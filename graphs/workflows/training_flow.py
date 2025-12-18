"""
Training pipeline workflow graph.

This is the core LangGraph definition that orchestrates the entire
training pipeline: SFT -> Quantization -> Evaluation -> Deployment.

Design:
- Uses conditional edges for routing decisions
- Supports checkpointing via pluggable backends (MongoDB, PostgreSQL)
- Designed for resume-after-pause pattern (scheduler-driven)
"""

from typing import Any, Optional

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from core.logging import get_logger
from graphs.state.agent_state import AgentState, JobPhase, JobStatus
from graphs.nodes.submit_ops import create_submit_node
from graphs.nodes.check_ops import create_check_node
from graphs.nodes.router import route_node
from tools.cluster_api.base import ClusterClient


logger = get_logger(__name__)


class TrainingWorkflow:
    """
    Training pipeline workflow manager.
    
    Usage:
        # Create workflow with dependencies
        workflow = TrainingWorkflow(
            cluster_client=MockClusterClient(),
            checkpointer=checkpointer,  # Any LangGraph-compatible checkpointer
        )
        
        # Run new task
        result = await workflow.invoke({
            "thread_id": "task-123",
            "base_model": "llama3-8b",
            "dataset_path": "s3://...",
        })
        
        # Resume existing task (by scheduler)
        result = await workflow.resume("task-123")
    """
    
    def __init__(
        self,
        cluster_client: ClusterClient,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """
        Initialize the workflow with dependencies.
        
        Args:
            cluster_client: Client for remote cluster operations
            checkpointer: LangGraph-compatible checkpointer for state persistence
                          (supports MongoDB, PostgreSQL, and other backends)
        """
        self.cluster_client = cluster_client
        self.checkpointer = checkpointer
        self._graph = self._build_graph()
        self._compiled = self._compile_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Construct the LangGraph state machine.
        
        Graph structure:
        
            [START]
               |
               v
            [router] ---> "submit" --> [submit_task] --> [router]
               |                                            |
               +--> "check" --> [check_status] --> [router]-+
               |
               +--> "complete" --> [END]
               |
               +--> "error" --> [END]
               |
               +--> "wait" --> [END]  (pause for scheduler)
        """
        # Create the graph with state schema
        graph = StateGraph(dict)
        
        # Create nodes with injected dependencies
        submit_node = create_submit_node(self.cluster_client)
        check_node = create_check_node(self.cluster_client)
        
        # Add nodes
        graph.add_node("router", self._router_wrapper)
        graph.add_node("submit_task", submit_node)
        graph.add_node("check_status", check_node)
        graph.add_node("on_complete", self._on_complete)
        graph.add_node("on_error", self._on_error)
        
        # Set entry point - always start with router
        graph.set_entry_point("router")
        
        # Add conditional edges from router
        graph.add_conditional_edges(
            "router",
            self._get_route,
            {
                "submit": "submit_task",
                "check": "check_status",
                "complete": "on_complete",
                "error": "on_error",
                "wait": END,  # End this invocation, wait for scheduler
            },
        )
        
        # After submit, go back to router (may need to wait)
        graph.add_edge("submit_task", "router")
        
        # After check, go back to router (may transition phase)
        graph.add_edge("check_status", "router")
        
        # Terminal nodes go to END
        graph.add_edge("on_complete", END)
        graph.add_edge("on_error", END)
        
        return graph
    
    def _compile_graph(self):
        """Compile the graph with checkpointer."""
        return self._graph.compile(checkpointer=self.checkpointer)
    
    def _router_wrapper(self, state: dict) -> dict:
        """
        Wrapper for router that logs decisions.
        
        The router doesn't modify state, just determines edges.
        """
        # Log current state for debugging
        phase = state.get("current_phase", "unknown")
        job_status = state.get("remote_job_status", "unknown")
        job_id = state.get("remote_job_id")
        
        logger.debug(
            "Router processing",
            thread_id=state.get("thread_id"),
            phase=phase,
            job_status=job_status,
            job_id=job_id,
        )
        
        return state  # Pass through unchanged
    
    def _get_route(self, state: dict) -> str:
        """Get routing decision from router node."""
        return route_node(state)
    
    def _on_complete(self, state: dict) -> dict:
        """Handle workflow completion."""
        logger.info(
            "Workflow completed",
            thread_id=state.get("thread_id"),
            deploy_url=state.get("deploy_url"),
        )
        return state
    
    def _on_error(self, state: dict) -> dict:
        """Handle workflow error."""
        logger.error(
            "Workflow failed",
            thread_id=state.get("thread_id"),
            error=state.get("error_message"),
        )
        return state
    
    async def invoke(
        self,
        initial_state: dict[str, Any],
        thread_id: str,
    ) -> dict[str, Any]:
        """
        Start a new workflow or resume an existing one.
        
        Args:
            initial_state: Initial state values (for new workflows)
            thread_id: Unique identifier for checkpointing
            
        Returns:
            Final state after this invocation
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # Ensure required fields
        if "thread_id" not in initial_state:
            initial_state["thread_id"] = thread_id
        if "current_phase" not in initial_state:
            initial_state["current_phase"] = JobPhase.SFT.value
        if "remote_job_status" not in initial_state:
            initial_state["remote_job_status"] = JobStatus.IDLE.value
        if "execution_logs" not in initial_state:
            initial_state["execution_logs"] = []
        if "config" not in initial_state:
            initial_state["config"] = {}
        
        logger.info(
            "Invoking workflow",
            thread_id=thread_id,
            initial_phase=initial_state.get("current_phase"),
        )
        
        result = await self._compiled.ainvoke(initial_state, config)
        return result
    
    async def resume(self, thread_id: str) -> Optional[dict[str, Any]]:
        """
        从 checkpoint 恢复工作流。
        
        关键: 恢复前清除 needs_wait 标志，允许执行一次 check。
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # 检查 checkpoint 是否存在
        state = await self.get_state(thread_id)
        if state is None:
            logger.warning(
                "No checkpoint found for thread",
                thread_id=thread_id,
            )
            return None
        
        logger.info(
            "Resuming workflow",
            thread_id=thread_id,
            current_phase=state.get("current_phase"),
            job_status=state.get("remote_job_status"),
        )
        
        # 清除 needs_wait 标志，允许本次 invoke 执行 check
        await self._compiled.aupdate_state(config, {"needs_wait": False})
        
        # 从 checkpoint 恢复执行
        result = await self._compiled.ainvoke(None, config)
        return result
    
    async def get_state(self, thread_id: str) -> Optional[dict[str, Any]]:
        """
        Get current state of a workflow without executing.
        
        Args:
            thread_id: Workflow identifier
            
        Returns:
            Current state dict or None if not found
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = await self._compiled.aget_state(config)
            if snapshot and snapshot.values:
                return snapshot.values
            return None
        except Exception as e:
            logger.debug(
                "Failed to get state",
                thread_id=thread_id,
                error=str(e),
            )
            return None


def create_training_workflow(
    cluster_client: ClusterClient,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> TrainingWorkflow:
    """
    Factory function to create a training workflow.
    
    This is the preferred way to instantiate workflows,
    allowing for easier testing and dependency injection.
    
    Args:
        cluster_client: Client for remote cluster operations
        checkpointer: LangGraph-compatible checkpointer (MongoDB, PostgreSQL, etc.)
    """
    return TrainingWorkflow(
        cluster_client=cluster_client,
        checkpointer=checkpointer,
    )


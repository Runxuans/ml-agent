"""
Task submission node operations.

This node handles submitting new tasks to the remote cluster.
It's triggered when entering a new phase with no active remote job.
"""

from datetime import datetime
from typing import Any

from core.logging import get_logger
from graphs.state.agent_state import AgentState, JobPhase, JobStatus
from tools.cluster_api.base import ClusterClient, TaskType


logger = get_logger(__name__)


def _phase_to_task_type(phase: JobPhase) -> TaskType:
    """Map pipeline phase to cluster task type."""
    mapping = {
        JobPhase.SFT: TaskType.SFT,
        JobPhase.QUANTIZATION: TaskType.QUANTIZATION,
        JobPhase.EVALUATION: TaskType.EVALUATION,
        JobPhase.DEPLOYMENT: TaskType.DEPLOYMENT,
    }
    task_type = mapping.get(phase)
    if not task_type:
        raise ValueError(f"Cannot submit task for phase: {phase}")
    return task_type


def _build_task_params(state: AgentState) -> dict[str, Any]:
    """
    Build task parameters based on current state and phase.
    
    Different phases need different inputs:
    - SFT: base_model + dataset
    - Quantization: sft_model_path + quant config
    - Evaluation: quant_model_path + eval config
    - Deployment: final model path + deploy config
    """
    phase = JobPhase(state.current_phase)
    base_params = {
        "thread_id": state.thread_id,
        "config": state.config,
    }
    
    if phase == JobPhase.SFT:
        return {
            **base_params,
            "base_model": state.base_model,
            "dataset_path": state.dataset_path,
        }
    elif phase == JobPhase.QUANTIZATION:
        return {
            **base_params,
            "model_path": state.sft_model_path,
            "quantization_type": state.config.get("quantization", "int4"),
        }
    elif phase == JobPhase.EVALUATION:
        return {
            **base_params,
            "model_path": state.quant_model_path or state.sft_model_path,
        }
    elif phase == JobPhase.DEPLOYMENT:
        return {
            **base_params,
            "model_path": state.quant_model_path or state.sft_model_path,
        }
    
    return base_params


async def submit_task_node(
    state: dict[str, Any],
    cluster_client: ClusterClient,
) -> dict[str, Any]:
    """
    Submit a new task to the remote cluster.
    
    This is a LangGraph node function that:
    1. Determines the task type from current phase
    2. Builds appropriate parameters
    3. Submits to remote cluster
    4. Updates state with job tracking info
    
    Args:
        state: Current graph state (dict form)
        cluster_client: Injected cluster client
        
    Returns:
        State updates (partial dict)
    """
    agent_state = AgentState.model_validate(state)
    phase = JobPhase(agent_state.current_phase)
    
    logger.info(
        "Submitting task",
        thread_id=agent_state.thread_id,
        phase=phase.value,
    )
    
    try:
        task_type = _phase_to_task_type(phase)
        params = _build_task_params(agent_state)
        
        # Submit to cluster
        job_info = await cluster_client.submit_task(task_type, params)
        
        # Log the submission
        log_message = f"[{phase.value.upper()}] Submitted job {job_info.job_id}"
        
        logger.info(
            "Task submitted successfully",
            thread_id=agent_state.thread_id,
            job_id=job_info.job_id,
            task_type=task_type.value,
        )
        
        # Return complete state (merge with existing to preserve all fields)
        return {
            **state,
            "remote_job_id": job_info.job_id,
            "remote_job_status": JobStatus.RUNNING.value,
            "remote_job_progress": job_info.progress,
            "retry_count": 0,  # Reset retry count on successful submit
            "execution_logs": agent_state.execution_logs + [
                f"[{datetime.utcnow().isoformat()}] {log_message}"
            ],
            "updated_at": datetime.utcnow(),
        }
        
    except Exception as e:
        logger.error(
            "Failed to submit task",
            thread_id=agent_state.thread_id,
            phase=phase.value,
            error=str(e),
        )
        
        # Return complete state with error info
        return {
            **state,
            "remote_job_status": JobStatus.FAILED.value,
            "error_message": f"Failed to submit {phase.value} task: {str(e)}",
            "execution_logs": agent_state.execution_logs + [
                f"[{datetime.utcnow().isoformat()}] [ERROR] Submit failed: {str(e)}"
            ],
            "updated_at": datetime.utcnow(),
        }


def create_submit_node(cluster_client: ClusterClient):
    """
    Factory function to create a submit node with injected dependencies.
    
    This allows the graph to use the node without knowing about
    the cluster client implementation.
    
    Usage:
        client = MockClusterClient()
        submit_node = create_submit_node(client)
        graph.add_node("submit", submit_node)
    """
    async def node(state: dict[str, Any]) -> dict[str, Any]:
        return await submit_task_node(state, cluster_client)
    return node


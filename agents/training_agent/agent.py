"""
Training Agent implementation.

Manages the full training pipeline workflow.
This is a thin wrapper around the orchestrator that conforms to
the BaseAgent interface, allowing for future agent type extensibility.
"""

from typing import Any, Optional

from agents.base import BaseAgent
from core.logging import get_logger
from manager.orchestrator import AgentOrchestrator
from tools.cluster_api.base import ClusterClient


logger = get_logger(__name__)


class TrainingAgent(BaseAgent):
    """
    Agent for managing ML training pipelines.
    
    Implements the training workflow:
    SFT -> Quantization -> Evaluation -> Deployment
    
    This agent delegates to the orchestrator for actual workflow management.
    It exists to provide a consistent interface if we add other agent types
    in the future (e.g., InferenceAgent, DataProcessingAgent).
    
    Usage:
        agent = TrainingAgent(cluster_client)
        await agent.initialize()
        
        task_id = await agent.create_task({
            "base_model": "llama3-8b",
            "dataset_path": "s3://...",
        })
    """
    
    def __init__(self, cluster_client: ClusterClient):
        """
        Initialize the training agent.
        
        Args:
            cluster_client: Client for remote cluster operations
        """
        self._orchestrator = AgentOrchestrator(cluster_client)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the agent's orchestrator."""
        if self._initialized:
            return
        
        logger.info("Initializing TrainingAgent")
        await self._orchestrator.initialize()
        self._initialized = True
        logger.info("TrainingAgent initialized")
    
    async def create_task(self, params: dict[str, Any]) -> str:
        """
        Create a new training task.
        
        Args:
            params: Must contain:
                - base_model: Base model identifier
                - dataset_path: Path to training data
                - config (optional): Additional configuration
                
        Returns:
            task_id for tracking
        """
        self._ensure_initialized()
        
        return await self._orchestrator.create_task(
            base_model=params["base_model"],
            dataset_path=params["dataset_path"],
            config=params.get("config"),
        )
    
    async def resume_task(self, task_id: str) -> Optional[dict[str, Any]]:
        """Resume an existing training task."""
        self._ensure_initialized()
        return await self._orchestrator.resume_task(task_id)
    
    async def get_task_status(self, task_id: str) -> Optional[dict[str, Any]]:
        """Get current status of a training task."""
        self._ensure_initialized()
        return await self._orchestrator.get_task_status(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running training task."""
        self._ensure_initialized()
        return await self._orchestrator.cancel_task(task_id)
    
    def _ensure_initialized(self) -> None:
        """Raise if not initialized."""
        if not self._initialized:
            raise RuntimeError(
                "TrainingAgent not initialized. Call initialize() first."
            )


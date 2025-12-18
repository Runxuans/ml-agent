"""
Base agent interface.

Defines the contract that all agents must implement.
This abstraction allows for different agent types in the future
(training agent, inference agent, data processing agent, etc.)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    An agent is responsible for managing a specific type of workflow.
    It provides a unified interface for:
    - Starting new tasks
    - Resuming existing tasks
    - Querying task status
    - Cancelling tasks
    
    Subclasses implement the specifics of their workflow type.
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the agent and its dependencies.
        
        Must be called before any other methods.
        """
        pass
    
    @abstractmethod
    async def create_task(
        self,
        params: dict[str, Any],
    ) -> str:
        """
        Create a new task.
        
        Args:
            params: Task-specific parameters
            
        Returns:
            task_id for tracking
        """
        pass
    
    @abstractmethod
    async def resume_task(self, task_id: str) -> Optional[dict[str, Any]]:
        """
        Resume an existing task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Updated state after resume
        """
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> Optional[dict[str, Any]]:
        """
        Get current status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Current state or None if not found
        """
        pass
    
    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if cancellation was successful
        """
        pass


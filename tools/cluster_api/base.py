"""
Abstract base class for cluster API clients.

This defines the contract that all cluster implementations must follow,
whether mock, real cloud providers (AWS, GCP), or on-prem solutions.

Design principles:
- Task-type agnostic: Same interface for SFT, quantization, etc.
- Async-first: All operations are coroutines
- Result objects: Return structured data, not raw dicts
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class TaskType(str, Enum):
    """Types of tasks that can be submitted to the cluster."""
    SFT = "sft"
    QUANTIZATION = "quantization"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"


class RemoteJobStatus(str, Enum):
    """Status reported by remote cluster."""
    PENDING = "pending"      # Queued, waiting for resources
    RUNNING = "running"      # Actively executing
    SUCCESS = "success"      # Completed successfully
    FAILED = "failed"        # Failed with error
    CANCELLED = "cancelled"  # Manually cancelled


@dataclass
class JobInfo:
    """
    Information about a remote job.
    
    Returned by submit_task and get_job_status methods.
    """
    job_id: str
    status: RemoteJobStatus
    task_type: TaskType
    progress: Optional[float] = None  # 0-100
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def is_terminal(self) -> bool:
        """Check if job has reached a terminal state."""
        return self.status in (
            RemoteJobStatus.SUCCESS,
            RemoteJobStatus.FAILED,
            RemoteJobStatus.CANCELLED,
        )


@dataclass
class JobResult:
    """
    Result of a completed job.
    
    Contains the outputs/artifacts produced by the remote task.
    """
    job_id: str
    task_type: TaskType
    success: bool
    output_path: Optional[str] = None  # Path to artifacts
    metrics: Optional[dict[str, Any]] = None  # Task-specific metrics
    error_message: Optional[str] = None
    logs: Optional[list[str]] = None


class ClusterClient(ABC):
    """
    Abstract interface for cluster API clients.
    
    Implementations handle the specifics of communicating with different
    cluster backends (mock, cloud, on-prem).
    
    Usage:
        client = MockClusterClient()
        
        # Submit a task
        job_info = await client.submit_task(
            task_type=TaskType.SFT,
            params={"base_model": "llama3-8b", "dataset": "s3://..."}
        )
        
        # Poll for status
        status = await client.get_job_status(job_info.job_id)
        
        # Get results when complete
        if status.status == RemoteJobStatus.SUCCESS:
            result = await client.get_job_result(job_info.job_id)
    """
    
    @abstractmethod
    async def submit_task(
        self,
        task_type: TaskType,
        params: dict[str, Any],
    ) -> JobInfo:
        """
        Submit a new task to the cluster.
        
        Args:
            task_type: Type of task to run
            params: Task-specific parameters
            
        Returns:
            JobInfo with the assigned job_id
            
        Raises:
            ClusterError: If submission fails
        """
        pass
    
    @abstractmethod
    async def get_job_status(self, job_id: str) -> JobInfo:
        """
        Get current status of a job.
        
        Args:
            job_id: ID returned from submit_task
            
        Returns:
            Current JobInfo
            
        Raises:
            JobNotFoundError: If job_id doesn't exist
        """
        pass
    
    @abstractmethod
    async def get_job_result(self, job_id: str) -> JobResult:
        """
        Get results of a completed job.
        
        Should only be called after job reaches terminal state.
        
        Args:
            job_id: ID returned from submit_task
            
        Returns:
            JobResult with outputs
            
        Raises:
            JobNotFoundError: If job_id doesn't exist
            JobNotCompleteError: If job hasn't finished
        """
        pass
    
    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """
        Attempt to cancel a running job.
        
        Args:
            job_id: ID returned from submit_task
            
        Returns:
            True if cancellation was successful
        """
        pass


class ClusterError(Exception):
    """Base exception for cluster operations."""
    pass


class JobNotFoundError(ClusterError):
    """Job ID doesn't exist in the cluster."""
    pass


class JobNotCompleteError(ClusterError):
    """Attempted to get results of incomplete job."""
    pass


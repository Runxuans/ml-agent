"""
Abstract base classes for storage backends.

This module defines the contracts that all storage implementations must follow,
enabling pluggable backends for checkpoint persistence and job index management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class JobIndexRecord:
    """
    Record structure for job index storage.
    
    This is the data model for tracking active jobs independently
    of the LangGraph checkpoint storage.
    """
    thread_id: str
    current_phase: str = "pending"
    remote_job_id: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed, cancelled
    error_message: Optional[str] = None
    retry_count: int = 0
    last_checked_at: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "thread_id": self.thread_id,
            "current_phase": self.current_phase,
            "remote_job_id": self.remote_job_id,
            "status": self.status,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "last_checked_at": self.last_checked_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobIndexRecord":
        """Create from dictionary."""
        return cls(
            thread_id=data["thread_id"],
            current_phase=data.get("current_phase", "pending"),
            remote_job_id=data.get("remote_job_id"),
            status=data.get("status", "pending"),
            error_message=data.get("error_message"),
            retry_count=data.get("retry_count", 0),
            last_checked_at=data.get("last_checked_at", datetime.utcnow()),
            created_at=data.get("created_at", datetime.utcnow()),
            updated_at=data.get("updated_at", datetime.utcnow()),
        )


class BaseCheckpointer(ABC):
    """
    Abstract base class for LangGraph checkpoint storage.
    
    Implementations must provide both sync and async interfaces
    to support LangGraph's requirements.
    
    Note: LangGraph has built-in support for various backends.
    This wrapper provides a unified interface for initialization
    and lifecycle management.
    """
    
    @abstractmethod
    def setup(self) -> None:
        """
        Initialize the storage backend (create tables/collections).
        
        This should be idempotent - safe to call multiple times.
        """
        pass
    
    @abstractmethod
    async def setup_async(self) -> None:
        """Async version of setup for async-first applications."""
        pass
    
    @abstractmethod
    def get_native_checkpointer(self) -> Any:
        """
        Get the native LangGraph checkpointer instance.
        
        Returns the actual checkpointer object that LangGraph
        can use directly (e.g., PostgresSaver, MongoDBSaver).
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (connections, pools)."""
        pass


class BaseJobIndexRepository(ABC):
    """
    Abstract base class for job index storage.
    
    The job index is a lightweight index of active jobs used by
    the scheduler for efficient polling. It's separate from
    the checkpoint storage for performance reasons.
    """
    
    @abstractmethod
    async def setup(self) -> None:
        """
        Initialize the storage (create tables/collections/indexes).
        
        This should be idempotent.
        """
        pass
    
    @abstractmethod
    async def create(self, record: JobIndexRecord) -> None:
        """
        Create a new job index record.
        
        If the record already exists (by thread_id), it should be updated.
        """
        pass
    
    @abstractmethod
    async def get(self, thread_id: str) -> Optional[JobIndexRecord]:
        """Get a job record by thread_id."""
        pass
    
    @abstractmethod
    async def update(
        self,
        thread_id: str,
        *,
        current_phase: Optional[str] = None,
        remote_job_id: Optional[str] = None,
        status: Optional[str] = None,
        error_message: Optional[str] = None,
        retry_count: Optional[int] = None,
    ) -> bool:
        """
        Update specific fields of a job record.
        
        Only provided fields will be updated.
        Returns True if record was found and updated.
        """
        pass
    
    @abstractmethod
    async def get_active_jobs(self) -> list[str]:
        """
        Get thread_ids of jobs that need processing.
        
        Returns jobs with status in ('running', 'pending'),
        ordered by last_checked_at ascending (oldest first).
        """
        pass
    
    @abstractmethod
    async def set_cancelled(self, thread_id: str) -> bool:
        """
        Mark a job as cancelled.
        
        Returns True if record was found and updated.
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass


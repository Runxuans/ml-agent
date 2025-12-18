"""
Storage factory for creating storage backend instances.

This module provides factory functions to create the appropriate
storage implementations based on configuration.
"""

from enum import Enum
from typing import TYPE_CHECKING

from core.logging import get_logger
from core.storage.base import BaseCheckpointer, BaseJobIndexRepository


if TYPE_CHECKING:
    from core.config import Settings


logger = get_logger(__name__)


class StorageBackend(str, Enum):
    """Supported storage backends."""
    MONGODB = "mongodb"
    POSTGRES = "postgres"


def get_storage_backend(settings: "Settings") -> StorageBackend:
    """
    Determine which storage backend to use based on settings.
    
    Args:
        settings: Application settings
        
    Returns:
        The configured storage backend
    """
    backend_str = settings.storage_backend.lower()
    
    try:
        return StorageBackend(backend_str)
    except ValueError:
        raise ValueError(
            f"Unsupported storage backend: {backend_str}. "
            f"Supported backends: {[b.value for b in StorageBackend]}"
        )


def create_checkpointer(settings: "Settings") -> BaseCheckpointer:
    """
    Create a checkpointer instance based on settings.
    
    Args:
        settings: Application settings
        
    Returns:
        Configured checkpointer instance (not yet initialized)
    """
    backend = get_storage_backend(settings)
    
    if backend == StorageBackend.MONGODB:
        from core.storage.mongodb import MongoDBCheckpointer
        
        logger.info(
            "Creating MongoDB checkpointer",
            database=settings.mongodb_database,
        )
        connection_string = f"mongodb://{settings.mongodb_user}:{settings.mongodb_password}@{settings.mongodb_url}"
        return MongoDBCheckpointer(
            connection_string=connection_string,
            database_name=settings.mongodb_database,
        )
    
    elif backend == StorageBackend.POSTGRES:
        from core.storage.postgres import PostgresCheckpointer
        
        logger.info("Creating PostgreSQL checkpointer")
        return PostgresCheckpointer(
            sync_connection_string=settings.postgres_sync_url,
        )
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def create_job_index_repository(settings: "Settings") -> BaseJobIndexRepository:
    """
    Create a job index repository instance based on settings.
    
    Args:
        settings: Application settings
        
    Returns:
        Configured repository instance (not yet initialized)
    """
    backend = get_storage_backend(settings)
    
    if backend == StorageBackend.MONGODB:
        from core.storage.mongodb import MongoDBJobIndexRepository
        
        logger.info(
            "Creating MongoDB job index repository",
            database=settings.mongodb_database,
        )
        return MongoDBJobIndexRepository(
            connection_string=settings.mongodb_url,
            database_name=settings.mongodb_database,
        )
    
    elif backend == StorageBackend.POSTGRES:
        from core.storage.postgres import PostgresJobIndexRepository
        
        logger.info("Creating PostgreSQL job index repository")
        return PostgresJobIndexRepository(
            async_connection_string=settings.postgres_async_url,
            echo=settings.debug,
        )
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")


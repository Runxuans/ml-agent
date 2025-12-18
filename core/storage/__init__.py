"""
Storage abstraction layer.

Provides pluggable storage backends for:
- Checkpoint persistence (LangGraph state)
- Job index management (active jobs tracking)

Supported backends:
- MongoDB (recommended for production)
- PostgreSQL (legacy support)
"""

from core.storage.base import (
    BaseCheckpointer,
    BaseJobIndexRepository,
    JobIndexRecord,
)
from core.storage.factory import (
    create_checkpointer,
    create_job_index_repository,
    get_storage_backend,
    StorageBackend,
)

__all__ = [
    # Abstract interfaces
    "BaseCheckpointer",
    "BaseJobIndexRepository",
    "JobIndexRecord",
    # Factory functions
    "create_checkpointer",
    "create_job_index_repository",
    "get_storage_backend",
    "StorageBackend",
]


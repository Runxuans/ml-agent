"""
MongoDB storage backend implementation.

Provides MongoDB implementations for:
- LangGraph checkpoint storage (via langgraph-checkpoint-mongodb)
- Job index repository
"""

from datetime import datetime
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient, ASCENDING

from core.logging import get_logger
from core.storage.base import BaseCheckpointer, BaseJobIndexRepository, JobIndexRecord


logger = get_logger(__name__)


class MongoDBCheckpointer(BaseCheckpointer):
    """
    MongoDB-based checkpoint storage for LangGraph.
    
    Uses langgraph-checkpoint-mongodb under the hood.
    """
    
    def __init__(
        self,
        connection_string: str,
        database_name: str = "ml_agent",
    ):
        """
        Initialize MongoDB checkpointer.
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Database name for checkpoints
        """
        self._connection_string = connection_string
        self._database_name = database_name
        self._sync_client: Optional[MongoClient] = None
        self._async_client: Optional[AsyncIOMotorClient] = None
        self._checkpointer: Optional[Any] = None
    
    def setup(self) -> None:
        """Initialize sync MongoDB client and LangGraph checkpointer."""
        from langgraph.checkpoint.mongodb import MongoDBSaver
        
        self._sync_client = MongoClient(self._connection_string)
        self._checkpointer = MongoDBSaver(self._sync_client, self._database_name)
        
        logger.info(
            "MongoDB checkpointer initialized",
            database=self._database_name,
        )
    
    async def setup_async(self) -> None:
        """Initialize async MongoDB client."""
        from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
        
        self._async_client = AsyncIOMotorClient(self._connection_string)
        self._checkpointer = AsyncMongoDBSaver(
            self._async_client,
            self._database_name,
        )
        
        logger.info(
            "Async MongoDB checkpointer initialized",
            database=self._database_name,
        )
    
    def get_native_checkpointer(self) -> Any:
        """Get the LangGraph MongoDBSaver instance."""
        if self._checkpointer is None:
            raise RuntimeError(
                "Checkpointer not initialized. Call setup() or setup_async() first."
            )
        return self._checkpointer
    
    async def close(self) -> None:
        """Close MongoDB connections."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        
        if self._async_client is not None:
            self._async_client.close()
            self._async_client = None
        
        self._checkpointer = None
        logger.info("MongoDB checkpointer closed")


class MongoDBJobIndexRepository(BaseJobIndexRepository):
    """
    MongoDB-based job index repository.
    
    Uses a dedicated collection for efficient job polling by the scheduler.
    """
    
    COLLECTION_NAME = "active_jobs"
    
    def __init__(
        self,
        connection_string: str,
        database_name: str = "ml_agent",
    ):
        """
        Initialize MongoDB job index repository.
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Database name
        """
        self._connection_string = connection_string
        self._database_name = database_name
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
    
    async def setup(self) -> None:
        """Initialize connection and create indexes."""
        self._client = AsyncIOMotorClient(self._connection_string)
        self._db = self._client[self._database_name]
        
        collection = self._db[self.COLLECTION_NAME]
        
        # Create indexes for efficient querying
        await collection.create_index("thread_id", unique=True)
        await collection.create_index(
            [("status", ASCENDING), ("last_checked_at", ASCENDING)],
            name="idx_status_last_checked",
        )
        await collection.create_index("current_phase")
        
        logger.info(
            "MongoDB job index repository initialized",
            database=self._database_name,
            collection=self.COLLECTION_NAME,
        )
    
    @property
    def _collection(self):
        """Get the active_jobs collection."""
        if self._db is None:
            raise RuntimeError(
                "Repository not initialized. Call setup() first."
            )
        return self._db[self.COLLECTION_NAME]
    
    async def create(self, record: JobIndexRecord) -> None:
        """Create or update a job index record (upsert)."""
        doc = record.to_dict()
        doc["_id"] = record.thread_id  # Use thread_id as _id
        
        await self._collection.update_one(
            {"_id": record.thread_id},
            {"$set": doc},
            upsert=True,
        )
        
        logger.debug(
            "Job index record created/updated",
            thread_id=record.thread_id,
        )
    
    async def get(self, thread_id: str) -> Optional[JobIndexRecord]:
        """Get a job record by thread_id."""
        doc = await self._collection.find_one({"_id": thread_id})
        if doc is None:
            return None
        
        # Remove MongoDB _id field
        doc.pop("_id", None)
        return JobIndexRecord.from_dict(doc)
    
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
        """Update specific fields of a job record."""
        update_fields: dict[str, Any] = {
            "last_checked_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        
        if current_phase is not None:
            update_fields["current_phase"] = current_phase
        if remote_job_id is not None:
            update_fields["remote_job_id"] = remote_job_id
        if status is not None:
            update_fields["status"] = status
        if error_message is not None:
            update_fields["error_message"] = error_message
        if retry_count is not None:
            update_fields["retry_count"] = retry_count
        
        result = await self._collection.update_one(
            {"_id": thread_id},
            {"$set": update_fields},
        )
        
        return result.matched_count > 0
    
    async def get_active_jobs(self) -> list[str]:
        """Get thread_ids of jobs that need processing."""
        cursor = self._collection.find(
            {"status": {"$in": ["running", "pending"]}},
            {"_id": 1},
        ).sort("last_checked_at", ASCENDING)
        
        docs = await cursor.to_list(length=None)
        return [doc["_id"] for doc in docs]
    
    async def set_cancelled(self, thread_id: str) -> bool:
        """Mark a job as cancelled."""
        result = await self._collection.update_one(
            {"_id": thread_id},
            {
                "$set": {
                    "status": "cancelled",
                    "last_checked_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }
            },
        )
        
        if result.matched_count > 0:
            logger.info("Job cancelled", thread_id=thread_id)
            return True
        return False
    
    async def close(self) -> None:
        """Close MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
        logger.info("MongoDB job index repository closed")


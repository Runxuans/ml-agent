"""
PostgreSQL storage backend implementation.

Provides PostgreSQL implementations for:
- LangGraph checkpoint storage (via langgraph-checkpoint-postgres)
- Job index repository
"""

from datetime import datetime
from typing import Any, Optional

from psycopg_pool import ConnectionPool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.logging import get_logger
from core.storage.base import BaseCheckpointer, BaseJobIndexRepository, JobIndexRecord


logger = get_logger(__name__)


class PostgresCheckpointer(BaseCheckpointer):
    """
    PostgreSQL-based checkpoint storage for LangGraph.
    
    Uses langgraph-checkpoint-postgres (PostgresSaver) under the hood.
    """
    
    def __init__(
        self,
        sync_connection_string: str,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ):
        """
        Initialize PostgreSQL checkpointer.
        
        Args:
            sync_connection_string: PostgreSQL sync connection URI (psycopg format)
            pool_min_size: Minimum pool size
            pool_max_size: Maximum pool size
        """
        self._connection_string = sync_connection_string
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool: Optional[ConnectionPool] = None
        self._checkpointer: Optional[Any] = None
    
    def setup(self) -> None:
        """Initialize sync PostgreSQL connection pool and LangGraph checkpointer."""
        from langgraph.checkpoint.postgres import PostgresSaver
        
        self._pool = ConnectionPool(
            conninfo=self._connection_string,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
            open=True,
        )
        
        self._checkpointer = PostgresSaver(self._pool)
        self._checkpointer.setup()  # Creates LangGraph checkpoint tables
        
        logger.info("PostgreSQL checkpointer initialized")
    
    async def setup_async(self) -> None:
        """
        Async setup - note that PostgresSaver requires sync connection.
        
        This method still uses sync setup internally.
        """
        self.setup()
    
    def get_native_checkpointer(self) -> Any:
        """Get the LangGraph PostgresSaver instance."""
        if self._checkpointer is None:
            raise RuntimeError(
                "Checkpointer not initialized. Call setup() first."
            )
        return self._checkpointer
    
    async def close(self) -> None:
        """Close PostgreSQL connection pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        
        self._checkpointer = None
        logger.info("PostgreSQL checkpointer closed")


class PostgresJobIndexRepository(BaseJobIndexRepository):
    """
    PostgreSQL-based job index repository.
    
    Uses SQLAlchemy async for database operations.
    """
    
    def __init__(
        self,
        async_connection_string: str,
        echo: bool = False,
    ):
        """
        Initialize PostgreSQL job index repository.
        
        Args:
            async_connection_string: PostgreSQL async connection URI (asyncpg format)
            echo: Whether to echo SQL statements
        """
        self._connection_string = async_connection_string
        self._echo = echo
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
    
    async def setup(self) -> None:
        """Initialize connection and create table/indexes if not exists."""
        self._engine = create_async_engine(
            self._connection_string,
            echo=self._echo,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        
        # Create table if not exists
        async with self._engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS active_jobs (
                    thread_id VARCHAR(255) PRIMARY KEY,
                    current_phase VARCHAR(50) NOT NULL DEFAULT 'pending',
                    remote_job_id VARCHAR(255),
                    status VARCHAR(50) NOT NULL DEFAULT 'pending',
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    last_checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_active_jobs_status 
                ON active_jobs(status) 
                WHERE status IN ('running', 'pending')
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_active_jobs_phase 
                ON active_jobs(current_phase)
            """))
        
        logger.info("PostgreSQL job index repository initialized")
    
    async def _get_session(self):
        """Get a new session."""
        if self._session_factory is None:
            raise RuntimeError(
                "Repository not initialized. Call setup() first."
            )
        return self._session_factory()
    
    async def create(self, record: JobIndexRecord) -> None:
        """Create or update a job index record (upsert)."""
        async with await self._get_session() as session:
            await session.execute(
                text("""
                    INSERT INTO active_jobs (
                        thread_id, current_phase, remote_job_id, status,
                        error_message, retry_count, last_checked_at, created_at, updated_at
                    ) VALUES (
                        :thread_id, :current_phase, :remote_job_id, :status,
                        :error_message, :retry_count, :last_checked_at, :created_at, :updated_at
                    )
                    ON CONFLICT (thread_id) DO UPDATE SET
                        current_phase = :current_phase,
                        remote_job_id = :remote_job_id,
                        status = :status,
                        error_message = :error_message,
                        retry_count = :retry_count,
                        last_checked_at = :last_checked_at,
                        updated_at = :updated_at
                """),
                record.to_dict(),
            )
            await session.commit()
        
        logger.debug(
            "Job index record created/updated",
            thread_id=record.thread_id,
        )
    
    async def get(self, thread_id: str) -> Optional[JobIndexRecord]:
        """Get a job record by thread_id."""
        async with await self._get_session() as session:
            result = await session.execute(
                text("SELECT * FROM active_jobs WHERE thread_id = :thread_id"),
                {"thread_id": thread_id},
            )
            row = result.mappings().first()
            
            if row is None:
                return None
            
            return JobIndexRecord.from_dict(dict(row))
    
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
        # Build dynamic SET clause
        set_parts = ["last_checked_at = :now", "updated_at = :now"]
        params: dict[str, Any] = {
            "thread_id": thread_id,
            "now": datetime.utcnow(),
        }
        
        if current_phase is not None:
            set_parts.append("current_phase = :current_phase")
            params["current_phase"] = current_phase
        if remote_job_id is not None:
            set_parts.append("remote_job_id = :remote_job_id")
            params["remote_job_id"] = remote_job_id
        if status is not None:
            set_parts.append("status = :status")
            params["status"] = status
        if error_message is not None:
            set_parts.append("error_message = :error_message")
            params["error_message"] = error_message
        if retry_count is not None:
            set_parts.append("retry_count = :retry_count")
            params["retry_count"] = retry_count
        
        query = f"""
            UPDATE active_jobs 
            SET {', '.join(set_parts)}
            WHERE thread_id = :thread_id
        """
        
        async with await self._get_session() as session:
            result = await session.execute(text(query), params)
            await session.commit()
            return result.rowcount > 0
    
    async def get_active_jobs(self) -> list[str]:
        """Get thread_ids of jobs that need processing."""
        async with await self._get_session() as session:
            result = await session.execute(
                text("""
                    SELECT thread_id FROM active_jobs 
                    WHERE status IN ('running', 'pending')
                    ORDER BY last_checked_at ASC
                """)
            )
            rows = result.fetchall()
            return [row[0] for row in rows]
    
    async def set_cancelled(self, thread_id: str) -> bool:
        """Mark a job as cancelled."""
        async with await self._get_session() as session:
            result = await session.execute(
                text("""
                    UPDATE active_jobs 
                    SET status = 'cancelled',
                        last_checked_at = :now,
                        updated_at = :now
                    WHERE thread_id = :thread_id
                """),
                {"thread_id": thread_id, "now": datetime.utcnow()},
            )
            await session.commit()
            
            if result.rowcount > 0:
                logger.info("Job cancelled", thread_id=thread_id)
                return True
            return False
    
    async def close(self) -> None:
        """Close database engine."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
        logger.info("PostgreSQL job index repository closed")


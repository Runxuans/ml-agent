"""
Database connection and session management.

DEPRECATED: This module is deprecated in favor of core.storage module.
The storage abstraction layer provides pluggable backends (MongoDB, PostgreSQL).

This module is kept for backward compatibility but will be removed in future versions.
New code should use:
    - core.storage.create_checkpointer() for checkpoint storage
    - core.storage.create_job_index_repository() for job index operations

For PostgreSQL-specific legacy code, the functions below are preserved
but will emit deprecation warnings.
"""

import warnings
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from psycopg_pool import ConnectionPool

from core.config import settings


# Async engine for application database operations
_async_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None

# Sync connection pool for LangGraph PostgresSaver
_sync_pool: ConnectionPool | None = None


def _warn_deprecated(func_name: str) -> None:
    """Emit deprecation warning for legacy database functions."""
    warnings.warn(
        f"{func_name}() is deprecated. Use core.storage module instead. "
        "See core.storage.create_checkpointer() and create_job_index_repository().",
        DeprecationWarning,
        stacklevel=3,
    )


def get_async_engine() -> AsyncEngine:
    """
    Get or create async SQLAlchemy engine.
    
    DEPRECATED: Use core.storage module instead.
    """
    _warn_deprecated("get_async_engine")
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            settings.postgres_async_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _async_engine


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create async session factory.
    
    DEPRECATED: Use core.storage module instead.
    """
    _warn_deprecated("get_async_session_factory")
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _async_session_factory


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.
    
    DEPRECATED: Use core.storage module instead.
    
    Usage:
        async with get_async_session() as session:
            result = await session.execute(...)
    """
    _warn_deprecated("get_async_session")
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_sync_connection_pool() -> ConnectionPool:
    """
    Get or create sync connection pool for LangGraph PostgresSaver.
    
    DEPRECATED: Use core.storage.create_checkpointer() instead.
    """
    _warn_deprecated("get_sync_connection_pool")
    global _sync_pool
    if _sync_pool is None:
        _sync_pool = ConnectionPool(
            conninfo=settings.postgres_sync_url,
            min_size=2,
            max_size=10,
            open=True,
        )
    return _sync_pool


async def init_database() -> None:
    """
    Initialize database connections.
    
    DEPRECATED: Storage initialization is now handled by the orchestrator.
    """
    _warn_deprecated("init_database")
    
    if not settings.is_postgres:
        warnings.warn(
            "init_database() only supports PostgreSQL. "
            "Current storage backend is MongoDB. This call will be ignored.",
            UserWarning,
        )
        return
    
    # Test async connection
    engine = get_async_engine()
    async with engine.connect() as conn:
        await conn.execute("SELECT 1")
    
    # Initialize sync pool
    get_sync_connection_pool()


async def close_database() -> None:
    """
    Close all database connections.
    
    DEPRECATED: Storage cleanup is now handled by the orchestrator.
    """
    _warn_deprecated("close_database")
    global _async_engine, _async_session_factory, _sync_pool
    
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None
    
    if _sync_pool is not None:
        _sync_pool.close()
        _sync_pool = None


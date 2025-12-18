"""
Application configuration management.

Uses pydantic-settings for type-safe environment variable parsing.
All configuration is centralized here to support dependency injection
and avoid scattering os.getenv() calls throughout the codebase.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with validation and type coercion.
    
    Values are loaded from environment variables or .env file.
    All fields have sensible defaults for local development.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Storage Backend Selection
    # Options: "mongodb", "postgres"
    storage_backend: Literal["mongodb", "postgres"] = "mongodb"
    
    # MongoDB Configuration (default backend)
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "ml_agent"
    mondodb_user: str = "admin"
    mondodb_password: str = "a123980"
    
    # PostgreSQL Configuration (legacy support)
    postgres_async_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ml_agent"
    postgres_sync_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/ml_agent"
    
    # Deprecated: Keep for backward compatibility, will be removed in future
    @property
    def database_url(self) -> str:
        """Deprecated: Use postgres_async_url instead."""
        return self.postgres_async_url
    
    @property
    def database_sync_url(self) -> str:
        """Deprecated: Use postgres_sync_url instead."""
        return self.postgres_sync_url
    
    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    debug: bool = True
    
    # Scheduler
    scheduler_poll_interval_seconds: int = 60
    scheduler_max_concurrent_jobs: int = 10
    
    # Mock configuration (Phase 1)
    mock_task_duration_seconds: int = 120
    
    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    @property
    def is_mongodb(self) -> bool:
        """Check if using MongoDB backend."""
        return self.storage_backend == "mongodb"
    
    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL backend."""
        return self.storage_backend == "postgres"


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings singleton.
    
    Use this function to get settings instance throughout the application.
    The @lru_cache ensures we only parse environment once.
    """
    return Settings()


# Convenience export for direct import
settings = get_settings()


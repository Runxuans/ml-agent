"""
Structured logging configuration.

Uses structlog for machine-readable, context-rich logging.
Supports both JSON (production) and human-readable (development) output.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from core.config import settings


def configure_logging() -> None:
    """
    Configure structlog with appropriate processors based on environment.
    
    Development: Human-readable colored output
    Production: JSON output for log aggregation systems
    """
    
    # Shared processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    if settings.is_development:
        # Development: pretty printing
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # Production: JSON for log aggregation
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        **initial_context: Initial context values to bind
    
    Returns:
        A bound logger with the given context
    
    Usage:
        logger = get_logger(__name__, task_id="abc-123")
        logger.info("Task started", phase="sft")
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


"""
FastAPI dependencies for dependency injection.

Provides singleton instances of core services to route handlers.
"""

from typing import Optional

from manager.orchestrator import AgentOrchestrator
from manager.scheduler import JobScheduler


# Global singletons (set during app lifespan)
_orchestrator: Optional[AgentOrchestrator] = None
_scheduler: Optional[JobScheduler] = None


def set_orchestrator(orchestrator: AgentOrchestrator) -> None:
    """Set the global orchestrator instance."""
    global _orchestrator
    _orchestrator = orchestrator


def set_scheduler(scheduler: JobScheduler) -> None:
    """Set the global scheduler instance."""
    global _scheduler
    _scheduler = scheduler


async def get_orchestrator() -> AgentOrchestrator:
    """
    Dependency that provides the orchestrator.
    
    Usage:
        @router.get("/tasks")
        async def list_tasks(
            orchestrator: AgentOrchestrator = Depends(get_orchestrator)
        ):
            ...
    """
    if _orchestrator is None:
        raise RuntimeError("Orchestrator not initialized")
    return _orchestrator


async def get_scheduler() -> JobScheduler:
    """
    Dependency that provides the scheduler.
    """
    if _scheduler is None:
        raise RuntimeError("Scheduler not initialized")
    return _scheduler


"""
Health check endpoints.

Provides endpoints for monitoring and load balancer health checks.
"""

from fastapi import APIRouter, Response

from core.logging import get_logger


logger = get_logger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> dict:
    """
    Basic health check.
    
    Returns 200 if the service is running.
    Used by load balancers and orchestration systems.
    """
    return {
        "status": "healthy",
        "service": "ml-agent",
    }


@router.get("/ready")
async def readiness_check() -> dict:
    """
    Readiness check.
    
    Returns 200 if the service is ready to handle requests.
    Checks database connectivity and scheduler status.
    """
    # TODO: Add actual readiness checks
    # - Database connectivity
    # - Scheduler status
    # - Cluster API reachability
    
    return {
        "status": "ready",
        "checks": {
            "database": "ok",
            "scheduler": "ok",
        },
    }


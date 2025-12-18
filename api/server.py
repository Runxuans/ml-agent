"""
FastAPI application entry point.

Sets up the application with:
- Lifespan management (startup/shutdown)
- Route registration
- Middleware configuration
- Error handling
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import set_orchestrator, set_scheduler
from api.routes import tasks_router, health_router
from core.config import settings
from core.logging import configure_logging, get_logger
from manager.orchestrator import AgentOrchestrator
from manager.scheduler import JobScheduler
from tools.cluster_api.mock_client import MockClusterClient


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    应用程序生命周期管理器。
    
    处理启动和关闭:
    - 启动: 初始化编排器（包括存储后端）, 创建并启动调度器
    - 关闭: 停止调度器, 关闭编排器（清理存储连接）
    """

    logger.info(
        "Starting ML Agent service...",
        storage_backend=settings.storage_backend,
    )
    
    configure_logging()
    
    # Create cluster client (Mock for Phase 1)
    # In Phase 2+, this will be a real cluster client
    cluster_client = MockClusterClient(
        task_duration=settings.mock_task_duration_seconds
    )
    logger.info(
        "Using MockClusterClient",
        task_duration=settings.mock_task_duration_seconds,
    )
    
    # Create and initialize orchestrator
    # The orchestrator handles storage backend initialization internally
    orchestrator = AgentOrchestrator(cluster_client)
    await orchestrator.initialize()
    set_orchestrator(orchestrator)
    
    # Create and start scheduler
    scheduler = JobScheduler(
        orchestrator=orchestrator,
        poll_interval_seconds=settings.scheduler_poll_interval_seconds,
        max_concurrent_jobs=settings.scheduler_max_concurrent_jobs,
    )
    await scheduler.start()
    set_scheduler(scheduler)
    
    logger.info(
        "ML Agent service started",
        host=settings.server_host,
        port=settings.server_port,
        poll_interval=settings.scheduler_poll_interval_seconds,
        storage_backend=settings.storage_backend,
    )
    
    yield
    
    # =========================================
    # Shutdown
    # =========================================
    logger.info("Shutting down ML Agent service...")
    
    # Stop scheduler
    await scheduler.shutdown()
    
    # Shutdown orchestrator (closes storage connections)
    await orchestrator.shutdown()
    
    logger.info("ML Agent service stopped")


def create_app() -> FastAPI:
    """
    Application factory.
    
    Creates and configures the FastAPI application.
    """
    app = FastAPI(
        title="ML Agent Platform",
        description=(
            "LLM Full-Pipeline Automation Agent Platform.\n\n"
            "Automatically handles: SFT -> Quantization -> Evaluation -> Deployment"
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    app.include_router(health_router)
    app.include_router(tasks_router)
    

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "message": str(exc) if settings.debug else "An error occurred",
            },
        )
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.server:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug,
    )


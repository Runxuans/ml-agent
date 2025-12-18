"""
Task management endpoints.

Provides CRUD operations for training tasks:
- POST /tasks - Create new task
- GET /tasks/{task_id} - Get task status
- DELETE /tasks/{task_id} - Cancel task
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query

from api.schemas.task import (
    TaskCreateRequest,
    TaskCreateResponse,
    TaskStatusResponse,
    TaskArtifacts,
)
from api.dependencies import get_orchestrator
from core.logging import get_logger
from manager.orchestrator import AgentOrchestrator


logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/tasks", tags=["Tasks"])


@router.post("", response_model=TaskCreateResponse)
async def create_task(
    request: TaskCreateRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> TaskCreateResponse:
    """
    Create a new training task.
    
    Starts the training pipeline for the specified model and dataset.
    The task will proceed through SFT -> Quantization -> Evaluation -> Deployment.
    
    Returns immediately with a task_id for tracking. Use GET /tasks/{task_id}
    to monitor progress.
    """
    logger.info(
        "Creating task",
        base_model=request.base_model,
        dataset_path=request.dataset_path,
    )
    
    try:
        # Build config from request
        config = request.config or {}
        if request.quantization:
            config["quantization"] = request.quantization
        
        task_id = await orchestrator.create_task(
            base_model=request.base_model,
            dataset_path=request.dataset_path,
            config=config,
        )
        
        return TaskCreateResponse(
            task_id=task_id,
            message="Task created successfully. Use GET /tasks/{task_id} to monitor progress.",
        )
        
    except Exception as e:
        logger.error(
            "Failed to create task",
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create task: {str(e)}",
        )


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> TaskStatusResponse:
    """
    Get the current status of a task.
    
    Returns detailed information including:
    - Current phase and status
    - Progress percentage (if available)
    - Execution logs
    - Artifacts produced so far
    """
    logger.debug(
        "Getting task status",
        task_id=task_id,
    )
    
    state = await orchestrator.get_task_status(task_id)
    
    if not state:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found",
        )
    
    return TaskStatusResponse(
        task_id=task_id,
        phase=state.get("current_phase", "unknown"),
        status=state.get("remote_job_status", "unknown"),
        progress=state.get("remote_job_progress"),
        is_complete=state.get("is_complete", False),
        error_message=state.get("error_message"),
        logs=state.get("execution_logs", []),
        artifacts=TaskArtifacts(
            sft_model=state.get("sft_model_path"),
            quant_model=state.get("quant_model_path"),
            eval_report=state.get("eval_report"),
            deploy_url=state.get("deploy_url"),
        ),
        created_at=state.get("created_at"),
        updated_at=state.get("updated_at"),
    )


@router.delete("/{task_id}")
async def cancel_task(
    task_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> dict:
    """
    Cancel a running task.
    
    Attempts to stop the current remote job and marks the task as cancelled.
    """
    logger.info(
        "Cancelling task",
        task_id=task_id,
    )
    
    success = await orchestrator.cancel_task(task_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found or already completed",
        )
    
    return {
        "task_id": task_id,
        "message": "Task cancelled successfully",
    }


@router.post("/{task_id}/retry")
async def retry_task(
    task_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> dict:
    """
    Retry a failed task.
    
    Resets the error state and attempts to resume from the failed phase.
    """
    logger.info(
        "Retrying task",
        task_id=task_id,
    )
    
    state = await orchestrator.get_task_status(task_id)
    
    if not state:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found",
        )
    
    if state.get("current_phase") != "error":
        raise HTTPException(
            status_code=400,
            detail="Task is not in error state. Cannot retry.",
        )
    
    # TODO: Implement retry logic
    # This would reset the error state and resume the workflow
    
    raise HTTPException(
        status_code=501,
        detail="Retry functionality not yet implemented",
    )


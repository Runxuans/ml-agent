"""
Task-related request and response schemas.

These Pydantic models define the API contract and provide
automatic validation and documentation.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskCreateRequest(BaseModel):
    """Request body for creating a new training task."""
    
    base_model: str = Field(
        ...,
        description="Base model identifier (e.g., 'llama3-8b', 'qwen2-7b')",
        examples=["llama3-8b"],
    )
    dataset_path: str = Field(
        ...,
        description="Path to training dataset (S3, local, etc.)",
        examples=["s3://bucket/data.json"],
    )
    quantization: Optional[str] = Field(
        default="int4",
        description="Quantization type for the model",
        examples=["int4", "int8", "fp16"],
    )
    config: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional configuration parameters",
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "base_model": "llama3-8b",
                    "dataset_path": "s3://bucket/training-data.json",
                    "quantization": "int4",
                    "config": {
                        "epochs": 3,
                        "learning_rate": 2e-5,
                    },
                }
            ]
        }
    }


class TaskCreateResponse(BaseModel):
    """Response after creating a new task."""
    
    task_id: str = Field(
        ...,
        description="Unique identifier for tracking the task",
    )
    message: str = Field(
        default="Task created successfully",
        description="Human-readable status message",
    )


class TaskArtifacts(BaseModel):
    """Artifacts produced by the training pipeline."""
    
    sft_model: Optional[str] = Field(
        default=None,
        description="Path to fine-tuned model",
    )
    quant_model: Optional[str] = Field(
        default=None,
        description="Path to quantized model",
    )
    eval_report: Optional[dict[str, Any]] = Field(
        default=None,
        description="Evaluation metrics and report",
    )
    deploy_url: Optional[str] = Field(
        default=None,
        description="URL of deployed inference service",
    )


class TaskStatusResponse(BaseModel):
    """Detailed status of a task."""
    
    task_id: str = Field(
        ...,
        description="Task identifier",
    )
    phase: str = Field(
        ...,
        description="Current pipeline phase",
        examples=["sft", "quantization", "evaluation", "deployment", "completed", "error"],
    )
    status: str = Field(
        ...,
        description="Job status within current phase",
        examples=["idle", "pending", "running", "success", "failed"],
    )
    progress: Optional[float] = Field(
        default=None,
        description="Progress percentage (0-100) if available",
    )
    is_complete: bool = Field(
        ...,
        description="Whether the task has finished (success or error)",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error details if task failed",
    )
    logs: list[str] = Field(
        default_factory=list,
        description="Execution log entries",
    )
    artifacts: TaskArtifacts = Field(
        default_factory=TaskArtifacts,
        description="Outputs from completed phases",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Task creation timestamp",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp",
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task_id": "task-abc123def456",
                    "phase": "quantization",
                    "status": "running",
                    "progress": 45.0,
                    "is_complete": False,
                    "error_message": None,
                    "logs": [
                        "[2024-01-15T10:00:00] [SFT] Submitted job mock-sft-12345678",
                        "[2024-01-15T10:02:00] [SFT] Job completed successfully",
                        "[2024-01-15T10:02:01] [QUANTIZATION] Submitted job mock-quant-87654321",
                    ],
                    "artifacts": {
                        "sft_model": "s3://mock-bucket/outputs/sft/...",
                        "quant_model": None,
                        "eval_report": None,
                        "deploy_url": None,
                    },
                    "created_at": "2024-01-15T10:00:00Z",
                    "updated_at": "2024-01-15T10:02:30Z",
                }
            ]
        }
    }


class TaskListResponse(BaseModel):
    """List of tasks with summary info."""
    
    tasks: list[TaskStatusResponse] = Field(
        default_factory=list,
        description="List of task statuses",
    )
    total: int = Field(
        ...,
        description="Total number of tasks",
    )


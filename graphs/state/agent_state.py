"""
Agent state schema definition.

This is the core data structure that LangGraph maintains throughout
the entire lifecycle of a training pipeline. It's designed to be:
- Serializable: Can be persisted to PostgreSQL
- Extensible: New fields can be added without breaking existing checkpoints
- Self-documenting: Clear typing and field organization
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class JobPhase(str, Enum):
    """
    Pipeline phases that an agent can be in.
    
    The typical flow is:
    PENDING -> SFT -> QUANTIZATION -> EVALUATION -> DEPLOYMENT -> COMPLETED
    
    Any phase can transition to ERROR if something fails.
    """
    PENDING = "pending"           # Initial state, not yet started
    SFT = "sft"                   # Supervised fine-tuning
    QUANTIZATION = "quantization" # Model quantization
    EVALUATION = "evaluation"     # Model evaluation
    DEPLOYMENT = "deployment"     # Service deployment
    COMPLETED = "completed"       # All phases finished
    ERROR = "error"               # Encountered unrecoverable error
    CANCELLED = "cancelled"       # Manually cancelled by user


class JobStatus(str, Enum):
    """
    Status of the current phase's remote job.
    
    This tracks the state of the currently executing remote task,
    not the overall pipeline status.
    """
    IDLE = "idle"           # No remote job active (between phases)
    PENDING = "pending"     # Remote job submitted, awaiting start
    RUNNING = "running"     # Remote job actively executing
    SUCCESS = "success"     # Remote job completed successfully
    FAILED = "failed"       # Remote job failed


class AgentState(BaseModel):
    """
    Complete state for the training pipeline agent.
    
    This TypedDict-compatible model is what LangGraph persists in checkpoints.
    It contains everything needed to resume the agent from any point.
    
    State is organized into logical sections:
    - Input Configuration: User-provided parameters
    - Flow Control: Current position in the pipeline
    - Remote Job Tracking: Status of external tasks
    - Artifacts: Outputs from completed phases
    - Execution Log: Audit trail of all actions
    """
    
    # =========================================
    # Input Configuration
    # =========================================
    base_model: str = Field(
        ...,
        description="Base model identifier (e.g., 'llama3-8b', 'qwen2-7b')"
    )
    dataset_path: str = Field(
        ...,
        description="Path to training dataset (e.g., 's3://bucket/data.json')"
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration parameters (quantization type, hyperparams, etc.)"
    )
    
    # =========================================
    # Flow Control
    # =========================================
    thread_id: str = Field(
        ...,
        description="Unique identifier for this pipeline run, used for tracking and checkpointing"
    )
    current_phase: JobPhase = Field(
        default=JobPhase.PENDING,
        description="Current phase in the pipeline"
    )
    retry_count: int = Field(
        default=0,
        description="Number of retries attempted for current phase"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts before marking as failed"
    )
    
    # =========================================
    # Remote Job Tracking
    # =========================================
    remote_job_id: Optional[str] = Field(
        default=None,
        description="ID of the currently running remote task"
    )
    remote_job_status: JobStatus = Field(
        default=JobStatus.IDLE,
        description="Status of the current remote job"
    )
    remote_job_progress: Optional[float] = Field(
        default=None,
        description="Progress percentage (0-100) if available"
    )
    needs_wait: bool = Field(
        default=False,
        description="Flag indicating workflow should wait for next scheduler poll"
    )
    
    # =========================================
    # Artifacts (outputs from each phase)
    # =========================================
    sft_model_path: Optional[str] = Field(
        default=None,
        description="Path to the fine-tuned model output"
    )
    quant_model_path: Optional[str] = Field(
        default=None,
        description="Path to the quantized model output"
    )
    eval_report: Optional[dict[str, Any]] = Field(
        default=None,
        description="Evaluation metrics and report"
    )
    deploy_url: Optional[str] = Field(
        default=None,
        description="URL of deployed inference service"
    )
    
    # =========================================
    # Execution Log
    # =========================================
    execution_logs: list[str] = Field(
        default_factory=list,
        description="Chronological log of all significant events"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if current_phase is ERROR"
    )
    
    # =========================================
    # Timestamps
    # =========================================
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this pipeline run was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last state update timestamp"
    )
    
    def add_log(self, message: str) -> "AgentState":
        """Add a timestamped log entry."""
        timestamp = datetime.utcnow().isoformat()
        self.execution_logs.append(f"[{timestamp}] {message}")
        self.updated_at = datetime.utcnow()
        return self
    
    def to_checkpoint_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LangGraph checkpoint."""
        return self.model_dump(mode="json")
    
    @classmethod
    def from_checkpoint_dict(cls, data: dict[str, Any]) -> "AgentState":
        """Reconstruct from checkpoint dictionary."""
        return cls.model_validate(data)
    
    model_config = {"use_enum_values": True}


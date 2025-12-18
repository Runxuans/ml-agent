"""
Mock cluster client for Phase 1 testing.

Simulates a remote GPU cluster without actual computation.
Jobs transition through states over time to test the polling mechanism.

Key features:
- Configurable task duration
- Simulated progress updates
- Controllable failure modes for testing error handling
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from core.config import settings
from tools.cluster_api.base import (
    ClusterClient,
    ClusterError,
    JobInfo,
    JobNotCompleteError,
    JobNotFoundError,
    JobResult,
    RemoteJobStatus,
    TaskType,
)


class MockJob:
    """Internal representation of a mock job."""
    
    def __init__(
        self,
        job_id: str,
        task_type: TaskType,
        params: dict[str, Any],
        duration_seconds: int,
    ):
        self.job_id = job_id
        self.task_type = task_type
        self.params = params
        self.duration_seconds = duration_seconds
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self._status = RemoteJobStatus.PENDING
        self._error: Optional[str] = None
        self._cancelled = False
    
    @property
    def status(self) -> RemoteJobStatus:
        """Calculate current status based on elapsed time."""
        if self._cancelled:
            return RemoteJobStatus.CANCELLED
        if self._error:
            return RemoteJobStatus.FAILED
        if self._status == RemoteJobStatus.PENDING:
            # Start after brief delay
            if (datetime.utcnow() - self.created_at).total_seconds() > 2:
                self.started_at = datetime.utcnow()
                self._status = RemoteJobStatus.RUNNING
            return self._status
        if self._status == RemoteJobStatus.RUNNING:
            elapsed = (datetime.utcnow() - self.started_at).total_seconds()
            if elapsed >= self.duration_seconds:
                self.completed_at = datetime.utcnow()
                self._status = RemoteJobStatus.SUCCESS
        return self._status
    
    @property
    def progress(self) -> Optional[float]:
        """Calculate progress percentage."""
        if self.status == RemoteJobStatus.PENDING:
            return 0.0
        if self.status in (RemoteJobStatus.SUCCESS, RemoteJobStatus.FAILED):
            return 100.0
        if self.status == RemoteJobStatus.RUNNING and self.started_at:
            elapsed = (datetime.utcnow() - self.started_at).total_seconds()
            return min(99.0, (elapsed / self.duration_seconds) * 100)
        return None
    
    def cancel(self) -> bool:
        """Attempt to cancel the job."""
        if self.status in (RemoteJobStatus.PENDING, RemoteJobStatus.RUNNING):
            self._cancelled = True
            self.completed_at = datetime.utcnow()
            return True
        return False
    
    def force_fail(self, error: str) -> None:
        """Force the job to fail (for testing)."""
        self._error = error
        self.completed_at = datetime.utcnow()


class MockClusterClient(ClusterClient):
    """
    Mock implementation of ClusterClient for testing.
    
    Maintains an in-memory store of jobs and simulates time-based
    state transitions. Thread-safe for concurrent polling.
    
    Usage:
        client = MockClusterClient(task_duration=60)  # 60 second tasks
        
        # Submit and poll
        job = await client.submit_task(TaskType.SFT, {"model": "test"})
        while not (await client.get_job_status(job.job_id)).is_terminal:
            await asyncio.sleep(10)
        result = await client.get_job_result(job.job_id)
    """
    
    def __init__(
        self,
        task_duration: Optional[int] = None,
        failure_rate: float = 0.0,
    ):
        """
        Initialize mock client.
        
        Args:
            task_duration: Seconds for each task. Defaults to config value.
            failure_rate: Probability (0-1) of random failure for testing.
        """
        self._jobs: dict[str, MockJob] = {}
        self._task_duration = task_duration or settings.mock_task_duration_seconds
        self._failure_rate = failure_rate
        self._lock = asyncio.Lock()
    
    async def submit_task(
        self,
        task_type: TaskType,
        params: dict[str, Any],
    ) -> JobInfo:
        """Submit a mock task."""
        job_id = f"mock-{task_type.value}-{uuid.uuid4().hex[:8]}"
        
        async with self._lock:
            job = MockJob(
                job_id=job_id,
                task_type=task_type,
                params=params,
                duration_seconds=self._task_duration,
            )
            self._jobs[job_id] = job
        
        # Simulate network latency
        await asyncio.sleep(0.1)
        
        return JobInfo(
            job_id=job_id,
            status=RemoteJobStatus.PENDING,
            task_type=task_type,
            progress=0.0,
            message=f"Task {task_type.value} submitted successfully",
        )
    
    async def get_job_status(self, job_id: str) -> JobInfo:
        """Get current status of a mock job."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise JobNotFoundError(f"Job {job_id} not found")
            
            return JobInfo(
                job_id=job.job_id,
                status=job.status,
                task_type=job.task_type,
                progress=job.progress,
                message=self._get_status_message(job),
                started_at=job.started_at,
                completed_at=job.completed_at,
            )
    
    async def get_job_result(self, job_id: str) -> JobResult:
        """Get results of a completed mock job."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise JobNotFoundError(f"Job {job_id} not found")
            
            if job.status not in (
                RemoteJobStatus.SUCCESS,
                RemoteJobStatus.FAILED,
                RemoteJobStatus.CANCELLED,
            ):
                raise JobNotCompleteError(
                    f"Job {job_id} is still {job.status.value}"
                )
            
            return JobResult(
                job_id=job.job_id,
                task_type=job.task_type,
                success=job.status == RemoteJobStatus.SUCCESS,
                output_path=self._generate_output_path(job) if job.status == RemoteJobStatus.SUCCESS else None,
                metrics=self._generate_metrics(job) if job.status == RemoteJobStatus.SUCCESS else None,
                error_message=job._error,
                logs=self._generate_logs(job),
            )
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a mock job."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise JobNotFoundError(f"Job {job_id} not found")
            return job.cancel()
    
    def _get_status_message(self, job: MockJob) -> str:
        """Generate human-readable status message."""
        messages = {
            RemoteJobStatus.PENDING: f"Waiting for resources ({job.task_type.value})",
            RemoteJobStatus.RUNNING: f"Processing {job.task_type.value} ({job.progress:.1f}%)",
            RemoteJobStatus.SUCCESS: f"Task {job.task_type.value} completed successfully",
            RemoteJobStatus.FAILED: f"Task {job.task_type.value} failed: {job._error}",
            RemoteJobStatus.CANCELLED: f"Task {job.task_type.value} was cancelled",
        }
        return messages.get(job.status, "Unknown status")
    
    def _generate_output_path(self, job: MockJob) -> str:
        """Generate mock output path based on task type."""
        base_path = "s3://mock-bucket/outputs"
        paths = {
            TaskType.SFT: f"{base_path}/sft/{job.job_id}/model",
            TaskType.QUANTIZATION: f"{base_path}/quant/{job.job_id}/model.int4",
            TaskType.EVALUATION: f"{base_path}/eval/{job.job_id}/report.json",
            TaskType.DEPLOYMENT: f"https://inference.mock-cluster.com/{job.job_id}",
        }
        return paths.get(job.task_type, f"{base_path}/{job.job_id}")
    
    def _generate_metrics(self, job: MockJob) -> dict[str, Any]:
        """Generate mock metrics based on task type."""
        metrics: dict[TaskType, dict[str, Any]] = {
            TaskType.SFT: {
                "final_loss": 0.342,
                "epochs": 3,
                "training_time_hours": 2.5,
            },
            TaskType.QUANTIZATION: {
                "compression_ratio": 4.0,
                "quantization_type": "int4",
                "perplexity_increase": 0.02,
            },
            TaskType.EVALUATION: {
                "accuracy": 0.87,
                "f1_score": 0.85,
                "latency_ms": 45.2,
            },
            TaskType.DEPLOYMENT: {
                "replicas": 2,
                "gpu_type": "A100",
                "avg_latency_ms": 32.1,
            },
        }
        return metrics.get(job.task_type, {})
    
    def _generate_logs(self, job: MockJob) -> list[str]:
        """Generate mock execution logs."""
        logs = [
            f"[{job.created_at.isoformat()}] Task {job.task_type.value} created",
        ]
        if job.started_at:
            logs.append(f"[{job.started_at.isoformat()}] Task started execution")
        if job.completed_at:
            status_str = "completed" if job.status == RemoteJobStatus.SUCCESS else "terminated"
            logs.append(f"[{job.completed_at.isoformat()}] Task {status_str}")
        return logs
    
    # =========================================
    # Testing utilities
    # =========================================
    
    async def force_job_failure(self, job_id: str, error: str) -> None:
        """Force a job to fail (for testing error handling)."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.force_fail(error)
    
    async def set_task_duration(self, seconds: int) -> None:
        """Change task duration for new tasks."""
        self._task_duration = seconds
    
    async def clear_all_jobs(self) -> None:
        """Clear all jobs (for testing)."""
        async with self._lock:
            self._jobs.clear()


"""
Job status checking node operations.

This node handles polling remote jobs and updating state based on results.
It's the most frequently executed node during long-running tasks.
"""

from datetime import datetime
from typing import Any, Optional

from core.logging import get_logger
from graphs.state.agent_state import AgentState, JobPhase, JobStatus
from tools.cluster_api.base import (
    ClusterClient,
    JobNotFoundError,
    RemoteJobStatus,
)


logger = get_logger(__name__)


# Phase transition mapping
PHASE_TRANSITIONS: dict[JobPhase, JobPhase] = {
    JobPhase.PENDING: JobPhase.SFT,
    JobPhase.SFT: JobPhase.QUANTIZATION,
    JobPhase.QUANTIZATION: JobPhase.EVALUATION,
    JobPhase.EVALUATION: JobPhase.DEPLOYMENT,
    JobPhase.DEPLOYMENT: JobPhase.COMPLETED,
}


def _get_next_phase(current: JobPhase) -> JobPhase:
    """Get the next phase in the pipeline."""
    return PHASE_TRANSITIONS.get(current, JobPhase.COMPLETED)


def _update_artifacts(
    state: AgentState,
    phase: JobPhase,
    output_path: Optional[str],
    metrics: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build artifact updates based on completed phase.
    
    Each phase produces different outputs that need to be stored
    in the appropriate state fields.
    """
    updates: dict[str, Any] = {}
    
    if phase == JobPhase.SFT and output_path:
        updates["sft_model_path"] = output_path
    elif phase == JobPhase.QUANTIZATION and output_path:
        updates["quant_model_path"] = output_path
    elif phase == JobPhase.EVALUATION:
        updates["eval_report"] = metrics
    elif phase == JobPhase.DEPLOYMENT and output_path:
        updates["deploy_url"] = output_path
    
    return updates


async def check_status_node(
    state: dict[str, Any],
    cluster_client: ClusterClient,
) -> dict[str, Any]:
    """
    检查远程任务的状态

    参数:
        state: 当前的图状态 (dict 形式)
        cluster_client: 工作集群客户端
        
    返回:
        状态更新 
    """
    agent_state = AgentState.model_validate(state)
    job_id = agent_state.remote_job_id
    phase = JobPhase(agent_state.current_phase)
    
    if not job_id:
        logger.warning(
            "No remote job to check",
            thread_id=agent_state.thread_id,
            phase=phase.value,
        )
        return {**state}
    
    logger.debug(
        "Checking job status",
        thread_id=agent_state.thread_id,
        job_id=job_id,
        phase=phase.value,
    )
    
    try:
        job_info = await cluster_client.get_job_status(job_id)
        
        if job_info.status == RemoteJobStatus.RUNNING:
            logger.debug(
                "Job still running",
                thread_id=agent_state.thread_id,
                job_id=job_id,
                progress=job_info.progress,
            )
            return {
                **state,
                "remote_job_progress": job_info.progress,
                "remote_job_status": JobStatus.RUNNING.value,
                "needs_wait": True,  # 标记需要等待，防止死循环
                "updated_at": datetime.utcnow(),
            }
        
        if job_info.status == RemoteJobStatus.PENDING:
            logger.debug(
                "Job pending",
                thread_id=agent_state.thread_id,
                job_id=job_id,
            )
            return {
                **state,
                "remote_job_status": JobStatus.PENDING.value,
                "needs_wait": True,  # 标记需要等待，防止死循环
                "updated_at": datetime.utcnow(),
            }
        
        # Job completed successfully
        if job_info.status == RemoteJobStatus.SUCCESS:
            result = await cluster_client.get_job_result(job_id)
            next_phase = _get_next_phase(phase)
            
            log_message = f"[{phase.value.upper()}] Job {job_id} completed successfully"
            
            logger.info(
                "Job completed successfully",
                thread_id=agent_state.thread_id,
                job_id=job_id,
                phase=phase.value,
                next_phase=next_phase.value,
                output_path=result.output_path,
            )
            
            # Build artifact updates
            artifact_updates = _update_artifacts(
                agent_state, phase, result.output_path, result.metrics
            )
            
            return {
                **state,
                **artifact_updates,
                "current_phase": next_phase.value,
                "remote_job_id": None,
                "remote_job_status": JobStatus.IDLE.value,
                "remote_job_progress": None,
                "retry_count": 0,
                "needs_wait": False,  # 任务完成，清除等待标志
                "execution_logs": agent_state.execution_logs + [
                    f"[{datetime.utcnow().isoformat()}] {log_message}"
                ],
                "updated_at": datetime.utcnow(),
            }
        
        # Job failed
        if job_info.status == RemoteJobStatus.FAILED:
            return await _handle_failure(
                state, agent_state, cluster_client, job_id, phase
            )
        
        # Job cancelled
        if job_info.status == RemoteJobStatus.CANCELLED:
            logger.info(
                "Job was cancelled",
                thread_id=agent_state.thread_id,
                job_id=job_id,
            )
            return {
                **state,
                "current_phase": JobPhase.ERROR.value,
                "remote_job_status": JobStatus.FAILED.value,
                "error_message": f"Job {job_id} was cancelled",
                "execution_logs": agent_state.execution_logs + [
                    f"[{datetime.utcnow().isoformat()}] [CANCELLED] Job {job_id} was cancelled"
                ],
                "updated_at": datetime.utcnow(),
            }
            
    except JobNotFoundError:
        logger.error(
            "Job not found in cluster",
            thread_id=agent_state.thread_id,
            job_id=job_id,
        )
        return {
            **state,
            "current_phase": JobPhase.ERROR.value,
            "remote_job_status": JobStatus.FAILED.value,
            "error_message": f"Job {job_id} not found in cluster",
            "updated_at": datetime.utcnow(),
        }
        
    except Exception as e:
        logger.error(
            "Error checking job status",
            thread_id=agent_state.thread_id,
            job_id=job_id,
            error=str(e),
        )
        # Network errors - increment retry but don't fail immediately
        return {
            **state,
            "retry_count": agent_state.retry_count + 1,
            "execution_logs": agent_state.execution_logs + [
                f"[{datetime.utcnow().isoformat()}] [WARN] Status check failed: {str(e)}"
            ],
            "updated_at": datetime.utcnow(),
        }


async def _handle_failure(
    raw_state: dict[str, Any],
    agent_state: AgentState,
    cluster_client: ClusterClient,
    job_id: str,
    phase: JobPhase,
) -> dict[str, Any]:
    """
    Handle a failed job.
    
    Implements retry logic with configurable max retries.
    """
    result = await cluster_client.get_job_result(job_id)
    error_msg = result.error_message or "Unknown error"
    
    # Check if we should retry
    if agent_state.retry_count < agent_state.max_retries:
        logger.warning(
            "Job failed, will retry",
            thread_id=agent_state.thread_id,
            job_id=job_id,
            retry_count=agent_state.retry_count + 1,
            max_retries=agent_state.max_retries,
            error=error_msg,
        )
        return {
            **raw_state,
            "remote_job_id": None,
            "remote_job_status": JobStatus.IDLE.value,
            "retry_count": agent_state.retry_count + 1,
            "execution_logs": agent_state.execution_logs + [
                f"[{datetime.utcnow().isoformat()}] [RETRY] Job {job_id} failed: {error_msg}. "
                f"Retrying ({agent_state.retry_count + 1}/{agent_state.max_retries})"
            ],
            "updated_at": datetime.utcnow(),
        }
    
    # Max retries exceeded - mark as error
    logger.error(
        "Job failed, max retries exceeded",
        thread_id=agent_state.thread_id,
        job_id=job_id,
        retry_count=agent_state.retry_count,
        error=error_msg,
    )
    return {
        **raw_state,
        "current_phase": JobPhase.ERROR.value,
        "remote_job_id": None,
        "remote_job_status": JobStatus.FAILED.value,
        "error_message": f"Max retries exceeded for {phase.value}: {error_msg}",
        "execution_logs": agent_state.execution_logs + [
            f"[{datetime.utcnow().isoformat()}] [ERROR] Job {job_id} failed permanently: {error_msg}"
        ],
        "updated_at": datetime.utcnow(),
    }


def create_check_node(cluster_client: ClusterClient):
    """
    Factory function to create a check node with injected dependencies.
    """
    async def node(state: dict[str, Any]) -> dict[str, Any]:
        return await check_status_node(state, cluster_client)
    return node


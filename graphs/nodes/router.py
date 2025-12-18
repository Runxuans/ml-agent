"""
Router node for conditional edge routing.

This node determines the next step in the workflow based on
the current state. It's the decision-making hub of the graph.
"""

from typing import Literal

from graphs.state.agent_state import AgentState, JobPhase, JobStatus


# Routing destinations
RouteDecision = Literal["submit", "check", "wait", "complete", "error"]


def route_node(state: dict) -> RouteDecision:
    """
    决定下一步执行哪个节点。
    
    路由逻辑:
    1. needs_wait=True -> "wait" (等待调度器下次轮询)
    2. ERROR phase -> "error" (终态)
    3. COMPLETED phase -> "complete" (终态)
    4. 无远程任务且未完成 -> "submit" (提交新阶段任务)
    5. 有远程任务且 running/pending -> "check" (检查状态)
    6. 默认 -> "wait"
    """
    agent_state = AgentState.model_validate(state)
    phase = JobPhase(agent_state.current_phase)
    job_status = JobStatus(agent_state.remote_job_status)
    
    # 关键: 如果已检查过且任务仍在运行，等待下次调度
    if agent_state.needs_wait:
        return "wait"
    
    if phase == JobPhase.ERROR:
        return "error"
    
    if phase == JobPhase.COMPLETED:
        return "complete"
    
    # 无活跃任务 - 需要提交
    if agent_state.remote_job_id is None and job_status == JobStatus.IDLE:
        if phase in (
            JobPhase.PENDING,
            JobPhase.SFT,
            JobPhase.QUANTIZATION,
            JobPhase.EVALUATION,
            JobPhase.DEPLOYMENT,
        ):
            return "submit"
    
    # 有活跃任务且正在运行 - 检查状态
    if agent_state.remote_job_id is not None:
        if job_status in (JobStatus.RUNNING, JobStatus.PENDING):
            return "check"
    
    if job_status == JobStatus.SUCCESS:
        return "submit"
    
    if job_status == JobStatus.FAILED:
        if agent_state.retry_count < agent_state.max_retries:
            return "submit"
        return "error"
    
    return "wait"


def should_continue(state: dict) -> bool:
    """
    Check if the workflow should continue processing.
    
    Used to determine if we should keep the graph running
    or end this invocation (to be resumed by scheduler).
    
    Returns True if we should continue to next node.
    Returns False if we should pause and wait for scheduler.
    """
    agent_state = AgentState.model_validate(state)
    phase = JobPhase(agent_state.current_phase)
    job_status = JobStatus(agent_state.remote_job_status)
    
    # Don't continue if in terminal state
    if phase in (JobPhase.COMPLETED, JobPhase.ERROR):
        return False
    
    # Don't continue if waiting for remote job
    if job_status in (JobStatus.RUNNING, JobStatus.PENDING):
        return False
    
    # Continue if we need to submit or process
    return True


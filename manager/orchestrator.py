"""
Agent 编排器 - 统一管理所有 Agent 实例。

作为 API 层、LangGraph 工作流、数据库和调度器之间的桥梁，
提供任务生命周期管理的统一接口。
"""

from typing import Any, Optional

from core.config import settings
from core.logging import get_logger
from core.storage import (
    BaseCheckpointer,
    BaseJobIndexRepository,
    JobIndexRecord,
    create_checkpointer,
    create_job_index_repository,
)
from graphs.state.agent_state import JobPhase, JobStatus
from graphs.workflows.training_flow import TrainingWorkflow, create_training_workflow
from tools.cluster_api.base import ClusterClient


logger = get_logger(__name__)


class AgentOrchestrator:
    """
    Agent 生命周期管理器。
    
    - 创建新训练任务
    - 恢复已有任务
    - 查询任务状态
    - 同步 LangGraph 状态索引表
    
    维护单一工作流实例，通过 thread_id 区分不同任务。
    使用抽象存储接口，支持多种后端（MongoDB、PostgreSQL）。
    """
    
    def __init__(
        self,
        cluster_client: ClusterClient,
        checkpointer: Optional[BaseCheckpointer] = None,
        job_index_repository: Optional[BaseJobIndexRepository] = None,
    ):
        """
        初始化编排器。
        
        Args:
            cluster_client: 集群 API 客户端
            checkpointer: 可选的 checkpoint 存储实例（默认从配置创建）
            job_index_repository: 可选的作业索引仓库（默认从配置创建）
        """
        self.cluster_client = cluster_client
        self._checkpointer = checkpointer
        self._job_index_repo = job_index_repository
        self._workflow: Optional[TrainingWorkflow] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化编排器，包括存储后端和工作流。"""
        if self._initialized:
            return
        
        logger.info(
            "Initializing orchestrator",
            storage_backend=settings.storage_backend,
        )
        
        # 创建或使用提供的存储实例
        if self._checkpointer is None:
            self._checkpointer = create_checkpointer(settings)
        
        if self._job_index_repo is None:
            self._job_index_repo = create_job_index_repository(settings)
        
        # 初始化存储
        await self._checkpointer.setup_async()
        await self._job_index_repo.setup()
        
        # 创建工作流
        self._workflow = create_training_workflow(
            cluster_client=self.cluster_client,
            checkpointer=self._checkpointer.get_native_checkpointer(),
        )
        
        self._initialized = True
        logger.info(
            "Orchestrator initialized",
            storage_backend=settings.storage_backend,
        )
    
    async def shutdown(self) -> None:
        """关闭编排器，清理资源。"""
        logger.info("Shutting down orchestrator")
        
        if self._checkpointer is not None:
            await self._checkpointer.close()
        
        if self._job_index_repo is not None:
            await self._job_index_repo.close()
        
        self._initialized = False
        logger.info("Orchestrator shut down")
    
    async def create_task(
        self,
        base_model: str,
        dataset_path: str,
        config: Optional[dict[str, Any]] = None,
    ) -> str:
        """创建新的训练任务，返回 task_id。"""
        self._ensure_initialized()
        
        import uuid
        thread_id = f"task-{uuid.uuid4().hex[:12]}"
        
        logger.info(
            "Creating new task",
            thread_id=thread_id,
            base_model=base_model,
            dataset_path=dataset_path,
        )
        
        initial_state = {
            "thread_id": thread_id,
            "base_model": base_model,
            "dataset_path": dataset_path,
            "config": config or {},
            "current_phase": JobPhase.SFT.value,
            "remote_job_status": JobStatus.IDLE.value,
            "execution_logs": [],
            "retry_count": 0,
            "max_retries": 3,
        }
        
        # 创建作业索引记录
        await self._job_index_repo.create(
            JobIndexRecord(
                thread_id=thread_id,
                current_phase=initial_state["current_phase"],
                status="pending",
            )
        )
        
        # 执行工作流
        result = await self._workflow.invoke(initial_state, thread_id)
        
        # 同步状态到索引
        await self.sync_job_status(thread_id, result)
        
        logger.info(
            "Task created",
            thread_id=thread_id,
            phase=result.get("current_phase"),
            job_status=result.get("remote_job_status"),
        )
        
        return thread_id
    
    async def resume_task(self, thread_id: str) -> Optional[dict[str, Any]]:
        """从 checkpoint 恢复任务，由调度器调用。"""
        self._ensure_initialized()
        
        logger.debug(
            "Resuming task",
            thread_id=thread_id,
        )
        
        result = await self._workflow.resume(thread_id)
        return result
    
    async def get_task_status(self, thread_id: str) -> Optional[dict[str, Any]]:
        """获取任务当前状态，直接从 LangGraph checkpoint 读取。"""
        self._ensure_initialized()
        
        state = await self._workflow.get_state(thread_id)
        
        if state:
            state["task_id"] = thread_id
            state["is_complete"] = state.get("current_phase") in (
                JobPhase.COMPLETED.value,
                JobPhase.ERROR.value,
            )
        
        return state
    
    async def get_active_jobs(self) -> list[str]:
        """获取需要处理的任务列表，供调度器轮询使用。"""
        self._ensure_initialized()
        return await self._job_index_repo.get_active_jobs()
    
    async def sync_job_status(
        self,
        thread_id: str,
        state: dict[str, Any],
    ) -> None:
        """同步 LangGraph 状态到索引表，每次工作流调用后调用。"""
        self._ensure_initialized()
        
        phase = state.get("current_phase", JobPhase.PENDING.value)
        job_status = state.get("remote_job_status", JobStatus.IDLE.value)
        remote_job_id = state.get("remote_job_id")
        error_message = state.get("error_message")
        
        # 根据状态确定索引状态
        if phase in (JobPhase.COMPLETED.value, JobPhase.ERROR.value):
            index_status = "completed" if phase == JobPhase.COMPLETED.value else "failed"
        elif job_status in (JobStatus.RUNNING.value, JobStatus.PENDING.value):
            index_status = "running"
        else:
            index_status = "pending"
        
        await self._job_index_repo.update(
            thread_id,
            current_phase=phase,
            remote_job_id=remote_job_id,
            status=index_status,
            error_message=error_message,
        )
    
    async def cancel_task(self, thread_id: str) -> bool:
        """取消运行中的任务，尝试取消远端任务并标记为已取消。"""
        self._ensure_initialized()
        
        state = await self.get_task_status(thread_id)
        if not state:
            return False
        
        remote_job_id = state.get("remote_job_id")
        if remote_job_id:
            try:
                await self.cluster_client.cancel_job(remote_job_id)
            except Exception as e:
                logger.warning(
                    "Failed to cancel remote job",
                    thread_id=thread_id,
                    remote_job_id=remote_job_id,
                    error=str(e),
                )
        
        # 使用仓库接口更新状态
        await self._job_index_repo.set_cancelled(thread_id)
        
        logger.info("Task cancelled", thread_id=thread_id)
        return True
    
    def _ensure_initialized(self) -> None:
        """检查是否已初始化，未初始化则抛出异常。"""
        if not self._initialized:
            raise RuntimeError(
                "Orchestrator not initialized. Call initialize() first."
            )

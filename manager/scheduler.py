"""
Background scheduler for polling active jobs.

Uses APScheduler to periodically wake up and check on running tasks.
This is the mechanism that makes long-running async workflows possible.

Design principles:
- Non-blocking: Scheduler runs in background, doesn't block API
- Resilient: Survives individual task failures
- Configurable: Poll interval and concurrency are adjustable
"""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from core.config import settings
from core.logging import get_logger

if TYPE_CHECKING:
    from manager.orchestrator import AgentOrchestrator


logger = get_logger(__name__)


class JobScheduler:
    """
    Background scheduler for periodic job polling.
    
    Integrates with APScheduler to periodically invoke the orchestrator
    to check on active jobs and drive state transitions.
    
    Features:
    - Configurable poll interval
    - Graceful shutdown
    - Error isolation (one failed poll doesn't stop the scheduler)
    
    Usage:
        scheduler = JobScheduler(orchestrator)
        await scheduler.start()
        # ... application runs ...
        await scheduler.shutdown()
    """
    
    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        poll_interval_seconds: Optional[int] = None,
        max_concurrent_jobs: Optional[int] = None,
    ):
        """
        Initialize the scheduler.
        
        Args:
            orchestrator: The agent orchestrator to drive
            poll_interval_seconds: Seconds between polls (default from config)
            max_concurrent_jobs: Max jobs to process per poll (default from config)
        """
        self.orchestrator = orchestrator
        self.poll_interval = poll_interval_seconds or settings.scheduler_poll_interval_seconds
        self.max_concurrent = max_concurrent_jobs or settings.scheduler_max_concurrent_jobs
        
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._is_polling = False
        self._poll_lock = asyncio.Lock()
    
    async def start(self) -> None:
        """
        Start the background scheduler.
        
        Should be called during application startup (in lifespan).
        """
        if self._scheduler is not None:
            logger.warning("Scheduler already running")
            return
        
        self._scheduler = AsyncIOScheduler()
        
        # Add the polling job
        self._scheduler.add_job(
            self._poll_jobs,
            trigger=IntervalTrigger(seconds=self.poll_interval),
            id="poll_active_jobs",
            name="Poll active jobs",
            max_instances=1,  # Prevent overlapping polls
            replace_existing=True,
        )
        
        self._scheduler.start()
        
        logger.info(
            "Scheduler started",
            poll_interval_seconds=self.poll_interval,
            max_concurrent_jobs=self.max_concurrent,
        )
        
        # Run initial poll immediately
        asyncio.create_task(self._poll_jobs())
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the scheduler.
        
        Waits for current poll to complete before stopping.
        """
        if self._scheduler is None:
            return
        
        logger.info("Shutting down scheduler...")
        
        # Wait for any ongoing poll to complete
        async with self._poll_lock:
            self._scheduler.shutdown(wait=True)
            self._scheduler = None
        
        logger.info("Scheduler shut down")
    
    async def _poll_jobs(self) -> None:
        """
        Single poll cycle: find active jobs and process them.
        
        This is called periodically by APScheduler.
        """
        if self._is_polling:
            logger.debug("Poll already in progress, skipping")
            return
        
        async with self._poll_lock:
            self._is_polling = True
            try:
                await self._do_poll()
            except Exception as e:
                logger.error(
                    "Poll cycle failed",
                    error=str(e),
                    exc_info=True,
                )
            finally:
                self._is_polling = False
    
    async def _do_poll(self) -> None:
        """
        Execute the poll logic.
        
        1. Query database for running jobs
        2. Resume each job's workflow
        3. Update job status based on results
        """
        logger.debug("Starting poll cycle")
        
        # Get list of active job thread_ids
        active_jobs = await self.orchestrator.get_active_jobs()
        
        if not active_jobs:
            logger.debug("No active jobs to process")
            return
        
        logger.info(
            "Processing active jobs",
            count=len(active_jobs),
        )
        
        # Process jobs with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(thread_id: str):
            async with semaphore:
                await self._process_job(thread_id)
        
        # Process all jobs concurrently (up to max_concurrent)
        tasks = [
            asyncio.create_task(process_with_semaphore(job_id))
            for job_id in active_jobs[:self.max_concurrent]  # Limit total per poll
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.debug("Poll cycle completed")
    
    async def _process_job(self, thread_id: str) -> None:
        """
        Process a single job.
        
        Resumes the workflow and updates the database index.
        """
        try:
            logger.debug(
                "Processing job",
                thread_id=thread_id,
            )
            
            # Resume the workflow
            result = await self.orchestrator.resume_task(thread_id)
            
            if result:
                # Update the active_jobs index
                await self.orchestrator.sync_job_status(thread_id, result)
            
        except Exception as e:
            logger.error(
                "Failed to process job",
                thread_id=thread_id,
                error=str(e),
            )
            # Don't re-raise - let other jobs continue
    
    async def trigger_poll(self) -> None:
        """
        Manually trigger a poll cycle.
        
        Useful for testing or when you want immediate processing
        after submitting a new task.
        """
        await self._poll_jobs()
    
    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._scheduler is not None and self._scheduler.running


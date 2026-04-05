"""Background job manager — runs agent tasks asynchronously with progress tracking."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from agent.schemas import AgentResult, Job, JobStatus

logger = logging.getLogger(__name__)

# Max completed jobs to keep in memory before eviction
_MAX_COMPLETED = 50


class JobManager:
    """Submit, track, and notify about background agent jobs.

    Jobs run as asyncio.Tasks, gated by a semaphore for concurrency control.
    Job metadata is persisted to Neo4j (if available) as the single source of truth.
    In-memory dict holds active + recent jobs for fast lookups and result delivery.
    """

    def __init__(
        self,
        agent: Any,  # Agent (avoid circular import)
        memory: Any,  # MemoryClient
        max_concurrent: int = 3,
    ):
        self._agent = agent
        self._memory = memory
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._tasks: dict[str, asyncio.Task] = {}

        # Callbacks set by the interface (Telegram bot)
        self._on_progress: Callable[[Job], Awaitable[None]] | None = None
        self._on_complete: Callable[[Job], Awaitable[None]] | None = None

    def set_progress_callback(self, cb: Callable[[Job], Awaitable[None]]) -> None:
        self._on_progress = cb

    def set_completion_callback(self, cb: Callable[[Job], Awaitable[None]]) -> None:
        self._on_complete = cb

    async def submit(
        self,
        user_input: str,
        chat_id: int = 0,
        message_id: int | None = None,
    ) -> Job:
        """Submit a new background job. Returns immediately."""
        job = Job(
            id=str(uuid.uuid4())[:16],
            user_input=user_input,
            status=JobStatus.pending,
            chat_id=chat_id,
            message_id=message_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._jobs[job.id] = job
        self._evict_old()

        # Persist to Neo4j (fire-and-forget)
        asyncio.ensure_future(self._persist(job))

        # Launch background task
        task = asyncio.create_task(self._run_job(job))
        self._tasks[job.id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job.id, None))

        logger.info("Job %s submitted: %s", job.id, user_input[:80])
        return job

    async def _run_job(self, job: Job) -> None:
        """Execute agent.run() in background with progress tracking."""
        async with self._semaphore:
            # Mark running
            job.status = JobStatus.running
            job.started_at = datetime.now(timezone.utc).isoformat()
            job.current_phase = "starting"
            await self._persist(job)

            async def _progress(phase: str, detail: str) -> None:
                job.current_phase = phase
                job.progress = detail
                # Throttle Neo4j updates — only persist on phase change
                if self._on_progress:
                    try:
                        await self._on_progress(job)
                    except Exception:
                        pass

            try:
                result = await self._agent.run(job.user_input, on_progress=_progress)

                # Success
                job.status = JobStatus.done
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.current_phase = "done"
                job.result = result
                job.episode_id = getattr(self._agent, "_last_episode_id", None)

                await self._persist(job)
                if job.episode_id:
                    await self._memory.link_job_episode(job.id, job.episode_id)

                logger.info("Job %s completed", job.id)

            except Exception as exc:
                job.status = JobStatus.failed
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.current_phase = "failed"
                job.error = str(exc)
                await self._persist(job)
                logger.error("Job %s failed: %s", job.id, exc)

            # Notify completion
            if self._on_complete:
                try:
                    await self._on_complete(job)
                except Exception as exc:
                    logger.warning("Completion callback failed: %s", exc)

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def get_active_jobs(self, chat_id: int | None = None) -> list[Job]:
        """Get pending/running jobs, optionally filtered by chat_id."""
        return [
            j for j in self._jobs.values()
            if j.status in (JobStatus.pending, JobStatus.running)
            and (chat_id is None or j.chat_id == chat_id)
        ]

    def get_recent_jobs(self, chat_id: int | None = None, limit: int = 5) -> list[Job]:
        """Get recent jobs (all statuses), most recent first."""
        jobs = [
            j for j in reversed(self._jobs.values())
            if chat_id is None or j.chat_id == chat_id
        ]
        return jobs[:limit]

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.failed
                job.error = "Cancelled by user"
                job.completed_at = datetime.now(timezone.utc).isoformat()
                await self._persist(job)
            logger.info("Job %s cancelled", job_id)
            return True
        return False

    async def _persist(self, job: Job) -> None:
        """Persist job state to Neo4j (best effort)."""
        try:
            await self._memory.persist_job(
                job_id=job.id,
                user_input=job.user_input,
                status=job.status.value,
                chat_id=job.chat_id,
                current_phase=job.current_phase,
                progress=job.progress,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                error=job.error,
            )
        except Exception as exc:
            logger.debug("Job persist to Neo4j failed (non-critical): %s", exc)

    def _evict_old(self) -> None:
        """Remove old completed jobs from memory to prevent leaks."""
        completed = [
            jid for jid, j in self._jobs.items()
            if j.status in (JobStatus.done, JobStatus.failed)
        ]
        while len(completed) > _MAX_COMPLETED:
            oldest = completed.pop(0)
            self._jobs.pop(oldest, None)

"""
job_manager.py — In-memory asynchronous background job manager for resume_builder.

Enables:
- Non-blocking execution of heavy document parsing, CV generation, and ATS analysis.
- Worker-compatible task scheduling without locking FastAPI HTTP request threads.
- Real-time job lifecycle tracking (QUEUED -> PROCESSING -> COMPLETED / FAILED).
- Automatic TTL pruning of completed jobs (zero database writes or tables).
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("resume_builder.job_manager")


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    status: JobStatus = JobStatus.QUEUED
    user_id: Optional[str] = None
    progress: int = 0
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class AsyncJobManager:
    """
    Thread-safe and async-safe in-memory job manager.
    Designed to seamlessly bridge current HTTP endpoints with future background worker fleets.
    """

    def __init__(self, max_records: int = 2000, job_ttl_seconds: float = 3600.0):
        self._jobs: Dict[str, JobRecord] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._max_records = max_records
        self._job_ttl_seconds = job_ttl_seconds

    async def submit_job(
        self,
        job_type: str,
        coro_fn: Callable[..., Any],
        *args: Any,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Submits an asynchronous coroutine for background execution and returns job_id instantly.
        """
        job_id = str(uuid.uuid4())
        record = JobRecord(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.QUEUED,
            user_id=user_id or "anonymous",
            progress=0,
        )

        async with self._lock:
            self._prune_expired_locked()
            self._jobs[job_id] = record

        # Spawn background task
        task = asyncio.create_task(
            self._execute_job_wrapper(job_id, coro_fn, *args, **kwargs),
            name=f"job_{job_type}_{job_id}",
        )
        self._tasks[job_id] = task

        logger.info(f"[JobManager] Enqueued background job '{job_id}' (type={job_type}, user={user_id})")
        return job_id

    async def _execute_job_wrapper(
        self,
        job_id: str,
        coro_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Internal execution wrapper tracking status and exceptions."""
        record = self._jobs.get(job_id)
        if not record:
            return

        record.status = JobStatus.PROCESSING
        record.started_at = time.time()
        record.progress = 10

        try:
            # Execute coroutine
            result = await coro_fn(*args, **kwargs)

            record.status = JobStatus.COMPLETED
            record.progress = 100
            record.result = result
            record.completed_at = time.time()
            record.latency_ms = round((record.completed_at - record.started_at) * 1000, 2)
            logger.info(f"[JobManager] ✓ Job '{job_id}' COMPLETED in {record.latency_ms}ms")

        except asyncio.CancelledError:
            record.status = JobStatus.CANCELLED
            record.completed_at = time.time()
            record.error = "Job was cancelled by client"
            logger.warning(f"[JobManager] Job '{job_id}' was CANCELLED")

        except Exception as e:
            record.status = JobStatus.FAILED
            record.completed_at = time.time()
            record.error = str(e)
            record.latency_ms = round((record.completed_at - record.started_at) * 1000, 2)
            logger.error(f"[JobManager] ❌ Job '{job_id}' FAILED after {record.latency_ms}ms: {e}", exc_info=True)

        finally:
            self._tasks.pop(job_id, None)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def get_job_dict(self, job_id: str) -> Optional[Dict[str, Any]]:
        record = self.get_job(job_id)
        return record.to_dict() if record else None

    def list_user_jobs(self, user_id: str) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._jobs.values() if r.user_id == user_id]

    async def cancel_job(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def _prune_expired_locked(self) -> None:
        """Prune old jobs to prevent memory growth (internal locked method)."""
        now = time.time()
        if len(self._jobs) > self._max_records:
            sorted_keys = sorted(
                self._jobs.keys(),
                key=lambda k: self._jobs[k].created_at,
            )
            # Remove oldest 20%
            to_remove = sorted_keys[: int(len(sorted_keys) * 0.2)]
            for k in to_remove:
                self._jobs.pop(k, None)
                self._tasks.pop(k, None)

        # Remove records older than TTL that are finished
        expired = [
            k for k, r in self._jobs.items()
            if r.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
            and (now - r.created_at) > self._job_ttl_seconds
        ]
        for k in expired:
            self._jobs.pop(k, None)

    def clear(self) -> None:
        """Clear all in-memory jobs (for testing)."""
        self._jobs.clear()
        self._tasks.clear()


# Global Singleton instance
_job_manager: Optional[AsyncJobManager] = None


def get_job_manager() -> AsyncJobManager:
    global _job_manager
    if _job_manager is None:
        _job_manager = AsyncJobManager()
    return _job_manager

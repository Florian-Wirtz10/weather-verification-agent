"""Small in-process job runner with observable stage events."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import time
import uuid


class JobQueueFull(RuntimeError):
    """The single-worker service has reached its admission limit."""


@dataclass
class Job:
    id: str
    request: dict
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str = "任务已进入队列"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    result: dict | None = None
    error: str | None = None
    events: list[dict] = field(default_factory=list)
    created_monotonic: float = field(default_factory=time.perf_counter, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def update(self, stage: str, message: str, progress: float) -> None:
        with self.lock:
            self.stage = stage
            self.message = message
            self.progress = max(0.0, min(1.0, float(progress)))
            self.events.append(
                {
                    "index": len(self.events),
                    "stage": stage,
                    "message": message,
                    "progress": self.progress,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "elapsed_seconds": round(time.perf_counter() - self.created_monotonic, 3),
                }
            )

    def public(self) -> dict:
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "stage": self.stage,
                "progress": self.progress,
                "message": self.message,
                "created_at": self.created_at,
                "request": self.request,
                "error": self.error,
                "result": self.result if self.status == "complete" else None,
            }


class JobManager:
    def __init__(self, *, max_pending=8, max_history=100):
        if max_pending < 1 or max_history < max_pending:
            raise ValueError("require 1 <= max_pending <= max_history")
        self._max_pending = max_pending
        self._max_history = max_history
        # Matplotlib and the large source readers are deliberately serialized.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verification")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, request: dict, runner) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], request=request)
        job.update("queued", "任务已进入队列", 0.0)
        with self._lock:
            pending = sum(job.status in {"queued", "running"} for job in self._jobs.values())
            if pending >= self._max_pending:
                raise JobQueueFull("任务队列已满，请等待现有任务完成后重试")
            for old_id, old_job in list(self._jobs.items()):
                if len(self._jobs) < self._max_history:
                    break
                if old_job.status not in {"queued", "running"}:
                    del self._jobs[old_id]
            self._jobs[job.id] = job
            try:
                self._executor.submit(self._run, job, runner)
            except Exception:
                del self._jobs[job.id]
                raise
        return job

    def _run(self, job: Job, runner) -> None:
        with job.lock:
            job.status = "running"
        try:
            result = runner(job.request, job.id, job.update)
            with job.lock:
                job.result = result
                job.update("complete", "检验、制图与解读均已完成", 1.0)
                job.status = "complete"
        except Exception as exc:  # surfaced verbatim to the local expert user
            with job.lock:
                job.error = str(exc)
                job.update("failed", str(exc), job.progress)
                job.status = "failed"

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def recent(self) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [job.public() for job in reversed(jobs[-20:])]

    def active_ids(self) -> set[str]:
        with self._lock:
            return {
                job_id
                for job_id, job in self._jobs.items()
                if job.status in {"queued", "running"}
            }

    def forget(self, job_ids) -> None:
        """Discard completed in-memory records whose artifacts were removed."""
        with self._lock:
            for job_id in job_ids:
                job = self._jobs.get(job_id)
                if job is not None and job.status not in {"queued", "running"}:
                    self._jobs.pop(job_id, None)


jobs = JobManager()

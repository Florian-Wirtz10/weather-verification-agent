"""Small in-process job runner with observable stage events."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import time
import uuid


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
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

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
    def __init__(self):
        # Matplotlib and the large source readers are deliberately serialized.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verification")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, request: dict, runner) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], request=request)
        job.update("queued", "任务已进入队列", 0.0)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job, runner)
        return job

    def _run(self, job: Job, runner) -> None:
        job.status = "running"
        try:
            job.result = runner(job.request, job.id, job.update)
            job.status = "complete"
            job.update("complete", "检验、制图与解读均已完成", 1.0)
        except Exception as exc:  # surfaced verbatim to the local expert user
            job.error = str(exc)
            job.status = "failed"
            job.update("failed", str(exc), job.progress)

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

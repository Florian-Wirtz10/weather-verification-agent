"""Retention policy for generated web-task artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Callable, Iterable


LOGGER = logging.getLogger(__name__)
_JOB_ID = re.compile(r"[0-9a-f]{12}\Z")


@dataclass(frozen=True)
class CleanupReport:
    removed_job_ids: tuple[str, ...]
    bytes_freed: int
    remaining_bytes: int


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _job_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        item
        for item in root.iterdir()
        if item.is_dir() and _JOB_ID.fullmatch(item.name) is not None
    ]


def cleanup_job_outputs(
    root: Path,
    *,
    retention_days: float,
    max_storage_gb: float,
    active_job_ids: Iterable[str] = (),
    now: float | None = None,
) -> CleanupReport:
    """Remove expired artifacts, then enforce a total storage ceiling.

    Only directories named like application job IDs are eligible.  Active jobs
    and unknown files are never removed, even when the storage ceiling is
    exceeded.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    active = set(active_job_ids)
    timestamp = time.time() if now is None else float(now)
    cutoff = timestamp - retention_days * 86_400 if retention_days > 0 else None
    max_bytes = int(max_storage_gb * 1024**3) if max_storage_gb > 0 else None
    removed: list[str] = []
    bytes_freed = 0

    def remove(path: Path, size: int) -> None:
        nonlocal bytes_freed
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        removed.append(path.name)
        bytes_freed += size

    entries: list[tuple[Path, float, int]] = []
    for path in _job_directories(root):
        try:
            entries.append((path, path.stat().st_mtime, _directory_size(path)))
        except FileNotFoundError:
            continue

    if cutoff is not None:
        for path, modified, size in entries:
            if path.name not in active and modified < cutoff:
                remove(path, size)

    remaining = [entry for entry in entries if entry[0].is_dir()]
    total_bytes = sum(size for _, _, size in remaining)
    if max_bytes is not None and total_bytes > max_bytes:
        for path, _, size in sorted(remaining, key=lambda entry: entry[1]):
            if total_bytes <= max_bytes:
                break
            if path.name in active:
                continue
            remove(path, size)
            total_bytes -= size

    return CleanupReport(tuple(removed), bytes_freed, max(0, total_bytes))


class JobOutputJanitor:
    """Run artifact cleanup once at startup and periodically afterwards."""

    def __init__(
        self,
        root: Path,
        *,
        retention_days: float,
        max_storage_gb: float,
        interval_minutes: float,
        active_job_ids: Callable[[], set[str]],
        forget_jobs: Callable[[Iterable[str]], None],
    ):
        self.root = Path(root)
        self.retention_days = retention_days
        self.max_storage_gb = max_storage_gb
        self.interval_seconds = interval_minutes * 60
        self.active_job_ids = active_job_ids
        self.forget_jobs = forget_jobs
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def cleanup_once(self) -> CleanupReport:
        report = cleanup_job_outputs(
            self.root,
            retention_days=self.retention_days,
            max_storage_gb=self.max_storage_gb,
            active_job_ids=self.active_job_ids(),
        )
        if report.removed_job_ids:
            self.forget_jobs(report.removed_job_ids)
            LOGGER.info(
                "removed %d expired web jobs (%.1f MiB)",
                len(report.removed_job_ids),
                report.bytes_freed / 1024**2,
            )
        return report

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.cleanup_once()
            except Exception:
                LOGGER.exception("scheduled web-job cleanup failed")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        try:
            self.cleanup_once()
        except Exception:
            LOGGER.exception("startup web-job cleanup failed")
        if self.interval_seconds <= 0:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="web-job-cleanup",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

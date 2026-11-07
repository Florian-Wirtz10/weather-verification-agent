"""Deployment settings loaded before any data catalogs are constructed."""

from __future__ import annotations

import os
import math
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Always resolve the project-local file explicitly.  Relying on ``find_dotenv``
# makes behavior depend on the entry point and, more importantly, loading it in
# ``webapi.main`` used to happen after ``webapi.config`` had already frozen the
# data paths at import time.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def project_path(value: str | os.PathLike[str]) -> Path:
    """Resolve an environment path consistently from the project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def environment_path(name: str, default: str | os.PathLike[str]) -> Path:
    return project_path(os.getenv(name, os.fspath(default)))


DATA_ROOT = environment_path("WEATHER_DATA_ROOT", "external/datasets")
WEB_JOB_OUTPUT_ROOT = environment_path("WEB_JOB_OUTPUT_DIR", "outputs/web_jobs")


def _nonnegative_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


WEB_JOB_RETENTION_DAYS = _nonnegative_float("WEB_JOB_RETENTION_DAYS", 7.0)
WEB_JOB_MAX_STORAGE_GB = _nonnegative_float("WEB_JOB_MAX_STORAGE_GB", 20.0)
WEB_JOB_CLEANUP_INTERVAL_MINUTES = _nonnegative_float(
    "WEB_JOB_CLEANUP_INTERVAL_MINUTES", 60.0
)

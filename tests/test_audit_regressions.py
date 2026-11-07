"""Regression coverage for security, resource and numerical audit findings."""

from threading import Event

import numpy as np
import pytest
from fastapi.testclient import TestClient

from verification.metrics import mae, mean_error, rmse
from verification.benchmark import _frame_row
from webapi import execution, settings
from webapi.jobs import JobManager, JobQueueFull
from webapi.main import app, jobs
from webapi.maintenance import cleanup_job_outputs
from webapi.schemas import CommonTimeRequest, TaskRequest


def test_integer_error_metrics_do_not_wrap():
    forecast = np.array([0, 255], dtype=np.uint8)
    truth = np.array([255, 0], dtype=np.uint8)
    assert mae(forecast, truth) == 255
    assert rmse(forecast, truth) == 255
    assert mean_error(forecast[:1], truth[:1]) == -255
    assert rmse(np.array([30000], dtype=np.int16), np.array([-30000], dtype=np.int16)) == 60000


def test_benchmark_rejects_broadcastable_mismatched_frames():
    with pytest.raises(ValueError, match="matching shapes"):
        _frame_row(np.zeros((2, 1)), np.zeros((2, 2)), 20, (1,))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1"])
def test_retention_rejects_nonfinite_configuration(monkeypatch, value):
    monkeypatch.setenv("AUDIT_RETENTION", value)
    with pytest.raises(ValueError):
        settings._nonnegative_float("AUDIT_RETENTION", 7)


@pytest.mark.parametrize("model", [CommonTimeRequest, TaskRequest])
def test_duplicate_initializations_rejected(model):
    with pytest.raises(ValueError, match="unique"):
        model(case_id="2025-04-11", models=["StormDiT"], scope="custom",
              initialization_ids=["same", "same"])


def test_oversized_prompt_rejected():
    with pytest.raises(ValueError):
        TaskRequest(case_id="2025-04-11", models=["StormDiT"], scope="all", prompt="x" * 10001)


def test_queue_admission_and_bounded_history():
    manager = JobManager(max_pending=1, max_history=2)
    release = Event()
    try:
        first = manager.submit({}, lambda *_: release.wait(5))
        with pytest.raises(JobQueueFull):
            manager.submit({}, lambda *_: None)
        assert len(manager._jobs) == 1
        release.set()
        manager._executor.submit(lambda: None).result(timeout=5)
        for _ in range(3):
            manager.submit({}, lambda *_: {"ok": True})
            manager._executor.submit(lambda: None).result(timeout=5)
        assert len(manager._jobs) == 2
        with pytest.raises(KeyError):
            manager.get(first.id)
        assert all(item["status"] == "complete" and item["progress"] == 1 for item in manager.recent())
    finally:
        release.set()
        manager._executor.shutdown(wait=True)


def test_full_queue_returns_retryable_http_error(monkeypatch):
    def reject(*_args):
        raise JobQueueFull("full")
    monkeypatch.setattr(jobs, "submit", reject)
    response = TestClient(app).post("/api/jobs", json={
        "case_id": "2025-04-11", "models": ["StormDiT"], "scope": "all",
    })
    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"


def test_artifact_job_symlink_cannot_escape_root(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    root.mkdir()
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (root / ("a" * 12)).symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(execution, "OUTPUT_ROOT", root)
    with pytest.raises(FileNotFoundError):
        execution.artifact_path("a" * 12, "secret.txt")
    report = cleanup_job_outputs(root, retention_days=0, max_storage_gb=1e-12)
    assert not report.removed_job_ids
    assert (outside / "secret.txt").read_text() == "secret"


def test_common_time_rejects_partially_missing_selection(monkeypatch):
    monkeypatch.setattr(execution, "regular_catalog", lambda *_: object())
    monkeypatch.setattr(execution, "discover_multimodel_initializations", lambda *_a, **_k: [
        {"case_id": "present"},
    ])
    with pytest.raises(ValueError, match="刷新"):
        execution.list_common_valid_times("2025-04-11", ["StormDiT"], ["present", "missing"])


def test_failed_job_closes_figures(monkeypatch):
    import matplotlib.pyplot as plt

    def fail(*_args):
        plt.figure()
        raise ValueError("render failed")

    monkeypatch.setattr(execution, "_regular_single", fail)
    with pytest.raises(ValueError, match="render failed"):
        execution.execute_task({"models": ["StormDiT"], "scope": "single"}, "a" * 12, None)
    assert not plt.get_fignums()

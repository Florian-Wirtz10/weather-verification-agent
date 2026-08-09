from __future__ import annotations

import time

from fastapi.testclient import TestClient
import pytest

from webapi import execution
from webapi.jobs import JobManager
from webapi.main import app
from webapi.schemas import TaskRequest


client = TestClient(app)


def test_health_and_public_options_do_not_touch_large_data_catalogs():
    assert client.get("/api/health").json()["status"] == "ready"

    cases = client.get("/api/catalog/options").json()["cases"]
    case_2025 = next(case for case in cases if case["id"] == "2025-04-11")
    assert case_2025["model_groups"]["case"] == ["StormDiT", "风雷"]
    assert "ConvLSTM" in case_2025["model_groups"]["benchmark"]


def test_default_prompt_changes_with_scope_and_model_family():
    regular = client.post(
        "/api/tasks/default-prompt",
        json={
            "case_id": "2025-04-11",
            "models": ["StormDiT", "风雷"],
            "scope": "single",
            "complete_only": True,
        },
    )
    assert regular.status_code == 200
    assert "空间演变" in regular.json()["prompt"]

    benchmark = client.post(
        "/api/tasks/default-prompt",
        json={
            "case_id": "2025-04-11",
            "models": ["ConvLSTM", "PredRNN"],
            "scope": "all",
            "complete_only": True,
        },
    )
    assert benchmark.status_code == 200
    assert "FSS" in benchmark.json()["prompt"]


def test_mixed_data_contract_models_are_rejected():
    response = client.post(
        "/api/tasks/default-prompt",
        json={
            "case_id": "2025-04-11",
            "models": ["风雷", "ConvLSTM"],
            "scope": "all",
            "complete_only": True,
        },
    )
    assert response.status_code == 422
    assert "不能混合选择" in response.text


@pytest.mark.parametrize(
    ("scope", "ids"),
    [("single", []), ("single", ["a", "b"]), ("custom", ["a"]), ("custom", ["a"] * 9)],
)
def test_task_scope_requires_the_right_number_of_initializations(scope, ids):
    with pytest.raises(ValueError):
        TaskRequest(
            case_id="2025-04-11",
            models=["StormDiT"],
            scope=scope,
            initialization_ids=ids,
        )


def test_job_manager_exposes_progress_and_result():
    manager = JobManager()

    def runner(payload, job_id, update):
        update("compute", f"计算 {payload['value']}", 0.5)
        return {"job_id": job_id, "answer": payload["value"] * 2}

    job = manager.submit({"value": 21}, runner)
    deadline = time.monotonic() + 2
    while job.status not in {"complete", "failed"} and time.monotonic() < deadline:
        time.sleep(0.01)

    public = job.public()
    assert public["status"] == "complete"
    assert public["progress"] == 1.0
    assert public["result"]["answer"] == 42
    assert any(event["stage"] == "compute" for event in job.events)


def test_artifact_path_rejects_unknown_job_ids_and_nested_names(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "OUTPUT_ROOT", tmp_path)
    directory = tmp_path / "abcdef123456"
    directory.mkdir()
    artifact = directory / "metrics.csv"
    artifact.write_text("csi\n0.5\n", encoding="utf-8")

    assert execution.artifact_path("abcdef123456", "metrics.csv") == artifact
    with pytest.raises(FileNotFoundError):
        execution.artifact_path("../../outside", "metrics.csv")
    with pytest.raises(FileNotFoundError):
        execution.artifact_path("abcdef123456", "../secret.txt")

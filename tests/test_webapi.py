from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
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

    metadata = client.get("/api/catalog/metadata").json()
    assert metadata["llm"]["model"] == "deepseek-chat"
    assert {item["name"] for item in metadata["products"]} >= {"StormDiT", "风雷", "CREF"}
    assert any("严格相同" in rule for rule in metadata["rules"])
    stormdit = next(item for item in metadata["products"] if item["name"] == "StormDiT")
    assert "起报频率为1小时" in stormdit["cadence"]
    assert "预报输出间隔为6分钟" in stormdit["cadence"]
    assert any("只有风雷是每6分钟起报" in rule for rule in metadata["rules"])


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
    assert "10 dBZ" not in regular.json()["prompt"]
    assert "使用20、30、40 dBZ" in regular.json()["prompt"]

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
    assert "在20 dBZ阈值下" in benchmark.json()["prompt"]
    assert "空间分布动画" not in benchmark.json()["prompt"]

    benchmark_single = client.post(
        "/api/tasks/default-prompt",
        json={
            "case_id": "2025-04-11",
            "models": ["StormDiT", "风雷", "Earthformer", "NowCastNet"],
            "scope": "single",
            "complete_only": True,
        },
    )
    assert benchmark_single.status_code == 200
    assert "生成空间分布动画，时效曲线、热力图" in benchmark_single.json()["prompt"]


def test_business_models_can_join_zero_shot_benchmark_selection():
    response = client.post(
        "/api/tasks/default-prompt",
        json={
            "case_id": "2025-04-11",
            "models": ["StormDiT", "风雷", "ConvLSTM"],
            "scope": "all",
            "complete_only": True,
        },
    )
    assert response.status_code == 200
    assert "StormDiT、风雷、ConvLSTM" in response.json()["prompt"]
    assert "+6—+72分钟" in response.json()["prompt"]
    assert "FSS" in response.json()["prompt"]


def test_benchmark_catalog_keeps_original_concise_time_label(monkeypatch):
    sample = SimpleNamespace(
        sample_id=119,
        start_time_utc=datetime(2025, 4, 11, 0, 0),
        analysis_time_utc=datetime(2025, 4, 11, 1, 36),
        analysis_time_bjt=datetime(2025, 4, 11, 9, 36),
    )
    monkeypatch.setattr(execution, "is_benchmark", lambda _models: True)
    monkeypatch.setattr(execution, "benchmark_catalog", lambda: object())
    monkeypatch.setattr(execution, "benchmark_model_ids", lambda _models: ("ConvLSTM",))
    monkeypatch.setattr(execution, "common_horizon_minutes", lambda _models: 72)
    monkeypatch.setattr(
        execution,
        "benchmark_available_samples",
        lambda _catalog, _models, lead_range: (sample,),
    )

    items = execution.list_initializations("2025-04-11", ["ConvLSTM"], True)

    assert items[0]["label"] == "2025-04-11 09:36 BJT（04-11 01:36 UTC）"
    assert items[0]["available_leads"] == list(range(6, 73, 6))


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
    assert public["request"] == {"value": 21}
    assert any(event["stage"] == "compute" for event in job.events)
    assert all(event["elapsed_seconds"] >= 0 for event in job.events)


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

"""Regression checks for undefined rankings and independently reusable exports."""

from datetime import datetime, timedelta
from io import BytesIO
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from agent.benchmark_insight import (
    build_benchmark_insight_facts,
    render_benchmark_deterministic_summary,
)
from agent.benchmark_request import default_benchmark_request
from data_io.benchmark import BenchmarkCatalog, BenchmarkSample
from webapi import execution


@pytest.mark.parametrize("values,leader", [([np.nan, np.nan], None), ([np.nan, 0.0], "PredRNN")])
def test_undefined_metrics_are_not_ranked(values, leader):
    summary = pd.DataFrame({"model": ["StormDiT", "PredRNN"], "lead_minutes": [6, 6],
                            "threshold_dbz": [20.0, 20.0], "far": values})
    result = {"summary": summary, "context": {"models": ["StormDiT", "PredRNN"],
              "lead_range_minutes": [6, 6]}, "coverage": {"sample_count": 1}}
    request = {"diagnostic_threshold_dbz": 20.0, "metrics": ["far"], "fss_windows": [1]}
    facts = build_benchmark_insight_facts(result, request, summary, [])
    assert facts["metrics"][0]["leader"] == leader
    text = render_benchmark_deterministic_summary(facts)
    if leader is None:
        assert "最优" not in text
        assert "未定义" in text
    else:
        assert "PredRNN最优（0.000）" in text


@pytest.mark.parametrize("threshold", [20.0, 75.0])
def test_benchmark_zip_preserves_statistics_and_resolved_context(tmp_path, monkeypatch, threshold):
    analysis = datetime(2025, 4, 11, 1, 36)
    samples = []
    for index in range(2):
        path = tmp_path / f"sample_{index}.npz"
        truth = np.zeros((2, 128, 128), dtype=np.float32)
        truth[:, :2, :2] = 30
        forecast = truth.copy()
        forecast[index, :2, 1:3] = 30
        np.savez(path, radar_condition_dbz=truth[:1], radar_truth_dbz=truth,
                 pred_PredRNN_dbz=forecast, lat=32 - np.arange(128) * .03,
                 lon=112 + np.arange(128) * .03)
        init = analysis + timedelta(minutes=6 * index)
        samples.append(BenchmarkSample(index, path, init - timedelta(minutes=96),
                                       init, init + timedelta(hours=8)))
    catalog = BenchmarkCatalog(tmp_path, tuple(samples), ("PredRNN",),
                               (28.19, 32), (112, 115.81), .03, {})
    request = default_benchmark_request(single=False, common_horizon=72)
    request.update(thresholds_dbz=[threshold], diagnostic_threshold_dbz=threshold,
                   lead_range_minutes={"start": 6, "end": 12}, views=["far_bar"])
    monkeypatch.setattr(execution, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(execution, "benchmark_catalog", lambda: catalog)
    monkeypatch.setattr(execution, "default_benchmark_request", lambda **kwargs: request)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = execution._benchmark({"models": ["PredRNN"], "scope": "all"},
                                  "abcdef123456", lambda *args: None)
    bundle = tmp_path / "outputs/abcdef123456/verification_result.zip"
    with zipfile.ZipFile(bundle) as archive:
        rows = pd.read_csv(BytesIO(archive.read("benchmark_metrics.csv")))
        required = {"absolute_error_sum", "squared_error_sum"} | {
            f"fss_{window}_{part}" for window in (1, 3, 5)
            for part in ("numerator", "denominator")}
        assert required <= set(rows.columns)
        summary = pd.read_csv(BytesIO(archive.read("benchmark_summary.csv")))
        period = pd.read_csv(BytesIO(archive.read("benchmark_period_summary.csv")))
        manifest = json.loads(archive.read("benchmark_manifest.json"))
        assert manifest["request"] == request
        assert manifest["context"]["models"] == ["PredRNN"]
        assert manifest["coverage"]["sample_count"] == 2
        assert [sample["sample_id"] for sample in manifest["samples"]] == [0, 1]
        assert manifest["samples"][0]["analysis_time_utc"] == analysis.isoformat()
        assert len(summary) == 2
        assert len(period) == 1
        assert period.iloc[0].mae == pytest.approx(rows.absolute_error_sum.sum() / rows.valid_count.sum())
        assert period.iloc[0].rmse == pytest.approx(np.sqrt(rows.squared_error_sum.sum() / rows.valid_count.sum()))
        for window in (1, 3, 5):
            denominator = rows[f"fss_{window}_denominator"].sum()
            if denominator:
                expected = 1 - rows[f"fss_{window}_numerator"].sum() / denominator
                assert period.iloc[0][f"fss_w{window}"] == pytest.approx(expected)
            else:
                assert pd.isna(period.iloc[0][f"fss_w{window}"])
    if threshold == 75:
        assert "未定义" in result["insight"]
        assert "最优" not in result["insight"]

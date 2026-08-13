import pandas as pd

from agent.benchmark_insight import (
    build_benchmark_insight_facts,
    render_benchmark_deterministic_summary,
)


def test_benchmark_insight_distinguishes_business_and_zero_shot_models():
    models = ["StormDiT", "Fenglei", "ConvLSTM", "Persistence"]
    summary = pd.DataFrame(
        [
            {
                "model": model,
                "lead_minutes": lead,
                "threshold_dbz": 20.0,
                "csi": 0.7 - model_index * 0.05 - lead / 1000,
            }
            for model_index, model in enumerate(models)
            for lead in (6, 12)
        ]
    )
    period = (
        summary.groupby(["model", "threshold_dbz"], as_index=False)["csi"]
        .mean()
    )
    result = {
        "context": {"models": models, "lead_range_minutes": [6, 12]},
        "coverage": {"sample_count": 2},
        "summary": summary,
    }
    request = {
        "diagnostic_threshold_dbz": 20.0,
        "metrics": ["csi"],
        "fss_windows": [1, 3, 5],
    }

    facts = build_benchmark_insight_facts(result, request, period, [])
    experiment = facts["experiment"]
    assert experiment["business_models"] == ["StormDiT", "Fenglei"]
    assert experiment["zero_shot_baselines"] == ["ConvLSTM"]
    assert experiment["reference_baselines"] == ["Persistence"]

    text = render_benchmark_deterministic_summary(facts)
    assert "StormDiT, Fenglei为业务模型" in text
    assert "ConvLSTM为SEVIR预训练后迁移到中国CREF的zero-shot基线" in text
    assert "Persistence为持续性参考" in text


def test_benchmark_insight_uses_only_the_requested_diagnostic_threshold():
    summary = pd.DataFrame(
        [
            {"model": model, "lead_minutes": lead, "threshold_dbz": threshold, "csi": value}
            for threshold, values in ((20.0, (0.6, 0.4)), (35.0, (0.2, 0.8)))
            for model, value in zip(("StormDiT", "Fenglei"), values, strict=True)
            for lead in (6, 12)
        ]
    )
    period = summary.groupby(["model", "threshold_dbz"], as_index=False)["csi"].mean()
    result = {
        "context": {"models": ["StormDiT", "Fenglei"], "lead_range_minutes": [6, 12]},
        "coverage": {"sample_count": 1},
        "summary": summary,
    }
    request = {
        "diagnostic_threshold_dbz": 20.0,
        "metrics": ["csi"],
        "fss_windows": [1, 3, 5],
    }

    facts = build_benchmark_insight_facts(result, request, period, [])
    text = render_benchmark_deterministic_summary(facts)

    assert facts["experiment"]["diagnostic_threshold_dbz"] == 20.0
    assert facts["metrics"][0]["pooled_values"] == {"Fenglei": 0.4, "StormDiT": 0.6}
    assert "在20 dBZ下" in text
    assert "在35 dBZ下" not in text


def test_benchmark_insight_keeps_every_requested_fss_scale():
    models = ["StormDiT", "Fenglei"]
    summary = pd.DataFrame(
        [
            {
                "model": model,
                "lead_minutes": lead,
                "threshold_dbz": 20.0,
                "fss_w1": 0.5,
                "fss_w3": 0.6,
                "fss_w5": 0.7,
            }
            for model in models
            for lead in (6, 12)
        ]
    )
    period = summary.groupby(["model", "threshold_dbz"], as_index=False).mean()
    result = {
        "context": {"models": models, "lead_range_minutes": [6, 12]},
        "coverage": {"sample_count": 1},
        "summary": summary,
    }
    request = {
        "diagnostic_threshold_dbz": 20.0,
        "metrics": ["fss"],
        "fss_windows": [1, 3, 5],
    }

    facts = build_benchmark_insight_facts(result, request, period, [])

    assert [item["metric"] for item in facts["metrics"]] == [
        "FSS 1×1像元",
        "FSS 3×3像元",
        "FSS 5×5像元",
    ]

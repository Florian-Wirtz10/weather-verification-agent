from datetime import datetime

import pandas as pd

from agent.comparison_insight import (
    build_comparison_insight_facts,
    render_comparison_deterministic_summary,
)


def _result():
    return {
        "context": {
            "model_ids": ["stormdit", "fenglei"],
            "init_time_utc": datetime(2025, 4, 11, 0),
            "init_time_bjt": datetime(2025, 4, 11, 8),
            "model_contract": "phase8-inferred-v1",
        },
        "per_lead": [
            {
                "lead_minutes": lead,
                "models": {
                    "stormdit": {15.0: {"csi": stormdit}},
                    "fenglei": {15.0: {"csi": fenglei}},
                },
            }
            for lead, stormdit, fenglei in ((6, 0.8, 0.7), (12, 0.6, 0.65))
        ],
    }


def test_comparison_facts_precompute_means_and_wins():
    facts = build_comparison_insight_facts(
        _result(), ["csi"], figure_catalog=[{"number": 1, "label": "CSI曲线"}]
    )

    threshold = facts["thresholds"][0]
    assert threshold["products"]["stormdit"]["csi"]["mean"] == 0.7
    assert threshold["wins_by_metric"]["csi"] == {
        "stormdit": 1,
        "fenglei": 1,
        "ties": 0,
    }
    assert facts["context"]["products"][0]["label"] == "StormDiT"
    assert "路径" not in str(facts)


def test_comparison_fallback_has_required_sections_and_figure_reference():
    text = render_comparison_deterministic_summary(
        build_comparison_insight_facts(
            _result(),
            ["csi"],
            figure_catalog=[{"number": 1, "label": "CSI曲线"}],
        )
    )

    assert "### 图表导读" in text
    assert "图1" in text
    assert "StormDiT" in text
    assert "### 结论摘要" in text
    assert "平均表现以" in text
    assert "### 数据证据与诊断分析" in text
    assert "均值" in text
    assert "首值" in text
    assert "末值" in text
    assert "口径与边界" not in text


def test_far_wins_use_lower_is_better_direction():
    result = _result()
    for entry, stormdit, fenglei in zip(
        result["per_lead"],
        (0.10, 0.30),
        (0.20, 0.25),
    ):
        entry["models"]["stormdit"][15.0]["far"] = stormdit
        entry["models"]["fenglei"][15.0]["far"] = fenglei

    facts = build_comparison_insight_facts(result, ["far"])

    assert facts["thresholds"][0]["wins_by_metric"]["far"] == {
        "stormdit": 1,
        "fenglei": 1,
        "ties": 0,
    }


def test_batch_facts_include_descriptive_physical_correlations():
    rows = []
    for model_id in ("stormdit", "fenglei"):
        for index in range(5):
            rows.append(
                {
                    "model_id": model_id,
                    "init_time_utc": f"2025-04-11T0{index}:00:00",
                    "lead_minutes": 6 * (index + 1),
                    "threshold_dbz": 20.0,
                    "valid_count": 100,
                    "hits": 50 - index,
                    "misses": 10 + index,
                    "false_alarms": 5 + index,
                    "csi": 0.8 - index * 0.1,
                    "forecast_mean_dbz": 20 - index,
                    "observation_mean_dbz": 19 - index / 2,
                    "reflectivity_mae_dbz": 2 + index,
                    "centroid_distance_km": 3 + index,
                }
            )
    frame = pd.DataFrame(rows)
    result = {
        "context": {
            "model_ids": ["stormdit", "fenglei"],
            "model_contract": "phase8-inferred-v1",
        },
        "coverage": {
            "selected_initialization_count": 5,
            "selected_pairs": 5,
        },
        "per_case": frame,
    }

    facts = build_comparison_insight_facts(
        result,
        ["csi"],
        diagnostic_threshold_dbz=20,
        correlation_method="spearman",
    )

    correlations = facts["physical_correlations"]
    assert correlations["method"] == "spearman"
    assert correlations["products"]["stormdit"]["sample_count"] == 5
    assert correlations["products"]["stormdit"]["strongest_score_relationships"]


def test_same_valid_time_facts_preserve_init_and_native_lead():
    valid_time = datetime(2025, 4, 11, 3)
    frame = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "init_time_utc": init_time.isoformat(),
                "lead_minutes": lead,
                "threshold_dbz": 20.0,
                "csi": score,
            }
            for init_time, lead, score in (
                (datetime(2025, 4, 11, 1), 120, 0.4),
                (datetime(2025, 4, 11, 2), 60, 0.6),
            )
            for model_id in ("stormdit", "fenglei")
        ]
    )
    result = {
        "context": {
            "model_ids": ["stormdit", "fenglei"],
            "model_contract": "phase8-inferred-v1",
        },
        "coverage": {
            "selected_initialization_count": 2,
            "selected_pairs": 2,
        },
        "per_case": frame,
        "same_valid_time_comparison": {
            "valid_time_utc": valid_time,
            "valid_time_bjt": valid_time + pd.Timedelta(hours=8),
            "items": [
                {
                    "init_time_utc": init_time,
                    "init_time_bjt": init_time + pd.Timedelta(hours=8),
                    "lead_minutes": lead,
                    "scores": {
                        model_id: {
                            20.0: {
                                "csi": score,
                                "reflectivity_mae_dbz": 3.0,
                            }
                        }
                        for model_id in ("stormdit", "fenglei")
                    },
                }
                for init_time, lead, score in (
                    (datetime(2025, 4, 11, 1), 120, 0.4),
                    (datetime(2025, 4, 11, 2), 60, 0.6),
                )
            ],
        },
    }

    facts = build_comparison_insight_facts(
        result,
        ["csi"],
        diagnostic_threshold_dbz=20,
    )

    comparison = facts["same_valid_time_comparison"]
    assert comparison["valid_time_bjt"] == "2025-04-11T11:00"
    assert [
        item["lead_minutes"] for item in comparison["initializations"]
    ] == [120, 60]

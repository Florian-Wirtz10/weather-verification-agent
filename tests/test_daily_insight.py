from types import SimpleNamespace

import pandas as pd

from agent.daily_insight import (
    build_daily_insight_facts,
    generate_daily_insight,
    render_daily_deterministic_summary,
)


def _result():
    per_case = pd.DataFrame(
        [
            {
                "init_time_bjt": f"2026-04-07T00:0{init}:00",
                "lead_minutes": lead,
                "threshold_dbz": threshold,
                "csi": value,
                "hits": 10,
                "misses": 2,
                "false_alarms": 3,
                "valid_count": 20,
            }
            for init in (0, 6)
            for lead, value in ((6, 0.6), (12, 0.4))
            for threshold in (5.0, 15.0)
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "lead_minutes": lead,
                "threshold_dbz": threshold,
                "case_count": 2,
                "csi": value,
            }
            for lead, value in ((6, 0.6), (12, 0.4))
            for threshold in (5.0, 15.0)
        ]
    )
    return {
        "context": {
            "case_date": "2026-04-07",
            "forecast_case_time_basis": "bjt",
            "observation_case_time_basis": "utc",
            "model_id": "fenglei",
            "bbox": [28, 35, 116.5, 123.5],
            "thresholds_dbz": [5, 15],
            "lead_range_minutes": [6, 12],
            "temporal_matching": "exact",
            "spatial_alignment": "nearest",
            "aggregation": "pooled counts",
        },
        "coverage": {"selected_pairs": 4, "completed_pairs": 4},
        "per_case": per_case,
        "summary": summary,
    }


def test_daily_insight_facts_are_path_free_and_number_figures():
    facts = build_daily_insight_facts(_result(), metric="csi", diagnostic_threshold_dbz=15)

    assert facts["requested_metric"] == "csi"
    assert [item["number"] for item in facts["figure_catalog"]] == [1, 2, 3, 4]
    assert facts["threshold_summary"][0]["first"]["value"] == 0.6
    assert facts["rate_diagnostics"]["sample_count"] == 4
    assert "path" not in str(facts).lower()
    assert "不预设" in render_daily_deterministic_summary(facts)


def test_generate_daily_insight_sends_only_fact_json():
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="## 图表导读\n如图1所示。"))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    facts = build_daily_insight_facts(_result(), metric="csi", diagnostic_threshold_dbz=15)

    text = generate_daily_insight(facts, client=client)

    assert text.startswith("## 图表导读")
    assert captured["temperature"] == 0
    assert "forecast_path" not in captured["messages"][1]["content"]


def test_daily_insight_catalog_contains_only_requested_figures():
    facts = build_daily_insight_facts(
        _result(),
        metric="csi",
        diagnostic_threshold_dbz=15,
        views=["daily_score_curve", "score_distribution"],
    )

    assert [item["type"] for item in facts["figure_catalog"]] == [
        "daily_score_curve",
        "score_distribution",
    ]
    assert [item["number"] for item in facts["figure_catalog"]] == [1, 2]

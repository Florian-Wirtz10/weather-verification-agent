from datetime import datetime

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
    assert "### 主要发现" in text
    assert "### 数据分析" in text
    assert "口径与边界" not in text

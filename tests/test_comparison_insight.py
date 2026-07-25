from datetime import datetime

from agent.comparison_insight import (
    build_comparison_insight_facts,
    render_comparison_deterministic_summary,
)


def _result():
    return {
        "context": {
            "init_time_utc": datetime(2025, 4, 11, 0),
            "init_time_bjt": datetime(2025, 4, 11, 8),
            "model_contract": "phase8-inferred-v1",
        },
        "per_lead": [
            {
                "lead_minutes": lead,
                "model": {15.0: {"csi": model}},
                "fenglei": {15.0: {"csi": fenglei}},
            }
            for lead, model, fenglei in ((6, 0.8, 0.7), (12, 0.6, 0.65))
        ],
    }


def test_comparison_facts_precompute_means_and_wins():
    facts = build_comparison_insight_facts(_result(), ["csi"])

    threshold = facts["thresholds"][0]
    assert threshold["products"]["model"]["csi"]["mean"] == 0.7
    assert threshold["wins_by_metric"]["csi"] == {
        "model": 1,
        "fenglei": 1,
        "ties": 0,
    }
    assert "路径" not in str(facts)


def test_comparison_fallback_has_required_sections():
    text = render_comparison_deterministic_summary(
        build_comparison_insight_facts(_result(), ["csi"])
    )

    assert "### 图表导读" in text
    assert "### 主要发现" in text
    assert "### 指标分析" in text
    assert "口径与边界" not in text

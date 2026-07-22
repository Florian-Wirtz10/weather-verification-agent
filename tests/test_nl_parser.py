from agent.nl_parser import (
    _apply_rate_threshold_mode_guard,
    _apply_spatial_animation_guard,
    _build_system_prompt,
    _validate,
)


def test_prompt_exposes_scope_schema_and_available_case():
    prompt = _build_system_prompt(
        [
            {
                "case_id": "case-1",
                "date": "2025-04-11",
                "init_time": "2025-04-11 02:12:00",
                "available_leads": [6, 12, 18],
            }
        ]
    )

    assert "不是实时天气问答助手" in prompt
    assert "2025-04-11" in prompt
    assert "unmatched_requests" in prompt
    assert "不得为了凑结果输出全部图" in prompt
    assert "空间回波的“演变、过程、动态、逐时变化”" in prompt
    assert 'layout="auto"' in prompt
    assert "偷换成评分曲线" in prompt
    assert 'threshold_mode="separate"' in prompt


def test_validate_uses_schema_and_does_not_default_unsupported_request():
    result = _validate(
        {
            "intent": "verify_case",
            "views": [{"type": "taylor_diagram"}],
        }
    )

    assert result["views"] == []
    assert any("taylor" in item for item in result["unmatched_requests"])


def test_spatial_comparison_evolution_becomes_animation_not_score_curve():
    parsed = _validate(
        {
            "intent": "verify_case",
            "filters": {"thresholds_dbz": [35], "lead_range_minutes": {"start": 0, "end": 180}},
            "views": [{"type": "score_time_series", "metric": "csi"}],
        }
    )

    guarded = _apply_spatial_animation_guard(
        "使用35 dBZ阈值，看所有时效的预测-观测对比图的时序变化，并简要解读结果。",
        parsed,
    )

    assert guarded["views"] == [
        {
            "type": "reflectivity_animation",
            "lead_range_minutes": {"start": 6, "end": 180},
            "fps": 2,
        }
    ]


def test_score_time_series_without_spatial_comparison_is_unchanged():
    parsed = _validate({"views": [{"type": "score_time_series", "metric": "csi"}]})

    assert _apply_spatial_animation_guard("看 CSI 随预报时效变化", parsed) == parsed


def test_explicit_rate_threshold_layout_wording_is_deterministic():
    parsed = _validate(
        {
            "filters": {"thresholds_dbz": [15, 35]},
            "views": [{"type": "forecast_observed_rate"}],
        }
    )

    combined = _apply_rate_threshold_mode_guard(
        "把15和35 dBZ合在一张图中比较预报率和观测率", parsed
    )
    assert combined["views"][0]["threshold_mode"] == "combined"

    separate = _apply_rate_threshold_mode_guard(
        "15和35 dBZ分别画预报率和观测率关系图", combined
    )
    assert separate["views"][0]["threshold_mode"] == "separate"

import math

import pytest

from agent.request_schema import (
    DEFAULT_THRESHOLDS_DBZ,
    SCHEMA_VERSION,
    RequestValidationError,
    default_request,
    validate_request,
)


def test_default_request_is_complete_and_returns_fresh_data():
    first = default_request()
    second = default_request()

    assert first == {
        "schema_version": SCHEMA_VERSION,
        "intent": "verify_case",
        "filters": {
            "thresholds_dbz": list(DEFAULT_THRESHOLDS_DBZ),
            "lead_range_minutes": {"start": 0, "end": 180},
            "requested_date": None,
        },
        "views": [{"type": "score_time_series", "metric": "csi"}],
        "presentation": {"layout": "auto", "strategy": "auto", "metric": None},
        "insight": {"enabled": True, "length": "detailed"},
        "unmatched_requests": [],
        "notes": "",
    }
    first["views"].append({"type": "score_time_series", "metric": "ets"})
    assert len(second["views"]) == 1


def test_validate_normalizes_parameterized_request_and_deduplicates_in_order():
    result = validate_request(
        {
            "schema_version": "1",
            "intent": "VERIFY_CASE",
            "filters": {
                "thresholds_dbz": [35, 20.0, 35],
                "lead_range_minutes": {"start": 6, "end": 60},
                "requested_date": "2025-04-11",
            },
            "views": [
                {"type": "score_time_series", "metric": "CSI"},
                {"type": "score_time_series", "metric": "ets"},
                {"type": "score_time_series", "metric": "csi"},
                {"type": "reflectivity_comparison", "lead": "LAST"},
                {"type": "reflectivity_error", "lead": 60},
            ],
            "insight": {"enabled": True, "length": "DETAILED"},
            "notes": "  只看前一小时  ",
        }
    )

    assert result["filters"] == {
        "thresholds_dbz": [35.0, 20.0],
        "lead_range_minutes": {"start": 6, "end": 60},
        "requested_date": "2025-04-11",
    }
    assert result["views"] == [
        {"type": "score_time_series", "metric": "csi"},
        {"type": "score_time_series", "metric": "ets"},
        {"type": "reflectivity_comparison", "lead": "last"},
        {"type": "reflectivity_error", "lead": 60},
    ]
    assert result["insight"] == {"enabled": True, "length": "detailed"}
    assert result["notes"] == "只看前一小时"


@pytest.mark.parametrize("parsed", [None, [], "request", 42])
def test_request_must_be_an_object(parsed):
    with pytest.raises(RequestValidationError):
        validate_request(parsed)


def test_missing_views_gets_one_csi_chart_not_all_charts():
    result = validate_request({})
    assert result["views"] == [{"type": "score_time_series", "metric": "csi"}]


def test_forecast_observed_rate_view_supports_safe_threshold_modes():
    result = validate_request({"views": [{"type": "forecast_observed_rate"}]})

    assert result["views"] == [
        {"type": "forecast_observed_rate", "threshold_mode": "separate"}
    ]
    assert result["unmatched_requests"] == []

    combined = validate_request(
        {
            "views": [
                {"type": "forecast_observed_rate", "threshold_mode": "combined"}
            ]
        }
    )
    assert combined["views"][0]["threshold_mode"] == "combined"

    with_extra = validate_request(
        {"views": [{"type": "forecast_observed_rate", "chart": "free-form"}]}
    )
    assert with_extra["views"] == [
        {"type": "forecast_observed_rate", "threshold_mode": "separate"}
    ]
    assert any("chart" in item for item in with_extra["unmatched_requests"])

    with pytest.raises(RequestValidationError):
        validate_request(
            {
                "views": [
                    {"type": "forecast_observed_rate", "threshold_mode": "free_form"}
                ]
            }
        )


def test_unknown_and_malicious_views_are_preserved_and_never_executed():
    result = validate_request(
        {
            "views": [
                {"type": "taylor_diagram"},
                {"type": "os.system('rm -rf /')"},
                "__import__('subprocess').run('bad')",
            ]
        }
    )

    assert result["views"] == []
    assert "taylor_diagram" in " ".join(result["unmatched_requests"])
    assert "os.system" in " ".join(result["unmatched_requests"])
    assert "subprocess" in " ".join(result["unmatched_requests"])


def test_unknown_metric_is_unmatched_while_supported_view_still_runs():
    result = validate_request(
        {
            "views": [
                {"type": "score_time_series", "metric": "fss"},
                {"type": "score_time_series", "metric": "csi"},
            ]
        }
    )

    assert result["views"] == [{"type": "score_time_series", "metric": "csi"}]
    assert any("fss" in item for item in result["unmatched_requests"])


def test_only_unsupported_metric_does_not_fall_back_to_csi():
    result = validate_request(
        {"views": [{"type": "score_time_series", "metric": "fss"}]}
    )
    assert result["views"] == []
    assert result["unmatched_requests"]


def test_unsupported_extra_view_parameter_is_reported_but_valid_view_survives():
    result = validate_request(
        {
            "views": [
                {
                    "type": "score_time_series",
                    "metric": "ets",
                    "scale": "log",
                }
            ]
        }
    )
    assert result["views"] == [{"type": "score_time_series", "metric": "ets"}]
    assert any("scale" in item for item in result["unmatched_requests"])


@pytest.mark.parametrize(
    "thresholds",
    [
        "35",
        [],
        [True],
        ["35"],
        [-1],
        [76],
        [math.nan],
        [math.inf],
        [-math.inf],
        list(range(9)),
    ],
)
def test_explicit_invalid_thresholds_fail_closed(thresholds):
    with pytest.raises(RequestValidationError):
        validate_request({"filters": {"thresholds_dbz": thresholds}})


def test_missing_thresholds_use_defaults_but_explicit_none_is_invalid():
    assert validate_request({"filters": {}})["filters"]["thresholds_dbz"] == list(
        DEFAULT_THRESHOLDS_DBZ
    )
    with pytest.raises(RequestValidationError):
        validate_request({"filters": {"thresholds_dbz": None}})


@pytest.mark.parametrize(
    "lead_range",
    [
        {"start": True, "end": 60},
        {"start": 0.0, "end": 60},
        {"start": -1, "end": 60},
        {"start": 0, "end": 181},
        {"start": 61, "end": 60},
        "0-60",
    ],
)
def test_invalid_lead_ranges_fail_closed(lead_range):
    with pytest.raises(RequestValidationError):
        validate_request({"filters": {"lead_range_minutes": lead_range}})


@pytest.mark.parametrize("lead", [True, 6.0, -1, 181, "middle"])
def test_invalid_reflectivity_view_leads_fail_closed(lead):
    with pytest.raises(RequestValidationError):
        validate_request({"views": [{"type": "reflectivity_error", "lead": lead}]})


@pytest.mark.parametrize(
    "requested_date",
    ["2025-4-11", "2025-02-29", "11/04/2025", "2025-04-11T00:00:00", 20250411],
)
def test_requested_date_must_be_a_real_iso_date(requested_date):
    with pytest.raises(RequestValidationError):
        validate_request({"filters": {"requested_date": requested_date}})


def test_non_verification_intent_does_not_gain_a_default_view():
    result = validate_request({"intent": "out_of_scope"})
    assert result["views"] == []


def test_unknown_intent_is_explicit_and_not_defaulted():
    result = validate_request({"intent": "forecast_tomorrow"})
    assert result["intent"] == "unknown"
    assert result["views"] == []
    assert result["unmatched_requests"]


def test_free_text_is_bounded_and_unmatched_is_deduplicated():
    result = validate_request({"unmatched_requests": ["Taylor 图", "Taylor 图"]})
    assert result["unmatched_requests"] == ["Taylor 图"]

    with pytest.raises(RequestValidationError):
        validate_request({"notes": "x" * 1_001})
    with pytest.raises(RequestValidationError):
        validate_request({"unmatched_requests": ["x" * 301]})


def test_spatial_view_count_is_bounded_for_memory_safety():
    views = [
        {"type": "reflectivity_error", "lead": lead}
        for lead in (6, 12, 18, 24, 30, 36, 42)
    ]

    with pytest.raises(RequestValidationError, match="空间图最多允许 6 张"):
        validate_request({"views": views})


def test_animation_and_presentation_are_declarative_and_strictly_bounded():
    result = validate_request(
        {
            "views": [
                {
                    "type": "reflectivity_animation",
                    "lead_range_minutes": {"start": 6, "end": 24},
                    "fps": 2,
                }
            ],
            "presentation": {"layout": "animation_focus"},
        }
    )
    assert result["views"] == [
        {
            "type": "reflectivity_animation",
            "lead_range_minutes": {"start": 6, "end": 24},
            "fps": 2,
        }
    ]
    assert result["presentation"] == {
        "layout": "animation_focus",
        "strategy": "auto",
        "metric": None,
    }

    linked = validate_request(
        {
            "views": [
                {"type": "score_time_series", "metric": "csi"},
                {
                    "type": "reflectivity_animation",
                    "lead_range_minutes": {"start": 6, "end": 24},
                },
            ],
            "presentation": {"strategy": "linked_score_animation", "metric": "csi"},
        }
    )
    assert linked["presentation"] == {
        "layout": "auto",
        "strategy": "linked_score_animation",
        "metric": "csi",
    }

    with pytest.raises(RequestValidationError):
        validate_request(
            {
                "views": [
                    {
                        "type": "reflectivity_animation",
                        "lead_range_minutes": {"start": 6, "end": 24},
                        "fps": 5,
                    }
                ]
            }
        )
    with pytest.raises(RequestValidationError):
        validate_request({"presentation": {"layout": "arbitrary_css"}})
    with pytest.raises(RequestValidationError):
        validate_request({"presentation": {"strategy": "linked_score_animation"}})

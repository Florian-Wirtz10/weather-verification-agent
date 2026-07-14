import pytest

from agent.daily_request import default_daily_request, validate_daily_request


def test_default_daily_request_selects_complete_diagnostic_suite():
    request = default_daily_request("2026-04-07")

    assert request["requested_date"] == "2026-04-07"
    assert request["views"] == [
        "daily_score_curve",
        "daily_score_heatmap",
        "score_distribution",
    ]


def test_daily_request_rejects_diagnostic_threshold_not_scored():
    request = default_daily_request()
    request["thresholds_dbz"] = [5, 25]

    with pytest.raises(ValueError, match="must be included"):
        validate_daily_request(request)


def test_daily_request_preserves_unsupported_view_without_fallback():
    request = default_daily_request()
    request["views"] = ["daily_score_curve", "taylor"]

    normalized = validate_daily_request(request)

    assert normalized["views"] == ["daily_score_curve"]
    assert normalized["unmatched_requests"] == ["暂不支持全天图表：taylor"]


def test_daily_request_requires_native_six_minute_leads():
    request = default_daily_request()
    request["lead_range_minutes"] = {"start": 7, "end": 180}

    with pytest.raises(ValueError, match="multiple of 6"):
        validate_daily_request(request)

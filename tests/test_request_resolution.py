from datetime import datetime

from agent.request_resolution import resolve_request
from agent.request_schema import default_request, validate_request


CASES = [
    {
        "case_id": "fenglei-refmosaic-202504110212",
        "date": "2025-04-11",
        "init_time": datetime(2025, 4, 11, 2, 12),
        "available_leads": [6, 12, 18],
    }
]


def test_ready_request_resolves_to_one_case_and_execution_plan():
    result = resolve_request(default_request(), CASES)
    assert result["status"] == "ready"
    assert result["execution"]["init_time"] == datetime(2025, 4, 11, 2, 12)
    assert result["execution"]["views"] == [
        {"type": "score_time_series", "metric": "csi"}
    ]


def test_unsupported_request_does_not_gain_default_execution():
    request = validate_request({"views": [{"type": "taylor_diagram"}]})
    result = resolve_request(request, CASES)
    assert result["status"] == "unsupported"
    assert "execution" not in result


def test_partial_request_executes_only_supported_views():
    request = validate_request(
        {
            "views": [
                {"type": "score_time_series", "metric": "fss"},
                {"type": "score_time_series", "metric": "csi"},
            ]
        }
    )
    result = resolve_request(request, CASES)
    assert result["status"] == "partial"
    assert result["execution"]["views"] == [
        {"type": "score_time_series", "metric": "csi"}
    ]


def test_unavailable_date_is_no_data_and_has_no_execution():
    request = validate_request({"filters": {"requested_date": "2025-04-10"}})
    result = resolve_request(request, CASES)
    assert result["status"] == "no_data"
    assert "2025-04-11" in result["message"]
    assert "execution" not in result


def test_live_weather_is_out_of_scope():
    request = validate_request({"intent": "out_of_scope"})
    result = resolve_request(request, CASES)
    assert result["status"] == "out_of_scope"
    assert "execution" not in result


def test_multiple_cases_require_selection():
    cases = CASES + [
        {
            "case_id": "fenglei-refmosaic-202504112212",
            "date": "2025-04-11",
            "init_time": datetime(2025, 4, 11, 22, 12),
            "available_leads": [6, 12],
        }
    ]
    result = resolve_request(default_request(), cases)
    assert result["status"] == "needs_clarification"
    assert "execution" not in result


def test_explain_last_result_is_deferred_in_single_turn_phase3():
    request = validate_request({"intent": "explain_last_result"})
    assert resolve_request(request, CASES)["status"] == "needs_clarification"

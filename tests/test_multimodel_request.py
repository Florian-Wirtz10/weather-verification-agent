import pytest

from agent.multimodel_request import (
    default_multimodel_request,
    validate_multimodel_request,
)


def test_single_default_includes_synchronized_animation():
    request = default_multimodel_request(batch=False)
    assert request["views"] == [
        "spatial_comparison",
        "reflectivity_animation",
        "score_curve",
    ]


def test_batch_rejects_spatial_view_explicitly():
    parsed = default_multimodel_request(batch=True)
    parsed["views"].append("spatial_comparison")
    request = validate_multimodel_request(parsed, batch=True)
    assert "spatial_comparison" not in request["views"]
    assert request["unmatched_requests"] == ["多起报模式暂不支持：spatial_comparison"]


def test_lead_range_stays_on_native_six_minute_cadence():
    parsed = default_multimodel_request(batch=False)
    parsed["lead_range_minutes"]["end"] = 61
    with pytest.raises(ValueError, match="multiple of 6"):
        validate_multimodel_request(parsed, batch=False)

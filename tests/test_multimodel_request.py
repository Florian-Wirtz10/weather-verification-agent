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
        "multi_metric_curve",
    ]
    assert request["metrics"] == ["csi", "pod", "far"]
    assert request["thresholds_dbz"] == [10.0, 20.0, 30.0, 40.0]
    assert request["diagnostic_threshold_dbz"] == 20.0
    assert request["diagnostic_metric"] == "csi"
    assert request["correlation_method"] == "spearman"


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


def test_product_request_rejects_ets_and_invalid_correlation_method():
    parsed = default_multimodel_request(batch=True)
    parsed["metrics"] = ["csi", "ets"]
    with pytest.raises(ValueError, match="unsupported metric: ets"):
        validate_multimodel_request(parsed, batch=True)

    parsed = default_multimodel_request(batch=True)
    parsed["correlation_method"] = "kendall"
    with pytest.raises(ValueError, match="correlation_method"):
        validate_multimodel_request(parsed, batch=True)

    parsed = default_multimodel_request(batch=True)
    parsed["diagnostic_metric"] = "bias_score"
    with pytest.raises(ValueError, match="diagnostic_metric"):
        validate_multimodel_request(parsed, batch=True)

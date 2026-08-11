import pytest

from agent.multimodel_request import (
    default_multimodel_request,
    validate_multimodel_request,
)


def test_single_default_includes_synchronized_animation():
    request = default_multimodel_request(batch=False)
    assert request["scope"] == "single"
    assert request["views"] == [
        "reflectivity_animation",
        "score_curve",
        "multi_metric_curve",
    ]
    assert request["spatial_lead_minutes"] is None
    assert request["metrics"] == ["csi", "pod", "far"]
    assert request["thresholds_dbz"] == [20.0, 30.0, 40.0]
    assert request["diagnostic_threshold_dbz"] == 20.0
    assert request["diagnostic_metric"] == "csi"
    assert request["correlation_method"] == "spearman"


def test_custom_default_compares_different_inits_at_one_valid_time():
    request = default_multimodel_request(batch=True, scope="custom")
    assert request["scope"] == "custom"
    assert request["views"] == ["same_valid_time_comparison"]
    assert request["thresholds_dbz"] == [20.0]


def test_all_default_keeps_population_views():
    request = default_multimodel_request(batch=True, scope="all")
    assert request["scope"] == "all"
    assert request["views"] == [
        "score_curve",
        "multi_metric_curve",
        "score_heatmap",
        "variable_correlation_heatmap",
        "score_distribution",
        "forecast_observed_rate_scatter",
    ]


def test_same_valid_time_view_is_custom_only():
    parsed = default_multimodel_request(batch=True, scope="all")
    parsed["views"] = ["same_valid_time_comparison"]
    request = validate_multimodel_request(parsed, batch=True, scope="all")
    assert request["views"] == []
    assert request["unmatched_requests"] == [
        "仅自定义多起报支持：same_valid_time_comparison"
    ]


def test_animation_subsumes_unrequested_static_spatial_frame():
    parsed = default_multimodel_request(batch=False)
    parsed["views"].insert(0, "spatial_comparison")
    request = validate_multimodel_request(parsed, batch=False)
    assert "spatial_comparison" not in request["views"]
    assert "reflectivity_animation" in request["views"]


def test_explicit_spatial_lead_keeps_static_frame_beside_animation():
    parsed = default_multimodel_request(batch=False)
    parsed["views"].insert(0, "spatial_comparison")
    parsed["spatial_lead_minutes"] = 60
    request = validate_multimodel_request(parsed, batch=False)
    assert request["views"][:2] == [
        "spatial_comparison",
        "reflectivity_animation",
    ]
    assert request["spatial_lead_minutes"] == 60


def test_spatial_lead_must_be_inside_selected_range():
    parsed = default_multimodel_request(batch=False)
    parsed["views"] = ["spatial_comparison"]
    parsed["lead_range_minutes"] = {"start": 60, "end": 120}
    parsed["spatial_lead_minutes"] = 30
    with pytest.raises(ValueError, match="inside lead_range_minutes"):
        validate_multimodel_request(parsed, batch=False)


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

from pathlib import Path
import re

import pytest

from agent.benchmark_request import (
    apply_explicit_benchmark_contract,
    default_benchmark_request,
    validate_benchmark_request,
)
from agent.multimodel_request import (
    apply_explicit_multimodel_contract,
    default_multimodel_request,
    validate_multimodel_request,
)


def _reference_prompts():
    source = Path("frontend/src/components/ModelPromptView.tsx").read_text(encoding="utf-8")
    return re.findall(r"prompt: '([^']+)'", source)


REGULAR_CONTRACTS = [
    {
        "scope": "single",
        "views": ["reflectivity_animation", "score_curve", "multi_metric_curve"],
        "thresholds": [20.0, 30.0, 40.0],
        "score_metrics": ["csi", "pod", "far"],
        "multi_metrics": ["csi", "pod", "far"],
        "diagnostic": 20.0,
        "spatial": None,
    },
    {
        "scope": "single",
        "views": ["spatial_comparison", "score_curve"],
        "thresholds": [20.0, 35.0],
        "score_metrics": ["csi", "pod", "far"],
        "multi_metrics": [],
        "diagnostic": 20.0,
        "spatial": 60,
    },
    {
        "scope": "single",
        "views": ["reflectivity_animation", "score_curve", "multi_metric_curve"],
        "thresholds": [20.0, 30.0, 40.0],
        "score_metrics": ["csi", "bias_score"],
        "multi_metrics": ["csi", "pod", "far"],
        "diagnostic": 20.0,
        "spatial": None,
    },
    {
        "scope": "custom",
        "views": ["same_valid_time_comparison"],
        "thresholds": [20.0],
        "score_metrics": [],
        "multi_metrics": [],
        "diagnostic": 20.0,
        "spatial": None,
    },
    {
        "scope": "custom",
        "views": ["score_curve", "score_heatmap", "score_distribution"],
        "thresholds": [20.0, 30.0, 40.0],
        "score_metrics": ["csi", "pod", "far"],
        "multi_metrics": [],
        "diagnostic": 20.0,
        "spatial": None,
    },
    {
        "scope": "custom",
        "views": [
            "score_curve",
            "variable_correlation_heatmap",
            "forecast_observed_rate_scatter",
        ],
        "thresholds": [20.0],
        "score_metrics": ["bias_score"],
        "multi_metrics": [],
        "diagnostic": 20.0,
        "spatial": None,
    },
    {
        "scope": "all",
        "views": [
            "score_curve",
            "multi_metric_curve",
            "score_heatmap",
            "variable_correlation_heatmap",
            "score_distribution",
            "forecast_observed_rate_scatter",
        ],
        "thresholds": [20.0, 30.0, 40.0],
        "score_metrics": ["csi", "pod", "far"],
        "multi_metrics": ["csi", "pod", "far"],
        "diagnostic": 20.0,
        "spatial": None,
    },
    {
        "scope": "all",
        "views": ["score_curve", "score_heatmap", "score_distribution"],
        "thresholds": [30.0, 40.0],
        "score_metrics": ["csi", "pod", "far", "bias_score"],
        "multi_metrics": [],
        "diagnostic": 40.0,
        "spatial": None,
    },
    {
        "scope": "all",
        "views": [
            "score_curve",
            "variable_correlation_heatmap",
            "forecast_observed_rate_scatter",
        ],
        "thresholds": [20.0],
        "score_metrics": ["bias_score"],
        "multi_metrics": [],
        "diagnostic": 20.0,
        "spatial": None,
    },
]


BENCHMARK_CONTRACTS = [
    {
        "single": False,
        "horizon": 120,
        "views": ["score_curve", "fss_curve", "period_summary", "score_heatmap"],
        "metrics": ["csi", "far", "mae", "rmse", "fss"],
        "thresholds": [20.0, 35.0, 45.0],
        "diagnostic": 35.0,
        "diagnostic_metric": "csi",
        "windows": [5],
        "spatial": None,
    },
    {
        "single": False,
        "horizon": 72,
        "views": ["score_curve", "period_summary", "score_heatmap"],
        "metrics": ["csi", "pod", "far"],
        "thresholds": [20.0, 35.0, 45.0],
        "diagnostic": 35.0,
        "diagnostic_metric": "csi",
        "windows": [1, 3, 5],
        "spatial": None,
    },
    {
        "single": False,
        "horizon": 72,
        "views": ["fss_curve", "period_summary", "score_heatmap"],
        "metrics": ["fss"],
        "thresholds": [20.0, 35.0],
        "diagnostic": 35.0,
        "diagnostic_metric": "fss",
        "windows": [1, 3, 5],
        "spatial": None,
    },
    {
        "single": True,
        "horizon": 120,
        "views": ["score_curve", "fss_curve", "spatial_comparison"],
        "metrics": ["csi", "pod", "far", "fss"],
        "thresholds": [20.0, 35.0],
        "diagnostic": 20.0,
        "diagnostic_metric": "csi",
        "windows": [5],
        "spatial": 60,
    },
]


def test_reference_page_has_all_current_examples_under_contract():
    assert len(_reference_prompts()) == 14


@pytest.mark.parametrize("index, expected", list(enumerate(REGULAR_CONTRACTS)))
def test_regular_reference_prompt_matches_its_figures(index, expected):
    prompt = _reference_prompts()[index]
    batch = expected["scope"] != "single"
    request = apply_explicit_multimodel_contract(
        default_multimodel_request(batch=batch, scope=expected["scope"]),
        prompt,
        scope=expected["scope"],
    )
    request = validate_multimodel_request(
        request, batch=batch, scope=expected["scope"]
    )

    assert request["views"] == expected["views"]
    assert request["thresholds_dbz"] == expected["thresholds"]
    assert request["score_curve_metrics"] == expected["score_metrics"]
    assert request["multi_metric_metrics"] == expected["multi_metrics"]
    assert request["diagnostic_threshold_dbz"] == expected["diagnostic"]
    assert request["spatial_lead_minutes"] == expected["spatial"]
    assert request["unmatched_requests"] == []


@pytest.mark.parametrize(
    "offset, expected", list(enumerate(BENCHMARK_CONTRACTS, start=9))
)
def test_benchmark_reference_prompt_matches_its_figures(offset, expected):
    prompt = _reference_prompts()[offset]
    request = apply_explicit_benchmark_contract(
        default_benchmark_request(
            single=expected["single"], common_horizon=expected["horizon"]
        ),
        prompt,
    )
    request = validate_benchmark_request(
        request,
        single=expected["single"],
        common_horizon=expected["horizon"],
    )

    assert request["views"] == expected["views"]
    assert request["metrics"] == expected["metrics"]
    assert request["thresholds_dbz"] == expected["thresholds"]
    assert request["diagnostic_threshold_dbz"] == expected["diagnostic"]
    assert request["diagnostic_metric"] == expected["diagnostic_metric"]
    assert request["fss_windows"] == expected["windows"]
    assert request["spatial_lead_minutes"] == expected["spatial"]
    assert request["unmatched_requests"] == []

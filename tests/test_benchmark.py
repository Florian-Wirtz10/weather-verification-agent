from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.benchmark_request import (
    default_benchmark_request,
    validate_benchmark_request,
)
from data_io.benchmark import build_benchmark_catalog, load_benchmark_arrays
from verification.benchmark import (
    _frame_row,
    benchmark_available_samples,
    common_horizon_minutes,
    run_benchmark,
    summarize_benchmark_period,
)
from visualization.benchmark_plots import (
    plot_benchmark_fss_curves,
    plot_benchmark_score_curves,
)


def test_common_horizon_uses_shortest_native_model():
    assert common_horizon_minutes(["StormDiT", "Persistence"]) == 360
    assert common_horizon_minutes(["StormDiT", "DiffCast"]) == 120
    assert common_horizon_minutes(["StormDiT", "ConvLSTM"]) == 72
    assert common_horizon_minutes(["StormDiT", "Fenglei"]) == 180


def test_request_rejects_lead_beyond_strict_common_horizon():
    request = default_benchmark_request(single=False, common_horizon=72)
    request["lead_range_minutes"]["end"] = 120

    with pytest.raises(ValueError, match=r"through \+72 minutes"):
        validate_benchmark_request(request, single=False, common_horizon=72)


def test_request_explicitly_rejects_spatial_view_for_multiple_initializations():
    request = default_benchmark_request(single=False, common_horizon=72)
    request["views"].append("spatial_comparison")

    validated = validate_benchmark_request(request, single=False, common_horizon=72)

    assert "spatial_comparison" not in validated["views"]
    assert "空间对比图仅支持单一起报" in validated["unmatched_requests"]


def test_fss_and_contingency_are_computed_on_the_same_grid():
    truth = np.array([[1, 1], [0, 0]], dtype=float)
    forecast = np.array([[1, 0], [1, 0]], dtype=float)

    row = _frame_row(forecast, truth, threshold=0.5, window_sizes=(1,))

    assert row["hits"] == 1
    assert row["misses"] == 1
    assert row["false_alarms"] == 1
    assert row["correct_negatives"] == 1
    assert row["csi"] == pytest.approx(1 / 3)
    assert row["fss_w1"] == pytest.approx(0.5)


def test_period_summary_pools_counts_instead_of_averaging_sample_scores():
    small = _frame_row(np.ones((1, 1)), np.ones((1, 1)), 0.5, (1,))
    large = _frame_row(np.zeros((3, 3)), np.ones((3, 3)), 0.5, (1,))
    rows = pd.DataFrame(
        [
            {"model": "StormDiT", "threshold_dbz": 20.0, **small},
            {"model": "StormDiT", "threshold_dbz": 20.0, **large},
        ]
    )

    summary = summarize_benchmark_period(rows, fss_windows=(1,))

    assert summary.iloc[0]["csi"] == pytest.approx(0.1)
    assert summary.iloc[0]["valid_count"] == 10


def test_benchmark_plots_render_selected_models_and_thresholds():
    rows = []
    for model, offset in (("StormDiT", 0.1), ("ConvLSTM", 0.0)):
        for lead in (6, 12):
            for threshold in (20.0, 35.0):
                rows.append(
                    {
                        "model": model,
                        "lead_minutes": lead,
                        "threshold_dbz": threshold,
                        "csi": 0.5 + offset,
                        "fss_w1": 0.6 + offset,
                        "fss_w3": 0.7 + offset,
                        "fss_w5": 0.8 + offset,
                    }
                )
    frame = pd.DataFrame(rows)

    score = plot_benchmark_score_curves(frame, "csi", [20.0, 35.0])
    fss = plot_benchmark_fss_curves(frame, 20.0, (1, 3, 5))

    assert len(score.axes) == 2
    assert len(fss.axes) == 3


@pytest.mark.skipif(
    not Path("external/delivery/apr11_wuhan_full/index.csv").is_file(),
    reason="private delivery package is not part of the repository",
)
def test_delivery_stormdit_scores_match_teacher_crosscheck_csv():
    package = Path("external/delivery/apr11_wuhan_full")
    catalog = build_benchmark_catalog(package)
    result = run_benchmark(
        catalog,
        ["StormDiT"],
        thresholds=(20.0, 35.0, 45.0),
        lead_range=(6, 72),
    )
    provided = pd.read_csv(package / "metrics/metrics_per_lead.csv")
    provided = provided[provided["model"] == "StormDiT"]
    merged = result["summary"].merge(
        provided, on=["model", "lead_minutes"], suffixes=("_ours", "_provided")
    )
    threshold_20 = merged[np.isclose(merged["threshold_dbz"], 20.0)]

    np.testing.assert_allclose(threshold_20["csi"], threshold_20["csi_20"], atol=1e-12)
    np.testing.assert_allclose(threshold_20["fss_w5"], threshold_20["fss_20_w5"], atol=5e-9)
    np.testing.assert_allclose(threshold_20["mae_ours"], threshold_20["mae_provided"], atol=1e-12)


@pytest.mark.skipif(
    not Path("external/delivery/apr11_wuhan_full/index.csv").is_file()
    or not Path("external/20250411风雷预测资料").is_dir(),
    reason="private benchmark and Fenglei packages are not part of the repository",
)
def test_fenglei_joins_the_benchmark_on_23_strict_common_samples():
    catalog = build_benchmark_catalog(
        "external/delivery/apr11_wuhan_full",
        fenglei_directory="external/20250411风雷预测资料",
    )
    samples = benchmark_available_samples(
        catalog,
        ["StormDiT", "Fenglei"],
        lead_range=(6, 72),
    )

    assert len(samples) == 23
    arrays = load_benchmark_arrays(samples[0], ["Fenglei"], catalog=catalog, leads=[6])
    assert arrays["forecasts"]["Fenglei"].shape == (30, 128, 128)
    assert np.isfinite(arrays["forecasts"]["Fenglei"][0]).all()

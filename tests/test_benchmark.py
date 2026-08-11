from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from agent.benchmark_request import (
    apply_explicit_benchmark_contract,
    default_benchmark_request,
    validate_benchmark_request,
)
from data_io.benchmark import (
    BenchmarkCatalog,
    BenchmarkSample,
    build_benchmark_catalog,
    index_benchmark_stormdit_forecasts,
    load_benchmark_arrays,
    read_benchmark_stormdit_forecasts,
)
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
    plot_benchmark_score_heatmap,
)


def test_common_horizon_uses_shortest_native_model():
    assert common_horizon_minutes(["StormDiT", "Persistence"]) == 360
    assert common_horizon_minutes(["StormDiT", "DiffCast"]) == 120
    assert common_horizon_minutes(["StormDiT", "ConvLSTM"]) == 72
    assert common_horizon_minutes(["StormDiT", "Fenglei"]) == 180


def test_benchmark_stormdit_can_join_business_catalog_with_native_domain(tmp_path):
    path = tmp_path / "sample_119.npz"
    analysis = datetime(2025, 4, 11, 1, 36)
    values = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    np.savez(
        path,
        analysis_time=analysis.isoformat(),
        pred_StormDiT_dbz=values,
        pred_StormDiT_lead_minutes=np.array([6, 12]),
        lat=np.array([32.0, 31.97]),
        lon=np.array([112.0, 112.03, 112.06]),
    )
    sample = BenchmarkSample(
        sample_id=119,
        path=path,
        start_time_utc=analysis - timedelta(minutes=96),
        analysis_time_utc=analysis,
        analysis_time_bjt=analysis + timedelta(hours=8),
    )
    catalog = BenchmarkCatalog(
        package_dir=tmp_path,
        samples=(sample,),
        models=("StormDiT",),
        lat_range=(31.97, 32.0),
        lon_range=(112.0, 112.06),
        resolution_degrees=0.03,
        fenglei_forecasts={},
    )

    records = index_benchmark_stormdit_forecasts(catalog)
    fields = read_benchmark_stormdit_forecasts(path, (6, 12))

    record = records[(analysis, 6)]
    assert record.source_kind == "benchmark_stormdit"
    assert record.domain == (31.97, 32.0, 112.0, 112.06)
    assert fields[6].valid_time == analysis + timedelta(minutes=6)
    assert fields[6].lat.tolist() == [31.97, 32.0]
    assert np.array_equal(fields[6].data, values[0, ::-1, :])


def test_request_rejects_lead_beyond_strict_common_horizon():
    request = default_benchmark_request(single=False, common_horizon=72)
    request["lead_range_minutes"]["end"] = 120

    with pytest.raises(ValueError, match=r"through \+72 minutes"):
        validate_benchmark_request(request, single=False, common_horizon=72)


def test_single_default_does_not_invent_a_static_spatial_figure():
    request = default_benchmark_request(single=True, common_horizon=72)

    assert request["views"] == [
        "score_curve",
        "fss_curve",
        "period_summary",
        "score_heatmap",
    ]
    assert request["diagnostic_threshold_dbz"] == 20.0
    assert request["spatial_lead_minutes"] is None


def test_explicit_prompt_controls_views_and_diagnostic_threshold():
    parsed = default_benchmark_request(single=True, common_horizon=72)
    parsed["views"].insert(0, "spatial_comparison")
    parsed["spatial_lead_minutes"] = 60
    parsed["diagnostic_threshold_dbz"] = 35

    contracted = apply_explicit_benchmark_contract(
        parsed,
        "查看6到72分钟CSI、POD、FAR和多尺度FSS，生成公共样本汇总、时效曲线和热力图，并在20 dBZ阈值下汇总和解读。",
    )

    assert contracted["views"] == [
        "score_curve",
        "fss_curve",
        "period_summary",
        "score_heatmap",
    ]
    assert contracted["diagnostic_threshold_dbz"] == 20.0
    assert contracted["spatial_lead_minutes"] is None


def test_unsupported_animation_is_not_replaced_by_spatial_comparison():
    parsed = default_benchmark_request(single=True, common_horizon=72)

    contracted = apply_explicit_benchmark_contract(
        parsed,
        "生成6到72分钟多模型预报动画和CSI曲线。",
    )

    assert contracted["views"] == ["score_curve"]
    assert contracted["unmatched_requests"] == ["多模型公共样本基准暂不支持动画"]


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


def test_benchmark_four_threshold_titles_use_a_two_by_two_layout():
    rows = []
    for threshold in (10.0, 20.0, 30.0, 40.0):
        for lead in (6, 12):
            rows.append(
                {
                    "model": "StormDiT",
                    "lead_minutes": lead,
                    "threshold_dbz": threshold,
                    "csi": 0.5,
                }
            )
    figure = plot_benchmark_score_curves(pd.DataFrame(rows), "csi", [10, 20, 30, 40])
    try:
        assert len(figure.axes) == 4
        assert [axis.get_title() for axis in figure.axes] == [
            "10 dBZ", "20 dBZ", "30 dBZ", "40 dBZ"
        ]
        assert len({tuple(axis.get_position().bounds[:2]) for axis in figure.axes}) == 4
    finally:
        plt.close(figure)


def test_benchmark_model_lead_heatmap_uses_fixed_blue_to_red_score_scale():
    frame = pd.DataFrame(
        [
            {
                "model": model,
                "lead_minutes": lead,
                "threshold_dbz": 20.0,
                "csi": score,
            }
            for model, score in (("StormDiT", 0.7), ("ConvLSTM", 0.4))
            for lead in (6, 12)
        ]
    )

    figure = plot_benchmark_score_heatmap(frame, "csi", 20.0)
    try:
        image = figure.axes[0].images[0]
        assert image.get_cmap().name == "RdBu_r"
        assert image.get_clim() == (0.0, 1.0)
    finally:
        plt.close(figure)


@pytest.mark.skipif(
    not Path("external/datasets/benchmarks/2025-04-11/wuhan-zero-shot/index.csv").is_file(),
    reason="private delivery package is not part of the repository",
)
def test_delivery_stormdit_scores_match_teacher_crosscheck_csv():
    package = Path("external/datasets/benchmarks/2025-04-11/wuhan-zero-shot")
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
    not Path("external/datasets/benchmarks/2025-04-11/wuhan-zero-shot/index.csv").is_file()
    or not Path("external/datasets/cases/2025-04-11/forecasts/fenglei").is_dir(),
    reason="private benchmark and Fenglei packages are not part of the repository",
)
def test_fenglei_joins_the_benchmark_on_23_strict_common_samples():
    catalog = build_benchmark_catalog(
        "external/datasets/benchmarks/2025-04-11/wuhan-zero-shot",
        fenglei_directory="external/datasets/cases/2025-04-11/forecasts/fenglei",
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

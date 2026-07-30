from datetime import datetime, timedelta
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from data_io.grid import GriddedField
from visualization.comparison_plots import (
    plot_multimodel_forecast_observed_rate_scatter,
    plot_multimodel_multi_metric_curves,
    plot_multimodel_same_valid_time_comparison,
    plot_multimodel_score_distribution,
    plot_multimodel_variable_correlation_heatmap,
    plot_triplet_score_curves,
    render_multimodel_reflectivity_animation,
)


def test_triplet_score_plot_uses_product_line_styles_without_grid():
    result = {
        "per_lead": [
            {
                "lead_minutes": lead,
                "valid_time_utc": datetime(2025, 4, 11, 0, lead),
                "model": {15.0: {"csi": model}},
                "fenglei": {15.0: {"csi": fenglei}},
            }
            for lead, model, fenglei in ((6, 0.8, 0.7), (12, 0.7, 0.6))
        ]
    }

    figure = plot_triplet_score_curves(result, "csi")

    try:
        axis = figure.axes[0]
        assert len(axis.lines) == 2
        assert axis.lines[0].get_linestyle() == "-"
        assert axis.lines[1].get_linestyle() == "--"
        assert not any(line.get_visible() for line in axis.get_xgridlines())
    finally:
        plt.close(figure)


def test_triplet_gif_stays_below_streamlit_resize_limit_and_keeps_frames():
    init_time = datetime(2025, 4, 11)
    lat = np.array([30.0, 31.0])
    lon = np.array([112.0, 113.0])
    frames = []
    for lead, offset in ((6, 0.0), (12, 8.0)):
        valid_time = init_time + timedelta(minutes=lead)

        def field(product, values):
            return GriddedField(
                data=np.asarray(values, dtype=float) + offset,
                lat=lat,
                lon=lon,
                valid_time=valid_time,
                product=product,
                units="dBZ",
                init_time=init_time,
                lead_minutes=lead,
            )

        frames.append(
            {
                "lead_minutes": lead,
                "forecasts": {
                    "stormdit": field("StormDiT", [[10, 20], [30, 40]]),
                    "fenglei": field("Fenglei", [[12, 18], [28, 38]]),
                },
                "observation": field("CREF", [[8, 22], [32, 36]]),
            }
        )

    gif = render_multimodel_reflectivity_animation(frames, fps=2)
    with Image.open(BytesIO(gif)) as image:
        assert image.width < 1460
        assert image.n_frames == 2


def test_same_valid_time_plot_labels_each_init_lead_and_one_observation():
    valid_time = datetime(2025, 4, 11, 3)
    lat = np.array([30.0, 31.0])
    lon = np.array([112.0, 113.0])

    def field(product, init_time, lead, offset):
        return GriddedField(
            data=np.array([[10, 20], [30, 40]], dtype=float) + offset,
            lat=lat,
            lon=lon,
            valid_time=valid_time,
            product=product,
            units="dBZ",
            init_time=init_time,
            lead_minutes=lead,
        )

    observation = field("CREF", None, None, 0)
    items = []
    for init_time, lead, offset in (
        (datetime(2025, 4, 11, 0), 180, 1),
        (datetime(2025, 4, 11, 1), 120, 2),
        (datetime(2025, 4, 11, 2), 60, 3),
    ):
        items.append(
            {
                "init_time_utc": init_time,
                "init_time_bjt": init_time + timedelta(hours=8),
                "lead_minutes": lead,
                "frame": {
                    "forecasts": {
                        "stormdit": field("StormDiT", init_time, lead, offset),
                    },
                    "observation": observation,
                },
            }
        )
    comparison = {
        "valid_time_utc": valid_time,
        "valid_time_bjt": valid_time + timedelta(hours=8),
        "model_ids": ["stormdit"],
        "items": items,
    }

    figure = plot_multimodel_same_valid_time_comparison(
        comparison,
        "stormdit",
    )
    try:
        titles = [axis.get_title() for axis in figure.axes]
        assert sum("Lead +" in title for title in titles) == 3
        assert sum("CREF observation" in title for title in titles) == 1
        assert "one valid time" in figure._suptitle.get_text()
    finally:
        plt.close(figure)


def _comparison_result():
    entries = []
    for lead, offset in ((6, 0.0), (12, -0.1), (18, -0.2)):
        entries.append(
            {
                "lead_minutes": lead,
                "models": {
                    model_id: {
                        20.0: {
                            "csi": 0.8 + offset - model_offset,
                            "pod": 0.9 + offset - model_offset,
                            "far": 0.1 - offset + model_offset,
                        }
                    }
                    for model_id, model_offset in (
                        ("stormdit", 0.0),
                        ("fenglei", 0.05),
                    )
                },
            }
        )
    return {
        "context": {"model_ids": ["stormdit", "fenglei"]},
        "per_lead": entries,
    }


def _batch_frame():
    rows = []
    for model_index, model_id in enumerate(("stormdit", "fenglei")):
        for init_index in range(4):
            for lead in (6, 12, 18):
                for threshold in (10.0, 20.0, 30.0, 40.0):
                    base = 0.75 - lead / 100 - threshold / 200 - model_index * 0.03
                    rows.append(
                        {
                            "model_id": model_id,
                            "init_time_utc": f"2026-04-08T00:{init_index * 6:02d}:00",
                            "init_time_bjt": f"2026-04-08T08:{init_index * 6:02d}:00",
                            "lead_minutes": lead,
                            "threshold_dbz": threshold,
                            "hits": 30 + init_index,
                            "misses": 10 + lead // 6,
                            "false_alarms": 8 + model_index,
                            "valid_count": 100,
                            "csi": base,
                            "pod": min(1.0, base + 0.12),
                            "far": max(0.0, 0.28 - base / 5),
                            "bias_score": 1.0 + model_index * 0.05,
                            "forecast_mean_dbz": 15 + init_index + model_index,
                            "observation_mean_dbz": 14 + init_index,
                            "forecast_max_dbz": 50 + init_index,
                            "observation_max_dbz": 49 + init_index,
                            "mean_reflectivity_error_dbz": 1 + model_index,
                            "reflectivity_mae_dbz": 3 + lead / 30 + model_index,
                            "centroid_distance_km": 5 + lead / 6 + model_index,
                        }
                    )
    return pd.DataFrame(rows)


def test_multi_metric_curve_uses_metric_colours_and_model_styles():
    figure = plot_multimodel_multi_metric_curves(
        _comparison_result(), ["csi", "pod", "far"], 20
    )
    try:
        axis = figure.axes[0]
        assert len(axis.lines) == 6
        assert {line.get_linestyle() for line in axis.lines} == {"-", "--"}
        assert len({line.get_color() for line in axis.lines}) == 3
        assert axis.get_ylim() == (0.0, 1.0)
    finally:
        plt.close(figure)


def test_batch_publication_views_support_multiple_thresholds_and_correlations():
    frame = _batch_frame()
    figures = [
        plot_multimodel_score_distribution(
            frame, "csi", [10, 20, 30, 40]
        ),
        plot_multimodel_forecast_observed_rate_scatter(
            frame, [10, 20, 30, 40]
        ),
        plot_multimodel_variable_correlation_heatmap(
            frame, 20, method="spearman"
        ),
    ]
    try:
        assert "thresholds" in figures[0]._suptitle.get_text()
        assert len(figures[0].axes) == 2
        assert "Forecast rate" in figures[1]._suptitle.get_text()
        assert figures[1].legends == []
        assert figures[1].axes[-1].get_legend().get_title().get_text() == "Threshold"
        figures[1].canvas.draw()
        renderer = figures[1].canvas.get_renderer()
        legend_box = figures[1].axes[-1].get_legend().get_window_extent(renderer)
        xlabel_box = figures[1].axes[-1].xaxis.label.get_window_extent(renderer)
        assert not legend_box.overlaps(xlabel_box)
        assert "Variable correlation" in figures[2]._suptitle.get_text()
        assert "Spearman" in figures[2].axes[-1].get_ylabel()
    finally:
        for figure in figures:
            plt.close(figure)

"""Regressions for scientific meaning retained by exported figures."""

from datetime import datetime
from types import SimpleNamespace
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.text import Text
import numpy as np
import pandas as pd
import pytest

from visualization import plots
from visualization.benchmark_plots import (
    plot_benchmark_far_bars,
    plot_benchmark_period_summary,
    plot_benchmark_score_heatmap,
    plot_benchmark_spatial_comparison,
)
from visualization.comparison_plots import plot_multimodel_score_heatmaps


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("view", ["lead", "period"])
def test_undefined_far_bar_is_marked_but_valid_zero_is_not(view):
    rows = pd.DataFrame([
        {"model": model, "lead_minutes": 6, "threshold_dbz": 35.0, "far": value}
        for model, value in (("StormDiT", .25), ("NowCastNet", np.nan), ("PredRNN", 0.0))
    ])
    figure = (plot_benchmark_far_bars(rows, [35]) if view == "lead" else
              plot_benchmark_period_summary(rows, ["far"], 35))
    axis = figure.axes[0]
    bars = axis.patches
    np.testing.assert_allclose([bar.get_height() for bar in bars], [0.0, np.nan, .25])
    np.testing.assert_allclose([bar.get_facecolor() for bar in bars], [
        to_rgba("#59A14F"), to_rgba("#E6B800"), to_rgba("#E15759")
    ])
    markers = [text for text in axis.texts if text.get_text() == "N/A"]
    assert len(markers) == 1
    assert markers[0].get_position()[0] == pytest.approx(bars[1].get_x() + bars[1].get_width() / 2)
    assert axis.get_ylim() == (0.0, 1.0)


def test_spatial_export_retains_init_lead_and_valid_time_without_top_titles():
    frame = {
        "sample": SimpleNamespace(analysis_time_bjt=datetime(2025, 4, 11, 23, 30)),
        "lead_minutes": 72,
        "lat": np.array([31., 30.]), "lon": np.array([113., 114.]),
        "truth": np.full((2, 2), 20.),
        "forecasts": {"PredRNN": np.full((2, 2), 20.)},
    }
    figure = plot_benchmark_spatial_comparison(frame)
    text = " ".join(item.get_text() for item in figure.findobj(Text))
    assert "Init 2025-04-11 23:30 BJT" in text
    assert "+72 min" in text
    assert "Valid 2025-04-12 00:42 BJT" in text
    assert figure._suptitle is None
    assert all(axis.get_title() == "" for axis in figure.axes)
    figure.canvas.draw()


def test_period_export_retains_selected_threshold_without_top_titles():
    period = pd.DataFrame([{"model": "PredRNN", "threshold_dbz": 35., "csi": .4}])
    figure = plot_benchmark_period_summary(period, ["csi"], 35.)
    assert "35 dBZ" in " ".join(item.get_text() for item in figure.findobj(Text))
    assert figure._suptitle is None
    assert all(axis.get_title() == "" for axis in figure.axes)
    figure.canvas.draw()


def _bias_rows(values):
    return pd.DataFrame([
        {"model_id": model_id, "model": model, "init_time_bjt": datetime(2025, 4, 11),
         "lead_minutes": lead, "threshold_dbz": 20., "bias_score": value}
        for model_id, model, model_values in (("stormdit", "StormDiT", values),
                                              ("fenglei", "Fenglei", values[::-1]))
        for lead, value in zip((6, 12, 18), model_values)
    ])


@pytest.mark.parametrize("view", ["benchmark", "comparison"])
def test_bias_heatmap_distinguishes_above_one_scores_with_shared_reference_scale(view):
    rows = _bias_rows([1., 2., 5.])
    rows.loc[rows["model_id"] == "fenglei", "bias_score"] = [.5, 1., 2.]
    figure = (plot_benchmark_score_heatmap(rows, "bias_score", 20.) if view == "benchmark" else
              plot_multimodel_score_heatmaps(rows, "bias_score", 20.))
    images = [axis.images[0] for axis in figure.axes[:1 if view == "benchmark" else 2]]
    for image in images:
        assert image.norm(1.) == pytest.approx(.5)
        assert image.norm.vmin <= 1. < image.norm.vmax
        assert image.norm.vmax >= 5.
        assert len({tuple(image.cmap(image.norm(value))) for value in (1., 2., 5.)}) == 3
    assert len({image.get_clim() for image in images}) == 1
    np.testing.assert_allclose(np.sort(images[0].get_array(), axis=None),
                               [.5, 1., 1., 2., 2., 5.] if view == "benchmark" else [1., 2., 5.])


@pytest.mark.parametrize("value", [1., 5., np.nan])
@pytest.mark.parametrize("view", ["benchmark", "comparison"])
def test_bias_heatmap_constant_or_undefined_scores_render_with_finite_scale(view, value):
    rows = _bias_rows([value] * 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        figure = (plot_benchmark_score_heatmap(rows, "bias_score", 20.) if view == "benchmark" else
                  plot_multimodel_score_heatmaps(rows, "bias_score", 20.))
        figure.canvas.draw()
    for axis in figure.axes:
        for image in axis.images:
            assert np.isfinite(image.get_clim()).all()
            assert image.norm.vmin < image.norm.vmax
    image = figure.axes[0].images[0]
    assert image.norm(1.) == pytest.approx(.5)
    if np.isnan(value):
        assert np.ma.getmaskarray(image.get_array()).all()


def test_reflectivity_colormap_supports_matplotlib_38_constructor(monkeypatch):
    # Matplotlib 3.8 accepts only colors/name/N. Keep the real colormap behavior.
    def compatible_colormap(colors, name="from_list", N=None):
        return ListedColormap(colors, name=name) if N is None else ListedColormap(colors, name=name, N=N)

    monkeypatch.setattr(plots, "ListedColormap", compatible_colormap)
    cmap, norm = plots._reflectivity_cmap_norm()
    np.testing.assert_allclose(cmap.get_under(), [1., 1., 1., 1.])
    np.testing.assert_allclose(cmap.get_bad(), [1., 1., 1., 1.])
    np.testing.assert_allclose(cmap.get_over(), to_rgba("#a891ea"))
    np.testing.assert_allclose(cmap(norm(2.)), to_rgba("#94f75f"))

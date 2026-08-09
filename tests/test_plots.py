from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, Normalize
import numpy as np
import pytest

from data_io.grid import GriddedField
from visualization.plots import (
    _REFLECTIVITY_BOUNDS,
    _break_time_gaps,
    _reflectivity_cmap_norm,
    _reflectivity_display_data,
    plot_forecast_observed_rate_trajectory,
    plot_forecast_observed_rate_threshold_comparison,
    plot_reflectivity_comparison,
    plot_reflectivity_error,
    plot_score_time_series,
)


@pytest.fixture(autouse=True)
def _close_test_figures():
    yield
    plt.close("all")


def test_reflectivity_display_uses_a_common_zero_no_echo_floor():
    displayed = _reflectivity_display_data(np.array([[-18.0, -0.5, 0.0, 12.0, np.nan]]))

    np.testing.assert_allclose(displayed[0, :4], [0.0, 0.0, 0.0, 12.0])
    assert np.isnan(displayed[0, 4])


def test_reflectivity_display_can_apply_a_visual_only_weak_echo_cutoff():
    displayed = _reflectivity_display_data(
        np.array([[-18.0, 0.0, 10.0, 15.0, 30.0, np.nan]]),
        minimum_display_dbz=15,
    )

    assert np.isnan(displayed[0, :3]).all()
    np.testing.assert_allclose(displayed[0, 3:5], [15.0, 30.0])
    assert np.isnan(displayed[0, 5])
    cmap, norm = _reflectivity_cmap_norm(15)
    np.testing.assert_allclose(norm.boundaries[0], 15.0)
    np.testing.assert_allclose(cmap(norm(17.0)), (1.0, 1.0, 84 / 255, 1.0))


def test_reflectivity_colour_table_matches_the_course_threshold_reference():
    cmap, norm = _reflectivity_cmap_norm()

    np.testing.assert_allclose(
        _REFLECTIVITY_BOUNDS,
        [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
    )
    np.testing.assert_allclose(cmap.get_under(), (1.0, 1.0, 1.0, 1.0))
    np.testing.assert_allclose(cmap(norm(2.0)), (148 / 255, 247 / 255, 95 / 255, 1.0))
    np.testing.assert_allclose(cmap(norm(72.0)), (168 / 255, 145 / 255, 234 / 255, 1.0))


def test_score_series_breaks_missing_native_leads_and_uses_30_minute_ticks():
    figure = plot_score_time_series(
        [6, 12, 18, 36],
        {35: [0.2, 0.3, 0.25, 0.1]},
        "csi",
    )
    line = figure.axes[0].lines[0]
    assert np.isnan(line.get_xdata()).sum() == 1
    ticks = figure.axes[0].xaxis.get_major_locator().tick_values(0, 60)
    np.testing.assert_allclose(np.diff(ticks), 30)
    assert not any(line.get_visible() for line in figure.axes[0].get_xgridlines())
    assert not any(line.get_visible() for line in figure.axes[0].get_ygridlines())


def test_rate_trajectory_uses_blue_red_continuous_lead_colours():
    tables = [
        {"hits": 4, "misses": 1, "false_alarms": 2, "valid_count": 10},
        {"hits": 3, "misses": 2, "false_alarms": 1, "valid_count": 10},
    ]

    figure = plot_forecast_observed_rate_trajectory([6, 12], tables, 15)

    main_axes = [axis for axis in figure.axes if axis.get_xlabel() == "Observed rate"]
    assert len(main_axes) == 2  # joint scatter plus bottom marginal density
    assert all(axis.collections for axis in main_axes)
    assert all(not any(line.get_visible() for line in axis.get_xgridlines()) for axis in main_axes)
    scatter = main_axes[0].collections[0]
    assert isinstance(scatter.norm, Normalize)
    assert not isinstance(scatter.norm, BoundaryNorm)
    np.testing.assert_allclose(scatter.cmap(0.0)[:3], (42 / 255, 120 / 255, 214 / 255))
    np.testing.assert_allclose(scatter.cmap(1.0)[:3], (227 / 255, 73 / 255, 72 / 255))
    np.testing.assert_allclose(figure.axes[-1].get_xticks(), [6, 12])


def test_rate_threshold_comparison_keeps_thresholds_in_one_joint_figure():
    tables = {
        15: [
            {"hits": 4, "misses": 1, "false_alarms": 2, "valid_count": 10},
            {"hits": 3, "misses": 2, "false_alarms": 1, "valid_count": 10},
        ],
        35: [
            {"hits": 1, "misses": 1, "false_alarms": 1, "valid_count": 10},
            {"hits": 1, "misses": 2, "false_alarms": 0, "valid_count": 10},
        ],
    }

    figure = plot_forecast_observed_rate_threshold_comparison([6, 12], tables)

    joint = figure.axes[0]
    assert len(joint.collections) == 2
    legend = figure.axes[-1].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["15 dBZ", "35 dBZ"]
    assert not any(line.get_visible() for line in joint.get_xgridlines())


def test_reflectivity_comparison_adds_boundaries_without_changing_extent_or_colourbar():
    common = {
        "lat": np.array([29.0, 33.5]),
        "lon": np.array([108.0, 116.5]),
        "valid_time": datetime(2025, 4, 11, 2, 18),
        "units": "dBZ",
    }
    forecast = GriddedField(data=np.array([[10.0, 20.0], [30.0, 40.0]]), product="forecast", **common)
    obs = GriddedField(data=np.array([[15.0, 25.0], [35.0, 45.0]]), product="obs", **common)

    figure = plot_reflectivity_comparison(forecast, obs)

    for axis in figure.axes[:2]:
        assert axis.lines
        np.testing.assert_allclose(axis.get_xlim(), [108.0, 116.5])
        np.testing.assert_allclose(axis.get_ylim(), [29.0, 33.5])
        np.testing.assert_allclose(axis.images[0].norm.boundaries, _REFLECTIVITY_BOUNDS)


def test_break_time_gaps_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        _break_time_gaps([6, 12], [0.1])


def test_reflectivity_error_tolerates_masked_outside_domain_cells():
    common = {
        "lat": np.array([28.0, 28.01]),
        "lon": np.array([110.0, 110.01]),
        "valid_time": datetime(2025, 4, 11, 2, 18),
        "units": "dBZ",
    }
    forecast = GriddedField(
        data=np.array([[np.nan, 20.0], [30.0, 40.0]]),
        product="forecast",
        **common,
    )
    obs = GriddedField(
        data=np.array([[10.0, 10.0], [20.0, 50.0]]),
        product="obs",
        **common,
    )

    figure = plot_reflectivity_error(forecast, obs)

    assert figure.axes[0].images[0].get_clim() == (-10.0, 10.0)

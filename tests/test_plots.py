from datetime import datetime

import numpy as np
import pytest

from data_io.grid import GriddedField
from visualization.plots import (
    _REFLECTIVITY_BOUNDS,
    _break_time_gaps,
    _reflectivity_cmap_norm,
    _reflectivity_display_data,
    plot_reflectivity_error,
    plot_score_time_series,
)


def test_reflectivity_display_uses_a_common_zero_no_echo_floor():
    displayed = _reflectivity_display_data(np.array([[-18.0, -0.5, 0.0, 12.0, np.nan]]))

    np.testing.assert_allclose(displayed[0, :4], [0.0, 0.0, 0.0, 12.0])
    assert np.isnan(displayed[0, 4])


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

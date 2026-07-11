from datetime import datetime

import numpy as np
import pytest

from data_io.grid import GriddedField
from visualization.plots import (
    _break_time_gaps,
    plot_reflectivity_error,
    plot_score_time_series,
)


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

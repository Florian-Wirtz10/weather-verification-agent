"""Matplotlib figure builders for forecast/observation/error fields.

Phase 2 adds a fixed menu of real-data plotting functions (reflectivity
field, reflectivity error, categorical-score time series) for the NL agent
layer to call by name — the agent picks which of these to run and with what
parameters, it does not write its own plotting code.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap


def plot_field(array, title, cmap="Blues"):
    """Render a single 2D field as a heatmap figure."""
    fig, ax = plt.subplots()
    im = ax.imshow(array, cmap=cmap, origin="lower")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mm")
    return fig


def plot_error(forecast, obs, title="Forecast - Obs"):
    """Render the forecast-minus-obs error field with a diverging, zero-centered colormap."""
    error = forecast - obs
    limit = float(np.max(np.abs(error))) or 1.0

    fig, ax = plt.subplots()
    im = ax.imshow(error, cmap="RdBu_r", origin="lower", vmin=-limit, vmax=limit)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mm")
    return fig


# ---------------------------------------------------------------------------
# Phase 2: real-data (geo-referenced) plots. Fixed categorical hue order —
# never cycled/reassigned per-series — per the dataviz skill's color formula.
# ---------------------------------------------------------------------------

CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]

_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"
_MUTED = "#898781"

# Standard 18-level radar reflectivity color table (5 dBZ bins, -10..75 dBZ) —
# the convention meteorologists expect for dBZ fields, not a generic palette.
_REFLECTIVITY_BOUNDS = np.arange(-10, 76, 5)
_REFLECTIVITY_COLORS = [
    "#646464", "#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00",
    "#fdf802", "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000", "#f800fd",
    "#9854c6", "#fdfdfd", "#000000",
]


def _reflectivity_cmap_norm():
    cmap = ListedColormap(_REFLECTIVITY_COLORS)
    norm = BoundaryNorm(_REFLECTIVITY_BOUNDS, cmap.N)
    return cmap, norm


def _diverging_cmap():
    return LinearSegmentedColormap.from_list("diverging_blue_red", ["#2a78d6", "#f0efec", "#e34948"])


def _style_axes(ax):
    ax.grid(True, color=_GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_AXIS)
    ax.spines["bottom"].set_color(_AXIS)
    ax.tick_params(colors=_MUTED)


def plot_reflectivity_field(field, title=None):
    """Render a GriddedField of dBZ reflectivity on its lat/lon extent, using the
    standard 18-level radar color table."""
    cmap, norm = _reflectivity_cmap_norm()
    extent = [field.lon.min(), field.lon.max(), field.lat.min(), field.lat.max()]

    fig, ax = plt.subplots()
    im = ax.imshow(field.data, cmap=cmap, norm=norm, origin="lower", extent=extent, aspect="auto")
    ax.set_title(title or f"{field.product} ({field.valid_time:%Y-%m-%d %H:%M} UTC)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, label="dBZ", ticks=_REFLECTIVITY_BOUNDS[::2])
    return fig


def plot_reflectivity_error(forecast_field, obs_field, title="Forecast - Obs (dBZ)"):
    """Render the forecast-minus-obs reflectivity error, cropped/regridded onto the
    same grid (see data_io.grid.align_pair), with a zero-centered diverging colormap."""
    diff = forecast_field.data - obs_field.data
    limit = float(np.max(np.abs(diff))) or 1.0
    extent = [obs_field.lon.min(), obs_field.lon.max(), obs_field.lat.min(), obs_field.lat.max()]

    fig, ax = plt.subplots()
    im = ax.imshow(
        diff, cmap=_diverging_cmap(), origin="lower", extent=extent,
        vmin=-limit, vmax=limit, aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, label="dBZ error")
    return fig


def plot_score_time_series(lead_minutes, scores_by_threshold, score_name, title=None):
    """Line chart of one categorical score vs forecast lead time, one line per
    threshold in a fixed hue order (never reassigned/cycled per call).

    `scores_by_threshold`: {threshold: [score value per lead_minutes entry]}.
    """
    fig, ax = plt.subplots()
    for i, threshold in enumerate(sorted(scores_by_threshold)):
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        ax.plot(
            lead_minutes, scores_by_threshold[threshold],
            marker="o", markersize=5, linewidth=2, color=color, label=str(threshold),
        )
    ax.set_xlabel("Forecast lead time (min)")
    ax.set_ylabel(score_name.upper())
    ax.set_title(title or f"{score_name.upper()} vs lead time")
    _style_axes(ax)
    ax.legend(title="Threshold", frameon=False)
    return fig

"""Matplotlib figure builders for forecast/observation/error fields."""

import matplotlib.pyplot as plt
import numpy as np


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

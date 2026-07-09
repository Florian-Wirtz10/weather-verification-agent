"""Plotting layer.

Renders forecast/observation/error fields and verification-metric charts
for the Streamlit app and generated reports. Takes arrays and metric
results as input; never computes metrics or touches storage itself.
"""

from visualization.plots import (
    plot_error,
    plot_field,
    plot_reflectivity_error,
    plot_reflectivity_field,
    plot_score_time_series,
)

__all__ = [
    "plot_error",
    "plot_field",
    "plot_reflectivity_field",
    "plot_reflectivity_error",
    "plot_score_time_series",
]

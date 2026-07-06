"""Precipitation forecast verification metrics.

Pure functions over forecast/observation arrays (categorical scores, error
statistics, threshold- and lead-time-conditioned skill scores). No I/O, no
plotting, no agent/LLM logic — keeps the metric math independently testable.
"""

from verification.metrics import (
    compute_all_metrics,
    contingency_table,
    mae,
    mean_error,
    rmse,
)

__all__ = [
    "compute_all_metrics",
    "contingency_table",
    "mae",
    "mean_error",
    "rmse",
]

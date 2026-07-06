"""Glue code driving the data_io -> verification -> visualization pipeline.

Phase 1: no natural-language parsing yet, just a typed entry point that
takes the parameters a future NL layer would extract (grid shape, seed,
thresholds) and runs the full pipeline.
"""

from data_io.synthetic import generate_synthetic_fields
from verification.metrics import compute_all_metrics
from visualization.plots import plot_error, plot_field


def run_verification(shape=(64, 64), seed=None, thresholds=(1, 5, 10)):
    """Run the synthetic-data verification pipeline and return arrays, metrics, and figures."""
    forecast, obs = generate_synthetic_fields(shape=shape, seed=seed)
    metrics = compute_all_metrics(forecast, obs, thresholds)

    figures = {
        "forecast": plot_field(forecast, "Forecast"),
        "obs": plot_field(obs, "Observation"),
        "error": plot_error(forecast, obs),
    }

    return {
        "forecast": forecast,
        "obs": obs,
        "metrics": metrics,
        "figures": figures,
    }

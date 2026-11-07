"""Precipitation verification metric implementations.

Categorical scores follow the standard 2x2 contingency table for a single
precipitation threshold; continuous scores work on the raw fields.
"""

import numpy as np


def _common_finite_values(forecast, obs):
    """Return input data and a mask where both fields contain finite values."""
    forecast_array = np.ma.asarray(forecast)
    obs_array = np.ma.asarray(obs)
    if forecast_array.shape != obs_array.shape:
        raise ValueError(
            "forecast and obs must have the same shape "
            f"(got {forecast_array.shape} and {obs_array.shape})"
        )

    forecast_data = np.asarray(np.ma.getdata(forecast_array), dtype=np.float64)
    obs_data = np.asarray(np.ma.getdata(obs_array), dtype=np.float64)
    valid = (
        ~np.ma.getmaskarray(forecast_array)
        & ~np.ma.getmaskarray(obs_array)
        & np.isfinite(forecast_data)
        & np.isfinite(obs_data)
    )
    return forecast_data, obs_data, valid


def contingency_table(forecast, obs, threshold):
    """Count a 2x2 table over cells where forecast and obs are both finite."""
    threshold_array = np.asarray(threshold)
    if threshold_array.ndim != 0 or not bool(np.isfinite(threshold_array)):
        raise ValueError(f"threshold must be a finite scalar (got {threshold!r})")

    forecast_data, obs_data, valid = _common_finite_values(forecast, obs)
    fcst_yes = forecast_data >= threshold
    obs_yes = obs_data >= threshold

    hits = int(np.sum(valid & fcst_yes & obs_yes))
    misses = int(np.sum(valid & ~fcst_yes & obs_yes))
    false_alarms = int(np.sum(valid & fcst_yes & ~obs_yes))
    correct_negatives = int(np.sum(valid & ~fcst_yes & ~obs_yes))
    valid_count = int(np.sum(valid))

    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "valid_count": valid_count,
        "invalid_count": int(valid.size - valid_count),
    }


def pod(table):
    """Probability of detection: hits / (hits + misses)."""
    denom = table["hits"] + table["misses"]
    return table["hits"] / denom if denom > 0 else float("nan")


def far(table):
    """False alarm ratio: false_alarms / (hits + false_alarms)."""
    denom = table["hits"] + table["false_alarms"]
    return table["false_alarms"] / denom if denom > 0 else float("nan")


def csi(table):
    """Critical success index: hits / (hits + misses + false_alarms)."""
    denom = table["hits"] + table["misses"] + table["false_alarms"]
    return table["hits"] / denom if denom > 0 else float("nan")


def bias_score(table):
    """Frequency bias: (hits + false_alarms) / (hits + misses)."""
    denom = table["hits"] + table["misses"]
    return (table["hits"] + table["false_alarms"]) / denom if denom > 0 else float("nan")


def ets(table):
    """Equitable threat score: hits adjusted for hits expected by chance."""
    hits, misses = table["hits"], table["misses"]
    false_alarms, correct_negatives = table["false_alarms"], table["correct_negatives"]
    total = hits + misses + false_alarms + correct_negatives
    if total == 0:
        return float("nan")

    hits_random = (hits + misses) * (hits + false_alarms) / total
    denom = hits + misses + false_alarms - hits_random
    return (hits - hits_random) / denom if denom != 0 else float("nan")


def mae(forecast, obs):
    """Mean absolute error over cells where both inputs are finite."""
    forecast_data, obs_data, valid = _common_finite_values(forecast, obs)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(forecast_data[valid] - obs_data[valid])))


def rmse(forecast, obs):
    """Root mean squared error over cells where both inputs are finite."""
    forecast_data, obs_data, valid = _common_finite_values(forecast, obs)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((forecast_data[valid] - obs_data[valid]) ** 2)))


def mean_error(forecast, obs):
    """Mean error over cells where both inputs are finite."""
    forecast_data, obs_data, valid = _common_finite_values(forecast, obs)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(forecast_data[valid] - obs_data[valid]))


CATEGORICAL_SCORES = {
    "pod": pod,
    "far": far,
    "csi": csi,
    "bias_score": bias_score,
    "ets": ets,
}


def compute_all_metrics(forecast, obs, thresholds):
    """Compute categorical scores per threshold plus overall continuous error metrics.

    Returns a dict with:
      - "categorical": {threshold: {score_name: value, ...}, ...}
      - "continuous": {"mae": ..., "rmse": ..., "mean_error": ...}
    """
    categorical = {}
    for threshold in thresholds:
        table = contingency_table(forecast, obs, threshold)
        categorical[threshold] = {
            name: score_fn(table) for name, score_fn in CATEGORICAL_SCORES.items()
        }

    continuous = {
        "mae": mae(forecast, obs),
        "rmse": rmse(forecast, obs),
        "mean_error": mean_error(forecast, obs),
    }

    return {"categorical": categorical, "continuous": continuous}

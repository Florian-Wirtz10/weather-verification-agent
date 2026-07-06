"""Precipitation verification metric implementations.

Categorical scores follow the standard 2x2 contingency table for a single
precipitation threshold; continuous scores work on the raw fields.
"""

import numpy as np


def contingency_table(forecast, obs, threshold):
    """Binarize forecast/obs at threshold and count the 2x2 contingency table."""
    fcst_yes = forecast >= threshold
    obs_yes = obs >= threshold

    hits = int(np.sum(fcst_yes & obs_yes))
    misses = int(np.sum(~fcst_yes & obs_yes))
    false_alarms = int(np.sum(fcst_yes & ~obs_yes))
    correct_negatives = int(np.sum(~fcst_yes & ~obs_yes))

    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
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
    """Mean absolute error."""
    return float(np.mean(np.abs(forecast - obs)))


def rmse(forecast, obs):
    """Root mean squared error."""
    return float(np.sqrt(np.mean((forecast - obs) ** 2)))


def mean_error(forecast, obs):
    """Mean error (bias): mean(forecast - obs)."""
    return float(np.mean(forecast - obs))


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

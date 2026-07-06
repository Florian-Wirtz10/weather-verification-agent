import numpy as np

from verification.metrics import (
    bias_score,
    compute_all_metrics,
    contingency_table,
    csi,
    ets,
    far,
    mae,
    mean_error,
    pod,
    rmse,
)


def test_contingency_table_perfect_forecast():
    obs = np.array([[0.0, 2.0], [5.0, 0.0]])
    forecast = obs.copy()

    table = contingency_table(forecast, obs, threshold=1.0)

    assert table == {"hits": 2, "misses": 0, "false_alarms": 0, "correct_negatives": 2}
    assert pod(table) == 1.0
    assert far(table) == 0.0
    assert csi(table) == 1.0
    assert bias_score(table) == 1.0
    assert ets(table) == 1.0


def test_contingency_table_all_misses():
    obs = np.array([[5.0, 5.0], [5.0, 5.0]])
    forecast = np.array([[0.0, 0.0], [0.0, 0.0]])

    table = contingency_table(forecast, obs, threshold=1.0)

    assert table == {"hits": 0, "misses": 4, "false_alarms": 0, "correct_negatives": 0}
    assert pod(table) == 0.0
    assert csi(table) == 0.0


def test_far_undefined_when_no_positive_forecasts():
    table = {"hits": 0, "misses": 3, "false_alarms": 0, "correct_negatives": 1}
    assert np.isnan(far(table))


def test_continuous_metrics():
    forecast = np.array([1.0, 2.0, 3.0])
    obs = np.array([1.0, 4.0, 3.0])

    assert mae(forecast, obs) == 2.0 / 3.0
    assert rmse(forecast, obs) == np.sqrt((0 + 4 + 0) / 3.0)
    assert mean_error(forecast, obs) == -2.0 / 3.0


def test_compute_all_metrics_shape():
    rng = np.random.default_rng(0)
    forecast = rng.random((10, 10)) * 10
    obs = rng.random((10, 10)) * 10

    result = compute_all_metrics(forecast, obs, thresholds=[1, 5])

    assert set(result["categorical"].keys()) == {1, 5}
    assert set(result["categorical"][1].keys()) == {"pod", "far", "csi", "bias_score", "ets"}
    assert set(result["continuous"].keys()) == {"mae", "rmse", "mean_error"}

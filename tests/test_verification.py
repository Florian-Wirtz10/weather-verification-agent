import numpy as np
import pytest

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

    assert table == {
        "hits": 2,
        "misses": 0,
        "false_alarms": 0,
        "correct_negatives": 2,
        "valid_count": 4,
        "invalid_count": 0,
    }
    assert pod(table) == 1.0
    assert far(table) == 0.0
    assert csi(table) == 1.0
    assert bias_score(table) == 1.0
    assert ets(table) == 1.0


def test_contingency_table_all_misses():
    obs = np.array([[5.0, 5.0], [5.0, 5.0]])
    forecast = np.array([[0.0, 0.0], [0.0, 0.0]])

    table = contingency_table(forecast, obs, threshold=1.0)

    assert table == {
        "hits": 0,
        "misses": 4,
        "false_alarms": 0,
        "correct_negatives": 0,
        "valid_count": 4,
        "invalid_count": 0,
    }
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


def test_metrics_exclude_cells_not_finite_in_both_fields():
    forecast = np.array([5.0, 0.0, np.nan, 5.0, np.inf, 100.0])
    obs = np.array([5.0, 5.0, 0.0, np.nan, 0.0, 98.0])

    table = contingency_table(forecast, obs, threshold=1.0)

    assert table == {
        "hits": 2,
        "misses": 1,
        "false_alarms": 0,
        "correct_negatives": 0,
        "valid_count": 3,
        "invalid_count": 3,
    }
    assert mae(forecast, obs) == pytest.approx(7.0 / 3.0)
    assert rmse(forecast, obs) == pytest.approx(np.sqrt(29.0 / 3.0))
    assert mean_error(forecast, obs) == pytest.approx(-1.0)


def test_continuous_metrics_return_nan_without_common_finite_cells():
    forecast = np.array([np.nan, np.inf])
    obs = np.array([1.0, 2.0])

    assert np.isnan(mae(forecast, obs))
    assert np.isnan(rmse(forecast, obs))
    assert np.isnan(mean_error(forecast, obs))


@pytest.mark.parametrize("metric", [contingency_table, mae, rmse, mean_error])
def test_metrics_reject_shape_mismatch(metric):
    forecast = np.zeros((2, 2))
    obs = np.zeros(4)

    if metric is contingency_table:
        with pytest.raises(ValueError, match="same shape"):
            metric(forecast, obs, threshold=1.0)
    else:
        with pytest.raises(ValueError, match="same shape"):
            metric(forecast, obs)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_contingency_table_rejects_nonfinite_threshold(threshold):
    with pytest.raises(ValueError, match="finite scalar"):
        contingency_table(np.zeros(1), np.zeros(1), threshold)


def test_compute_all_metrics_shape():
    rng = np.random.default_rng(0)
    forecast = rng.random((10, 10)) * 10
    obs = rng.random((10, 10)) * 10

    result = compute_all_metrics(forecast, obs, thresholds=[1, 5])

    assert set(result["categorical"].keys()) == {1, 5}
    assert set(result["categorical"][1].keys()) == {"pod", "far", "csi", "bias_score", "ets"}
    assert set(result["continuous"].keys()) == {"mae", "rmse", "mean_error"}

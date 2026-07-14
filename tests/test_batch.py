from tests.test_catalog import _write_forecast
from tests.test_radar_binary import _write_binary_cref
from verification.batch import run_daily_fenglei_verification


def test_daily_batch_streams_aggregates_and_reuses_metric_cache(tmp_path):
    forecast_dir = tmp_path / "forecast"
    obs_dir = tmp_path / "obs"
    forecast_dir.mkdir()
    obs_dir.mkdir()
    _write_forecast(forecast_dir)
    _write_binary_cref(obs_dir)
    kwargs = {
        "forecast_dir": forecast_dir,
        "observation_dir": obs_dir,
        "case_date_bjt": "2026-04-07",
        "bbox": (28.0, 30.0, 116.0, 120.0),
        "thresholds": (5, 15),
        "catalog_cache_dir": tmp_path / "catalog-cache",
        "metric_cache_path": tmp_path / "metrics.sqlite3",
    }

    first = run_daily_fenglei_verification(**kwargs)

    assert first["coverage"]["selected_pairs"] == 1
    assert first["cache"]["metric_pairs_computed"] == 1
    assert first["cache"]["metric_pairs_reused"] == 0
    assert first["cache"]["complete"] is True
    assert len(first["per_case"]) == 2
    assert len(first["summary"]) == 2
    assert set(first["summary"]["case_count"]) == {1}
    assert set(first["summary"]["threshold_dbz"]) == {5.0, 15.0}
    assert first["summary"]["valid_count"].min() > 0

    second = run_daily_fenglei_verification(**kwargs)

    assert second["cache"]["metric_pairs_computed"] == 0
    assert second["cache"]["metric_pairs_reused"] == 1
    assert second["catalog"].cache_hit is True
    assert second["per_case"].equals(first["per_case"])


def test_daily_batch_cache_key_changes_with_scoring_contract(tmp_path):
    forecast_dir = tmp_path / "forecast"
    obs_dir = tmp_path / "obs"
    forecast_dir.mkdir()
    obs_dir.mkdir()
    _write_forecast(forecast_dir)
    _write_binary_cref(obs_dir)
    common = {
        "forecast_dir": forecast_dir,
        "observation_dir": obs_dir,
        "case_date_bjt": "2026-04-07",
        "bbox": (28.0, 30.0, 116.0, 120.0),
        "catalog_cache_dir": tmp_path / "catalog-cache",
        "metric_cache_path": tmp_path / "metrics.sqlite3",
    }

    five = run_daily_fenglei_verification(thresholds=(5,), **common)
    fifteen = run_daily_fenglei_verification(thresholds=(15,), **common)

    assert five["cache"]["run_key"] != fifteen["cache"]["run_key"]
    assert five["cache"]["metric_pairs_computed"] == 1
    assert fifteen["cache"]["metric_pairs_computed"] == 1

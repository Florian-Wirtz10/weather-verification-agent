from datetime import datetime

from tests.test_catalog import _write_forecast
from tests.test_radar_binary import _write_binary_cref
from verification.batch import (
    filter_daily_result_by_initializations,
    run_daily_fenglei_verification,
)


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
        "case_date": "2026-04-07",
        "bbox": (28.0, 30.0, 116.0, 120.0),
        "thresholds": (5, 15),
        "forecast_case_time_basis": "bjt",
        "observation_case_time_basis": "utc",
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
        "case_date": "2026-04-07",
        "bbox": (28.0, 30.0, 116.0, 120.0),
        "forecast_case_time_basis": "bjt",
        "observation_case_time_basis": "utc",
        "catalog_cache_dir": tmp_path / "catalog-cache",
        "metric_cache_path": tmp_path / "metrics.sqlite3",
    }

    five = run_daily_fenglei_verification(thresholds=(5,), **common)
    fifteen = run_daily_fenglei_verification(thresholds=(15,), **common)

    assert five["cache"]["run_key"] != fifteen["cache"]["run_key"]
    assert five["cache"]["metric_pairs_computed"] == 1
    assert fifteen["cache"]["metric_pairs_computed"] == 1


def test_daily_result_can_be_reaggregated_for_selected_initializations(tmp_path):
    forecast_dir = tmp_path / "forecast"
    obs_dir = tmp_path / "obs"
    forecast_dir.mkdir()
    obs_dir.mkdir()
    for init_bjt, valid, generated in (
        ("202604071848", "20260407_105400", "20260407110012"),
        ("202604071854", "20260407_110000", "20260407110612"),
        ("202604071900", "20260407_110600", "20260407111212"),
    ):
        _write_forecast(forecast_dir, init_bjt=init_bjt)
        _write_binary_cref(obs_dir, valid=valid, generated=generated)

    result = run_daily_fenglei_verification(
        forecast_dir=forecast_dir,
        observation_dir=obs_dir,
        case_date="2026-04-07",
        bbox=(28.0, 30.0, 116.0, 120.0),
        thresholds=(15,),
        forecast_case_time_basis="bjt",
        observation_case_time_basis="utc",
        catalog_cache_dir=tmp_path / "catalog-cache",
        metric_cache_path=tmp_path / "metrics.sqlite3",
    )
    selected = filter_daily_result_by_initializations(
        result,
        [datetime(2026, 4, 7, 10, 48), datetime(2026, 4, 7, 11, 0)],
    )

    assert len(result["per_case"]) == 3
    assert len(selected["per_case"]) == 2
    assert selected["summary"].iloc[0]["case_count"] == 2
    assert selected["coverage"]["selected_initialization_count"] == 2
    assert selected["coverage"]["selected_pairs"] == 2
    assert selected["context"]["selection_scope"] == "selected_initializations"

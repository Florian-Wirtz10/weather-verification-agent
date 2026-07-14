from datetime import date, datetime

import numpy as np
import pytest

from data_io.catalog import build_daily_catalog
from tests.test_radar_binary import _write_binary_cref
from tests.test_real_data_io import _make_swan_mosaic_bytes


def _write_forecast(root, *, init_bjt="202604071848", lead=6):
    init_utc_hour = int(init_bjt[8:10]) - 8
    blob = _make_swan_mosaic_bytes(
        nx=4,
        ny=3,
        cell_bytes=1,
        scale=0.5,
        offset=-33.0,
        values=np.zeros((3, 4), dtype="u1"),
        header_overrides={
            "year": 2026,
            "month": 4,
            "day": 7,
            "hour": init_utc_hour,
            "minute": int(init_bjt[10:12]),
            "interval": lead,
            "StartLon": 116.0,
            "StartLat": 30.0,
            "XReso": 4.0 / 3.0,
            "YReso": 1.0,
        },
    )
    path = root / (
        f"Z_SEVP_C_BABJ_20260407190510_P_FENGL-V11-REFMOSAIC-"
        f"{init_bjt}-{lead:03d}MIN.bz2"
    )
    path.write_bytes(blob)
    return path


def test_daily_catalog_matches_times_and_chooses_latest_correction(tmp_path):
    forecast_dir = tmp_path / "forecast"
    obs_dir = tmp_path / "obs"
    cache_dir = tmp_path / "cache"
    forecast_dir.mkdir()
    obs_dir.mkdir()
    forecast = _write_forecast(forecast_dir)
    older = _write_binary_cref(
        obs_dir,
        valid="20260407_185400",
        generated="20260407190012",
    )
    newer = _write_binary_cref(
        obs_dir,
        valid="20260407_185400",
        generated="20260407190512",
    )

    catalog = build_daily_catalog(
        forecast_dir,
        obs_dir,
        date(2026, 4, 7),
        cache_dir=cache_dir,
    )

    assert catalog.cache_hit is False
    assert len(catalog.forecasts) == 1
    assert catalog.forecasts[0].path == forecast
    assert catalog.forecasts[0].init_time_utc == datetime(2026, 4, 7, 10, 48)
    assert catalog.forecasts[0].valid_time_utc == datetime(2026, 4, 7, 10, 54)
    assert catalog.observations[0].path == newer
    assert catalog.matched_forecasts == catalog.forecasts
    assert catalog.duplicates[0].selected == newer
    assert catalog.duplicates[0].discarded == (older,)

    cached = build_daily_catalog(
        forecast_dir,
        obs_dir,
        "2026-04-07",
        cache_dir=cache_dir,
    )
    assert cached.cache_hit is True


def test_daily_catalog_duplicate_error_policy(tmp_path):
    forecast_dir = tmp_path / "forecast"
    obs_dir = tmp_path / "obs"
    forecast_dir.mkdir()
    obs_dir.mkdir()
    _write_forecast(forecast_dir)
    _write_binary_cref(obs_dir, generated="20260407190012")
    _write_binary_cref(obs_dir, generated="20260407190512")

    with pytest.raises(ValueError, match="duplicate observations"):
        build_daily_catalog(
            forecast_dir,
            obs_dir,
            "2026-04-07",
            duplicate_policy="error",
            cache_dir=tmp_path / "cache",
        )

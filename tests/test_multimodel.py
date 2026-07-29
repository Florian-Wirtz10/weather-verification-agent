from datetime import datetime, timedelta
import sqlite3

import numpy as np

from data_io.forecast_sources import (
    MODEL_FENGLEI,
    MODEL_STORMDIT,
    ForecastRecord,
    MultiModelCatalog,
    discover_multimodel_initializations,
)
from data_io.grid import GriddedField
from verification import multimodel


def _field(data, valid, *, product, init=None, lead=None):
    return GriddedField(
        data=np.asarray(data, dtype=np.float32),
        lat=np.array([29.0, 29.03]),
        lon=np.array([110.0, 110.03]),
        valid_time=valid,
        product=product,
        units="dBZ",
        init_time=init,
        lead_minutes=lead,
    )


def _catalog(tmp_path):
    init = datetime(2026, 4, 8, 0)
    stormdit = tmp_path / "stormdit.npz"
    fenglei_6 = tmp_path / "fenglei_006.bz2"
    obs_6 = tmp_path / "obs_006.nc"
    obs_12 = tmp_path / "obs_012.nc"
    for path in (stormdit, fenglei_6, obs_6, obs_12):
        path.write_bytes(b"x")
    forecasts = {
        MODEL_STORMDIT: {
            (init, lead): ForecastRecord(
                MODEL_STORMDIT,
                stormdit,
                init,
                init + timedelta(hours=8),
                lead,
                init + timedelta(minutes=lead),
            )
            for lead in (6, 12)
        },
        MODEL_FENGLEI: {
            (init, 6): ForecastRecord(
                MODEL_FENGLEI,
                fenglei_6,
                init,
                init + timedelta(hours=8),
                6,
                init + timedelta(minutes=6),
            )
        },
    }
    return MultiModelCatalog(
        observations={
            init + timedelta(minutes=6): obs_6,
            init + timedelta(minutes=12): obs_12,
        },
        forecasts=forecasts,
        stormdit_contract="phase8-inferred-v1",
    ), init


def test_dual_model_catalog_uses_strict_common_leads(tmp_path):
    catalog, init = _catalog(tmp_path)
    cases = discover_multimodel_initializations(
        catalog,
        (MODEL_STORMDIT, MODEL_FENGLEI),
        lead_range=(6, 12),
        common_only=True,
    )

    assert cases[0]["init_time"] == init
    assert cases[0]["available_leads"] == [6]
    assert cases[0]["missing_leads"] == [12]
    assert cases[0]["complete"] is False


def test_case_scoring_and_persistent_cache_share_identical_samples(
    tmp_path, monkeypatch
):
    catalog, init = _catalog(tmp_path)
    observation = _field(
        [[20, 0], [20, 0]], init + timedelta(minutes=6), product="CREF"
    )
    stormdit = _field(
        [[20, 0], [0, 0]],
        init + timedelta(minutes=6),
        product="StormDiT",
        init=init,
        lead=6,
    )
    fenglei = _field(
        [[20, 0], [20, 0]],
        init + timedelta(minutes=6),
        product="Fenglei",
        init=init,
        lead=6,
    )
    monkeypatch.setattr(
        multimodel,
        "read_radar_observation",
        lambda *_args, **_kwargs: observation,
    )
    monkeypatch.setattr(
        multimodel,
        "read_upstream_model_forecasts",
        lambda *_args, **_kwargs: {6: stormdit},
    )
    monkeypatch.setattr(
        multimodel, "read_swan_mosaic", lambda *_args, **_kwargs: fenglei
    )

    result = multimodel.run_multimodel_case(
        catalog,
        (MODEL_STORMDIT, MODEL_FENGLEI),
        init,
        (29, 30, 110, 111),
        thresholds=(15,),
        lead_range=(6, 6),
    )
    assert result["coverage"]["selected_leads"] == [6]
    assert result["per_lead"][0]["models"][MODEL_STORMDIT][15.0]["csi"] == 0.5
    assert result["per_lead"][0]["models"][MODEL_FENGLEI][15.0]["csi"] == 1.0
    stormdit_score = result["per_lead"][0]["models"][MODEL_STORMDIT][15.0]
    assert stormdit_score["reflectivity_mae_dbz"] == 5.0
    assert stormdit_score["forecast_mean_dbz"] == 5.0
    assert 1.6 < stormdit_score["centroid_distance_km"] < 1.8

    cache = tmp_path / "scores.sqlite3"
    first = multimodel.run_multimodel_batch(
        catalog,
        (MODEL_STORMDIT, MODEL_FENGLEI),
        (29, 30, 110, 111),
        thresholds=(15,),
        lead_range=(6, 6),
        cache_path=cache,
    )
    second = multimodel.run_multimodel_batch(
        catalog,
        (MODEL_STORMDIT, MODEL_FENGLEI),
        (29, 30, 110, 111),
        thresholds=(15,),
        lead_range=(6, 6),
        cache_path=cache,
    )
    assert first["cache"]["computed_initializations"] == 1
    assert second["cache"]["computed_initializations"] == 0
    assert len(second["per_case"]) == 2
    assert {
        "reflectivity_mae_dbz",
        "mean_reflectivity_error_dbz",
        "centroid_distance_km",
    } <= set(second["per_case"].columns)


def test_phase8_cache_adds_physical_columns_to_legacy_database(tmp_path):
    cache = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(cache)
    connection.execute(
        """
        CREATE TABLE multimodel_scores (
            run_key TEXT NOT NULL,
            model_id TEXT NOT NULL,
            init_time_utc TEXT NOT NULL,
            init_time_bjt TEXT NOT NULL,
            valid_time_utc TEXT NOT NULL,
            lead_minutes INTEGER NOT NULL,
            threshold_dbz REAL NOT NULL,
            hits INTEGER NOT NULL,
            misses INTEGER NOT NULL,
            false_alarms INTEGER NOT NULL,
            correct_negatives INTEGER NOT NULL,
            valid_count INTEGER NOT NULL,
            invalid_count INTEGER NOT NULL,
            pod REAL,
            far REAL,
            csi REAL,
            bias_score REAL,
            ets REAL,
            PRIMARY KEY (
                run_key, model_id, init_time_utc, lead_minutes, threshold_dbz
            )
        )
        """
    )
    connection.commit()
    connection.close()

    migrated = multimodel._connect(cache)
    try:
        columns = {
            row[1] for row in migrated.execute("PRAGMA table_info(multimodel_scores)")
        }
    finally:
        migrated.close()

    assert {
        "forecast_mean_dbz",
        "reflectivity_mae_dbz",
        "centroid_distance_km",
    } <= columns

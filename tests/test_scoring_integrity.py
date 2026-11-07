"""Regression cases with known scores, independent of the scoring implementation."""

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from data_io.forecast_sources import ForecastRecord, MultiModelCatalog, MODEL_FENGLEI
from data_io.grid import GriddedField
from verification import multimodel, triplet


INIT = datetime(2025, 4, 11, 2)
VALID = INIT + timedelta(minutes=12)
BBOX = (29.0, 30.0, 110.0, 111.0)
LAT = np.array([29.0, 29.03])
LON = np.array([110.0, 110.03])


def field(values, *, init=INIT, lead=12, valid=VALID):
    return GriddedField(
        np.ma.asarray(values, dtype=float), LAT, LON, valid, "test", "dBZ", init, lead
    )


def catalog(observation_path, pairs=((INIT, 12),)):
    return MultiModelCatalog(
        {VALID: observation_path},
        {MODEL_FENGLEI: {
            (init, lead): ForecastRecord(
                MODEL_FENGLEI, Path(str(lead)), init, init + timedelta(hours=8), lead, VALID
            ) for init, lead in pairs
        }},
        "test-contract",
    )


def run_case(cat):
    return multimodel.run_multimodel_case(
        cat, [MODEL_FENGLEI], INIT, BBOX, thresholds=(15,), lead_range=(12, 12)
    )["per_lead"][0]["models"][MODEL_FENGLEI][15.0]


def test_netcdf_masked_fill_is_excluded_from_counts_and_error(tmp_path, monkeypatch):
    path = tmp_path / "masked.nc"
    with netCDF4.Dataset(path, "w") as ds:
        for dim, size in (("time", 1), ("lat", 2), ("lon", 2)):
            ds.createDimension(dim, size)
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "minutes since 2025-04-11 02:00:00"
        time[:] = [12]
        ds.createVariable("lat", "f8", ("lat",))[:] = LAT
        ds.createVariable("lon", "f8", ("lon",))[:] = LON
        ds.createVariable("CREF", "f4", ("time", "lat", "lon"), fill_value=-9999)[:] = [
            [[20, -9999], [20, 0]]
        ]
    monkeypatch.setattr(multimodel, "read_swan_mosaic", lambda *a, **kw: field(
        [[20, 20], [20, 0]]
    ))
    score = run_case(catalog(path))
    assert score["contingency"]["valid_count"] == 3
    assert score["contingency"]["invalid_count"] == 1
    assert score["csi"] == 1
    assert score["reflectivity_mae_dbz"] == 0


@pytest.mark.parametrize("all_missing", [False, True])
def test_masked_forecast_values_never_enter_scores(monkeypatch, all_missing):
    values = np.ma.array([[20, -9999], [20, 0]], mask=True if all_missing else [
        [False, True], [False, False]
    ])
    monkeypatch.setattr(multimodel, "read_radar_observation", lambda *a, **kw: field(
        [[20, 20], [20, 0]]
    ))
    monkeypatch.setattr(multimodel, "read_swan_mosaic", lambda *a, **kw: field(values))
    score = run_case(catalog(Path("unused")))
    assert score["contingency"]["valid_count"] == (0 if all_missing else 3)
    if all_missing:
        assert np.isnan(score["csi"])
        assert np.isnan(score["reflectivity_mae_dbz"])
    else:
        assert score["csi"] == 1
        assert score["reflectivity_mae_dbz"] == 0


def test_same_valid_time_scores_use_common_cells_across_initializations(monkeypatch):
    second = INIT + timedelta(minutes=6)
    obs = field([[20, 20], [0, 0]])
    first_fcst = field([[20, np.nan], [0, 0]])
    second_fcst = field([[20, 0], [0, 0]], init=second, lead=6)
    monkeypatch.setattr(multimodel, "read_radar_observation", lambda *a, **kw: obs)
    monkeypatch.setattr(multimodel, "read_swan_mosaic", lambda p, **kw:
                        first_fcst if str(p) == "12" else second_fcst)
    result = multimodel.run_multimodel_same_valid_time(
        catalog(Path("unused"), ((INIT, 12), (second, 6))), [MODEL_FENGLEI],
        [INIT, second], VALID, BBOX, thresholds=(15,),
    )
    assert result["per_case"]["valid_count"].tolist() == [3, 3]
    assert result["per_case"]["csi"].tolist() == [1, 1]
    assert [item["scores"][MODEL_FENGLEI][15.0]["csi"] for item in result["items"]] == [1, 1]


@pytest.mark.parametrize("mismatch", ["init", "lead", "observation_valid"])
def test_case_rejects_inconsistent_time_identity(monkeypatch, mismatch):
    forecast = field([[20, 0], [0, 0]])
    observation = field([[20, 0], [0, 0]])
    if mismatch == "init":
        forecast.init_time += timedelta(minutes=6)
    elif mismatch == "lead":
        forecast.lead_minutes = 6
    else:
        observation.valid_time += timedelta(minutes=6)
    monkeypatch.setattr(multimodel, "read_radar_observation", lambda *a, **kw: observation)
    monkeypatch.setattr(multimodel, "read_swan_mosaic", lambda *a, **kw: forecast)
    with pytest.raises(ValueError, match="time mismatch"):
        run_case(catalog(Path("unused")))


def setup_triplet(monkeypatch, *, wrong_header=False):
    metadata = SimpleNamespace(init_time_utc=INIT, init_time_bjt=INIT + timedelta(hours=8),
                               available_leads=(12,))
    monkeypatch.setattr(triplet, "read_upstream_model_metadata", lambda *a, **kw: metadata)
    monkeypatch.setattr(triplet, "_index_fenglei_init", lambda *a, **kw: {12: Path("f")})
    monkeypatch.setattr(triplet, "index_radar_observation_files", lambda *a, **kw: {
        VALID: Path("o")
    })
    monkeypatch.setattr(triplet, "read_upstream_model_forecasts", lambda *a, **kw: {
        12: field([[20, np.nan], [0, 0]])
    })
    forecast = field([[20, 0], [0, 0]])
    if wrong_header:
        forecast.init_time += timedelta(minutes=6)
        forecast.lead_minutes = 6
    monkeypatch.setattr(triplet, "read_swan_mosaic", lambda *a, **kw: forecast)
    monkeypatch.setattr(triplet, "read_radar_observation", lambda *a, **kw: field(
        [[20, 20], [0, 0]]
    ))


def test_triplet_uses_common_support_for_both_models(monkeypatch):
    setup_triplet(monkeypatch)
    row = triplet.run_triplet_case("m", "f", "o", BBOX, thresholds=(15,))["per_lead"][0]
    assert row["model"][15.0]["contingency"]["valid_count"] == 3
    assert row["fenglei"][15.0]["contingency"]["valid_count"] == 3
    assert row["model"][15.0]["csi"] == row["fenglei"][15.0]["csi"] == 1


def test_triplet_rejects_same_valid_time_with_wrong_init_and_lead(monkeypatch):
    setup_triplet(monkeypatch, wrong_header=True)
    with pytest.raises(ValueError, match="time mismatch"):
        triplet.run_triplet_case("m", "f", "o", BBOX, thresholds=(15,))

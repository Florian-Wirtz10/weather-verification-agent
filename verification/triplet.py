"""Single-initialization comparison of upstream model, Fenglei, and CREF."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re

from data_io.fenglei_mosaic import read_swan_mosaic
from data_io.grid import regrid_nearest
from data_io.radar_obs import index_radar_observation_files, read_radar_observation
from data_io.upstream_model import (
    PHASE8_INFERRED_CONTRACT,
    UpstreamModelContract,
    read_upstream_model_forecasts,
    read_upstream_model_metadata,
)
from verification.metrics import CATEGORICAL_SCORES, common_valid_arrays, contingency_table

_FENGLEI_RE = re.compile(
    r"(?:FENGL(?:-V11)?-)?REFMOSAIC-(?P<init>\d{12})-(?P<lead>\d{3})MIN\.bz2\Z"
)


def _index_fenglei_init(directory, init_time_utc: datetime) -> dict[int, Path]:
    selected: dict[int, Path] = {}
    for path in Path(directory).glob("*REFMOSAIC*MIN.bz2"):
        match = _FENGLEI_RE.search(path.name)
        if match is None:
            continue
        filename_init = datetime.strptime(match.group("init"), "%Y%m%d%H%M")
        # V11 filenames are BJT; the older delivery uses UTC.
        if "-V11-" in path.name:
            filename_init -= timedelta(hours=8)
        if filename_init != init_time_utc:
            continue
        lead = int(match.group("lead"))
        if lead in selected:
            raise ValueError(
                f"duplicate Fenglei lead {lead} for {init_time_utc} UTC: "
                f"{selected[lead]} and {path}"
            )
        selected[lead] = path
    return selected


def _score(field, observation, thresholds):
    forecast_on_obs = regrid_nearest(field, observation.lat, observation.lon)
    return _score_arrays(forecast_on_obs, observation.data, thresholds)


def _score_arrays(forecast, observation, thresholds):
    result = {}
    for threshold in thresholds:
        table = contingency_table(forecast, observation, threshold)
        result[float(threshold)] = {
            "contingency": table,
            **{name: score_fn(table) for name, score_fn in CATEGORICAL_SCORES.items()},
        }
    return result


def run_triplet_case(
    model_path,
    fenglei_dir,
    observation_dir,
    bbox,
    *,
    thresholds=(5, 15, 25, 35),
    lead_range=(6, 180),
    contract: UpstreamModelContract = PHASE8_INFERRED_CONTRACT,
) -> dict:
    """Score one exact initialization under one shared verification definition."""
    metadata = read_upstream_model_metadata(model_path, contract=contract)
    fenglei_by_lead = _index_fenglei_init(fenglei_dir, metadata.init_time_utc)
    observation_by_time = index_radar_observation_files(observation_dir)

    start_lead, end_lead = map(int, lead_range)
    requested = [
        lead
        for lead in metadata.available_leads
        if start_lead <= lead <= end_lead
    ]
    selected = [
        lead
        for lead in requested
        if lead in fenglei_by_lead
        and metadata.init_time_utc + timedelta(minutes=lead) in observation_by_time
    ]
    if not selected:
        raise ValueError("no exact model/Fenglei/CREF triplet is available")

    lat_min, lat_max, lon_min, lon_max = map(float, bbox)
    model_by_lead = read_upstream_model_forecasts(
        model_path, selected, contract=contract
    )
    per_lead = []
    for lead in selected:
        model = model_by_lead[lead]
        fenglei = read_swan_mosaic(
            fenglei_by_lead[lead],
            bbox=(lat_min - 0.02, lat_max + 0.02, lon_min - 0.02, lon_max + 0.02),
        )
        valid_time = metadata.init_time_utc + timedelta(minutes=lead)
        observation = read_radar_observation(
            observation_by_time[valid_time],
            bbox=(lat_min, lat_max, lon_min, lon_max),
        )
        if observation.valid_time != valid_time or any(
            forecast.valid_time != valid_time
            or forecast.init_time != metadata.init_time_utc
            or forecast.lead_minutes != lead
            for forecast in (model, fenglei)
        ):
            raise ValueError(f"triplet header time mismatch at lead {lead}")
        obs_data, model_data, fenglei_data = common_valid_arrays(
            observation.data,
            regrid_nearest(model, observation.lat, observation.lon),
            regrid_nearest(fenglei, observation.lat, observation.lon),
        )
        per_lead.append(
            {
                "lead_minutes": lead,
                "valid_time_utc": valid_time,
                "model": _score_arrays(model_data, obs_data, thresholds),
                "fenglei": _score_arrays(fenglei_data, obs_data, thresholds),
            }
        )

    return {
        "context": {
            "model_path": str(Path(model_path)),
            "model_contract": contract.name,
            "value_zero_missing_semantics": "unconfirmed",
            "init_time_utc": metadata.init_time_utc,
            "init_time_bjt": metadata.init_time_bjt,
            "bbox": [lat_min, lat_max, lon_min, lon_max],
            "thresholds_dbz": [float(value) for value in thresholds],
            "temporal_matching": "exact initialization and exact valid time",
            "spatial_alignment": "nearest forecast to the same CREF observation grid",
        },
        "coverage": {
            "requested_leads": requested,
            "selected_leads": selected,
            "missing_leads": [lead for lead in requested if lead not in selected],
        },
        "per_lead": per_lead,
    }

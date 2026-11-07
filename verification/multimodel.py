"""Unified Phase 8 verification for StormDiT, Fenglei, and CREF."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Callable

import numpy as np
import pandas as pd

from data_io.benchmark import read_benchmark_stormdit_forecasts
from data_io.fenglei_mosaic import read_swan_mosaic
from data_io.forecast_sources import (
    MODEL_FENGLEI,
    MODEL_LABELS,
    MODEL_STORMDIT,
    MultiModelCatalog,
    discover_multimodel_initializations,
)
from data_io.grid import GriddedField, regrid_nearest
from data_io.radar_obs import read_radar_observation
from data_io.upstream_model import read_upstream_model_forecasts
from verification.metrics import CATEGORICAL_SCORES, common_valid_arrays, contingency_table

MULTIMODEL_CACHE_VERSION = 5
_COUNT_COLUMNS = (
    "hits",
    "misses",
    "false_alarms",
    "correct_negatives",
    "valid_count",
    "invalid_count",
)
_SCORE_COLUMNS = tuple(CATEGORICAL_SCORES)
_DIAGNOSTIC_COLUMNS = (
    "forecast_mean_dbz",
    "observation_mean_dbz",
    "forecast_max_dbz",
    "observation_max_dbz",
    "mean_reflectivity_error_dbz",
    "reflectivity_mae_dbz",
    "centroid_distance_km",
)


def _echo_centroid(data, valid, threshold, lat, lon):
    echo = valid & (data >= threshold)
    if not np.any(echo):
        return None
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return float(np.mean(lat_grid[echo])), float(np.mean(lon_grid[echo]))


def _centroid_distance_km(left, right):
    if left is None or right is None:
        return float("nan")
    mean_lat = np.deg2rad((left[0] + right[0]) / 2)
    north_south = (left[0] - right[0]) * 111.32
    east_west = (left[1] - right[1]) * 111.32 * np.cos(mean_lat)
    return float(np.hypot(north_south, east_west))


def _score_array(forecast_data, observation_data, thresholds, *, lat, lon):
    """Score categorical events and cache auditable physical diagnostics.

    Negative finite reflectivities are normalized to the common 0 dBZ no-echo
    floor only for the new continuous diagnostics. Existing categorical scores
    keep their original threshold comparison semantics.
    """
    forecast, observation = common_valid_arrays(forecast_data, observation_data)
    valid = np.isfinite(forecast) & np.isfinite(observation)
    normalized_forecast = np.maximum(forecast, 0.0)
    normalized_observation = np.maximum(observation, 0.0)
    if np.any(valid):
        difference = normalized_forecast[valid] - normalized_observation[valid]
        common = {
            "forecast_mean_dbz": float(np.mean(normalized_forecast[valid])),
            "observation_mean_dbz": float(np.mean(normalized_observation[valid])),
            "forecast_max_dbz": float(np.max(normalized_forecast[valid])),
            "observation_max_dbz": float(np.max(normalized_observation[valid])),
            "mean_reflectivity_error_dbz": float(np.mean(difference)),
            "reflectivity_mae_dbz": float(np.mean(np.abs(difference))),
        }
    else:
        common = {
            name: float("nan")
            for name in _DIAGNOSTIC_COLUMNS
            if name != "centroid_distance_km"
        }
    result = {}
    for threshold in thresholds:
        table = contingency_table(forecast, observation, threshold)
        forecast_centroid = _echo_centroid(
            normalized_forecast, valid, threshold, lat, lon
        )
        observation_centroid = _echo_centroid(
            normalized_observation, valid, threshold, lat, lon
        )
        result[float(threshold)] = {
            "contingency": table,
            **{name: score_fn(table) for name, score_fn in CATEGORICAL_SCORES.items()},
            **common,
            "centroid_distance_km": _centroid_distance_km(
                forecast_centroid, observation_centroid
            ),
        }
    return result


def _field_on_observation_grid(field, observation) -> GriddedField:
    return GriddedField(
        data=regrid_nearest(field, observation.lat, observation.lon),
        lat=observation.lat,
        lon=observation.lon,
        valid_time=observation.valid_time,
        product=field.product,
        units=field.units,
        init_time=field.init_time,
        lead_minutes=field.lead_minutes,
    )


def _common_spatial_bbox(bbox, records) -> tuple[float, float, float, float]:
    """Intersect the requested area with every finite forecast-source domain."""
    lat_min, lat_max, lon_min, lon_max = map(float, bbox)
    for record in records:
        if record.domain is None:
            continue
        source_lat_min, source_lat_max, source_lon_min, source_lon_max = record.domain
        lat_min = max(lat_min, source_lat_min)
        lat_max = min(lat_max, source_lat_max)
        lon_min = max(lon_min, source_lon_min)
        lon_max = min(lon_max, source_lon_max)
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError("所选检验区域与预报资料的公共空间范围不相交")
    return lat_min, lat_max, lon_min, lon_max


def _read_stormdit_fields(record, leads):
    if record.source_kind == "benchmark_stormdit":
        return read_benchmark_stormdit_forecasts(record.path, leads)
    return read_upstream_model_forecasts(record.path, leads)


def run_multimodel_case(
    catalog: MultiModelCatalog,
    model_ids,
    init_time_utc: datetime,
    bbox,
    *,
    thresholds=(5, 15, 25, 35),
    lead_range=(6, 180),
    frame_leads=(),
) -> dict:
    """Score one initialization, using common samples when multiple models are selected."""
    model_ids = tuple(dict.fromkeys(model_ids))
    if not model_ids or any(model_id not in MODEL_LABELS for model_id in model_ids):
        raise ValueError(f"unsupported model selection: {model_ids!r}")
    common_only = len(model_ids) > 1
    start_lead, end_lead = map(int, lead_range)
    leads = [
        lead
        for lead in catalog.available_leads(
            model_ids, init_time_utc, common_only=common_only
        )
        if start_lead <= lead <= end_lead
    ]
    if not leads:
        raise ValueError("selected initialization has no exact forecast/observation matches")

    requested_bbox = tuple(map(float, bbox))
    representative_records = [
        catalog.forecasts[model_id][(init_time_utc, leads[0])]
        for model_id in model_ids
    ]
    lat_min, lat_max, lon_min, lon_max = _common_spatial_bbox(
        requested_bbox, representative_records
    )

    stormdit_fields = {}
    if MODEL_STORMDIT in model_ids:
        record = catalog.forecasts[MODEL_STORMDIT][(init_time_utc, leads[0])]
        stormdit_fields = _read_stormdit_fields(record, leads)

    requested_frames = set(frame_leads)
    per_lead = []
    frames = {}
    for lead in leads:
        valid_time = init_time_utc + timedelta(minutes=lead)
        observation = read_radar_observation(
            catalog.observations[valid_time],
            bbox=(lat_min, lat_max, lon_min, lon_max),
        )
        if observation.valid_time != valid_time:
            raise ValueError(f"CREF header time mismatch at +{lead} minutes")
        aligned_fields = {}
        for model_id in model_ids:
            if model_id == MODEL_STORMDIT:
                field = stormdit_fields[lead]
            else:
                record = catalog.forecasts[MODEL_FENGLEI][(init_time_utc, lead)]
                field = read_swan_mosaic(
                    record.path,
                    bbox=(
                        lat_min - 0.02,
                        lat_max + 0.02,
                        lon_min - 0.02,
                        lon_max + 0.02,
                    ),
                )
            if (
                field.valid_time != valid_time
                or field.init_time != init_time_utc
                or field.lead_minutes != lead
            ):
                raise ValueError(
                    f"{MODEL_LABELS[model_id]} header time mismatch at +{lead} minutes"
                )
            aligned = _field_on_observation_grid(field, observation)
            aligned_fields[model_id] = aligned
        observation_for_score, *forecast_arrays = common_valid_arrays(
            observation.data, *(field.data for field in aligned_fields.values())
        )
        scores = {}
        for model_id, forecast_array in zip(aligned_fields, forecast_arrays, strict=True):
            scores[model_id] = _score_array(
                forecast_array,
                observation_for_score,
                thresholds,
                lat=observation.lat,
                lon=observation.lon,
            )
        per_lead.append(
            {
                "lead_minutes": lead,
                "valid_time_utc": valid_time,
                "models": scores,
            }
        )
        if lead in requested_frames:
            frames[lead] = {
                "lead_minutes": lead,
                "observation": observation,
                "forecasts": aligned_fields,
            }

    expected = list(range(max(6, start_lead), end_lead + 1, 6))
    return {
        "context": {
            "model_ids": list(model_ids),
            "model_labels": [MODEL_LABELS[value] for value in model_ids],
            "model_contract": catalog.stormdit_contract,
            "value_zero_missing_semantics": (
                "StormDiT 0 dBZ is treated as no echo under phase8-inferred-v1"
            ),
            "init_time_utc": init_time_utc,
            "init_time_bjt": init_time_utc + timedelta(hours=8),
            "bbox": [lat_min, lat_max, lon_min, lon_max],
            "requested_bbox": list(requested_bbox),
            "thresholds_dbz": [float(value) for value in thresholds],
            "sample_policy": "common intersection" if common_only else "model-specific matches",
            "temporal_matching": "exact initialization and exact valid time",
            "spatial_alignment": "nearest forecast to the same CREF observation grid",
            "valid_mask": "finite CREF and every selected forecast product",
        },
        "coverage": {
            "requested_leads": expected,
            "selected_leads": leads,
            "missing_leads": [lead for lead in expected if lead not in leads],
            "complete": leads == expected,
        },
        "per_lead": per_lead,
        "frames": frames,
    }


def common_valid_times(cases, *, lead_range=(6, 180)) -> list[datetime]:
    """Return exact valid times shared by every selected initialization."""
    cases = list(cases)
    if not cases:
        return []
    start_lead, end_lead = map(int, lead_range)
    valid_time_sets = []
    for case in cases:
        init_time = case["init_time"]
        valid_time_sets.append(
            {
                init_time + timedelta(minutes=int(lead))
                for lead in case["available_leads"]
                if start_lead <= int(lead) <= end_lead
            }
        )
    return sorted(set.intersection(*valid_time_sets))


def run_multimodel_same_valid_time(
    catalog: MultiModelCatalog,
    model_ids,
    initializations,
    valid_time_utc: datetime,
    bbox,
    *,
    thresholds=(20,),
) -> dict:
    """Read different initializations that verify at one exact valid time."""
    model_ids = tuple(dict.fromkeys(model_ids))
    thresholds = tuple(thresholds)
    initializations = sorted(dict.fromkeys(initializations))
    if not 2 <= len(initializations) <= 8:
        raise ValueError("same-valid-time comparison requires 2—8 initializations")
    init_leads = []
    for init_time in initializations:
        lead_seconds = (valid_time_utc - init_time).total_seconds()
        lead = int(round(lead_seconds / 60))
        if (
            lead_seconds != lead * 60
            or lead < 6
            or lead > 180
            or lead % 6
        ):
            raise ValueError(
                f"{valid_time_utc:%Y-%m-%d %H:%M} UTC is not a native "
                f"6—180 minute valid time for initialization "
                f"{init_time:%Y-%m-%d %H:%M} UTC"
            )
        init_leads.append((init_time, lead))
    effective_bbox = _common_spatial_bbox(
        bbox,
        [
            catalog.forecasts[model_id][(init_time, lead)]
            for init_time, lead in init_leads
            for model_id in model_ids
        ],
    )
    items = []
    rows = []
    for init_time, lead in init_leads:
        result = run_multimodel_case(
            catalog,
            model_ids,
            init_time,
            effective_bbox,
            thresholds=thresholds,
            lead_range=(lead, lead),
            frame_leads=(lead,),
        )
        items.append(
            {
                "init_time_utc": init_time,
                "init_time_bjt": result["context"]["init_time_bjt"],
                "lead_minutes": lead,
                "scores": result["per_lead"][0]["models"],
                "frame": result["frames"][lead],
            }
        )
    # A fair update comparison uses the intersection across initializations as
    # well as models. Per-initialization masks can otherwise hide different misses.
    fields = [item["frame"]["observation"] for item in items]
    fields.extend(item["frame"]["forecasts"][model] for item in items for model in model_ids)
    reference = fields[0]
    if any(
        not np.array_equal(field.lat, reference.lat)
        or not np.array_equal(field.lon, reference.lon)
        for field in fields[1:]
    ):
        raise ValueError("same-valid-time fields must share the observation grid")
    common_arrays = common_valid_arrays(*(field.data for field in fields))
    for index, item in enumerate(items):
        observation = common_arrays[index]
        item["scores"] = {
            model_id: _score_array(
                common_arrays[len(items) + index * len(model_ids) + model_index],
                observation, thresholds, lat=reference.lat, lon=reference.lon,
            )
            for model_index, model_id in enumerate(model_ids)
        }
        for model_id, threshold_map in item["scores"].items():
            for threshold, values in threshold_map.items():
                rows.append(
                    {
                        "model_id": model_id,
                        "init_time_utc": item["init_time_utc"].isoformat(),
                        "init_time_bjt": item["init_time_bjt"].isoformat(),
                        "valid_time_utc": valid_time_utc.isoformat(),
                        "lead_minutes": item["lead_minutes"],
                        "threshold_dbz": threshold,
                        **values["contingency"],
                        **{
                            name: values.get(name)
                            for name in (*_SCORE_COLUMNS, *_DIAGNOSTIC_COLUMNS)
                        },
                    }
                )
    return {
        "valid_time_utc": valid_time_utc,
        "valid_time_bjt": valid_time_utc + timedelta(hours=8),
        "model_ids": list(dict.fromkeys(model_ids)),
        "bbox": list(effective_bbox),
        "items": items,
        "per_case": pd.DataFrame(rows),
    }


def _path_fingerprint(paths) -> str:
    digest = hashlib.sha256()
    for path in sorted({Path(value) for value in paths}):
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    return digest.hexdigest()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS multimodel_runs (
            run_key TEXT PRIMARY KEY,
            config_json TEXT NOT NULL,
            expected_rows INTEGER NOT NULL,
            complete INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS multimodel_scores (
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
            forecast_mean_dbz REAL,
            observation_mean_dbz REAL,
            forecast_max_dbz REAL,
            observation_max_dbz REAL,
            mean_reflectivity_error_dbz REAL,
            reflectivity_mae_dbz REAL,
            centroid_distance_km REAL,
            PRIMARY KEY (
                run_key, model_id, init_time_utc, lead_minutes, threshold_dbz
            )
        );
        """
    )
    existing_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(multimodel_scores)")
    }
    for column in _DIAGNOSTIC_COLUMNS:
        if column not in existing_columns:
            connection.execute(
                f"ALTER TABLE multimodel_scores ADD COLUMN {column} REAL"
            )
    return connection


def _aggregate(per_case: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_id",
        "lead_minutes",
        "threshold_dbz",
        "case_count",
        *_COUNT_COLUMNS,
        *_SCORE_COLUMNS,
    ]
    if per_case.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        per_case.groupby(
            ["model_id", "lead_minutes", "threshold_dbz"], as_index=False
        )
        .agg(
            case_count=("init_time_utc", "nunique"),
            **{column: (column, "sum") for column in _COUNT_COLUMNS},
        )
    )
    grouped = pd.DataFrame(grouped).sort_values(
        by=["model_id", "lead_minutes", "threshold_dbz"]
    )
    for score_name, score_fn in CATEGORICAL_SCORES.items():
        grouped[score_name] = [
            score_fn(row)
            for row in grouped.loc[:, _COUNT_COLUMNS].to_dict(orient="records")
        ]
    return pd.DataFrame(grouped.loc[:, columns])


def run_multimodel_batch(
    catalog: MultiModelCatalog,
    model_ids,
    bbox,
    *,
    thresholds=(5, 15, 25, 35),
    lead_range=(6, 180),
    initializations=None,
    complete_only=True,
    cache_path="outputs/cache/phase8_multimodel_metrics.sqlite3",
    force=False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Stream selected initializations into a persistent metric cache."""
    model_ids = tuple(dict.fromkeys(model_ids))
    common_only = len(model_ids) > 1
    cases = discover_multimodel_initializations(
        catalog,
        model_ids,
        lead_range=lead_range,
        common_only=common_only,
    )
    if complete_only:
        cases = [case for case in cases if case["complete"]]
    if initializations is not None:
        selected_set = set(initializations)
        cases = [case for case in cases if case["init_time"] in selected_set]
    if not cases:
        raise ValueError("no initializations satisfy the selected coverage policy")

    requested_bbox = tuple(map(float, bbox))
    effective_bbox = _common_spatial_bbox(
        requested_bbox,
        [
            catalog.forecasts[model_id][(case["init_time"], case["available_leads"][0])]
            for case in cases
            for model_id in model_ids
        ],
    )

    selected_paths = []
    expected_rows = 0
    for case in cases:
        init_time = case["init_time"]
        for lead in case["available_leads"]:
            selected_paths.append(catalog.observations[init_time + pd.Timedelta(minutes=lead)])
            for model_id in model_ids:
                selected_paths.append(catalog.forecasts[model_id][(init_time, lead)].path)
                expected_rows += len(thresholds)
    config = {
        "version": MULTIMODEL_CACHE_VERSION,
        "models": list(model_ids),
        "contract": catalog.stormdit_contract,
        "bbox": list(effective_bbox),
        "requested_bbox": list(requested_bbox),
        "thresholds": [float(value) for value in thresholds],
        "lead_range": [int(value) for value in lead_range],
        "complete_only": bool(complete_only),
        "initializations": [case["init_time"].isoformat() for case in cases],
        "source_fingerprint": _path_fingerprint(selected_paths),
        "temporal_matching": "exact",
        "spatial_alignment": "nearest forecast to CREF grid",
        "valid_mask": "finite CREF and every selected forecast product",
    }
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    run_key = hashlib.sha256(config_json.encode()).hexdigest()
    connection = _connect(Path(cache_path))
    try:
        if force:
            connection.execute("DELETE FROM multimodel_scores WHERE run_key = ?", (run_key,))
            connection.execute("DELETE FROM multimodel_runs WHERE run_key = ?", (run_key,))
        connection.execute(
            "INSERT OR IGNORE INTO multimodel_runs(run_key, config_json, expected_rows) "
            "VALUES (?, ?, ?)",
            (run_key, config_json, expected_rows),
        )
        connection.commit()
        completed_counts = {
            (row[0], datetime.fromisoformat(row[1])): int(row[2])
            for row in connection.execute(
                "SELECT model_id, init_time_utc, COUNT(*) FROM multimodel_scores "
                "WHERE run_key = ? GROUP BY model_id, init_time_utc",
                (run_key,),
            )
        }
        processed_now = 0
        total = len(cases)
        for index, case in enumerate(cases, 1):
            init_time = case["init_time"]
            expected_per_model = len(case["available_leads"]) * len(thresholds)
            if all(
                completed_counts.get((model_id, init_time)) == expected_per_model
                for model_id in model_ids
            ):
                if progress_callback:
                    progress_callback(index, total, f"cache {init_time:%Y-%m-%d %H:%M}")
                continue
            result = run_multimodel_case(
                catalog,
                model_ids,
                init_time,
                effective_bbox,
                thresholds=thresholds,
                lead_range=lead_range,
            )
            rows = []
            for entry in result["per_lead"]:
                for model_id, threshold_map in entry["models"].items():
                    for threshold, values in threshold_map.items():
                        table = values["contingency"]
                        rows.append(
                            (
                                run_key,
                                model_id,
                                init_time.isoformat(),
                                result["context"]["init_time_bjt"].isoformat(),
                                entry["valid_time_utc"].isoformat(),
                                entry["lead_minutes"],
                                threshold,
                                *(table[column] for column in _COUNT_COLUMNS),
                                *(values[column] for column in _SCORE_COLUMNS),
                                *(values[column] for column in _DIAGNOSTIC_COLUMNS),
                            )
                        )
            connection.executemany(
                "INSERT OR REPLACE INTO multimodel_scores ("
                "run_key,model_id,init_time_utc,init_time_bjt,valid_time_utc,"
                "lead_minutes,threshold_dbz,"
                + ",".join(_COUNT_COLUMNS)
                + ","
                + ",".join(_SCORE_COLUMNS)
                + ","
                + ",".join(_DIAGNOSTIC_COLUMNS)
                + ") VALUES ("
                + ",".join("?" for _ in range(
                    7
                    + len(_COUNT_COLUMNS)
                    + len(_SCORE_COLUMNS)
                    + len(_DIAGNOSTIC_COLUMNS)
                ))
                + ")",
                rows,
            )
            connection.commit()
            processed_now += 1
            if progress_callback:
                progress_callback(index, total, f"{init_time:%Y-%m-%d %H:%M} UTC")
        count = connection.execute(
            "SELECT COUNT(*) FROM multimodel_scores WHERE run_key = ?", (run_key,)
        ).fetchone()[0]
        complete = count == expected_rows
        connection.execute(
            "UPDATE multimodel_runs SET complete = ? WHERE run_key = ?",
            (int(complete), run_key),
        )
        connection.commit()
        per_case = pd.read_sql_query(
            "SELECT * FROM multimodel_scores WHERE run_key = ? "
            "ORDER BY init_time_utc, lead_minutes, model_id, threshold_dbz",
            connection,
            params=[run_key],
        )
    finally:
        connection.close()
    return {
        "context": {
            "model_ids": list(model_ids),
            "model_labels": [MODEL_LABELS[value] for value in model_ids],
            "model_contract": catalog.stormdit_contract,
            "bbox": list(effective_bbox),
            "requested_bbox": list(requested_bbox),
            "thresholds_dbz": list(map(float, thresholds)),
            "lead_range_minutes": list(map(int, lead_range)),
            "sample_policy": "common intersection" if common_only else "model-specific matches",
            "complete_only": bool(complete_only),
        },
        "coverage": {
            "selected_initialization_count": len(cases),
            "selected_pairs": sum(len(case["available_leads"]) for case in cases),
            "complete_initializations": sum(case["complete"] for case in cases),
        },
        "cases": cases,
        "per_case": per_case,
        "summary": _aggregate(per_case),
        "cache": {
            "path": str(cache_path),
            "run_key": run_key,
            "computed_initializations": processed_now,
            "complete": complete,
        },
    }

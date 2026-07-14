"""Streaming full-day verification with an incremental SQLite metric cache."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Callable

import pandas as pd

from data_io.catalog import DailyCatalog, DuplicatePolicy, build_daily_catalog
from data_io.fenglei_mosaic import read_swan_mosaic
from data_io.grid import regrid_nearest
from data_io.radar_obs import read_radar_observation
from verification.metrics import CATEGORICAL_SCORES, contingency_table

BATCH_CACHE_VERSION = 1
_COUNT_COLUMNS = (
    "hits",
    "misses",
    "false_alarms",
    "correct_negatives",
    "valid_count",
    "invalid_count",
)
_SCORE_COLUMNS = tuple(CATEGORICAL_SCORES)


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="seconds") + "Z"


def _score_table(table: dict) -> dict[str, float]:
    return {name: score_fn(table) for name, score_fn in CATEGORICAL_SCORES.items()}


def _normalize_thresholds(thresholds) -> tuple[float, ...]:
    normalized = tuple(dict.fromkeys(float(value) for value in thresholds))
    if not normalized:
        raise ValueError("at least one reflectivity threshold is required")
    if len(normalized) > 8 or any(value < 0 or value > 75 for value in normalized):
        raise ValueError("thresholds must contain at most eight values between 0 and 75 dBZ")
    return normalized


def _run_config(
    catalog: DailyCatalog,
    *,
    bbox,
    thresholds,
    lead_range,
    duplicate_policy,
    model_id,
    max_pairs,
) -> dict:
    return {
        "version": BATCH_CACHE_VERSION,
        "case_date_bjt": catalog.case_date_bjt.isoformat(),
        "forecast_fingerprint": catalog.forecast_fingerprint,
        "observation_fingerprint": catalog.observation_fingerprint,
        "bbox": [float(value) for value in bbox],
        "thresholds_dbz": list(thresholds),
        "lead_range_minutes": [int(lead_range[0]), int(lead_range[1])],
        "duplicate_policy": duplicate_policy,
        "model_id": model_id,
        "max_pairs": max_pairs,
        "temporal_matching": "exact_valid_time",
        "spatial_alignment": "nearest_forecast_to_observation_grid",
        "observation_missing_raw": -32768,
    }


def _connect_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_key TEXT PRIMARY KEY,
            config_json TEXT NOT NULL,
            expected_pairs INTEGER NOT NULL,
            complete INTEGER NOT NULL DEFAULT 0,
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scores (
            run_key TEXT NOT NULL,
            model_id TEXT NOT NULL,
            forecast_path TEXT NOT NULL,
            observation_path TEXT NOT NULL,
            init_time_utc TEXT NOT NULL,
            valid_time_utc TEXT NOT NULL,
            init_time_bjt TEXT NOT NULL,
            valid_time_bjt TEXT NOT NULL,
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
            PRIMARY KEY (run_key, forecast_path, threshold_dbz),
            FOREIGN KEY (run_key) REFERENCES runs(run_key)
        );
        CREATE INDEX IF NOT EXISTS scores_run_lead_threshold
            ON scores(run_key, lead_minutes, threshold_dbz);
        """
    )
    return connection


def _aggregate_scores(per_case: pd.DataFrame) -> pd.DataFrame:
    """Pool contingency counts by lead/threshold and recompute the scores."""
    columns = [
        "lead_minutes",
        "threshold_dbz",
        "case_count",
        *_COUNT_COLUMNS,
        *_SCORE_COLUMNS,
    ]
    if per_case.empty:
        return pd.DataFrame(columns=columns)
    grouped = pd.DataFrame(
        per_case.groupby(["lead_minutes", "threshold_dbz"], as_index=False)
        .agg(
            case_count=("forecast_path", "nunique"),
            **{column: (column, "sum") for column in _COUNT_COLUMNS},
        )
    )
    grouped = grouped.sort_values(  # pyright: ignore[reportCallIssue]
        ["lead_minutes", "threshold_dbz"]
    )
    scores = []
    for row in grouped.to_dict(orient="records"):
        scores.append(_score_table(row))
    score_frame = pd.DataFrame(scores, index=grouped.index)
    combined = pd.concat([grouped, score_frame], axis=1)
    return pd.DataFrame(combined.loc[:, columns])


def _load_scores(connection: sqlite3.Connection, run_key: str) -> pd.DataFrame:
    return pd.DataFrame(
        pd.read_sql_query(
            "SELECT * FROM scores WHERE run_key = ? "
            "ORDER BY init_time_bjt, lead_minutes, threshold_dbz",
            connection,
            params=[run_key],
        )
    )


def run_daily_fenglei_verification(
    forecast_dir,
    observation_dir,
    case_date_bjt,
    bbox,
    thresholds=(10, 20, 35),
    *,
    lead_range=(6, 180),
    duplicate_policy: DuplicatePolicy = "latest_generation",
    catalog_cache_dir="outputs/cache/catalogs",
    metric_cache_path="outputs/cache/phase5_metrics.sqlite3",
    model_id="fenglei",
    force=False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    max_pairs: int | None = None,
) -> dict:
    """Score every exactly matched forecast in one Beijing-calendar day.

    Forecasts are decoded one at a time and observations one valid time at a
    time.  Each completed forecast/observation pair is committed to SQLite, so
    an interrupted long run resumes from its last complete pair.
    """
    if not str(forecast_dir).strip() or not str(observation_dir).strip():
        raise ValueError("both full-day forecast and observation directories are required")
    thresholds = _normalize_thresholds(thresholds)
    lat_min, lat_max, lon_min, lon_max = map(float, bbox)
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError(f"invalid bbox: {bbox!r}")
    start_lead, end_lead = map(int, lead_range)
    if start_lead < 0 or start_lead > end_lead:
        raise ValueError(f"invalid lead range: {lead_range!r}")
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be positive")

    catalog = build_daily_catalog(
        forecast_dir,
        observation_dir,
        case_date_bjt,
        duplicate_policy=duplicate_policy,
        cache_dir=catalog_cache_dir,
    )
    observation_by_time = catalog.observation_by_valid_time
    records = [
        record
        for record in catalog.matched_forecasts
        if start_lead <= record.lead_minutes <= end_lead
    ]
    if max_pairs is not None:
        records = records[:max_pairs]
    if not records:
        raise ValueError("no exactly matched forecast/observation pairs in the requested range")

    config = _run_config(
        catalog,
        bbox=(lat_min, lat_max, lon_min, lon_max),
        thresholds=thresholds,
        lead_range=(start_lead, end_lead),
        duplicate_policy=duplicate_policy,
        model_id=model_id,
        max_pairs=max_pairs,
    )
    config_json = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    run_key = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    connection = _connect_cache(Path(metric_cache_path))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        if force:
            connection.execute("DELETE FROM scores WHERE run_key = ?", (run_key,))
            connection.execute("DELETE FROM runs WHERE run_key = ?", (run_key,))
        connection.execute(
            """
            INSERT INTO runs(run_key, config_json, expected_pairs, complete, created_utc, updated_utc)
            VALUES (?, ?, ?, 0, ?, ?)
            ON CONFLICT(run_key) DO UPDATE SET
                config_json=excluded.config_json,
                expected_pairs=excluded.expected_pairs,
                updated_utc=excluded.updated_utc
            """,
            (run_key, config_json, len(records), now, now),
        )
        connection.commit()

        completed_paths = {
            row[0]
            for row in connection.execute(
                """
                SELECT forecast_path FROM scores WHERE run_key = ?
                GROUP BY forecast_path HAVING COUNT(*) = ?
                """,
                (run_key, len(thresholds)),
            )
        }
        cached_before = len(completed_paths)
        processed_now = 0
        current_obs_time = None
        observation = None
        total = len(records)
        # Retain one extra source cell around the domain so nearest-neighbour
        # alignment at the requested edge cannot be clipped prematurely.
        forecast_bbox = (lat_min - 0.02, lat_max + 0.02, lon_min - 0.02, lon_max + 0.02)
        for index, record in enumerate(records, start=1):
            forecast_path = str(record.path)
            if forecast_path in completed_paths:
                if progress_callback is not None:
                    progress_callback(index, total, f"cache: {record.path.name}")
                continue
            obs_record = observation_by_time[record.valid_time_utc]
            if current_obs_time != record.valid_time_utc:
                observation = read_radar_observation(
                    obs_record.path,
                    bbox=(lat_min, lat_max, lon_min, lon_max),
                )
                if observation.valid_time != record.valid_time_utc:
                    raise ValueError(
                        f"observation catalog/header time mismatch: {obs_record.path}"
                    )
                current_obs_time = record.valid_time_utc
            assert observation is not None
            forecast = read_swan_mosaic(record.path, bbox=forecast_bbox)
            if (
                forecast.init_time != record.init_time_utc
                or forecast.valid_time != record.valid_time_utc
                or forecast.lead_minutes != record.lead_minutes
            ):
                raise ValueError(f"forecast catalog/header time mismatch: {record.path}")
            forecast_on_obs = regrid_nearest(forecast, observation.lat, observation.lon)
            rows = []
            for threshold in thresholds:
                table = contingency_table(forecast_on_obs, observation.data, threshold)
                scores = _score_table(table)
                rows.append(
                    (
                        run_key,
                        model_id,
                        forecast_path,
                        str(obs_record.path),
                        _utc_text(record.init_time_utc),
                        _utc_text(record.valid_time_utc),
                        record.init_time_bjt.isoformat(timespec="seconds"),
                        record.valid_time_bjt.isoformat(timespec="seconds"),
                        record.lead_minutes,
                        threshold,
                        *(table[column] for column in _COUNT_COLUMNS),
                        *(scores[column] for column in _SCORE_COLUMNS),
                    )
                )
            connection.executemany(
                """
                INSERT OR REPLACE INTO scores(
                    run_key, model_id, forecast_path, observation_path,
                    init_time_utc, valid_time_utc, init_time_bjt, valid_time_bjt,
                    lead_minutes, threshold_dbz,
                    hits, misses, false_alarms, correct_negatives, valid_count, invalid_count,
                    pod, far, csi, bias_score, ets
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            processed_now += 1
            connection.commit()
            if progress_callback is not None:
                progress_callback(index, total, record.path.name)

        completed_total = connection.execute(
            "SELECT COUNT(DISTINCT forecast_path) FROM scores WHERE run_key = ?",
            (run_key,),
        ).fetchone()[0]
        complete = completed_total == len(records)
        connection.execute(
            "UPDATE runs SET complete = ?, updated_utc = ? WHERE run_key = ?",
            (int(complete), datetime.now(timezone.utc).isoformat(timespec="seconds"), run_key),
        )
        connection.commit()
        per_case = _load_scores(connection, run_key)
    finally:
        connection.close()

    summary = _aggregate_scores(per_case)
    matched_valid_times = [record.valid_time_bjt for record in catalog.matched_forecasts]
    matched_init_times = [record.init_time_bjt for record in catalog.matched_forecasts]
    return {
        "context": {
            "case_date_bjt": catalog.case_date_bjt.isoformat(),
            "model_id": model_id,
            "bbox": [lat_min, lat_max, lon_min, lon_max],
            "thresholds_dbz": list(thresholds),
            "lead_range_minutes": [start_lead, end_lead],
            "temporal_matching": "exact valid-time equality",
            "spatial_alignment": "nearest-neighbour forecast to CREF observation grid",
            "aggregation": "pooled contingency counts by lead and threshold",
        },
        "catalog": catalog,
        "per_case": per_case,
        "summary": summary,
        "coverage": {
            "indexed_forecasts": len(catalog.forecasts),
            "unique_observations": len(catalog.observations),
            "matched_pairs_available": len(catalog.matched_forecasts),
            "matched_valid_start_bjt": (
                min(matched_valid_times).isoformat(timespec="seconds")
                if matched_valid_times
                else None
            ),
            "matched_valid_end_bjt": (
                max(matched_valid_times).isoformat(timespec="seconds")
                if matched_valid_times
                else None
            ),
            "matched_init_start_bjt": (
                min(matched_init_times).isoformat(timespec="seconds")
                if matched_init_times
                else None
            ),
            "matched_init_end_bjt": (
                max(matched_init_times).isoformat(timespec="seconds")
                if matched_init_times
                else None
            ),
            "selected_pairs": len(records),
            "completed_pairs": int(completed_total),
            "duplicate_observation_times": len(catalog.duplicates),
            "discarded_observation_revisions": sum(
                len(item.discarded) for item in catalog.duplicates
            ),
        },
        "cache": {
            "run_key": run_key,
            "path": str(Path(metric_cache_path)),
            "catalog_cache_hit": catalog.cache_hit,
            "metric_pairs_reused": cached_before,
            "metric_pairs_computed": processed_now,
            "complete": complete,
        },
    }

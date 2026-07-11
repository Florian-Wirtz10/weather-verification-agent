"""Glue code driving the data_io -> verification -> visualization pipeline.

Phase 1: no natural-language parsing yet, just a typed entry point that
takes the parameters a future NL layer would extract (grid shape, seed,
thresholds) and runs the full pipeline.
"""

from pathlib import Path
from typing import TypedDict

from data_io.fenglei_mosaic import read_swan_mosaic, read_swan_mosaic_valid_time
from data_io.grid import GriddedField, align_pair
from data_io.radar_obs import index_cref_files, read_cref
from data_io.synthetic import generate_synthetic_fields
from verification.metrics import CATEGORICAL_SCORES, compute_all_metrics, contingency_table
from visualization.plots import (
    plot_error,
    plot_field,
    plot_reflectivity_error,
    plot_reflectivity_field,
    plot_score_time_series,
)


class VerificationFrame(TypedDict):
    """One forecast/observation pair retained for endpoint plots."""

    lead_minutes: int
    forecast: GriddedField
    obs: GriddedField


def run_verification(shape=(64, 64), seed=None, thresholds=(1, 5, 10)):
    """Run the synthetic-data verification pipeline and return arrays, metrics, and figures."""
    forecast, obs = generate_synthetic_fields(shape=shape, seed=seed)
    metrics = compute_all_metrics(forecast, obs, thresholds)

    figures = {
        "forecast": plot_field(forecast, "Forecast"),
        "obs": plot_field(obs, "Observation"),
        "error": plot_error(forecast, obs),
    }

    return {
        "forecast": forecast,
        "obs": obs,
        "metrics": metrics,
        "figures": figures,
    }


def run_fenglei_case_verification(
    fenglei_dir,
    radar_dir,
    bbox,
    thresholds=(10, 20, 35),
    fenglei_pattern="*REFMOSAIC*MIN.bz2",
):
    """Verify 风雷 REFMOSAIC reflectivity forecasts (one init run) against radar CREF
    observations across every forecast lead time found in `fenglei_dir`.

    `bbox` is (lat_min, lat_max, lon_min, lon_max). Real 风雷/雷达 grids don't share an
    exact origin, so each lead's forecast is regridded onto the observation's native
    grid via data_io.grid.align_pair before scoring — see that module for why. Forecast
    and observation timestamps must match exactly. Files are indexed from metadata and
    matching pairs are loaded one at a time to keep memory bounded.

    Returns per-lead categorical scores plus the earliest/latest lead's forecast,
    obs, and error figures, and CSI/POD/FAR/ETS/bias time-series figures.
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    obs_files = index_cref_files(radar_dir)
    if not obs_files:
        raise ValueError(f"no radar observation files found under {radar_dir}")

    forecast_paths = sorted(Path(fenglei_dir).glob(fenglei_pattern))
    if not forecast_paths:
        raise ValueError(f"no fenglei mosaic files matching {fenglei_pattern!r} under {fenglei_dir}")

    per_lead = []
    first_frame: VerificationFrame | None = None
    last_frame: VerificationFrame | None = None
    for path in forecast_paths:
        valid_time = read_swan_mosaic_valid_time(path)
        obs_path = obs_files.get(valid_time)
        if obs_path is None:
            continue

        forecast = read_swan_mosaic(path)
        if forecast.lead_minutes is None:
            raise ValueError(f"{path}: forecast lead time is missing")
        lead_minutes = forecast.lead_minutes
        obs = read_cref(obs_path)

        forecast_on_grid, obs_crop = align_pair(forecast, obs, lat_min, lat_max, lon_min, lon_max)
        categorical = {
            threshold: {
                name: score_fn(contingency_table(forecast_on_grid.data, obs_crop.data, threshold))
                for name, score_fn in CATEGORICAL_SCORES.items()
            }
            for threshold in thresholds
        }
        per_lead.append(
            {
                "lead_minutes": lead_minutes,
                "valid_time": forecast.valid_time,
                "categorical": categorical,
            }
        )

        frame: VerificationFrame = {
            "lead_minutes": lead_minutes,
            "forecast": forecast_on_grid,
            "obs": obs_crop,
        }
        if first_frame is None or lead_minutes < first_frame["lead_minutes"]:
            first_frame = frame
        if last_frame is None or lead_minutes > last_frame["lead_minutes"]:
            last_frame = frame

    if not per_lead or first_frame is None or last_frame is None:
        raise ValueError(
            "no fenglei forecast frame found an exactly matching radar observation timestamp"
        )

    per_lead.sort(key=lambda entry: entry["lead_minutes"])
    lead_minutes = [entry["lead_minutes"] for entry in per_lead]

    figures = {}
    for score_name in CATEGORICAL_SCORES:
        scores_by_threshold = {
            threshold: [entry["categorical"][threshold][score_name] for entry in per_lead]
            for threshold in thresholds
        }
        figures[f"{score_name}_time_series"] = plot_score_time_series(
            lead_minutes, scores_by_threshold, score_name
        )

    figures["forecast_first_lead"] = plot_reflectivity_field(
        first_frame["forecast"], title=f"Forecast lead={first_frame['lead_minutes']}min"
    )
    figures["obs_first_lead"] = plot_reflectivity_field(
        first_frame["obs"], title=f"Obs @ {first_frame['obs'].valid_time:%Y-%m-%d %H:%M} UTC"
    )
    figures["error_first_lead"] = plot_reflectivity_error(first_frame["forecast"], first_frame["obs"])

    figures["forecast_last_lead"] = plot_reflectivity_field(
        last_frame["forecast"], title=f"Forecast lead={last_frame['lead_minutes']}min"
    )
    figures["obs_last_lead"] = plot_reflectivity_field(
        last_frame["obs"], title=f"Obs @ {last_frame['obs'].valid_time:%Y-%m-%d %H:%M} UTC"
    )
    figures["error_last_lead"] = plot_reflectivity_error(last_frame["forecast"], last_frame["obs"])

    return {
        "per_lead": per_lead,
        "lead_minutes": lead_minutes,
        "figures": figures,
    }

"""Deterministic execution layer for synthetic and real-data verification."""

from collections import defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import TypedDict

from agent.errors import AmbiguousCaseError, DataContractError, NoDataError
from data_io.fenglei_mosaic import read_swan_mosaic, read_swan_mosaic_metadata
from data_io.grid import GriddedField, align_pair
from data_io.radar_obs import index_cref_files, read_cref
from data_io.synthetic import generate_synthetic_fields
from verification.metrics import CATEGORICAL_SCORES, compute_all_metrics, contingency_table
from visualization.plots import (
    plot_error,
    plot_field,
    plot_reflectivity_comparison,
    plot_reflectivity_error,
    plot_score_time_series,
)


class VerificationFrame(TypedDict):
    """One forecast/observation pair retained for endpoint plots."""

    lead_minutes: int
    forecast: GriddedField
    obs: GriddedField


_DEFAULT_VIEWS = ({"type": "score_time_series", "metric": "csi"},)
_NATIVE_LEAD_STEP_MINUTES = 6


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


def _index_fenglei_cases(fenglei_dir, radar_dir, fenglei_pattern):
    """Return radar paths and exactly time-matched forecast files grouped by init."""
    obs_files = index_cref_files(radar_dir)
    if not obs_files:
        raise NoDataError(f"no radar observation files found under {radar_dir}")

    forecast_paths = sorted(Path(fenglei_dir).glob(fenglei_pattern))
    if not forecast_paths:
        raise NoDataError(
            f"no fenglei mosaic files matching {fenglei_pattern!r} under {fenglei_dir}"
        )

    cases = defaultdict(dict)
    for path in forecast_paths:
        metadata = read_swan_mosaic_metadata(path)
        if metadata.valid_time not in obs_files:
            continue
        lead_map = cases[metadata.init_time]
        if metadata.lead_minutes in lead_map:
            previous = lead_map[metadata.lead_minutes][1]
            raise DataContractError(
                f"duplicate forecast lead {metadata.lead_minutes} minutes for init "
                f"{metadata.init_time:%Y-%m-%d %H:%M} UTC: {previous} and {path}"
            )
        lead_map[metadata.lead_minutes] = (metadata, path)
    return obs_files, cases


def discover_fenglei_cases(
    fenglei_dir,
    radar_dir,
    fenglei_pattern="*REFMOSAIC*MIN.bz2",
):
    """Discover single-initialization cases with exact forecast/observation matches."""
    _, indexed = _index_fenglei_cases(fenglei_dir, radar_dir, fenglei_pattern)
    cases = []
    for init_time, lead_map in sorted(indexed.items()):
        leads = sorted(lead_map)
        if not leads:
            continue
        metadata = lead_map[leads[0]][0]
        cases.append(
            {
                "case_id": f"fenglei-refmosaic-{init_time:%Y%m%d%H%M}",
                "init_time": init_time,
                "date": init_time.date().isoformat(),
                "product": "Fenglei V1.0 REFMOSAIC",
                "product_header": metadata.product,
                "available_leads": leads,
            }
        )
    return cases


def _coerce_init_time(value):
    if value is None or isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise DataContractError(f"invalid forecast initialization time: {value!r}") from exc
    if parsed is not None and parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _resolve_spatial_lead(selection, selected_leads):
    if selection == "first":
        return selected_leads[0]
    if selection == "last":
        return selected_leads[-1]
    if isinstance(selection, int) and not isinstance(selection, bool):
        return selection if selection in selected_leads else None
    raise DataContractError(f"invalid spatial lead selection: {selection!r}")


def _normalize_execution_views(views, selected_leads):
    normalized = []
    unavailable = []
    for view in views or _DEFAULT_VIEWS:
        view_type = view.get("type")
        if view_type == "score_time_series":
            metric = view.get("metric")
            if metric not in CATEGORICAL_SCORES:
                raise DataContractError(f"unsupported score metric: {metric!r}")
            normalized.append({"type": view_type, "metric": metric})
        elif view_type in {"reflectivity_comparison", "reflectivity_error"}:
            selection = view.get("lead", "last")
            lead = _resolve_spatial_lead(selection, selected_leads)
            if lead is None:
                unavailable.append({"type": view_type, "lead": selection})
            else:
                normalized.append({"type": view_type, "lead": lead})
        else:
            raise DataContractError(f"unsupported view type: {view_type!r}")
    if not normalized:
        raise NoDataError("none of the requested views has data in the selected lead range")
    return normalized, unavailable


def run_fenglei_case_verification(
    fenglei_dir,
    radar_dir,
    bbox,
    thresholds=(10, 20, 35),
    *,
    init_time=None,
    lead_range=(0, 180),
    views=None,
    fenglei_pattern="*REFMOSAIC*MIN.bz2",
):
    """Verify one explicitly resolved REFMOSAIC initialization against exact CREF times.

    File headers are indexed first. Only requested, exactly matched leads are fully decoded,
    and only requested figures are rendered, keeping both scope and memory auditable.
    """
    lat_min, lat_max, lon_min, lon_max = map(float, bbox)
    if lat_min >= lat_max or lon_min >= lon_max:
        raise DataContractError("bbox minimums must be smaller than maximums")
    start_lead, end_lead = lead_range
    if start_lead < 0 or start_lead > end_lead:
        raise DataContractError(f"invalid lead range: {lead_range!r}")

    obs_files, indexed_cases = _index_fenglei_cases(fenglei_dir, radar_dir, fenglei_pattern)
    if not indexed_cases:
        raise NoDataError("no exactly time-matched fenglei/radar case is available")

    requested_init = _coerce_init_time(init_time)
    if requested_init is None:
        if len(indexed_cases) != 1:
            choices = ", ".join(f"{value:%Y-%m-%d %H:%M}" for value in sorted(indexed_cases))
            raise AmbiguousCaseError(f"select one forecast initialization: {choices} UTC")
        requested_init = next(iter(indexed_cases))
    if requested_init not in indexed_cases:
        raise NoDataError(
            f"no exactly matched data for init {requested_init:%Y-%m-%d %H:%M} UTC"
        )

    case_files = indexed_cases[requested_init]
    available_leads = sorted(case_files)
    selected_leads = [lead for lead in available_leads if start_lead <= lead <= end_lead]
    if not selected_leads:
        raise NoDataError(
            f"no matched forecast leads in requested range {start_lead}–{end_lead} minutes"
        )

    first_expected = math.ceil(start_lead / _NATIVE_LEAD_STEP_MINUTES) * _NATIVE_LEAD_STEP_MINUTES
    expected_leads = list(
        range(first_expected, end_lead + 1, _NATIVE_LEAD_STEP_MINUTES)
    )
    missing_leads = [lead for lead in expected_leads if lead not in available_leads]
    resolved_views, unavailable_views = _normalize_execution_views(views, selected_leads)
    needed_frame_leads = {
        view["lead"] for view in resolved_views if view["type"] != "score_time_series"
    }

    per_lead = []
    frames: dict[int, VerificationFrame] = {}
    for lead_minutes in selected_leads:
        metadata, path = case_files[lead_minutes]
        forecast = read_swan_mosaic(path)
        obs = read_cref(obs_files[metadata.valid_time])
        forecast_on_grid, obs_crop = align_pair(
            forecast, obs, lat_min, lat_max, lon_min, lon_max
        )

        categorical = {}
        tables = {}
        for threshold in thresholds:
            table = contingency_table(forecast_on_grid.data, obs_crop.data, threshold)
            tables[threshold] = table
            categorical[threshold] = {
                name: score_fn(table) for name, score_fn in CATEGORICAL_SCORES.items()
            }
        per_lead.append(
            {
                "lead_minutes": lead_minutes,
                "valid_time": metadata.valid_time,
                "categorical": categorical,
                "contingency": tables,
            }
        )
        if lead_minutes in needed_frame_leads:
            frames[lead_minutes] = {
                "lead_minutes": lead_minutes,
                "forecast": forecast_on_grid,
                "obs": obs_crop,
            }

    figures = {}
    figure_metadata = {}
    for index, view in enumerate(resolved_views, start=1):
        key = f"view_{index:02d}"
        view_type = view["type"]
        if view_type == "score_time_series":
            metric = view["metric"]
            scores_by_threshold = {
                threshold: [entry["categorical"][threshold][metric] for entry in per_lead]
                for threshold in thresholds
            }
            figures[key] = plot_score_time_series(
                selected_leads, scores_by_threshold, metric
            )
        elif view_type == "reflectivity_comparison":
            frame = frames[view["lead"]]
            figures[key] = plot_reflectivity_comparison(
                frame["forecast"],
                frame["obs"],
                title=f"Forecast vs observation — lead {view['lead']} min",
            )
        else:
            frame = frames[view["lead"]]
            figures[key] = plot_reflectivity_error(
                frame["forecast"],
                frame["obs"],
                title=f"Forecast - observation — lead {view['lead']} min",
            )
        figure_metadata[key] = view

    warnings = []
    if missing_leads:
        warnings.append(f"requested range contains missing leads: {missing_leads}")
    if unavailable_views:
        warnings.append(f"some spatial views had no exact lead: {unavailable_views}")

    return {
        "context": {
            "case_id": f"fenglei-refmosaic-{requested_init:%Y%m%d%H%M}",
            "product": "Fenglei V1.0 REFMOSAIC",
            "variable": "radar composite reflectivity",
            "units": "dBZ",
            "init_time": requested_init,
            "bbox": [lat_min, lat_max, lon_min, lon_max],
            "temporal_matching": "exact valid-time equality",
            "spatial_alignment": "nearest-neighbor forecast to radar grid; outside masked",
            "single_case": True,
        },
        "coverage": {
            "requested_range": [start_lead, end_lead],
            "native_step_minutes": _NATIVE_LEAD_STEP_MINUTES,
            "available_leads": available_leads,
            "selected_leads": selected_leads,
            "missing_leads": missing_leads,
            "unavailable_views": unavailable_views,
        },
        "per_lead": per_lead,
        "lead_minutes": selected_leads,
        "figures": figures,
        "figure_metadata": figure_metadata,
        "warnings": warnings,
    }

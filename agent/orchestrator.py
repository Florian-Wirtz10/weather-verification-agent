"""Deterministic execution layer for synthetic and real-data verification."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import re
from typing import TypedDict

from agent.errors import AmbiguousCaseError, DataContractError, NoDataError
from agent.request_schema import (
    MAX_ANIMATION_FRAMES,
    MAX_SPATIAL_VIEWS,
    MAX_VIEWS,
    SUPPORTED_PRESENTATION_STRATEGIES,
)
from data_io.fenglei_mosaic import (
    SwanMosaicMetadata,
    read_swan_mosaic,
    read_swan_mosaic_metadata,
)
from data_io.grid import GriddedField, align_pair
from data_io.radar_obs import index_radar_observation_files, read_radar_observation
from data_io.synthetic import generate_synthetic_fields
from verification.metrics import CATEGORICAL_SCORES, compute_all_metrics, contingency_table
from visualization.plots import (
    plot_error,
    plot_field,
    plot_reflectivity_comparison,
    plot_reflectivity_error,
    plot_forecast_observed_rate_trajectory,
    plot_score_time_series,
    render_linked_score_animation,
    render_reflectivity_animation,
)


class VerificationFrame(TypedDict):
    """One forecast/observation pair retained for endpoint plots."""

    lead_minutes: int
    forecast: GriddedField
    obs: GriddedField


_DEFAULT_VIEWS = ({"type": "score_time_series", "metric": "csi"},)
_NATIVE_LEAD_STEP_MINUTES = 6
_FILENAME_FORECAST_RE = re.compile(
    r"(?:FENGL(?P<v11>-V11)?-)?REFMOSAIC-"
    r"(?P<init>\d{12})-(?P<lead>\d{3})MIN\.bz2\Z"
)


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


def _metadata_from_forecast_filename(path: Path):
    """Return timing from a standard REFMOSAIC filename, if it is unambiguous.

    The operational filename is sufficient for discovery.  Full SWAN-header
    validation is deliberately deferred to the selected files at execution
    time, rather than bzip-reading every file on every Streamlit rerun.
    """
    match = _FILENAME_FORECAST_RE.search(path.name)
    if match is None:
        return None
    init_time = datetime.strptime(match.group("init"), "%Y%m%d%H%M")
    # V1.1 delivery filenames use BJT; their SWAN headers remain UTC.
    if match.group("v11"):
        init_time -= timedelta(hours=8)
    lead_minutes = int(match.group("lead"))
    return SwanMosaicMetadata(
        init_time=init_time,
        valid_time=init_time + timedelta(minutes=lead_minutes),
        lead_minutes=lead_minutes,
        product="REFMOSAIC (filename-indexed)",
    )


def _index_fenglei_cases(fenglei_dir, radar_dir, fenglei_pattern):
    """Return radar paths and exactly time-matched forecast files grouped by init.

    Standard operational filenames are indexed without opening compressed
    payloads.  Files with non-standard names retain the conservative
    header-reading fallback.
    """
    obs_files = index_radar_observation_files(radar_dir)
    if not obs_files:
        raise NoDataError(f"no radar observation files found under {radar_dir}")

    forecast_paths = sorted(Path(fenglei_dir).glob(fenglei_pattern))
    if not forecast_paths:
        raise NoDataError(
            f"no fenglei mosaic files matching {fenglei_pattern!r} under {fenglei_dir}"
        )

    cases = defaultdict(dict)
    for path in forecast_paths:
        filename_metadata = _metadata_from_forecast_filename(path)
        metadata = (
            read_swan_mosaic_metadata(path)
            if filename_metadata is None
            else filename_metadata
        )
        init_time = metadata.init_time
        valid_time = metadata.valid_time
        lead_minutes = metadata.lead_minutes
        if valid_time not in obs_files:
            continue
        lead_map = cases[init_time]
        if lead_minutes in lead_map:
            previous = lead_map[lead_minutes][1]
            raise DataContractError(
                f"duplicate forecast lead {lead_minutes} minutes for init "
                f"{init_time:%Y-%m-%d %H:%M} UTC: {previous} and {path}"
            )
        lead_map[lead_minutes] = (metadata, path)
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
    requested_views = views or _DEFAULT_VIEWS
    if len(requested_views) > MAX_VIEWS:
        raise DataContractError(f"at most {MAX_VIEWS} views can be rendered")
    spatial_count = sum(
        view.get("type")
        in {"reflectivity_comparison", "reflectivity_error", "reflectivity_animation"}
        for view in requested_views
    )
    if spatial_count > MAX_SPATIAL_VIEWS:
        raise DataContractError(
            f"at most {MAX_SPATIAL_VIEWS} spatial views can be rendered"
        )
    normalized = []
    unavailable = []
    for view in requested_views:
        view_type = view.get("type")
        if view_type == "score_time_series":
            metric = view.get("metric")
            if metric not in CATEGORICAL_SCORES:
                raise DataContractError(f"unsupported score metric: {metric!r}")
            normalized.append({"type": view_type, "metric": metric})
        elif view_type == "forecast_observed_rate":
            normalized.append({"type": view_type})
        elif view_type in {"reflectivity_comparison", "reflectivity_error"}:
            selection = view.get("lead", "last")
            lead = _resolve_spatial_lead(selection, selected_leads)
            if lead is None:
                unavailable.append({"type": view_type, "lead": selection})
            else:
                normalized.append({"type": view_type, "lead": lead})
        elif view_type == "reflectivity_animation":
            lead_range = view.get("lead_range_minutes")
            if not isinstance(lead_range, dict):
                raise DataContractError("reflectivity_animation requires lead_range_minutes")
            start = lead_range.get("start")
            end = lead_range.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or start > end:
                raise DataContractError("invalid reflectivity_animation lead range")
            animation_leads = [lead for lead in selected_leads if start <= lead <= end]
            if not animation_leads:
                unavailable.append({"type": view_type, "lead_range_minutes": lead_range})
            elif len(animation_leads) > MAX_ANIMATION_FRAMES:
                raise DataContractError(
                    f"reflectivity animation supports at most {MAX_ANIMATION_FRAMES} matched frames"
                )
            else:
                normalized.append(
                    {
                        "type": view_type,
                        "lead_range_minutes": {"start": start, "end": end},
                        "leads": animation_leads,
                        "fps": view.get("fps", 2),
                    }
                )
        else:
            raise DataContractError(f"unsupported view type: {view_type!r}")
    if not normalized:
        raise NoDataError("none of the requested views has data in the selected lead range")
    return normalized, unavailable


def _resolve_presentation_strategy(presentation, resolved_views):
    """Validate the planner's strategy and deterministically resolve fallbacks."""
    raw = presentation if isinstance(presentation, dict) else {}
    strategy = raw.get("strategy", "auto")
    if not isinstance(strategy, str) or strategy not in SUPPORTED_PRESENTATION_STRATEGIES:
        raise DataContractError(f"unsupported presentation strategy: {strategy!r}")
    metric = raw.get("metric")
    plan = {
        "requested_strategy": strategy,
        "resolved_strategy": "separate_media",
        "metric": metric if isinstance(metric, str) else None,
    }
    if strategy != "linked_score_animation":
        return plan

    if metric not in CATEGORICAL_SCORES:
        raise DataContractError("linked_score_animation requires a supported score metric")
    score_indices = [
        index
        for index, view in enumerate(resolved_views)
        if view["type"] == "score_time_series" and view["metric"] == metric
    ]
    animation_indices = [
        index
        for index, view in enumerate(resolved_views)
        if view["type"] == "reflectivity_animation"
    ]
    if len(score_indices) == 1 and animation_indices:
        plan.update(
            {
                "resolved_strategy": "linked_score_animation",
                "score_index": score_indices[0],
                "animation_index": animation_indices[0],
            }
        )
    else:
        plan["fallback_reason"] = (
            "联动动画需要同时具有一个指定评分曲线和至少一个空间动画；"
            "本次条件不满足，已分别展示已生成媒体。"
        )
    return plan


def run_fenglei_case_verification(
    fenglei_dir,
    radar_dir,
    bbox,
    thresholds=(10, 20, 35),
    *,
    init_time=None,
    lead_range=(0, 180),
    views=None,
    presentation=None,
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

    first_expected = max(
        _NATIVE_LEAD_STEP_MINUTES,
        math.ceil(start_lead / _NATIVE_LEAD_STEP_MINUTES) * _NATIVE_LEAD_STEP_MINUTES,
    )
    expected_leads = list(
        range(first_expected, end_lead + 1, _NATIVE_LEAD_STEP_MINUTES)
    )
    missing_leads = [lead for lead in expected_leads if lead not in available_leads]
    resolved_views, unavailable_views = _normalize_execution_views(views, selected_leads)
    estimated_static_outputs = sum(
        len(thresholds) if view["type"] == "forecast_observed_rate" else 1
        for view in resolved_views
        if view["type"] != "reflectivity_animation"
    )
    if estimated_static_outputs > MAX_VIEWS:
        raise DataContractError(
            f"the requested thresholds and views would render {estimated_static_outputs} "
            f"figures; at most {MAX_VIEWS} are allowed"
        )
    presentation_plan = _resolve_presentation_strategy(presentation, resolved_views)
    needed_frame_leads = {
        view["lead"]
        for view in resolved_views
        if view["type"] in {"reflectivity_comparison", "reflectivity_error"}
    }
    for view in resolved_views:
        if view["type"] == "reflectivity_animation":
            needed_frame_leads.update(view["leads"])

    per_lead = []
    frames: dict[int, VerificationFrame] = {}
    for lead_minutes in selected_leads:
        metadata, path = case_files[lead_minutes]
        forecast = read_swan_mosaic(path)
        expected_init = metadata.init_time
        expected_valid = metadata.valid_time
        expected_lead = metadata.lead_minutes
        if (
            forecast.init_time != expected_init
            or forecast.valid_time != expected_valid
            or forecast.lead_minutes != expected_lead
        ):
            raise DataContractError(f"forecast filename/header mismatch: {path}")
        obs = read_radar_observation(obs_files[expected_valid])
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
                "valid_time": expected_valid,
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
    animations = {}
    animation_metadata = {}
    for zero_index, view in enumerate(resolved_views):
        index = zero_index + 1
        key = f"view_{index:02d}"
        view_type = view["type"]
        if view_type == "score_time_series":
            if (
                presentation_plan["resolved_strategy"] == "linked_score_animation"
                and zero_index == presentation_plan["score_index"]
            ):
                continue
            metric = view["metric"]
            scores_by_threshold = {
                threshold: [entry["categorical"][threshold][metric] for entry in per_lead]
                for threshold in thresholds
            }
            figures[key] = plot_score_time_series(
                selected_leads, scores_by_threshold, metric
            )
        elif view_type == "forecast_observed_rate":
            for threshold in thresholds:
                threshold_key = (
                    key
                    if len(thresholds) == 1
                    else f"{key}_threshold_{float(threshold):g}dbz"
                )
                figures[threshold_key] = plot_forecast_observed_rate_trajectory(
                    selected_leads,
                    [entry["contingency"][threshold] for entry in per_lead],
                    threshold,
                    title="Forecast rate vs observed rate with marginal distributions",
                )
                figure_metadata[threshold_key] = {
                    "type": view_type,
                    "threshold_dbz": float(threshold),
                }
            continue
        elif view_type == "reflectivity_comparison":
            frame = frames[view["lead"]]
            figures[key] = plot_reflectivity_comparison(
                frame["forecast"],
                frame["obs"],
                title=f"Forecast vs observation — lead {view['lead']} min",
            )
        elif view_type == "reflectivity_error":
            frame = frames[view["lead"]]
            figures[key] = plot_reflectivity_error(
                frame["forecast"],
                frame["obs"],
                title=f"Forecast - observation — lead {view['lead']} min",
            )
        elif view_type == "reflectivity_animation":
            animation_key = f"animation_{len(animations) + 1:02d}"
            animation_frames = [frames[lead] for lead in view["leads"]]
            if (
                presentation_plan["resolved_strategy"] == "linked_score_animation"
                and zero_index == presentation_plan["animation_index"]
            ):
                metric = presentation_plan["metric"]
                scores_by_threshold = {
                    threshold: [entry["categorical"][threshold][metric] for entry in per_lead]
                    for threshold in thresholds
                }
                animations[animation_key] = render_linked_score_animation(
                    animation_frames,
                    lead_minutes=selected_leads,
                    scores_by_threshold=scores_by_threshold,
                    score_name=metric,
                    fps=view["fps"],
                )
                animation_metadata[animation_key] = {**view, "linked_metric": metric}
            else:
                animations[animation_key] = render_reflectivity_animation(
                    animation_frames, fps=view["fps"]
                )
                animation_metadata[animation_key] = view
        else:
            raise DataContractError(f"unsupported resolved view type: {view_type!r}")
        if view_type != "reflectivity_animation":
            figure_metadata[key] = view

    warnings = []
    if missing_leads:
        warnings.append(
            f"请求范围内缺少 {len(missing_leads)} 个精确匹配时效，完整列表见“时效覆盖”。"
        )
    if unavailable_views:
        warnings.append(
            f"有 {len(unavailable_views)} 张空间图没有对应的精确预报时效，已不生成这些图。"
        )
    if "fallback_reason" in presentation_plan:
        warnings.append(presentation_plan["fallback_reason"])

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
        "animations": animations,
        "animation_metadata": animation_metadata,
        "presentation": presentation_plan,
        "warnings": warnings,
    }

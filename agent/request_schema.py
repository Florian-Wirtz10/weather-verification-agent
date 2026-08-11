"""Validated, declarative requests for the verification agent.

The language model may propose a request, but only the normalized dictionary
returned by :func:`validate_request` is safe for the execution layer to use.
No value in this module is executable code and unsupported capabilities are
preserved in ``unmatched_requests`` instead of silently falling back.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from numbers import Real
from typing import Any

SCHEMA_VERSION = "1"

SUPPORTED_METRICS = ("csi", "pod", "far", "bias_score", "ets")
SUPPORTED_VIEW_TYPES = (
    "score_time_series",
    "forecast_observed_rate",
    "reflectivity_comparison",
    "reflectivity_error",
    "reflectivity_animation",
)
SUPPORTED_RATE_THRESHOLD_MODES = ("separate", "combined")
DEFAULT_THRESHOLDS_DBZ = (20.0, 35.0)
MAX_VIEWS = 12
MAX_SPATIAL_VIEWS = 6
MAX_ANIMATION_FRAMES = 30
SUPPORTED_LAYOUTS = (
    "auto",
    "single",
    "two_column",
    "grid_2x2",
    "report",
    "animation_focus",
)
SUPPORTED_PRESENTATION_STRATEGIES = (
    "auto",
    "separate_media",
    "linked_score_animation",
)

_SUPPORTED_INTENTS = {
    "verify_case",
    "show_capabilities",
    "explain_last_result",
    "out_of_scope",
    "unknown",
}
_SUPPORTED_INSIGHT_LENGTHS = {"brief", "detailed"}
_MAX_NOTES_LENGTH = 1_000
_MAX_UNMATCHED_TEXT_LENGTH = 300
_MAX_UNMATCHED_REQUESTS = 20
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_MISSING = object()


class RequestValidationError(ValueError):
    """Raised when a proposed request violates the executable schema."""


def default_request() -> dict[str, Any]:
    """Return a fresh, fully normalized default verification request."""
    return {
        "schema_version": SCHEMA_VERSION,
        "intent": "verify_case",
        "filters": {
            "thresholds_dbz": list(DEFAULT_THRESHOLDS_DBZ),
            "lead_range_minutes": {"start": 0, "end": 180},
            "requested_date": None,
        },
        "views": [{"type": "score_time_series", "metric": "csi"}],
        "presentation": {"layout": "auto", "strategy": "auto", "metric": None},
        "insight": {"enabled": True, "length": "detailed"},
        "unmatched_requests": [],
        "notes": "",
    }


def validate_request(parsed: object) -> dict[str, Any]:
    """Validate an untrusted parsed request and return a plain dictionary.

    Known fields with invalid values fail closed. Unknown view types and score
    metrics are capability requests rather than malformed data, so they are
    retained as human-readable ``unmatched_requests`` entries.
    """
    if not isinstance(parsed, dict):
        raise RequestValidationError("请求必须是 JSON 对象")

    schema_version = parsed.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        raise RequestValidationError(
            f"不支持 schema_version={schema_version!r}，当前版本为 {SCHEMA_VERSION}"
        )

    unmatched = _validate_unmatched(parsed.get("unmatched_requests", []))
    intent = _validate_intent(parsed.get("intent", "verify_case"), unmatched)
    filters = _validate_filters(parsed.get("filters", {}))
    views = _validate_views(parsed.get("views", []), unmatched)
    if len(views) > MAX_VIEWS:
        raise RequestValidationError(f"views 最多允许 {MAX_VIEWS} 项")
    spatial_count = sum(
        view["type"]
        in {"reflectivity_comparison", "reflectivity_error", "reflectivity_animation"}
        for view in views
    )
    if spatial_count > MAX_SPATIAL_VIEWS:
        raise RequestValidationError(f"空间图最多允许 {MAX_SPATIAL_VIEWS} 张")
    insight = _validate_insight(parsed.get("insight", {}))
    presentation = _validate_presentation(parsed.get("presentation", {}))
    notes = _validate_free_text(
        parsed.get("notes", ""),
        field="notes",
        maximum=_MAX_NOTES_LENGTH,
    )

    unmatched = _deduplicate(unmatched)
    if len(unmatched) > _MAX_UNMATCHED_REQUESTS:
        raise RequestValidationError(
            f"unmatched_requests 最多允许 {_MAX_UNMATCHED_REQUESTS} 项"
        )

    # A genuinely unspecified verification request gets one harmless default
    # view. An unsupported request must remain empty so the caller can refuse
    # it explicitly rather than surprising the user with unrelated charts.
    if intent == "verify_case" and not views and not unmatched:
        views = [{"type": "score_time_series", "metric": "csi"}]

    return {
        "schema_version": SCHEMA_VERSION,
        "intent": intent,
        "filters": filters,
        "views": views,
        "presentation": presentation,
        "insight": insight,
        "unmatched_requests": unmatched,
        "notes": notes,
    }


def _validate_intent(value: object, unmatched: list[str]) -> str:
    if not isinstance(value, str):
        raise RequestValidationError("intent 必须是字符串")
    normalized = value.strip().lower()
    if normalized in _SUPPORTED_INTENTS:
        return normalized
    _append_unmatched(unmatched, f"暂不支持请求意图：{normalized or '(空)'}")
    return "unknown"


def _validate_filters(value: object) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise RequestValidationError("filters 必须是 JSON 对象")

    thresholds = _validate_thresholds(value.get("thresholds_dbz", _MISSING))
    lead_range = _validate_lead_range(value.get("lead_range_minutes", _MISSING))
    requested_date = _validate_requested_date(value.get("requested_date", None))
    return {
        "thresholds_dbz": thresholds,
        "lead_range_minutes": lead_range,
        "requested_date": requested_date,
    }


def _validate_thresholds(value: object) -> list[float]:
    if value is _MISSING:
        return list(DEFAULT_THRESHOLDS_DBZ)
    if not isinstance(value, list) or not value:
        raise RequestValidationError("thresholds_dbz 必须是非空数字数组")

    thresholds: list[float] = []
    seen: set[float] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise RequestValidationError("thresholds_dbz 中的每一项都必须是数字")
        threshold = float(item)
        if not math.isfinite(threshold):
            raise RequestValidationError("thresholds_dbz 不允许 NaN 或无穷大")
        if not 0.0 <= threshold <= 75.0:
            raise RequestValidationError("thresholds_dbz 必须位于 0 到 75 dBZ")
        if threshold not in seen:
            seen.add(threshold)
            thresholds.append(threshold)

    if len(thresholds) > 8:
        raise RequestValidationError("thresholds_dbz 去重后最多允许 8 个阈值")
    return thresholds


def _validate_lead_range(value: object) -> dict[str, int]:
    if value is _MISSING:
        return {"start": 0, "end": 180}
    if not isinstance(value, dict):
        raise RequestValidationError("lead_range_minutes 必须是 JSON 对象")

    start = value.get("start", 0)
    end = value.get("end", 180)
    start = _validate_lead(start, field="lead_range_minutes.start")
    end = _validate_lead(end, field="lead_range_minutes.end")
    if start > end:
        raise RequestValidationError("lead_range_minutes.start 不能大于 end")
    return {"start": start, "end": end}


def _validate_requested_date(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _ISO_DATE_RE.fullmatch(value):
        raise RequestValidationError("requested_date 必须是 YYYY-MM-DD 格式")
    try:
        parsed_date = date.fromisoformat(value)
    except ValueError as exc:
        raise RequestValidationError("requested_date 不是有效日期") from exc
    return parsed_date.isoformat()


def _validate_views(value: object, unmatched: list[str]) -> list[dict[str, Any]]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise RequestValidationError("views 必须是数组")

    views: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_view in value:
        if not isinstance(raw_view, dict):
            _append_unmatched(unmatched, f"无法识别图表请求：{_safe_label(raw_view)}")
            continue

        raw_type = raw_view.get("type")
        if not isinstance(raw_type, str):
            _append_unmatched(unmatched, "图表请求缺少有效的 type")
            continue
        view_type = raw_type.strip().lower()
        if view_type not in SUPPORTED_VIEW_TYPES:
            _append_unmatched(unmatched, f"暂不支持图表：{view_type or '(空)'}")
            continue

        if view_type == "score_time_series":
            view = _validate_score_view(raw_view, unmatched)
        elif view_type == "forecast_observed_rate":
            view = _validate_rate_view(raw_view)
        elif view_type == "reflectivity_animation":
            view = _validate_animation_view(raw_view)
        else:
            view = _validate_reflectivity_view(raw_view, view_type)
        if view is None:
            continue

        _record_extra_view_parameters(raw_view, view_type, unmatched)
        fingerprint = json.dumps(view, sort_keys=True, ensure_ascii=True)
        if fingerprint not in seen:
            seen.add(fingerprint)
            views.append(view)
    return views


def _validate_score_view(
    raw_view: dict[str, Any], unmatched: list[str]
) -> dict[str, Any] | None:
    raw_metric = raw_view.get("metric")
    if not isinstance(raw_metric, str):
        _append_unmatched(unmatched, "评分曲线请求缺少有效的 metric")
        return None
    metric = raw_metric.strip().lower()
    if metric not in SUPPORTED_METRICS:
        _append_unmatched(unmatched, f"暂不支持评分指标：{metric or '(空)'}")
        return None
    return {"type": "score_time_series", "metric": metric}


def _validate_rate_view(raw_view: dict[str, Any]) -> dict[str, Any]:
    mode = raw_view.get("threshold_mode", "separate")
    if not isinstance(mode, str):
        raise RequestValidationError("forecast_observed_rate.threshold_mode 必须是字符串")
    mode = mode.strip().lower()
    if mode not in SUPPORTED_RATE_THRESHOLD_MODES:
        raise RequestValidationError(
            "forecast_observed_rate.threshold_mode 只能是 separate 或 combined"
        )
    return {"type": "forecast_observed_rate", "threshold_mode": mode}


def _validate_reflectivity_view(raw_view: dict[str, Any], view_type: str) -> dict[str, Any]:
    if "lead" not in raw_view:
        raise RequestValidationError(f"{view_type} 必须指定 lead")
    raw_lead = raw_view["lead"]
    if isinstance(raw_lead, str):
        lead = raw_lead.strip().lower()
        if lead not in {"first", "last"}:
            raise RequestValidationError(f"{view_type}.lead 只能是 first、last 或 0 到 180 的整数")
    else:
        lead = _validate_lead(raw_lead, field=f"{view_type}.lead")
    return {"type": view_type, "lead": lead}


def _validate_animation_view(raw_view: dict[str, Any]) -> dict[str, Any]:
    value = raw_view.get("lead_range_minutes")
    if not isinstance(value, dict):
        raise RequestValidationError("reflectivity_animation 必须指定 lead_range_minutes")
    start = _validate_lead(value.get("start"), field="reflectivity_animation.lead_range_minutes.start")
    end = _validate_lead(value.get("end"), field="reflectivity_animation.lead_range_minutes.end")
    if start > end:
        raise RequestValidationError("reflectivity_animation.lead_range_minutes.start 不能大于 end")
    fps = raw_view.get("fps", 2)
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 4:
        raise RequestValidationError("reflectivity_animation.fps 必须是 1 到 4 的整数")
    return {
        "type": "reflectivity_animation",
        "lead_range_minutes": {"start": start, "end": end},
        "fps": fps,
    }


def _record_extra_view_parameters(
    raw_view: dict[str, Any], view_type: str, unmatched: list[str]
) -> None:
    if view_type in {"score_time_series", "forecast_observed_rate"}:
        allowed = {"type", "metric"}
        if view_type == "forecast_observed_rate":
            allowed = {"type", "threshold_mode"}
    elif view_type == "reflectivity_animation":
        allowed = {"type", "lead_range_minutes", "fps"}
    else:
        allowed = {"type", "lead"}
    extras = sorted(str(key) for key in raw_view if key not in allowed)
    if extras:
        _append_unmatched(
            unmatched,
            f"图表 {view_type} 暂不支持参数：{', '.join(extras)}",
        )


def _validate_insight(value: object) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise RequestValidationError("insight 必须是 JSON 对象")
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise RequestValidationError("insight.enabled 必须是布尔值")
    length = value.get("length", "brief")
    if not isinstance(length, str) or length.strip().lower() not in _SUPPORTED_INSIGHT_LENGTHS:
        raise RequestValidationError("insight.length 只能是 brief 或 detailed")
    return {"enabled": enabled, "length": length.strip().lower()}


def _validate_presentation(value: object) -> dict[str, str | None]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise RequestValidationError("presentation 必须是 JSON 对象")
    layout = value.get("layout", "auto")
    if not isinstance(layout, str) or layout.strip().lower() not in SUPPORTED_LAYOUTS:
        choices = "、".join(SUPPORTED_LAYOUTS)
        raise RequestValidationError(f"presentation.layout 只能是 {choices}")
    strategy = value.get("strategy", "auto")
    if not isinstance(strategy, str) or strategy.strip().lower() not in SUPPORTED_PRESENTATION_STRATEGIES:
        choices = "、".join(SUPPORTED_PRESENTATION_STRATEGIES)
        raise RequestValidationError(f"presentation.strategy 只能是 {choices}")
    strategy = strategy.strip().lower()
    metric = value.get("metric")
    if strategy == "linked_score_animation":
        if not isinstance(metric, str) or metric.strip().lower() not in SUPPORTED_METRICS:
            raise RequestValidationError("linked_score_animation 必须指定受支持的 metric")
        metric = metric.strip().lower()
    elif metric is not None:
        raise RequestValidationError("presentation.metric 只用于 linked_score_animation")
    return {"layout": layout.strip().lower(), "strategy": strategy, "metric": metric}


def _validate_unmatched(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RequestValidationError("unmatched_requests 必须是字符串数组")
    if len(value) > _MAX_UNMATCHED_REQUESTS:
        raise RequestValidationError(
            f"unmatched_requests 最多允许 {_MAX_UNMATCHED_REQUESTS} 项"
        )
    return [
        _validate_free_text(item, field="unmatched_requests", maximum=_MAX_UNMATCHED_TEXT_LENGTH)
        for item in value
    ]


def _validate_free_text(value: object, *, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise RequestValidationError(f"{field} 必须是字符串")
    text = value.strip()
    if len(text) > maximum:
        raise RequestValidationError(f"{field} 最长允许 {maximum} 个字符")
    return text


def _validate_lead(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestValidationError(f"{field} 必须是整数")
    if not 0 <= value <= 180:
        raise RequestValidationError(f"{field} 必须位于 0 到 180 分钟")
    return value


def _safe_label(value: object) -> str:
    try:
        label = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        label = type(value).__name__
    if len(label) > 120:
        return f"{label[:117]}..."
    return label


def _append_unmatched(unmatched: list[str], message: str) -> None:
    if len(message) > _MAX_UNMATCHED_TEXT_LENGTH:
        message = f"{message[: _MAX_UNMATCHED_TEXT_LENGTH - 3]}..."
    unmatched.append(message)


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))

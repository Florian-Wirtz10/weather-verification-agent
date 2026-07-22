"""In-memory, browser-safe downloads for one real-case verification result."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from numbers import Real
import re
from typing import Any, Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


MAX_EXPORT_IMAGES = 12
MAX_EXPORT_ANIMATIONS = 3
MAX_EXPORT_BYTES = 100 * 1024 * 1024
_ALLOWED_VIEW_TYPES = {
    "score_time_series",
    "forecast_observed_rate",
    "reflectivity_comparison",
    "reflectivity_error",
    "reflectivity_animation",
}
_ALLOWED_METRICS = {"csi", "pod", "far", "bias_score", "ets"}
_METRIC_ROW_COLUMNS = (
    "lead_minutes",
    "valid_time_utc",
    "threshold_dbz",
    "csi",
    "pod",
    "far",
    "bias_score",
    "ets",
    "hits",
    "misses",
    "false_alarms",
    "correct_negatives",
    "valid_count",
    "invalid_count",
)

_SAFE_CONTEXT_KEYS = (
    "case_id",
    "product",
    "variable",
    "units",
    "init_time",
    "bbox",
    "temporal_matching",
    "spatial_alignment",
)
_SAFE_COVERAGE_KEYS = (
    "requested_range",
    "native_step_minutes",
    "available_leads",
    "selected_leads",
    "missing_leads",
    "unavailable_views",
)


def figure_to_png(figure, *, dpi=180) -> bytes:
    """Serialize a Matplotlib figure without writing to the workspace."""
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "Weather Verification Agent"},
    )
    return buffer.getvalue()


def describe_view(view: Mapping[str, Any]) -> str:
    """Return a short Chinese label for a validated, resolved view."""
    view_type = view.get("type")
    if view_type == "score_time_series":
        return f"{str(view.get('metric', '')).upper()} 随预报时效变化曲线"
    if view_type == "forecast_observed_rate":
        if view.get("threshold_mode") == "combined":
            thresholds = view.get("thresholds_dbz")
            if isinstance(thresholds, Sequence) and not isinstance(
                thresholds, (str, bytes)
            ):
                labels = "、".join(f"{float(value):g}" for value in thresholds)
                return f"{labels} dBZ 预报率—观测率阈值对比"
            return "预报率—观测率阈值对比"
        threshold = view.get("threshold_dbz")
        if isinstance(threshold, Real) and not isinstance(threshold, bool):
            return f"{float(threshold):g} dBZ 预报率—观测率时效轨迹"
        return "预报率—观测率时效轨迹"
    lead = view.get("lead")
    if view_type == "reflectivity_comparison":
        return f"{lead} 分钟预报与观测反射率对比"
    if view_type == "reflectivity_error":
        return f"{lead} 分钟预报减观测反射率误差"
    if view_type == "reflectivity_animation":
        lead_range = view.get("lead_range_minutes")
        if isinstance(lead_range, Mapping):
            linked_metric = view.get("linked_metric")
            if isinstance(linked_metric, str) and linked_metric in _ALLOWED_METRICS:
                return (
                    f"{lead_range.get('start')}—{lead_range.get('end')} 分钟"
                    f"预报观测与 {linked_metric.upper()} 联动动画"
                )
            return (
                f"{lead_range.get('start')}—{lead_range.get('end')} 分钟"
                "预报与观测反射率动画"
            )
        return "预报与观测反射率动画"
    return "检验图表"


def build_case_download_artifacts(
    *,
    result: Mapping[str, Any],
    request: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    facts: Mapping[str, Any],
    deterministic_summary: str,
    ai_insight: str | None,
    execution_status: str = "ready",
    rendered_images: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build individual files plus a ZIP bundle entirely in memory."""
    if execution_status not in {"ready", "partial"}:
        raise ValueError("download execution status must be ready or partial")
    context = _safe_pick(result.get("context"), _SAFE_CONTEXT_KEYS)
    coverage = _safe_pick(result.get("coverage"), _SAFE_COVERAGE_KEYS)
    safe_request = _safe_request(request)
    unmatched = request.get("unmatched_requests")
    unmatched_count = (
        len(unmatched)
        if isinstance(unmatched, Sequence) and not isinstance(unmatched, (str, bytes))
        else 0
    )
    base_name = _artifact_base_name(context, coverage, request, execution_status)

    images: dict[str, dict[str, Any]] = {}
    animations: dict[str, dict[str, Any]] = {}
    figures = result.get("figures")
    figure_metadata = result.get("figure_metadata")
    if not isinstance(figure_metadata, Mapping):
        raise ValueError("figure_metadata must be a mapping")
    if len(figure_metadata) > MAX_EXPORT_IMAGES:
        raise ValueError(f"at most {MAX_EXPORT_IMAGES} images can be exported")
    if rendered_images is not None:
        if set(rendered_images) != set(figure_metadata):
            raise ValueError("rendered image keys must exactly match figure metadata")
        for key, item in rendered_images.items():
            if not isinstance(item, Mapping):
                raise ValueError("rendered image entries must be mappings")
            view = _validated_view(figure_metadata[key])
            data = item.get("data")
            if not isinstance(data, bytes) or not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("rendered image data must be PNG bytes")
            images[str(key)] = {
                "file_name": f"{base_name}_{_figure_filename(len(images) + 1, view)}",
                "label": describe_view(view),
                "data": data,
            }
    elif isinstance(figures, Mapping):
        if set(figures) != set(figure_metadata):
            raise ValueError("figure keys must exactly match figure metadata")
        for index, (key, figure) in enumerate(figures.items(), start=1):
            view = _validated_view(figure_metadata[key])
            file_name = f"{base_name}_{_figure_filename(index, view)}"
            images[str(key)] = {
                "file_name": file_name,
                "label": describe_view(view),
                "data": figure_to_png(figure),
            }
    else:
        raise ValueError("figures must be a mapping when rendered_images is absent")

    raw_animations = result.get("animations", {})
    raw_animation_metadata = result.get("animation_metadata", {})
    if not isinstance(raw_animations, Mapping) or not isinstance(raw_animation_metadata, Mapping):
        raise ValueError("animations and animation_metadata must be mappings")
    if set(raw_animations) != set(raw_animation_metadata):
        raise ValueError("animation keys must exactly match animation metadata")
    if len(raw_animations) > MAX_EXPORT_ANIMATIONS:
        raise ValueError(f"at most {MAX_EXPORT_ANIMATIONS} animations can be exported")
    for index, (key, data) in enumerate(raw_animations.items(), start=1):
        view = _validated_view(raw_animation_metadata[key])
        if not isinstance(data, bytes) or not data.startswith((b"GIF87a", b"GIF89a")):
            raise ValueError("animation data must be GIF bytes")
        animations[str(key)] = {
            "file_name": f"{base_name}_{index:02d}_{_safe_filename(view['type'])}.gif",
            "label": describe_view(view),
            "data": data,
        }

    metrics_name = f"{base_name}_metrics.csv"
    metrics_data = (
        _metric_frame(metric_rows, context)
        .to_csv(index=False, na_rep="", float_format="%.6f", lineterminator="\n")
        .encode("utf-8-sig")
    )
    summary_name = f"{base_name}_summary.md"
    summary_data = _summary_markdown(
        context=context,
        coverage=coverage,
        request=safe_request,
        execution_status=execution_status,
        unmatched_count=unmatched_count,
        deterministic_summary=deterministic_summary,
        ai_insight=ai_insight,
    ).encode("utf-8")
    request_data = _json_bytes(safe_request)
    facts_data = _json_bytes(_safe_facts(facts))
    manifest_data = _json_bytes(
        {
            "execution": {
                "status": execution_status,
                "unmatched_request_count": unmatched_count,
            },
            "context": context,
            "coverage": coverage,
            "figures": [
                {
                    "file_name": item["file_name"],
                    "label": item["label"],
                    "sha256": hashlib.sha256(item["data"]).hexdigest(),
                }
                for item in images.values()
            ]
            + [
                {
                    "file_name": item["file_name"],
                    "label": item["label"],
                    "sha256": hashlib.sha256(item["data"]).hexdigest(),
                }
                for item in animations.values()
            ],
        }
    )

    total_bytes = (
        sum(len(item["data"]) for item in images.values())
        + sum(len(item["data"]) for item in animations.values())
        + len(metrics_data)
        + len(summary_data)
        + len(request_data)
        + len(facts_data)
        + len(manifest_data)
    )
    if total_bytes > MAX_EXPORT_BYTES:
        raise ValueError("download artifacts exceed the 100 MiB safety limit")

    bundle_buffer = BytesIO()
    with ZipFile(bundle_buffer, "w", compression=ZIP_DEFLATED) as archive:
        for item in images.values():
            archive.writestr(f"figures/{item['file_name']}", item["data"])
        for item in animations.values():
            archive.writestr(f"animations/{item['file_name']}", item["data"])
        archive.writestr(f"tables/{metrics_name}", metrics_data)
        archive.writestr(f"reports/{summary_name}", summary_data)
        archive.writestr("metadata/execution_request.json", request_data)
        archive.writestr("metadata/facts.json", facts_data)
        archive.writestr("metadata/manifest.json", manifest_data)

    return {
        "base_name": base_name,
        "images": images,
        "animations": animations,
        "metrics": {"file_name": metrics_name, "data": metrics_data},
        "summary": {"file_name": summary_name, "data": summary_data},
        "bundle": {
            "file_name": f"{base_name}_bundle.zip",
            "data": bundle_buffer.getvalue(),
        },
    }


def _figure_filename(index: int, view: Mapping[str, Any]) -> str:
    view_type = str(view.get("type") or "figure")
    if view_type == "score_time_series":
        detail = str(view.get("metric") or "score")
    elif view_type == "forecast_observed_rate":
        if view.get("threshold_mode") == "combined":
            detail = "threshold-rate-comparison"
        else:
            threshold = view.get("threshold_dbz")
            detail = (
                f"{float(threshold):g}dbz-rate-trajectory"
                if isinstance(threshold, Real) and not isinstance(threshold, bool)
                else "rate-trajectory"
            )
    else:
        lead = view.get("lead")
        detail = f"{int(lead):03d}min" if isinstance(lead, int) else "lead"
    return f"{index:02d}_{_safe_filename(view_type)}_{_safe_filename(detail)}.png"


def _artifact_base_name(context, coverage, request, execution_status: str) -> str:
    parts = [str(context.get("case_id") or "fenglei-verification")]
    selected_leads = coverage.get("selected_leads")
    if (
        isinstance(selected_leads, Sequence)
        and not isinstance(selected_leads, (str, bytes))
        and selected_leads
    ):
        parts.append(f"lead{int(selected_leads[0]):03d}-{int(selected_leads[-1]):03d}")
    filters = request.get("filters") if isinstance(request, Mapping) else None
    thresholds = filters.get("thresholds_dbz") if isinstance(filters, Mapping) else None
    if isinstance(thresholds, Sequence) and not isinstance(thresholds, (str, bytes)):
        threshold_text = "-".join(_format_threshold(value) for value in thresholds)
        if threshold_text:
            parts.append(f"thr{threshold_text}dbz")
    parts.append(execution_status)
    request_fingerprint = hashlib.sha256(
        _json_bytes(
            {
                "bbox": context.get("bbox"),
                "request": _safe_request(request),
            }
        )
    ).hexdigest()[:8]
    parts.append(f"req{request_fingerprint}")
    return _safe_filename("_".join(parts))


def _format_threshold(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _safe_filename(str(value))
    text = str(int(number)) if number.is_integer() else f"{number:g}"
    return text.replace(".", "p")


def _validated_view(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("figure metadata entries must be mappings")
    view_type = value.get("type")
    if view_type not in _ALLOWED_VIEW_TYPES:
        raise ValueError(f"unsupported export view type: {view_type!r}")
    if view_type == "score_time_series":
        metric = value.get("metric")
        if metric not in _ALLOWED_METRICS:
            raise ValueError(f"unsupported export metric: {metric!r}")
        return {"type": view_type, "metric": metric}
    if view_type == "forecast_observed_rate":
        if value.get("threshold_mode") == "combined":
            raw_thresholds = value.get("thresholds_dbz")
            if (
                not isinstance(raw_thresholds, Sequence)
                or isinstance(raw_thresholds, (str, bytes))
                or not raw_thresholds
                or len(raw_thresholds) > 8
            ):
                raise ValueError("invalid export comparison thresholds")
            thresholds = []
            for threshold in raw_thresholds:
                if (
                    isinstance(threshold, bool)
                    or not isinstance(threshold, Real)
                    or not 0 <= float(threshold) <= 75
                ):
                    raise ValueError(
                        f"invalid export comparison threshold: {threshold!r}"
                    )
                numeric = float(threshold)
                if numeric not in thresholds:
                    thresholds.append(numeric)
            return {
                "type": view_type,
                "threshold_mode": "combined",
                "thresholds_dbz": thresholds,
            }
        threshold = value.get("threshold_dbz")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not 0 <= float(threshold) <= 75
        ):
            raise ValueError(f"invalid export threshold: {threshold!r}")
        return {
            "type": view_type,
            "threshold_mode": "separate",
            "threshold_dbz": float(threshold),
        }
    if view_type == "reflectivity_animation":
        lead_range = value.get("lead_range_minutes")
        fps = value.get("fps")
        if not isinstance(lead_range, Mapping):
            raise ValueError("invalid animation lead range")
        start, end = lead_range.get("start"), lead_range.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start <= end <= 180
        ):
            raise ValueError("invalid animation lead range")
        if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 4:
            raise ValueError("invalid animation fps")
        linked_metric = value.get("linked_metric")
        if linked_metric is not None and linked_metric not in _ALLOWED_METRICS:
            raise ValueError("invalid linked animation metric")
        validated = {
            "type": view_type,
            "lead_range_minutes": {"start": start, "end": end},
            "fps": fps,
        }
        if linked_metric is not None:
            validated["linked_metric"] = linked_metric
        return validated
    lead = value.get("lead")
    if isinstance(lead, bool) or not isinstance(lead, int) or not 0 <= lead <= 180:
        raise ValueError(f"invalid export lead: {lead!r}")
    return {"type": view_type, "lead": lead}


def _summary_markdown(
    *,
    context,
    coverage,
    request,
    execution_status,
    unmatched_count,
    deterministic_summary,
    ai_insight,
) -> str:
    filters = request.get("filters", {})
    insight = request.get("insight", {})
    requested_metrics = [
        str(view["metric"]).upper()
        for view in request.get("views", [])
        if view.get("type") == "score_time_series" and "metric" in view
    ]
    requested_metric_text = "、".join(requested_metrics) or "无（仅请求空间图）"
    threshold_text = "、".join(
        _format_threshold(value) for value in filters.get("thresholds_dbz", [])
    )
    status_text = "完整执行" if execution_status == "ready" else "部分执行"
    lines = [
        "# 风雷预报检验结果",
        "",
        "## 检验范围",
        "",
        f"- 执行状态：{status_text}",
        f"- 个例：{context.get('case_id', '未提供')}",
        f"- 产品：{context.get('product', '未提供')}",
        f"- 起报时间：{_utc_text(context.get('init_time'))}",
        f"- 区域：[lat_min, lat_max, lon_min, lon_max] = {context.get('bbox')}",
        f"- 请求阈值：{threshold_text or '未提供'} dBZ",
        f"- 请求评分指标：{requested_metric_text}",
        f"- 实际时效：{coverage.get('selected_leads', [])} 分钟",
        f"- 缺失时效：{coverage.get('missing_leads', [])} 分钟",
        f"- 未生成的空间图：{coverage.get('unavailable_views', []) or '无'}",
        f"- 时间配对：{context.get('temporal_matching', '未提供')}",
        f"- 空间对齐：{context.get('spatial_alignment', '未提供')}",
        "",
        "## 确定性事实摘要",
        "",
        deterministic_summary,
    ]
    lines.extend(["", "## AI 解读", ""])
    if ai_insight:
        lines.append(ai_insight)
    elif insight.get("enabled"):
        lines.append("本结果包未包含 AI 解读；确定性指标、图表和摘要不受影响。")
    else:
        lines.append("本次请求未启用 AI 解读。")
    if execution_status == "partial" and unmatched_count:
        lines.extend(
            [
                "",
                f"> 本次有 {unmatched_count} 项请求超出当前能力，结果包只包含已支持并实际执行的部分。",
            ]
        )
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "本报告针对单个起报个例，不能推广为模型整体能力结论。",
        ]
    )
    return "\n".join(lines) + "\n"


def _metric_frame(
    rows: Sequence[Mapping[str, Any]], context: Mapping[str, Any]
) -> pd.DataFrame:
    """Create a standalone, provenance-bearing CSV from whitelisted row fields."""
    safe_rows = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("metric rows must be mappings")
        selected = {key: row[key] for key in _METRIC_ROW_COLUMNS if key in row}
        if "valid_time_utc" in selected:
            selected["valid_time_utc"] = _utc_text(selected["valid_time_utc"])
        safe_rows.append(selected)

    frame = pd.DataFrame(safe_rows)
    bbox = context.get("bbox")
    coordinates = (
        list(bbox)
        if isinstance(bbox, Sequence)
        and not isinstance(bbox, (str, bytes))
        and len(bbox) == 4
        else [None] * 4
    )
    provenance = (
        ("case_id", _spreadsheet_text(context.get("case_id", ""))),
        ("init_time_utc", _utc_text(context.get("init_time"))),
        ("lat_min", coordinates[0]),
        ("lat_max", coordinates[1]),
        ("lon_min", coordinates[2]),
        ("lon_max", coordinates[3]),
    )
    for column, value in reversed(provenance):
        frame.insert(0, column, value)
    return frame


def _safe_pick(value: Any, keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {key: value[key] for key in keys if key in value}


def _safe_request(value: Mapping[str, Any]) -> dict[str, Any]:
    filters = value.get("filters")
    insight = value.get("insight")
    views = value.get("views")
    return {
        "schema_version": value.get("schema_version"),
        "intent": value.get("intent"),
        "filters": _safe_pick(
            filters,
            ("thresholds_dbz", "lead_range_minutes", "requested_date"),
        ),
        "views": [
            _safe_pick(
                view,
                (
                    "type",
                    "metric",
                    "lead",
                    "lead_range_minutes",
                    "fps",
                    "threshold_mode",
                ),
            )
            for view in views
            if isinstance(view, Mapping)
        ]
        if isinstance(views, Sequence) and not isinstance(views, (str, bytes))
        else [],
        "insight": _safe_pick(insight, ("enabled", "length")),
        "presentation": _safe_pick(value.get("presentation"), ("layout", "strategy", "metric")),
    }


def _safe_facts(value: Mapping[str, Any]) -> dict[str, Any]:
    selected = _safe_pick(
        value,
        (
            "context",
            "coverage",
            "requested_metrics",
            "thresholds",
            "rate_diagnostics",
            "undefined_values",
            "limitations",
            "figure_catalog",
        ),
    )
    return _drop_sensitive(selected)


def _drop_sensitive(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned = {}
        for key, nested in value.items():
            name = str(key).lower()
            if (
                name in {"api_key", "secret", "token", "path", "file", "directory"}
                or name.endswith(("_path", "_file", "_dir", "_directory"))
            ):
                continue
            cleaned[str(key)] = _drop_sensitive(nested)
        return cleaned
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_drop_sensitive(item) for item in value]
    return value


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned[:120] or "artifact"


def _utc_text(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.isoformat(timespec="seconds") + "Z"
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)


def _spreadsheet_text(value: Any) -> str:
    text = str(value)
    return "'" + text if text.startswith(("=", "+", "-", "@")) else text


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(value) if isinstance(value, int) else number
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    raise TypeError(f"unsupported export metadata type: {type(value).__name__}")

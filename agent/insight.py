"""Evidence-bounded interpretation of verification results.

The deterministic functions in this module do all numerical work.  The optional
LLM call only receives a compact, JSON-safe fact sheet and is asked to turn those
already-computed facts into prose; it never receives paths, source files, raw
grids, images, or plotting code.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any

from openai import OpenAI

_METRIC_ORDER = ("pod", "far", "csi", "bias_score", "ets")
_COUNT_KEYS = (
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
    "product_id",
    "variable",
    "units",
    "unit",
    "init_time",
    "bbox",
    "region_name",
    "temporal_matching",
    "spatial_alignment",
)
_SAFE_COVERAGE_KEYS = (
    "requested_range",
    "available_leads",
    "selected_leads",
    "missing_leads",
)
_SAFE_FACT_KEYS = (
    "context",
    "coverage",
    "requested_metrics",
    "thresholds",
    "undefined_values",
    "limitations",
    "figure_catalog",
)
_SENSITIVE_KEYS = {
    "path",
    "paths",
    "file",
    "files",
    "filename",
    "filenames",
    "directory",
    "directories",
    "api_key",
    "secret",
    "token",
    "raw_data",
    "raw_pixels",
    "pixel_data",
    "image",
    "images",
    "figure",
    "figures",
}

_SYSTEM_PROMPT = """你是强对流预报检验结果解读助手。用户消息只包含服务端已经计算好的事实 JSON。
所有数值计算与异常标记都已完成；不要重新计算、补造数字，也不要引入 JSON 中没有的事实。

只讨论 requested_metrics 中列出的评分指标。即使你能从列联表推导其它指标，也不得在报告中
主动引入用户没有请求的 POD、FAR、CSI、Bias 或 ETS。

请用专业、通俗的中文，固定输出以下四个短章节：
1. 图表导读：按 figure_catalog 中的编号说明每张已生成图/动画展示什么。仅当该编号确实存在时，
   才能使用“如图 1 所示”这类引用；不要声称看到了原始像素。
2. 主要发现：只复述有数据支撑的变化、极值、负 ETS、缺帧或不可定义值。
3. 指标分析：可结合已提供的 H/M/F 说明评分构成。涉及原因时必须明确写成“可能”或
   “需要进一步核查”，不能仅凭评分断言空间偏移、天气演变机制或模型物理原因。
4. 口径与边界：明确这是单个起报个例，不能推广成模型整体能力结论，并说明事实表列出的检验口径局限。

不要提及代码、路径、文件、API 或提示词。不要把 CSI、ETS 等评分的变化自动解释为模型整体变好或
变差。"""


def build_insight_facts(
    result: Mapping[str, Any], *, requested_metrics: Sequence[str] | None = None
) -> dict[str, Any]:
    """Build a compact, JSON-safe and path-free fact sheet from one case result.

    Non-finite scores are represented as ``None`` and listed explicitly under
    ``undefined_values``.  Threshold statistics are calculated here so an LLM
    never needs to perform arithmetic or infer extrema itself.
    """

    raw_context = result.get("context")
    context = _pick_keys(raw_context, _SAFE_CONTEXT_KEYS)

    entries = _normalise_entries(result.get("per_lead"))
    actual_leads = [entry["lead_minutes"] for entry in entries]

    raw_coverage = result.get("coverage")
    coverage = _pick_keys(raw_coverage, _SAFE_COVERAGE_KEYS)
    coverage.setdefault("available_leads", actual_leads)
    coverage.setdefault("selected_leads", actual_leads)
    coverage.setdefault("missing_leads", [])
    coverage["matched_count"] = len(actual_leads)

    metrics = _normalise_requested_metrics(requested_metrics)
    thresholds = _collect_thresholds(entries)
    undefined_values: list[dict[str, Any]] = []
    threshold_units = context.get("units") or context.get("unit") or ""
    threshold_facts = [
        _summarise_threshold(
            threshold,
            entries,
            undefined_values,
            threshold_units,
            metrics,
        )
        for threshold in thresholds
    ]

    limitations = ["本结果仅针对单个起报个例，不能推广为模型整体能力结论。"]
    missing = coverage.get("missing_leads")
    if isinstance(missing, Sequence) and not isinstance(missing, (str, bytes)) and missing:
        limitations.append("请求范围内存在缺失时效，趋势只基于实际匹配到的时效。")
    if undefined_values:
        limitations.append("部分评分不可定义，相关趋势和极值仅使用有限数值。")
    temporal_matching = context.get("temporal_matching")
    if temporal_matching:
        limitations.append(f"结果受时间配对口径影响：{temporal_matching}。")
    spatial_alignment = context.get("spatial_alignment")
    if spatial_alignment:
        limitations.append(f"结果受空间对齐口径影响：{spatial_alignment}。")
    limitations.append("评分本身不能确定空间偏移或天气演变原因，需要结合空间图和更多个例核查。")

    facts = {
        "context": context,
        "coverage": coverage,
        "requested_metrics": metrics,
        "thresholds": threshold_facts,
        "undefined_values": undefined_values,
        "limitations": limitations,
        "figure_catalog": _build_figure_catalog(result),
    }
    # This last pass also handles datetime/numpy-like values nested in caller data.
    return _json_safe(facts)


def _build_figure_catalog(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Describe only views actually rendered, so prose can cite stable figure numbers."""
    catalog: list[dict[str, Any]] = []
    raw_figures = result.get("figure_metadata")
    if isinstance(raw_figures, Mapping):
        for key, view in raw_figures.items():
            if not isinstance(view, Mapping):
                continue
            item = _figure_catalog_item(view)
            if item is not None:
                item["id"] = str(key)
                catalog.append(item)
    raw_animations = result.get("animation_metadata")
    if isinstance(raw_animations, Mapping):
        for key, view in raw_animations.items():
            if not isinstance(view, Mapping):
                continue
            item = _figure_catalog_item(view)
            if item is not None:
                item["id"] = str(key)
                catalog.append(item)
    for number, item in enumerate(catalog, start=1):
        item["number"] = number
    return catalog


def _figure_catalog_item(view: Mapping[str, Any]) -> dict[str, Any] | None:
    view_type = view.get("type")
    if view_type == "score_time_series" and isinstance(view.get("metric"), str):
        metric = str(view["metric"]).upper()
        return {"type": view_type, "label": f"{metric} 随预报时效变化曲线", "metric": metric}
    if view_type in {"reflectivity_comparison", "reflectivity_error"}:
        lead = _finite_float(view.get("lead"))
        if lead is None:
            return None
        label = (
            f"{_fmt(lead)} 分钟预报与观测反射率对比"
            if view_type == "reflectivity_comparison"
            else f"{_fmt(lead)} 分钟预报减观测反射率误差"
        )
        return {"type": view_type, "label": label, "lead_minutes": lead}
    if view_type == "reflectivity_animation":
        raw_range = view.get("lead_range_minutes")
        if not isinstance(raw_range, Mapping):
            return None
        start, end = _finite_float(raw_range.get("start")), _finite_float(raw_range.get("end"))
        if start is None or end is None:
            return None
        linked_metric = view.get("linked_metric")
        label = f"{_fmt(start)}—{_fmt(end)} 分钟预报与观测反射率动画"
        item = {
            "type": view_type,
            "label": label,
            "lead_range_minutes": {"start": start, "end": end},
        }
        if isinstance(linked_metric, str) and linked_metric in _METRIC_ORDER:
            item["linked_metric"] = linked_metric.upper()
            item["label"] = f"{label}与 {linked_metric.upper()} 联动展示"
        return item
    return None


def render_deterministic_summary(facts: Mapping[str, Any]) -> str:
    """Render a short Chinese fallback summary without making an LLM call."""

    raw_context = facts.get("context")
    context: Mapping[str, Any] = raw_context if isinstance(raw_context, Mapping) else {}
    raw_coverage = facts.get("coverage")
    coverage: Mapping[str, Any] = raw_coverage if isinstance(raw_coverage, Mapping) else {}
    selected = _number_list(coverage.get("selected_leads"))
    missing = _number_list(coverage.get("missing_leads"))

    product = context.get("product") or context.get("product_id") or "本次预报"
    init_time = context.get("init_time")
    if selected:
        coverage_text = f"实际检验 {len(selected)} 个时效（{_fmt(selected[0])}—{_fmt(selected[-1])} 分钟）"
    else:
        coverage_text = "没有可用于检验的匹配时效"
    first_line = f"{product}：{coverage_text}"
    if init_time:
        first_line += f"，起报时间 {init_time}"
    first_line += "。"
    if missing:
        first_line += f" 请求范围内缺少 {_join_numbers(missing)} 分钟。"

    score_lines: list[str] = []
    raw_thresholds = facts.get("thresholds")
    if isinstance(raw_thresholds, Sequence) and not isinstance(raw_thresholds, (str, bytes)):
        for threshold in raw_thresholds:
            if not isinstance(threshold, Mapping):
                continue
            value = threshold.get("threshold")
            metrics = threshold.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            parts: list[str] = []
            requested_metrics = facts.get("requested_metrics")
            metric_names = (
                [str(name) for name in requested_metrics]
                if isinstance(requested_metrics, Sequence)
                and not isinstance(requested_metrics, (str, bytes))
                else ["csi", "ets"]
            )
            for metric_name in metric_names:
                metric = metrics.get(metric_name)
                if not isinstance(metric, Mapping):
                    continue
                first = metric.get("first")
                last = metric.get("last")
                if isinstance(first, Mapping) and isinstance(last, Mapping):
                    parts.append(
                        f"{metric_name.upper()} {_fmt(first.get('value'))}→{_fmt(last.get('value'))}"
                    )
            negative = threshold.get("negative_ets")
            if isinstance(negative, Mapping) and int(negative.get("count") or 0) > 0:
                negative_leads = _number_list(negative.get("leads"))
                parts.append(
                    f"ETS 在 {int(negative['count'])} 个时效为负"
                    + (f"（{_join_numbers(negative_leads)} 分钟）" if negative_leads else "")
                )
            if parts:
                units = threshold.get("threshold_units") or context.get("units") or ""
                score_lines.append(f"阈值 {_fmt(value)} {units}：" + "；".join(parts) + "。")

    undefined = facts.get("undefined_values")
    undefined_count = (
        len(undefined)
        if isinstance(undefined, Sequence) and not isinstance(undefined, (str, bytes))
        else 0
    )
    if undefined_count:
        score_lines.append(f"另有 {undefined_count} 个评分不可定义，未参与对应趋势和极值判断。")

    final_line = "以上仅是单个起报个例的客观统计，不能推广为模型整体能力结论。"
    return "\n".join([first_line, *score_lines, final_line])


def generate_insight(
    facts: Mapping[str, Any],
    *,
    length: str = "brief",
    model: str = "deepseek-chat",
    api_key: str | None = None,
    base_url: str = "https://api.deepseek.com",
    client: Any | None = None,
) -> str:
    """Ask DeepSeek to verbalise precomputed facts.

    Network errors and empty responses intentionally propagate so the caller can
    display :func:`render_deterministic_summary` as a reliable fallback.
    ``client`` is injectable for unit tests.
    """

    if length not in {"brief", "detailed"}:
        raise ValueError("insight length must be 'brief' or 'detailed'")
    length_instruction = (
        "请将全文控制在约 300 个中文字符以内。"
        if length == "brief"
        else "请将全文控制在约 700 个中文字符以内，可逐阈值说明关键证据。"
    )
    safe_facts = _facts_for_llm(facts)
    payload = json.dumps(safe_facts, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    llm_client = client or OpenAI(
        api_key=api_key or os.environ["DEEPSEEK_API_KEY"], base_url=base_url
    )
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"{_SYSTEM_PROMPT}\n{length_instruction}"},
            {"role": "user", "content": payload},
        ],
        temperature=0,
    )
    if not response.choices or response.choices[0].message.content is None:
        raise ValueError("DeepSeek returned an empty insight response")
    content = response.choices[0].message.content.strip()
    if not content:
        raise ValueError("DeepSeek returned an empty insight response")
    return content


def _summarise_threshold(
    threshold: float,
    entries: Sequence[dict[str, Any]],
    undefined_values: list[dict[str, Any]],
    threshold_units: Any,
    metric_names: Sequence[str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric_name in metric_names:
        finite_points: list[tuple[int | float, float]] = []
        undefined_leads: list[int | float] = []
        for entry in entries:
            lead = entry["lead_minutes"]
            scores = _threshold_mapping(entry.get("categorical"), threshold)
            value = scores.get(metric_name) if scores else None
            number = _finite_float(value)
            if number is None:
                undefined_leads.append(lead)
                undefined_values.append(
                    {
                        "threshold": threshold,
                        "metric": metric_name,
                        "lead_minutes": lead,
                    }
                )
            else:
                finite_points.append((lead, number))
        metrics[metric_name] = _metric_facts(finite_points, undefined_leads)

    negative_points = []
    if "ets" in metric_names:
        negative_points = [
            point
            for point in (
                (
                    entry["lead_minutes"],
                    _finite_float(
                        (_threshold_mapping(entry.get("categorical"), threshold) or {}).get(
                            "ets"
                        )
                    ),
                )
                for entry in entries
            )
            if point[1] is not None and point[1] < 0
        ]

    by_lead: list[dict[str, Any]] = []
    totals = {key: 0 for key in _COUNT_KEYS}
    for entry in entries:
        counts = _threshold_mapping(entry.get("contingency"), threshold)
        if not counts:
            continue
        row: dict[str, Any] = {"lead_minutes": entry["lead_minutes"]}
        for key in _COUNT_KEYS:
            count = _nonnegative_int(counts.get(key))
            if count is not None:
                row[key] = count
                totals[key] += count
        # Older callers may omit valid_count; derive it from the 2x2 table.
        if "valid_count" not in row and all(key in row for key in _COUNT_KEYS[:4]):
            row["valid_count"] = sum(int(row[key]) for key in _COUNT_KEYS[:4])
            totals["valid_count"] += int(row["valid_count"])
        by_lead.append(row)

    summary = {
        "threshold": threshold,
        "threshold_units": threshold_units,
        "metrics": metrics,
        "contingency": {"by_lead": by_lead, "totals": totals},
    }
    if "ets" in metric_names:
        summary["negative_ets"] = {
            "count": len(negative_points),
            "leads": [lead for lead, _ in negative_points],
            "values": [value for _, value in negative_points],
        }
    return summary


def _normalise_requested_metrics(value: Sequence[str] | None) -> list[str]:
    if value is None:
        return list(_METRIC_ORDER)
    metrics = list(dict.fromkeys(str(metric).lower() for metric in value))
    unsupported = [metric for metric in metrics if metric not in _METRIC_ORDER]
    if unsupported:
        raise ValueError(f"unsupported insight metrics: {unsupported}")
    return metrics


def _metric_facts(
    points: Sequence[tuple[int | float, float]], undefined_leads: Sequence[int | float]
) -> dict[str, Any]:
    if not points:
        return {
            "first": None,
            "last": None,
            "minimum": None,
            "maximum": None,
            "trend": {
                "direction": "insufficient_data",
                "change": None,
                "non_monotonic": False,
            },
            "undefined_leads": list(undefined_leads),
        }

    first_lead, first_value = points[0]
    last_lead, last_value = points[-1]
    min_lead, min_value = min(points, key=lambda item: item[1])
    max_lead, max_value = max(points, key=lambda item: item[1])
    change = last_value - first_value if len(points) >= 2 else None
    tolerance = 1e-12
    if change is None:
        direction = "insufficient_data"
    elif change > tolerance:
        direction = "increased"
    elif change < -tolerance:
        direction = "decreased"
    else:
        direction = "unchanged"

    differences = [later[1] - earlier[1] for earlier, later in zip(points, points[1:])]
    has_increase = any(diff > tolerance for diff in differences)
    has_decrease = any(diff < -tolerance for diff in differences)

    return {
        "first": {"lead_minutes": first_lead, "value": first_value},
        "last": {"lead_minutes": last_lead, "value": last_value},
        "minimum": {"lead_minutes": min_lead, "value": min_value},
        "maximum": {"lead_minutes": max_lead, "value": max_value},
        "trend": {
            "direction": direction,
            "change": change,
            "non_monotonic": has_increase and has_decrease,
        },
        "undefined_leads": list(undefined_leads),
    }


def _normalise_entries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    entries: list[dict[str, Any]] = []
    for raw_entry in value:
        if not isinstance(raw_entry, Mapping):
            continue
        lead = _finite_float(raw_entry.get("lead_minutes"))
        if lead is None or lead < 0:
            continue
        normalised_lead: int | float = int(lead) if lead.is_integer() else lead
        entries.append(
            {
                "lead_minutes": normalised_lead,
                "valid_time": raw_entry.get("valid_time"),
                "categorical": raw_entry.get("categorical"),
                "contingency": raw_entry.get("contingency"),
            }
        )
    return sorted(entries, key=lambda entry: entry["lead_minutes"])


def _collect_thresholds(entries: Sequence[Mapping[str, Any]]) -> list[float]:
    thresholds: set[float] = set()
    for entry in entries:
        for field in ("categorical", "contingency"):
            mapping = entry.get(field)
            if not isinstance(mapping, Mapping):
                continue
            for raw_threshold in mapping:
                threshold = _finite_float(raw_threshold)
                if threshold is not None:
                    thresholds.add(threshold)
    return sorted(thresholds)


def _threshold_mapping(value: Any, threshold: float) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    for raw_threshold, nested in value.items():
        candidate = _finite_float(raw_threshold)
        if candidate is not None and candidate == threshold and isinstance(nested, Mapping):
            return nested
    return None


def _pick_keys(value: Any, keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {key: value[key] for key in keys if key in value}


def _facts_for_llm(facts: Mapping[str, Any]) -> dict[str, Any]:
    selected = {key: facts[key] for key in _SAFE_FACT_KEYS if key in facts}
    return _json_safe(_drop_sensitive(selected))


def _drop_sensitive(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned = {}
        for key, nested in value.items():
            key_text = str(key)
            lower = key_text.lower()
            if (
                lower in _SENSITIVE_KEYS
                or lower.endswith("_path")
                or lower.endswith("_paths")
                or lower.endswith("_file")
                or lower.endswith("_files")
                or lower.endswith("_dir")
                or lower.endswith("_directory")
            ):
                continue
            cleaned[key_text] = _drop_sensitive(nested)
        return cleaned
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_drop_sensitive(item) for item in value]
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return str(value)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_int(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _number_list(value: Any) -> list[int | float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    numbers = []
    for raw_number in value:
        number = _finite_float(raw_number)
        if number is not None:
            numbers.append(int(number) if number.is_integer() else number)
    return sorted(numbers)


def _fmt(value: Any) -> str:
    number = _finite_float(value)
    if number is None:
        return "不可定义"
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _join_numbers(values: Sequence[int | float]) -> str:
    return "、".join(_fmt(value) for value in values)

"""Evidence-bounded interpretation for StormDiT/Fenglei verification."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

from openai import OpenAI

from data_io.forecast_sources import MODEL_LABELS
from verification.metrics import CATEGORICAL_SCORES

_SYSTEM_PROMPT = """你是强对流预报检验平台的AI解读助手。用户消息只包含服务端已经计算好的数字事实JSON。
不得补造数字，不得声称查看过原始像素，不得把个例或有限起报外推成模型总体结论。
只讨论 requested_metrics 中的指标，并引用 figure_catalog 里的图号：
1. 图表导读：用“如图1所示”说明每幅图展示什么。
2. 主要发现：按产品、阈值和时效概括均值、首末变化及双模型胜出次数。
3. 数据分析：指出随时效和阈值的变化，并结合physical_correlations解释相关性热图。
若 same_valid_time_comparison 非空，重点比较同一有效时间下不同起报对应的不同时效，
只能依据其中的结构化评分和诊断量解释，不能声称直接看到了图像像素。
CSI和POD越高越好，FAR越低越好，Bias越接近1越好。
相关系数只能描述共同变化，不能表述为因果关系；成因只能作为待核查假设。
不要增加“口径与边界”章节，不要提路径、代码、API或提示词。"""


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stats(values) -> dict[str, float | None]:
    values = [_finite(value) for value in values]
    values = [value for value in values if value is not None]
    return {
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "mean": sum(values) / len(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "count": len(values),
    }


def _wins(metric, left_values, right_values):
    pairs = [
        (left, right)
        for left, right in zip(left_values, right_values)
        if left is not None and right is not None
    ]
    if metric == "far":
        left_wins = sum(left < right for left, right in pairs)
        right_wins = sum(right < left for left, right in pairs)
    elif metric == "bias_score":
        left_wins = sum(abs(left - 1) < abs(right - 1) for left, right in pairs)
        right_wins = sum(abs(right - 1) < abs(left - 1) for left, right in pairs)
    else:
        left_wins = sum(left > right for left, right in pairs)
        right_wins = sum(right > left for left, right in pairs)
    return {
        "left": left_wins,
        "right": right_wins,
        "ties": len(pairs) - left_wins - right_wins,
    }


def _physical_correlation_facts(
    frame,
    requested_metrics,
    *,
    threshold,
    method,
):
    if frame is None or frame.empty or threshold is None:
        return {}
    selected = frame[
        (frame["threshold_dbz"] - float(threshold)).abs() < 1e-7
    ].copy()
    required = {"valid_count", "hits", "misses", "false_alarms"}
    if selected.empty or not required <= set(selected):
        return {}
    selected = selected[selected["valid_count"] > 0]
    valid = selected["valid_count"].astype(float)
    selected["forecast_rate"] = (
        selected["hits"] + selected["false_alarms"]
    ) / valid
    selected["observed_rate"] = (
        selected["hits"] + selected["misses"]
    ) / valid
    physical = [
        "lead_minutes",
        "forecast_rate",
        "observed_rate",
        "forecast_mean_dbz",
        "observation_mean_dbz",
        "mean_reflectivity_error_dbz",
        "reflectivity_mae_dbz",
        "centroid_distance_km",
    ]
    physical = [
        column
        for column in physical
        if column in selected
        and selected[column].notna().sum() >= 3
        and selected[column].nunique(dropna=True) > 1
    ]
    metrics = [
        metric
        for metric in requested_metrics
        if metric in selected
        and selected[metric].notna().sum() >= 3
        and selected[metric].nunique(dropna=True) > 1
    ]
    if not physical or not metrics:
        return {}
    products = {}
    for model_id, rows in selected.groupby("model_id", sort=False):
        correlations = rows[physical + metrics].corr(
            method=method, min_periods=3
        )
        relationships = []
        for metric in metrics:
            candidates = []
            for variable in physical:
                coefficient = _finite(correlations.loc[variable, metric])
                if coefficient is not None:
                    candidates.append(
                        {
                            "variable": variable,
                            "metric": metric,
                            "coefficient": coefficient,
                        }
                    )
            relationships.extend(
                sorted(
                    candidates,
                    key=lambda item: abs(item["coefficient"]),
                    reverse=True,
                )[:3]
            )
        products[str(model_id)] = {
            "sample_count": int(len(rows)),
            "strongest_score_relationships": relationships,
        }
    return {
        "threshold_dbz": float(threshold),
        "method": method,
        "products": products,
        "interpretation": "descriptive association only; no causal claim",
    }


def _same_valid_time_facts(result, metrics, threshold):
    comparison = result.get("same_valid_time_comparison")
    if not comparison:
        return {}
    selected_threshold = float(threshold)
    items = []
    for item in comparison["items"]:
        products = {}
        for model_id, threshold_map in item["scores"].items():
            values = threshold_map.get(selected_threshold)
            if values is None:
                continue
            products[model_id] = {
                **{metric: _finite(values.get(metric)) for metric in metrics},
                **{
                    name: _finite(values.get(name))
                    for name in (
                        "forecast_mean_dbz",
                        "observation_mean_dbz",
                        "forecast_max_dbz",
                        "observation_max_dbz",
                        "mean_reflectivity_error_dbz",
                        "reflectivity_mae_dbz",
                        "centroid_distance_km",
                    )
                },
            }
        items.append(
            {
                "init_time_utc": item["init_time_utc"].isoformat(timespec="minutes"),
                "init_time_bjt": item["init_time_bjt"].isoformat(timespec="minutes"),
                "lead_minutes": item["lead_minutes"],
                "products": products,
            }
        )
    return {
        "valid_time_utc": comparison["valid_time_utc"].isoformat(timespec="minutes"),
        "valid_time_bjt": comparison["valid_time_bjt"].isoformat(timespec="minutes"),
        "threshold_dbz": selected_threshold,
        "initializations": items,
    }


def build_comparison_insight_facts(
    result: Mapping[str, Any],
    requested_metrics: Sequence[str],
    *,
    figure_catalog: Sequence[Mapping[str, Any]] | None = None,
    diagnostic_threshold_dbz: float | None = None,
    correlation_method: str = "spearman",
) -> dict[str, Any]:
    """Reduce either a single-init or cached batch result to numerical facts."""
    metrics = list(dict.fromkeys(requested_metrics))
    if not metrics or any(metric not in CATEGORICAL_SCORES for metric in metrics):
        raise ValueError("comparison insight requires supported categorical metrics")
    context = result["context"]
    model_ids = list(context["model_ids"])
    threshold_summaries = []
    if "per_lead" in result:
        entries = list(result["per_lead"])
        if not entries:
            raise ValueError("comparison result contains no leads")
        thresholds = sorted(entries[0]["models"][model_ids[0]])
        for threshold in thresholds:
            products = {
                model_id: {
                    metric: _stats(
                        entry["models"][model_id][threshold][metric]
                        for entry in entries
                    )
                    for metric in metrics
                }
                for model_id in model_ids
            }
            wins = {}
            if len(model_ids) == 2:
                left, right = model_ids
                for metric in metrics:
                    left_values = []
                    right_values = []
                    for entry in entries:
                        left_value = _finite(
                            entry["models"][left][threshold][metric]
                        )
                        right_value = _finite(
                            entry["models"][right][threshold][metric]
                        )
                        left_values.append(left_value)
                        right_values.append(right_value)
                    counts = _wins(metric, left_values, right_values)
                    wins[metric] = {
                        left: counts["left"],
                        right: counts["right"],
                        "ties": counts["ties"],
                    }
            threshold_summaries.append(
                {
                    "threshold_dbz": float(threshold),
                    "products": products,
                    "wins_by_metric": wins,
                }
            )
        coverage = {
            "scope": "single_initialization",
            "initialization_count": 1,
            "lead_start_minutes": entries[0]["lead_minutes"],
            "lead_end_minutes": entries[-1]["lead_minutes"],
            "matched_count": len(entries),
        }
        initialization = {
            "init_time_utc": context["init_time_utc"].isoformat(timespec="minutes"),
            "init_time_bjt": context["init_time_bjt"].isoformat(timespec="minutes"),
        }
    else:
        frame = result["per_case"]
        if frame.empty:
            raise ValueError("comparison batch contains no scores")
        for threshold in sorted(frame["threshold_dbz"].unique()):
            selected = frame[frame["threshold_dbz"] == threshold]
            products = {
                model_id: {
                    metric: _stats(
                        selected.loc[selected["model_id"] == model_id]
                        .sort_values(["init_time_utc", "lead_minutes"])[metric]
                    )
                    for metric in metrics
                }
                for model_id in model_ids
            }
            wins = {}
            if len(model_ids) == 2:
                left, right = model_ids
                keys = ["init_time_utc", "lead_minutes"]
                left_frame = selected[selected["model_id"] == left]
                right_frame = selected[selected["model_id"] == right]
                joined = left_frame.merge(right_frame, on=keys, suffixes=("_left", "_right"))
                for metric in metrics:
                    a = joined[f"{metric}_left"]
                    b = joined[f"{metric}_right"]
                    finite = a.notna() & b.notna()
                    counts = _wins(
                        metric,
                        [_finite(value) for value in a[finite]],
                        [_finite(value) for value in b[finite]],
                    )
                    wins[metric] = {
                        left: counts["left"],
                        right: counts["right"],
                        "ties": counts["ties"],
                    }
            threshold_summaries.append(
                {
                    "threshold_dbz": float(threshold),
                    "products": products,
                    "wins_by_metric": wins,
                }
            )
        coverage = {
            "scope": "multiple_initializations",
            "initialization_count": result["coverage"]["selected_initialization_count"],
            "matched_count": result["coverage"]["selected_pairs"],
            "lead_start_minutes": int(frame["lead_minutes"].min()),
            "lead_end_minutes": int(frame["lead_minutes"].max()),
        }
        initialization = {}
    same_valid_time = _same_valid_time_facts(
        result,
        metrics,
        diagnostic_threshold_dbz,
    ) if diagnostic_threshold_dbz is not None else {}
    physical_correlations = (
        {}
        if same_valid_time
        else _physical_correlation_facts(
            result.get("per_case"),
            metrics,
            threshold=diagnostic_threshold_dbz,
            method=correlation_method,
        )
    )
    return {
        "context": {
            **initialization,
            "products": [
                {"id": model_id, "label": MODEL_LABELS[model_id]}
                for model_id in model_ids
            ],
            "contract": context["model_contract"],
        },
        "coverage": coverage,
        "requested_metrics": metrics,
        "thresholds": threshold_summaries,
        "physical_correlations": physical_correlations,
        "same_valid_time_comparison": same_valid_time,
        "figure_catalog": list(figure_catalog or []),
    }


def _format(value):
    return "无有限值" if value is None else f"{value:.3f}"


def render_comparison_deterministic_summary(facts: Mapping[str, Any]) -> str:
    coverage = facts["coverage"]
    labels = {
        item["id"]: item["label"] for item in facts["context"]["products"]
    }
    catalog = facts.get("figure_catalog") or []
    guide = "；".join(
        f"图{item['number']}为{item['label']}" for item in catalog
    ) or "结果图展示所选产品与CREF的同口径检验结果"
    lines = [
        "### 图表导读",
        f"{guide}。",
        "",
        "### 主要发现",
        (
            f"本次纳入 {coverage['initialization_count']} 个起报、"
            f"{coverage['matched_count']} 组严格匹配时效，"
            f"时效范围 {coverage['lead_start_minutes']}—"
            f"{coverage['lead_end_minutes']} 分钟。"
        ),
    ]
    same_valid_time = facts.get("same_valid_time_comparison") or {}
    if same_valid_time:
        lines.append(
            f"空间对比固定在 {same_valid_time['valid_time_bjt']} BJT；"
            "不同起报对应不同预报时效，并共同对照同一CREF观测。"
        )
    for item in facts["thresholds"]:
        threshold = item["threshold_dbz"]
        for metric in facts["requested_metrics"]:
            fragments = [
                f"{labels[model_id]} {_format(values[metric]['mean'])}"
                for model_id, values in item["products"].items()
            ]
            lines.append(
                f"{threshold:g} dBZ 的 {metric.upper()} 均值："
                + "、".join(fragments)
                + "。"
            )
    lines.extend(
        [
            "",
            "### 数据分析",
            "以上结论来自页面所选起报、阈值、时效和区域的结构化评分。",
        ]
    )
    return "\n".join(lines)


def generate_comparison_insight(
    facts: Mapping[str, Any],
    *,
    model: str = "deepseek-chat",
    api_key: str | None = None,
    base_url: str = "https://api.deepseek.com",
    client: Any | None = None,
) -> str:
    payload = json.dumps(facts, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    llm_client = client or OpenAI(
        api_key=api_key or os.environ["DEEPSEEK_API_KEY"], base_url=base_url
    )
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        temperature=0,
    )
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("DeepSeek returned an empty comparison insight response")
    return response.choices[0].message.content.strip()

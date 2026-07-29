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
3. 数据分析：指出随时效和阈值的变化；成因只能作为待核查假设。
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


def build_comparison_insight_facts(
    result: Mapping[str, Any],
    requested_metrics: Sequence[str],
    *,
    figure_catalog: Sequence[Mapping[str, Any]] | None = None,
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
                    pairs: list[tuple[float, float]] = []
                    for entry in entries:
                        left_value = _finite(
                            entry["models"][left][threshold][metric]
                        )
                        right_value = _finite(
                            entry["models"][right][threshold][metric]
                        )
                        if left_value is not None and right_value is not None:
                            pairs.append((left_value, right_value))
                    wins[metric] = {
                        left: sum(a > b for a, b in pairs),
                        right: sum(b > a for a, b in pairs),
                        "ties": sum(a == b for a, b in pairs),
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
                    wins[metric] = {
                        left: int((a[finite] > b[finite]).sum()),
                        right: int((b[finite] > a[finite]).sum()),
                        "ties": int((a[finite] == b[finite]).sum()),
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

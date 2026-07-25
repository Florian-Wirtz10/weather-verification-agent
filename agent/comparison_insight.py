"""Evidence-bounded interpretation for the Phase 8 two-product comparison."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

from openai import OpenAI

from verification.metrics import CATEGORICAL_SCORES

_SYSTEM_PROMPT = """你是强对流预报检验对比解读助手。用户消息只包含服务端已经计算好的事实JSON。
不得补造数字，不得声称看过原始图片或像素，也不得把单个起报外推成模型整体结论。

只讨论 requested_metrics 中列出的指标，固定输出三个短章节：
1. 图表导读：说明图中实线为我们模型、虚线为风雷，相同颜色表示相同阈值。
2. 主要发现：比较两个产品在本次起报、各阈值和时效范围内的首值、末值、均值与胜出时效数。
3. 指标分析：指出优势是否随阈值变化；原因只能写成待核查假设。

不要增加“口径与边界”章节，不要提及路径、代码、API或提示词。"""


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_comparison_insight_facts(
    result: Mapping[str, Any], requested_metrics: Sequence[str]
) -> dict[str, Any]:
    metrics = list(dict.fromkeys(requested_metrics))
    if not metrics or any(metric not in CATEGORICAL_SCORES for metric in metrics):
        raise ValueError("comparison insight requires supported categorical metrics")
    entries = list(result.get("per_lead") or [])
    if not entries:
        raise ValueError("comparison result contains no leads")
    thresholds = sorted(entries[0]["model"])
    summaries = []
    for threshold in thresholds:
        products = {}
        wins = {}
        for metric in metrics:
            model_values = [_finite(entry["model"][threshold][metric]) for entry in entries]
            fenglei_values = [_finite(entry["fenglei"][threshold][metric]) for entry in entries]
            valid_pairs = [
                (model, fenglei)
                for model, fenglei in zip(model_values, fenglei_values)
                if model is not None and fenglei is not None
            ]
            wins[metric] = {
                "model": sum(model > fenglei for model, fenglei in valid_pairs),
                "fenglei": sum(fenglei > model for model, fenglei in valid_pairs),
                "ties": sum(model == fenglei for model, fenglei in valid_pairs),
            }
            for name, values in (
                ("model", model_values),
                ("fenglei", fenglei_values),
            ):
                finite_values = [value for value in values if value is not None]
                products.setdefault(name, {})[metric] = {
                    "first": finite_values[0] if finite_values else None,
                    "last": finite_values[-1] if finite_values else None,
                    "mean": (
                        sum(finite_values) / len(finite_values) if finite_values else None
                    ),
                    "minimum": min(finite_values) if finite_values else None,
                    "maximum": max(finite_values) if finite_values else None,
                }
        summaries.append(
            {
                "threshold_dbz": float(threshold),
                "products": products,
                "wins_by_metric": wins,
            }
        )
    context = result["context"]
    return {
        "context": {
            "init_time_utc": context["init_time_utc"].isoformat(timespec="minutes"),
            "init_time_bjt": context["init_time_bjt"].isoformat(timespec="minutes"),
            "contract": context["model_contract"],
        },
        "coverage": {
            "lead_start_minutes": entries[0]["lead_minutes"],
            "lead_end_minutes": entries[-1]["lead_minutes"],
            "matched_count": len(entries),
        },
        "requested_metrics": metrics,
        "thresholds": summaries,
        "figure_catalog": [
            {
                "number": index,
                "metric": metric.upper(),
                "label": f"我们模型与风雷的 {metric.upper()} 随时效对比",
            }
            for index, metric in enumerate(metrics, 1)
        ],
    }


def render_comparison_deterministic_summary(facts: Mapping[str, Any]) -> str:
    coverage = facts["coverage"]
    lines = [
        "### 图表导读",
        "实线表示我们模型，虚线表示风雷；相同颜色表示相同反射率阈值。",
        "",
        "### 主要发现",
        (
            f"本次包含 {coverage['matched_count']} 个严格匹配时效，"
            f"范围为 {coverage['lead_start_minutes']}—{coverage['lead_end_minutes']} 分钟。"
        ),
    ]
    for item in facts["thresholds"]:
        threshold = item["threshold_dbz"]
        for metric in facts["requested_metrics"]:
            model = item["products"]["model"][metric]["mean"]
            fenglei = item["products"]["fenglei"][metric]["mean"]
            wins = item["wins_by_metric"][metric]["model"]
            lines.append(
                f"{threshold:g} dBZ：{metric.upper()} 均值为我们模型 {model:.3f}、"
                f"风雷 {fenglei:.3f}；我们模型在 {wins}/{coverage['matched_count']} 个时效较高。"
            )
    lines.extend(
        [
            "",
            "### 指标分析",
            "以上比较只描述当前起报的同口径评分；不同阈值下的相对表现可能不同。",
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

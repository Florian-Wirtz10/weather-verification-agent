"""Evidence-bounded AI interpretation for multi-initialization summaries."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping

import numpy as np
import pandas as pd
from openai import OpenAI

_METRICS = {"csi", "ets", "pod", "far", "bias_score"}
_SYSTEM_PROMPT = """你是强对流预报检验平台的结果解读助手。用户消息只有服务端已经
计算好的结构化事实；不得重新计算、补造数字，不得声称看过原始图片或网格。

固定输出三个简短章节：
1. 图表导读：只按 figure_catalog 中实际存在的编号使用“如图1所示”介绍图表内容。
2. 总体结果：客观说明不同阈值下该指标随时效的首末值、范围和极值。
3. 离散性与诊断：结合热力图事实、分位数、最弱样本和列联表，指出值得复核的起报/时效；
   原因只能表述为“可能”或“需要结合空间回波进一步核查”。

requested_metric 非空时，只讨论该评分，不主动引入其它评分；为空时不要讨论任何评分。
仅当 rate_diagnostics 非空时，可以客观说明预报率、
观测率及散点相对 1:1 线的数量关系；不要把二者误称为 CSI、Bias 等评分。
不要提到代码、路径、API、提示词或缓存实现。严格使用上述三个章节，不得新增其它章节，
也不要讨论结果能否推广。"""


def _number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _point(row: Mapping[str, Any], metric: str) -> dict[str, Any]:
    return {
        "lead_minutes": int(row["lead_minutes"]),
        "value": _number(row[metric]),
        "case_count": int(row.get("case_count", 0)),
    }


def build_daily_insight_facts(
    result: Mapping[str, Any], *, metric="csi", diagnostic_threshold_dbz=15, views=None
) -> dict[str, Any]:
    """Reduce cached DataFrames to a compact, path-free numerical fact sheet."""
    metric = str(metric).lower()
    if metric not in _METRICS:
        raise ValueError(f"unsupported daily insight metric: {metric!r}")
    summary = result.get("summary")
    per_case = result.get("per_case")
    if not isinstance(summary, pd.DataFrame) or not isinstance(per_case, pd.DataFrame):
        raise ValueError("daily insight requires summary and per_case DataFrames")
    if metric not in summary or metric not in per_case:
        raise ValueError(f"daily result does not contain {metric}")

    threshold_facts = []
    for threshold, rows in summary.groupby("threshold_dbz", sort=True):
        rows = rows.sort_values("lead_minutes")
        metric_values = np.asarray(rows[metric].to_numpy(), dtype=float)
        finite = rows.iloc[np.flatnonzero(np.isfinite(metric_values))]
        threshold_value = _number(threshold)
        if threshold_value is None:
            continue
        if finite.empty:
            threshold_facts.append(
                {
                    "threshold_dbz": threshold_value,
                    "finite_count": 0,
                    "undefined_count": len(rows),
                }
            )
            continue
        finite_values = np.asarray(finite[metric].to_numpy(), dtype=float)
        min_row = finite.iloc[int(np.argmin(finite_values))]
        max_row = finite.iloc[int(np.argmax(finite_values))]
        threshold_facts.append(
            {
                "threshold_dbz": threshold_value,
                "finite_count": int(len(finite)),
                "undefined_count": int(len(rows) - len(finite)),
                "first": _point(finite.iloc[0], metric),
                "last": _point(finite.iloc[-1], metric),
                "minimum": _point(min_row, metric),
                "maximum": _point(max_row, metric),
                "mean": _number(np.mean(finite_values)),
            }
        )

    selected = per_case[
        (per_case["threshold_dbz"] - float(diagnostic_threshold_dbz)).abs() < 1e-7
    ].copy()
    if selected.empty:
        raise ValueError(
            f"diagnostic threshold {diagnostic_threshold_dbz:g} dBZ is not in the result"
        )
    selected["metric_value"] = np.asarray(selected[metric], dtype=float)
    lead_distributions = []
    for lead, rows in selected.groupby("lead_minutes", sort=True):
        values = np.asarray(rows["metric_value"], dtype=float)
        values = values[np.isfinite(values)]
        lead_number = _number(lead)
        if lead_number is None:
            continue
        lead_distributions.append(
            {
                "lead_minutes": int(lead_number),
                "sample_count": int(len(values)),
                "undefined_count": int(len(rows) - len(values)),
                "q25": _number(np.quantile(values, 0.25)) if len(values) else None,
                "median": _number(np.median(values)) if len(values) else None,
                "q75": _number(np.quantile(values, 0.75)) if len(values) else None,
            }
        )
    selected_values = np.asarray(selected["metric_value"], dtype=float)
    weakest = selected.iloc[np.flatnonzero(np.isfinite(selected_values))].sort_values(
        "metric_value"
    ).head(5)
    weakest = weakest.loc[
        :, ["init_time_bjt", "lead_minutes", "metric_value", "hits", "misses", "false_alarms"]
    ]
    weakest_cases = [
        {
            "init_time_bjt": str(row.init_time_bjt),
            "lead_minutes": int(row.lead_minutes),
            "value": _number(row.metric_value),
            "hits": int(row.hits),
            "misses": int(row.misses),
            "false_alarms": int(row.false_alarms),
        }
        for row in weakest.itertuples(index=False)
    ]
    selected["init_time_bjt"] = pd.to_datetime(selected["init_time_bjt"])
    selected["init_time_bin_bjt"] = selected["init_time_bjt"].dt.floor("30min")
    initialization_bins = []
    for init_bin, rows in selected.groupby("init_time_bin_bjt", sort=True):
        values = np.asarray(rows["metric_value"], dtype=float)
        values = values[np.isfinite(values)]
        initialization_bins.append(
            {
                "init_time_bjt": init_bin.isoformat(timespec="minutes"),
                "sample_count": int(len(values)),
                "median": _number(np.median(values)) if len(values) else None,
            }
        )

    raw_context = result.get("context")
    context: Mapping[str, Any] = raw_context if isinstance(raw_context, Mapping) else {}
    raw_coverage = result.get("coverage")
    coverage: Mapping[str, Any] = raw_coverage if isinstance(raw_coverage, Mapping) else {}
    view_labels = {
        "daily_score_curve": f"汇总 {metric.upper()} 随预报时效曲线",
        "daily_score_heatmap": (
            f"{diagnostic_threshold_dbz:g} dBZ 下起报时刻×时效 {metric.upper()} 热力图"
        ),
        "score_distribution": (
            f"{diagnostic_threshold_dbz:g} dBZ 下各时效 {metric.upper()} 分布箱线图"
        ),
        "forecast_observed_rate_distribution": (
            f"{diagnostic_threshold_dbz:g} dBZ 下预报率—观测率联合分布"
        ),
    }
    requested_views = list(views) if views is not None else list(view_labels)
    score_view_types = {"daily_score_curve", "daily_score_heatmap", "score_distribution"}
    has_score_view = any(view in score_view_types for view in requested_views)
    figure_catalog = [
        {"number": number, "type": view, "label": view_labels[view]}
        for number, view in enumerate(requested_views, start=1)
        if view in view_labels
    ]
    rate_diagnostics = None
    rate_columns = {"valid_count", "hits", "misses", "false_alarms"}
    if (
        "forecast_observed_rate_distribution" in requested_views
        and rate_columns.issubset(selected.columns)
    ):
        valid_rate_rows = selected[selected["valid_count"] > 0].copy()
        if not valid_rate_rows.empty:
            valid_count = valid_rate_rows["valid_count"].to_numpy(dtype=float)
            forecast_rate = (
                valid_rate_rows["hits"].to_numpy(dtype=float)
                + valid_rate_rows["false_alarms"].to_numpy(dtype=float)
            ) / valid_count
            observed_rate = (
                valid_rate_rows["hits"].to_numpy(dtype=float)
                + valid_rate_rows["misses"].to_numpy(dtype=float)
            ) / valid_count
            rate_diagnostics = {
                "sample_count": int(len(valid_rate_rows)),
                "mean_forecast_rate": _number(np.mean(forecast_rate)),
                "mean_observed_rate": _number(np.mean(observed_rate)),
                "mean_forecast_minus_observed": _number(
                    np.mean(forecast_rate - observed_rate)
                ),
                "above_one_to_one_count": int(np.sum(forecast_rate > observed_rate)),
                "below_one_to_one_count": int(np.sum(forecast_rate < observed_rate)),
            }
    return {
        "context": {
            "case_date": context.get("case_date"),
            "model_id": context.get("model_id"),
            "thresholds_dbz": context.get("thresholds_dbz"),
            "lead_range_minutes": context.get("lead_range_minutes"),
        },
        "coverage": {
            key: coverage.get(key)
            for key in (
                "unique_observations",
                "selected_pairs",
                "completed_pairs",
                "duplicate_observation_times",
                "discarded_observation_revisions",
            )
        },
        "requested_metric": metric if has_score_view else None,
        "diagnostic_threshold_dbz": float(diagnostic_threshold_dbz),
        "threshold_summary": threshold_facts if has_score_view else [],
        "lead_distributions": lead_distributions if has_score_view else [],
        "initialization_time_summary": initialization_bins if has_score_view else [],
        "weakest_cases": weakest_cases if has_score_view else [],
        "rate_diagnostics": rate_diagnostics,
        "figure_catalog": figure_catalog,
    }


def render_daily_deterministic_summary(facts: Mapping[str, Any]) -> str:
    requested_metric = facts.get("requested_metric")
    metric = str(requested_metric).upper() if requested_metric else "评分"
    figure_count = len(facts.get("figure_catalog", []))
    figure_text = f"共生成 {figure_count} 张已请求视图" if figure_count else "未生成图表"
    lines = [f"本次结果汇总 {metric}，{figure_text}。"]
    for item in facts.get("threshold_summary", []):
        if not isinstance(item, Mapping) or "first" not in item:
            continue
        first, last = item["first"], item["last"]
        lines.append(
            f"{item['threshold_dbz']:g} dBZ：{first['lead_minutes']}分钟 {first['value']:.3f}，"
            f"{last['lead_minutes']}分钟 {last['value']:.3f}。"
        )
    rate_diagnostics = facts.get("rate_diagnostics")
    if isinstance(rate_diagnostics, Mapping):
        lines.append(
            f"预报率—观测率图包含 {rate_diagnostics.get('sample_count', 0)} 个匹配样本；"
            f"平均预报率 {_number(rate_diagnostics.get('mean_forecast_rate')) or 0:.3f}，"
            f"平均观测率 {_number(rate_diagnostics.get('mean_observed_rate')) or 0:.3f}。"
        )
    return "\n".join(lines)


def generate_daily_insight(
    facts: Mapping[str, Any],
    *,
    model="deepseek-chat",
    api_key: str | None = None,
    base_url="https://api.deepseek.com",
    client: Any | None = None,
) -> str:
    """Ask DeepSeek to verbalize the precomputed multi-initialization fact sheet."""
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
        raise ValueError("DeepSeek returned an empty daily insight response")
    return response.choices[0].message.content.strip()

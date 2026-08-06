"""Evidence-bounded interpretation for the zero-shot benchmark."""

from __future__ import annotations

import json
import math
import os

from openai import OpenAI


_SYSTEM_PROMPT = """你是中国雷达CREF Zero-shot Benchmark的结果解读助手。
用户消息只有服务端计算好的数字事实JSON，不得补造数字或声称看过原始像素。
只输出三个章节，标题必须是“### 图表导读”“### 结论摘要”“### 数据证据与诊断分析”。
图表导读逐图使用“如图1所示”；结论摘要说明不同模型在什么指标、阈值和时效下更优；
数据证据与诊断分析提供可核对数字，并明确说明这些模型是SEVIR预训练权重到中国CREF的zero-shot迁移。
CSI/POD/FSS越高越好，FAR/MAE/RMSE越低越好，Bias越接近1越好。
不得把有限个例外推为模型总体能力，不得编造因果，不得增加“口径与边界”章节。"""


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _better(metric, value):
    if value is None:
        return float("inf")
    if metric in {"far", "mae", "rmse"}:
        return value
    if metric == "bias_score":
        return abs(value - 1)
    return -value


def build_benchmark_insight_facts(result, request, period, figure_catalog):
    threshold = float(request["diagnostic_threshold_dbz"])
    selected_period = period[(period["threshold_dbz"] - threshold).abs() < 1e-7]
    metrics = []
    fss_window = max(request["fss_windows"])
    for requested in request["metrics"]:
        column = f"fss_w{fss_window}" if requested == "fss" else requested
        values = {
            str(row.model): _finite(getattr(row, column))
            for row in selected_period.itertuples(index=False)
        }
        if not values:
            continue
        leader = min(values, key=lambda model: _better(requested, values[model]))
        trends = {}
        selected_summary = result["summary"][
            (result["summary"]["threshold_dbz"] - threshold).abs() < 1e-7
        ]
        for model, rows in selected_summary.groupby("model", sort=False):
            rows = rows.sort_values("lead_minutes")
            trends[str(model)] = {
                "first": _finite(rows.iloc[0][column]),
                "last": _finite(rows.iloc[-1][column]),
            }
        metrics.append(
            {
                "metric": requested,
                "pooled_values": values,
                "leader": leader,
                "lead_trends": trends,
            }
        )
    return {
        "experiment": {
            "name": "China CREF zero-shot benchmark",
            "case": "2025-04-11 Wuhan convective event",
            "models": result["context"]["models"],
            "sample_count": result["coverage"]["sample_count"],
            "lead_range_minutes": result["context"]["lead_range_minutes"],
            "diagnostic_threshold_dbz": threshold,
            "fss_window_pixels": fss_window,
            "grid": "128x128 common patch; 0.03 degree; no verification-stage interpolation",
            "fairness": (
                "SEVIR-pretrained baselines transferred zero-shot to China CREF; "
                "all selected models are compared only on their strict common native leads"
            ),
        },
        "metrics": metrics,
        "figures": list(figure_catalog),
    }


def render_benchmark_deterministic_summary(facts):
    experiment = facts["experiment"]
    guide = (
        "；".join(f"图{item['number']}为{item['label']}" for item in facts["figures"])
        or "本次请求未生成图表"
    )
    lines = [
        "### 图表导读",
        guide + "。",
        "",
        "### 结论摘要",
        (
            f"本次纳入{experiment['sample_count']}个起报，严格公共时效为"
            f"+{experiment['lead_range_minutes'][0]}—+{experiment['lead_range_minutes'][1]}分钟。"
        ),
    ]
    for item in facts["metrics"]:
        value = item["pooled_values"][item["leader"]]
        lines.append(
            f"在{experiment['diagnostic_threshold_dbz']:g} dBZ下，"
            f"{item['metric'].upper()}汇总表现以{item['leader']}最优（{value:.3f}）。"
        )
    lines.extend(["", "### 数据证据与诊断分析"])
    for item in facts["metrics"]:
        evidence = "、".join(
            f"{model} {value:.3f}" if value is not None else f"{model}无有限值"
            for model, value in item["pooled_values"].items()
        )
        lines.append(f"{item['metric'].upper()}汇总值：{evidence}。")
    lines.append(
        "这些数字反映SEVIR预训练Baseline向中国CREF的zero-shot迁移表现；"
        "模型的条件长度、推理分辨率和上下文范围不同，不能外推为同训练条件下的总体排名。"
    )
    return "\n".join(lines)


def generate_benchmark_insight(
    facts,
    *,
    model="deepseek-chat",
    api_key=None,
    base_url="https://api.deepseek.com",
):
    client = OpenAI(api_key=api_key or os.environ["DEEPSEEK_API_KEY"], base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False, allow_nan=False)},
        ],
        temperature=0,
    )
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("DeepSeek returned an empty benchmark insight")
    return response.choices[0].message.content.strip()

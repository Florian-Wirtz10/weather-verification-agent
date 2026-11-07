"""Evidence-bounded interpretation for the zero-shot benchmark."""

from __future__ import annotations

import json
import math
import os

from openai import OpenAI

from agent.insight_text import remove_meta_question_phrasing


_BUSINESS_MODELS = {"StormDiT", "Fenglei"}
_ZERO_SHOT_BASELINES = {
    "ConvLSTM",
    "PredRNN",
    "Earthformer",
    "NowCastNet",
    "SimVP",
    "DiffCast",
}
_REFERENCE_BASELINES = {"Persistence"}

_SYSTEM_PROMPT = """你是中国雷达CREF多模型基准检验的结果解读助手。
用户消息只有服务端计算好的数字事实JSON，不得补造数字或声称看过原始像素。
只输出三个章节，标题必须是“### 图表导读”“### 结论摘要”“### 数据证据与诊断分析”。
图表导读逐图使用“如图1所示”，只说明图形编码、比较对象和展示内容；不得出现
“可回答……”“用于回答……”等元话语。结论摘要说明不同模型在什么指标、阈值和时效下更优；
数据证据与诊断分析提供可核对数字，并严格区分模型身份：StormDiT和Fenglei是业务模型，
ConvLSTM、PredRNN、Earthformer、NowCastNet、SimVP和DiffCast是SEVIR预训练权重到中国CREF的
zero-shot基线，Persistence是持续性参考。不得把业务模型或Persistence称为zero-shot模型。
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
    metric_specs = []
    for requested in request["metrics"]:
        if requested == "fss":
            metric_specs.extend(
                (f"fss_w{window}", f"FSS {window}×{window}像元")
                for window in request["fss_windows"]
            )
        else:
            metric_specs.append((requested, requested.upper()))
    for column, label in metric_specs:
        values = {
            str(row.model): _finite(getattr(row, column))
            for row in selected_period.itertuples(index=False)
        }
        if not values:
            continue
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
                "metric": label,
                "pooled_values": values,
                "leader": min(values, key=lambda model: _better(column, values[model])),
                "lead_trends": trends,
            }
        )
    selected_models = [str(model) for model in result["context"]["models"]]
    return {
        "experiment": {
            "name": (
                "China CREF multi-model benchmark"
                if any(model in _BUSINESS_MODELS for model in selected_models)
                else "China CREF zero-shot benchmark"
            ),
            "case": "2025-04-11 Wuhan convective event",
            "models": selected_models,
            "business_models": [
                model for model in selected_models if model in _BUSINESS_MODELS
            ],
            "zero_shot_baselines": [
                model for model in selected_models if model in _ZERO_SHOT_BASELINES
            ],
            "reference_baselines": [
                model for model in selected_models if model in _REFERENCE_BASELINES
            ],
            "sample_count": result["coverage"]["sample_count"],
            "lead_range_minutes": result["context"]["lead_range_minutes"],
            "diagnostic_threshold_dbz": threshold,
            "fss_window_pixels": list(request["fss_windows"]),
            "grid": "128x128 common patch; 0.03 degree; no verification-stage interpolation",
            "fairness": (
                "Business products, SEVIR-pretrained zero-shot baselines, and persistence "
                "references are compared only on their strict common samples and native leads"
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
    identities = []
    if experiment["business_models"]:
        identities.append(
            f"{', '.join(experiment['business_models'])}为业务模型"
        )
    if experiment["zero_shot_baselines"]:
        identities.append(
            f"{', '.join(experiment['zero_shot_baselines'])}为SEVIR预训练后迁移到中国CREF的zero-shot基线"
        )
    if experiment["reference_baselines"]:
        identities.append(
            f"{', '.join(experiment['reference_baselines'])}为持续性参考"
        )
    lines.append(
        "；".join(identities)
        + "。所有结果只代表严格公共样本和公共时效下的比较，不能外推为同训练条件下的总体排名。"
    )
    return "\n".join(lines)


def generate_benchmark_insight(
    facts,
    *,
    model="deepseek-chat",
    api_key=None,
    base_url="https://api.deepseek.com",
):
    client = OpenAI(
        api_key=api_key or os.environ["DEEPSEEK_API_KEY"], base_url=base_url,
        timeout=60.0, max_retries=1,
    )
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
    return remove_meta_question_phrasing(response.choices[0].message.content)

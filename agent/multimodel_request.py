"""Closed request schema for the Phase 8 multi-model verification page."""

from __future__ import annotations

import json
import math
from numbers import Real
import os

from openai import OpenAI

MULTIMODEL_VIEWS = (
    "spatial_comparison",
    "reflectivity_animation",
    "score_curve",
    "multi_metric_curve",
    "score_heatmap",
    "variable_correlation_heatmap",
    "score_distribution",
    "forecast_observed_rate_scatter",
    "forecast_observed_rate_distribution",
)
MULTIMODEL_METRICS = ("csi", "pod", "far", "bias_score")
COMBINABLE_METRICS = ("csi", "pod", "far")
CORRELATION_METHODS = ("spearman", "pearson")

_SYSTEM_PROMPT = f"""你是强对流预报检验平台的受限请求规划器，只输出 JSON。
页面已经由用户选择 StormDiT、风雷或 StormDiT vs 风雷，以及单一起报、自定义多起报或全部起报。
你只能选择检验参数和已实现图表，不能计算、写绘图代码、改路径、改变时间匹配、空间配准或评分公式。

支持指标：{list(MULTIMODEL_METRICS)}
支持图表：
- spatial_comparison：同一有效时刻的预报和 CREF 空间对比；
- reflectivity_animation：所选时效的同步预报—观测动画；
- score_curve：一个指标在多个阈值下的时效曲线；
- multi_metric_curve：固定诊断阈值，在一张图比较CSI、POD、FAR；
- score_heatmap：起报时刻×预报时效评分矩阵，双模型时同时给差值图；
- variable_correlation_heatmap：物理诊断量与评分的相关性热力图；
- score_distribution：跨起报、多阈值指标分布；
- forecast_observed_rate_scatter：简洁预报率—观测率散点图；
- forecast_observed_rate_distribution：预报率—观测率散点与边缘分布。

规则：
1. 单一起报支持空间图、动画和两类时效曲线；多起报/全部起报支持评分矩阵、
   相关性、分布和预报率图，但不支持空间图和动画。
2. “全套、全面、演示”时：单起报选 spatial_comparison、reflectivity_animation、score_curve；
   多起报选 score_curve、multi_metric_curve、score_heatmap、
   variable_correlation_heatmap、score_distribution、forecast_observed_rate_scatter。
3. 未指定指标用 [csi,pod,far]；不支持ETS。未指定阈值用 [10,20,30,40]；
   诊断阈值默认20并必须包含于阈值数组；评分矩阵和分布图的diagnostic_metric
   默认csi并必须包含于metrics。
4. 未指定时效用 6—180 分钟，起止必须为6的倍数。
5. 动画最多30帧、1—4 fps。frame_step 表示每隔多少个原生6分钟时效取一帧，默认1。
6. 不支持的要求原样放到 unmatched_requests，不能悄悄换图。
7. multi_metric_curve 只组合 CSI、POD、FAR；Bias需使用score_curve单独展示。
8. 相关性方法只能是spearman或pearson，默认spearman。

JSON结构：
{{
  "metrics": ["csi", "pod", "far"],
  "thresholds_dbz": [10,20,30,40],
  "diagnostic_threshold_dbz": 20,
  "diagnostic_metric": "csi",
  "lead_range_minutes": {{"start": 6, "end": 180}},
  "views": ["spatial_comparison","reflectivity_animation","score_curve","multi_metric_curve"],
  "correlation_method": "spearman",
  "animation": {{"fps": 2, "frame_step": 1}},
  "unmatched_requests": [],
  "notes": ""
}}"""


def default_multimodel_request(*, batch: bool) -> dict:
    return {
        "metrics": ["csi", "pod", "far"],
        "thresholds_dbz": [10.0, 20.0, 30.0, 40.0],
        "diagnostic_threshold_dbz": 20.0,
        "diagnostic_metric": "csi",
        "lead_range_minutes": {"start": 6, "end": 180},
        "views": (
            [
                "score_curve",
                "multi_metric_curve",
                "score_heatmap",
                "variable_correlation_heatmap",
                "score_distribution",
                "forecast_observed_rate_scatter",
            ]
            if batch
            else [
                "spatial_comparison",
                "reflectivity_animation",
                "score_curve",
                "multi_metric_curve",
            ]
        ),
        "correlation_method": "spearman",
        "animation": {"fps": 2, "frame_step": 1},
        "unmatched_requests": [],
        "notes": "",
    }


def _lead(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 6 <= value <= 180 or value % 6:
        raise ValueError(f"{name} must be a multiple of 6 from 6 to 180")
    return value


def validate_multimodel_request(parsed, *, batch: bool) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("multi-model request must be a JSON object")
    raw_metrics = parsed.get("metrics", ["csi", "pod", "far"])
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ValueError("metrics must be a non-empty list")
    metrics = []
    for value in raw_metrics:
        metric = str(value).strip().lower()
        if metric not in MULTIMODEL_METRICS:
            raise ValueError(f"unsupported metric: {metric}")
        if metric not in metrics:
            metrics.append(metric)
    raw_thresholds = parsed.get("thresholds_dbz", [10, 20, 30, 40])
    if not isinstance(raw_thresholds, list) or not raw_thresholds:
        raise ValueError("thresholds_dbz must be a non-empty list")
    thresholds = []
    for value in raw_thresholds:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("threshold values must be numeric")
        threshold = float(value)
        if not math.isfinite(threshold) or not 0 <= threshold <= 75:
            raise ValueError("threshold values must be between 0 and 75 dBZ")
        if threshold not in thresholds:
            thresholds.append(threshold)
    if len(thresholds) > 8:
        raise ValueError("at most eight thresholds are supported")
    diagnostic = float(parsed.get("diagnostic_threshold_dbz", 20))
    if diagnostic not in thresholds:
        raise ValueError("diagnostic_threshold_dbz must be in thresholds_dbz")
    diagnostic_metric = str(
        parsed.get("diagnostic_metric", metrics[0])
    ).strip().lower()
    if diagnostic_metric not in metrics:
        raise ValueError("diagnostic_metric must be included in metrics")
    lead_range = parsed.get("lead_range_minutes", {})
    if not isinstance(lead_range, dict):
        raise ValueError("lead_range_minutes must be an object")
    start = _lead(lead_range.get("start", 6), "lead start")
    end = _lead(lead_range.get("end", 180), "lead end")
    if start > end:
        raise ValueError("lead start cannot exceed lead end")
    unmatched = parsed.get("unmatched_requests", [])
    if not isinstance(unmatched, list) or not all(isinstance(item, str) for item in unmatched):
        raise ValueError("unmatched_requests must be a string list")
    views = []
    raw_views = parsed.get("views", default_multimodel_request(batch=batch)["views"])
    if not isinstance(raw_views, list):
        raise ValueError("views must be a list")
    for value in raw_views:
        view = str(value).strip().lower()
        if view not in MULTIMODEL_VIEWS:
            unmatched.append(f"暂不支持图表：{view}")
        elif batch and view in {"spatial_comparison", "reflectivity_animation"}:
            unmatched.append(f"多起报模式暂不支持：{view}")
        elif not batch and view in {
            "score_heatmap",
            "variable_correlation_heatmap",
            "score_distribution",
            "forecast_observed_rate_scatter",
            "forecast_observed_rate_distribution",
        }:
            unmatched.append(f"单一起报模式暂不支持：{view}")
        elif view not in views:
            views.append(view)
    if "multi_metric_curve" in views:
        combined = [metric for metric in metrics if metric in COMBINABLE_METRICS]
        if len(combined) < 2:
            views.remove("multi_metric_curve")
            unmatched.append("多指标同图至少需要CSI、POD、FAR中的两项")
    correlation_method = str(
        parsed.get("correlation_method", "spearman")
    ).strip().lower()
    if correlation_method not in CORRELATION_METHODS:
        raise ValueError(
            "correlation_method must be 'spearman' or 'pearson'"
        )
    animation = parsed.get("animation", {})
    if not isinstance(animation, dict):
        raise ValueError("animation must be an object")
    fps = animation.get("fps", 2)
    frame_step = animation.get("frame_step", 1)
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 4:
        raise ValueError("animation.fps must be 1—4")
    if (
        isinstance(frame_step, bool)
        or not isinstance(frame_step, int)
        or not 1 <= frame_step <= 30
    ):
        raise ValueError("animation.frame_step must be 1—30")
    return {
        "metrics": metrics,
        "thresholds_dbz": thresholds,
        "diagnostic_threshold_dbz": diagnostic,
        "diagnostic_metric": diagnostic_metric,
        "lead_range_minutes": {"start": start, "end": end},
        "views": views,
        "correlation_method": correlation_method,
        "animation": {"fps": fps, "frame_step": frame_step},
        "unmatched_requests": list(dict.fromkeys(item.strip() for item in unmatched if item.strip())),
        "notes": str(parsed.get("notes", ""))[:1000],
    }


def parse_multimodel_request(
    user_text,
    *,
    batch: bool,
    model="deepseek-chat",
    api_key=None,
    base_url="https://api.deepseek.com",
) -> dict:
    client = OpenAI(api_key=api_key or os.environ["DEEPSEEK_API_KEY"], base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": str(user_text)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("DeepSeek returned an empty multi-model request")
    return validate_multimodel_request(
        json.loads(response.choices[0].message.content), batch=batch
    )

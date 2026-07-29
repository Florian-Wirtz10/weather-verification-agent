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
    "score_heatmap",
    "score_distribution",
    "forecast_observed_rate_distribution",
)
MULTIMODEL_METRICS = ("csi", "ets", "pod", "far", "bias_score")

_SYSTEM_PROMPT = f"""你是强对流预报检验平台的受限请求规划器，只输出 JSON。
页面已经由用户选择 StormDiT、风雷或 StormDiT vs 风雷，以及单一起报、自定义多起报或全部起报。
你只能选择检验参数和已实现图表，不能计算、写绘图代码、改路径、改变时间匹配、空间配准或评分公式。

支持指标：{list(MULTIMODEL_METRICS)}
支持图表：
- spatial_comparison：同一有效时刻的预报和 CREF 空间对比；
- reflectivity_animation：所选时效的同步预报—观测动画；
- score_curve：指标随时效曲线；
- score_heatmap：起报时刻×预报时效热力图，双模型时同时给差值图；
- score_distribution：跨起报指标分布；
- forecast_observed_rate_distribution：预报率—观测率散点与边缘分布。

规则：
1. 单一起报可以使用全部六类图；多起报/全部起报不支持空间图和动画。
2. “全套、全面、演示”时：单起报选 spatial_comparison、reflectivity_animation、score_curve；
   多起报选 score_curve、score_heatmap、score_distribution、forecast_observed_rate_distribution。
3. 未指定指标用 csi；未指定阈值用 [5,15,25,35]；诊断阈值默认15并必须包含于阈值数组。
4. 未指定时效用 6—180 分钟，起止必须为6的倍数。
5. 动画最多30帧、1—4 fps。frame_step 表示每隔多少个原生6分钟时效取一帧，默认1。
6. 不支持的要求原样放到 unmatched_requests，不能悄悄换图。

JSON结构：
{{
  "metrics": ["csi", "ets"],
  "thresholds_dbz": [5,15,25,35],
  "diagnostic_threshold_dbz": 15,
  "lead_range_minutes": {{"start": 6, "end": 180}},
  "views": ["spatial_comparison","reflectivity_animation","score_curve"],
  "animation": {{"fps": 2, "frame_step": 1}},
  "unmatched_requests": [],
  "notes": ""
}}"""


def default_multimodel_request(*, batch: bool) -> dict:
    return {
        "metrics": ["csi", "ets"],
        "thresholds_dbz": [5.0, 15.0, 25.0, 35.0],
        "diagnostic_threshold_dbz": 15.0,
        "lead_range_minutes": {"start": 6, "end": 180},
        "views": (
            [
                "score_curve",
                "score_heatmap",
                "score_distribution",
                "forecast_observed_rate_distribution",
            ]
            if batch
            else ["spatial_comparison", "reflectivity_animation", "score_curve"]
        ),
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
    raw_metrics = parsed.get("metrics", ["csi"])
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ValueError("metrics must be a non-empty list")
    metrics = []
    for value in raw_metrics:
        metric = str(value).strip().lower()
        if metric not in MULTIMODEL_METRICS:
            raise ValueError(f"unsupported metric: {metric}")
        if metric not in metrics:
            metrics.append(metric)
    raw_thresholds = parsed.get("thresholds_dbz", [5, 15, 25, 35])
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
    diagnostic = float(parsed.get("diagnostic_threshold_dbz", 15))
    if diagnostic not in thresholds:
        raise ValueError("diagnostic_threshold_dbz must be in thresholds_dbz")
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
        elif view not in views:
            views.append(view)
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
        "lead_range_minutes": {"start": start, "end": end},
        "views": views,
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

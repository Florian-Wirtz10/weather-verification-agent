"""Closed natural-language request schema for multi-initialization diagnostics."""

from __future__ import annotations

from datetime import date
import json
import math
from numbers import Real
import os

from openai import OpenAI

DAILY_VIEW_TYPES = (
    "daily_score_curve",
    "daily_score_heatmap",
    "score_distribution",
    "forecast_observed_rate_distribution",
)
DAILY_METRICS = ("csi", "ets", "pod", "far", "bias_score")
_SYSTEM_PROMPT = f"""你是强对流预报检验平台 Phase 7 的请求解析器，只输出 JSON。
平台按页面声明的源资料时钟（2025 预报/观测均为 UTC；2026 风雷文件名为 BJT、SWAN 头和 CREF 观测均为 UTC），对风雷预报和 CREF 观测做严格同有效时刻、同区域、同网格口径检验。
你只选择参数，不计算、不画图、不写代码，也不能修改数据路径、插值、掩膜、缓存或评分公式。

支持指标：{list(DAILY_METRICS)}
支持视图：
- daily_score_curve：所选起报汇总指标随预报时效曲线；
- daily_score_heatmap：起报时刻×预报时效热力图；
- score_distribution：各时效跨起报样本的分布箱线图。
- forecast_observed_rate_distribution：固定诊断阈值下，所选起报×时效样本的
  预报率—观测率散点及边缘分布；按 6—60、66—120、126—180 分钟分组。

规则：
1. “全套/全面/录屏展示”默认输出四种视图；未指定指标用 csi。
2. 未指定阈值用 [5,15,25,35]，诊断阈值默认 15 且必须包含在 thresholds_dbz 中。
3. 未指定时效用 6—180 分钟；所有时效必须是 6 的倍数。
4. Taylor、FSS、动画、自由排版等未支持要求放入 unmatched_requests，不得替换成别的图。
5. 用户只要求某一种支持视图时，只输出该视图。
6. requested_date 使用 YYYY-MM-DD；没有日期则为 null。
7. 用户要求“预报率和观测率”“forecast rate 与 observed/base rate”或“频率偏差散点”时，
   输出 forecast_observed_rate_distribution；其阈值使用 diagnostic_threshold_dbz。

JSON 结构：
{{
  "requested_date": "2026-04-07",
  "metric": "csi",
  "thresholds_dbz": [5, 15, 25, 35],
  "diagnostic_threshold_dbz": 15,
  "lead_range_minutes": {{"start": 6, "end": 180}},
  "views": ["daily_score_curve", "daily_score_heatmap", "score_distribution", "forecast_observed_rate_distribution"],
  "unmatched_requests": [],
  "notes": ""
}}"""


def default_daily_request(requested_date=None) -> dict:
    return {
        "requested_date": requested_date,
        "metric": "csi",
        "thresholds_dbz": [5.0, 15.0, 25.0, 35.0],
        "diagnostic_threshold_dbz": 15.0,
        "lead_range_minutes": {"start": 6, "end": 180},
        "views": list(DAILY_VIEW_TYPES),
        "unmatched_requests": [],
        "notes": "",
    }


def _lead(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or not 6 <= value <= 180:
        raise ValueError(f"{field} must be an integer from 6 to 180")
    if value % 6:
        raise ValueError(f"{field} must be a multiple of 6 minutes")
    return value


def validate_daily_request(parsed) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("daily request must be a JSON object")
    requested_date = parsed.get("requested_date")
    if requested_date is not None:
        if not isinstance(requested_date, str):
            raise ValueError("requested_date must be YYYY-MM-DD")
        requested_date = date.fromisoformat(requested_date).isoformat()
    metric = str(parsed.get("metric", "csi")).strip().lower()
    if metric not in DAILY_METRICS:
        raise ValueError(f"unsupported daily metric: {metric}")
    raw_thresholds = parsed.get("thresholds_dbz", [5, 15, 25, 35])
    if not isinstance(raw_thresholds, list) or not raw_thresholds:
        raise ValueError("thresholds_dbz must be a non-empty list")
    thresholds = []
    for value in raw_thresholds:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("thresholds_dbz values must be numeric")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 75:
            raise ValueError("thresholds_dbz values must be between 0 and 75")
        if value not in thresholds:
            thresholds.append(value)
    if len(thresholds) > 8:
        raise ValueError("at most eight thresholds are supported")
    diagnostic = parsed.get("diagnostic_threshold_dbz", 15)
    if isinstance(diagnostic, bool) or not isinstance(diagnostic, Real):
        raise ValueError("diagnostic_threshold_dbz must be numeric")
    diagnostic = float(diagnostic)
    if diagnostic not in thresholds:
        raise ValueError("diagnostic_threshold_dbz must be included in thresholds_dbz")
    lead_range = parsed.get("lead_range_minutes", {})
    if not isinstance(lead_range, dict):
        raise ValueError("lead_range_minutes must be an object")
    start = _lead(lead_range.get("start", 6), "lead_range_minutes.start")
    end = _lead(lead_range.get("end", 180), "lead_range_minutes.end")
    if start > end:
        raise ValueError("lead range start cannot exceed end")
    raw_views = parsed.get("views", list(DAILY_VIEW_TYPES))
    if not isinstance(raw_views, list):
        raise ValueError("views must be a list")
    views = []
    unmatched = parsed.get("unmatched_requests", [])
    if not isinstance(unmatched, list) or not all(isinstance(item, str) for item in unmatched):
        raise ValueError("unmatched_requests must be a string list")
    unmatched = [item.strip()[:300] for item in unmatched if item.strip()]
    for item in raw_views:
        normalized = str(item).strip().lower()
        if normalized in DAILY_VIEW_TYPES and normalized not in views:
            views.append(normalized)
        elif normalized:
            unmatched.append(f"暂不支持图表：{normalized}")
    if not views and not unmatched:
        views = list(DAILY_VIEW_TYPES)
    notes = str(parsed.get("notes", ""))[:1000]
    return {
        "requested_date": requested_date,
        "metric": metric,
        "thresholds_dbz": thresholds,
        "diagnostic_threshold_dbz": diagnostic,
        "lead_range_minutes": {"start": start, "end": end},
        "views": views,
        "unmatched_requests": list(dict.fromkeys(unmatched)),
        "notes": notes,
    }


def parse_daily_request(
    user_text,
    *,
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
        raise ValueError("DeepSeek returned an empty daily request")
    return validate_daily_request(json.loads(response.choices[0].message.content))

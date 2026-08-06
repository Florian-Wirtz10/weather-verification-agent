"""Closed natural-language request schema for the zero-shot benchmark."""

from __future__ import annotations

import json
import math
from numbers import Real
import os

from openai import OpenAI


BENCHMARK_METRICS = ("csi", "pod", "far", "bias_score", "mae", "rmse", "fss")
BENCHMARK_VIEWS = (
    "score_curve",
    "fss_curve",
    "period_summary",
    "score_heatmap",
    "spatial_comparison",
)

_SYSTEM_PROMPT = f"""你是中国雷达CREF Zero-shot Benchmark的受限请求规划器，只输出JSON。
页面已由用户选择模型和起报样本。你不能更改模型、样本、数据、坐标、时间匹配、评分公式或写绘图代码。
可选指标：{list(BENCHMARK_METRICS)}。
可选图表：
- score_curve：CSI/POD/FAR/Bias/MAE/RMSE随预报时效变化；
- fss_curve：FSS随预报时效变化，可选1/3/5像元邻域；
- period_summary：整个所选共同时间段的多模型汇总比较；
- score_heatmap：模型×预报时效热力图；
- spatial_comparison：单一起报、单个时效的多模型预报与CREF空间对比。
规则：
1. 全部或自定义多起报不支持空间图；单一起报可支持所有图。
2. 默认阈值20、35、45 dBZ，默认指标CSI、POD、FAR、FSS。
3. 时效必须为6分钟倍数，且不得超过服务端给出的共同最大时效。
4. FSS邻域只能是1、3、5像元。
5. 不支持的要求必须放入unmatched_requests，不能偷偷替换。
6. 用户要求全面展示时，多起报使用全部四类统计图，单起报额外加入空间图。
JSON结构：
{{"metrics":["csi","pod","far","fss"],"thresholds_dbz":[20,35,45],
"lead_range_minutes":{{"start":6,"end":72}},"views":["score_curve","fss_curve","period_summary","score_heatmap"],
"diagnostic_threshold_dbz":35,"spatial_lead_minutes":null,"fss_windows":[1,3,5],
"unmatched_requests":[],"notes":""}}"""


def default_benchmark_request(*, single: bool, common_horizon: int) -> dict:
    return {
        "metrics": ["csi", "pod", "far", "fss"],
        "thresholds_dbz": [20.0, 35.0, 45.0],
        "lead_range_minutes": {"start": 6, "end": int(common_horizon)},
        "views": [
            *(["spatial_comparison"] if single else []),
            "score_curve",
            "fss_curve",
            "period_summary",
            "score_heatmap",
        ],
        "diagnostic_threshold_dbz": 35.0,
        "spatial_lead_minutes": min(60, int(common_horizon)) if single else None,
        "fss_windows": [1, 3, 5],
        "unmatched_requests": [],
        "notes": "",
    }


def validate_benchmark_request(parsed, *, single: bool, common_horizon: int) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("benchmark request must be a JSON object")
    metrics = []
    for value in parsed.get("metrics", ["csi", "pod", "far", "fss"]):
        metric = str(value).strip().lower()
        if metric not in BENCHMARK_METRICS:
            raise ValueError(f"unsupported benchmark metric: {metric}")
        if metric not in metrics:
            metrics.append(metric)
    if not metrics:
        raise ValueError("select at least one benchmark metric")
    thresholds = []
    for value in parsed.get("thresholds_dbz", [20, 35, 45]):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("benchmark thresholds must be numeric")
        threshold = float(value)
        if not math.isfinite(threshold) or not 0 <= threshold <= 75:
            raise ValueError("benchmark thresholds must be between 0 and 75 dBZ")
        if threshold not in thresholds:
            thresholds.append(threshold)
    if not thresholds or len(thresholds) > 6:
        raise ValueError("select one to six benchmark thresholds")
    lead = parsed.get("lead_range_minutes", {})
    if not isinstance(lead, dict):
        raise ValueError("lead_range_minutes must be an object")
    start, end = lead.get("start", 6), lead.get("end", common_horizon)
    for value, label in ((start, "start"), (end, "end")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 6 or value % 6:
            raise ValueError(f"benchmark lead {label} must be a positive multiple of 6")
    if start > end or end > common_horizon:
        raise ValueError(f"selected models only share forecasts through +{common_horizon} minutes")
    unmatched = parsed.get("unmatched_requests", [])
    if not isinstance(unmatched, list) or not all(isinstance(item, str) for item in unmatched):
        raise ValueError("unmatched_requests must be a string list")
    views = []
    for value in parsed.get(
        "views", default_benchmark_request(single=single, common_horizon=common_horizon)["views"]
    ):
        view = str(value).strip().lower()
        if view not in BENCHMARK_VIEWS:
            unmatched.append(f"暂不支持图表：{view}")
        elif view == "spatial_comparison" and not single:
            unmatched.append("空间对比图仅支持单一起报")
        elif view not in views:
            views.append(view)
    diagnostic = float(parsed.get("diagnostic_threshold_dbz", thresholds[0]))
    if diagnostic not in thresholds:
        raise ValueError("diagnostic_threshold_dbz must be included in thresholds_dbz")
    spatial = parsed.get("spatial_lead_minutes")
    if spatial is not None:
        if isinstance(spatial, bool) or not isinstance(spatial, int) or spatial % 6:
            raise ValueError("spatial lead must be a 6-minute step")
        if not start <= spatial <= end:
            raise ValueError("spatial lead must be inside the requested lead range")
    windows = []
    for value in parsed.get("fss_windows", [1, 3, 5]):
        if value not in {1, 3, 5}:
            raise ValueError("FSS windows must be selected from 1, 3 and 5 pixels")
        if value not in windows:
            windows.append(value)
    if not windows:
        windows = [1]
    if "fss_curve" in views and "fss" not in metrics:
        views.remove("fss_curve")
        unmatched.append("FSS曲线需要在metrics中选择fss")
    return {
        "metrics": metrics,
        "thresholds_dbz": thresholds,
        "lead_range_minutes": {"start": start, "end": end},
        "views": views,
        "diagnostic_threshold_dbz": diagnostic,
        "spatial_lead_minutes": spatial,
        "fss_windows": windows,
        "unmatched_requests": list(
            dict.fromkeys(item.strip() for item in unmatched if item.strip())
        ),
        "notes": str(parsed.get("notes", ""))[:1000],
    }


def parse_benchmark_request(
    user_text,
    *,
    single: bool,
    common_horizon: int,
    model="deepseek-chat",
    api_key=None,
    base_url="https://api.deepseek.com",
) -> dict:
    client = OpenAI(api_key=api_key or os.environ["DEEPSEEK_API_KEY"], base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{_SYSTEM_PROMPT}\n当前是否单一起报：{single}；"
                    f"当前模型共同最大时效：{common_horizon}分钟。"
                ),
            },
            {"role": "user", "content": str(user_text)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("DeepSeek returned an empty benchmark request")
    return validate_benchmark_request(
        json.loads(response.choices[0].message.content),
        single=single,
        common_horizon=common_horizon,
    )

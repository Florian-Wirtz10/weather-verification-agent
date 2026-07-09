"""Natural-language request parsing via DeepSeek (OpenAI-compatible API).

The model is only allowed to select from and parameterize a fixed menu of
already-implemented plotting functions (see visualization.plots) — it never
generates, sees, or suggests plotting code. `_validate` is the enforcement
point: anything the model returns that isn't a literal key of PLOT_MENU is
dropped before it ever reaches the pipeline.
"""

import json
import os

from openai import OpenAI

PLOT_MENU = {
    "reflectivity_field": "单个时刻的反射率空间分布图 (dBZ)",
    "reflectivity_error": "预报减观测的反射率误差空间分布图",
    "csi_time_series": "CSI 评分随预报时效变化曲线",
    "pod_time_series": "命中率 (POD) 随预报时效变化曲线",
    "far_time_series": "空报率 (FAR) 随预报时效变化曲线",
    "bias_score_time_series": "频率偏差随预报时效变化曲线",
    "ets_time_series": "公平技巧评分 (ETS) 随预报时效变化曲线",
}

DEFAULT_THRESHOLDS = (10.0, 20.0, 35.0)

_SYSTEM_PROMPT = """你是天气预报检验智能体的请求解析器。将用户的自然语言请求解析为结构化 JSON 参数，
供已经实现好的检验流程调用。你只能从下面这个"图表菜单"里选择要生成哪些图，绝不能编写、
输出或建议任何绘图代码或其它代码——只输出图表名称和参数。

可选图表 (plots 字段，从下列 key 中选择一个或多个):
{plot_menu}

只输出如下 JSON，不要输出其它文字、不要用 markdown 代码块包裹:
{{
  "thresholds": [数字, ...],
  "plots": ["图表菜单里的 key", ...],
  "notes": "一句话说明你是如何理解这个请求的"
}}
""".format(plot_menu=json.dumps(PLOT_MENU, ensure_ascii=False, indent=2))


def parse_request(user_text, *, model="deepseek-chat", api_key=None, base_url="https://api.deepseek.com"):
    """Turn a natural-language verification request into {thresholds, plots, notes}."""
    client = OpenAI(api_key=api_key or os.environ["DEEPSEEK_API_KEY"], base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return _validate(json.loads(response.choices[0].message.content))


FIGURE_KEYS_BY_PLOT = {
    "reflectivity_field": ["forecast_first_lead", "obs_first_lead", "forecast_last_lead", "obs_last_lead"],
    "reflectivity_error": ["error_first_lead", "error_last_lead"],
    "csi_time_series": ["csi_time_series"],
    "pod_time_series": ["pod_time_series"],
    "far_time_series": ["far_time_series"],
    "bias_score_time_series": ["bias_score_time_series"],
    "ets_time_series": ["ets_time_series"],
}


def select_figures(figures, plots):
    """Map the model's chosen PLOT_MENU keys onto the actual figures a run produced,
    preserving `plots` order and dropping duplicates."""
    ordered_keys = []
    seen = set()
    for plot in plots:
        for key in FIGURE_KEYS_BY_PLOT.get(plot, []):
            if key not in seen:
                seen.add(key)
                ordered_keys.append(key)
    return {key: figures[key] for key in ordered_keys if key in figures}


def _validate(parsed):
    """Coerce/whitelist the model's raw JSON. Never trust plot names it invents."""
    try:
        thresholds = [float(t) for t in parsed.get("thresholds") or DEFAULT_THRESHOLDS]
    except (TypeError, ValueError):
        thresholds = list(DEFAULT_THRESHOLDS)

    requested_plots = parsed.get("plots") or []
    plots = [name for name in requested_plots if name in PLOT_MENU]
    if not plots:
        plots = list(PLOT_MENU)

    return {
        "thresholds": thresholds,
        "plots": plots,
        "notes": str(parsed.get("notes", "")),
    }

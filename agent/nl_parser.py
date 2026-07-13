"""DeepSeek request parsing into the closed verification-request schema.

The model only proposes JSON. :func:`agent.request_schema.validate_request`
is the authority that decides which values are safe to execute.
"""

import json
import os

from openai import OpenAI

from agent.request_schema import (
    DEFAULT_THRESHOLDS_DBZ,
    SCHEMA_VERSION,
    SUPPORTED_METRICS,
    SUPPORTED_LAYOUTS,
    SUPPORTED_VIEW_TYPES,
    validate_request,
)


def _build_system_prompt(available_cases=None, previous_request=None):
    cases = available_cases or []
    case_summary = [
        {
            "case_id": case.get("case_id"),
            "date": case.get("date"),
            "init_time_utc": str(case.get("init_time")),
            "available_leads": case.get("available_leads", []),
        }
        for case in cases
    ]
    previous = previous_request if isinstance(previous_request, dict) else None
    return f"""你是风雷预报检验助手的请求解析器，不是实时天气问答助手，也不是自由制图工具。
你的唯一任务是把用户请求转换成受限 JSON；你不计算指标、不画图、不编写代码。

产品当前只支持：已接入的风雷 V1.0 REFMOSAIC 反射率预报与雷达 CREF 观测检验。
当前可用个例：
{json.dumps(case_summary, ensure_ascii=False)}

intent 只能是：
- verify_case：检验当前已接入个例
- show_capabilities：询问系统能做什么
- explain_last_result：要求解释上一轮结果
- out_of_scope：实时天气、未来天气问答或其它业务范围外请求
- unknown：无法可靠理解，需要追问

支持的指标：{json.dumps(SUPPORTED_METRICS, ensure_ascii=False)}
支持的视图类型：{json.dumps(SUPPORTED_VIEW_TYPES, ensure_ascii=False)}
- score_time_series：必须带一个 metric，一个请求对应一张图
- reflectivity_comparison：必须带 lead，值为 first、last 或 0—180 整数
- reflectivity_error：必须带 lead，值为 first、last 或 0—180 整数
- reflectivity_animation：必须带 lead_range_minutes={{"start": 6, "end": 24}} 和可选 fps（1—4）；
  只生成确定性“预报｜实况”对比动画，不能要求任意动画或自由代码。

支持的展示布局：{json.dumps(SUPPORTED_LAYOUTS, ensure_ascii=False)}。
布局只决定已支持图表的网页排列和顺序，不能改变数据、评分、配对或绘图方法。
支持的展示策略：auto、separate_media、linked_score_animation。
linked_score_animation 只用于“空间回波演变与某一评分变化之间的关系”：它要求同时提出
一个 reflectivity_animation 和对应的 score_time_series，并在 presentation.metric 中指定该评分。

上一轮已验证请求（若为空则表示没有上一轮）：
{json.dumps(previous, ensure_ascii=False)}

严格规则：
1. 用户没有给阈值时使用 {list(DEFAULT_THRESHOLDS_DBZ)}；“35 dBZ以上”解析为 [35]。
2. “前1小时”解析为 lead_range_minutes={{"start": 0, "end": 60}}。
3. 用户要求首尾对比时，分别输出 lead=first 和 lead=last 的两个视图。
4. Taylor、FSS、log坐标等未支持要求必须原样概括到 unmatched_requests；不得静默丢弃，
   不得擅自替换成其它图，也不得为了凑结果输出全部图。
5. 同时含支持和不支持项时，只把明确支持项放进 views，不支持项放 unmatched_requests。
6. 询问未接入日期的风雷检验仍属于 verify_case，并填写 requested_date；是否有数据由服务端判断。
7. “今天天气怎么样”等实时天气请求必须是 out_of_scope，views 为空。
8. 不得让自然语言修改数据路径、API地址、模型名、Python代码或执行环境。
9. notes 只简述理解，不承担是否可执行的判断。
10. 如果用户说“保留刚才”“在上一结果基础上修改”等，以上一轮请求为基线，仅修改用户明确
    要求修改的字段；仍必须输出完整 JSON。
11. 只有用户明确说“单列、双列、2×2、报告式、动画优先”等布局时，才设置对应 layout；
    否则输出 layout="auto"，由确定性页面根据实际生成的媒体决定布局。
12. 用户明确要看空间回波的“演变、过程、动态、逐时变化”时，应提出一个
    reflectivity_animation（使用其时效范围，默认 fps=2）；普通 CSI/ETS 等评分时序曲线
    仍应保持 score_time_series，不能仅因横轴是时效就擅自增加动画。
13. 用户要求把空间演变与 CSI、ETS 等某项评分联系起来、同屏对应或解释两者关系时，设置
    strategy="linked_score_animation" 和对应 metric。其它情形 strategy="auto"；不得把多个
    评分曲线或未请求的指标塞进联动动画。

只输出 JSON 对象，不使用 Markdown。结构如下：
{{
  "schema_version": "{SCHEMA_VERSION}",
  "intent": "verify_case",
  "filters": {{
    "thresholds_dbz": [10, 20, 35],
    "lead_range_minutes": {{"start": 0, "end": 180}},
    "requested_date": null
  }},
  "views": [{{"type": "score_time_series", "metric": "csi"}}],
  "presentation": {{"layout": "auto", "strategy": "auto", "metric": null}},
  "insight": {{"enabled": true, "length": "detailed"}},
  "unmatched_requests": [],
  "notes": ""
}}
"""


def parse_request(
    user_text,
    *,
    available_cases=None,
    previous_request=None,
    model="deepseek-chat",
    api_key=None,
    base_url="https://api.deepseek.com",
):
    """Parse text into a validated declarative verification request."""
    client = OpenAI(api_key=api_key or os.environ["DEEPSEEK_API_KEY"], base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _build_system_prompt(available_cases, previous_request)},
            {"role": "user", "content": str(user_text)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    if not response.choices or response.choices[0].message.content is None:
        raise ValueError("DeepSeek returned an empty response")
    return validate_request(json.loads(response.choices[0].message.content))


def _validate(parsed):
    """Backward-compatible test seam; validation lives in request_schema."""
    return validate_request(parsed)

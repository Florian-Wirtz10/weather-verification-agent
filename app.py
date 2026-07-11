"""Streamlit entry point for deterministic verification plus constrained LLM I/O."""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import (
    AmbiguousCaseError,
    DataContractError,
    NoDataError,
    RequestValidationError,
    build_insight_facts,
    default_request,
    discover_fenglei_cases,
    generate_insight,
    parse_request,
    render_deterministic_summary,
    resolve_request,
    run_fenglei_case_verification,
    run_verification,
)
from reports import save_report

load_dotenv()

st.set_page_config(page_title="Weather Verification Agent", layout="wide")
st.title("Weather Verification Agent")

tab_synthetic, tab_real_case = st.tabs(["Phase 1: 合成数据 demo", "Phase 3: 风雷检验助手"])

with tab_synthetic:
    st.caption("Phase 1: synthetic precipitation fields, categorical + continuous verification metrics.")

    default_thresholds = os.getenv("DEFAULT_THRESHOLDS", "1,5,10")

    with st.sidebar:
        st.header("Phase 1 run settings")
        grid_size = st.number_input("Grid size (N x N)", min_value=8, max_value=256, value=64, step=8)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        thresholds_input = st.text_input(
            "Precipitation thresholds (mm, comma-separated)", value=default_thresholds
        )
        run_clicked = st.button("Generate & verify", type="primary")

    if run_clicked:
        thresholds = [float(t.strip()) for t in thresholds_input.split(",") if t.strip()]
        result = run_verification(shape=(grid_size, grid_size), seed=seed, thresholds=thresholds)
        st.session_state["result"] = result

    result = st.session_state.get("result")

    if result is None:
        st.info("Set parameters in the sidebar and click **Generate & verify** to run the pipeline.")
    else:
        figures = result["figures"]
        col1, col2, col3 = st.columns(3)
        col1.pyplot(figures["forecast"])
        col2.pyplot(figures["obs"])
        col3.pyplot(figures["error"])

        metrics = result["metrics"]

        st.subheader("Continuous error metrics")
        st.table(pd.DataFrame([metrics["continuous"]]))

        st.subheader("Categorical scores by threshold")
        categorical_df = pd.DataFrame(metrics["categorical"]).T
        categorical_df.index.name = "threshold"
        st.table(categorical_df)

        if st.button("Save report to outputs/"):
            outdir = save_report(metrics, figures)
            st.success(f"Saved figures/tables/report to {outdir}/")

def _execute_real_case_request(
    request,
    *,
    cases,
    selected_case_id,
    fenglei_dir,
    radar_dir,
    bbox,
    allow_llm_insight,
):
    """Resolve, execute and interpret one validated request without silent fallbacks."""
    decision = resolve_request(
        request,
        cases,
        selected_case_id=selected_case_id,
    )
    attempt = {"decision": decision, "request": request}
    if decision["status"] not in {"ready", "partial"}:
        return attempt

    execution = decision["execution"]
    lead_filter = execution["lead_range"]
    result = run_fenglei_case_verification(
        fenglei_dir=fenglei_dir,
        radar_dir=radar_dir,
        bbox=bbox,
        thresholds=tuple(execution["thresholds"]),
        init_time=execution["init_time"],
        lead_range=(lead_filter["start"], lead_filter["end"]),
        views=execution["views"],
    )
    if result["coverage"]["unavailable_views"] and decision["status"] == "ready":
        decision = {
            **decision,
            "status": "partial",
            "message": "部分空间图没有对应的精确预报时效，已只生成有数据的视图。",
        }
        attempt["decision"] = decision
    facts = build_insight_facts(result)
    deterministic_summary = render_deterministic_summary(facts)
    ai_insight = None
    insight_error = None
    if execution["insight"]["enabled"] and allow_llm_insight:
        try:
            ai_insight = generate_insight(facts, length=execution["insight"]["length"])
        except Exception as exc:
            insight_error = str(exc)

    attempt.update(
        {
            "result": result,
            "facts": facts,
            "deterministic_summary": deterministic_summary,
            "ai_insight": ai_insight,
            "insight_error": insight_error,
        }
    )
    return attempt


def _metric_rows(result):
    rows = []
    for entry in result["per_lead"]:
        for threshold, scores in entry["categorical"].items():
            table = entry["contingency"][threshold]
            rows.append(
                {
                    "lead_minutes": entry["lead_minutes"],
                    "valid_time_utc": entry["valid_time"],
                    "threshold_dbz": threshold,
                    **scores,
                    **table,
                }
            )
    return rows


with tab_real_case:
    st.caption(
        "受限的风雷 V1.0 REFMOSAIC 反射率预报检验助手：自然语言只生成经过白名单验证的"
        "检验计划，指标计算和绘图全部由确定性代码完成。"
    )

    with st.sidebar:
        st.header("Phase 3 个例设置")
        fenglei_dir = st.text_input(
            "风雷 REFMOSAIC 目录", value=os.getenv("FENGLEI_CASE_DIR", "")
        )
        radar_dir = st.text_input(
            "雷达 CREF 观测目录", value=os.getenv("RADAR_OBS_DIR", "")
        )
        lat_min = st.number_input("lat_min", value=28.0)
        lat_max = st.number_input("lat_max", value=35.0)
        lon_min = st.number_input("lon_min", value=108.0)
        lon_max = st.number_input("lon_max", value=118.0)
        allow_llm_insight = st.checkbox(
            "允许将汇总指标发送给 DeepSeek 生成文字解读",
            value=False,
            help="只发送白名单事实摘要，不发送路径、文件名、原始网格或图片。",
        )

    cases = []
    catalog_error = None
    if fenglei_dir and radar_dir:
        try:
            cases = discover_fenglei_cases(fenglei_dir, radar_dir)
        except Exception as exc:
            catalog_error = str(exc)

    selected_case_id = None
    if cases:
        labels = {
            case["case_id"]: (
                f"{case['init_time']:%Y-%m-%d %H:%M} UTC — "
                f"{len(case['available_leads'])} 个匹配时效"
            )
            for case in cases
        }
        selected_case_id = st.selectbox(
            "选择风雷起报",
            options=list(labels),
            format_func=lambda value: labels[value],
        )
    elif catalog_error:
        st.error(f"资料目录检查失败：{catalog_error}")
    else:
        st.warning("当前目录下没有严格时间匹配的风雷/雷达个例。")

    nl_request = st.text_area(
        "用自然语言描述检验要求",
        value="使用35 dBZ阈值，只看前1小时，生成CSI和ETS两张曲线，并简要解读结果。",
        help="不支持或无数据的请求会明确说明，不会自动换成全部图表。",
    )
    button_col1, button_col2 = st.columns(2)
    parse_clicked = button_col1.button("解析并运行", type="primary")
    manual_clicked = button_col2.button("使用明确默认方案运行")
    current_scope = (
        fenglei_dir,
        radar_dir,
        float(lat_min),
        float(lat_max),
        float(lon_min),
        float(lon_max),
        selected_case_id,
    )

    previous_attempt = st.session_state.get("fenglei_attempt")
    if previous_attempt is not None and previous_attempt.get("scope") != current_scope:
        st.session_state.pop("fenglei_attempt", None)

    if parse_clicked or manual_clicked:
        st.session_state.pop("fenglei_attempt", None)
        run_attempt: dict
        try:
            request = default_request() if manual_clicked else parse_request(
                nl_request, available_cases=cases
            )
        except (RequestValidationError, ValueError, KeyError, TypeError) as exc:
            run_attempt = {
                "decision": {
                    "status": "parser_error",
                    "message": str(exc),
                }
            }
        except Exception as exc:
            run_attempt = {"decision": {"status": "parser_error", "message": str(exc)}}
        else:
            try:
                run_attempt = _execute_real_case_request(
                    request,
                    cases=cases,
                    selected_case_id=selected_case_id,
                    fenglei_dir=fenglei_dir,
                    radar_dir=radar_dir,
                    bbox=(lat_min, lat_max, lon_min, lon_max),
                    allow_llm_insight=allow_llm_insight,
                )
            except NoDataError as exc:
                run_attempt = {
                    "request": request,
                    "decision": {"status": "no_data", "message": str(exc)},
                }
            except AmbiguousCaseError as exc:
                run_attempt = {
                    "request": request,
                    "decision": {"status": "needs_clarification", "message": str(exc)},
                }
            except DataContractError as exc:
                run_attempt = {
                    "request": request,
                    "decision": {"status": "execution_error", "message": str(exc)},
                }
            except Exception as exc:
                run_attempt = {
                    "request": request,
                    "decision": {"status": "execution_error", "message": str(exc)},
                }
        run_attempt["scope"] = current_scope
        st.session_state["fenglei_attempt"] = run_attempt

    attempt = st.session_state.get("fenglei_attempt")
    if attempt is None:
        st.info("输入检验要求并运行；也可以绕过自然语言，主动使用页面明确展示的默认方案。")
    else:
        decision = attempt["decision"]
        status = decision["status"]
        if status == "ready":
            st.success(decision["message"])
        elif status == "partial":
            st.warning(decision["message"])
        elif status in {"no_data", "unsupported", "parser_error", "execution_error"}:
            st.error(decision["message"])
        else:
            st.info(decision["message"])

        if "request" in attempt:
            with st.expander("查看规范化请求与执行口径", expanded=status in {"partial", "unsupported"}):
                st.json(attempt["request"])

        result = attempt.get("result")
        if result is not None:
            coverage = result["coverage"]
            st.write(
                f"**实际时效：** {coverage['selected_leads']} 分钟  "
                f"\n**缺失时效：** {coverage['missing_leads'] or '无'}"
            )
            for warning in result["warnings"]:
                st.warning(warning)

            st.subheader("结果解读")
            st.text(attempt["deterministic_summary"])
            if attempt.get("ai_insight"):
                st.info("AI 解读（基于上述确定性事实）\n\n" + attempt["ai_insight"])
            elif attempt.get("insight_error"):
                st.caption("DeepSeek 解读不可用，已保留确定性摘要：" + attempt["insight_error"])

            st.subheader("按请求生成的图表")
            for key, figure in result["figures"].items():
                st.caption(str(result["figure_metadata"][key]))
                st.pyplot(figure)

            st.subheader("各预报时效分类检验指标与样本数")
            st.dataframe(pd.DataFrame(_metric_rows(result)), width="stretch")

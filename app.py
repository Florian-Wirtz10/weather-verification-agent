"""Streamlit entry point for deterministic verification plus constrained LLM I/O."""

import base64
import os

import matplotlib.pyplot as plt
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
    resolve_media_layout,
    render_deterministic_summary,
    resolve_request,
    run_fenglei_case_verification,
    run_verification,
)
from reports import build_case_download_artifacts, describe_view, save_report

load_dotenv()

st.set_page_config(page_title="Weather Verification Agent", layout="wide")
st.title("Weather Verification Agent")
st.markdown(
    """
    <style>
    @media (min-width: 960px) {
      div[data-testid="stHorizontalBlock"]:has(.ai-insight-anchor)
        > div[data-testid="stColumn"]:last-child {
          align-self: flex-start;
          position: sticky;
          top: 4.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tab_synthetic, tab_real_case = st.tabs(["Phase 1: 合成数据 demo", "Phase 4: 风雷检验助手"])

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
        presentation=execution["presentation"],
    )
    coverage_reasons = []
    if result["coverage"]["missing_leads"]:
        coverage_reasons.append(
            f"请求范围内有 {len(result['coverage']['missing_leads'])} 个时效没有严格匹配资料"
        )
    if result["coverage"]["unavailable_views"]:
        coverage_reasons.append(
            f"有 {len(result['coverage']['unavailable_views'])} 张空间图没有对应的精确预报时效"
        )
    if coverage_reasons:
        coverage_message = "；".join(coverage_reasons)
        message = (
            f"{decision['message']} 另外，{coverage_message}。"
            if decision["status"] == "partial"
            else f"请求只能部分执行：{coverage_message}；已仅生成有数据的结果。"
        )
        decision = {
            **decision,
            "status": "partial",
            "message": message,
        }
        attempt["decision"] = decision
    requested_metrics = [
        view["metric"]
        for view in execution["views"]
        if view["type"] == "score_time_series"
    ]
    facts = build_insight_facts(result, requested_metrics=requested_metrics)
    deterministic_summary = render_deterministic_summary(facts)
    ai_insight = None
    insight_error = None
    try:
        ai_insight = generate_insight(facts, length=execution["insight"]["length"])
    except Exception as exc:
        insight_error = str(exc)

    metric_rows = _metric_rows(result, requested_metrics)
    try:
        downloads = build_case_download_artifacts(
            result=result,
            request=request,
            metric_rows=metric_rows,
            facts=facts,
            deterministic_summary=deterministic_summary,
            ai_insight=ai_insight,
            execution_status=attempt["decision"]["status"],
        )
    finally:
        for figure in result["figures"].values():
            plt.close(figure)
    result["figures"] = {}

    attempt.update(
        {
            "result": result,
            "facts": facts,
            "deterministic_summary": deterministic_summary,
            "ai_insight": ai_insight,
            "insight_error": insight_error,
            "insight_retry_failed": False,
            "requested_metrics": requested_metrics,
            "metric_rows": metric_rows,
            "downloads": downloads,
        }
    )
    return attempt


def _metric_rows(result, requested_metrics):
    rows = []
    for entry in result["per_lead"]:
        for threshold, scores in entry["categorical"].items():
            table = entry["contingency"][threshold]
            selected_scores = {
                metric: scores[metric]
                for metric in requested_metrics
                if metric in scores
            }
            valid_time = entry["valid_time"]
            valid_time_utc = (
                valid_time.isoformat(timespec="seconds") + "Z"
                if hasattr(valid_time, "isoformat")
                else str(valid_time)
            )
            rows.append(
                {
                    "lead_minutes": entry["lead_minutes"],
                    "valid_time_utc": valid_time_utc,
                    "threshold_dbz": threshold,
                    **selected_scores,
                    **table,
                }
            )
    return rows


def _render_professional_insight(attempt, result, metric_rows, downloads):
    """Render a sticky, independently scrollable evidence-bounded AI panel."""
    st.markdown('<span class="ai-insight-anchor"></span>', unsafe_allow_html=True)
    st.subheader("AI 解读")
    with st.container(height=680, border=False):
        st.caption("仅基于本次确定性评分、样本数与图表目录生成；不读取原始网格或图片。")
        if st.button("重新生成解读", key=f"insight_{result['context']['case_id']}"):
            with st.spinner("正在基于当前确定性事实生成 AI 解读……"):
                previous_insight = attempt.get("ai_insight")
                try:
                    candidate_insight = generate_insight(
                        attempt["facts"], length=attempt["request"]["insight"]["length"]
                    )
                    candidate_downloads = build_case_download_artifacts(
                        result=result,
                        request=attempt["request"],
                        metric_rows=metric_rows,
                        facts=attempt["facts"],
                        deterministic_summary=attempt["deterministic_summary"],
                        ai_insight=candidate_insight,
                        execution_status=attempt["decision"]["status"],
                        rendered_images=downloads["images"],
                    )
                except Exception as exc:
                    attempt["insight_error"] = str(exc)
                    attempt["insight_retry_failed"] = previous_insight is not None
                else:
                    attempt["ai_insight"] = candidate_insight
                    attempt["insight_error"] = None
                    attempt["insight_retry_failed"] = False
                    attempt["downloads"] = candidate_downloads
                st.session_state["fenglei_attempt"] = attempt
                st.rerun()
        if attempt.get("ai_insight"):
            st.markdown(attempt["ai_insight"])
        else:
            st.info("确定性摘要")
            st.write(attempt["deterministic_summary"])
        if attempt.get("insight_retry_failed"):
            st.warning("重新生成失败，页面和下载包仍保留上一次成功解读。")
        elif attempt.get("insight_error"):
            st.caption("AI 解读暂不可用；已保留确定性摘要。")


def _render_requested_media(result, downloads, request):
    """Display pre-rendered bytes in one of the schema-approved layouts."""
    layout = resolve_media_layout(
        request.get("presentation", {}).get("layout", "auto"),
        figure_metadata=result["figure_metadata"],
        animation_metadata=result["animation_metadata"],
    )
    static_items = [
        ("image", key, item, result["figure_metadata"][key])
        for key, item in downloads["images"].items()
    ]
    animation_items = [
        ("animation", key, item, result["animation_metadata"][key])
        for key, item in downloads.get("animations", {}).items()
    ]
    items = animation_items + static_items if layout == "animation_focus" else static_items + animation_items
    if not items:
        st.info("本次请求未生成可展示的图表或动画。")
        return

    threshold_text = "、".join(f"{float(value):g}" for value in request["filters"]["thresholds_dbz"])

    def show_item(index, item):
        kind, key, media, view = item
        caption = [f"第 {index}/{len(items)} 项", describe_view(view)]
        if view["type"] == "score_time_series":
            caption.append(f"阈值 {threshold_text} dBZ")
        caption.append(f"起报 {result['context']['init_time']:%Y-%m-%d %H:%M} UTC")
        st.caption("｜".join(caption))
        if kind == "animation":
            # st.image resizes wide GIFs through Pillow, which serializes only
            # the current/first frame.  A browser data URI preserves the exact
            # original multi-frame GIF and lets CSS handle visual scaling.
            encoded = base64.b64encode(media["data"]).decode("ascii")
            st.markdown(
                '<img alt="预报观测联动动画" '
                f'src="data:image/gif;base64,{encoded}" '
                'style="display:block;width:100%;height:auto;" />',
                unsafe_allow_html=True,
            )
        else:
            st.image(media["data"])
        extension, mime, label = (
            ("gif", "image/gif", "下载该动画（GIF）")
            if kind == "animation"
            else ("png", "image/png", "下载这张图（PNG）")
        )
        st.download_button(
            label,
            data=media["data"],
            file_name=media["file_name"],
            mime=mime,
            key=f"{extension}_{downloads['base_name']}_{key}",
            on_click="ignore",
        )

    if layout in {"two_column", "grid_2x2", "animation_focus"}:
        for start in range(0, len(items), 2):
            columns = st.columns(2)
            for offset, column in enumerate(columns):
                if start + offset < len(items):
                    with column:
                        show_item(start + offset + 1, items[start + offset])
    else:
        for index, item in enumerate(items, start=1):
            show_item(index, item)


with tab_real_case:
    st.caption(
        "受限的风雷 V1.0 REFMOSAIC 反射率预报检验助手：自然语言只生成经过白名单验证的"
        "检验计划，指标计算和绘图全部由确定性代码完成。"
    )

    with st.sidebar:
        st.header("Phase 4 个例设置")
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
    parse_clicked = button_col1.button("解析并运行", type="primary", width="stretch")
    manual_clicked = button_col2.button("使用明确默认方案运行", width="stretch")
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
            with st.spinner("正在解析检验要求……"):
                previous_request = (
                    previous_attempt.get("request")
                    if isinstance(previous_attempt, dict)
                    else None
                )
                request = default_request() if manual_clicked else parse_request(
                    nl_request,
                    available_cases=cases,
                    previous_request=previous_request,
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
                with st.spinner("正在索引资料、计算指标并生成图表……"):
                    run_attempt = _execute_real_case_request(
                        request,
                        cases=cases,
                        selected_case_id=selected_case_id,
                        fenglei_dir=fenglei_dir,
                        radar_dir=radar_dir,
                        bbox=(lat_min, lat_max, lon_min, lon_max),
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
        run_attempt["submitted_prompt"] = None if manual_clicked else nl_request
        st.session_state["fenglei_attempt"] = run_attempt

    attempt = st.session_state.get("fenglei_attempt")
    if attempt is None:
        st.info("输入检验要求并运行；也可以绕过自然语言，主动使用页面明确展示的默认方案。")
    else:
        decision = attempt["decision"]
        status = decision["status"]
        submitted_prompt = attempt.get("submitted_prompt")
        if submitted_prompt is not None and submitted_prompt != nl_request:
            st.warning("当前展示的是上一次已执行结果；文本框中的新要求尚未运行。")
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
        required_result_keys = {
            "request",
            "facts",
            "deterministic_summary",
            "metric_rows",
            "downloads",
        }
        if result is not None and not required_result_keys <= attempt.keys():
            for figure in result.get("figures", {}).values():
                plt.close(figure)
            st.warning("当前页面缓存的是旧版本结果，请重新点击运行以生成可下载的新结果。")
            result = None
        if result is not None:
            coverage = result["coverage"]
            metric_rows = attempt["metric_rows"]
            downloads = attempt["downloads"]
            result_column, insight_column = st.columns([2.2, 1], gap="large")
            with insight_column:
                _render_professional_insight(attempt, result, metric_rows, downloads)
            with result_column:
                missing_leads = coverage["missing_leads"]
                missing_text = "无" if not missing_leads else f"{len(missing_leads)} 个"
                st.write(
                    f"**匹配 {len(coverage['selected_leads'])} 个时效**｜"
                    f"范围 {coverage['selected_leads'][0]}–{coverage['selected_leads'][-1]} 分钟｜"
                    f"缺失 {missing_text}"
                )
                with st.expander("查看完整时效覆盖"):
                    st.write("实际时效（分钟）：", coverage["selected_leads"])
                    st.write("缺失时效（分钟）：", missing_leads or "无")

                st.subheader("下载结果")
                download_col1, download_col2, download_col3 = st.columns(3)
                download_col1.download_button(
                    "下载已生成的部分结果 ZIP" if status == "partial" else "下载完整结果包 ZIP",
                    data=downloads["bundle"]["data"],
                    file_name=downloads["bundle"]["file_name"],
                    mime="application/zip",
                    key=f"bundle_{downloads['base_name']}",
                    on_click="ignore",
                    type="primary",
                    width="stretch",
                )
                download_col2.download_button(
                    "下载本次请求指标 CSV",
                    data=downloads["metrics"]["data"],
                    file_name=downloads["metrics"]["file_name"],
                    mime="text/csv",
                    key=f"metrics_{downloads['base_name']}",
                    on_click="ignore",
                    width="stretch",
                )
                download_col3.download_button(
                    "下载结果解读 Markdown",
                    data=downloads["summary"]["data"],
                    file_name=downloads["summary"]["file_name"],
                    mime="text/markdown",
                    key=f"summary_{downloads['base_name']}",
                    on_click="ignore",
                    width="stretch",
                )

                st.subheader("按请求生成的图表与动画")
                _render_requested_media(result, downloads, attempt["request"])
                st.subheader("本次请求指标与样本数")
                st.dataframe(pd.DataFrame(metric_rows), width="stretch")

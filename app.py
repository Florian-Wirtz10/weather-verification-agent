"""Streamlit entry point for deterministic verification plus constrained LLM I/O."""

from datetime import date
import os
from typing import Any

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
    build_daily_insight_facts,
    default_daily_request,
    default_request,
    discover_fenglei_cases,
    generate_insight,
    generate_daily_insight,
    parse_daily_request,
    parse_request,
    resolve_media_layout,
    render_deterministic_summary,
    render_daily_deterministic_summary,
    resolve_request,
    run_fenglei_case_verification,
    run_verification,
)
from reports import build_case_download_artifacts, describe_view, figure_to_png, save_report
from verification.batch import run_daily_fenglei_verification
from visualization.batch_plots import (
    plot_daily_score_curves,
    plot_daily_score_distribution,
    plot_daily_score_heatmap,
)

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

tab_synthetic, tab_real_case, tab_daily = st.tabs(
    ["Phase 1: 合成数据 demo", "Phase 4: 单起报检验", "Phase 5: 全天公平检验"]
)

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
            # Do not pass a width here: Streamlit's GIF width-processing path
            # can flatten an animation to its first frame.  The renderer emits
            # a 1000 px-wide source, which is large enough for this column and
            # retains Streamlit's native fullscreen viewer at natural size.
            st.image(media["data"])
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
            result_column, insight_column = st.columns([3.2, 1], gap="large")
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


def _daily_media(result, request):
    """Render only schema-approved full-day plots and retain browser-safe PNG bytes."""
    media = {}
    metric = request["metric"]
    threshold = request["diagnostic_threshold_dbz"]
    factories = {
        "daily_score_curve": lambda number: plot_daily_score_curves(
            result["summary"], metric, figure_number=number
        ),
        "daily_score_heatmap": lambda number: plot_daily_score_heatmap(
            result["per_case"], metric, threshold, figure_number=number
        ),
        "score_distribution": lambda number: plot_daily_score_distribution(
            result["per_case"], metric, threshold, figure_number=number
        ),
    }
    labels = {
        "daily_score_curve": f"全天汇总 {metric.upper()} 随预报时效曲线",
        "daily_score_heatmap": f"{threshold:g} dBZ 起报时刻×时效热力图",
        "score_distribution": f"{threshold:g} dBZ 跨起报样本分布图",
    }
    for number, view in enumerate(request["views"], start=1):
        figure = factories[view](number)
        try:
            media[view] = {
                "number": number,
                "label": labels[view],
                "data": figure_to_png(figure),
                "file_name": (
                    f"phase5_{result['context']['case_date_bjt']}_{metric}_{number:02d}_{view}.png"
                ),
            }
        finally:
            plt.close(figure)
    return media


def _execute_daily_request(
    request,
    *,
    forecast_dir,
    observation_dir,
    fallback_date,
    bbox,
    force,
    progress_callback,
):
    if not request["views"]:
        return {
            "status": "unsupported",
            "message": "本次请求没有已实现的全天视图，因此没有执行计算。",
            "request": request,
        }
    requested_date = request["requested_date"] or fallback_date
    lead_range = request["lead_range_minutes"]
    result = run_daily_fenglei_verification(
        forecast_dir=forecast_dir,
        observation_dir=observation_dir,
        case_date_bjt=requested_date,
        bbox=bbox,
        thresholds=request["thresholds_dbz"],
        lead_range=(lead_range["start"], lead_range["end"]),
        force=force,
        progress_callback=progress_callback,
    )
    media = _daily_media(result, request)
    facts = build_daily_insight_facts(
        result,
        metric=request["metric"],
        diagnostic_threshold_dbz=request["diagnostic_threshold_dbz"],
        views=request["views"],
    )
    deterministic_summary = render_daily_deterministic_summary(facts)
    ai_insight = None
    insight_error = None
    try:
        ai_insight = generate_daily_insight(facts)
    except Exception as exc:
        insight_error = str(exc)
    status = "partial" if request["unmatched_requests"] else "ready"
    return {
        "status": status,
        "message": (
            "已执行全部支持的全天诊断；未支持项已保留在规范化请求中。"
            if status == "partial"
            else "全天严格匹配、评分、缓存与诊断图已完成。"
        ),
        "request": request,
        "result": result,
        "media": media,
        "facts": facts,
        "deterministic_summary": deterministic_summary,
        "ai_insight": ai_insight,
        "insight_error": insight_error,
    }


with tab_daily:
    st.caption(
        "对同一北京时间自然日内的所有起报做公平检验：严格匹配有效时刻，统一观测、区域、"
        "最近邻空间配准、有效掩膜、阈值与评分公式；平台不预设哪个模型更好。"
    )
    daily_cases = {
        "2026-04-07": {
            "label": "2026-04-07｜CREF 二进制观测",
            "forecast_dir": os.getenv("FENGLEI_DAILY_DIR", ""),
            "observation_dir": os.getenv("RADAR_DAILY_DIR", ""),
        },
        "2025-04-11": {
            "label": "2025-04-11｜CREF NetCDF 观测",
            "forecast_dir": os.getenv("FENGLEI_CASE_DIR", ""),
            "observation_dir": os.getenv("RADAR_OBS_DIR", ""),
        },
    }
    with st.sidebar:
        st.header("Phase 5 全天资料")
        daily_case_key = st.selectbox(
            "全天个例",
            options=list(daily_cases),
            format_func=lambda key: daily_cases[key]["label"],
            help="两个案例共用同一检验口径；可在下方临时修改目录。",
        )
        daily_case = daily_cases[daily_case_key]
        daily_forecast_dir = st.text_input(
            "全天风雷 REFMOSAIC 目录",
            value=daily_case["forecast_dir"],
            key=f"daily_forecast_dir_{daily_case_key}",
        )
        daily_observation_dir = st.text_input(
            "全天雷达 CREF 观测目录（.bin 或 .nc）",
            value=daily_case["observation_dir"],
            key=f"daily_observation_dir_{daily_case_key}",
        )
        daily_default_date = date.fromisoformat(daily_case_key)
        daily_date_raw = st.date_input(
            "个例日期（北京时间）",
            value=daily_default_date,
            key=f"daily_date_{daily_case_key}",
        )
        daily_date = daily_date_raw if isinstance(daily_date_raw, date) else daily_default_date
        daily_lat_min = st.number_input("全天 lat_min", value=28.0)
        daily_lat_max = st.number_input("全天 lat_max", value=35.0)
        daily_lon_min = st.number_input("全天 lon_min", value=116.5)
        daily_lon_max = st.number_input("全天 lon_max", value=123.5)
        daily_force = st.checkbox("忽略已有指标缓存并重新计算", value=False)

    daily_prompt = st.text_area(
        "用自然语言描述全天检验要求",
        value=(
            f"检验{daily_date:%Y年%m月%d日}全天风雷预报，使用5、15、25、35 dBZ阈值，查看6—180分钟"
            "CSI曲线；再用15 dBZ画起报时刻×时效热力图和跨起报分布图，并诊断异常时段。"
        ),
        help="当前支持全天曲线、热力图和分布图；不支持项会明确保留，不会自动替换。",
        key=f"daily_prompt_{daily_case_key}",
    )
    daily_col1, daily_col2 = st.columns(2)
    daily_parse_clicked = daily_col1.button(
        "解析并运行全天检验", type="primary", width="stretch"
    )
    daily_manual_clicked = daily_col2.button("使用全天默认方案运行", width="stretch")
    daily_scope = (
        daily_forecast_dir,
        daily_observation_dir,
        daily_date.isoformat(),
        float(daily_lat_min),
        float(daily_lat_max),
        float(daily_lon_min),
        float(daily_lon_max),
    )
    previous_daily = st.session_state.get("daily_attempt")
    if previous_daily is not None and previous_daily.get("scope") != daily_scope:
        st.session_state.pop("daily_attempt", None)

    if daily_parse_clicked or daily_manual_clicked:
        daily_run_attempt: dict[str, Any]
        progress = st.progress(0.0, text="准备全天资料索引……")

        def update_daily_progress(done, total, label):
            step = max(1, total // 100)
            if done == total or done == 1 or done % step == 0:
                progress.progress(done / total, text=f"{done}/{total}｜{label}")

        try:
            request = (
                default_daily_request(daily_date.isoformat())
                if daily_manual_clicked
                else parse_daily_request(daily_prompt)
            )
        except Exception as exc:
            daily_run_attempt = {
                "status": "parser_error",
                "message": str(exc),
            }
        else:
            try:
                daily_run_attempt = _execute_daily_request(
                    request,
                    forecast_dir=daily_forecast_dir,
                    observation_dir=daily_observation_dir,
                    fallback_date=daily_date.isoformat(),
                    bbox=(daily_lat_min, daily_lat_max, daily_lon_min, daily_lon_max),
                    force=daily_force,
                    progress_callback=update_daily_progress,
                )
            except Exception as exc:
                daily_run_attempt = {
                    "status": "execution_error",
                    "message": str(exc),
                    "request": request,
                }
        finally:
            progress.empty()
        daily_run_attempt["scope"] = daily_scope
        daily_run_attempt["submitted_prompt"] = None if daily_manual_clicked else daily_prompt
        st.session_state["daily_attempt"] = daily_run_attempt

    daily_attempt = st.session_state.get("daily_attempt")
    if daily_attempt is None:
        st.info(
            "目录准备好后运行。首次全天计算会逐文件流式处理；中断后再次使用同一参数会从指标缓存续跑。"
        )
    else:
        daily_status = daily_attempt["status"]
        if daily_status == "ready":
            st.success(daily_attempt["message"])
        elif daily_status == "partial":
            st.warning(daily_attempt["message"])
        else:
            st.error(daily_attempt["message"])
        if "request" in daily_attempt:
            with st.expander("查看全天规范化请求与未支持项", expanded=daily_status == "partial"):
                st.json(daily_attempt["request"])

        daily_result = daily_attempt.get("result")
        if daily_result is not None:
            main_column, ai_column = st.columns([3.2, 1], gap="large")
            with ai_column:
                st.markdown('<span class="ai-insight-anchor"></span>', unsafe_allow_html=True)
                st.subheader("AI 解读")
                with st.container(height=680, border=False):
                    st.caption("只读取全天汇总、分位数、异常样本和图表目录，不读取原始网格或图片。")
                    if daily_attempt.get("ai_insight"):
                        st.markdown(daily_attempt["ai_insight"])
                    else:
                        st.write(daily_attempt["deterministic_summary"])
                        if daily_attempt.get("insight_error"):
                            st.caption("AI 解读暂不可用；已显示确定性摘要。")
            with main_column:
                coverage = daily_result["coverage"]
                cache = daily_result["cache"]
                st.write(
                    f"**完成 {coverage['completed_pairs']}/{coverage['selected_pairs']} 对严格匹配样本**｜"
                    f"唯一观测 {coverage['unique_observations']} 帧｜"
                    f"订正时刻 {coverage['duplicate_observation_times']} 个｜"
                    f"本次复用缓存 {cache['metric_pairs_reused']} 对"
                )
                if coverage["matched_valid_start_bjt"]:
                    st.caption(
                        "可用严格匹配样本的有效时刻覆盖（北京时间）："
                        f"{coverage['matched_valid_start_bjt']} 至 "
                        f"{coverage['matched_valid_end_bjt']}。"
                    )
                st.caption(
                    "全天曲线按每个时效汇总 H/M/F/CN 后重新计算指标；"
                    "不是把各起报评分做简单平均。"
                )
                media_items = list(daily_attempt["media"].values())
                for start in range(0, len(media_items), 2):
                    columns = st.columns(2)
                    for offset, column in enumerate(columns):
                        if start + offset >= len(media_items):
                            continue
                        item = media_items[start + offset]
                        with column:
                            st.caption(f"图{item['number']}｜{item['label']}")
                            st.image(item["data"])
                            st.download_button(
                                "下载这张全天图（PNG）",
                                data=item["data"],
                                file_name=item["file_name"],
                                mime="image/png",
                                key=f"daily_{item['file_name']}",
                                on_click="ignore",
                            )
                st.subheader("下载全天结果")
                csv_col1, csv_col2, csv_col3 = st.columns(3)
                csv_col1.download_button(
                    "下载逐起报指标 CSV",
                    data=daily_result["per_case"].to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"phase5_{daily_result['context']['case_date_bjt']}_per_case.csv",
                    mime="text/csv",
                    on_click="ignore",
                    width="stretch",
                )
                csv_col2.download_button(
                    "下载全天汇总指标 CSV",
                    data=daily_result["summary"].to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"phase5_{daily_result['context']['case_date_bjt']}_summary.csv",
                    mime="text/csv",
                    on_click="ignore",
                    width="stretch",
                )
                daily_report = (
                    f"# Phase 5 全天检验解读\n\n"
                    f"{daily_attempt.get('ai_insight') or daily_attempt['deterministic_summary']}\n\n"
                    "## 确定性摘要\n\n"
                    f"{daily_attempt['deterministic_summary']}\n"
                )
                csv_col3.download_button(
                    "下载全天解读 Markdown",
                    data=daily_report.encode("utf-8"),
                    file_name=f"phase5_{daily_result['context']['case_date_bjt']}_insight.md",
                    mime="text/markdown",
                    on_click="ignore",
                    width="stretch",
                )
                with st.expander("查看重复观测订正版审计"):
                    duplicate_rows = [
                        {
                            "valid_time_bjt": item.valid_time_bjt,
                            "selected": item.selected.name,
                            "discarded": ", ".join(path.name for path in item.discarded),
                            "policy": item.policy,
                        }
                        for item in daily_result["catalog"].duplicates
                    ]
                    st.dataframe(pd.DataFrame(duplicate_rows), width="stretch")
                st.subheader("全天汇总指标与样本数")
                st.dataframe(daily_result["summary"], width="stretch")

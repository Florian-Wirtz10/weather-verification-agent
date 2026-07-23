"""Streamlit demo for fair Fenglei/CREF verification on the delivered cases."""

from datetime import date, timedelta
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
    build_daily_insight_facts,
    build_insight_facts,
    default_daily_request,
    default_request,
    discover_fenglei_cases,
    generate_daily_insight,
    generate_insight,
    parse_daily_request,
    parse_request,
    render_daily_deterministic_summary,
    render_deterministic_summary,
    resolve_media_layout,
    resolve_request,
    run_fenglei_case_verification,
)
from reports import build_case_download_artifacts, describe_view, figure_to_png
from verification.batch import (
    filter_daily_result_by_initializations,
    run_daily_fenglei_verification,
)
from visualization.batch_plots import (
    plot_forecast_observed_rate_distribution,
    plot_daily_score_curves,
    plot_daily_score_distribution,
    plot_daily_score_heatmap,
)

load_dotenv()
st.set_page_config(page_title="强对流预报检验智能体", layout="wide")

DEFAULT_HUBEI_BBOX = {
    "lat_min": 29.0,
    "lat_max": 33.5,
    "lon_min": 108.0,
    "lon_max": 116.5,
}


@st.cache_data(show_spinner="正在建立可用起报索引…")
def _cached_cases(forecast_dir: str, observation_dir: str, generation: int):
    del generation
    return discover_fenglei_cases(forecast_dir, observation_dir)


CASES = {
    "2026-04-07": {
        "label": "2026-04-07｜8:00— 次日02:54 BJT",
        "forecast_dir": os.getenv("FENGLEI_DAILY_DIR", ""),
        "observation_dir": os.getenv("RADAR_DAILY_DIR", ""),
        "display_clock": "BJT",
        "forecast_case_time_basis": "bjt",
        "observation_case_time_basis": "utc",
    },
    "2025-04-11": {
        "label": "2025-04-11｜00:06—23:54 UTC",
        "forecast_dir": os.getenv("FENGLEI_CASE_DIR", ""),
        "observation_dir": os.getenv("RADAR_OBS_DIR", ""),
        "display_clock": "UTC",
        "forecast_case_time_basis": "utc",
        "observation_case_time_basis": "utc",
    },
}


def _clear_verification_attempt():
    st.session_state.pop("verification_attempt", None)


def _metric_rows(result, metrics):
    rows = []
    for entry in result["per_lead"]:
        for threshold, scores in entry["categorical"].items():
            rows.append(
                {
                    "lead_minutes": entry["lead_minutes"],
                    "valid_time_utc": entry["valid_time"].isoformat(timespec="seconds") + "Z",
                    "threshold_dbz": threshold,
                    **{name: scores[name] for name in metrics if name in scores},
                    **entry["contingency"][threshold],
                }
            )
    return rows


def _execute_single(request, *, cases, selected_case_id, config, bbox):
    decision = resolve_request(request, cases, selected_case_id=selected_case_id)
    attempt: dict[str, Any] = {"kind": "single", "decision": decision, "request": request}
    if decision["status"] not in {"ready", "partial"}:
        return attempt
    execution = decision["execution"]
    result = run_fenglei_case_verification(
        fenglei_dir=config["forecast_dir"],
        radar_dir=config["observation_dir"],
        bbox=bbox,
        thresholds=tuple(execution["thresholds"]),
        init_time=execution["init_time"],
        lead_range=(execution["lead_range"]["start"], execution["lead_range"]["end"]),
        views=execution["views"],
        presentation=execution["presentation"],
    )
    metrics = [view["metric"] for view in execution["views"] if view["type"] == "score_time_series"]
    facts = build_insight_facts(result, requested_metrics=metrics)
    deterministic = render_deterministic_summary(facts)
    try:
        insight, insight_error = (
            generate_insight(facts, length=execution["insight"]["length"]),
            None,
        )
    except Exception as exc:  # External AI is optional.
        insight, insight_error = None, str(exc)
    rows = _metric_rows(result, metrics)
    try:
        downloads = build_case_download_artifacts(
            result=result,
            request=request,
            metric_rows=rows,
            facts=facts,
            deterministic_summary=deterministic,
            ai_insight=insight,
            execution_status=decision["status"],
        )
    finally:
        for figure in result["figures"].values():
            plt.close(figure)
    result["figures"] = {}
    attempt.update(
        result=result,
        metric_rows=rows,
        downloads=downloads,
        insight=insight,
        deterministic=deterministic,
        insight_error=insight_error,
    )
    return attempt


def _daily_media(result, request):
    metric, threshold = request["metric"], request["diagnostic_threshold_dbz"]
    selected_scope = (
        result["context"].get("selection_scope") == "selected_initializations"
    )
    scope_label = "Selected-initialization" if selected_scope else "Full-day"
    label_prefix = "所选起报" if selected_scope else "全天"
    factories = {
        "daily_score_curve": lambda n: plot_daily_score_curves(
            result["summary"], metric, figure_number=n, scope_label=scope_label
        ),
        "daily_score_heatmap": lambda n: plot_daily_score_heatmap(
            result["per_case"],
            metric,
            threshold,
            figure_number=n,
            scope_label=scope_label,
        ),
        "score_distribution": lambda n: plot_daily_score_distribution(
            result["per_case"], metric, threshold, figure_number=n
        ),
        "forecast_observed_rate_distribution": lambda n: (
            plot_forecast_observed_rate_distribution(
                result["per_case"], threshold, figure_number=n
            )
        ),
    }
    labels = {
        "daily_score_curve": f"{label_prefix}汇总 {metric.upper()} 随时效变化",
        "daily_score_heatmap": f"{threshold:g} dBZ 起报时刻×时效",
        "score_distribution": f"{threshold:g} dBZ 跨起报分布",
        "forecast_observed_rate_distribution": (
            f"{threshold:g} dBZ {label_prefix}预报率—观测率联合分布"
        ),
    }
    media = []
    for number, view in enumerate(request["views"], 1):
        figure = factories[view](number)
        try:
            media.append(
                {
                    "number": number,
                    "label": labels[view],
                    "data": figure_to_png(figure),
                    "file_name": f"{result['context']['case_date']}_{metric}_{view}.png",
                }
            )
        finally:
            plt.close(figure)
    return media


def _execute_daily(
    request,
    *,
    config,
    case_date,
    bbox,
    progress_callback,
    selected_initializations_utc=None,
):
    if not request["views"]:
        return {
            "kind": "daily",
            "status": "unsupported",
            "message": "没有可执行的全天图表。",
            "request": request,
        }
    lead = request["lead_range_minutes"]
    result = run_daily_fenglei_verification(
        forecast_dir=config["forecast_dir"],
        observation_dir=config["observation_dir"],
        case_date=request["requested_date"] or case_date,
        forecast_case_time_basis=config["forecast_case_time_basis"],
        observation_case_time_basis=config["observation_case_time_basis"],
        bbox=bbox,
        thresholds=request["thresholds_dbz"],
        lead_range=(lead["start"], lead["end"]),
        progress_callback=progress_callback,
    )
    if selected_initializations_utc is not None:
        result = filter_daily_result_by_initializations(
            result, selected_initializations_utc
        )
    facts = build_daily_insight_facts(
        result,
        metric=request["metric"],
        diagnostic_threshold_dbz=request["diagnostic_threshold_dbz"],
        views=request["views"],
    )
    deterministic = render_daily_deterministic_summary(facts)
    try:
        insight, insight_error = generate_daily_insight(facts), None
    except Exception as exc:  # External AI is optional.
        insight, insight_error = None, str(exc)
    return {
        "kind": "daily",
        "status": "partial" if request["unmatched_requests"] else "ready",
        "message": "检验完成。",
        "request": request,
        "result": result,
        "media": _daily_media(result, request),
        "insight": insight,
        "deterministic": deterministic,
        "insight_error": insight_error,
    }


def _render_single(attempt):
    decision, result = attempt["decision"], attempt.get("result")
    if decision["status"] not in {"ready", "partial"}:
        st.error(decision["message"])
        return
    if result is None:
        st.warning(decision["message"])
        return
    main, insight = st.columns([3.2, 1], gap="large")
    with insight:
        st.subheader("AI 解读")
        with st.container(height=680, border=False):
            st.markdown(attempt["insight"] or attempt["deterministic"])
            if attempt["insight_error"]:
                st.caption("AI 解读暂不可用，已显示确定性解读。")
    with main:
        downloads = attempt["downloads"]
        columns = st.columns(3)
        columns[0].download_button(
            "下载结果包 ZIP",
            downloads["bundle"]["data"],
            downloads["bundle"]["file_name"],
            "application/zip",
            width="stretch",
        )
        columns[1].download_button(
            "下载指标 CSV",
            downloads["metrics"]["data"],
            downloads["metrics"]["file_name"],
            "text/csv",
            width="stretch",
        )
        columns[2].download_button(
            "下载解读 Markdown",
            downloads["summary"]["data"],
            downloads["summary"]["file_name"],
            "text/markdown",
            width="stretch",
        )
        layout = resolve_media_layout(
            attempt["request"].get("presentation", {}).get("layout", "auto"),
            figure_metadata=result["figure_metadata"],
            animation_metadata=result["animation_metadata"],
        )
        items = [
            ("image", key, item, result["figure_metadata"][key])
            for key, item in downloads["images"].items()
        ] + [
            ("animation", key, item, result["animation_metadata"][key])
            for key, item in downloads.get("animations", {}).items()
        ]
        for start in range(0, len(items), 2 if layout != "single_column" else 1):
            columns = st.columns(2) if layout != "single_column" else [st.container()]
            for column, (kind, key, media, view) in zip(
                columns, items[start : start + len(columns)]
            ):
                with column:
                    st.caption(describe_view(view))
                    st.image(media["data"])
                    suffix, mime = (
                        ("gif", "image/gif") if kind == "animation" else ("png", "image/png")
                    )
                    st.download_button(
                        f"下载 {suffix.upper()}",
                        media["data"],
                        media["file_name"],
                        mime,
                        key=f"{suffix}_{key}",
                    )
        with st.expander("查看本次指标"):
            st.dataframe(pd.DataFrame(attempt["metric_rows"]), width="stretch")


def _render_daily(attempt):
    if attempt["status"] not in {"ready", "partial"}:
        st.error(attempt["message"])
        return
    result = attempt["result"]
    main, insight = st.columns([3.2, 1], gap="large")
    with insight:
        st.subheader("AI 解读")
        with st.container(height=680, border=False):
            st.markdown(attempt["insight"] or attempt["deterministic"])
            if attempt["insight_error"]:
                st.caption("AI 解读暂不可用，已显示确定性解读。")
    with main:
        coverage, cache = result["coverage"], result["cache"]
        selected_count = coverage.get("selected_initialization_count")
        selection_text = f"｜已选择 {selected_count} 个起报" if selected_count else ""
        st.write(
            f"**{coverage['completed_pairs']}/{coverage['selected_pairs']} 对严格匹配样本**"
            f"{selection_text}｜本次复用缓存 {cache['metric_pairs_reused']} 对"
        )
        for start in range(0, len(attempt["media"]), 2):
            columns = st.columns(2)
            for column, item in zip(columns, attempt["media"][start : start + 2]):
                with column:
                    st.caption(item["label"])
                    st.image(item["data"])
                    st.download_button(
                        "下载 PNG",
                        item["data"],
                        item["file_name"],
                        "image/png",
                        key=f"daily_{item['file_name']}",
                    )
        columns = st.columns(3)
        columns[0].download_button(
            "下载逐起报 CSV",
            result["per_case"].to_csv(index=False).encode("utf-8-sig"),
            f"{result['context']['case_date']}_per_case.csv",
            "text/csv",
            width="stretch",
        )
        columns[1].download_button(
            "下载汇总 CSV",
            result["summary"].to_csv(index=False).encode("utf-8-sig"),
            f"{result['context']['case_date']}_summary.csv",
            "text/csv",
            width="stretch",
        )
        columns[2].download_button(
            "下载解读 Markdown",
            attempt["deterministic"].encode("utf-8"),
            f"{result['context']['case_date']}_insight.md",
            "text/markdown",
            width="stretch",
        )
        with st.expander("查看汇总指标"):
            st.dataframe(result["summary"], width="stretch")


st.title("强对流预报检验智能体")
with st.sidebar:
    st.header("检验区域")
    st.caption("默认：湖北区域")
    lat_min = st.number_input("lat_min", value=DEFAULT_HUBEI_BBOX["lat_min"])
    lat_max = st.number_input("lat_max", value=DEFAULT_HUBEI_BBOX["lat_max"])
    lon_min = st.number_input("lon_min", value=DEFAULT_HUBEI_BBOX["lon_min"])
    lon_max = st.number_input("lon_max", value=DEFAULT_HUBEI_BBOX["lon_max"])

case_column, mode_column = st.columns([1.45, 1])
with case_column:
    case_key = st.selectbox(
        "检验个例", list(CASES), format_func=lambda value: CASES[value]["label"]
    )
with mode_column:
    mode = st.radio(
        "检验范围",
        ["全部起报（全天）", "自定义多起报", "单一起报"],
        horizontal=True,
    )

config = CASES[case_key]
bbox = (lat_min, lat_max, lon_min, lon_max)
scope = (case_key, mode, *bbox)
if st.session_state.get("active_scope") != scope:
    st.session_state["active_scope"] = scope
    st.session_state.pop("verification_attempt", None)

available_cases = None
labels = None
select = None
if mode in {"单一起报", "自定义多起报"}:
    generation = st.session_state.get("case_index_generation", 0)
    refresh, select = st.columns([1, 3])
    if refresh.button("刷新可用起报"):
        generation += 1
        st.session_state["case_index_generation"] = generation
    try:
        available_cases = _cached_cases(
            config["forecast_dir"], config["observation_dir"], generation
        )
    except Exception as exc:
        available_cases, catalog_error = [], str(exc)
    else:
        catalog_error = None
    if catalog_error or not available_cases:
        st.error(catalog_error or "当前个例没有可用起报。")
        st.stop()

    def _case_label(item):
        init = item["init_time"]
        if config["display_clock"] == "BJT":
            init += timedelta(hours=8)
        return f"{init:%Y-%m-%d %H:%M} {config['display_clock']}｜{len(item['available_leads'])} 个时效"

    labels = {item["case_id"]: _case_label(item) for item in available_cases}

if mode == "单一起报":
    with select:
        selected_case_id = st.selectbox(
            "风雷起报", list(labels), format_func=lambda value: labels[value]
        )
    prompt = st.text_area(
        "检验要求",
        "使用35 dBZ阈值，只看前1小时，生成 CSI ， ETS 曲线和预报-观测时序变化图，并简要解读结果。",
    )
    parse, default = st.columns(2)
    run_parse = parse.button("智能解析并运行", type="primary", width="stretch")
    run_default = default.button("使用推荐方案", width="stretch")
    if run_parse or run_default:
        try:
            request = (
                default_request()
                if run_default
                else parse_request(prompt, available_cases=available_cases)
            )
            with st.spinner("正在计算…"):
                st.session_state["verification_attempt"] = _execute_single(
                    request,
                    cases=available_cases,
                    selected_case_id=selected_case_id,
                    config=config,
                    bbox=bbox,
                )
        except (
            AmbiguousCaseError,
            DataContractError,
            NoDataError,
            RequestValidationError,
            ValueError,
        ) as exc:
            st.session_state["verification_attempt"] = {
                "kind": "single",
                "decision": {"status": "execution_error", "message": str(exc)},
            }
    attempt = st.session_state.get("verification_attempt")
    if attempt and attempt.get("kind") == "single":
        _render_single(attempt)
elif mode == "自定义多起报":
    if len(available_cases) < 2:
        st.error("当前个例至少需要两个可用起报，才能进行多起报比较。")
        st.stop()
    if len(available_cases) <= 3:
        default_case_ids = [item["case_id"] for item in available_cases]
    else:
        default_case_ids = [
            available_cases[index]["case_id"]
            for index in (0, len(available_cases) // 2, len(available_cases) - 1)
        ]
    with select:
        selected_case_ids = st.multiselect(
            "选择风雷起报（2—8个）",
            list(labels),
            default=default_case_ids,
            max_selections=8,
            format_func=lambda value: labels[value],
            on_change=_clear_verification_attempt,
        )
    prompt = st.text_area(
        "检验要求",
        "比较页面选中的多个起报，使用5、15、25、35 dBZ阈值，查看6—180分钟CSI曲线；"
        "再用15 dBZ画起报时刻×时效热力图、跨起报分布图和预报率—观测率联合分布，并全面解读。",
    )
    parse, default = st.columns(2)
    run_parse = parse.button("智能解析并运行", type="primary", width="stretch")
    run_default = default.button("使用推荐方案", width="stretch")
    if run_parse or run_default:
        if not 2 <= len(selected_case_ids) <= 8:
            st.session_state["verification_attempt"] = {
                "kind": "daily",
                "status": "execution_error",
                "message": "请选择2—8个不同的起报时间。",
            }
        else:
            progress = st.progress(0.0)

            def _progress(done, total, _label):
                progress.progress(done / total)

            selected_by_id = {item["case_id"]: item for item in available_cases}
            selected_times = [
                selected_by_id[case_id]["init_time"] for case_id in selected_case_ids
            ]
            try:
                request = (
                    default_daily_request(case_key)
                    if run_default
                    else parse_daily_request(prompt)
                )
                st.session_state["verification_attempt"] = _execute_daily(
                    request,
                    config=config,
                    case_date=case_key,
                    bbox=bbox,
                    progress_callback=_progress,
                    selected_initializations_utc=selected_times,
                )
            except Exception as exc:
                st.session_state["verification_attempt"] = {
                    "kind": "daily",
                    "status": "execution_error",
                    "message": str(exc),
                }
            finally:
                progress.empty()
    attempt = st.session_state.get("verification_attempt")
    if attempt and attempt.get("kind") == "daily":
        _render_daily(attempt)
else:
    prompt = st.text_area(
        "检验要求",
        f"检验{date.fromisoformat(case_key):%Y年%m月%d日}全部起报，使用5、15、25、35 dBZ阈值，"
        "查看 CSI 曲线、15 dBZ 热力图、跨起报分布图和预报率—观测率联合分布，并全面解读。",
    )
    parse, default = st.columns(2)
    run_parse = parse.button("智能解析并运行", type="primary", width="stretch")
    run_default = default.button("使用推荐方案", width="stretch")
    if run_parse or run_default:
        progress = st.progress(0.0)

        def _progress(done, total, _label):
            progress.progress(done / total)

        try:
            request = (
                default_daily_request(case_key) if run_default else parse_daily_request(prompt)
            )
            st.session_state["verification_attempt"] = _execute_daily(
                request, config=config, case_date=case_key, bbox=bbox, progress_callback=_progress
            )
        except Exception as exc:
            st.session_state["verification_attempt"] = {
                "kind": "daily",
                "status": "execution_error",
                "message": str(exc),
            }
        finally:
            progress.empty()
    attempt = st.session_state.get("verification_attempt")
    if attempt and attempt.get("kind") == "daily":
        _render_daily(attempt)

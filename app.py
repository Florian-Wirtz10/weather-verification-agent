"""Streamlit interface for StormDiT/Fenglei/CREF verification."""

from __future__ import annotations

from datetime import timedelta
from io import BytesIO
import os
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import (
    build_comparison_insight_facts,
    default_multimodel_request,
    generate_comparison_insight,
    parse_multimodel_request,
    render_comparison_deterministic_summary,
)
from data_io.forecast_sources import (
    MODEL_FENGLEI,
    MODEL_LABELS,
    MODEL_STORMDIT,
    build_multimodel_catalog,
    catalog_source_signature,
    discover_multimodel_initializations,
)
from data_io.benchmark import BENCHMARK_MODEL_LABELS
from reports import figure_to_png, figure_to_svg
from verification.multimodel import (
    common_valid_times,
    run_multimodel_batch,
    run_multimodel_case,
    run_multimodel_same_valid_time,
)
from visualization.batch_plots import plot_forecast_observed_rate_distribution
from visualization.comparison_plots import (
    plot_multimodel_batch_multi_metric_curves,
    plot_multimodel_batch_score_curves,
    plot_multimodel_forecast_observed_rate_scatter,
    plot_multimodel_multi_metric_curves,
    plot_multimodel_reflectivity_frame,
    plot_multimodel_score_curves,
    plot_multimodel_score_distribution,
    plot_multimodel_score_heatmaps,
    plot_multimodel_same_valid_time_comparison,
    plot_multimodel_variable_correlation_heatmap,
    render_multimodel_reflectivity_animation,
)

load_dotenv()
st.set_page_config(page_title="强对流预报检验智能体", layout="wide")

DEFAULT_HUBEI_BBOX = (29.0, 33.5, 108.0, 116.5)
CASES = {
    "2025-04-11": {
        "label": "2025-04-11 强对流个例",
        "forecast_dir": os.getenv("FENGLEI_CASE_DIR", "external/20250411风雷预测资料"),
        "observation_dir": os.getenv("RADAR_OBS_DIR", "external/20250411观测资料"),
    },
    "2026-04-08": {
        "label": "2026-04-08 强对流个例",
        "forecast_dir": os.getenv(
            "FENGLEI_20260408_DIR",
            "external/20260408风雷预测资料",
        ),
        "observation_dir": os.getenv(
            "RADAR_20260408_DIR",
            "external/20260408观测资料",
        ),
    },
}
BENCHMARK_LABEL_TO_MODEL = {label: model for model, label in BENCHMARK_MODEL_LABELS.items()}
CASE_MODEL_OPTIONS = {
    "2025-04-11": [
        "StormDiT",
        "风雷",
        "ConvLSTM",
        "PredRNN",
        "Earthformer",
        "NowCastNet",
        "SimVP",
        "DiffCast",
        "Persistence",
    ],
    "2026-04-08": ["StormDiT", "风雷"],
}
BENCHMARK_ONLY_MODELS = set(CASE_MODEL_OPTIONS["2025-04-11"][2:])
AI_PANEL_CSS = """
<style>
.st-key-ai_insight_panel {
    position: sticky;
    top: 1rem;
    max-height: calc(100vh - 2rem);
    overflow-y: auto;
    overscroll-behavior: contain;
    scrollbar-gutter: stable;
    padding-right: 0.5rem;
}
</style>
"""


def _stormdit_directories() -> tuple[str, ...]:
    raw = os.getenv(
        "UPSTREAM_MODEL_DIRS",
        "external/湖北两个个例;external/湖北个例补充",
    )
    return tuple(item.strip() for item in raw.split(";") if item.strip())


@st.cache_data(show_spinner="正在建立 StormDiT、风雷与CREF时空索引…")
def _catalog(stormdit_dirs, fenglei_dir, observation_dir, source_signature):
    # ``source_signature`` intentionally participates in Streamlit's cache key.
    # The catalog builder itself only needs the three source paths.
    _ = source_signature
    return build_multimodel_catalog(
        stormdit_directories=stormdit_dirs,
        fenglei_directory=fenglei_dir,
        observation_directory=observation_dir,
    )


def _clear_result():
    st.session_state.pop("phase8_result", None)


def _clear_result_and_prompt():
    _clear_result()
    st.session_state.pop("verification_prompt", None)
    st.session_state.pop("verification_prompt_context", None)
    st.session_state.pop("benchmark_attempt", None)
    st.session_state.pop("benchmark_prompt", None)
    st.session_state.pop("benchmark_prompt_context", None)


def _case_changed():
    _clear_result_and_prompt()
    st.session_state.pop("selected_models", None)


def _case_label(case):
    return (
        f"{case['init_time_bjt']:%Y-%m-%d %H:%M} BJT"
        f"（{case['init_time']:%m-%d %H:%M} UTC）｜"
        f"{len(case['available_leads'])} 个匹配时效"
        + ("｜完整" if case["complete"] else f"｜缺 {len(case['missing_leads'])} 帧")
    )


def _single_rows(result) -> pd.DataFrame:
    rows = []
    for entry in result["per_lead"]:
        for model_id, threshold_map in entry["models"].items():
            for threshold, values in threshold_map.items():
                rows.append(
                    {
                        "model_id": model_id,
                        "model": MODEL_LABELS[model_id],
                        "init_time_utc": result["context"]["init_time_utc"].isoformat(),
                        "init_time_bjt": result["context"]["init_time_bjt"].isoformat(),
                        "valid_time_utc": entry["valid_time_utc"].isoformat(),
                        "lead_minutes": entry["lead_minutes"],
                        "threshold_dbz": threshold,
                        **values["contingency"],
                        **{
                            metric: values[metric] for metric in ("pod", "far", "csi", "bias_score")
                        },
                        **{
                            metric: values[metric]
                            for metric in (
                                "forecast_mean_dbz",
                                "observation_mean_dbz",
                                "forecast_max_dbz",
                                "observation_max_dbz",
                                "mean_reflectivity_error_dbz",
                                "reflectivity_mae_dbz",
                                "centroid_distance_km",
                            )
                            if metric in values
                        },
                    }
                )
    return pd.DataFrame(rows)


def _add_figure(media, figure, label, stem, *, full_width=False):
    try:
        media.append(
            {
                "number": len(media) + 1,
                "label": label,
                "data": figure_to_png(figure, dpi=300),
                "file_name": f"{stem}.png",
                "mime": "image/png",
                "kind": "PNG",
                "full_width": bool(full_width),
                "svg_data": figure_to_svg(figure),
                "svg_file_name": f"{stem}.svg",
            }
        )
    finally:
        plt.close(figure)


def _frame_selection(leads, request):
    step = request["animation"]["frame_step"]
    selected = list(leads)[::step]
    if len(selected) > 30:
        positions = {round(index * (len(selected) - 1) / 29) for index in range(30)}
        selected = [selected[index] for index in sorted(positions)]
    return selected


def _single_media(result, request):
    media = []
    init = result["context"]["init_time_utc"]
    views = request["views"]
    if "spatial_comparison" in views:
        target = result["coverage"]["spatial_lead"]
        _add_figure(
            media,
            plot_multimodel_reflectivity_frame(result["frames"][target]),
            f"+{target} 分钟预报—CREF空间对比",
            f"{init:%Y%m%d%H%M}_spatial_{target:03d}",
        )
    if "reflectivity_animation" in views:
        ordered = [result["frames"][lead] for lead in result["coverage"]["animation_leads"]]
        media.append(
            {
                "number": len(media) + 1,
                "label": "预报—CREF同步时序动画",
                "data": render_multimodel_reflectivity_animation(
                    ordered, fps=request["animation"]["fps"]
                ),
                "file_name": f"{init:%Y%m%d%H%M}_reflectivity.gif",
                "mime": "image/gif",
                "kind": "GIF",
            }
        )
    if "score_curve" in views:
        for metric in request["metrics"]:
            _add_figure(
                media,
                plot_multimodel_score_curves(result, metric),
                f"{metric.upper()} 随预报时效变化",
                f"{init:%Y%m%d%H%M}_{metric}",
            )
    if "multi_metric_curve" in views:
        _add_figure(
            media,
            plot_multimodel_multi_metric_curves(
                result,
                request["metrics"],
                request["diagnostic_threshold_dbz"],
            ),
            f"{request['diagnostic_threshold_dbz']:g} dBZ CSI、POD、FAR同图比较",
            f"{init:%Y%m%d%H%M}_multi_metric",
        )
    return media


def _batch_media(result, request):
    media = []
    views = request["views"]
    if "same_valid_time_comparison" in views:
        comparison = result["same_valid_time_comparison"]
        for model_id in comparison["model_ids"]:
            _add_figure(
                media,
                plot_multimodel_same_valid_time_comparison(
                    comparison,
                    model_id,
                ),
                f"{MODEL_LABELS[model_id]}不同起报在"
                f"{comparison['valid_time_bjt']:%m-%d %H:%M} BJT的预报对比",
                f"phase8_same_valid_{model_id}_{comparison['valid_time_utc']:%Y%m%d%H%M}",
                full_width=True,
            )
    if "score_curve" in views:
        for metric in request["metrics"]:
            _add_figure(
                media,
                plot_multimodel_batch_score_curves(result["summary"], metric),
                f"汇总 {metric.upper()} 随预报时效变化",
                f"phase8_batch_{metric}",
            )
    if "multi_metric_curve" in views:
        _add_figure(
            media,
            plot_multimodel_batch_multi_metric_curves(
                result["summary"],
                request["metrics"],
                request["diagnostic_threshold_dbz"],
            ),
            f"{request['diagnostic_threshold_dbz']:g} dBZ 汇总CSI、POD、FAR同图比较",
            "phase8_batch_multi_metric",
        )
    if "score_heatmap" in views:
        metric = request["diagnostic_metric"]
        _add_figure(
            media,
            plot_multimodel_score_heatmaps(
                result["per_case"],
                metric,
                request["diagnostic_threshold_dbz"],
            ),
            f"{request['diagnostic_threshold_dbz']:g} dBZ {metric.upper()} 起报—时效评分矩阵",
            f"phase8_batch_{metric}_heatmap",
        )
    if "variable_correlation_heatmap" in views:
        _add_figure(
            media,
            plot_multimodel_variable_correlation_heatmap(
                result["per_case"],
                request["diagnostic_threshold_dbz"],
                method=request["correlation_method"],
            ),
            f"{request['diagnostic_threshold_dbz']:g} dBZ "
            f"{request['correlation_method'].title()}变量相关性热力图",
            "phase8_batch_variable_correlation",
        )
    if "score_distribution" in views:
        metric = request["diagnostic_metric"]
        _add_figure(
            media,
            plot_multimodel_score_distribution(
                result["per_case"],
                metric,
                request["thresholds_dbz"],
            ),
            f"{metric.upper()} 多阈值分布",
            f"phase8_batch_{metric}_distribution",
        )
    if "forecast_observed_rate_scatter" in views:
        _add_figure(
            media,
            plot_multimodel_forecast_observed_rate_scatter(
                result["per_case"],
                request["thresholds_dbz"],
            ),
            "多阈值预报率—观测率散点比较",
            "phase8_batch_rate_scatter",
        )
    if "forecast_observed_rate_distribution" in views:
        for model_id in result["context"]["model_ids"]:
            subset = result["per_case"][result["per_case"]["model_id"] == model_id]
            figure = plot_forecast_observed_rate_distribution(
                subset, request["diagnostic_threshold_dbz"]
            )
            figure.suptitle(
                f"{'StormDiT' if model_id == MODEL_STORMDIT else 'Fenglei'} "
                "forecast rate vs observed rate",
                fontsize=14,
                fontweight="bold",
            )
            _add_figure(
                media,
                figure,
                f"{MODEL_LABELS[model_id]}预报率—观测率联合分布",
                f"phase8_batch_{model_id}_rates",
            )
    return media


def _finish_attempt(result, request, media, rows):
    catalog = [{"number": item["number"], "label": item["label"]} for item in media]
    facts = build_comparison_insight_facts(
        result,
        request["metrics"],
        figure_catalog=catalog,
        diagnostic_threshold_dbz=request["diagnostic_threshold_dbz"],
        correlation_method=request["correlation_method"],
    )
    fallback = render_comparison_deterministic_summary(facts)
    try:
        insight, insight_error = generate_comparison_insight(facts), None
    except Exception as exc:
        insight, insight_error = None, str(exc)
    status = (
        "partial"
        if request["unmatched_requests"]
        or ("coverage" in result and result["coverage"].get("complete") is False)
        else "ready"
    )
    return {
        "status": status,
        "request": request,
        "result": result,
        "media": media,
        "rows": rows,
        "insight": insight,
        "fallback": fallback,
        "insight_error": insight_error,
    }


def _execute_single(catalog, model_ids, case, bbox, request):
    lead = request["lead_range_minutes"]
    available = [
        value for value in case["available_leads"] if lead["start"] <= value <= lead["end"]
    ]
    if not available:
        raise ValueError("所选起报在请求的时效范围内没有严格匹配资料。")
    animation_leads = (
        _frame_selection(available, request) if "reflectivity_animation" in request["views"] else []
    )
    frame_leads = set(animation_leads)
    spatial_lead = None
    if "spatial_comparison" in request["views"]:
        spatial_lead = request["spatial_lead_minutes"] or available[0]
        if spatial_lead not in available:
            raise ValueError(f"请求的 +{spatial_lead} 分钟没有严格匹配的预报与观测资料。")
        frame_leads.add(spatial_lead)
    result = run_multimodel_case(
        catalog,
        model_ids,
        case["init_time"],
        bbox,
        thresholds=request["thresholds_dbz"],
        lead_range=(lead["start"], lead["end"]),
        frame_leads=sorted(frame_leads),
    )
    result["coverage"]["animation_leads"] = animation_leads
    result["coverage"]["spatial_lead"] = spatial_lead
    return _finish_attempt(result, request, _single_media(result, request), _single_rows(result))


def _execute_batch(
    catalog,
    model_ids,
    bbox,
    request,
    *,
    initializations,
    comparison_valid_time,
    complete_only,
    progress_callback,
):
    lead = request["lead_range_minutes"]
    same_valid_time_requested = "same_valid_time_comparison" in request["views"]
    comparison = None
    if same_valid_time_requested:
        if initializations is None or comparison_valid_time is None:
            raise ValueError("请选择至少两个具有共同有效时间的起报。")
        comparison = run_multimodel_same_valid_time(
            catalog,
            model_ids,
            initializations,
            comparison_valid_time,
            bbox,
            thresholds=request["thresholds_dbz"],
        )
    if set(request["views"]) == {"same_valid_time_comparison"}:
        result = {
            "context": {
                "model_ids": list(model_ids),
                "model_labels": [MODEL_LABELS[value] for value in model_ids],
                "model_contract": catalog.stormdit_contract,
                "bbox": list(map(float, bbox)),
                "thresholds_dbz": request["thresholds_dbz"],
                "lead_range_minutes": [lead["start"], lead["end"]],
                "sample_policy": (
                    "common intersection" if len(model_ids) > 1 else "model-specific matches"
                ),
            },
            "coverage": {
                "selected_initialization_count": len(comparison["items"]),
                "selected_pairs": len(comparison["items"]),
                "complete_initializations": len(comparison["items"]),
            },
            "per_case": comparison["per_case"],
            "summary": pd.DataFrame(),
            "cache": {
                "path": "",
                "run_key": "same-valid-time-direct",
                "computed_initializations": len(comparison["items"]),
                "complete": True,
            },
            "same_valid_time_comparison": comparison,
        }
    else:
        result = run_multimodel_batch(
            catalog,
            model_ids,
            bbox,
            thresholds=request["thresholds_dbz"],
            lead_range=(lead["start"], lead["end"]),
            initializations=initializations,
            complete_only=complete_only,
            progress_callback=progress_callback,
        )
        if comparison is not None:
            result["same_valid_time_comparison"] = comparison
    return _finish_attempt(
        result,
        request,
        _batch_media(result, request),
        result["per_case"].drop(columns=["ets"], errors="ignore"),
    )


def _bundle(attempt):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in attempt["media"]:
            archive.writestr(item["file_name"], item["data"])
            if item.get("svg_data"):
                archive.writestr(item["svg_file_name"], item["svg_data"])
        archive.writestr("metrics.csv", attempt["rows"].to_csv(index=False).encode("utf-8-sig"))
        archive.writestr(
            "ai_insight.md",
            (attempt["insight"] or attempt["fallback"]).encode("utf-8"),
        )
    return buffer.getvalue()


def _render(attempt):
    if attempt.get("status") == "execution_error":
        st.error(attempt["message"])
        return
    st.markdown(AI_PANEL_CSS, unsafe_allow_html=True)
    main, right = st.columns([3.25, 1], gap="large")
    with right:
        with st.container(key="ai_insight_panel"):
            st.subheader("AI 解读")
            st.markdown(attempt["insight"] or attempt["fallback"])
            if attempt["insight_error"]:
                st.caption("AI服务暂不可用，已显示由同一数字事实生成的确定性解读。")
    with main:
        result = attempt["result"]
        models = " vs ".join(result["context"]["model_labels"])
        if "per_lead" in result:
            context, coverage = result["context"], result["coverage"]
            st.info(
                f"{models}｜{context['init_time_bjt']:%Y-%m-%d %H:%M} BJT"
                f"（{context['init_time_utc']:%m-%d %H:%M} UTC）｜"
                f"{len(coverage['selected_leads'])} 个严格匹配时效"
            )
        else:
            coverage, cache = result["coverage"], result["cache"]
            comparison = result.get("same_valid_time_comparison")
            if comparison and set(attempt["request"]["views"]) == {"same_valid_time_comparison"}:
                lead_text = "、".join(f"+{item['lead_minutes']}" for item in comparison["items"])
                st.info(
                    f"{models}｜共同有效时间 "
                    f"{comparison['valid_time_bjt']:%Y-%m-%d %H:%M} BJT｜"
                    f"{coverage['selected_initialization_count']} 个起报｜"
                    f"对应时效 {lead_text} 分钟"
                )
            else:
                st.info(
                    f"{models}｜{coverage['selected_initialization_count']} 个起报｜"
                    f"{coverage['selected_pairs']} 组严格匹配时效｜"
                    f"本次新计算 {cache['computed_initializations']} 个起报"
                )
        if attempt["status"] == "partial":
            notes = "；".join(attempt["request"]["unmatched_requests"])
            st.warning(notes or "所选起报资料不完整，已执行可用时效。")
        controls = st.columns(3)
        controls[0].download_button(
            "下载结果包 ZIP",
            _bundle(attempt),
            "phase8_verification.zip",
            "application/zip",
            width="stretch",
        )
        controls[1].download_button(
            "下载指标 CSV",
            attempt["rows"].to_csv(index=False).encode("utf-8-sig"),
            "phase8_metrics.csv",
            "text/csv",
            width="stretch",
        )
        controls[2].download_button(
            "下载AI解读",
            (attempt["insight"] or attempt["fallback"]).encode("utf-8"),
            "phase8_ai_insight.md",
            "text/markdown",
            width="stretch",
        )

        def render_item(item):
            st.caption(f"图{item['number']}｜{item['label']}")
            # Comparison GIFs are encoded below Streamlit's internal
            # 1460 px media limit, so this native component receives
            # their original multi-frame bytes and retains fullscreen.
            st.image(item["data"], width="stretch")
            st.download_button(
                f"下载 {item['kind']}",
                item["data"],
                item["file_name"],
                item["mime"],
                key=f"media_{item['number']}_{item['file_name']}",
            )
            if item.get("svg_data"):
                st.download_button(
                    "下载 SVG",
                    item["svg_data"],
                    item["svg_file_name"],
                    "image/svg+xml",
                    key=f"media_svg_{item['number']}_{item['svg_file_name']}",
                )

        position = 0
        while position < len(attempt["media"]):
            item = attempt["media"][position]
            if item.get("full_width"):
                render_item(item)
                position += 1
                continue
            row = []
            while (
                position < len(attempt["media"])
                and not attempt["media"][position].get("full_width")
                and len(row) < 2
            ):
                row.append(attempt["media"][position])
                position += 1
            for column, row_item in zip(st.columns(2), row):
                with column:
                    render_item(row_item)
        with st.expander("查看结构化指标"):
            st.dataframe(attempt["rows"], width="stretch")


st.title("强对流预报检验智能体")
case_column, product_column, mode_column = st.columns([1.1, 1.8, 1.5])
with case_column:
    case_key = st.selectbox(
        "检验个例",
        list(CASES),
        format_func=lambda value: CASES[value]["label"],
        on_change=_case_changed,
    )
with product_column:
    selected_model_labels = st.multiselect(
        "预报模型",
        CASE_MODEL_OPTIONS[case_key],
        default=["StormDiT", "风雷"],
        key="selected_models",
        on_change=_clear_result_and_prompt,
    )
with mode_column:
    mode = st.radio(
        "检验范围",
        ["全部起报", "自定义多起报", "单一起报"],
        horizontal=True,
        on_change=_clear_result_and_prompt,
    )

if not selected_model_labels:
    st.error("请至少选择一个预报模型。")
    st.stop()

benchmark_mode = bool(set(selected_model_labels) & BENCHMARK_ONLY_MODELS)
with st.sidebar:
    st.header("检验区域")
    if benchmark_mode:
        st.caption("武汉固定区域：28.70—32.51°N，112.38—116.19°E")
        lat_min = st.number_input("lat_min", value=28.70, disabled=True)
        lat_max = st.number_input("lat_max", value=32.51, disabled=True)
        lon_min = st.number_input("lon_min", value=112.38, disabled=True)
        lon_max = st.number_input("lon_max", value=116.19, disabled=True)
    else:
        st.caption("默认：湖北区域")
        lat_min = st.number_input("lat_min", value=DEFAULT_HUBEI_BBOX[0])
        lat_max = st.number_input("lat_max", value=DEFAULT_HUBEI_BBOX[1])
        lon_min = st.number_input("lon_min", value=DEFAULT_HUBEI_BBOX[2])
        lon_max = st.number_input("lon_max", value=DEFAULT_HUBEI_BBOX[3])

if benchmark_mode:
    from benchmark_ui import render_benchmark_page

    render_benchmark_page(
        models=tuple(BENCHMARK_LABEL_TO_MODEL[label] for label in selected_model_labels),
        scope=mode,
    )
    st.stop()

complete_only = st.checkbox(
    "仅使用6—180分钟完整起报",
    value=True,
    help="关闭后可选择资料不完整的起报；缺帧不会被时间插值。",
    on_change=_clear_result,
)
bbox = (lat_min, lat_max, lon_min, lon_max)
config = CASES[case_key]
model_ids = tuple(
    MODEL_STORMDIT if label == "StormDiT" else MODEL_FENGLEI for label in selected_model_labels
)
try:
    stormdit_dirs = _stormdit_directories()
    catalog = _catalog(
        stormdit_dirs,
        config["forecast_dir"],
        config["observation_dir"],
        catalog_source_signature(
            stormdit_directories=stormdit_dirs,
            fenglei_directory=config["forecast_dir"],
            observation_directory=config["observation_dir"],
        ),
    )
    available_cases = discover_multimodel_initializations(
        catalog, model_ids, lead_range=(6, 180), common_only=len(model_ids) > 1
    )
    if complete_only:
        available_cases = [case for case in available_cases if case["complete"]]
except Exception as exc:
    st.error(f"资料索引失败：{exc}")
    st.stop()
if not available_cases:
    st.error("当前个例、产品和完整性条件下没有可检验起报。")
    st.stop()

case_by_id = {case["case_id"]: case for case in available_cases}
selected_case_ids = None
selected_case_id = None
comparison_valid_time = None
if mode == "单一起报":
    selected_case_id = st.selectbox(
        "起报时间",
        list(case_by_id),
        format_func=lambda value: _case_label(case_by_id[value]),
        on_change=_clear_result,
    )
elif mode == "自定义多起报":
    if len(available_cases) < 2:
        st.error("当前条件下不足两个起报，无法进行多起报比较。")
        st.stop()
    selection_count = min(3, len(available_cases))
    middle = len(available_cases) // 2
    start = max(
        0,
        min(middle - selection_count // 2, len(available_cases) - selection_count),
    )
    indexes = list(range(start, start + selection_count))
    selected_case_ids = st.multiselect(
        "选择起报（2—8个）",
        list(case_by_id),
        default=[available_cases[index]["case_id"] for index in indexes],
        max_selections=8,
        format_func=lambda value: _case_label(case_by_id[value]),
        on_change=_clear_result,
    )
    selected_cases = [case_by_id[value] for value in selected_case_ids or [] if value in case_by_id]
    shared_valid_times = common_valid_times(selected_cases) if len(selected_cases) >= 2 else []
    if shared_valid_times:

        def _valid_time_label(value):
            lead_text = "、".join(
                f"{case['init_time_bjt']:%m-%d %H:%M}起报→"
                f"+{int((value - case['init_time']).total_seconds() / 60)}分钟"
                for case in selected_cases
            )
            return (
                f"{value + timedelta(hours=8):%Y-%m-%d %H:%M} BJT"
                f"（{value:%m-%d %H:%M} UTC）｜{lead_text}"
            )

        comparison_valid_time = st.selectbox(
            "共同有效时间",
            shared_valid_times,
            index=len(shared_valid_times) // 2,
            format_func=_valid_time_label,
            help="各起报在该时刻均有严格匹配的原生预报，并共同对照同一张CREF观测。",
            on_change=_clear_result,
        )
    elif len(selected_cases) >= 2:
        st.warning("所选起报在6—180分钟范围内没有共同有效时间，请选择更接近的起报。")

batch = mode != "单一起报"
request_scope = {
    "单一起报": "single",
    "自定义多起报": "custom",
    "全部起报": "all",
}[mode]
product_subject = "StormDiT和风雷" if len(selected_model_labels) > 1 else selected_model_labels[0]
if request_scope == "single":
    default_prompt = (
        f"使用10、20、30、40 dBZ阈值，查看6—180分钟{product_subject}预报"
        "与CREF观测的同步动画和"
        "CSI、POD、FAR曲线，并在20 dBZ下将三个指标放在一张图比较和解读。"
    )
elif request_scope == "custom":
    spatial_instruction = (
        "StormDiT和风雷分别生成空间对比图"
        if len(model_ids) > 1
        else f"为{product_subject}生成空间对比图"
    )
    default_prompt = (
        "对比我选择的多个起报在页面所选同一有效时间的预报信息。"
        f"{spatial_instruction}：每个面板对应一个起报，明确标注"
        "起报时间和预报时效，并共同对照该有效时间的CREF观测；使用统一地理范围"
        "和反射率色标，结合20 dBZ评分说明随着起报更新、时效缩短，回波位置、"
        "强度和命中空报发生了什么变化。"
    )
else:
    analysis_verb = "比较" if len(model_ids) > 1 else "分析"
    default_prompt = (
        f"使用10、20、30、40 dBZ阈值，查看6—180分钟{product_subject}的"
        f"CSI、POD、FAR曲线，在20 dBZ下把三个指标放在一张图{analysis_verb}，并生成"
        "起报—时效评分矩阵、Spearman变量相关性热力图、多阈值评分分布和"
        "预报率—观测率散点图，全面解读。"
    )
prompt_context = (tuple(selected_model_labels), request_scope)
if st.session_state.get("verification_prompt_context") != prompt_context:
    st.session_state["verification_prompt"] = default_prompt
    st.session_state["verification_prompt_context"] = prompt_context
prompt = st.text_area(
    "检验要求",
    key="verification_prompt",
)
left, right = st.columns(2)
run_ai = left.button("智能解析并运行", type="primary", width="stretch")
run_default = right.button("使用推荐方案", width="stretch")
if run_ai or run_default:
    try:
        request = (
            default_multimodel_request(batch=batch, scope=request_scope)
            if run_default
            else parse_multimodel_request(
                prompt,
                batch=batch,
                scope=request_scope,
            )
        )
        if mode == "单一起报":
            if selected_case_id is None:
                raise ValueError("请选择一个起报时间。")
            with st.spinner("正在读取并严格对齐预报与CREF…"):
                attempt = _execute_single(
                    catalog,
                    model_ids,
                    case_by_id[selected_case_id],
                    bbox,
                    request,
                )
        else:
            if mode == "自定义多起报":
                if not selected_case_ids or not 2 <= len(selected_case_ids) <= 8:
                    raise ValueError("请选择2—8个起报时间。")
                initializations = [case_by_id[value]["init_time"] for value in selected_case_ids]
            else:
                initializations = None
            progress = st.progress(0.0)

            def _progress(done, total, _label):
                progress.progress(done / total)

            try:
                attempt = _execute_batch(
                    catalog,
                    model_ids,
                    bbox,
                    request,
                    initializations=initializations,
                    comparison_valid_time=comparison_valid_time,
                    complete_only=complete_only,
                    progress_callback=_progress,
                )
            finally:
                progress.empty()
        st.session_state["phase8_result"] = attempt
    except Exception as exc:
        st.session_state["phase8_result"] = {
            "status": "execution_error",
            "message": str(exc),
        }

attempt = st.session_state.get("phase8_result")
if attempt:
    _render(attempt)

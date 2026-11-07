"""Streamlit UI for the delivered China CREF zero-shot benchmark."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import streamlit as st

from agent.benchmark_insight import (
    build_benchmark_insight_facts,
    generate_benchmark_insight,
    render_benchmark_deterministic_summary,
)
from agent.benchmark_request import default_benchmark_request, parse_benchmark_request
from data_io.benchmark import BENCHMARK_MODEL_LABELS, build_benchmark_catalog
from reports import figure_to_png, figure_to_svg
from verification.benchmark import (
    benchmark_available_samples,
    benchmark_frame,
    common_horizon_minutes,
    run_benchmark,
    summarize_benchmark_period,
)
from visualization.benchmark_plots import (
    plot_benchmark_pod_distribution,
    plot_benchmark_far_bars,
    plot_benchmark_fss_curves,
    plot_benchmark_period_summary,
    plot_benchmark_score_curves,
    plot_benchmark_score_heatmap,
    plot_benchmark_spatial_comparison,
)
from webapi.settings import DATA_ROOT, environment_path


BENCHMARK_PACKAGE = environment_path(
    "BENCHMARK_PACKAGE_DIR", DATA_ROOT / "benchmarks/2025-04-11/wuhan-zero-shot"
)
BENCHMARK_FENGLEI_DIR = environment_path(
    "FENGLEI_CASE_DIR", DATA_ROOT / "cases/2025-04-11/forecasts/fenglei"
)
_INTERNAL_COLUMNS = {
    "absolute_error_sum",
    "squared_error_sum",
    "fss_1_numerator",
    "fss_1_denominator",
    "fss_3_numerator",
    "fss_3_denominator",
    "fss_5_numerator",
    "fss_5_denominator",
}


def _package_signature(path: Path, fenglei_directory: Path):
    index = path / "index.csv"
    files = sorted((path / "forecasts").glob("*.npz"))
    return (
        str(path.resolve()),
        index.stat().st_mtime_ns,
        index.stat().st_size,
        tuple((item.name, item.stat().st_size, item.stat().st_mtime_ns) for item in files),
        (
            str(fenglei_directory.resolve()),
            fenglei_directory.stat().st_mtime_ns,
            sum(1 for _ in fenglei_directory.glob("*REFMOSAIC*MIN.bz2")),
        ),
    )


@st.cache_resource(show_spinner="正在验证数据…")
def _benchmark_catalog(package, fenglei_directory, signature):
    _ = signature
    return build_benchmark_catalog(package, fenglei_directory=fenglei_directory)


@st.cache_data(show_spinner="正在计算检验指标…", max_entries=16)
def _benchmark_result(
    package,
    fenglei_directory,
    signature,
    models,
    sample_ids,
    thresholds,
    start,
    end,
    windows,
):
    catalog = build_benchmark_catalog(package, fenglei_directory=fenglei_directory)
    return run_benchmark(
        catalog,
        models,
        sample_ids=sample_ids,
        thresholds=thresholds,
        lead_range=(start, end),
        fss_windows=windows,
    )


def _sample_label(sample):
    return (
        f"{sample.analysis_time_bjt:%Y-%m-%d %H:%M} BJT"
        f"（{sample.analysis_time_utc:%m-%d %H:%M} UTC）｜ID {sample.sample_id}"
    )


def _add_figure(media, figure, label, stem, *, full_width=False):
    try:
        media.append(
            {
                "number": len(media) + 1,
                "label": label,
                "data": figure_to_png(figure, dpi=240),
                "file_name": f"{stem}.png",
                "mime": "image/png",
                "kind": "PNG",
                "full_width": full_width,
                "svg_data": figure_to_svg(figure),
                "svg_file_name": f"{stem}.svg",
            }
        )
    finally:
        plt.close(figure)


def _media(catalog, result, request, *, single_sample_id=None):
    media = []
    summary = result["summary"]
    thresholds = request["thresholds_dbz"]
    if "spatial_comparison" in request["views"]:
        lead = request["spatial_lead_minutes"] or request["lead_range_minutes"]["start"]
        frame = benchmark_frame(catalog, single_sample_id, result["context"]["models"], lead)
        _add_figure(
            media,
            plot_benchmark_spatial_comparison(frame),
            f"+{lead}分钟多模型—CREF空间对比",
            f"benchmark_spatial_{single_sample_id}_{lead:03d}",
            full_width=True,
        )
    if "score_curve" in request["views"]:
        for metric in request.get("score_curve_metrics", request["metrics"]):
            if metric == "fss":
                continue
            _add_figure(
                media,
                plot_benchmark_score_curves(summary, metric, thresholds),
                f"多模型{metric.upper()}随预报时效变化",
                f"benchmark_{metric}_curve",
                full_width=len(thresholds) >= 3,
            )
    if "fss_curve" in request["views"] and "fss" in request["metrics"]:
        _add_figure(
            media,
            plot_benchmark_fss_curves(
                summary, request.get("fss_thresholds_dbz", thresholds), request["fss_windows"]
            ),
            "多阈值多尺度FSS随预报时效变化",
            "benchmark_fss_curve",
            full_width=True,
        )
    period = summarize_benchmark_period(result["per_sample"], fss_windows=request["fss_windows"])
    if "period_summary" in request["views"]:
        threshold = request["diagnostic_threshold_dbz"]
        fss_window = max(request["fss_windows"])
        _add_figure(
            media,
            plot_benchmark_period_summary(
                period,
                request["metrics"],
                threshold,
                fss_window=fss_window,
            ),
            f"{threshold:g} dBZ严格公共时效汇总比较",
            "benchmark_period_summary",
            full_width=True,
        )
    if "score_heatmap" in request["views"]:
        metric = request["metrics"][0]
        threshold = request["diagnostic_threshold_dbz"]
        fss_window = max(request["fss_windows"])
        _add_figure(
            media,
            plot_benchmark_score_heatmap(
                summary,
                metric,
                threshold,
                fss_window=fss_window,
            ),
            f"{threshold:g} dBZ模型—时效{metric.upper()}热力图",
            "benchmark_score_heatmap",
            full_width=True,
        )
    if "pod_distribution" in request["views"]:
        for threshold in thresholds:
            _add_figure(
                media, plot_benchmark_pod_distribution(result["per_sample"], threshold),
                f"{threshold:g} dBZ POD箱线图", f"benchmark_pod_distribution_{threshold:g}",
                full_width=True,
            )
    if "far_bar" in request["views"]:
        _add_figure(
            media, plot_benchmark_far_bars(summary, thresholds),
            "FAR分组柱状图", "benchmark_far_bars", full_width=True,
        )
    return media, period


def _bundle(attempt):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in attempt["media"]:
            archive.writestr(item["file_name"], item["data"])
            archive.writestr(item["svg_file_name"], item["svg_data"])
        archive.writestr(
            "benchmark_per_sample.csv",
            attempt["rows"].to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(
            "benchmark_summary.csv",
            attempt["summary"].to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr("benchmark_ai_insight.md", attempt["text"].encode("utf-8"))
    return buffer.getvalue()


def _render(attempt):
    st.markdown(
        """
        <style>
        .st-key-benchmark_ai_panel {position: sticky; top: 1rem; max-height: calc(100vh - 2rem);
        overflow-y: auto; overscroll-behavior: contain; padding-right: .45rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    main, right = st.columns([3.25, 1], gap="large")
    with right:
        with st.container(key="benchmark_ai_panel"):
            st.subheader("AI 解读")
            st.markdown(attempt["text"])
            if attempt["insight_error"]:
                st.caption("AI服务暂不可用，已显示由相同数字事实生成的确定性解读。")
    with main:
        context, coverage = attempt["result"]["context"], attempt["result"]["coverage"]
        st.info(
            f"{'、'.join(BENCHMARK_MODEL_LABELS[model] for model in context['models'])}｜"
            f"{coverage['sample_count']}个起报｜"
            f"严格公共时效 +{context['lead_range_minutes'][0]}—"
            f"+{context['lead_range_minutes'][1]}分钟｜128×128、0.03°统一网格"
        )
        if attempt["request"]["unmatched_requests"]:
            st.warning("；".join(attempt["request"]["unmatched_requests"]))
        controls = st.columns(3)
        controls[0].download_button(
            "下载结果包 ZIP",
            _bundle(attempt),
            "zero_shot_benchmark.zip",
            "application/zip",
            width="stretch",
        )
        controls[1].download_button(
            "下载逐样本 CSV",
            attempt["rows"].to_csv(index=False).encode("utf-8-sig"),
            "benchmark_per_sample.csv",
            "text/csv",
            width="stretch",
        )
        controls[2].download_button(
            "下载汇总 CSV",
            attempt["summary"].to_csv(index=False).encode("utf-8-sig"),
            "benchmark_summary.csv",
            "text/csv",
            width="stretch",
        )

        def render_item(item):
            st.caption(f"图{item['number']}｜{item['label']}")
            st.image(item["data"], width="stretch")
            buttons = st.columns([1, 1, 3])
            buttons[0].download_button(
                "下载 PNG",
                item["data"],
                item["file_name"],
                item["mime"],
                key=f"benchmark_png_{item['number']}",
                width="stretch",
            )
            buttons[1].download_button(
                "下载 SVG",
                item["svg_data"],
                item["svg_file_name"],
                "image/svg+xml",
                key=f"benchmark_svg_{item['number']}",
                width="stretch",
            )

        position = 0
        while position < len(attempt["media"]):
            item = attempt["media"][position]
            if item["full_width"]:
                render_item(item)
                position += 1
                continue
            row = []
            while (
                position < len(attempt["media"])
                and not attempt["media"][position]["full_width"]
                and len(row) < 2
            ):
                row.append(attempt["media"][position])
                position += 1
            for column, row_item in zip(st.columns(2), row):
                with column:
                    render_item(row_item)
        with st.expander("查看结构化指标"):
            st.dataframe(attempt["rows"], width="stretch", hide_index=True)


def render_benchmark_page(*, models, scope):
    signature = _package_signature(BENCHMARK_PACKAGE, BENCHMARK_FENGLEI_DIR)
    catalog = _benchmark_catalog(str(BENCHMARK_PACKAGE), str(BENCHMARK_FENGLEI_DIR), signature)
    models = list(models)
    horizon = common_horizon_minutes(models)
    available_samples = list(benchmark_available_samples(catalog, models, lead_range=(6, horizon)))
    if not available_samples:
        st.error("当前模型组合没有严格匹配的公共起报。")
        return
    selected_samples = list(available_samples)
    if scope == "单一起报":
        sample_id = st.selectbox(
            "起报时间",
            [sample.sample_id for sample in available_samples],
            format_func=lambda value: _sample_label(
                next(sample for sample in available_samples if sample.sample_id == value)
            ),
        )
        selected_samples = [sample for sample in available_samples if sample.sample_id == sample_id]
    elif scope == "自定义多起报":
        middle = max(0, len(available_samples) // 2 - 1)
        default = [sample.sample_id for sample in available_samples[middle : middle + 3]]
        selected_ids = st.multiselect(
            "选择起报（2—8个）",
            [sample.sample_id for sample in available_samples],
            default=default,
            max_selections=8,
            format_func=lambda value: _sample_label(
                next(sample for sample in available_samples if sample.sample_id == value)
            ),
        )
        if not 2 <= len(selected_ids) <= 8:
            st.warning("请选择2—8个起报。")
            return
        selected_samples = [
            sample for sample in available_samples if sample.sample_id in selected_ids
        ]

    single = scope == "单一起报"
    request = default_benchmark_request(single=single, common_horizon=horizon)
    if single:
        default_prompt = (
            f"比较所选{len(models)}个模型，使用20、35、45 dBZ阈值，查看+6—+{horizon}分钟"
            "CSI、POD、FAR和FSS曲线，并生成+60分钟多模型—CREF空间对比、"
            "公共时效汇总图和模型—时效热力图，全面解读。"
        )
    else:
        default_prompt = (
            f"比较所选{len(models)}个模型和{len(selected_samples)}个起报，使用20、35、45 dBZ阈值，"
            f"只看严格公共的+6—+{horizon}分钟，生成CSI、POD、FAR和多尺度FSS曲线、"
            "公共时效汇总图及模型—时效热力图，全面解读。"
        )
    prompt_context = (tuple(models), scope, horizon, len(selected_samples))
    if st.session_state.get("benchmark_prompt_context") != prompt_context:
        st.session_state["benchmark_prompt"] = default_prompt
        st.session_state["benchmark_prompt_context"] = prompt_context
    prompt = st.text_area(
        "检验要求",
        key="benchmark_prompt",
        height=115,
    )
    if st.button("生成检验结果", type="primary", width="stretch"):
        try:
            with st.status("正在生成检验结果…", expanded=True) as status:
                if prompt.strip() == default_prompt.strip():
                    status.write("已载入推荐方案。")
                else:
                    status.write("正在解析自然语言要求…")
                    request = parse_benchmark_request(
                        prompt,
                        single=single,
                        common_horizon=horizon,
                    )
                lead = request["lead_range_minutes"]
                status.write("正在逐样本计算列联表、误差和多尺度FSS…")
                result = _benchmark_result(
                    str(BENCHMARK_PACKAGE),
                    str(BENCHMARK_FENGLEI_DIR),
                    signature,
                    tuple(models),
                    tuple(sample.sample_id for sample in selected_samples),
                    tuple(request["thresholds_dbz"]),
                    lead["start"],
                    lead["end"],
                    tuple(request["fss_windows"]),
                )
                status.write("正在生成图表与证据约束型解读…")
                media, period = _media(
                    catalog,
                    result,
                    request,
                    single_sample_id=selected_samples[0].sample_id if single else None,
                )
                facts = build_benchmark_insight_facts(
                    result,
                    request,
                    period,
                    [{"number": item["number"], "label": item["label"]} for item in media],
                )
                fallback = render_benchmark_deterministic_summary(facts)
                if os.getenv("DEEPSEEK_API_KEY"):
                    try:
                        text, insight_error = generate_benchmark_insight(facts), None
                    except Exception as exc:
                        text, insight_error = fallback, str(exc)
                else:
                    text, insight_error = fallback, "DEEPSEEK_API_KEY未配置"
                rows = result["per_sample"].drop(
                    columns=[
                        column for column in _INTERNAL_COLUMNS if column in result["per_sample"]
                    ]
                )
                summary = result["summary"].drop(
                    columns=[column for column in _INTERNAL_COLUMNS if column in result["summary"]]
                )
                st.session_state["benchmark_attempt"] = {
                    "request": request,
                    "result": result,
                    "media": media,
                    "rows": rows,
                    "summary": summary,
                    "text": text,
                    "insight_error": insight_error,
                }
                status.update(label="检验完成", state="complete", expanded=False)
        except Exception as exc:
            st.error(str(exc))
    attempt = st.session_state.get("benchmark_attempt")
    if attempt:
        _render(attempt)

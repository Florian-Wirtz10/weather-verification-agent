"""Bridge the existing deterministic verification core to JSON web results."""

from __future__ import annotations

from datetime import datetime, timedelta
from io import BytesIO
import math
import os
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent import (
    build_comparison_insight_facts,
    default_multimodel_request,
    generate_comparison_insight,
    parse_multimodel_request,
    render_comparison_deterministic_summary,
)
from agent.benchmark_insight import (
    build_benchmark_insight_facts,
    generate_benchmark_insight,
    render_benchmark_deterministic_summary,
)
from agent.benchmark_request import default_benchmark_request, parse_benchmark_request
from data_io.benchmark import BENCHMARK_MODEL_LABELS
from data_io.forecast_sources import MODEL_LABELS, discover_multimodel_initializations
from reports import figure_to_png, figure_to_svg
from verification.benchmark import (
    benchmark_available_samples,
    benchmark_frames,
    common_horizon_minutes,
    run_benchmark,
    summarize_benchmark_period,
)
from verification.multimodel import (
    common_valid_times,
    run_multimodel_batch,
    run_multimodel_case,
    run_multimodel_same_valid_time,
)
from visualization.benchmark_plots import (
    plot_benchmark_pod_distribution,
    plot_benchmark_far_bars,
    plot_benchmark_fss_curves,
    plot_benchmark_period_summary,
    plot_benchmark_score_curves,
    plot_benchmark_score_heatmap,
    plot_benchmark_spatial_comparison,
    render_benchmark_reflectivity_animation,
)
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
from webapi.config import (
    benchmark_catalog,
    benchmark_model_ids,
    is_benchmark,
    regular_catalog,
    regular_model_ids,
)
from webapi.settings import WEB_JOB_OUTPUT_ROOT


OUTPUT_ROOT = WEB_JOB_OUTPUT_ROOT


def default_prompt(case_id: str, models: list[str], scope: str) -> str:
    """Return the visible deterministic recommendation for the selected workflow."""
    if is_benchmark(models):
        horizon = common_horizon_minutes(benchmark_model_ids(models))
        subject = "、".join(models)
        spatial = "生成空间分布动画，" if scope == "single" else "生成"
        return (
            f"比较{subject}，使用20、35、45 dBZ阈值，查看+6—+{horizon}分钟的"
            f"CSI、POD、FAR和多尺度FSS，{spatial}时效曲线、热力图，并在20 dBZ阈值下"
            "生成公共样本汇总和解读。"
        )
    subject = "和".join(models)
    if scope == "single":
        return (
            f"使用20、30、40 dBZ阈值，查看6—180分钟{subject}预报与CREF观测的"
            "空间演变、CSI、POD、FAR曲线，并在20 dBZ下综合解读。"
        )
    if scope == "custom":
        return (
            f"比较所选多个起报的{subject}预报，使用20 dBZ阈值，对照共同有效时间的"
            "CREF观测，并分析起报更新后回波位置、强度和评分的变化。"
        )
    return (
        f"使用20、30、40 dBZ阈值，汇总{subject}的CSI、POD、FAR曲线，生成"
        "起报—时效热力图、多阈值分布、变量相关性和预报率—观测率图并全面解读。"
    )


def list_initializations(case_id: str, models: list[str], complete_only: bool) -> list[dict]:
    if is_benchmark(models):
        catalog = benchmark_catalog()
        selected = benchmark_model_ids(models)
        horizon = common_horizon_minutes(selected)
        samples = benchmark_available_samples(catalog, selected, lead_range=(6, horizon))
        return [
            {
                "id": str(sample.sample_id),
                "time_utc": sample.analysis_time_utc.isoformat(),
                "time_bjt": sample.analysis_time_bjt.isoformat(),
                "label": (
                    f"{sample.analysis_time_bjt:%Y-%m-%d %H:%M} BJT"
                    f"（{sample.analysis_time_utc:%m-%d %H:%M} UTC）"
                ),
                "available_leads": list(range(6, horizon + 1, 6)),
                "complete": True,
            }
            for sample in samples
        ]
    catalog = regular_catalog(case_id)
    model_ids = regular_model_ids(models)
    cases = discover_multimodel_initializations(
        catalog,
        model_ids,
        lead_range=(6, 180),
        common_only=len(model_ids) > 1,
    )
    if complete_only:
        cases = [case for case in cases if case["complete"]]
    return [
        {
            "id": case["case_id"],
            "time_utc": case["init_time"].isoformat(),
            "time_bjt": case["init_time_bjt"].isoformat(),
            "label": (
                f"{case['init_time_bjt']:%Y-%m-%d %H:%M} BJT｜{len(case['available_leads'])}个时效"
            ),
            "available_leads": case["available_leads"],
            "complete": case["complete"],
            "domain": case.get("domain"),
        }
        for case in cases
    ]


def list_common_valid_times(
    case_id: str, models: list[str], initialization_ids: list[str]
) -> list[dict]:
    if is_benchmark(models):
        return []
    catalog = regular_catalog(case_id)
    model_ids = regular_model_ids(models)
    cases = discover_multimodel_initializations(
        catalog,
        model_ids,
        lead_range=(6, 180),
        common_only=len(model_ids) > 1,
    )
    selected = [case for case in cases if case["case_id"] in set(initialization_ids)]
    if {case["case_id"] for case in selected} != set(initialization_ids):
        raise ValueError("部分所选起报已不在当前严格匹配目录中，请刷新后重选")
    values = common_valid_times(selected)
    return [
        {
            "value": value.isoformat(),
            "label": f"{value + timedelta(hours=8):%Y-%m-%d %H:%M} BJT",
        }
        for value in values
    ]


class ArtifactStore:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.directory = OUTPUT_ROOT / job_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.gallery: list[dict] = []

    def add_bytes(self, name: str, data: bytes, mime: str, label: str | None = None) -> dict:
        path = self.directory / name
        path.write_bytes(data)
        item = {
            "name": name,
            "label": label or name,
            "mime": mime,
            "url": f"/api/jobs/{self.job_id}/artifacts/{name}",
        }
        return item

    def add_figure(
        self,
        stem: str,
        label: str,
        figure,
        *,
        gallery: bool = True,
        svg: bool = True,
    ) -> dict:
        try:
            png = self.add_bytes(f"{stem}.png", figure_to_png(figure, dpi=220), "image/png", label)
            if svg:
                vector = self.add_bytes(
                    f"{stem}.svg", figure_to_svg(figure), "image/svg+xml", label
                )
                png["svg_url"] = vector["url"]
            if gallery:
                self.gallery.append(png)
            return png
        finally:
            plt.close(figure)

    def add_table(self, name: str, frame: pd.DataFrame) -> dict:
        return self.add_bytes(
            name,
            frame.to_csv(index=False).encode("utf-8-sig"),
            "text/csv",
            name,
        )

    def bundle(self, name: str = "verification_result.zip") -> dict:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.directory.iterdir()):
                if path.name != name and path.is_file():
                    archive.write(path, path.name)
        return self.add_bytes(name, buffer.getvalue(), "application/zip", "完整结果包")


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _frame_subset(leads, maximum=30):
    leads = list(leads)
    if len(leads) <= maximum:
        return leads
    positions = {round(index * (len(leads) - 1) / (maximum - 1)) for index in range(maximum)}
    return [leads[index] for index in sorted(positions)]


def _single_rows(result) -> pd.DataFrame:
    rows = []
    for entry in result["per_lead"]:
        for model_id, threshold_map in entry["models"].items():
            for threshold, values in threshold_map.items():
                rows.append(
                    {
                        "model_id": model_id,
                        "model": MODEL_LABELS[model_id],
                        "init_time_utc": result["context"]["init_time_utc"],
                        "valid_time_utc": entry["valid_time_utc"],
                        "lead_minutes": entry["lead_minutes"],
                        "threshold_dbz": threshold,
                        **values["contingency"],
                        **{
                            key: values.get(key)
                            for key in (
                                "csi",
                                "pod",
                                "far",
                                "bias_score",
                                "forecast_mean_dbz",
                                "observation_mean_dbz",
                                "forecast_max_dbz",
                                "observation_max_dbz",
                                "reflectivity_mae_dbz",
                                "centroid_distance_km",
                            )
                        },
                    }
                )
    return pd.DataFrame(rows)


def _curve_series(
    frame: pd.DataFrame,
    model_column: str = "model_id",
    *,
    metrics=None,
    fss_windows=(5,),
) -> list[dict]:
    series = []
    requested = list(metrics or ("csi", "pod", "far", "bias_score", "fss"))
    curve_metrics = []
    for metric in requested:
        candidates = [f"fss_w{window}" for window in fss_windows] if metric == "fss" else [metric]
        curve_metrics.extend(candidate for candidate in candidates if candidate in frame)
    curve_metrics = list(dict.fromkeys(curve_metrics))
    for (model, threshold), rows in frame.groupby([model_column, "threshold_dbz"], sort=False):
        rows = rows.sort_values("lead_minutes")
        for metric in curve_metrics:
            points = [
                [int(row.lead_minutes), _finite(getattr(row, metric))] for row in rows.itertuples()
            ]
            series.append(
                {
                    "model": str(model),
                    "threshold_dbz": float(threshold),
                    "metric": metric,
                    "points": points,
                }
            )
    return series


def _summary_table(frame: pd.DataFrame, model_column="model_id") -> list[dict]:
    metrics = [
        metric
        for metric in ("csi", "pod", "far", "bias_score", "mae", "rmse", "fss_w5")
        if metric in frame
    ]
    if not metrics or frame.empty:
        return []
    grouped = frame.groupby([model_column, "threshold_dbz"], as_index=False)[metrics].mean()
    return _json_safe(grouped.to_dict("records"))


def _insight(result, request, catalog_items):
    facts = build_comparison_insight_facts(
        result,
        request["metrics"],
        figure_catalog=catalog_items,
        diagnostic_threshold_dbz=request["diagnostic_threshold_dbz"],
        correlation_method=request["correlation_method"],
    )
    fallback = render_comparison_deterministic_summary(facts)
    if not os.getenv("DEEPSEEK_API_KEY"):
        return fallback, "确定性解读"
    try:
        return generate_comparison_insight(facts), "DeepSeek"
    except Exception:
        return fallback, "确定性解读"


def _regular_request(payload, *, batch):
    scope = payload["scope"]
    if payload.get("use_default_plan", True) or not payload.get("prompt", "").strip():
        return default_multimodel_request(batch=batch, scope=scope)
    return parse_multimodel_request(payload["prompt"], batch=batch, scope=scope)


def _regular_single(payload, job_id, update):
    store = ArtifactStore(job_id)
    catalog = regular_catalog(payload["case_id"])
    model_ids = regular_model_ids(payload["models"])
    cases = discover_multimodel_initializations(
        catalog, model_ids, lead_range=(6, 180), common_only=len(model_ids) > 1
    )
    case = next(
        (item for item in cases if item["case_id"] == payload["initialization_ids"][0]),
        None,
    )
    if case is None:
        raise ValueError("所选起报已不在当前严格匹配目录中")
    update("parse", "正在解析并校验检验要求", 0.08)
    request = _regular_request(payload, batch=False)
    lead = request["lead_range_minutes"]
    available = [
        value for value in case["available_leads"] if lead["start"] <= value <= lead["end"]
    ]
    views = set(request["views"])
    animation_leads = []
    if "reflectivity_animation" in views:
        step = request["animation"]["frame_step"]
        animation_leads = _frame_subset(available[::step])
    static_leads = []
    if "spatial_comparison" in views and available:
        requested_lead = request.get("spatial_lead_minutes")
        static_leads = [requested_lead if requested_lead in available else available[0]]
    frame_leads = sorted(set(animation_leads + static_leads))
    update("align", "正在严格匹配预报、观测及空间网格", 0.18)
    result = run_multimodel_case(
        catalog,
        model_ids,
        case["init_time"],
        payload["bbox"],
        thresholds=request["thresholds_dbz"],
        lead_range=(lead["start"], lead["end"]),
        frame_leads=frame_leads,
    )
    update("render", "正在生成时效空间帧", 0.48)
    spatial = []
    for index, frame_lead in enumerate(frame_leads, start=1):
        item = store.add_figure(
            f"spatial_{frame_lead:03d}",
            f"+{frame_lead}分钟预报—CREF空间对比",
            plot_multimodel_reflectivity_frame(result["frames"][frame_lead]),
            gallery=False,
            svg=False,
        )
        spatial.append({"lead_minutes": frame_lead, **item})
        update(
            "render",
            f"正在生成空间帧 {index}/{len(frame_leads)}",
            0.48 + 0.22 * index / max(1, len(frame_leads)),
        )
    if animation_leads:
        animation = store.add_bytes(
            "reflectivity_animation.gif",
            render_multimodel_reflectivity_animation(
                [result["frames"][value] for value in animation_leads],
                fps=request["animation"]["fps"],
            ),
            "image/gif",
            "预报—CREF同步空间演变动画",
        )
        store.gallery.append(animation)
    elif static_leads:
        store.gallery.append(spatial[0])
    rows = _single_rows(result)
    if "score_curve" in views:
        for metric in request["score_curve_metrics"]:
            store.add_figure(
                f"score_{metric}",
                f"{metric.upper()}随预报时效变化",
                plot_multimodel_score_curves(result, metric),
            )
    if "multi_metric_curve" in views:
        store.add_figure(
            "multi_metric",
            f"{request['diagnostic_threshold_dbz']:g} dBZ多指标比较",
            plot_multimodel_multi_metric_curves(
                result,
                request["multi_metric_metrics"],
                request["diagnostic_threshold_dbz"],
            ),
        )
    update("interpret", "正在生成证据约束型AI解读", 0.82)
    insight, insight_source = _insight(
        result,
        request,
        [
            {"number": index, "label": item["label"]}
            for index, item in enumerate(store.gallery, start=1)
        ],
    )
    metrics_file = store.add_table("metrics.csv", rows)
    insight_file = store.add_bytes("ai_insight.md", insight.encode(), "text/markdown", "AI解读")
    bundle = store.bundle()
    return _json_safe(
        {
            "kind": "single",
            "request": request,
            "context": result["context"],
            "coverage": result["coverage"],
            "curves": (
                _curve_series(
                    rows,
                    metrics=list(
                        dict.fromkeys(
                            request["score_curve_metrics"]
                            + request["multi_metric_metrics"]
                        )
                    ),
                )
                if views.intersection({"score_curve", "multi_metric_curve"})
                else []
            ),
            "summary_table": _summary_table(rows),
            "spatial_frames": spatial,
            "gallery": store.gallery,
            "insight": insight,
            "insight_source": insight_source,
            "downloads": [bundle, metrics_file, insight_file],
        }
    )


def _regular_batch(payload, job_id, update):
    store = ArtifactStore(job_id)
    catalog = regular_catalog(payload["case_id"])
    model_ids = regular_model_ids(payload["models"])
    request = _regular_request(payload, batch=True)
    lead = request["lead_range_minutes"]
    cases = discover_multimodel_initializations(
        catalog,
        model_ids,
        lead_range=(lead["start"], lead["end"]),
        common_only=len(model_ids) > 1,
    )
    selected_ids = set(payload.get("initialization_ids") or [])
    selected_cases = [case for case in cases if case["case_id"] in selected_ids]
    if payload["scope"] == "custom" and len(selected_cases) != len(selected_ids):
        raise ValueError("部分所选起报已不在当前严格匹配目录中，请刷新后重选")
    initializations = (
        [case["init_time"] for case in selected_cases] if payload["scope"] == "custom" else None
    )
    comparison = None
    if "same_valid_time_comparison" in request["views"] and not payload.get("comparison_valid_time"):
        raise ValueError("同有效时间对比需要选择共同有效时间")
    if (
        payload["scope"] == "custom"
        and payload.get("comparison_valid_time")
        and "same_valid_time_comparison" in request["views"]
    ):
        update("align", "正在构建共同有效时间对比", 0.15)
        comparison = run_multimodel_same_valid_time(
            catalog,
            model_ids,
            initializations,
            datetime.fromisoformat(payload["comparison_valid_time"]),
            payload["bbox"],
            thresholds=request["thresholds_dbz"],
        )

    def progress(done, total, label):
        update("compute", f"正在计算起报 {done}/{total}｜{label}", 0.2 + 0.45 * done / total)

    update("compute", "正在流式计算逐起报评分", 0.2)
    result = run_multimodel_batch(
        catalog,
        model_ids,
        payload["bbox"],
        thresholds=request["thresholds_dbz"],
        lead_range=(lead["start"], lead["end"]),
        initializations=initializations,
        complete_only=payload["complete_only"],
        progress_callback=progress,
    )
    if comparison is not None:
        result["same_valid_time_comparison"] = comparison
    update("render", "正在生成综合统计图表", 0.7)
    views = set(request["views"])
    if comparison is not None and "same_valid_time_comparison" in views:
        for model_id in comparison["model_ids"]:
            store.add_figure(
                f"same_valid_{model_id}",
                f"{MODEL_LABELS[model_id]}不同起报在同一有效时间的预报",
                plot_multimodel_same_valid_time_comparison(comparison, model_id),
            )
    if "score_curve" in views:
        for metric in request["score_curve_metrics"]:
            store.add_figure(
                f"batch_{metric}",
                f"汇总{metric.upper()}随预报时效变化",
                plot_multimodel_batch_score_curves(result["summary"], metric),
            )
    if "multi_metric_curve" in views:
        store.add_figure(
            "batch_multi_metric",
            f"{request['diagnostic_threshold_dbz']:g} dBZ多指标比较",
            plot_multimodel_batch_multi_metric_curves(
                result["summary"],
                request["multi_metric_metrics"],
                request["diagnostic_threshold_dbz"],
            ),
        )
    if "score_heatmap" in views:
        store.add_figure(
            "batch_heatmap",
            "起报—时效评分矩阵",
            plot_multimodel_score_heatmaps(
                result["per_case"],
                request["diagnostic_metric"],
                request["diagnostic_threshold_dbz"],
            ),
        )
    if "score_distribution" in views:
        store.add_figure(
            "batch_distribution",
            "多阈值评分分布",
            plot_multimodel_score_distribution(
                result["per_case"], request["diagnostic_metric"], request["thresholds_dbz"]
            ),
        )
    if "forecast_observed_rate_scatter" in views:
        store.add_figure(
            "batch_rate_scatter",
            "预报率—观测率散点比较",
            plot_multimodel_forecast_observed_rate_scatter(
                result["per_case"], request["thresholds_dbz"]
            ),
        )
    if "variable_correlation_heatmap" in views:
        store.add_figure(
            "batch_correlation",
            "变量相关性热力图",
            plot_multimodel_variable_correlation_heatmap(
                result["per_case"],
                request["diagnostic_threshold_dbz"],
                method=request["correlation_method"],
            ),
        )
    update("interpret", "正在生成综合AI解读", 0.88)
    insight, insight_source = _insight(
        result,
        request,
        [
            {"number": index, "label": item["label"]}
            for index, item in enumerate(store.gallery, start=1)
        ],
    )
    rows = result["per_case"].drop(columns=["ets"], errors="ignore")
    metrics_file = store.add_table("metrics.csv", rows)
    insight_file = store.add_bytes("ai_insight.md", insight.encode(), "text/markdown", "AI解读")
    bundle = store.bundle()
    return _json_safe(
        {
            "kind": "custom" if payload["scope"] == "custom" else "all",
            "request": request,
            "context": result["context"],
            "coverage": result["coverage"],
            "curves": (
                _curve_series(
                    result["summary"],
                    metrics=list(
                        dict.fromkeys(
                            request["score_curve_metrics"]
                            + request["multi_metric_metrics"]
                        )
                    ),
                )
                if views.intersection({"score_curve", "multi_metric_curve"})
                else []
            ),
            "summary_table": _summary_table(rows),
            "spatial_frames": [],
            "gallery": store.gallery,
            "insight": insight,
            "insight_source": insight_source,
            "downloads": [bundle, metrics_file, insight_file],
        }
    )


def _benchmark_artifact_manifest(request, models, animation_leads, spatial, gallery):
    """Validate and describe the exact renderer output promised by the request."""
    expected_gallery = []
    views = set(request["views"])
    if animation_leads:
        expected_gallery.append("benchmark_reflectivity_animation.gif")
    if "spatial_comparison" in views:
        expected_gallery.append("benchmark_spatial.png")
    if "score_curve" in views:
        expected_gallery.extend(
            f"benchmark_{metric}.png"
            for metric in request.get("score_curve_metrics", request["metrics"])
            if metric != "fss"
        )
    if "fss" in request["metrics"] and "fss_curve" in views:
        expected_gallery.append("benchmark_fss.png")
    if "period_summary" in views:
        expected_gallery.append("benchmark_period.png")
    if "score_heatmap" in views:
        expected_gallery.append("benchmark_heatmap.png")
    if "pod_distribution" in views:
        expected_gallery.extend(
            f"benchmark_pod_distribution_{threshold:g}.png"
            for threshold in request["thresholds_dbz"]
        )
    if "far_bar" in views:
        expected_gallery.append("benchmark_far_bars.png")
    actual_gallery = [item["name"] for item in gallery]
    if actual_gallery != expected_gallery:
        raise RuntimeError(
            "benchmark artifacts do not match the validated request: "
            f"expected {expected_gallery}, got {actual_gallery}"
        )
    actual_leads = [int(item["lead_minutes"]) for item in spatial]
    if actual_leads != animation_leads:
        raise RuntimeError(
            "benchmark animation frames do not match the requested lead sequence: "
            f"expected {animation_leads}, got {actual_leads}"
        )
    return {
        "models": list(models),
        "thresholds_dbz": list(request["thresholds_dbz"]),
        "metrics": list(request["metrics"]),
        "fss_windows": list(request["fss_windows"]),
        "diagnostic_threshold_dbz": request["diagnostic_threshold_dbz"],
        "diagnostic_metric": request["diagnostic_metric"],
        "animation": {
            "enabled": bool(animation_leads),
            "frame_count": len(animation_leads),
            "lead_minutes": list(animation_leads),
            "panel_order": [*models, "CREF"],
            "fps": request["animation"]["fps"],
        },
        "gallery_files": actual_gallery,
    }


def _benchmark(payload, job_id, update):
    store = ArtifactStore(job_id)
    catalog = benchmark_catalog()
    models = benchmark_model_ids(payload["models"])
    horizon = common_horizon_minutes(models)
    single = payload["scope"] == "single"
    update("parse", "正在解析公共样本检验要求", 0.08)
    request = (
        default_benchmark_request(single=single, common_horizon=horizon)
        if payload.get("use_default_plan", True) or not payload.get("prompt", "").strip()
        else parse_benchmark_request(payload["prompt"], single=single, common_horizon=horizon)
    )
    lead = request["lead_range_minutes"]
    available = benchmark_available_samples(
        catalog, models, lead_range=(lead["start"], lead["end"])
    )
    selected_ids = {int(value) for value in payload.get("initialization_ids") or []}
    samples = (
        [sample for sample in available if sample.sample_id in selected_ids]
        if payload["scope"] != "all"
        else list(available)
    )
    if not samples:
        raise ValueError("当前模型、起报和时效范围没有严格公共样本")
    if payload["scope"] != "all" and selected_ids != {sample.sample_id for sample in samples}:
        raise ValueError("部分所选起报已不在当前严格公共样本中，请刷新后重选")

    def progress(done, total, sample_id):
        update(
            "compute", f"正在计算公共样本 {done}/{total}｜ID {sample_id}", 0.18 + 0.5 * done / total
        )

    update("compute", "正在计算列联表、连续误差和多尺度FSS", 0.18)
    result = run_benchmark(
        catalog,
        models,
        sample_ids=[sample.sample_id for sample in samples],
        thresholds=request["thresholds_dbz"],
        lead_range=(lead["start"], lead["end"]),
        fss_windows=request["fss_windows"],
        progress_callback=progress,
    )
    update("render", "正在生成多模型公共样本图表", 0.72)
    views = set(request["views"])
    animation_leads = []
    if single and "reflectivity_animation" in views:
        step = request["animation"]["frame_step"]
        native_leads = list(range(lead["start"], lead["end"] + 1, 6))
        animation_leads = _frame_subset(native_leads[::step])
    static_leads = []
    if single and "spatial_comparison" in views:
        static_leads = [request["spatial_lead_minutes"] or lead["start"]]
    frame_leads = sorted(set(animation_leads + static_leads))
    spatial = []
    frame_map = {}
    if frame_leads:
        loaded_frames = benchmark_frames(catalog, samples[0].sample_id, models, frame_leads)
        frame_map = {frame["lead_minutes"]: frame for frame in loaded_frames}
        for index, frame_lead in enumerate(animation_leads, start=1):
            item = store.add_figure(
                f"benchmark_spatial_{frame_lead:03d}",
                f"+{frame_lead}分钟多模型—CREF空间对比",
                plot_benchmark_spatial_comparison(frame_map[frame_lead]),
                gallery=False,
                svg=False,
            )
            spatial.append({"lead_minutes": frame_lead, **item})
            update(
                "render",
                f"正在生成多模型空间帧 {index}/{len(animation_leads)}",
                0.72 + 0.08 * index / max(1, len(animation_leads)),
            )
    if animation_leads:
        animation = store.add_bytes(
            "benchmark_reflectivity_animation.gif",
            render_benchmark_reflectivity_animation(
                [frame_map[value] for value in animation_leads],
                fps=request["animation"]["fps"],
            ),
            "image/gif",
            "多模型—CREF同步空间分布动画",
        )
        store.gallery.append(animation)
    if static_leads:
        static_lead = static_leads[0]
        store.add_figure(
            "benchmark_spatial",
            f"+{static_lead}分钟多模型—CREF空间对比",
            plot_benchmark_spatial_comparison(frame_map[static_lead]),
        )
    if "score_curve" in views:
        for metric in request.get("score_curve_metrics", request["metrics"]):
            if metric == "fss":
                continue
            store.add_figure(
                f"benchmark_{metric}",
                f"多模型{metric.upper()}随预报时效变化",
                plot_benchmark_score_curves(result["summary"], metric, request["thresholds_dbz"]),
            )
    if "fss" in request["metrics"] and "fss_curve" in views:
        store.add_figure(
            "benchmark_fss",
            "多模型多尺度FSS随预报时效变化",
            plot_benchmark_fss_curves(
                result["summary"],
                request.get("fss_thresholds_dbz", request["thresholds_dbz"]),
                request["fss_windows"],
            ),
        )
    period = summarize_benchmark_period(result["per_sample"], fss_windows=request["fss_windows"])
    if "period_summary" in views:
        store.add_figure(
            "benchmark_period",
            "严格公共时效综合比较",
            plot_benchmark_period_summary(
                period,
                request["metrics"],
                request["diagnostic_threshold_dbz"],
                fss_windows=request["fss_windows"],
            ),
        )
    if "score_heatmap" in views:
        store.add_figure(
            "benchmark_heatmap",
            "模型—时效评分热力图",
            plot_benchmark_score_heatmap(
                result["summary"],
                request["diagnostic_metric"],
                request["diagnostic_threshold_dbz"],
                fss_window=max(request["fss_windows"]),
            ),
        )
    if "pod_distribution" in views:
        for threshold in request["thresholds_dbz"]:
            store.add_figure(
                f"benchmark_pod_distribution_{threshold:g}",
                f"{threshold:g} dBZ三模型POD箱线图（起报×时效样本）",
                plot_benchmark_pod_distribution(result["per_sample"], threshold),
            )
    if "far_bar" in views:
        store.add_figure(
            "benchmark_far_bars", "FAR分组柱状图（横轴预报时效）",
            plot_benchmark_far_bars(result["summary"], request["thresholds_dbz"]),
        )
    artifact_manifest = _benchmark_artifact_manifest(
        request, models, animation_leads, spatial, store.gallery
    )
    facts = build_benchmark_insight_facts(
        result,
        request,
        period,
        [
            {"number": index, "label": item["label"]}
            for index, item in enumerate(store.gallery, start=1)
        ],
    )
    fallback = render_benchmark_deterministic_summary(facts)
    if os.getenv("DEEPSEEK_API_KEY"):
        try:
            insight, insight_source = generate_benchmark_insight(facts), "DeepSeek"
        except Exception:
            insight, insight_source = fallback, "确定性解读"
    else:
        insight, insight_source = fallback, "确定性解读"
    rows = result["per_sample"].drop(
        columns=[
            column
            for column in result["per_sample"]
            if column.endswith(("_sum", "_numerator", "_denominator"))
        ],
        errors="ignore",
    )
    metrics_file = store.add_table("benchmark_metrics.csv", rows)
    insight_file = store.add_bytes("ai_insight.md", insight.encode(), "text/markdown", "AI解读")
    bundle = store.bundle()
    labels = {model: BENCHMARK_MODEL_LABELS[model] for model in models}
    curves = (
        _curve_series(
            result["summary"],
            model_column="model",
            metrics=request["metrics"],
            fss_windows=request["fss_windows"],
        )
        if views.intersection({"score_curve", "fss_curve"})
        else []
    )
    for series in curves:
        series["model"] = labels.get(series["model"], series["model"])
    table = _summary_table(period, model_column="model")
    for row in table:
        row["model"] = labels.get(row["model"], row["model"])
    return _json_safe(
        {
            "kind": "benchmark",
            "request": request,
            "context": {**result["context"], "model_labels": list(labels.values())},
            "coverage": {
                **result["coverage"],
                "animation_frame_count": len(animation_leads),
                "animation_leads": animation_leads,
            },
            "artifact_manifest": artifact_manifest,
            "curves": curves,
            "summary_table": table,
            "spatial_frames": spatial,
            "gallery": store.gallery,
            "insight": insight,
            "insight_source": insight_source,
            "downloads": [bundle, metrics_file, insight_file],
        }
    )


def execute_task(payload: dict, job_id: str, update):
    """JobManager entry point."""
    try:
        if is_benchmark(payload["models"]):
            return _benchmark(payload, job_id, update)
        if payload["scope"] == "single":
            return _regular_single(payload, job_id, update)
        return _regular_batch(payload, job_id, update)
    finally:
        # Jobs are serialized; release figures even when rendering fails before
        # a figure can be handed to ArtifactStore.add_figure().
        plt.close("all")


def artifact_path(job_id: str, name: str) -> Path:
    if len(job_id) != 12 or any(character not in "0123456789abcdef" for character in job_id):
        raise FileNotFoundError(name)
    safe_name = Path(name).name
    if safe_name != name:
        raise FileNotFoundError(name)
    path = (OUTPUT_ROOT / job_id / safe_name).resolve()
    root = (OUTPUT_ROOT / job_id).resolve()
    if root.parent != OUTPUT_ROOT.resolve() or path.parent != root or not path.is_file():
        raise FileNotFoundError(name)
    return path

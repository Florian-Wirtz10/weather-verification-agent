"""Publication-style figures for the China CREF zero-shot benchmark."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.geography import add_admin_boundaries
from visualization.plots import _reflectivity_cmap_norm, _reflectivity_display_data
from visualization.style import style_axes


MODEL_COLORS = {
    "ConvLSTM": "#4E79A7",
    "PredRNN": "#59A14F",
    "Earthformer": "#9C755F",
    "NowCastNet": "#F28E2B",
    "SimVP": "#76B7B2",
    "DiffCast": "#B07AA1",
    "StormDiT": "#E15759",
    "Fenglei": "#EDC948",
    "Persistence": "#6B7280",
}
METRIC_LABELS = {
    "csi": "CSI ↑",
    "pod": "POD ↑",
    "far": "FAR ↓",
    "bias_score": "Frequency bias → 1",
    "mae": "MAE (dBZ) ↓",
    "rmse": "RMSE (dBZ) ↓",
    "fss": "FSS ↑",
}


def _score_limits(axis, metric):
    if metric in {"csi", "pod", "far", "fss"}:
        axis.set_ylim(0, 1)
    elif metric == "bias_score":
        axis.axhline(1, color="#7A7A7A", linestyle=":", linewidth=1)


def plot_benchmark_score_curves(summary: pd.DataFrame, metric: str, thresholds):
    """Plot one selected score for all models, faceted by threshold."""
    metric = str(metric)
    thresholds = list(thresholds)
    if metric not in {"csi", "pod", "far", "bias_score", "mae", "rmse"}:
        raise ValueError(f"unsupported benchmark curve metric: {metric}")
    if summary.empty:
        raise ValueError("benchmark summary is empty")
    if metric in {"mae", "rmse"}:
        thresholds = [float(summary["threshold_dbz"].min())]
    column_count = min(2, len(thresholds))
    row_count = math.ceil(len(thresholds) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6.4 * column_count, 4.7 * row_count),
        sharey=metric not in {"mae", "rmse"},
        squeeze=False,
        layout="constrained",
    )
    for axis, threshold in zip(axes.flat, thresholds):
        selected = summary[np.isclose(summary["threshold_dbz"], float(threshold))]
        for model, rows in selected.groupby("model", sort=False):
            rows = rows.sort_values("lead_minutes")
            axis.plot(
                rows["lead_minutes"],
                rows[metric],
                color=MODEL_COLORS.get(model, "#333333"),
                linewidth=2,
                marker="o",
                markersize=3,
                label=model,
            )
        axis.set_title(f"{threshold:g} dBZ" if metric not in {"mae", "rmse"} else "Common samples")
        axis.set_xlabel("Lead time (min)")
        axis.set_ylabel(METRIC_LABELS[metric])
        _score_limits(axis, metric)
        style_axes(axis)
    for axis in axes.flat[len(thresholds) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle(f"Zero-shot benchmark | {METRIC_LABELS[metric]} by forecast lead")
    return fig


def plot_benchmark_fss_curves(summary, threshold: float, windows=(1, 3, 5)):
    selected = summary[np.isclose(summary["threshold_dbz"], float(threshold))]
    if selected.empty:
        raise ValueError("benchmark summary has no selected FSS threshold")
    windows = tuple(windows)
    fig, axes = plt.subplots(
        1,
        len(windows),
        figsize=(6.2 * len(windows), 5.2),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for axis, window in zip(axes.flat, windows):
        column = f"fss_w{window}"
        for model, rows in selected.groupby("model", sort=False):
            rows = rows.sort_values("lead_minutes")
            axis.plot(
                rows["lead_minutes"],
                rows[column],
                color=MODEL_COLORS.get(model, "#333333"),
                linewidth=2,
                marker="o",
                markersize=3,
                label=model,
            )
        axis.set(
            title=f"{window}×{window} pixels",
            xlabel="Lead time (min)",
            ylabel="FSS ↑",
            ylim=(0, 1),
        )
        style_axes(axis)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle(f"Zero-shot benchmark | FSS at {threshold:g} dBZ")
    return fig


def plot_benchmark_period_summary(period: pd.DataFrame, metrics, threshold: float, *, fss_window=5):
    selected = period[np.isclose(period["threshold_dbz"], float(threshold))].copy()
    if selected.empty:
        raise ValueError("benchmark period summary is empty")
    metrics = list(dict.fromkeys(metrics))[:6]
    columns = 2 if len(metrics) > 1 else 1
    rows = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.2 * columns, 4.6 * rows),
        squeeze=False,
        layout="constrained",
    )
    order = [model for model in MODEL_COLORS if model in set(selected["model"])]
    selected = selected.set_index("model").reindex(order)
    for axis, metric in zip(axes.flat, metrics):
        column = f"fss_w{fss_window}" if metric == "fss" else metric
        axis.bar(
            np.arange(len(order)),
            selected[column],
            color=[MODEL_COLORS[model] for model in order],
            width=0.7,
        )
        axis.set_xticks(np.arange(len(order)), order, rotation=32, ha="right")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.set_title(METRIC_LABELS[metric])
        _score_limits(axis, metric)
        style_axes(axis)
    for axis in axes.flat[len(metrics) :]:
        axis.set_visible(False)
    fig.suptitle(
        f"Pooled common-window comparison | {threshold:g} dBZ | "
        f"FSS uses {fss_window}×{fss_window} pixels"
    )
    return fig


def plot_benchmark_score_heatmap(summary, metric: str, threshold: float, *, fss_window=5):
    column = f"fss_w{fss_window}" if metric == "fss" else metric
    selected = summary[np.isclose(summary["threshold_dbz"], float(threshold))]
    matrix = selected.pivot(index="model", columns="lead_minutes", values=column)
    order = [model for model in MODEL_COLORS if model in matrix.index]
    matrix = matrix.reindex(order)
    fig, axis = plt.subplots(
        figsize=(11.5, max(4.2, 0.55 * len(order) + 1.8)), layout="constrained"
    )
    values = matrix.to_numpy(dtype=float)
    if metric in {"mae", "rmse", "far"}:
        cmap = "magma_r"
    elif metric == "bias_score":
        cmap = "RdBu_r"
    else:
        cmap = "viridis"
    image = axis.imshow(values, aspect="auto", cmap=cmap)
    axis.set_xticks(np.arange(matrix.shape[1]), [str(value) for value in matrix.columns])
    axis.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    axis.set_xlabel("Lead time (min)")
    axis.set_ylabel("Model")
    axis.set_title(f"{METRIC_LABELS[metric]} | {threshold:g} dBZ")
    fig.colorbar(image, ax=axis, label=METRIC_LABELS[metric])
    return fig


def plot_benchmark_spatial_comparison(frame):
    panels = list(frame["forecasts"].items()) + [("CREF observation", frame["truth"])]
    columns = 3 if len(panels) > 4 else 2
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 4.5 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    lat, lon = frame["lat"], frame["lon"]
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
    cmap, norm = _reflectivity_cmap_norm()
    image = None
    for axis, (label, values) in zip(axes.flat, panels):
        image = axis.imshow(
            _reflectivity_display_data(values)[::-1, :],
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            aspect="auto",
        )
        axis.set_title(label)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        add_admin_boundaries(axis, extent)
    for axis in axes.flat[len(panels) :]:
        axis.set_visible(False)
    if image is not None:
        fig.colorbar(
            image,
            ax=[axis for axis in axes.flat if axis.get_visible()],
            label="Reflectivity (dBZ)",
            shrink=0.88,
        )
    sample = frame["sample"]
    fig.suptitle(
        f"Zero-shot spatial comparison | Init {sample.analysis_time_bjt:%Y-%m-%d %H:%M} BJT | "
        f"+{frame['lead_minutes']} min"
    )
    return fig

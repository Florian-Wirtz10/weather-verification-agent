"""Publication-style figures for the China CREF zero-shot benchmark."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.geography import add_admin_boundaries
from visualization.plots import (
    _figures_to_gif,
    _reflectivity_cmap_norm,
    _reflectivity_display_data,
)
from visualization.style import style_axes


MODEL_COLORS = {
    "ConvLSTM": "#4E79A7",
    "PredRNN": "#59A14F",
    "Earthformer": "#9C755F",
    "NowCastNet": "#E6B800",
    "SimVP": "#76B7B2",
    "DiffCast": "#B07AA1",
    "StormDiT": "#E15759",
    "Fenglei": "#EDC948",
    "Persistence": "#6B7280",
}
METRIC_LABELS = {
    "csi": "CSI",
    "pod": "POD",
    "far": "FAR",
    "bias_score": "Frequency bias → 1",
    "mae": "MAE (dBZ) ↓",
    "rmse": "RMSE (dBZ) ↓",
    "fss": "FSS",
}


def _publication_fonts(fig):
    """Keep text legible in exported multi-panel figures without global rc changes."""
    for axis in fig.axes:
        axis.tick_params(labelsize=13)
        axis.xaxis.label.set_size(15)
        axis.yaxis.label.set_size(15)
        axis.title.set_size(16)
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(14)
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(18)
    return fig


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
    fig.suptitle(f"Multi-model CREF benchmark | {METRIC_LABELS[metric]} by forecast lead")
    return _publication_fonts(fig)


def plot_benchmark_fss_curves(summary, thresholds, windows=(1, 3, 5)):
    thresholds = (
        [float(thresholds)]
        if np.isscalar(thresholds)
        else [float(value) for value in thresholds]
    )
    if not thresholds:
        raise ValueError("select at least one FSS threshold")
    windows = tuple(windows)
    fig, axes = plt.subplots(
        len(thresholds),
        len(windows),
        figsize=(5.7 * len(windows), 4.3 * len(thresholds)),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for row_index, threshold in enumerate(thresholds):
        selected = summary[np.isclose(summary["threshold_dbz"], threshold)]
        if selected.empty:
            raise ValueError(f"benchmark summary has no FSS threshold {threshold:g} dBZ")
        for column_index, window in enumerate(windows):
            axis = axes[row_index, column_index]
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
                title=f"{threshold:g} dBZ | {window}×{window} pixels",
                xlabel="Lead time (min)",
                ylabel="FSS",
                ylim=(0, 1),
            )
            style_axes(axis)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle("Multi-model CREF benchmark | Multi-threshold, multi-scale FSS")
    return _publication_fonts(fig)


def plot_benchmark_pod_distribution(per_sample, threshold):
    """One threshold per figure; boxes use individual matched sample/lead PODs."""
    selected = per_sample[np.isclose(per_sample["threshold_dbz"], threshold)]
    models = [model for model in MODEL_COLORS if model in set(selected["model"])]
    if not models:
        raise ValueError(f"no POD samples for {threshold:g} dBZ")
    fig, axis = plt.subplots(figsize=(8, 6.5), layout="constrained")
    rng = np.random.default_rng(0)
    labels = []
    for index, model in enumerate(models, start=1):
        values = selected.loc[selected["model"] == model, "pod"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        labels.append(f"{model}\n(n={len(values)})")
        if not len(values):
            continue
        color = MODEL_COLORS[model]
        axis.boxplot(
            [values], positions=[index], widths=0.5, patch_artist=True,
            showfliers=False, showmeans=True,
            boxprops={"facecolor": color, "alpha": 0.4},
            medianprops={"color": "#222222", "linewidth": 1.5},
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#222222"},
        )
        # Box statistics use ALL valid samples. Thin points only for readability.
        points = values if len(values) <= 400 else rng.choice(values, 400, replace=False)
        axis.scatter(index + rng.uniform(-0.16, 0.16, len(points)), points,
                     color=color, s=12, alpha=0.5, edgecolors="none")
    axis.set_xticks(range(1, len(models) + 1), labels)
    axis.set(xlim=(0.5, len(models) + 0.5), ylim=(0, 1), ylabel="POD", xlabel="Model",
             title=f"POD distribution | {threshold:g} dBZ")
    style_axes(axis)
    return _publication_fonts(fig)


def plot_benchmark_far_bars(summary, thresholds):
    """Grouped bars by native lead using pooled FAR, never averaged sample ratios."""
    thresholds = list(thresholds)
    if not thresholds:
        raise ValueError("select at least one FAR threshold")
    columns = min(2, len(thresholds))
    fig, axes = plt.subplots(math.ceil(len(thresholds) / columns), columns,
                             figsize=(8 * columns, 5.5 * math.ceil(len(thresholds) / columns)),
                             squeeze=False, layout="constrained")
    for axis, threshold in zip(axes.flat, thresholds):
        selected = summary[np.isclose(summary["threshold_dbz"], threshold)]
        models = [model for model in MODEL_COLORS if model in set(selected["model"])]
        leads = sorted(selected["lead_minutes"].unique())
        if not models or not leads:
            plt.close(fig)
            raise ValueError(f"no FAR scores for {threshold:g} dBZ")
        width = 0.8 / len(models)
        for index, model in enumerate(models):
            values = selected[selected["model"] == model].set_index("lead_minutes").reindex(leads)
            axis.bar(np.arange(len(leads)) + (index - (len(models) - 1) / 2) * width,
                     values["far"], width=width, color=MODEL_COLORS[model], label=model)
        axis.set_xticks(range(len(leads)), [str(value) for value in leads])
        axis.set(xlabel="Lead time (min)", ylabel="FAR", ylim=(0, 1),
                 title=f"{threshold:g} dBZ")
        style_axes(axis)
    for axis in axes.flat[len(thresholds):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle("Multi-model CREF benchmark | FAR by forecast lead")
    return _publication_fonts(fig)


def plot_benchmark_period_summary(
    period: pd.DataFrame,
    metrics,
    threshold: float,
    *,
    fss_window=5,
    fss_windows=None,
):
    selected = period[np.isclose(period["threshold_dbz"], float(threshold))].copy()
    if selected.empty:
        raise ValueError("benchmark period summary is empty")
    windows = tuple(dict.fromkeys(fss_windows or (fss_window,)))
    plot_specs = []
    for metric in dict.fromkeys(metrics):
        if metric == "fss":
            plot_specs.extend(
                (f"fss_w{window}", f"FSS {window}×{window} pixels ↑", "fss")
                for window in windows
            )
        else:
            plot_specs.append((metric, METRIC_LABELS[metric], metric))
    plot_specs = plot_specs[:8]
    columns = 2 if len(plot_specs) > 1 else 1
    rows = math.ceil(len(plot_specs) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.2 * columns, 4.6 * rows),
        squeeze=False,
        layout="constrained",
    )
    order = [model for model in MODEL_COLORS if model in set(selected["model"])]
    selected = selected.set_index("model").reindex(order)
    for axis, (column, label, metric) in zip(axes.flat, plot_specs):
        axis.bar(
            np.arange(len(order)),
            selected[column],
            color=[MODEL_COLORS[model] for model in order],
            width=0.7,
        )
        axis.set_xticks(np.arange(len(order)), order, rotation=32, ha="right")
        axis.set_ylabel(label)
        axis.set_title(label)
        _score_limits(axis, metric)
        style_axes(axis)
    for axis in axes.flat[len(plot_specs) :]:
        axis.set_visible(False)
    fig.suptitle(
        f"Pooled common-window comparison | {threshold:g} dBZ | "
        f"FSS scales {', '.join(f'{window}×{window}' for window in windows)} pixels"
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
    limits = {"csi": (0.0, 1.0), "pod": (0.0, 1.0), "far": (0.0, 1.0), "fss": (0.0, 1.0)}
    vmin, vmax = limits.get(metric, (None, None))
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
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
        f"Multi-model spatial comparison | Init {sample.analysis_time_bjt:%Y-%m-%d %H:%M} BJT | "
        f"+{frame['lead_minutes']} min"
    )
    return fig


def render_benchmark_reflectivity_animation(frames, *, fps=2):
    """Encode ordered benchmark spatial comparisons as a deterministic GIF."""
    frames = list(frames)
    if not frames:
        raise ValueError("benchmark animation requires at least one frame")
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 4:
        raise ValueError("fps must be an integer from 1 to 4")
    leads = [int(frame["lead_minutes"]) for frame in frames]
    if leads != sorted(set(leads)):
        raise ValueError("benchmark animation frames must use unique increasing leads")
    panel_order = list(frames[0]["forecasts"])
    if any(list(frame["forecasts"]) != panel_order for frame in frames):
        raise ValueError("benchmark animation model panels must keep a fixed order")
    return _figures_to_gif(
        (plot_benchmark_spatial_comparison(frame) for frame in frames),
        fps=fps,
    )

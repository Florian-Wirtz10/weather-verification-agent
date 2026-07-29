"""Figures for the Phase 8 StormDiT/Fenglei comparison."""

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_io.forecast_sources import MODEL_STORMDIT
from verification.metrics import CATEGORICAL_SCORES
from visualization.geography import add_admin_boundaries
from visualization.plots import (
    CATEGORICAL_COLORS,
    _figures_to_gif,
    _reflectivity_cmap_norm,
    _reflectivity_display_data,
)

_MODEL_STYLES = {
    "stormdit": {"linestyle": "-", "linewidth": 2.3, "marker": "o"},
    "fenglei": {"linestyle": "--", "linewidth": 1.9, "marker": None},
}
_PLOT_LABELS = {"stormdit": "StormDiT", "fenglei": "Fenglei"}


def _style_score_axis(ax, metric, leads):
    ax.set(
        xlabel="Lead time (min)",
        ylabel=metric.upper(),
        xlim=(min(leads), max(leads)),
    )
    if metric in {"csi", "pod", "far"}:
        ax.set_ylim(0.0, 1.0)
    elif metric == "ets":
        ax.set_ylim(-0.2, 1.0)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_multimodel_score_curves(result, metric: str):
    """Plot selected products with common threshold colours and model line styles."""
    if metric not in CATEGORICAL_SCORES:
        raise ValueError(f"unsupported comparison metric: {metric!r}")
    entries = result["per_lead"]
    if not entries:
        raise ValueError("comparison contains no matched leads")
    model_ids = result["context"]["model_ids"]
    thresholds = sorted(entries[0]["models"][model_ids[0]])
    leads = [entry["lead_minutes"] for entry in entries]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for threshold_index, threshold in enumerate(thresholds):
        color = CATEGORICAL_COLORS[threshold_index % len(CATEGORICAL_COLORS)]
        for model_id in model_ids:
            style = _MODEL_STYLES[model_id]
            ax.plot(
                leads,
                [
                    entry["models"][model_id][threshold][metric]
                    for entry in entries
                ],
                color=color,
                markersize=3.2,
                **style,
            )
    model_legend = ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#333333",
                label=_PLOT_LABELS[model_id],
                **_MODEL_STYLES[model_id],
            )
            for model_id in model_ids
        ],
        title="Forecast product",
        loc="upper right",
    )
    ax.add_artist(model_legend)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=CATEGORICAL_COLORS[index % len(CATEGORICAL_COLORS)],
                linewidth=2.2,
                label=f"{threshold:g} dBZ",
            )
            for index, threshold in enumerate(thresholds)
        ],
        title="Threshold",
        loc="lower left",
    )
    ax.set_title(f"{metric.upper()} comparison by forecast lead")
    _style_score_axis(ax, metric, leads)
    fig.tight_layout()
    return fig


def plot_triplet_score_curves(result, metric: str):
    """Backward-compatible adapter for the first Phase 8 result structure."""
    converted = {
        "context": {"model_ids": ["stormdit", "fenglei"]},
        "per_lead": [
            {
                "lead_minutes": entry["lead_minutes"],
                "models": {
                    "stormdit": entry["model"],
                    "fenglei": entry["fenglei"],
                },
            }
            for entry in result["per_lead"]
        ],
    }
    return plot_multimodel_score_curves(converted, metric)


def plot_multimodel_reflectivity_frame(frame, *, title=None):
    """Render one or two forecasts beside the exact CREF observation."""
    model_ids = list(frame["forecasts"])
    fields = [frame["forecasts"][model_id] for model_id in model_ids]
    observation = frame["observation"]
    extent = (
        float(observation.lon.min()),
        float(observation.lon.max()),
        float(observation.lat.min()),
        float(observation.lat.max()),
    )
    cmap, norm = _reflectivity_cmap_norm()
    panels = [
        (field, _PLOT_LABELS[model_id])
        for model_id, field in zip(model_ids, fields)
    ] + [(observation, "CREF observation")]
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.2 * len(panels), 5.0),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    image = None
    for axis, (field, label) in zip(axes, panels):
        image = axis.imshow(
            _reflectivity_display_data(field.data),
            cmap=cmap,
            norm=norm,
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        axis.set_title(label)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        add_admin_boundaries(axis, extent)
    if image is not None:
        fig.colorbar(
            image,
            ax=list(axes),
            label="Reflectivity (dBZ; display floor 0)",
            shrink=0.88,
        )
    fig.suptitle(
        title
        or (
            f"Forecast–observation comparison | "
            f"+{frame['lead_minutes']} min | {observation.valid_time:%Y-%m-%d %H:%M} UTC"
        )
    )
    return fig


def render_multimodel_reflectivity_animation(frames, *, fps=2):
    if not frames:
        raise ValueError("animation requires at least one frame")
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 4:
        raise ValueError("fps must be an integer from 1 to 4")
    return _figures_to_gif(
        (plot_multimodel_reflectivity_frame(frame) for frame in frames),
        fps=fps,
    )


def plot_multimodel_batch_score_curves(summary: pd.DataFrame, metric: str):
    if metric not in CATEGORICAL_SCORES:
        raise ValueError(f"unsupported comparison metric: {metric!r}")
    if summary.empty:
        raise ValueError("batch summary is empty")
    model_ids = list(dict.fromkeys(summary["model_id"]))
    thresholds = sorted(summary["threshold_dbz"].unique())
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for threshold_index, threshold in enumerate(thresholds):
        color = CATEGORICAL_COLORS[threshold_index % len(CATEGORICAL_COLORS)]
        for model_id in model_ids:
            subset = pd.DataFrame(
                summary[
                    (summary["model_id"] == model_id)
                    & (summary["threshold_dbz"] == threshold)
                ]
            ).sort_values(by="lead_minutes")
            if subset.empty:
                continue
            ax.plot(
                subset["lead_minutes"],
                subset[metric],
                color=color,
                markersize=3.0,
                label=None,
                **_MODEL_STYLES[model_id],
            )
    model_legend = ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#333333",
                label=_PLOT_LABELS[model_id],
                **_MODEL_STYLES[model_id],
            )
            for model_id in model_ids
        ],
        title="Forecast product",
        loc="upper right",
    )
    ax.add_artist(model_legend)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=CATEGORICAL_COLORS[index % len(CATEGORICAL_COLORS)],
                linewidth=2.2,
                label=f"{threshold:g} dBZ",
            )
            for index, threshold in enumerate(thresholds)
        ],
        title="Threshold",
        loc="lower left",
    )
    leads = sorted(summary["lead_minutes"].unique())
    ax.set_title(f"Pooled {metric.upper()} by forecast lead")
    _style_score_axis(ax, metric, leads)
    fig.tight_layout()
    return fig


def plot_multimodel_score_heatmaps(
    per_case: pd.DataFrame,
    metric: str,
    threshold: float,
):
    """Render one heatmap per model plus StormDiT-minus-Fenglei when available."""
    subset = per_case[per_case["threshold_dbz"] == float(threshold)].copy()
    if subset.empty:
        raise ValueError("no rows for the selected diagnostic threshold")
    subset["init_time_bjt"] = pd.to_datetime(subset["init_time_bjt"])
    model_ids = list(dict.fromkeys(subset["model_id"]))
    pivots = {
        model_id: pd.DataFrame(
            subset[subset["model_id"] == model_id]
        ).pivot(index="init_time_bjt", columns="lead_minutes", values=metric)
        for model_id in model_ids
    }
    include_difference = {"stormdit", "fenglei"} <= set(model_ids)
    panel_count = len(model_ids) + int(include_difference)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(6.2 * panel_count, 5.8),
        squeeze=False,
        layout="constrained",
    )
    axes = axes[0]
    score_min = -0.2 if metric == "ets" else 0.0
    score_max = 1.0
    for axis, model_id in zip(axes, model_ids):
        pivot = pivots[model_id]
        image = axis.imshow(
            pivot.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=score_min,
            vmax=score_max,
        )
        axis.set_title(_PLOT_LABELS[model_id])
        axis.set_xlabel("Lead time (min)")
        axis.set_ylabel("Initialization (BJT)")
        axis.set_xticks(range(len(pivot.columns))[:: max(1, len(pivot.columns) // 6)])
        axis.set_xticklabels(pivot.columns[:: max(1, len(pivot.columns) // 6)])
        labels = [value.strftime("%m-%d %H:%M") for value in pivot.index]
        step = max(1, len(labels) // 8)
        axis.set_yticks(range(len(labels))[::step])
        axis.set_yticklabels(labels[::step])
        axis.grid(False)
        fig.colorbar(image, ax=axis, label=metric.upper(), shrink=0.82)
    if include_difference:
        common_index = pivots["stormdit"].index.intersection(pivots["fenglei"].index)
        common_columns = pivots["stormdit"].columns.intersection(pivots["fenglei"].columns)
        difference = (
            pivots["stormdit"].loc[common_index, common_columns]
            - pivots["fenglei"].loc[common_index, common_columns]
        )
        limit = float(np.nanmax(np.abs(difference.to_numpy()))) if difference.size else 1.0
        limit = limit or 1.0
        axis = axes[-1]
        image = axis.imshow(
            difference.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title("StormDiT − Fenglei")
        axis.set_xlabel("Lead time (min)")
        axis.set_ylabel("Initialization (BJT)")
        axis.set_xticks(
            range(len(common_columns))[:: max(1, len(common_columns) // 6)]
        )
        axis.set_xticklabels(
            common_columns[:: max(1, len(common_columns) // 6)]
        )
        labels = [value.strftime("%m-%d %H:%M") for value in common_index]
        step = max(1, len(labels) // 8)
        axis.set_yticks(range(len(labels))[::step])
        axis.set_yticklabels(labels[::step])
        axis.grid(False)
        fig.colorbar(image, ax=axis, label=f"Δ{metric.upper()}", shrink=0.82)
    fig.suptitle(f"{metric.upper()} at {threshold:g} dBZ")
    return fig


def plot_multimodel_score_distribution(
    per_case: pd.DataFrame,
    metric: str,
    threshold: float,
):
    subset = per_case[per_case["threshold_dbz"] == float(threshold)]
    model_ids = list(dict.fromkeys(subset["model_id"]))
    values = [
        subset.loc[subset["model_id"] == model_id, metric].dropna().to_numpy()
        for model_id in model_ids
    ]
    if not any(len(value) for value in values):
        raise ValueError("no finite scores for distribution plot")
    fig, ax = plt.subplots(figsize=(8, 5))
    boxes = ax.boxplot(
        values,
        tick_labels=[_PLOT_LABELS[value] for value in model_ids],
        patch_artist=True,
        showmeans=True,
    )
    for patch, model_id in zip(boxes["boxes"], model_ids):
        patch.set_facecolor("#2a78d6" if model_id == MODEL_STORMDIT else "#e34948")
        patch.set_alpha(0.55)
    ax.set_title(f"{metric.upper()} distribution at {threshold:g} dBZ")
    ax.set_ylabel(metric.upper())
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

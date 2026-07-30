"""Figures for the Phase 8 StormDiT/Fenglei comparison."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from verification.metrics import CATEGORICAL_SCORES
from visualization.geography import add_admin_boundaries
from visualization.plots import (
    _figures_to_gif,
    _reflectivity_cmap_norm,
    _reflectivity_display_data,
)
from visualization.style import (
    METRIC_COLORS,
    METRIC_LABELS,
    MODEL_LABELS as PLOT_LABELS,
    MODEL_STYLES,
    model_legend_handles,
    style_axes,
    threshold_color,
    threshold_legend_handles,
)


def _ordered_model_ids(values):
    present = list(dict.fromkeys(str(value) for value in values))
    return [
        model_id for model_id in PLOT_LABELS if model_id in present
    ] + [model_id for model_id in present if model_id not in PLOT_LABELS]


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
    style_axes(ax)


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
        color = threshold_color(threshold, threshold_index)
        for model_id in model_ids:
            style = MODEL_STYLES[model_id]
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
        handles=model_legend_handles(model_ids),
        title="Forecast product",
        loc="upper right",
    )
    ax.add_artist(model_legend)
    ax.legend(
        handles=threshold_legend_handles(thresholds),
        title="Threshold",
        loc="lower left",
    )
    ax.set_title(f"{metric.upper()} comparison by forecast lead")
    _style_score_axis(ax, metric, leads)
    fig.tight_layout()
    return fig


def _combined_metric_names(metrics):
    selected = [
        str(metric).lower()
        for metric in dict.fromkeys(metrics)
        if str(metric).lower() in METRIC_COLORS
    ]
    if len(selected) < 2:
        raise ValueError("multi-metric curve requires at least two of CSI, POD and FAR")
    return selected


def _add_combined_metric_legends(axis, model_ids, metrics):
    model_legend = axis.legend(
        handles=model_legend_handles(model_ids),
        title="Forecast product",
        loc="upper right",
        frameon=False,
    )
    axis.add_artist(model_legend)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=METRIC_COLORS[metric],
                linewidth=2.2,
                label=METRIC_LABELS[metric],
            )
            for metric in metrics
        ],
        title="Metric",
        loc="lower left",
        frameon=False,
    )


def plot_multimodel_multi_metric_curves(result, metrics, threshold: float):
    """Compare CSI/POD/FAR in one panel at one fixed threshold."""
    metrics = _combined_metric_names(metrics)
    entries = result["per_lead"]
    if not entries:
        raise ValueError("comparison contains no matched leads")
    model_ids = result["context"]["model_ids"]
    threshold = float(threshold)
    leads = [entry["lead_minutes"] for entry in entries]
    fig, axis = plt.subplots(figsize=(10.5, 5.9))
    for metric in metrics:
        for model_id in model_ids:
            values = [
                entry["models"][model_id][threshold][metric]
                for entry in entries
            ]
            axis.plot(
                leads,
                values,
                color=METRIC_COLORS[metric],
                markersize=3.2,
                **MODEL_STYLES[model_id],
            )
    axis.set(
        xlabel="Lead time (min)",
        ylabel="Categorical score",
        xlim=(min(leads), max(leads)),
        ylim=(0.0, 1.0),
        title=f"Multi-metric comparison at {threshold:g} dBZ",
    )
    style_axes(axis)
    _add_combined_metric_legends(axis, model_ids, metrics)
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
        (field, PLOT_LABELS[model_id])
        for model_id, field in zip(model_ids, fields)
    ] + [(observation, "CREF observation")]
    fig, axes = plt.subplots(
        1,
        len(panels),
        # At GIF encoding's 100 dpi, three panels are exactly 1440 px wide.
        # This stays below Streamlit's 1460 px media-processing threshold;
        # crossing that threshold makes Pillow resize only the first GIF frame.
        figsize=(4.8 * len(panels), 5.0),
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


def plot_multimodel_same_valid_time_comparison(comparison, model_id: str):
    """Compare forecasts from different initializations at one valid time."""
    if model_id not in comparison["model_ids"]:
        raise ValueError(f"model is absent from same-valid-time comparison: {model_id}")
    items = list(comparison["items"])
    if not items:
        raise ValueError("same-valid-time comparison contains no initializations")
    observation = items[0]["frame"]["observation"]
    extent = (
        float(observation.lon.min()),
        float(observation.lon.max()),
        float(observation.lat.min()),
        float(observation.lat.max()),
    )
    panels = [
        (
            item["frame"]["forecasts"][model_id],
            f"Init {item['init_time_bjt']:%m-%d %H:%M} BJT\n"
            f"Lead +{item['lead_minutes']} min",
        )
        for item in items
    ]
    panels.append(
        (
            observation,
            f"CREF observation\n{comparison['valid_time_bjt']:%m-%d %H:%M} BJT",
        )
    )
    if len(panels) <= 3:
        column_count = len(panels)
    elif len(panels) == 4:
        column_count = 2
    elif len(panels) <= 6 or len(panels) == 9:
        column_count = 3
    else:
        column_count = 4
    row_count = int(np.ceil(len(panels) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.0 * column_count, 4.4 * row_count),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    axes_flat = list(axes.flat)
    cmap, norm = _reflectivity_cmap_norm()
    image = None
    for axis, (field, title) in zip(axes_flat, panels):
        image = axis.imshow(
            _reflectivity_display_data(field.data),
            cmap=cmap,
            norm=norm,
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        add_admin_boundaries(axis, extent)
    for axis in axes_flat[len(panels) :]:
        axis.set_visible(False)
    if image is not None:
        fig.colorbar(
            image,
            ax=[axis for axis in axes_flat if axis.get_visible()],
            label="Reflectivity (dBZ; display floor 0)",
            shrink=0.86,
        )
    fig.suptitle(
        f"{PLOT_LABELS[model_id]} forecasts from different initializations "
        f"at one valid time | {comparison['valid_time_bjt']:%Y-%m-%d %H:%M} BJT",
        fontsize=14,
        fontweight="bold",
    )
    return fig


def plot_multimodel_batch_score_curves(summary: pd.DataFrame, metric: str):
    if metric not in CATEGORICAL_SCORES:
        raise ValueError(f"unsupported comparison metric: {metric!r}")
    if summary.empty:
        raise ValueError("batch summary is empty")
    model_ids = _ordered_model_ids(summary["model_id"])
    thresholds = sorted(summary["threshold_dbz"].unique())
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for threshold_index, threshold in enumerate(thresholds):
        color = threshold_color(threshold, threshold_index)
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
                **MODEL_STYLES[model_id],
            )
    model_legend = ax.legend(
        handles=model_legend_handles(model_ids),
        title="Forecast product",
        loc="upper right",
    )
    ax.add_artist(model_legend)
    ax.legend(
        handles=threshold_legend_handles(thresholds),
        title="Threshold",
        loc="lower left",
    )
    leads = sorted(summary["lead_minutes"].unique())
    ax.set_title(f"Pooled {metric.upper()} by forecast lead")
    _style_score_axis(ax, metric, leads)
    fig.tight_layout()
    return fig


def plot_multimodel_batch_multi_metric_curves(
    summary: pd.DataFrame,
    metrics,
    threshold: float,
):
    """Compare pooled CSI/POD/FAR curves in one panel at one threshold."""
    if summary.empty:
        raise ValueError("batch summary is empty")
    metrics = _combined_metric_names(metrics)
    threshold = float(threshold)
    selected = summary[np.isclose(summary["threshold_dbz"], threshold)]
    if selected.empty:
        raise ValueError("diagnostic threshold is absent from batch summary")
    model_ids = _ordered_model_ids(selected["model_id"])
    leads = sorted(int(value) for value in selected["lead_minutes"].unique())
    fig, axis = plt.subplots(figsize=(10.5, 5.9))
    for metric in metrics:
        for model_id in model_ids:
            rows = pd.DataFrame(
                selected[selected["model_id"] == model_id]
            ).sort_values("lead_minutes")
            axis.plot(
                rows["lead_minutes"],
                rows[metric],
                color=METRIC_COLORS[metric],
                markersize=3.0,
                **MODEL_STYLES[model_id],
            )
    axis.set(
        xlabel="Lead time (min)",
        ylabel="Pooled categorical score",
        xlim=(min(leads), max(leads)),
        ylim=(0.0, 1.0),
        title=f"Pooled multi-metric comparison at {threshold:g} dBZ",
    )
    style_axes(axis)
    _add_combined_metric_legends(axis, model_ids, metrics)
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
    model_ids = _ordered_model_ids(subset["model_id"])
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
    score_image = None
    for axis, model_id in zip(axes, model_ids):
        pivot = pivots[model_id]
        score_image = axis.imshow(
            pivot.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=score_min,
            vmax=score_max,
        )
        axis.set_title(PLOT_LABELS[model_id])
        axis.set_xlabel("Lead time (min)")
        axis.set_ylabel("Initialization (BJT)")
        axis.set_xticks(range(len(pivot.columns))[:: max(1, len(pivot.columns) // 6)])
        axis.set_xticklabels(pivot.columns[:: max(1, len(pivot.columns) // 6)])
        labels = [value.strftime("%m-%d %H:%M") for value in pivot.index]
        step = max(1, len(labels) // 8)
        axis.set_yticks(range(len(labels))[::step])
        axis.set_yticklabels(labels[::step])
        axis.grid(False)
    if score_image is not None:
        fig.colorbar(
            score_image,
            ax=list(axes[: len(model_ids)]),
            label=metric.upper(),
            shrink=0.82,
        )
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
    fig.suptitle(
        f"Initialization–lead {metric.upper()} score matrix at {threshold:g} dBZ"
    )
    return fig


def plot_multimodel_score_distribution(
    per_case: pd.DataFrame,
    metric: str,
    thresholds=None,
):
    """Show raw samples and robust summaries for multiple thresholds by model."""
    if thresholds is None:
        thresholds = sorted(float(value) for value in per_case["threshold_dbz"].unique())
    elif np.isscalar(thresholds):
        thresholds = [float(thresholds)]
    else:
        thresholds = list(dict.fromkeys(float(value) for value in thresholds))
    subset = per_case[per_case["threshold_dbz"].isin(thresholds)]
    model_ids = _ordered_model_ids(subset["model_id"])
    if not model_ids or metric not in subset:
        raise ValueError("no finite scores for distribution plot")
    if not np.isfinite(subset[metric].to_numpy(dtype=float)).any():
        raise ValueError("no finite scores for distribution plot")
    fig, axes = plt.subplots(
        1,
        len(model_ids),
        figsize=(6.2 * len(model_ids), 5.6),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    rng = np.random.default_rng(20260729)
    for axis, model_id in zip(axes[0], model_ids):
        model_rows = subset[subset["model_id"] == model_id]
        values = [
            model_rows.loc[
                np.isclose(model_rows["threshold_dbz"], threshold), metric
            ].dropna().to_numpy(dtype=float)
            for threshold in thresholds
        ]
        boxes = axis.boxplot(
            values,
            positions=np.arange(len(thresholds)),
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            medianprops={"color": "#202124", "linewidth": 1.4},
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#202124",
                "markersize": 4,
            },
        )
        for index, (patch, sample) in enumerate(zip(boxes["boxes"], values)):
            color = threshold_color(thresholds[index], index)
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.35)
            shown = sample
            if len(shown) > 180:
                shown = rng.choice(shown, size=180, replace=False)
            jitter = rng.uniform(-0.14, 0.14, size=len(shown))
            axis.scatter(
                index + jitter,
                shown,
                s=10,
                facecolor=color,
                edgecolor="none",
                alpha=0.42,
                rasterized=True,
            )
        axis.set_xticks(
            range(len(thresholds)),
            [
                f"{threshold:g}\n(n={len(values[index])})"
                for index, threshold in enumerate(thresholds)
            ],
        )
        axis.set_xlabel("Threshold (dBZ)")
        axis.set_title(PLOT_LABELS[model_id])
        style_axes(axis)
    axes[0][0].set_ylabel(metric.upper())
    if metric in {"csi", "pod", "far"}:
        axes[0][0].set_ylim(0.0, 1.0)
    fig.suptitle(
        f"{metric.upper()} distribution across thresholds\n"
        "points: matched initialization/lead samples; box: IQR and median"
    )
    return fig


def _with_rate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "valid_count",
        "hits",
        "misses",
        "false_alarms",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"rate plot is missing columns: {sorted(missing)}")
    result = frame[frame["valid_count"] > 0].copy()
    valid = result["valid_count"].to_numpy(dtype=float)
    result["observed_rate"] = (
        result["hits"].to_numpy(dtype=float)
        + result["misses"].to_numpy(dtype=float)
    ) / valid
    result["forecast_rate"] = (
        result["hits"].to_numpy(dtype=float)
        + result["false_alarms"].to_numpy(dtype=float)
    ) / valid
    return result


def plot_multimodel_forecast_observed_rate_scatter(
    per_case: pd.DataFrame,
    thresholds,
):
    """Render an uncluttered raw rate scatter with one panel per product."""
    thresholds = list(dict.fromkeys(float(value) for value in thresholds))
    selected = _with_rate_columns(
        per_case[per_case["threshold_dbz"].isin(thresholds)]
    )
    if selected.empty:
        raise ValueError("no valid forecast/observed rate samples")
    model_ids = _ordered_model_ids(selected["model_id"])
    fig, axes = plt.subplots(
        1,
        len(model_ids),
        figsize=(6.2 * len(model_ids), 5.7),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for axis, model_id in zip(axes[0], model_ids):
        model_rows = selected[selected["model_id"] == model_id]
        sample_keys = [
            column
            for column in ("init_time_utc", "lead_minutes")
            if column in model_rows
        ]
        sample_count = (
            len(model_rows[sample_keys].drop_duplicates())
            if sample_keys
            else len(model_rows)
        )
        for index, threshold in enumerate(thresholds):
            rows = model_rows[np.isclose(model_rows["threshold_dbz"], threshold)]
            if rows.empty:
                continue
            axis.scatter(
                rows["observed_rate"],
                rows["forecast_rate"],
                s=16,
                facecolor=threshold_color(threshold, index),
                edgecolor="#30343b",
                linewidth=0.25,
                alpha=0.52,
                rasterized=True,
                label=f"{threshold:g} dBZ",
            )
        axis.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1.1)
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Observed rate",
            title=f"{PLOT_LABELS[model_id]} (n={sample_count} matched pairs)",
        )
        style_axes(axis)
    axes[0][0].set_ylabel("Forecast rate")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Threshold",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.055),
        ncols=max(1, len(thresholds)),
        frameon=False,
    )
    fig.suptitle("Forecast rate versus observed rate")
    return fig


_CORRELATION_VARIABLES = (
    ("lead_minutes", "Lead"),
    ("forecast_rate", "Forecast rate"),
    ("observed_rate", "Observed rate"),
    ("forecast_mean_dbz", "Forecast mean"),
    ("observation_mean_dbz", "Observed mean"),
    ("mean_reflectivity_error_dbz", "Mean error"),
    ("reflectivity_mae_dbz", "MAE"),
    ("centroid_distance_km", "Centroid distance"),
    ("csi", "CSI"),
    ("pod", "POD"),
    ("far", "FAR"),
    ("bias_score", "Bias"),
)


def plot_multimodel_variable_correlation_heatmap(
    per_case: pd.DataFrame,
    threshold: float,
    *,
    method="spearman",
):
    """Plot descriptive physical/score correlations for each forecast product."""
    method = str(method).lower()
    if method not in {"spearman", "pearson"}:
        raise ValueError("correlation method must be spearman or pearson")
    selected = _with_rate_columns(
        per_case[np.isclose(per_case["threshold_dbz"], float(threshold))]
    )
    available = [
        (column, label)
        for column, label in _CORRELATION_VARIABLES
        if column in selected.columns
        and selected[column].notna().sum() >= 3
        and selected[column].nunique(dropna=True) > 1
    ]
    if len(available) < 3:
        raise ValueError(
            "physical diagnostics are unavailable; rebuild the Phase 8 metric cache"
        )
    columns = [column for column, _label in available]
    labels = [label for _column, label in available]
    model_ids = _ordered_model_ids(selected["model_id"])
    matrices = {}
    for model_id in model_ids:
        rows = selected[selected["model_id"] == model_id]
        matrices[model_id] = rows[columns].corr(method=method, min_periods=3)
    fig, axes = plt.subplots(
        1,
        len(model_ids),
        figsize=(7.4 * len(model_ids), 7.0),
        squeeze=False,
        layout="constrained",
    )
    image = None
    for axis, model_id in zip(axes[0], model_ids):
        matrix = matrices[model_id].to_numpy(dtype=float)
        masked = np.ma.masked_invalid(matrix)
        image = axis.imshow(
            masked,
            cmap="RdBu_r",
            norm=Normalize(vmin=-1, vmax=1),
            aspect="equal",
        )
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(
            f"{PLOT_LABELS[model_id]} | "
            f"n={len(selected[selected['model_id'] == model_id])}"
        )
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = matrix[row, column]
                if np.isfinite(value):
                    axis.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.2,
                        color="white" if abs(value) >= 0.60 else "#202124",
                    )
        axis.grid(False)
    if image is not None:
        fig.colorbar(
            image,
            ax=list(axes[0]),
            shrink=0.78,
            label=f"{method.title()} correlation",
        )
    fig.suptitle(
        f"Variable correlation at {float(threshold):g} dBZ\n"
        "descriptive correlations over matched initialization/lead samples"
    )
    return fig

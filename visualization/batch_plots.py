"""Deterministic visualizations for full-day verification caches."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

_METRIC_LABELS = {
    "csi": "CSI",
    "ets": "ETS",
    "pod": "POD",
    "far": "FAR",
    "bias_score": "Frequency bias",
}


def _validate_metric(frame: pd.DataFrame, metric: str) -> str:
    metric = str(metric).lower()
    if metric not in _METRIC_LABELS or metric not in frame.columns:
        raise ValueError(f"unsupported daily score metric: {metric!r}")
    return metric


def plot_daily_score_curves(summary: pd.DataFrame, metric="csi", *, figure_number=1):
    """Plot pooled full-day score against lead time for every threshold."""
    metric = _validate_metric(summary, metric)
    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    for threshold, rows in summary.groupby("threshold_dbz", sort=True):
        rows = rows.sort_values("lead_minutes")
        ax.plot(
            rows["lead_minutes"],
            rows[metric],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"{threshold:g} dBZ",
        )
    ax.set_xlabel("Forecast lead time (min)")
    ax.set_ylabel(_METRIC_LABELS[metric])
    ax.set_title(
        f"Figure {figure_number}. Full-day pooled {_METRIC_LABELS[metric]} by lead time"
    )
    lead_values = np.unique(summary["lead_minutes"].to_numpy(dtype=int))
    ax.set_xticks(lead_values[::2])
    if metric in {"csi", "ets", "pod", "far"}:
        metric_values = summary[metric].to_numpy(dtype=float)
        finite_values = metric_values[np.isfinite(metric_values)]
        lower = (
            min(-0.05, float(np.min(finite_values)))
            if metric == "ets" and finite_values.size
            else (-0.05 if metric == "ets" else 0.0)
        )
        ax.set_ylim(lower, 1.0)
    ax.grid(True, alpha=0.28)
    ax.legend(title="Threshold", ncols=2)
    return fig


def plot_daily_score_heatmap(
    per_case: pd.DataFrame, metric="csi", threshold_dbz=15, *, figure_number=2
):
    """Plot score by initialization time and lead for one threshold."""
    metric = _validate_metric(per_case, metric)
    selected = per_case[np.isclose(per_case["threshold_dbz"], float(threshold_dbz))].copy()
    if selected.empty:
        raise ValueError(f"threshold {threshold_dbz:g} dBZ is not present in cached scores")
    selected["init_time_bjt"] = pd.to_datetime(selected["init_time_bjt"])
    pivot = selected.pivot_table(
        index="init_time_bjt",
        columns="lead_minutes",
        values=metric,
        aggfunc="mean",
    ).sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(12.5, 7.0), constrained_layout=True)
    values = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    vmin = min(-0.1, float(values.min())) if metric == "ets" and values.count() else 0.0
    vmax = 1.0 if metric != "bias_score" else max(2.0, float(values.max()))
    image = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    leads = list(pivot.columns)
    x_positions = np.arange(0, len(leads), max(1, len(leads) // 10))
    ax.set_xticks(x_positions, [str(leads[index]) for index in x_positions])
    times = list(pivot.index)
    y_positions = np.arange(0, len(times), max(1, len(times) // 12))
    ax.set_yticks(y_positions, [times[index].strftime("%H:%M") for index in y_positions])
    ax.set_xlabel("Forecast lead time (min)")
    ax.set_ylabel("Initialization time (BJT)")
    ax.set_title(
        f"Figure {figure_number}. {_METRIC_LABELS[metric]} heatmap at "
        f"{float(threshold_dbz):g} dBZ"
    )
    fig.colorbar(image, ax=ax, label=_METRIC_LABELS[metric], fraction=0.035, pad=0.02)
    return fig


def plot_daily_score_distribution(
    per_case: pd.DataFrame, metric="csi", threshold_dbz=15, *, figure_number=3
):
    """Plot the between-initialization score distribution at every lead."""
    metric = _validate_metric(per_case, metric)
    selected = per_case[np.isclose(per_case["threshold_dbz"], float(threshold_dbz))]
    if selected.empty:
        raise ValueError(f"threshold {threshold_dbz:g} dBZ is not present in cached scores")
    leads = sorted(int(value) for value in np.unique(np.asarray(selected["lead_minutes"], dtype=int)))
    distributions = [
        selected.loc[selected["lead_minutes"] == lead, metric].dropna().to_numpy(dtype=float)
        for lead in leads
    ]
    if not any(values.size for values in distributions):
        raise ValueError(f"{metric} is undefined for every selected daily sample")

    fig, ax = plt.subplots(figsize=(13.5, 6.2), constrained_layout=True)
    artists = ax.boxplot(
        distributions,
        positions=np.arange(len(leads)),
        widths=0.65,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#7f0000", "linewidth": 1.4},
    )
    for box in artists["boxes"]:
        box.set_facecolor("#9ecae1")
        box.set_edgecolor("#2171b5")
    tick_positions = np.arange(0, len(leads), max(1, len(leads) // 10))
    ax.set_xticks(tick_positions, [str(leads[index]) for index in tick_positions])
    ax.set_xlabel("Forecast lead time (min)")
    ax.set_ylabel(_METRIC_LABELS[metric])
    ax.set_title(
        f"Figure {figure_number}. {_METRIC_LABELS[metric]} distribution across initializations "
        f"at {float(threshold_dbz):g} dBZ"
    )
    if metric in {"csi", "pod", "far"}:
        ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.28)
    return fig

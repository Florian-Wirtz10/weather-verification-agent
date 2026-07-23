"""Deterministic visualizations for full-day verification caches."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

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
    ax.grid(False)
    ax.legend(title="Threshold", ncols=2)
    return fig


def _smooth_density(values, *, bins=80, sigma_bins=2.0):
    """Return a small dependency-free smoothed histogram on the 0—1 rate range."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    histogram, edges = np.histogram(values, bins=bins, range=(0, 1), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radius = max(2, int(round(4 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return centers, np.convolve(histogram, kernel, mode="same")


def plot_forecast_observed_rate_distribution(
    per_case: pd.DataFrame, threshold_dbz=15, *, figure_number=1
):
    """Plot a full-day rate scatter with forecast/observed marginal densities.

    Every point is one cached initialization/lead pair at one fixed threshold.
    Lead-time bands provide a meaningful grouping when geographic subregions
    have not been defined.  The chart is descriptive; it does not claim that
    overlapping six-minute initializations are statistically independent.
    """
    required = {
        "lead_minutes",
        "threshold_dbz",
        "hits",
        "misses",
        "false_alarms",
        "valid_count",
    }
    missing = required.difference(per_case.columns)
    if missing:
        raise ValueError(f"rate distribution is missing columns: {sorted(missing)}")
    selected = per_case[
        np.isclose(per_case["threshold_dbz"], float(threshold_dbz))
    ].copy()
    selected = selected[selected["valid_count"] > 0]
    if selected.empty:
        raise ValueError(
            f"threshold {float(threshold_dbz):g} dBZ has no valid rate samples"
        )
    valid = selected["valid_count"].to_numpy(dtype=float)
    selected["observed_rate"] = (
        selected["hits"].to_numpy(dtype=float)
        + selected["misses"].to_numpy(dtype=float)
    ) / valid
    selected["forecast_rate"] = (
        selected["hits"].to_numpy(dtype=float)
        + selected["false_alarms"].to_numpy(dtype=float)
    ) / valid

    bands = (
        ("6—60 min", 0, 60, "#12bde5"),
        ("66—120 min", 60, 120, "#ef8f83"),
        ("126—180 min", 120, 180, "#16bf65"),
    )
    fig = plt.figure(figsize=(10.5, 8.8))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(4.4, 1.15),
        height_ratios=(4.4, 1.15),
        left=0.09,
        right=0.95,
        bottom=0.11,
        top=0.87,
        wspace=0.12,
        hspace=0.12,
    )
    joint = fig.add_subplot(grid[0, 0])
    forecast_density = fig.add_subplot(grid[0, 1], sharey=joint)
    observed_density = fig.add_subplot(grid[1, 0], sharex=joint)
    legend_axis = fig.add_subplot(grid[1, 1])

    plotted_groups = 0
    for label, lower, upper, color in bands:
        in_band = (selected["lead_minutes"] > lower) & (
            selected["lead_minutes"] <= upper
        )
        rows = selected[in_band]
        if rows.empty:
            continue
        observed = rows["observed_rate"].to_numpy(dtype=float)
        forecast = rows["forecast_rate"].to_numpy(dtype=float)
        joint.scatter(
            observed,
            forecast,
            s=17,
            facecolor=color,
            edgecolor="#30343b",
            linewidth=0.3,
            alpha=0.42,
            rasterized=True,
            label=f"{label} (n={len(rows)})",
        )
        x, x_density = _smooth_density(observed)
        y, y_density = _smooth_density(forecast)
        observed_density.fill_between(x, 0, x_density, color=color, alpha=0.38)
        observed_density.plot(x, x_density, color=color, linewidth=1.25)
        forecast_density.fill_betweenx(y, 0, y_density, color=color, alpha=0.38)
        forecast_density.plot(y_density, y, color=color, linewidth=1.25)
        plotted_groups += 1
    if plotted_groups == 0:
        raise ValueError("no lead-time groups are available for the rate distribution")

    joint.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1.1)
    joint.set_xlim(0, 1)
    joint.set_ylim(0, 1)
    joint.set_xlabel("Observed rate")
    joint.set_ylabel("Forecast rate")
    joint.set_title(
        f"(a) {float(threshold_dbz):g} dBZ | {len(selected)} matched samples",
        loc="left",
    )
    joint.grid(False)
    joint.tick_params(direction="in", top=True, right=True)

    forecast_density.set_xlabel("Density")
    forecast_density.tick_params(axis="y", labelleft=False, labelright=True, direction="in")
    forecast_density.tick_params(axis="x", direction="in")
    forecast_density.grid(False)
    observed_density.set_xlabel("Observed rate")
    observed_density.set_ylabel("Density")
    observed_density.tick_params(direction="in")
    observed_density.grid(False)

    legend_axis.axis("off")
    handles, labels = joint.get_legend_handles_labels()
    legend_axis.legend(handles, labels, title="Forecast lead", loc="center", frameon=False)
    fig.suptitle(
        f"Figure {figure_number}. Forecast rate vs observed rate with marginal distributions",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.09,
        0.025,
        "Each point is one exactly matched initialization/lead sample; the dashed line is 1:1.",
        fontsize=9.2,
        color="#565d66",
    )
    return fig


def plot_daily_score_heatmap(
    per_case: pd.DataFrame, metric="csi", threshold_dbz=15, *, figure_number=2
):
    """Plot a 30-minute median score heatmap with two marginal summaries."""
    metric = _validate_metric(per_case, metric)
    selected = per_case[np.isclose(per_case["threshold_dbz"], float(threshold_dbz))].copy()
    if selected.empty:
        raise ValueError(f"threshold {threshold_dbz:g} dBZ is not present in cached scores")
    selected["init_time_bjt"] = pd.to_datetime(selected["init_time_bjt"])
    selected["init_time_bin_bjt"] = selected["init_time_bjt"].dt.floor("30min")
    aggregated = (
        selected.groupby(["init_time_bin_bjt", "lead_minutes"], as_index=False)[metric]
        .median()
    )
    pivot = aggregated.pivot_table(
        index="init_time_bin_bjt",
        columns="lead_minutes",
        values=metric,
        aggfunc="median",
    ).sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    values = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    if not values.count():
        raise ValueError(f"{metric} is undefined for every selected daily sample")
    raw_values = values.filled(np.nan)
    leads = np.asarray(pivot.columns, dtype=int)
    times = pd.DatetimeIndex(pivot.index)
    lead_median = np.nanmedian(raw_values, axis=0)
    lead_q25 = np.nanpercentile(raw_values, 25, axis=0)
    lead_q75 = np.nanpercentile(raw_values, 75, axis=0)
    time_median = np.nanmedian(raw_values, axis=1)
    cmap, norm, bounds = _daily_heatmap_scale(metric)

    fig = plt.figure(figsize=(14.8, 8.8))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(5.5, 1.25),
        height_ratios=(1.3, 5.3),
        left=0.08,
        right=0.92,
        bottom=0.10,
        top=0.90,
        wspace=0.08,
        hspace=0.08,
    )
    lead_axis = fig.add_subplot(grid[0, 0])
    heat_axis = fig.add_subplot(grid[1, 0], sharex=lead_axis)
    time_axis = fig.add_subplot(grid[1, 1], sharey=heat_axis)
    summary_axis = fig.add_subplot(grid[0, 1])

    image = heat_axis.imshow(
        values, aspect="auto", origin="lower", cmap=cmap, norm=norm
    )
    x_positions = np.arange(0, len(leads), max(1, len(leads) // 6))
    heat_axis.set_xticks(x_positions, [str(leads[index]) for index in x_positions])
    y_positions = np.arange(0, len(times), max(1, len(times) // 10))
    heat_axis.set_yticks(
        y_positions, [times[index].strftime("%H:%M") for index in y_positions]
    )
    heat_axis.set_xlabel("Forecast lead time (min)")
    heat_axis.set_ylabel("Initialization time (BJT, 30-minute median)")
    heat_axis.grid(False)

    x_values = np.arange(len(leads))
    lead_axis.fill_between(
        x_values, lead_q25, lead_q75, color="#8cbbe8", alpha=0.35, label="IQR"
    )
    lead_axis.plot(
        x_values,
        lead_median,
        color="#2a78d6",
        marker="o",
        markersize=3,
        linewidth=1.6,
        label="Median",
    )
    lead_axis.set_ylabel(_METRIC_LABELS[metric])
    lead_axis.set_ylim(*_daily_summary_limits(metric, lead_q25, lead_q75))
    lead_axis.grid(False)
    lead_axis.tick_params(axis="x", labelbottom=False)
    lead_axis.legend(frameon=False, ncols=2, loc="upper right")
    lead_axis.set_title(
        f"Figure {figure_number}. Lead-time summary across initialization bins",
        loc="left",
    )

    y_values = np.arange(len(times))
    time_axis.plot(time_median, y_values, color="#e34948", linewidth=1.7)
    time_axis.scatter(time_median, y_values, color="#e34948", s=12)
    time_axis.set_xlabel(f"Median {_METRIC_LABELS[metric]}")
    time_axis.set_xlim(*_daily_summary_limits(metric, time_median, time_median))
    time_axis.grid(False)
    time_axis.tick_params(axis="y", labelleft=False)
    time_axis.set_title("Time-window\nsummary", fontsize=10)

    best_index = int(np.nanargmax(raw_values))
    best_row, best_column = np.unravel_index(best_index, raw_values.shape)
    finite_values = raw_values[np.isfinite(raw_values)]
    summary_axis.axis("off")
    summary_axis.text(
        0.02,
        0.90,
        f"30-min bins: {len(times)}\n"
        f"Lead times: {len(leads)}\n"
        f"Median {_METRIC_LABELS[metric]}: {np.median(finite_values):.3f}\n"
        f"Best bin: {times[best_row]:%H:%M}, {leads[best_column]} min\n"
        f"Best median {_METRIC_LABELS[metric]}: {raw_values[best_row, best_column]:.3f}",
        va="top",
        fontsize=10,
        bbox={
            "facecolor": "#f7f7f7",
            "edgecolor": "#c7c7c7",
            "boxstyle": "round,pad=0.45",
        },
    )
    colorbar = fig.colorbar(
        image, ax=[heat_axis, time_axis], fraction=0.025, pad=0.02, ticks=bounds
    )
    colorbar.set_label(f"Median {_METRIC_LABELS[metric]} (discrete score bands)")
    fig.suptitle(
        f"Full-day {_METRIC_LABELS[metric]} structure with 30-minute initialization bins\n"
        f"{float(threshold_dbz):g} dBZ | white cells indicate no strictly matched sample",
        fontsize=15,
        fontweight="bold",
    )
    return fig


def _daily_heatmap_scale(metric: str):
    if metric in {"csi", "pod", "far"}:
        bounds = np.asarray([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
        base = plt.get_cmap("viridis")
    elif metric == "ets":
        bounds = np.asarray([-0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
        base = plt.get_cmap("RdYlGn")
    else:
        bounds = np.asarray([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4])
        base = plt.get_cmap("viridis")
    colors = base(np.linspace(0.05, 0.95, len(bounds) - 1))
    cmap = ListedColormap(colors).with_extremes(
        bad="white", under=colors[0], over=colors[-1]
    )
    return cmap, BoundaryNorm(bounds, cmap.N, clip=True), bounds


def _daily_summary_limits(metric: str, lower_values, upper_values):
    lower_finite = np.asarray(lower_values, dtype=float)
    upper_finite = np.asarray(upper_values, dtype=float)
    lower_finite = lower_finite[np.isfinite(lower_finite)]
    upper_finite = upper_finite[np.isfinite(upper_finite)]
    if metric in {"csi", "pod", "far"}:
        upper = max(0.5, float(np.max(upper_finite)) * 1.12)
        return 0.0, min(1.0, upper)
    if metric == "ets":
        lower = min(-0.1, float(np.min(lower_finite)) * 1.12)
        upper = max(0.5, float(np.max(upper_finite)) * 1.12)
        return max(-0.4, lower), min(1.0, upper)
    upper = max(2.0, float(np.max(upper_finite)) * 1.12)
    return 0.0, upper


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

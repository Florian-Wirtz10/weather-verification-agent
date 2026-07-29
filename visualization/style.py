"""Shared publication-style visual grammar for Phase 8 comparison figures."""

from __future__ import annotations

from matplotlib.lines import Line2D


MODEL_STYLES = {
    "stormdit": {"linestyle": "-", "linewidth": 2.3, "marker": "o"},
    "fenglei": {"linestyle": "--", "linewidth": 1.9, "marker": "s"},
}
MODEL_LABELS = {"stormdit": "StormDiT", "fenglei": "Fenglei"}

THRESHOLD_COLORS = (
    "#0072B2",
    "#009E73",
    "#E69F00",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
)
_CANONICAL_THRESHOLD_INDEX = {10.0: 0, 20.0: 1, 30.0: 2, 40.0: 3}
METRIC_COLORS = {
    "csi": "#0072B2",
    "pod": "#009E73",
    "far": "#D55E00",
}
METRIC_LABELS = {
    "csi": "CSI ↑",
    "pod": "POD ↑",
    "far": "FAR ↓",
    "bias_score": "Frequency bias",
}


def style_axes(axis):
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out")


def model_legend_handles(model_ids):
    return [
        Line2D(
            [0],
            [0],
            color="#30343b",
            label=MODEL_LABELS[model_id],
            markersize=4,
            **MODEL_STYLES[model_id],
        )
        for model_id in model_ids
    ]


def threshold_legend_handles(thresholds):
    return [
        Line2D(
            [0],
            [0],
            color=threshold_color(threshold, index),
            linewidth=2.2,
            label=f"{threshold:g} dBZ",
        )
        for index, threshold in enumerate(thresholds)
    ]


def threshold_color(threshold, fallback_index=0):
    index = _CANONICAL_THRESHOLD_INDEX.get(
        float(threshold),
        int(fallback_index) % len(THRESHOLD_COLORS),
    )
    return THRESHOLD_COLORS[index]

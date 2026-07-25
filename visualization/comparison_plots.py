"""Figures for the Phase 8 upstream-model/Fenglei comparison."""

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from verification.metrics import CATEGORICAL_SCORES
from visualization.plots import CATEGORICAL_COLORS


def plot_triplet_score_curves(result, metric: str):
    """Plot both forecasts under the same threshold colours and score axes."""
    if metric not in CATEGORICAL_SCORES:
        raise ValueError(f"unsupported comparison metric: {metric!r}")
    entries = result["per_lead"]
    if not entries:
        raise ValueError("triplet comparison contains no matched leads")

    thresholds = sorted(entries[0]["model"])
    leads = [entry["lead_minutes"] for entry in entries]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for index, threshold in enumerate(thresholds):
        color = CATEGORICAL_COLORS[index % len(CATEGORICAL_COLORS)]
        ax.plot(
            leads,
            [entry["model"][threshold][metric] for entry in entries],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=3.5,
        )
        ax.plot(
            leads,
            [entry["fenglei"][threshold][metric] for entry in entries],
            color=color,
            linewidth=1.8,
            linestyle="--",
        )

    product_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="#333333", linewidth=2.2, marker="o", label="Our model"),
            Line2D([0], [0], color="#333333", linewidth=1.8, linestyle="--", label="Fenglei"),
        ],
        title="Forecast product",
        loc="upper right",
    )
    ax.add_artist(product_legend)
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
    ax.set(
        title=f"{metric.upper()} comparison by forecast lead",
        xlabel="Lead time (min)",
        ylabel=metric.upper(),
        xlim=(min(leads), max(leads)),
    )
    if metric in {"csi", "pod", "far", "ets"}:
        ax.set_ylim(-0.05 if metric == "ets" else 0.0, 1.0)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

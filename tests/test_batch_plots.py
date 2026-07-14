import matplotlib.pyplot as plt
import pandas as pd

from visualization.batch_plots import (
    plot_daily_score_curves,
    plot_daily_score_distribution,
    plot_daily_score_heatmap,
)


def _frames():
    rows = []
    for init in ("2026-04-07T00:00:00", "2026-04-07T00:06:00"):
        for lead, value in ((6, 0.6), (12, 0.4)):
            for threshold in (5.0, 15.0):
                rows.append(
                    {
                        "forecast_path": f"{init}-{lead}",
                        "init_time_bjt": init,
                        "lead_minutes": lead,
                        "threshold_dbz": threshold,
                        "csi": value - threshold / 100,
                        "ets": value - threshold / 100 - 0.1,
                    }
                )
    per_case = pd.DataFrame(rows)
    summary = pd.DataFrame(
        per_case.groupby(["lead_minutes", "threshold_dbz"], as_index=False).agg(
            csi=("csi", "mean"), ets=("ets", "mean")
        )
    )
    return per_case, summary


def test_daily_batch_plot_suite():
    per_case, summary = _frames()

    figures = [
        plot_daily_score_curves(summary, "csi"),
        plot_daily_score_heatmap(per_case, "csi", 15),
        plot_daily_score_distribution(per_case, "csi", 15),
    ]

    assert all(figure.axes for figure in figures)
    assert "Figure 1" in figures[0].axes[0].get_title()
    assert "Figure 2" in figures[1].axes[0].get_title()
    assert "Figure 3" in figures[2].axes[0].get_title()
    for figure in figures:
        plt.close(figure)

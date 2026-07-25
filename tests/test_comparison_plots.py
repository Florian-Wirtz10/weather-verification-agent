from datetime import datetime

import matplotlib.pyplot as plt

from visualization.comparison_plots import plot_triplet_score_curves


def test_triplet_score_plot_uses_product_line_styles_without_grid():
    result = {
        "per_lead": [
            {
                "lead_minutes": lead,
                "valid_time_utc": datetime(2025, 4, 11, 0, lead),
                "model": {15.0: {"csi": model}},
                "fenglei": {15.0: {"csi": fenglei}},
            }
            for lead, model, fenglei in ((6, 0.8, 0.7), (12, 0.7, 0.6))
        ]
    }

    figure = plot_triplet_score_curves(result, "csi")

    try:
        axis = figure.axes[0]
        assert len(axis.lines) == 2
        assert axis.lines[0].get_linestyle() == "-"
        assert axis.lines[1].get_linestyle() == "--"
        assert not any(line.get_visible() for line in axis.get_xgridlines())
    finally:
        plt.close(figure)

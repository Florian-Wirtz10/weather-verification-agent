"""Assemble verification metrics and figures into on-disk artifacts."""

from pathlib import Path

import pandas as pd


def _metrics_to_dataframe(metrics):
    rows = []
    for threshold, scores in metrics["categorical"].items():
        row = {"threshold": threshold}
        row.update(scores)
        rows.append(row)
    return pd.DataFrame(rows).set_index("threshold")


def save_report(metrics, figures, outdir="outputs"):
    """Save figures to outdir/figures, metrics table to outdir/tables, summary to outdir/reports.

    Returns the outdir Path.
    """
    outdir = Path(outdir)
    figures_dir = outdir / "figures"
    tables_dir = outdir / "tables"
    reports_dir = outdir / "reports"
    for d in (figures_dir, tables_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        fig.savefig(figures_dir / f"{name}.png", dpi=150, bbox_inches="tight")

    table = _metrics_to_dataframe(metrics)
    table.to_csv(tables_dir / "categorical_metrics.csv")

    continuous = metrics["continuous"]
    summary_lines = ["# Verification Summary", "", "## Continuous error metrics", ""]
    summary_lines += [f"- {name}: {value:.4f}" for name, value in continuous.items()]
    summary_lines += ["", "## Categorical scores by threshold", "", table.to_string()]
    (reports_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    return outdir

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.benchmark_request import TEACHER_FIGURE_PROMPT, parse_benchmark_request
from visualization.benchmark_plots import (
    MODEL_COLORS, plot_benchmark_far_bars, plot_benchmark_pod_distribution,
    plot_benchmark_score_curves, plot_benchmark_fss_curves,
)
from webapi.execution import _benchmark_artifact_manifest


def test_teacher_template_is_offline_and_matches_frontend(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    request = parse_benchmark_request(TEACHER_FIGURE_PROMPT, single=False, common_horizon=72)
    assert request["thresholds_dbz"] == [10, 20, 35, 45]
    assert request["score_curve_metrics"] == ["csi"]
    assert request.get("fss_thresholds_dbz", request["thresholds_dbz"]) == [10, 20, 35, 45]
    assert request["views"] == ["score_curve", "pod_distribution", "far_bar", "fss_curve"]
    source = Path(__file__).resolve().parents[1] / "frontend/src/components/ModelPromptView.tsx"
    assert TEACHER_FIGURE_PROMPT in source.read_text()
    names = ["benchmark_csi.png", "benchmark_fss.png"] + [
        f"benchmark_pod_distribution_{t}.png" for t in (10, 20, 35, 45)
    ] + ["benchmark_far_bars.png"]
    manifest = _benchmark_artifact_manifest(request, ["PredRNN", "NowCastNet", "StormDiT"],
                                          [], [], [{"name": name} for name in names])
    assert manifest["gallery_files"] == names


def _rows():
    return pd.DataFrame([
        {"model": model, "lead_minutes": lead, "threshold_dbz": threshold,
         "csi": 0.5, "pod": pod, "far": 0.2 + lead / 100,
         "fss_w1": 0.5, "fss_w3": 0.6, "fss_w5": 0.7}
        for model in ("PredRNN", "NowCastNet", "StormDiT")
        for lead, pod in ((6, 0.8), (12, np.nan))
        for threshold in (10, 20, 35, 45)
    ])


def test_teacher_curves_have_large_text_and_no_arrows():
    rows = _rows().iloc[::-1]
    for figure in (plot_benchmark_score_curves(rows, "csi", [10, 20, 35, 45]),
                   plot_benchmark_fss_curves(rows, [10, 20, 35, 45])):
        assert figure._suptitle is None
        assert [t.get_text() for t in figure.legends[0].get_texts()] == [
            "PredRNN", "NowCastNet", "StormDiT"
        ]
        for axis in figure.axes:
            assert axis.get_title() == ""
            assert "↑" not in axis.get_ylabel()
            assert axis.yaxis.label.get_fontsize() >= 15
            assert axis.get_xticklabels()[0].get_fontsize() >= 13
            for line in axis.lines:
                assert line.get_color() == MODEL_COLORS[line.get_label()]


def test_teacher_distributions_exclude_undefined_scores_and_bars_keep_values():
    rows = _rows()
    for threshold in (10, 20, 35, 45):
        figure = plot_benchmark_pod_distribution(rows, threshold)
        axis = figure.axes[0]
        assert axis.get_ylabel() == "POD"
        assert len(axis.patches) == 3
        assert all("n=1" in label.get_text() for label in axis.get_xticklabels())
    bars = plot_benchmark_far_bars(rows, [10, 20, 35, 45])
    assert len(bars.axes) == 4
    for axis in bars.axes:
        assert axis.get_title() == ""
        assert axis.get_ylabel() == "FAR"
        np.testing.assert_allclose([p.get_height() for p in axis.patches], [0.26, 0.32] * 3)


@pytest.mark.parametrize("thresholds", ["10、20、35、45、55", "35、55", "5、10、15、20、30、35、45、55"])
def test_default_threshold_edits_work_without_llm(monkeypatch, thresholds):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    request = parse_benchmark_request(
        TEACHER_FIGURE_PROMPT.replace("10、20、35、45", thresholds),
        single=False, common_horizon=72,
    )
    expected = [float(t) for t in thresholds.split("、")]
    assert request["thresholds_dbz"] == expected
    assert request.get("fss_thresholds_dbz", expected) == expected
    assert request["diagnostic_threshold_dbz"] in expected
    assert request["score_curve_metrics"] == ["csi"]

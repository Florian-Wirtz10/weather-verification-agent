import json
from datetime import datetime, timezone
from types import SimpleNamespace

from agent.insight import build_insight_facts, generate_insight, render_deterministic_summary


def _result():
    scores = [
        (6, {"pod": 0.7, "far": 0.3, "csi": 0.5, "bias_score": 0.9, "ets": -0.1}),
        (
            18,
            {"pod": 0.6, "far": 0.4, "csi": float("nan"), "bias_score": 0.8, "ets": 0.05},
        ),
        (24, {"pod": 0.65, "far": 0.35, "csi": 0.3, "bias_score": 0.85, "ets": -0.02}),
    ]
    entries = []
    for index, (lead, categorical) in enumerate(scores):
        table = {
            "hits": 10 - index,
            "misses": 2 + index,
            "false_alarms": 3,
            "correct_negatives": 20,
            "valid_count": 35,
            "invalid_count": index,
        }
        entries.append(
            {
                "lead_minutes": lead,
                "valid_time": datetime(2025, 4, 11, 2, lead, tzinfo=timezone.utc),
                "categorical": {35: categorical},
                "contingency": {35: table},
            }
        )
    return {
        "context": {
            "product": "风雷 V1.0 REFMOSAIC",
            "variable": "组合反射率",
            "units": "dBZ",
            "init_time": datetime(2025, 4, 11, 2, 12, tzinfo=timezone.utc),
            "bbox": (28.0, 35.0, 108.0, 118.0),
            "temporal_matching": "有效时间严格相等",
            "spatial_alignment": "预报最近邻重采样到雷达网格",
            "source_path": "/secret/fenglei",
        },
        "coverage": {
            "requested_range": {"start": 6, "end": 24},
            "available_leads": [6, 12, 18, 24],
            "selected_leads": [6, 18, 24],
            "missing_leads": [12],
        },
        "per_lead": entries,
    }


def test_build_insight_facts_computes_evidence_and_is_strict_json():
    facts = build_insight_facts(_result())

    assert facts["context"]["init_time"] == "2025-04-11T02:12:00+00:00"
    assert "source_path" not in facts["context"]
    assert facts["coverage"]["missing_leads"] == [12]

    threshold = facts["thresholds"][0]
    assert threshold["threshold"] == 35.0
    assert threshold["metrics"]["ets"]["first"] == {"lead_minutes": 6, "value": -0.1}
    assert threshold["metrics"]["ets"]["maximum"] == {"lead_minutes": 18, "value": 0.05}
    assert threshold["metrics"]["ets"]["trend"]["non_monotonic"] is True
    assert threshold["negative_ets"] == {
        "count": 2,
        "leads": [6, 24],
        "values": [-0.1, -0.02],
    }
    assert threshold["metrics"]["csi"]["undefined_leads"] == [18]
    assert threshold["contingency"]["totals"]["hits"] == 27
    assert threshold["contingency"]["totals"]["valid_count"] == 105

    # No permissive NaN literals: this must be valid strict JSON.
    encoded = json.dumps(facts, ensure_ascii=False, allow_nan=False)
    assert "NaN" not in encoded


def test_deterministic_summary_reports_missing_negative_and_single_case_limit():
    summary = render_deterministic_summary(build_insight_facts(_result()))

    assert "缺少 12 分钟" in summary
    assert "ETS 在 2 个时效为负" in summary
    assert "单个起报个例" in summary
    assert "不能推广为模型整体能力结论" in summary


class _FakeCompletions:
    def __init__(self, captured):
        self.captured = captured

    def create(self, **kwargs):
        self.captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="客观事实：测试解读"))]
        )


def test_generate_insight_sends_only_sanitised_fact_json():
    captured = {}
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(captured)),
    )
    facts = build_insight_facts(_result())
    facts["source_path"] = "/do/not/send"
    facts["context"]["radar_file"] = "/secret/radar.nc"
    facts["raw_pixels"] = [[1, 2, 3]]

    text = generate_insight(facts, client=client)

    assert text == "客观事实：测试解读"
    payload = captured["messages"][1]["content"]
    decoded = json.loads(payload)
    assert decoded["context"]["product"] == "风雷 V1.0 REFMOSAIC"
    assert "/do/not/send" not in payload
    assert "/secret/radar.nc" not in payload
    assert "raw_pixels" not in payload
    assert captured["temperature"] == 0


def test_generate_insight_rejects_empty_response():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="  "))]
                )
            )
        )
    )

    try:
        generate_insight(build_insight_facts(_result()), client=client)
    except ValueError as exc:
        assert "empty insight response" in str(exc)
    else:
        raise AssertionError("an empty LLM response must raise")


def test_requested_metric_scope_excludes_unrequested_scores_from_llm_payload():
    captured = {}
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(captured)),
    )
    facts = build_insight_facts(_result(), requested_metrics=["csi", "ets"])

    assert facts["requested_metrics"] == ["csi", "ets"]
    assert set(facts["thresholds"][0]["metrics"]) == {"csi", "ets"}

    generate_insight(facts, client=client)
    payload = json.loads(captured["messages"][1]["content"])
    metric_names = set(payload["thresholds"][0]["metrics"])
    assert metric_names == {"csi", "ets"}
    assert "pod" not in captured["messages"][1]["content"]
    assert "far" not in captured["messages"][1]["content"]
    assert "bias_score" not in captured["messages"][1]["content"]

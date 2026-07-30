from datetime import datetime, timedelta

import pandas as pd
from streamlit.testing.v1 import AppTest

import agent
import data_io.forecast_sources
import verification.multimodel


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def _case():
    init = datetime(2026, 4, 8, 0)
    return {
        "case_id": "multimodel-202604080000",
        "init_time": init,
        "init_time_bjt": init + timedelta(hours=8),
        "available_leads": [6, 12],
        "complete": True,
        "missing_leads": [],
    }


def _single_result(model_ids=("stormdit",)):
    init = datetime(2026, 4, 8, 0)
    per_lead = []
    for lead, score in ((6, 0.8), (12, 0.7)):
        per_lead.append(
            {
                "lead_minutes": lead,
                "valid_time_utc": init + timedelta(minutes=lead),
                "models": {
                    model_id: {
                        15.0: {
                            "contingency": {
                                "hits": 8,
                                "misses": 1,
                                "false_alarms": 1,
                                "correct_negatives": 10,
                                "valid_count": 20,
                                "invalid_count": 0,
                            },
                            "pod": score,
                            "far": 1 - score,
                            "csi": score,
                            "bias_score": 1.0,
                            "ets": score - 0.1,
                        }
                    }
                    for model_id in model_ids
                },
            }
        )
    return {
        "context": {
            "model_ids": list(model_ids),
            "model_labels": ["StormDiT" if value == "stormdit" else "风雷" for value in model_ids],
            "model_contract": "phase8-inferred-v1",
            "init_time_utc": init,
            "init_time_bjt": init + timedelta(hours=8),
        },
        "coverage": {
            "selected_leads": [6, 12],
            "missing_leads": [],
            "complete": True,
        },
        "per_lead": per_lead,
        "frames": {},
    }


def _patch_catalog(monkeypatch):
    monkeypatch.setattr(
        data_io.forecast_sources, "build_multimodel_catalog", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        data_io.forecast_sources,
        "discover_multimodel_initializations",
        lambda *_args, **_kwargs: [_case()],
    )
    monkeypatch.setattr(
        agent,
        "generate_comparison_insight",
        lambda *_args, **_kwargs: "StormDiT测试解读",
    )


def test_streamlit_initial_screen_has_final_phase8_controls(monkeypatch):
    _patch_catalog(monkeypatch)

    app = AppTest.from_file("app.py", default_timeout=30).run()

    assert not app.exception
    assert [item.value for item in app.number_input] == [29.0, 33.5, 108.0, 116.5]
    assert app.selectbox[1].options == ["StormDiT", "风雷", "StormDiT vs 风雷"]
    assert app.radio[0].options == ["全部起报", "自定义多起报", "单一起报"]
    assert app.checkbox[0].value is True
    assert "刷新可用起报" not in {button.label for button in app.button}
    assert any("支持范围：" in item.value for item in app.caption)


def test_default_prompt_tracks_selected_product(monkeypatch):
    _patch_catalog(monkeypatch)

    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert "StormDiT和风雷" in app.text_area[0].value

    app = app.selectbox[1].set_value("StormDiT").run()
    assert "StormDiT的CSI" in app.text_area[0].value
    assert "StormDiT和风雷" not in app.text_area[0].value

    app = app.selectbox[1].set_value("风雷").run()
    assert "风雷的CSI" in app.text_area[0].value
    assert "StormDiT" not in app.text_area[0].value

    app = app.radio[0].set_value("单一起报").run()
    assert "风雷预报与CREF观测" in app.text_area[0].value


def test_single_stormdit_result_has_ai_and_downloads(monkeypatch):
    _patch_catalog(monkeypatch)
    request = agent.default_multimodel_request(batch=False)
    request["thresholds_dbz"] = [15.0]
    request["metrics"] = ["csi"]
    request["views"] = ["score_curve"]
    monkeypatch.setattr(agent, "default_multimodel_request", lambda **_kwargs: request)
    monkeypatch.setattr(
        verification.multimodel,
        "run_multimodel_case",
        lambda *_args, **_kwargs: _single_result(),
    )

    app = AppTest.from_file("app.py", default_timeout=30).run()
    app = app.selectbox[1].set_value("StormDiT").run()
    app = app.radio[0].set_value("单一起报").run()
    app = _button(app, "使用推荐方案").click().run()

    assert not app.exception
    assert len(app.image) == 1
    assert any("StormDiT测试解读" in item.value for item in app.markdown)
    labels = {button.label for button in app.download_button}
    assert {
        "下载结果包 ZIP",
        "下载指标 CSV",
        "下载AI解读",
        "下载 PNG",
        "下载 SVG",
    } <= labels


def test_two_model_batch_uses_final_product_name(monkeypatch):
    _patch_catalog(monkeypatch)
    request = agent.default_multimodel_request(batch=True)
    request["thresholds_dbz"] = [15.0]
    request["metrics"] = ["csi"]
    request["views"] = ["score_curve"]
    monkeypatch.setattr(agent, "default_multimodel_request", lambda **_kwargs: request)
    single = _single_result(("stormdit", "fenglei"))
    rows = []
    for entry in single["per_lead"]:
        for model_id, values in entry["models"].items():
            score = values[15.0]
            rows.append(
                {
                    "model_id": model_id,
                    "init_time_utc": "2026-04-08T00:00:00",
                    "init_time_bjt": "2026-04-08T08:00:00",
                    "valid_time_utc": entry["valid_time_utc"].isoformat(),
                    "lead_minutes": entry["lead_minutes"],
                    "threshold_dbz": 15.0,
                    **score["contingency"],
                    **{key: score[key] for key in ("pod", "far", "csi", "bias_score", "ets")},
                }
            )
    frame = pd.DataFrame(rows)
    summary = frame.copy()
    summary["case_count"] = 1
    result = {
        "context": {
            "model_ids": ["stormdit", "fenglei"],
            "model_labels": ["StormDiT", "风雷"],
            "model_contract": "phase8-inferred-v1",
        },
        "coverage": {
            "selected_initialization_count": 1,
            "selected_pairs": 2,
            "complete_initializations": 1,
        },
        "per_case": frame,
        "summary": summary,
        "cache": {"computed_initializations": 0},
    }
    monkeypatch.setattr(
        verification.multimodel,
        "run_multimodel_batch",
        lambda *_args, **_kwargs: result,
    )

    app = AppTest.from_file("app.py", default_timeout=30).run()
    app = app.selectbox[1].set_value("StormDiT vs 风雷").run()
    app = _button(app, "使用推荐方案").click().run()

    assert not app.exception
    assert len(app.image) == 1
    assert any("StormDiT vs 风雷" in item.value for item in app.info)

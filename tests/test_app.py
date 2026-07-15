import matplotlib.pyplot as plt
import numpy as np
from streamlit.testing.v1 import AppTest

import agent
from tests.test_real_data_io import _make_cref_nc, _make_swan_mosaic_bytes
from tests.test_catalog import _write_forecast
from tests.test_radar_binary import _write_binary_cref


def _write_app_case(tmp_path):
    fenglei_dir = tmp_path / "fenglei"
    radar_dir = tmp_path / "radar"
    fenglei_dir.mkdir()
    radar_dir.mkdir()

    raw = np.zeros((6, 6), dtype="u1")
    raw[2:4, 2:5] = 140
    (fenglei_dir / "REFMOSAIC-202504110212-006MIN.bz2").write_bytes(
        _make_swan_mosaic_bytes(
            nx=6,
            ny=6,
            cell_bytes=1,
            scale=0.5,
            offset=-33.0,
            values=raw,
        )
    )
    obs = np.zeros((6, 6), dtype="f4")
    obs[2:4, 2:5] = 40.0
    _make_cref_nc(
        radar_dir / "CREF_with_time_fixture.nc",
        lat=np.linspace(29.95, 30.0, 6, dtype="f4"),
        lon=np.linspace(110.0, 110.05, 6, dtype="f4"),
        data=obs,
        valid_time_str="2025-04-11 02:18:00",
    )
    return fenglei_dir, radar_dir


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def test_streamlit_initial_screen_renders_without_exceptions():
    app = AppTest.from_file("app.py", default_timeout=30).run()

    assert not app.exception
    labels = {button.label for button in app.button}
    assert "智能解析并运行" in labels
    assert "使用推荐方案" in labels


def test_phase4_result_has_browser_downloads_and_releases_figures(tmp_path, monkeypatch):
    fenglei_dir, radar_dir = _write_app_case(tmp_path)
    monkeypatch.setenv("FENGLEI_CASE_DIR", str(fenglei_dir))
    monkeypatch.setenv("RADAR_OBS_DIR", str(radar_dir))
    monkeypatch.setattr(agent, "generate_insight", lambda *_args, **_kwargs: "测试解读")
    plt.close("all")

    app = AppTest.from_file("app.py", default_timeout=30).run()
    app = app.selectbox[0].set_value("2025-04-11").run()
    app = app.radio[0].set_value("单一起报").run()
    for control, value in zip(app.number_input, (29.9, 30.1, 109.9, 110.1)):
        app = control.set_value(value).run()
    app = _button(app, "使用推荐方案").click().run()

    assert not app.exception
    assert len(app.image) == 1
    download_labels = {button.label for button in app.download_button}
    assert {
        "下载结果包 ZIP",
        "下载指标 CSV",
        "下载解读 Markdown",
        "下载 PNG",
    } <= download_labels
    assert any(item.value == "AI 解读" for item in app.subheader)
    assert not plt.get_fignums()


def test_single_init_mode_accepts_binary_cref(tmp_path, monkeypatch):
    forecast_dir = tmp_path / "forecast"
    obs_dir = tmp_path / "obs"
    forecast_dir.mkdir()
    obs_dir.mkdir()
    _write_forecast(forecast_dir)
    _write_binary_cref(obs_dir)
    monkeypatch.setenv("FENGLEI_DAILY_DIR", str(forecast_dir))
    monkeypatch.setenv("RADAR_DAILY_DIR", str(obs_dir))
    monkeypatch.setattr(agent, "generate_insight", lambda *_args, **_kwargs: "二进制观测测试解读")

    app = AppTest.from_file("app.py", default_timeout=30).run()
    app = app.radio[0].set_value("单一起报").run()
    app = _button(app, "使用推荐方案").click().run()

    assert not app.exception
    assert len(app.image) == 1
    assert any("二进制观测测试解读" in item.value for item in app.markdown)


def test_phase5_full_day_result_has_three_diagnostics_and_downloads(tmp_path, monkeypatch):
    forecast_dir = tmp_path / "daily-forecast"
    obs_dir = tmp_path / "daily-obs"
    forecast_dir.mkdir()
    obs_dir.mkdir()
    _write_forecast(forecast_dir)
    _write_binary_cref(obs_dir)
    monkeypatch.setenv("FENGLEI_DAILY_DIR", str(forecast_dir))
    monkeypatch.setenv("RADAR_DAILY_DIR", str(obs_dir))
    monkeypatch.setattr(agent, "generate_daily_insight", lambda *_args, **_kwargs: "全天测试解读")

    app = AppTest.from_file("app.py", default_timeout=30).run()
    app = _button(app, "使用推荐方案").click().run()

    assert not app.exception
    assert len(app.image) == 3
    download_labels = [button.label for button in app.download_button]
    assert download_labels.count("下载 PNG") == 3
    assert "下载逐起报 CSV" in download_labels
    assert "下载汇总 CSV" in download_labels
    assert "下载解读 Markdown" in download_labels
    assert any("1/1 对严格匹配样本" in item.value for item in app.markdown)

from streamlit.testing.v1 import AppTest


def test_streamlit_initial_screen_renders_without_exceptions():
    app = AppTest.from_file("app.py", default_timeout=30).run()

    assert not app.exception
    labels = {button.label for button in app.button}
    assert "解析并运行" in labels
    assert "使用明确默认方案运行" in labels

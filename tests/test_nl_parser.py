from agent.nl_parser import DEFAULT_THRESHOLDS, PLOT_MENU, _validate, select_figures


def test_validate_passes_through_known_plots_and_thresholds():
    parsed = {"thresholds": [15, 25], "plots": ["csi_time_series", "reflectivity_field"], "notes": "ok"}
    result = _validate(parsed)
    assert result["thresholds"] == [15.0, 25.0]
    assert result["plots"] == ["csi_time_series", "reflectivity_field"]
    assert result["notes"] == "ok"


def test_validate_drops_unknown_plot_names():
    # A hallucinated/injected plot name (or arbitrary code) must never survive.
    parsed = {"plots": ["csi_time_series", "os.system('rm -rf /')", "made_up_plot"]}
    result = _validate(parsed)
    assert result["plots"] == ["csi_time_series"]


def test_validate_falls_back_to_all_plots_when_nothing_valid_selected():
    result = _validate({"plots": ["nonsense"]})
    assert result["plots"] == list(PLOT_MENU)


def test_validate_falls_back_to_default_thresholds_on_bad_input():
    result = _validate({"thresholds": "not a list"})
    assert result["thresholds"] == list(DEFAULT_THRESHOLDS)


def test_validate_handles_missing_fields():
    result = _validate({})
    assert result["thresholds"] == list(DEFAULT_THRESHOLDS)
    assert result["plots"] == list(PLOT_MENU)
    assert result["notes"] == ""


def test_select_figures_maps_plot_names_to_figure_keys_in_order():
    figures = {
        "csi_time_series": "csi-fig",
        "forecast_first_lead": "fc-first",
        "obs_first_lead": "obs-first",
        "error_first_lead": "err-first",
    }
    selected = select_figures(figures, ["reflectivity_field", "csi_time_series", "reflectivity_error"])
    assert list(selected) == ["forecast_first_lead", "obs_first_lead", "csi_time_series", "error_first_lead"]


def test_select_figures_skips_figures_the_run_didnt_produce():
    figures = {"csi_time_series": "csi-fig"}
    selected = select_figures(figures, ["reflectivity_field", "csi_time_series"])
    assert selected == {"csi_time_series": "csi-fig"}

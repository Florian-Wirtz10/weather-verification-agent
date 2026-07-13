from agent.presentation import resolve_media_layout


def test_auto_layout_uses_actual_generated_media_not_parser_guess():
    assert (
        resolve_media_layout(
            "auto",
            figure_metadata={"view_01": {"type": "score_time_series"}},
            animation_metadata={},
        )
        == "report"
    )
    assert (
        resolve_media_layout(
            "auto",
            figure_metadata={
                "view_01": {"type": "reflectivity_comparison"},
                "view_02": {"type": "reflectivity_error"},
            },
            animation_metadata={},
        )
        == "grid_2x2"
    )
    assert (
        resolve_media_layout(
            "auto",
            figure_metadata={},
            animation_metadata={"animation_01": {"type": "reflectivity_animation"}},
        )
        == "animation_focus"
    )


def test_explicit_safe_layout_overrides_auto_policy():
    assert (
        resolve_media_layout(
            "single",
            figure_metadata={},
            animation_metadata={"animation_01": {"type": "reflectivity_animation"}},
        )
        == "single"
    )

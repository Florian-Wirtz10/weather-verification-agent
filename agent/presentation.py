"""Deterministic presentation policy for already-rendered verification media."""

from __future__ import annotations

from collections.abc import Mapping

from agent.request_schema import SUPPORTED_LAYOUTS


def resolve_media_layout(
    requested_layout: object,
    *,
    figure_metadata: Mapping[str, Mapping[str, object]],
    animation_metadata: Mapping[str, Mapping[str, object]],
) -> str:
    """Choose a safe layout from the media that actually exist.

    Explicit schema-approved choices win. ``auto`` is deliberately resolved
    after rendering so a parser cannot request a visually unsuitable layout
    for a result that became partial because some requested leads were absent.
    """
    layout = requested_layout if isinstance(requested_layout, str) else "auto"
    layout = layout.strip().lower()
    if layout not in SUPPORTED_LAYOUTS:
        layout = "auto"
    if layout != "auto":
        return layout

    if animation_metadata:
        return "animation_focus"

    static_views = list(figure_metadata.values())
    spatial_count = sum(
        view.get("type") in {"reflectivity_comparison", "reflectivity_error"}
        for view in static_views
    )
    if spatial_count >= 2:
        return "grid_2x2"
    if len(static_views) >= 3:
        return "two_column"
    return "report"

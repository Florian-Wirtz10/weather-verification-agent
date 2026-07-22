"""Lightweight, deterministic administrative-boundary overlays.

The bundled boundary geometry is a simplified Natural Earth 1:10m Admin-1
extract for China.  Plotters use it only as a visual overlay: adding geography
must never alter the plotted data, colour normalization, colourbar, or extent.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np


_BOUNDARY_PATH = Path(__file__).with_name("assets") / "china_admin1_simplified.geojson"


@lru_cache(maxsize=1)
def _features() -> tuple[dict, ...]:
    with _BOUNDARY_PATH.open(encoding="utf-8") as handle:
        collection = json.load(handle)
    features = collection.get("features")
    if not isinstance(features, list):
        raise ValueError(f"invalid administrative-boundary asset: {_BOUNDARY_PATH}")
    return tuple(feature for feature in features if isinstance(feature, dict))


def _rings(geometry):
    if not isinstance(geometry, dict):
        return
    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") == "Polygon":
        yield from coordinates
    elif geometry.get("type") == "MultiPolygon":
        for polygon in coordinates:
            yield from polygon


def add_admin_boundaries(ax, extent) -> int:
    """Overlay province boundaries and restore the caller's exact map extent.

    ``extent`` is ``(lon_min, lon_max, lat_min, lat_max)``.  The return value is
    the number of line artists added, which keeps the behaviour easy to test.
    """
    lon_min, lon_max, lat_min, lat_max = map(float, extent)
    added = 0
    for feature in _features():
        properties = feature.get("properties")
        province = properties.get("name") if isinstance(properties, dict) else ""
        is_hubei = province == "Hubei"
        for ring in _rings(feature.get("geometry")):
            coordinates = np.asarray(ring, dtype=float)
            if coordinates.ndim != 2 or coordinates.shape[1] < 2:
                continue
            if (
                coordinates[:, 0].max() < lon_min
                or coordinates[:, 0].min() > lon_max
                or coordinates[:, 1].max() < lat_min
                or coordinates[:, 1].min() > lat_max
            ):
                continue
            ax.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                color="#1d252c" if is_hubei else "#555f68",
                linewidth=1.05 if is_hubei else 0.55,
                alpha=0.95,
                zorder=8,
            )
            added += 1

    # Matplotlib line artists participate in autoscaling.  Restore the image's
    # original limits so the boundary layer cannot expand or crop the map.
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    return added

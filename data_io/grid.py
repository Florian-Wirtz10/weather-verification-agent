"""Shared gridded-field representation for real observation/forecast readers.

Both the fenglei mosaic reader and the radar CREF reader normalize into this
shape so verification/visualization code doesn't need to know which file
format a field came from.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class GriddedField:
    """A single 2D field on a regular lat/lon grid.

    `data` is south-to-north, west-to-east (row 0 = southernmost latitude),
    matching `lat`/`lon`, which are both ascending — the same convention
    `matplotlib`'s `imshow(..., origin="lower")` expects.
    """

    data: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    valid_time: datetime
    product: str
    units: str
    init_time: datetime | None = None
    lead_minutes: int | None = None

    def crop(self, lat_min, lat_max, lon_min, lon_max):
        """Return a new GriddedField cropped to the given lat/lon bounding box."""
        lat_idx = np.nonzero((self.lat >= lat_min) & (self.lat <= lat_max))[0]
        lon_idx = np.nonzero((self.lon >= lon_min) & (self.lon <= lon_max))[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            raise ValueError(
                f"crop bbox ({lat_min}, {lat_max}, {lon_min}, {lon_max}) does not "
                f"intersect field extent (lat {self.lat.min()}..{self.lat.max()}, "
                f"lon {self.lon.min()}..{self.lon.max()})"
            )
        return GriddedField(
            data=self.data[np.ix_(lat_idx, lon_idx)],
            lat=self.lat[lat_idx],
            lon=self.lon[lon_idx],
            valid_time=self.valid_time,
            product=self.product,
            units=self.units,
            init_time=self.init_time,
            lead_minutes=self.lead_minutes,
        )


def _nearest_indices(source_coords, target_coords):
    """For each value in target_coords, return the index of its nearest neighbor in
    source_coords (assumed sorted ascending)."""
    idx = np.searchsorted(source_coords, target_coords)
    idx0 = np.clip(idx - 1, 0, len(source_coords) - 1)
    idx1 = np.clip(idx, 0, len(source_coords) - 1)
    use_idx1 = np.abs(source_coords[idx1] - target_coords) < np.abs(
        source_coords[idx0] - target_coords
    )
    return np.where(use_idx1, idx1, idx0)


def regrid_nearest(source, target_lat, target_lon):
    """Resample `source.data` onto (target_lat, target_lon) by nearest-neighbor lookup.

    Real 风雷/雷达 products don't share an exact common grid origin — resolutions match
    (0.01 deg) but origins differ by a fraction of a cell — so index-aligning crops isn't
    safe; this regrids source onto the target's native grid instead.
    """
    target_lat = np.asarray(target_lat)
    target_lon = np.asarray(target_lon)
    lat_idx = _nearest_indices(source.lat, target_lat)
    lon_idx = _nearest_indices(source.lon, target_lon)
    result = source.data[np.ix_(lat_idx, lon_idx)]

    # Nearest-neighbor lookup is meaningful only within the source domain.  Clipping
    # indices outside it would otherwise copy edge cells into places with no forecast.
    outside = (
        (target_lat[:, None] < source.lat[0])
        | (target_lat[:, None] > source.lat[-1])
        | (target_lon[None, :] < source.lon[0])
        | (target_lon[None, :] > source.lon[-1])
    )
    if np.any(outside):
        if not np.issubdtype(result.dtype, np.inexact):
            result = result.astype(np.result_type(result.dtype, np.float32))
        result[outside] = np.nan
    return result


def align_pair(forecast, obs, lat_min, lat_max, lon_min, lon_max):
    """Crop forecast/obs to a common bounding box and resample forecast onto obs's native
    grid (nearest neighbor), so both come back with obs's shape/coordinates — the shape
    verification.metrics needs for a per-cell forecast/obs comparison.
    """
    obs_crop = obs.crop(lat_min, lat_max, lon_min, lon_max)

    # Regrid from the complete forecast domain.  Cropping it first can discard the
    # nearest source cell merely because that cell lies just outside the requested bbox.
    regridded_data = regrid_nearest(forecast, obs_crop.lat, obs_crop.lon)
    forecast_on_obs_grid = GriddedField(
        data=regridded_data,
        lat=obs_crop.lat,
        lon=obs_crop.lon,
        valid_time=forecast.valid_time,
        product=forecast.product,
        units=forecast.units,
        init_time=forecast.init_time,
        lead_minutes=forecast.lead_minutes,
    )
    return forecast_on_obs_grid, obs_crop

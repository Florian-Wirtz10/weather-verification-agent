"""Reader for CREF (combined reflectivity) radar observation NetCDF files.

These are the ground-truth field for reflectivity verification — not to be
confused with 风雷 REFMOSAIC, which is 风雷's own reflectivity *forecast*
product (see data_io.fenglei_mosaic).
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import netCDF4

# The CREF files' `time` units string encodes Beijing Time (UTC+8), verified
# against the 2025-04-11 case: file "..._1018.nc" lines up exactly with the
# 风雷 REFMOSAIC file whose (UTC) header valid time is 02:18 — i.e. the two
# only match once this offset is applied. 风雷 headers are UTC, so normalize
# radar observations to UTC here to keep valid_time comparable across readers.
_CREF_TIME_IS_BEIJING = timedelta(hours=8)


def _decode_valid_time(time_var) -> datetime:
    """Decode a CREF timestamp and normalize Beijing Time to UTC."""
    valid_time = cast(
        datetime,
        netCDF4.num2date(
            time_var[0],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        ),
    )
    return valid_time - _CREF_TIME_IS_BEIJING


def read_cref_valid_time(path) -> datetime:
    """Read only a CREF file's timestamp, without loading its 2D data field."""
    with netCDF4.Dataset(Path(path)) as ds:
        return _decode_valid_time(ds.variables["time"])


def index_cref_files(directory, pattern="CREF_with_time_*.nc") -> dict[datetime, Path]:
    """Map UTC valid times to CREF paths while loading metadata only.

    Keeping paths instead of :class:`GriddedField` instances lets callers load
    one large radar grid at a time rather than retaining an entire time series
    in memory.
    """
    index: dict[datetime, Path] = {}
    for path in sorted(Path(directory).glob(pattern)):
        valid_time = read_cref_valid_time(path)
        if valid_time in index:
            raise ValueError(
                f"duplicate radar observations for {valid_time:%Y-%m-%d %H:%M:%S} UTC: "
                f"{index[valid_time]} and {path}"
            )
        index[valid_time] = path
    return index


def index_radar_observation_files(directory) -> dict[datetime, Path]:
    """Index NetCDF and MOC-binary CREF files by their verified UTC times."""
    directory = Path(directory)
    paths = sorted(
        path
        for pattern in ("*.nc", "*.bin")
        for path in directory.glob(pattern)
    )
    index: dict[datetime, Path] = {}
    for path in paths:
        if path.suffix.lower() == ".bin":
            from data_io.radar_binary import read_cref_binary_valid_time

            valid_time = read_cref_binary_valid_time(path)
        else:
            valid_time = read_cref_valid_time(path)
        if valid_time in index:
            raise ValueError(
                f"duplicate radar observations for {valid_time:%Y-%m-%d %H:%M:%S} UTC: "
                f"{index[valid_time]} and {path}"
            )
        index[valid_time] = path
    return index


def read_cref(path):
    """Read a single-timestep CREF observation NetCDF file into a GriddedField."""
    from data_io.grid import GriddedField

    path = Path(path)

    with netCDF4.Dataset(path) as ds:
        cref = ds.variables["CREF"][:]
        lat = ds.variables["lat"][:].astype("float64")
        lon = ds.variables["lon"][:].astype("float64")
        valid_time = _decode_valid_time(ds.variables["time"])

    data = cref[0]
    if lat[0] > lat[-1]:
        data = data[::-1, :]
        lat = lat[::-1]

    return GriddedField(
        data=data.astype("float32"),
        lat=lat,
        lon=lon,
        valid_time=valid_time,
        product="CREF_obs",
        units="dBZ",
        init_time=valid_time,
        lead_minutes=0,
    )


def read_cref_series(directory, pattern="CREF_with_time_*.nc"):
    """Eagerly read every matching CREF file, sorted by valid time.

    Batch verification should prefer :func:`index_cref_files` and load one
    indexed path at a time; this helper remains for small, already-bounded sets.
    """
    directory = Path(directory)
    fields = [read_cref(p) for p in sorted(directory.glob(pattern))]
    fields.sort(key=lambda field: field.valid_time)
    return fields


def read_radar_observation(path, *, bbox=None):
    """Read either the legacy NetCDF CREF or MOC V3.0 binary CREF product."""
    path = Path(path)
    if path.suffix.lower() == ".bin":
        from data_io.radar_binary import read_cref_binary

        return read_cref_binary(path, bbox=bbox)
    if path.suffix.lower() != ".nc":
        raise ValueError(f"unsupported radar observation format: {path.suffix or path}")
    field = read_cref(path)
    if bbox is not None:
        return field.crop(*bbox)
    return field


def nearest_in_time(fields, target_time, tolerance=timedelta(minutes=3)):
    """Return the field in `fields` closest to `target_time`, within `tolerance`."""
    best = min(fields, key=lambda field: abs(field.valid_time - target_time))
    if abs(best.valid_time - target_time) > tolerance:
        return None
    return best

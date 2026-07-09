"""Reader for CREF (combined reflectivity) radar observation NetCDF files.

These are the ground-truth field for reflectivity verification — not to be
confused with 风雷 REFMOSAIC, which is 风雷's own reflectivity *forecast*
product (see data_io.fenglei_mosaic).
"""

from datetime import timedelta
from pathlib import Path

import netCDF4

# The CREF files' `time` units string encodes Beijing Time (UTC+8), verified
# against the 2025-04-11 case: file "..._1018.nc" lines up exactly with the
# 风雷 REFMOSAIC file whose (UTC) header valid time is 02:18 — i.e. the two
# only match once this offset is applied. 风雷 headers are UTC, so normalize
# radar observations to UTC here to keep valid_time comparable across readers.
_CREF_TIME_IS_BEIJING = timedelta(hours=8)


def read_cref(path):
    """Read a single-timestep CREF observation NetCDF file into a GriddedField."""
    from data_io.grid import GriddedField

    path = Path(path)

    with netCDF4.Dataset(path) as ds:
        cref = ds.variables["CREF"][:]
        lat = ds.variables["lat"][:].astype("float64")
        lon = ds.variables["lon"][:].astype("float64")
        time_var = ds.variables["time"]
        valid_time = netCDF4.num2date(
            time_var[0],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        valid_time -= _CREF_TIME_IS_BEIJING

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
    """Read every CREF file in `directory` matching `pattern`, sorted by valid time."""
    directory = Path(directory)
    fields = [read_cref(p) for p in sorted(directory.glob(pattern))]
    fields.sort(key=lambda field: field.valid_time)
    return fields


def nearest_in_time(fields, target_time, tolerance=timedelta(minutes=3)):
    """Return the field in `fields` closest to `target_time`, within `tolerance`."""
    best = min(fields, key=lambda field: abs(field.valid_time - target_time))
    if abs(best.valid_time - target_time) > tolerance:
        return None
    return best

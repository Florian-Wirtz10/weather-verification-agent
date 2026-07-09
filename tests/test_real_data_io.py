"""Tests for the real-data readers (风雷 mosaic bz2, radar CREF NetCDF) and grid
alignment, using small synthetic fixtures built in-memory — never the private
sample data under external/, which isn't tracked in this repo."""

import bz2
from datetime import datetime, timedelta

import netCDF4
import numpy as np
import pytest

from data_io.fenglei_mosaic import SwanHdr, read_swan_mosaic
from data_io.grid import GriddedField, align_pair, regrid_nearest
from data_io.radar_obs import nearest_in_time, read_cref


def _pack_text(text, width, encoding="gbk"):
    raw = text.encode(encoding)
    return np.frombuffer(raw.ljust(width, b"\x00"), dtype="S1")


def _make_swan_mosaic_bytes(
    *, nx, ny, cell_bytes, scale, offset, values, header_overrides=None, product_name="REF_TEST"
):
    header = np.zeros(1, dtype=SwanHdr)[0]
    header["ZoneName"] = _pack_text("diamond 131", 12)
    header["DataName"] = _pack_text(product_name, 38)
    header["year"], header["month"], header["day"] = 2025, 4, 11
    header["hour"], header["minute"] = 2, 12
    header["interval"] = 6
    header["XNumGrids"], header["YNumGrids"], header["ZNumGrids"] = nx, ny, 1
    header["StartLon"], header["StartLat"] = 110.0, 30.0
    header["XReso"], header["YReso"] = 0.01, 0.01
    header["scale"], header["offset"] = scale, offset
    header["m_iDataType"] = 0  # deliberately "unreliable" per the real format
    for key, value in (header_overrides or {}).items():
        header[key] = value

    dtype = {1: "u1", 2: "<i2", 4: "<f4"}[cell_bytes]
    body = np.asarray(values, dtype=dtype).tobytes()
    return bz2.compress(header.tobytes() + body)


def test_read_swan_mosaic_infers_dtype_and_scales(tmp_path):
    nx, ny = 4, 3
    # north-to-south, west-to-east raw storage per the real format
    raw = np.arange(nx * ny, dtype="u1").reshape(ny, nx)
    blob = _make_swan_mosaic_bytes(nx=nx, ny=ny, cell_bytes=1, scale=0.5, offset=-33.0, values=raw)
    path = tmp_path / "sample-REFMOSAIC-202504110212-006MIN.bz2"
    path.write_bytes(blob)

    field = read_swan_mosaic(path)

    assert field.data.shape == (ny, nx)
    # row 0 of `data` is south -> the last raw row (north-to-south storage flipped)
    np.testing.assert_allclose(field.data[0], raw[-1] * 0.5 - 33.0)
    np.testing.assert_allclose(field.data[-1], raw[0] * 0.5 - 33.0)
    assert list(field.lat) == sorted(field.lat)
    assert list(field.lon) == sorted(field.lon)
    assert field.lon[0] == pytest.approx(110.0)
    assert field.init_time == datetime(2025, 4, 11, 2, 12)
    assert field.lead_minutes == 6
    assert field.valid_time == datetime(2025, 4, 11, 2, 18)
    assert field.units == "dBZ"


def test_read_swan_mosaic_two_byte_precip(tmp_path):
    nx, ny = 2, 2
    raw = np.array([[0, 50], [100, 109]], dtype="<i2")
    blob = _make_swan_mosaic_bytes(
        nx=nx, ny=ny, cell_bytes=2, scale=0.1, offset=0.0, values=raw, product_name="PRE_TEST"
    )
    path = tmp_path / "sample-V11-PRE-202504110212-006MIN.bz2"
    path.write_bytes(blob)

    field = read_swan_mosaic(path)

    assert field.data.dtype == np.float32
    np.testing.assert_allclose(sorted(field.data.ravel()), [0.0, 5.0, 10.0, 10.9], atol=1e-4)
    assert field.units == "mm"


def test_read_swan_mosaic_rejects_unresolvable_cell_width(tmp_path):
    nx, ny = 3, 3
    blob_header = np.zeros(1, dtype=SwanHdr)[0]
    blob_header["XNumGrids"], blob_header["YNumGrids"], blob_header["ZNumGrids"] = nx, ny, 1
    # body length not a whole multiple of nx*ny cells for any known dtype width
    bad_body = b"\x00" * (nx * ny * 3)
    path = tmp_path / "bad.bz2"
    path.write_bytes(bz2.compress(blob_header.tobytes() + bad_body))

    with pytest.raises(ValueError, match="can't infer"):
        read_swan_mosaic(path)


def _make_cref_nc(path, *, lat, lon, data, valid_time_str):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))

        time_var = ds.createVariable("time", "i8", ("time",))
        time_var.units = f"days since {valid_time_str}"
        time_var.calendar = "proleptic_gregorian"
        time_var[:] = [0]

        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lat_var[:] = lat
        lon_var = ds.createVariable("lon", "f4", ("lon",))
        lon_var[:] = lon

        cref_var = ds.createVariable("CREF", "f4", ("time", "lat", "lon"))
        cref_var[:] = data[None, :, :]


def test_read_cref_converts_beijing_time_to_utc(tmp_path):
    lat = np.array([28.0, 28.01, 28.02], dtype="f4")
    lon = np.array([110.0, 110.01], dtype="f4")
    data = np.array([[0.0, 10.0], [20.0, 30.0], [40.0, 50.0]], dtype="f4")
    path = tmp_path / "cref.nc"
    _make_cref_nc(path, lat=lat, lon=lon, data=data, valid_time_str="2025-04-11 10:18:00")

    field = read_cref(path)

    assert field.valid_time == datetime(2025, 4, 11, 2, 18)  # BJT 10:18 -> UTC 02:18
    assert field.units == "dBZ"
    np.testing.assert_allclose(field.lat, lat)
    np.testing.assert_array_equal(field.data, data)


def test_read_cref_flips_descending_lat(tmp_path):
    lat = np.array([28.02, 28.01, 28.0], dtype="f4")  # descending
    lon = np.array([110.0, 110.01], dtype="f4")
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="f4")
    path = tmp_path / "cref_desc.nc"
    _make_cref_nc(path, lat=lat, lon=lon, data=data, valid_time_str="2025-04-11 10:00:00")

    field = read_cref(path)

    assert list(field.lat) == sorted(field.lat)
    np.testing.assert_array_equal(field.data[0], data[-1])


def test_nearest_in_time_respects_tolerance():
    base = datetime(2025, 4, 11, 2, 0)
    fields = [
        GriddedField(
            data=np.zeros((1, 1)), lat=np.array([0.0]), lon=np.array([0.0]),
            valid_time=base + timedelta(minutes=step), product="obs", units="dBZ",
        )
        for step in (0, 6, 12)
    ]
    close = nearest_in_time(fields, base + timedelta(minutes=7))
    assert close is not None and close.valid_time == base + timedelta(minutes=6)

    far = nearest_in_time(fields, base + timedelta(hours=5))
    assert far is None


def test_align_pair_regrids_forecast_onto_obs_grid():
    # Same resolution, slightly different origin — like the real fenglei/radar pair.
    forecast = GriddedField(
        data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        lat=np.array([29.98, 29.99, 30.00]),
        lon=np.array([110.00, 110.01, 110.02]),
        valid_time=datetime(2025, 4, 11, 2, 18),
        product="forecast",
        units="dBZ",
    )
    obs = GriddedField(
        data=np.array([[10.0, 20.0], [30.0, 40.0]]),
        lat=np.array([29.987, 29.997]),
        lon=np.array([110.007, 110.017]),
        valid_time=datetime(2025, 4, 11, 2, 18),
        product="obs",
        units="dBZ",
    )

    forecast_on_grid, obs_crop = align_pair(
        forecast, obs, lat_min=29.9, lat_max=30.0, lon_min=110.0, lon_max=110.02
    )

    assert forecast_on_grid.data.shape == obs_crop.data.shape
    np.testing.assert_array_equal(forecast_on_grid.lat, obs_crop.lat)
    np.testing.assert_array_equal(forecast_on_grid.lon, obs_crop.lon)
    # nearest neighbor: obs (29.987, 110.007) -> nearest forecast (29.99, 110.01) = 5.0
    assert forecast_on_grid.data[0, 0] == 5.0


def test_regrid_nearest_picks_closest_source_cell():
    source = GriddedField(
        data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        lat=np.array([10.0, 20.0]),
        lon=np.array([100.0, 200.0]),
        valid_time=datetime(2025, 1, 1),
        product="src",
        units="mm",
    )
    out = regrid_nearest(source, target_lat=np.array([9.0, 21.0]), target_lon=np.array([101.0, 199.0]))
    np.testing.assert_array_equal(out, np.array([[1.0, 2.0], [3.0, 4.0]]))

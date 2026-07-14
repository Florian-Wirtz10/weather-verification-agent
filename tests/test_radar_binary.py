from datetime import datetime
from pathlib import Path
import bz2
import struct

import numpy as np
import pytest

from data_io.radar_binary import read_cref_binary, read_cref_binary_header


def _write_binary_cref(
    tmp_path: Path,
    *,
    valid="20260407_185400",
    generated="20260407190012",
) -> Path:
    nx, ny = 4, 3
    # Stored north-to-south.  The minimum int16 is the only missing sentinel;
    # other negative reflectivity encodings remain finite below thresholds.
    raw_grid = np.array(
        [
            [-32768, -1280, 100, 200],
            [-1280, 50, 150, 250],
            [0, 10, 20, 30],
        ],
        dtype="<i2",
    )
    payload = bz2.compress(raw_grid.tobytes())
    name = f"Z_RADA_C_BABJ_{generated}_P_DOR_ACHN_CREF_{valid}.bin"
    path = tmp_path / name
    header = bytearray(256)
    header[0:4] = b"MOC\0"
    struct.pack_into("<i", header, 8, 256 + len(payload))
    struct.pack_into("<hh", header, 12, 1129, 3)
    header[16:24] = b"CREF\0\0\0\0"
    header[24:47] = b"Composite Refelectivity"
    struct.pack_into("<iii", header, 88, 256, len(payload), 0)
    valid_time = datetime.strptime(valid, "%Y%m%d_%H%M%S")
    struct.pack_into(
        "<6h",
        header,
        100,
        valid_time.year,
        valid_time.month,
        valid_time.day,
        valid_time.hour,
        valid_time.minute,
        valid_time.second,
    )
    struct.pack_into("<4i", header, 124, 28000, 116000, 30000, 120000)
    struct.pack_into("<2i", header, 140, 118, 29)
    struct.pack_into("<4i", header, 148, nx, ny, 100, 100)
    struct.pack_into("<2h", header, 164, 0, 1)
    struct.pack_into("<2i", header, 168, 188, raw_grid.nbytes)
    struct.pack_into("<h", header, 176, 10)
    path.write_bytes(header + payload)
    return path


def test_binary_cref_header_and_beijing_time_normalization(tmp_path):
    path = _write_binary_cref(tmp_path)

    header = read_cref_binary_header(path)

    assert header.valid_time_bjt == datetime(2026, 4, 7, 18, 54)
    assert header.valid_time_utc == datetime(2026, 4, 7, 10, 54)
    assert header.generated_time_bjt == datetime(2026, 4, 7, 19, 0, 12)
    assert header.raw_timezone_seconds == 0
    assert (header.nx, header.ny) == (4, 3)
    assert (header.center_lon, header.center_lat) == (118.0, 29.0)


def test_binary_cref_decodes_scales_flips_and_crops(tmp_path):
    path = _write_binary_cref(tmp_path)

    field = read_cref_binary(path, bbox=(28.0, 30.0, 116.0, 120.0))

    assert field.data.shape == (3, 4)
    np.testing.assert_allclose(field.lat, [28.0, 29.0, 30.0])
    np.testing.assert_allclose(field.lon, [116.0, 117.3333333333, 118.6666666667, 120.0])
    np.testing.assert_allclose(
        field.data,
        [[0.0, 1.0, 2.0, 3.0], [-128.0, 5.0, 15.0, 25.0], [np.nan, -128.0, 10.0, 20.0]],
        equal_nan=True,
    )


def test_binary_cref_rejects_header_filename_time_disagreement(tmp_path):
    path = _write_binary_cref(tmp_path)
    raw = bytearray(path.read_bytes())
    struct.pack_into("<h", raw, 108, 48)
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="header/filename valid time mismatch"):
        read_cref_binary_header(path)

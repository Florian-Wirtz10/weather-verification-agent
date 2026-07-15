"""Reader for CMA MOC weather-radar mosaic V3.0 CREF binary products.

The operational files contain a 256-byte little-endian product header followed
by a (usually bzip2-compressed) ``int16`` raster.  Grid values are divided by
the header scale to recover dBZ.  The storage order is north-to-south and is
flipped here to the project's common south-to-north convention.

For the 2026-04-07 delivery, ``TimeZone`` is zero and both timestamps in the
filename and product header are UTC.  ``BABJ`` identifies the issuing centre;
it does not make the timestamps Beijing Time.  The parser requires the header
and filename clocks to agree and rejects a non-zero timezone instead of
silently applying an unsupported offset.
"""

from __future__ import annotations

import bz2
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import struct

import numpy as np

from data_io.grid import GriddedField

_HEADER_BYTES = 256
_MISSING_RAW = np.iinfo(np.int16).min
_FILENAME_RE = re.compile(
    r"Z_RADA_C_BABJ_(?P<generated>\d{14})_P_DOR_(?P<region>[A-Z0-9]+)_"
    r"(?P<product>[A-Z0-9]+)_(?P<date>\d{8})_(?P<time>\d{6})\.bin\Z"
)


@dataclass(frozen=True)
class CrefBinaryHeader:
    """Validated metadata from one MOC CREF V3.0 product header."""

    path: Path
    version: str
    file_bytes: int
    mosaic_id: int
    coordinate_type: int
    product: str
    description: str
    block_position: int
    block_bytes: int
    raw_timezone_seconds: int
    valid_time_utc: datetime
    generated_time_utc: datetime
    south: float
    west: float
    north: float
    east: float
    center_lon: float
    center_lat: float
    nx: int
    ny: int
    dx: float
    dy: float
    compression: int
    radar_count: int
    uncompressed_bytes: int
    scale: int


def _text(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="strict").strip()


def _decode_center(raw_value: int, *, longitude: bool) -> float:
    """Handle the operational header's unscaled centre-coordinate variant."""
    limit = 180 if longitude else 90
    return float(raw_value) if abs(raw_value) <= limit else raw_value / 1000.0


def _parse_filename(path: Path) -> tuple[datetime, datetime, str, str]:
    match = _FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"{path}: unsupported radar binary filename")
    generated_utc = datetime.strptime(match.group("generated"), "%Y%m%d%H%M%S")
    valid_utc = datetime.strptime(
        match.group("date") + match.group("time"), "%Y%m%d%H%M%S"
    )
    return generated_utc, valid_utc, match.group("region"), match.group("product")


def read_cref_binary_header(path) -> CrefBinaryHeader:
    """Read and validate the fixed header without decompressing the grid."""
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(_HEADER_BYTES)
    if len(raw) != _HEADER_BYTES:
        raise ValueError(f"{path}: truncated radar header ({len(raw)} of 256 bytes)")

    label = _text(raw[0:4])
    version = _text(raw[4:8])
    file_bytes = struct.unpack_from("<i", raw, 8)[0]
    mosaic_id, coordinate_type = struct.unpack_from("<hh", raw, 12)
    product = _text(raw[16:24])
    description = _text(raw[24:88])
    block_position, block_bytes, raw_timezone_seconds = struct.unpack_from("<iii", raw, 88)
    year, month, day, hour, minute, second = struct.unpack_from("<6h", raw, 100)
    south_raw, west_raw, north_raw, east_raw = struct.unpack_from("<4i", raw, 124)
    center_lon_raw, center_lat_raw = struct.unpack_from("<2i", raw, 140)
    nx, ny, dx_raw, dy_raw = struct.unpack_from("<4i", raw, 148)
    _height, compression = struct.unpack_from("<2h", raw, 164)
    radar_count, uncompressed_bytes = struct.unpack_from("<2i", raw, 168)
    scale = struct.unpack_from("<h", raw, 176)[0]

    generated_utc, filename_valid_utc, region, filename_product = _parse_filename(path)
    try:
        header_valid_utc = datetime(year, month, day, hour, minute, second)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid observation clock in radar header") from exc

    actual_file_bytes = path.stat().st_size
    errors = []
    if label != "MOC":
        errors.append(f"label={label!r}, expected 'MOC'")
    if product != "CREF" or filename_product != "CREF":
        errors.append(f"product={product!r}/{filename_product!r}, expected CREF")
    if region != "ACHN":
        errors.append(f"region={region!r}, expected ACHN")
    if coordinate_type != 3:
        errors.append(f"coordinate_type={coordinate_type}, expected regular lat/lon (3)")
    if file_bytes != actual_file_bytes:
        errors.append(f"FileBytes={file_bytes}, actual={actual_file_bytes}")
    if block_position != _HEADER_BYTES:
        errors.append(f"BlockPos={block_position}, expected {_HEADER_BYTES}")
    if block_bytes != actual_file_bytes - block_position:
        errors.append(
            f"BlockLen={block_bytes}, actual payload={actual_file_bytes - block_position}"
        )
    if raw_timezone_seconds != 0:
        errors.append(
            f"unsupported TimeZone={raw_timezone_seconds}; delivered CREF clock must be UTC"
        )
    if header_valid_utc != filename_valid_utc:
        errors.append(
            "header/filename valid time mismatch: "
            f"{header_valid_utc} versus {filename_valid_utc} UTC"
        )
    if nx <= 0 or ny <= 0 or dx_raw <= 0 or dy_raw <= 0:
        errors.append(f"invalid grid nx={nx}, ny={ny}, dx={dx_raw}, dy={dy_raw}")
    expected_uncompressed = nx * ny * np.dtype("<i2").itemsize
    if uncompressed_bytes != expected_uncompressed:
        errors.append(
            f"UnZipBytes={uncompressed_bytes}, expected {expected_uncompressed}"
        )
    if compression not in {0, 1}:
        errors.append(f"unsupported compression={compression}")
    if scale <= 0:
        errors.append(f"invalid scale={scale}")
    if errors:
        raise ValueError(f"{path}: invalid radar binary contract: {'; '.join(errors)}")

    return CrefBinaryHeader(
        path=path,
        version=version,
        file_bytes=file_bytes,
        mosaic_id=mosaic_id,
        coordinate_type=coordinate_type,
        product=product,
        description=description,
        block_position=block_position,
        block_bytes=block_bytes,
        raw_timezone_seconds=raw_timezone_seconds,
        valid_time_utc=filename_valid_utc,
        generated_time_utc=generated_utc,
        south=south_raw / 1000.0,
        west=west_raw / 1000.0,
        north=north_raw / 1000.0,
        east=east_raw / 1000.0,
        center_lon=_decode_center(center_lon_raw, longitude=True),
        center_lat=_decode_center(center_lat_raw, longitude=False),
        nx=nx,
        ny=ny,
        dx=dx_raw / 10000.0,
        dy=dy_raw / 10000.0,
        compression=compression,
        radar_count=radar_count,
        uncompressed_bytes=uncompressed_bytes,
        scale=scale,
    )


def read_cref_binary_valid_time(path) -> datetime:
    """Return the verified UTC observation time without loading the raster."""
    return read_cref_binary_header(path).valid_time_utc


def _coordinate_slice(coords: np.ndarray, minimum: float, maximum: float) -> slice:
    start = int(np.searchsorted(coords, minimum, side="left"))
    stop = int(np.searchsorted(coords, maximum, side="right"))
    if start >= stop:
        raise ValueError(
            f"requested range {minimum}..{maximum} does not intersect "
            f"{coords[0]}..{coords[-1]}"
        )
    return slice(start, stop)


def read_cref_binary(path, *, bbox=None) -> GriddedField:
    """Decode one CREF binary product, optionally retaining only ``bbox``.

    ``bbox`` is ``(lat_min, lat_max, lon_min, lon_max)``.  The compressed body
    must still be decompressed as one bzip2 stream, but cropping before float
    conversion avoids retaining a 6200 x 4200 float array in memory.
    """
    header = read_cref_binary_header(path)
    with header.path.open("rb") as handle:
        handle.seek(header.block_position)
        payload = handle.read(header.block_bytes)
    try:
        decoded = bz2.decompress(payload) if header.compression == 1 else payload
    except OSError as exc:
        raise ValueError(f"{header.path}: radar payload decompression failed") from exc
    if len(decoded) != header.uncompressed_bytes:
        raise ValueError(
            f"{header.path}: decoded {len(decoded)} bytes, "
            f"expected {header.uncompressed_bytes}"
        )

    raw = np.frombuffer(decoded, dtype="<i2", count=header.nx * header.ny).reshape(
        header.ny, header.nx
    )
    # Stored north-to-south; normalize to ascending latitude.
    raw = raw[::-1, :]
    # The header describes outer domain limits.  Use both limits so the binary
    # reader reproduces the coordinate convention of the delivered CREF
    # NetCDF products (6200 points from 73 to 135, 4200 from 12.2 to 54.2).
    lon = np.linspace(header.west, header.east, header.nx, dtype=np.float64)
    lat = np.linspace(header.south, header.north, header.ny, dtype=np.float64)

    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = map(float, bbox)
        if lat_min >= lat_max or lon_min >= lon_max:
            raise ValueError(f"invalid bbox: {bbox!r}")
        y_slice = _coordinate_slice(lat, lat_min, lat_max)
        x_slice = _coordinate_slice(lon, lon_min, lon_max)
        raw = raw[y_slice, x_slice]
        lat = lat[y_slice]
        lon = lon[x_slice]

    values = raw.astype(np.float32) / float(header.scale)
    values[raw == _MISSING_RAW] = np.nan
    return GriddedField(
        data=values,
        lat=lat,
        lon=lon,
        valid_time=header.valid_time_utc,
        product="CREF_obs_v3",
        units="dBZ",
        init_time=header.valid_time_utc,
        lead_minutes=0,
    )

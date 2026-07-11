"""Reader for 风雷 SWAN/diamond-131-style mosaic products (REFMOSAIC, V11-PRE, ...).

File format: a 1024-byte binary header (`SwanHdr` below) followed by a raw
grid of scaled integers, the whole thing bz2-compressed.

The header's own `m_iDataType` field is not reliable — real 风雷 samples
(both REFMOSAIC and V11-PRE) carry `m_iDataType == 0`, which the format
notionally reserves for "unspecified". The per-cell byte width is instead
inferred from the file size: `remaining_bytes / (nx * ny)` after the header,
which reproduces the known-correct widths (1 byte for REFMOSAIC reflectivity,
2 bytes for V11-PRE precipitation) without trusting that field.
"""

import bz2
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from data_io.grid import GriddedField

SwanHdr = np.dtype(
    {
        "names": [
            "ZoneName", "DataName", "Flag", "Version",
            "year", "month", "day", "hour", "minute", "interval",
            "XNumGrids", "YNumGrids", "ZNumGrids", "RadarCount",
            "StartLon", "StartLat", "CenterLon", "CenterLat",
            "XReso", "YReso", "ZHighGrids", "RadarStationName",
            "RadarLongitude", "RadarLatitude", "RadarAltitude", "MosaicFlag",
            "m_iDataType", "m_iLevelDimension", "offset", "scale", "Reserved",
        ],
        "formats": [
            "12c", "38c", "8c", "8c",
            "h", "h", "h", "h", "h", "h",
            "h", "h", "h", "i",
            "f", "f", "f", "f", "f", "f",
            "40f", "320c", "20f", "20f", "20f", "20c",
            "H", "H", "f", "f", "160c",
        ],
    }
)

# byte width -> numpy dtype for the raw (unscaled) data body.
_WIDTH_TO_DTYPE = {1: np.dtype("u1"), 2: np.dtype("<i2"), 4: np.dtype("<f4")}


@dataclass(frozen=True)
class SwanMosaicMetadata:
    """Small, header-only description used to filter files before full decoding."""

    init_time: datetime
    valid_time: datetime
    lead_minutes: int
    product: str


def _decode_text(field):
    """Decode a fixed-width char[] header field (GBK, NUL-padded) to str."""
    return bytes(field).split(b"\x00")[0].decode("gbk", errors="ignore")


def _read_header(path):
    """Read only the fixed-size SWAN header from a compressed mosaic file."""
    with bz2.open(Path(path), "rb") as f:
        raw_header = f.read(SwanHdr.itemsize)
    if len(raw_header) != SwanHdr.itemsize:
        raise ValueError(
            f"{path}: truncated SWAN header ({len(raw_header)} of {SwanHdr.itemsize} bytes)"
        )
    return np.frombuffer(raw_header, dtype=SwanHdr)[0]


def _metadata_from_header(header) -> SwanMosaicMetadata:
    init_time = datetime(
        int(header["year"]),
        int(header["month"]),
        int(header["day"]),
        int(header["hour"]),
        int(header["minute"]),
    )
    lead_minutes = int(header["interval"])
    product = _decode_text(header["DataName"]) or _decode_text(header["ZoneName"])
    return SwanMosaicMetadata(
        init_time=init_time,
        valid_time=init_time + timedelta(minutes=lead_minutes),
        lead_minutes=lead_minutes,
        product=product,
    )


def read_swan_mosaic_metadata(path) -> SwanMosaicMetadata:
    """Read forecast identity and timing without decompressing the full grid."""
    return _metadata_from_header(_read_header(path))


def read_swan_mosaic_valid_time(path) -> datetime:
    """Read a forecast's UTC valid time without decompressing its full grid."""
    return read_swan_mosaic_metadata(path).valid_time


def read_swan_mosaic(path):
    """Read one bz2-compressed SWAN/diamond-131 mosaic file into a GriddedField.

    Returns a GriddedField with `data` scaled to physical units (dBZ for
    REFMOSAIC, mm/interval for V11-PRE), south-to-north / west-to-east.
    """
    path = Path(path)

    with bz2.open(path, "rb") as f:
        raw = f.read()

    header = np.frombuffer(raw[: SwanHdr.itemsize], dtype=SwanHdr)[0]
    body = raw[SwanHdr.itemsize :]

    nx = int(header["XNumGrids"])
    ny = int(header["YNumGrids"])
    nz = int(header["ZNumGrids"]) or 1
    cell_count = nx * ny * nz

    width, remainder = divmod(len(body), cell_count)
    if remainder != 0 or width not in _WIDTH_TO_DTYPE:
        raise ValueError(
            f"{path}: can't infer a whole-byte-per-cell dtype from body size "
            f"{len(body)} over {cell_count} cells (grid {nx}x{ny}x{nz})"
        )
    dtype = _WIDTH_TO_DTYPE[width]

    raw_values = np.frombuffer(body, dtype=dtype, count=cell_count).reshape(nz, ny, nx)
    raw_values = raw_values[0]  # drop the singleton z dimension

    scale = float(header["scale"])
    offset = float(header["offset"])
    values = raw_values.astype(np.float32) * scale + offset

    # Stored north-to-south; flip to south-to-north so lat ends up ascending.
    values = np.flipud(values)

    start_lon = float(header["StartLon"])
    start_lat = float(header["StartLat"])
    dx = float(header["XReso"])
    dy = float(header["YReso"])
    lon = start_lon + dx * np.arange(nx, dtype=np.float64)
    lat_north_to_south = start_lat - dy * np.arange(ny, dtype=np.float64)
    lat = lat_north_to_south[::-1]

    init_time = datetime(
        int(header["year"]), int(header["month"]), int(header["day"]),
        int(header["hour"]), int(header["minute"]),
    )
    lead_minutes = int(header["interval"])
    valid_time = init_time + timedelta(minutes=lead_minutes)

    product = _decode_text(header["DataName"]) or _decode_text(header["ZoneName"])
    units = "dBZ" if "REF" in product.upper() else "mm"

    return GriddedField(
        data=values,
        lat=lat,
        lon=lon,
        valid_time=valid_time,
        product=product,
        units=units,
        init_time=init_time,
        lead_minutes=lead_minutes,
    )

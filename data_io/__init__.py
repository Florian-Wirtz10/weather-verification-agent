"""Data access layer.

Loads synthetic and real forecast/observation fields and normalizes them into
the array shapes the verification package expects. Owns file-format and I/O
concerns; request interpretation stays in ``agent``.
"""

from data_io.fenglei_mosaic import (
    SwanMosaicMetadata,
    read_swan_mosaic,
    read_swan_mosaic_metadata,
    read_swan_mosaic_valid_time,
)
from data_io.grid import GriddedField
from data_io.radar_obs import (
    index_cref_files,
    nearest_in_time,
    read_cref,
    read_cref_series,
    read_cref_valid_time,
)
from data_io.synthetic import generate_synthetic_fields

__all__ = [
    "generate_synthetic_fields",
    "GriddedField",
    "read_swan_mosaic",
    "read_swan_mosaic_metadata",
    "read_swan_mosaic_valid_time",
    "SwanMosaicMetadata",
    "read_cref",
    "read_cref_valid_time",
    "index_cref_files",
    "read_cref_series",
    "nearest_in_time",
]

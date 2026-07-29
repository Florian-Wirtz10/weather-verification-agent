"""Reader for the StormDiT NPZ radar-reflectivity forecasts.

The delivered files do not contain latitude/longitude arrays or a formal field
contract.  Phase 8 therefore keeps the independently verified interpretation
in one explicit contract object instead of scattering guessed constants through
the application.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re

import numpy as np

from data_io.grid import GriddedField


@dataclass(frozen=True)
class UpstreamModelContract:
    """Deterministic interpretation of one upstream NPZ delivery."""

    ny: int
    nx: int
    history_count: int
    prediction_count: int
    step_minutes: int
    north_lat: float
    west_lon: float
    resolution_degrees: float
    source_clock: str = "BJT"
    units: str = "dBZ"
    name: str = "phase8-inferred-v1"


# Verified by matching the delivered 2025 history frames to CREF observations:
# bilinear sampling on this grid gives about 0.994 mean spatial correlation.
# Value-zero/missing semantics still require an upstream declaration.
PHASE8_INFERRED_CONTRACT = UpstreamModelContract(
    ny=640,
    nx=720,
    history_count=17,
    prediction_count=60,
    step_minutes=6,
    north_lat=41.0,
    west_lon=101.4,
    resolution_degrees=0.03,
)

_BJT_TO_UTC = timedelta(hours=8)
_DELIVERY_DIRECTORY_RE = re.compile(r"(?P<time>\d{10})_BJT\Z")
_REQUIRED_KEYS = {
    "predicted_radar",
    "history_frames",
    "start_time",
    "history_times",
    "num_history",
    "num_prediction",
}


@dataclass(frozen=True)
class UpstreamModelMetadata:
    path: Path
    init_time_bjt: datetime
    init_time_utc: datetime
    history_times_bjt: tuple[datetime, ...]
    available_leads: tuple[int, ...]
    contract_name: str


def discover_upstream_model_files(directories) -> dict[datetime, Path]:
    """Index delivered NPZ files by UTC initialization without decoding arrays.

    Directory names are only used for fast discovery.  The selected archive is
    still fully validated against its internal ``start_time`` before scoring.
    """
    indexed: dict[datetime, Path] = {}
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/*.npz")):
            match = _DELIVERY_DIRECTORY_RE.fullmatch(path.parent.name)
            if match is None:
                continue
            init_bjt = datetime.strptime(match.group("time"), "%Y%m%d%H")
            init_utc = init_bjt - _BJT_TO_UTC
            if init_utc in indexed:
                raise ValueError(
                    f"duplicate upstream-model initialization {init_utc} UTC: "
                    f"{indexed[init_utc]} and {path}"
                )
            indexed[init_utc] = path
    return indexed


def _parse_bjt_timestamp(value, fmt: str) -> datetime:
    try:
        return datetime.strptime(str(value), fmt)
    except ValueError as exc:
        raise ValueError(f"invalid upstream-model BJT timestamp: {value!r}") from exc


def _validate_archive(archive, path: Path, contract: UpstreamModelContract):
    missing = _REQUIRED_KEYS.difference(archive.files)
    if missing:
        raise ValueError(f"{path}: missing NPZ fields: {sorted(missing)}")

    num_history = int(archive["num_history"])
    num_prediction = int(archive["num_prediction"])
    if (num_history, num_prediction) != (
        contract.history_count,
        contract.prediction_count,
    ):
        raise ValueError(
            f"{path}: count metadata {(num_history, num_prediction)} does not match "
            f"contract {(contract.history_count, contract.prediction_count)}"
        )

    expected_history_shape = (contract.history_count, contract.ny, contract.nx)
    expected_prediction_shape = (
        contract.history_count + contract.prediction_count,
        contract.ny,
        contract.nx,
    )
    history = archive["history_frames"]
    predicted = archive["predicted_radar"]
    if history.shape != expected_history_shape:
        raise ValueError(
            f"{path}: history_frames shape {history.shape} != {expected_history_shape}"
        )
    if predicted.shape != expected_prediction_shape:
        raise ValueError(
            f"{path}: predicted_radar shape {predicted.shape} != {expected_prediction_shape}"
        )
    if not np.issubdtype(predicted.dtype, np.floating):
        raise ValueError(f"{path}: predicted_radar must be floating point")
    return history, predicted


def _metadata_from_archive(
    archive,
    path: Path,
    contract: UpstreamModelContract,
) -> tuple[UpstreamModelMetadata, np.ndarray]:
    _, predicted = _validate_archive(archive, path, contract)
    if contract.source_clock != "BJT":
        raise ValueError(f"unsupported upstream-model source clock: {contract.source_clock}")

    init_bjt = _parse_bjt_timestamp(archive["start_time"], "%Y-%m-%d %H:%M:%S")
    history_times = tuple(
        _parse_bjt_timestamp(value, "%Y%m%d%H%M") for value in archive["history_times"]
    )
    expected_history_times = tuple(
        init_bjt
        - timedelta(minutes=contract.step_minutes * offset)
        for offset in range(contract.history_count - 1, -1, -1)
    )
    if history_times != expected_history_times:
        raise ValueError(f"{path}: history_times do not end at start_time on the native cadence")

    leads = tuple(
        contract.step_minutes * index
        for index in range(1, contract.prediction_count + 1)
    )
    return (
        UpstreamModelMetadata(
            path=path,
            init_time_bjt=init_bjt,
            init_time_utc=init_bjt - _BJT_TO_UTC,
            history_times_bjt=history_times,
            available_leads=leads,
            contract_name=contract.name,
        ),
        predicted,
    )


def read_upstream_model_metadata(
    path,
    *,
    contract: UpstreamModelContract = PHASE8_INFERRED_CONTRACT,
) -> UpstreamModelMetadata:
    """Validate an NPZ delivery and return its inferred timing metadata."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata, _ = _metadata_from_archive(archive, path, contract)
        return metadata


def read_upstream_model_forecast(
    path,
    lead_minutes: int,
    *,
    contract: UpstreamModelContract = PHASE8_INFERRED_CONTRACT,
) -> GriddedField:
    """Read one future frame as an ascending-latitude :class:`GriddedField`.

    ``predicted_radar`` contains 17 reconstructed history frames followed by
    60 future frames.  Consequently array index 17 is +6 minutes and index 76
    is +360 minutes under the phase-8 inferred contract.
    """
    return read_upstream_model_forecasts(
        path, (lead_minutes,), contract=contract
    )[int(lead_minutes)]


def read_upstream_model_forecasts(
    path,
    leads,
    *,
    contract: UpstreamModelContract = PHASE8_INFERRED_CONTRACT,
) -> dict[int, GriddedField]:
    """Read several leads while decompressing the large NPZ array only once."""
    path = Path(path)
    normalized_leads = tuple(dict.fromkeys(leads))
    for lead_minutes in normalized_leads:
        if (
            isinstance(lead_minutes, bool)
            or not isinstance(lead_minutes, (int, np.integer))
            or lead_minutes <= 0
            or lead_minutes % contract.step_minutes
            or lead_minutes > contract.prediction_count * contract.step_minutes
        ):
            raise ValueError(
                f"lead_minutes must be one native step between {contract.step_minutes} and "
                f"{contract.prediction_count * contract.step_minutes}"
            )

    with np.load(path, allow_pickle=False) as archive:
        metadata, predicted = _metadata_from_archive(archive, path, contract)

    lat_descending = contract.north_lat - (
        np.arange(contract.ny, dtype=np.float64) * contract.resolution_degrees
    )
    lat = lat_descending[::-1]
    lon = contract.west_lon + (
        np.arange(contract.nx, dtype=np.float64) * contract.resolution_degrees
    )
    fields = {}
    for lead_minutes in normalized_leads:
        lead_minutes = int(lead_minutes)
        frame_index = (
            contract.history_count + lead_minutes // contract.step_minutes - 1
        )
        # Delivered rows run north to south.  GriddedField requires ascending
        # latitude, so flip only the latitude dimension.
        data = np.asarray(predicted[frame_index], dtype=np.float32)[::-1, :].copy()
        fields[lead_minutes] = GriddedField(
            data=data,
            lat=lat,
            lon=lon,
            valid_time=metadata.init_time_utc + timedelta(minutes=lead_minutes),
            product="StormDiT reflectivity",
            units=contract.units,
            init_time=metadata.init_time_utc,
            lead_minutes=lead_minutes,
        )
    return fields

"""Fast, auditable file catalogs for full-day forecast verification."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

from data_io.fenglei_mosaic import read_swan_mosaic_metadata
from data_io.radar_binary import read_cref_binary_header

CATALOG_VERSION = 1
_BEIJING_OFFSET = timedelta(hours=8)
_FORECAST_RE = re.compile(
    r"Z_SEVP_C_BABJ_(?P<generated>\d{14})_P_FENGL(?P<v11>-V11)?-REFMOSAIC-"
    r"(?P<init>\d{12})-(?P<lead>\d{3})MIN\.bz2\Z"
)
_OBS_RE = re.compile(
    r"Z_RADA_C_BABJ_(?P<generated>\d{14})_P_DOR_ACHN_CREF_"
    r"(?P<date>\d{8})_(?P<time>\d{6})\.bin\Z"
)
DuplicatePolicy = Literal["latest_generation", "earliest_generation", "error"]


@dataclass(frozen=True)
class ForecastRecord:
    path: Path
    init_time_utc: datetime
    valid_time_utc: datetime
    init_time_bjt: datetime
    valid_time_bjt: datetime
    lead_minutes: int
    generated_label_time: datetime
    filename_clock: str


@dataclass(frozen=True)
class ObservationRecord:
    path: Path
    valid_time_utc: datetime
    valid_time_bjt: datetime
    generated_time_bjt: datetime


@dataclass(frozen=True)
class DuplicateObservation:
    valid_time_bjt: datetime
    selected: Path
    discarded: tuple[Path, ...]
    policy: str


@dataclass(frozen=True)
class DailyCatalog:
    case_date_bjt: date
    forecasts: tuple[ForecastRecord, ...]
    observations: tuple[ObservationRecord, ...]
    duplicates: tuple[DuplicateObservation, ...]
    forecast_fingerprint: str
    observation_fingerprint: str
    cache_hit: bool

    @property
    def observation_by_valid_time(self) -> dict[datetime, ObservationRecord]:
        return {record.valid_time_utc: record for record in self.observations}

    @property
    def matched_forecasts(self) -> tuple[ForecastRecord, ...]:
        times = set(self.observation_by_valid_time)
        return tuple(record for record in self.forecasts if record.valid_time_utc in times)


def _files_and_fingerprint(root: Path, pattern: str) -> tuple[list[Path], str]:
    if not root.is_dir():
        raise ValueError(f"data directory does not exist: {root}")
    paths = sorted(path for path in root.rglob(pattern) if path.is_file())
    digest = hashlib.sha256()
    digest.update(f"catalog-v{CATALOG_VERSION}\0{root.resolve()}\0".encode())
    for path in paths:
        stat = path.stat()
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    return paths, digest.hexdigest()


def _parse_forecasts(paths: list[Path], case_date_bjt: date) -> list[ForecastRecord]:
    records = []
    identities: dict[tuple[datetime, int], Path] = {}
    for path in paths:
        match = _FORECAST_RE.fullmatch(path.name)
        if match is None:
            continue
        filename_init = datetime.strptime(match.group("init"), "%Y%m%d%H%M")
        lead = int(match.group("lead"))
        # V11 filenames use Beijing Time while their SWAN headers use UTC.
        # The older FENGL-REFMOSAIC delivery uses UTC in both filename/header.
        if match.group("v11"):
            init_bjt = filename_init
            init_utc = filename_init - _BEIJING_OFFSET
            filename_clock = "BJT"
        else:
            init_utc = filename_init
            init_bjt = filename_init + _BEIJING_OFFSET
            filename_clock = "UTC"
        if init_bjt.date() != case_date_bjt:
            continue
        valid_utc = init_utc + timedelta(minutes=lead)
        identity = (init_utc, lead)
        if identity in identities:
            raise ValueError(
                f"duplicate forecast init/lead {init_utc} UTC +{lead}: "
                f"{identities[identity]} and {path}"
            )
        identities[identity] = path
        records.append(
            ForecastRecord(
                path=path,
                init_time_utc=init_utc,
                valid_time_utc=valid_utc,
                init_time_bjt=init_bjt,
                valid_time_bjt=valid_utc + _BEIJING_OFFSET,
                lead_minutes=lead,
                generated_label_time=datetime.strptime(
                    match.group("generated"), "%Y%m%d%H%M%S"
                ),
                filename_clock=filename_clock,
            )
        )
    records.sort(key=lambda item: (item.valid_time_utc, item.init_time_utc, item.lead_minutes))
    return records


def _parse_observations(
    paths: list[Path], case_date_bjt: date, policy: DuplicatePolicy
) -> tuple[list[ObservationRecord], list[DuplicateObservation]]:
    groups: dict[datetime, list[ObservationRecord]] = defaultdict(list)
    for path in paths:
        match = _OBS_RE.fullmatch(path.name)
        if match is None:
            continue
        valid_bjt = datetime.strptime(
            match.group("date") + match.group("time"), "%Y%m%d%H%M%S"
        )
        if valid_bjt.date() != case_date_bjt:
            continue
        record = ObservationRecord(
            path=path,
            valid_time_utc=valid_bjt - _BEIJING_OFFSET,
            valid_time_bjt=valid_bjt,
            generated_time_bjt=datetime.strptime(match.group("generated"), "%Y%m%d%H%M%S"),
        )
        groups[valid_bjt].append(record)

    selected = []
    duplicates = []
    for valid_bjt, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: (item.generated_time_bjt, item.path.name))
        if len(ordered) > 1 and policy == "error":
            files = ", ".join(str(item.path) for item in ordered)
            raise ValueError(f"duplicate observations for {valid_bjt} BJT: {files}")
        chosen = ordered[0] if policy == "earliest_generation" else ordered[-1]
        selected.append(chosen)
        if len(ordered) > 1:
            duplicates.append(
                DuplicateObservation(
                    valid_time_bjt=valid_bjt,
                    selected=chosen.path,
                    discarded=tuple(item.path for item in ordered if item is not chosen),
                    policy=policy,
                )
            )
    return selected, duplicates


def _record_to_json(record) -> dict:
    values = asdict(record)
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, datetime):
            values[key] = value.isoformat()
    return values


def _save_cache(
    path: Path,
    *,
    case_date_bjt: date,
    forecast_root: Path,
    observation_root: Path,
    forecast_fingerprint: str,
    observation_fingerprint: str,
    forecasts: list[ForecastRecord],
    observation_candidates: list[Path],
) -> None:
    payload = {
        "version": CATALOG_VERSION,
        "case_date_bjt": case_date_bjt.isoformat(),
        "forecast_root": str(forecast_root.resolve()),
        "observation_root": str(observation_root.resolve()),
        "forecast_fingerprint": forecast_fingerprint,
        "observation_fingerprint": observation_fingerprint,
        "forecasts": [_record_to_json(record) for record in forecasts],
        # Keep every observation candidate; duplicate policy is applied at load
        # time so switching policy never requires rebuilding the file index.
        "observation_candidates": [str(item) for item in observation_candidates],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_forecasts(payload: dict) -> list[ForecastRecord]:
    records = []
    for item in payload["forecasts"]:
        records.append(
            ForecastRecord(
                path=Path(item["path"]),
                init_time_utc=datetime.fromisoformat(item["init_time_utc"]),
                valid_time_utc=datetime.fromisoformat(item["valid_time_utc"]),
                init_time_bjt=datetime.fromisoformat(item["init_time_bjt"]),
                valid_time_bjt=datetime.fromisoformat(item["valid_time_bjt"]),
                lead_minutes=int(item["lead_minutes"]),
                generated_label_time=datetime.fromisoformat(item["generated_label_time"]),
                filename_clock=item["filename_clock"],
            )
        )
    return records


def _validate_samples(
    forecasts: list[ForecastRecord], observations: list[ObservationRecord]
) -> None:
    """Cross-check a bounded sample against both binary headers."""
    forecast_samples = forecasts[:1] + forecasts[-1:] if forecasts else []
    obs_samples = observations[:1] + observations[-1:] if observations else []
    for record in forecast_samples:
        header = read_swan_mosaic_metadata(record.path)
        if (
            header.init_time != record.init_time_utc
            or header.valid_time != record.valid_time_utc
            or header.lead_minutes != record.lead_minutes
        ):
            raise ValueError(f"forecast filename/header mismatch: {record.path}")
    for record in obs_samples:
        header = read_cref_binary_header(record.path)
        if header.valid_time_utc != record.valid_time_utc:
            raise ValueError(f"observation filename/header mismatch: {record.path}")


def build_daily_catalog(
    forecast_dir,
    observation_dir,
    case_date_bjt: date | str,
    *,
    duplicate_policy: DuplicatePolicy = "latest_generation",
    cache_dir="outputs/cache/catalogs",
    validate_samples: bool = True,
) -> DailyCatalog:
    """Index a Beijing-calendar day without opening every compressed grid."""
    if isinstance(case_date_bjt, str):
        case_date_bjt = date.fromisoformat(case_date_bjt)
    if duplicate_policy not in {"latest_generation", "earliest_generation", "error"}:
        raise ValueError(f"unsupported duplicate policy: {duplicate_policy}")
    forecast_root = Path(forecast_dir)
    observation_root = Path(observation_dir)
    forecast_paths, forecast_fingerprint = _files_and_fingerprint(forecast_root, "*.bz2")
    observation_paths, observation_fingerprint = _files_and_fingerprint(
        observation_root, "*.bin"
    )

    cache_name = hashlib.sha256(
        f"{forecast_root.resolve()}\0{observation_root.resolve()}\0{case_date_bjt}".encode()
    ).hexdigest()[:20]
    cache_path = Path(cache_dir) / f"daily_catalog_{cache_name}.json"
    cache_hit = False
    forecasts: list[ForecastRecord]
    obs_candidates: list[Path]
    if cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            valid_cache = (
                payload.get("version") == CATALOG_VERSION
                and payload.get("case_date_bjt") == case_date_bjt.isoformat()
                and payload.get("forecast_fingerprint") == forecast_fingerprint
                and payload.get("observation_fingerprint") == observation_fingerprint
            )
            if not valid_cache:
                raise ValueError("stale catalog")
            forecasts = _load_forecasts(payload)
            obs_candidates = [Path(item) for item in payload["observation_candidates"]]
            cache_hit = True
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            forecasts = _parse_forecasts(forecast_paths, case_date_bjt)
            obs_candidates = observation_paths
    else:
        forecasts = _parse_forecasts(forecast_paths, case_date_bjt)
        obs_candidates = observation_paths

    if not cache_hit:
        _save_cache(
            cache_path,
            case_date_bjt=case_date_bjt,
            forecast_root=forecast_root,
            observation_root=observation_root,
            forecast_fingerprint=forecast_fingerprint,
            observation_fingerprint=observation_fingerprint,
            forecasts=forecasts,
            observation_candidates=obs_candidates,
        )

    observations, duplicates = _parse_observations(
        obs_candidates, case_date_bjt, duplicate_policy
    )
    if validate_samples:
        _validate_samples(forecasts, observations)
    return DailyCatalog(
        case_date_bjt=case_date_bjt,
        forecasts=tuple(forecasts),
        observations=tuple(observations),
        duplicates=tuple(duplicates),
        forecast_fingerprint=forecast_fingerprint,
        observation_fingerprint=observation_fingerprint,
        cache_hit=cache_hit,
    )

"""Unified deterministic catalog for Fenglei and StormDiT forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re

from data_io.radar_obs import index_radar_observation_files
from data_io.upstream_model import (
    PHASE8_INFERRED_CONTRACT,
    discover_upstream_model_files,
)

MODEL_STORMDIT = "stormdit"
MODEL_FENGLEI = "fenglei"
MODEL_LABELS = {
    MODEL_STORMDIT: "StormDiT",
    MODEL_FENGLEI: "风雷",
}
_FENGLEI_RE = re.compile(
    r"(?:FENGL(?P<v11>-V11)?-)?REFMOSAIC-"
    r"(?P<init>\d{12})-(?P<lead>\d{3})MIN\.bz2\Z"
)
_BJT_OFFSET = timedelta(hours=8)


def catalog_source_signature(
    *,
    stormdit_directories,
    fenglei_directory,
    observation_directory,
) -> tuple:
    """Return a lightweight, hashable version stamp for catalog source files.

    Streamlit caches the parsed catalog because indexing thousands of Fenglei
    files on every widget interaction is wasteful.  Directory paths alone are
    not a sufficient cache key, however: adding a new delivery under the same
    path would otherwise leave the old catalog alive.  Flat Fenglei/CREF
    deliveries are stamped by directory metadata and relevant file counts;
    the much smaller nested StormDiT delivery is stamped per NPZ archive.
    """

    def flat_stamp(directory, suffixes):
        root = Path(directory)
        if not root.is_dir():
            return (str(root), None, 0)
        suffixes = tuple(value.lower() for value in suffixes)
        count = sum(
            entry.is_file() and entry.name.lower().endswith(suffixes) for entry in root.iterdir()
        )
        return (str(root), root.stat().st_mtime_ns, count)

    stormdit_stamp = []
    for directory in stormdit_directories:
        root = Path(directory)
        if not root.is_dir():
            stormdit_stamp.append((str(root), None, ()))
            continue
        archives = []
        for path in sorted(root.glob("*/*.npz")):
            stat = path.stat()
            archives.append(
                (
                    str(path.relative_to(root)),
                    stat.st_size,
                    stat.st_mtime_ns,
                )
            )
        stormdit_stamp.append((str(root), root.stat().st_mtime_ns, tuple(archives)))

    return (
        tuple(stormdit_stamp),
        flat_stamp(fenglei_directory, (".bz2",)),
        flat_stamp(observation_directory, (".nc", ".bin")),
    )


@dataclass(frozen=True)
class ForecastRecord:
    model_id: str
    path: Path
    init_time_utc: datetime
    init_time_bjt: datetime
    lead_minutes: int
    valid_time_utc: datetime
    source_kind: str = "native"
    domain: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class MultiModelCatalog:
    observations: dict[datetime, Path]
    forecasts: dict[str, dict[tuple[datetime, int], ForecastRecord]]
    stormdit_contract: str

    def initializations(self, model_id: str) -> tuple[datetime, ...]:
        records = self.forecasts.get(model_id, {})
        return tuple(sorted({key[0] for key in records}))

    def matched_leads(self, model_id: str, init_time_utc: datetime) -> tuple[int, ...]:
        records = self.forecasts.get(model_id, {})
        return tuple(
            sorted(
                lead
                for (init_time, lead), record in records.items()
                if init_time == init_time_utc and record.valid_time_utc in self.observations
            )
        )

    def available_leads(
        self,
        model_ids,
        init_time_utc: datetime,
        *,
        common_only: bool,
    ) -> tuple[int, ...]:
        lead_sets = [set(self.matched_leads(model_id, init_time_utc)) for model_id in model_ids]
        if not lead_sets:
            return ()
        values = set.intersection(*lead_sets) if common_only else set.union(*lead_sets)
        return tuple(sorted(values))


def index_fenglei_forecasts(directory) -> dict[tuple[datetime, int], ForecastRecord]:
    records = {}
    for path in sorted(Path(directory).glob("*REFMOSAIC*MIN.bz2")):
        match = _FENGLEI_RE.search(path.name)
        if match is None:
            continue
        filename_init = datetime.strptime(match.group("init"), "%Y%m%d%H%M")
        init_utc = filename_init - _BJT_OFFSET if match.group("v11") else filename_init
        lead = int(match.group("lead"))
        key = (init_utc, lead)
        if key in records:
            raise ValueError(
                f"duplicate Fenglei initialization/lead {init_utc} UTC +{lead}: "
                f"{records[key].path} and {path}"
            )
        records[key] = ForecastRecord(
            model_id=MODEL_FENGLEI,
            path=path,
            init_time_utc=init_utc,
            init_time_bjt=init_utc + _BJT_OFFSET,
            lead_minutes=lead,
            valid_time_utc=init_utc + timedelta(minutes=lead),
        )
    return records


def index_stormdit_forecasts(directories) -> dict[tuple[datetime, int], ForecastRecord]:
    records = {}
    for init_utc, path in discover_upstream_model_files(directories).items():
        for lead in range(
            PHASE8_INFERRED_CONTRACT.step_minutes,
            PHASE8_INFERRED_CONTRACT.prediction_count * PHASE8_INFERRED_CONTRACT.step_minutes + 1,
            PHASE8_INFERRED_CONTRACT.step_minutes,
        ):
            records[(init_utc, lead)] = ForecastRecord(
                model_id=MODEL_STORMDIT,
                path=path,
                init_time_utc=init_utc,
                init_time_bjt=init_utc + _BJT_OFFSET,
                lead_minutes=lead,
                valid_time_utc=init_utc + timedelta(minutes=lead),
            )
    return records


def build_multimodel_catalog(
    *,
    stormdit_directories,
    fenglei_directory,
    observation_directory,
) -> MultiModelCatalog:
    return MultiModelCatalog(
        observations=index_radar_observation_files(observation_directory),
        forecasts={
            MODEL_STORMDIT: index_stormdit_forecasts(stormdit_directories),
            MODEL_FENGLEI: index_fenglei_forecasts(fenglei_directory),
        },
        stormdit_contract=PHASE8_INFERRED_CONTRACT.name,
    )


def discover_multimodel_initializations(
    catalog: MultiModelCatalog,
    model_ids,
    *,
    lead_range=(6, 180),
    common_only: bool,
) -> list[dict]:
    model_ids = tuple(model_ids)
    init_sets = [set(catalog.initializations(model_id)) for model_id in model_ids]
    if not init_sets:
        return []
    initializations = set.intersection(*init_sets) if common_only else set.union(*init_sets)
    start_lead, end_lead = map(int, lead_range)
    expected = tuple(range(max(6, start_lead), end_lead + 1, 6))
    cases = []
    for init_time in sorted(initializations):
        leads = tuple(
            lead
            for lead in catalog.available_leads(model_ids, init_time, common_only=common_only)
            if start_lead <= lead <= end_lead
        )
        if not leads:
            continue
        domains = {
            record.domain
            for model_id in model_ids
            for lead in leads
            if (
                record := catalog.forecasts.get(model_id, {}).get((init_time, lead))
            ) is not None
            if record.domain is not None
        }
        if len(domains) > 1:
            raise ValueError(
                f"initialization {init_time} has inconsistent forecast domains: {domains}"
            )
        cases.append(
            {
                "case_id": f"multimodel-{init_time:%Y%m%d%H%M}",
                "init_time": init_time,
                "init_time_bjt": init_time + _BJT_OFFSET,
                "available_leads": list(leads),
                "complete": leads == expected,
                "missing_leads": [lead for lead in expected if lead not in leads],
                "domain": list(next(iter(domains))) if domains else None,
            }
        )
    return cases

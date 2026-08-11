"""Strict reader for the delivered China CREF zero-shot benchmark package."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_io.forecast_sources import ForecastRecord, MODEL_STORMDIT, index_fenglei_forecasts
from data_io.grid import GriddedField, regrid_nearest
from data_io.fenglei_mosaic import read_swan_mosaic


BENCHMARK_MODEL_FRAMES = {
    "ConvLSTM": 12,
    "PredRNN": 12,
    "Earthformer": 12,
    "NowCastNet": 12,
    "SimVP": 12,
    "DiffCast": 20,
    "StormDiT": 60,
    "Fenglei": 30,
    "Persistence": 60,
}
PACKAGE_MODELS = (
    "ConvLSTM",
    "PredRNN",
    "Earthformer",
    "NowCastNet",
    "SimVP",
    "DiffCast",
    "StormDiT",
)
BENCHMARK_MODEL_LABELS = {**{model: model for model in BENCHMARK_MODEL_FRAMES}, "Fenglei": "风雷"}
BENCHMARK_STEP_MINUTES = 6
BENCHMARK_THRESHOLDS = (20.0, 35.0, 45.0)


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: int
    path: Path
    start_time_utc: datetime
    analysis_time_utc: datetime
    analysis_time_bjt: datetime


@dataclass(frozen=True)
class BenchmarkCatalog:
    package_dir: Path
    samples: tuple[BenchmarkSample, ...]
    models: tuple[str, ...]
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    resolution_degrees: float
    fenglei_forecasts: dict


def _parse_time(value) -> datetime:
    return datetime.fromisoformat(str(value))


def build_benchmark_catalog(package_dir, *, fenglei_directory=None) -> BenchmarkCatalog:
    """Validate the 24-sample Wuhan package without retaining forecast arrays."""
    package_dir = Path(package_dir)
    index_path = package_dir / "index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"benchmark index does not exist: {index_path}")
    index = pd.read_csv(index_path)
    required_columns = {
        "sample_id",
        "start_time_utc",
        "analysis_time_utc",
        "analysis_time_bjt",
        "truth_frames",
        "file",
    }
    missing = required_columns.difference(index.columns)
    if missing:
        raise ValueError(f"benchmark index is missing columns: {sorted(missing)}")

    samples = []
    common_models = None
    reference_lat = reference_lon = None
    for row in index.itertuples(index=False):
        path = package_dir / str(row.file)
        if not path.is_file():
            raise FileNotFoundError(f"benchmark sample does not exist: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required_fields = {
                "sample_id",
                "start_time",
                "analysis_time",
                "condition_frames",
                "frame_interval_minutes",
                "y_start",
                "x_start",
                "patch_size",
                "lat",
                "lon",
                "radar_condition_dbz",
                "radar_truth_dbz",
                "truth_lead_minutes",
                "truth_valid_time",
            }
            missing_fields = required_fields.difference(archive.files)
            if missing_fields:
                raise ValueError(f"{path}: missing fields {sorted(missing_fields)}")
            sample_id = int(archive["sample_id"])
            if sample_id != int(row.sample_id):
                raise ValueError(f"{path}: sample_id does not match index")
            start = _parse_time(archive["start_time"])
            analysis = _parse_time(archive["analysis_time"])
            if start != _parse_time(row.start_time_utc):
                raise ValueError(f"{path}: start_time does not match index")
            if analysis != _parse_time(row.analysis_time_utc):
                raise ValueError(f"{path}: analysis_time does not match index")
            if analysis - start != timedelta(minutes=96):
                raise ValueError(f"{path}: analysis_time must follow 17 conditions by 96 minutes")
            if _parse_time(row.analysis_time_bjt) != analysis + timedelta(hours=8):
                raise ValueError(f"{path}: BJT analysis time must equal UTC plus 8 hours")
            if int(archive["condition_frames"]) != 17:
                raise ValueError(f"{path}: condition_frames must be 17")
            if int(archive["frame_interval_minutes"]) != 6:
                raise ValueError(f"{path}: frame interval must be 6 minutes")
            if int(archive["patch_size"]) != 128:
                raise ValueError(f"{path}: patch_size must be 128")
            lat = np.asarray(archive["lat"], dtype=np.float64)
            lon = np.asarray(archive["lon"], dtype=np.float64)
            if lat.shape != (128,) or lon.shape != (128,):
                raise ValueError(f"{path}: benchmark coordinates must be length 128")
            if not np.allclose(np.diff(lat), -0.03, atol=1e-6):
                raise ValueError(f"{path}: latitude must decrease north-to-south by 0.03 degrees")
            if not np.allclose(np.diff(lon), 0.03, atol=1e-6):
                raise ValueError(f"{path}: longitude must increase west-to-east by 0.03 degrees")
            if archive["radar_condition_dbz"].shape != (17, 128, 128):
                raise ValueError(f"{path}: radar_condition_dbz must have shape (17,128,128)")
            if archive["radar_truth_dbz"].shape != (60, 128, 128):
                raise ValueError(f"{path}: radar_truth_dbz must have shape (60,128,128)")
            expected_truth_leads = np.arange(6, 361, 6)
            if not np.array_equal(archive["truth_lead_minutes"], expected_truth_leads):
                raise ValueError(f"{path}: truth lead axis must be +6..+360 minutes")
            truth_valid_times = tuple(_parse_time(value) for value in archive["truth_valid_time"])
            expected_valid_times = tuple(
                analysis + timedelta(minutes=int(lead)) for lead in expected_truth_leads
            )
            if truth_valid_times != expected_valid_times:
                raise ValueError(f"{path}: truth valid times do not match analysis time and leads")
            models = {
                key[5:-4]
                for key in archive.files
                if key.startswith("pred_") and key.endswith("_dbz")
            }
            for model in models:
                expected = BENCHMARK_MODEL_FRAMES.get(model)
                if expected is None:
                    raise ValueError(f"{path}: unsupported benchmark model {model}")
                if archive[f"pred_{model}_dbz"].shape != (expected, 128, 128):
                    raise ValueError(f"{path}: invalid {model} forecast shape")
                expected_leads = np.arange(6, expected * 6 + 1, 6)
                if not np.array_equal(archive[f"pred_{model}_lead_minutes"], expected_leads):
                    raise ValueError(f"{path}: invalid {model} lead axis")
            common_models = models if common_models is None else common_models.intersection(models)
            if reference_lat is None:
                reference_lat, reference_lon = lat, lon
            elif not np.array_equal(lat, reference_lat) or not np.array_equal(lon, reference_lon):
                raise ValueError(f"{path}: coordinates differ from the benchmark common grid")
        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                path=path,
                start_time_utc=start,
                analysis_time_utc=analysis,
                analysis_time_bjt=_parse_time(row.analysis_time_bjt),
            )
        )

    if not samples:
        raise ValueError(f"benchmark package contains no samples: {package_dir}")
    models = tuple(model for model in PACKAGE_MODELS if model in (common_models or set()))
    if models != PACKAGE_MODELS:
        raise ValueError(f"benchmark package does not contain all seven learning models: {models}")
    fenglei_forecasts = (
        index_fenglei_forecasts(fenglei_directory) if fenglei_directory is not None else {}
    )
    return BenchmarkCatalog(
        package_dir=package_dir,
        samples=tuple(samples),
        models=models + (("Fenglei",) if fenglei_forecasts else ()) + ("Persistence",),
        lat_range=(float(reference_lat[-1]), float(reference_lat[0])),
        lon_range=(float(reference_lon[0]), float(reference_lon[-1])),
        resolution_degrees=0.03,
        fenglei_forecasts=fenglei_forecasts,
    )


def index_benchmark_stormdit_forecasts(
    catalog: BenchmarkCatalog,
) -> dict[tuple[datetime, int], ForecastRecord]:
    """Expose benchmark-window StormDiT runs to the regular business catalog.

    These runs use the same StormDiT forecast target, but are available only on
    the delivered 128×128 Wuhan patch.  Keeping the domain on every record lets
    downstream verification enforce one common spatial support.
    """
    domain = (
        float(catalog.lat_range[0]),
        float(catalog.lat_range[1]),
        float(catalog.lon_range[0]),
        float(catalog.lon_range[1]),
    )
    records = {}
    for sample in catalog.samples:
        for lead in range(6, BENCHMARK_MODEL_FRAMES["StormDiT"] * 6 + 1, 6):
            key = (sample.analysis_time_utc, lead)
            records[key] = ForecastRecord(
                model_id=MODEL_STORMDIT,
                path=sample.path,
                init_time_utc=sample.analysis_time_utc,
                init_time_bjt=sample.analysis_time_bjt,
                lead_minutes=lead,
                valid_time_utc=sample.analysis_time_utc + timedelta(minutes=lead),
                source_kind="benchmark_stormdit",
                domain=domain,
            )
    return records


def read_benchmark_stormdit_forecasts(path, leads) -> dict[int, GriddedField]:
    """Read selected StormDiT future frames from one validated benchmark sample."""
    path = Path(path)
    normalized_leads = tuple(dict.fromkeys(int(value) for value in leads))
    valid_leads = set(range(6, BENCHMARK_MODEL_FRAMES["StormDiT"] * 6 + 1, 6))
    if not normalized_leads or any(lead not in valid_leads for lead in normalized_leads):
        raise ValueError("benchmark StormDiT leads must use +6..+360 minute steps")
    with np.load(path, allow_pickle=False) as archive:
        analysis = _parse_time(archive["analysis_time"])
        stored_leads = np.asarray(archive["pred_StormDiT_lead_minutes"], dtype=int)
        values = np.asarray(archive["pred_StormDiT_dbz"], dtype=np.float32)
        lat = np.asarray(archive["lat"], dtype=np.float64)
        lon = np.asarray(archive["lon"], dtype=np.float64)
    lead_to_index = {int(lead): index for index, lead in enumerate(stored_leads)}
    if any(lead not in lead_to_index for lead in normalized_leads):
        raise ValueError(f"{path}: requested StormDiT lead is absent from benchmark sample")
    if lat[0] > lat[-1]:
        lat = lat[::-1].copy()
        values = values[:, ::-1, :].copy()
    if lon[0] > lon[-1]:
        lon = lon[::-1].copy()
        values = values[:, :, ::-1].copy()
    return {
        lead: GriddedField(
            data=values[lead_to_index[lead]].copy(),
            lat=lat,
            lon=lon,
            valid_time=analysis + timedelta(minutes=lead),
            product="StormDiT reflectivity",
            units="dBZ",
            init_time=analysis,
            lead_minutes=lead,
        )
        for lead in normalized_leads
    }


def load_benchmark_arrays(
    sample: BenchmarkSample,
    models,
    *,
    catalog: BenchmarkCatalog | None = None,
    leads=None,
) -> dict:
    """Load one sample and return only the requested, already aligned arrays."""
    requested = tuple(dict.fromkeys(str(model) for model in models))
    unknown = set(requested).difference(BENCHMARK_MODEL_FRAMES)
    if unknown:
        raise ValueError(f"unsupported benchmark models: {sorted(unknown)}")
    with np.load(sample.path, allow_pickle=False) as archive:
        condition = np.asarray(archive["radar_condition_dbz"], dtype=np.float32)
        truth = np.asarray(archive["radar_truth_dbz"], dtype=np.float32)
        forecasts = {}
        requested_leads = tuple(leads or range(6, 361, 6))
        for model in requested:
            if model == "Persistence":
                forecasts[model] = np.repeat(condition[-1:, :, :], 60, axis=0)
            elif model == "Fenglei":
                if catalog is None:
                    raise ValueError("Fenglei benchmark loading requires a catalog")
                frame_count = BENCHMARK_MODEL_FRAMES[model]
                values = np.full((frame_count, 128, 128), np.nan, dtype=np.float32)
                target_lat = np.asarray(archive["lat"], dtype=np.float64)
                target_lon = np.asarray(archive["lon"], dtype=np.float64)
                # Keep one source-grid cell around the target patch.  Cropping to
                # the exact decimal endpoints can otherwise remove the nearest
                # Fenglei cell because its float32 grid origin is a few ulps away.
                source_buffer = 0.02
                bbox = (
                    float(target_lat.min() - source_buffer),
                    float(target_lat.max() + source_buffer),
                    float(target_lon.min() - source_buffer),
                    float(target_lon.max() + source_buffer),
                )
                for lead in requested_leads:
                    if lead > frame_count * 6:
                        continue
                    record = catalog.fenglei_forecasts.get((sample.analysis_time_utc, int(lead)))
                    if record is None:
                        raise ValueError(
                            f"Fenglei is unavailable for {sample.analysis_time_utc} UTC +{lead} min"
                        )
                    field = read_swan_mosaic(record.path, bbox=bbox)
                    # Benchmark rows are north-to-south. GriddedField rows and
                    # regrid_nearest targets are ascending latitude, so flip back.
                    aligned = regrid_nearest(field, target_lat[::-1], target_lon)[::-1, :]
                    values[lead // 6 - 1] = np.maximum(aligned, 0.0)
                forecasts[model] = values
            else:
                forecasts[model] = np.asarray(archive[f"pred_{model}_dbz"], dtype=np.float32)
        return {
            "sample": sample,
            "lat": np.asarray(archive["lat"], dtype=np.float64),
            "lon": np.asarray(archive["lon"], dtype=np.float64),
            "condition": condition,
            "truth": truth,
            "forecasts": forecasts,
        }

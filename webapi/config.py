"""Shared product and case configuration for the web API."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from data_io.benchmark import BENCHMARK_MODEL_LABELS, build_benchmark_catalog
from data_io.forecast_sources import (
    MODEL_FENGLEI,
    MODEL_STORMDIT,
    build_multimodel_catalog,
)


DEFAULT_HUBEI_BBOX = (29.0, 33.5, 108.0, 116.5)
CASES = {
    "2025-04-11": {
        "id": "2025-04-11",
        "label": "2025-04-11 强对流个例",
        "forecast_dir": os.getenv("FENGLEI_CASE_DIR", "external/20250411风雷预测资料"),
        "observation_dir": os.getenv("RADAR_OBS_DIR", "external/20250411观测资料"),
        "models": [
            "StormDiT",
            "风雷",
            "ConvLSTM",
            "PredRNN",
            "Earthformer",
            "NowCastNet",
            "SimVP",
            "DiffCast",
            "Persistence",
        ],
    },
    "2026-04-08": {
        "id": "2026-04-08",
        "label": "2026-04-08 强对流个例",
        "forecast_dir": os.getenv("FENGLEI_20260408_DIR", "external/20260408风雷预测资料"),
        "observation_dir": os.getenv("RADAR_20260408_DIR", "external/20260408观测资料"),
        "models": ["StormDiT", "风雷"],
    },
}
BENCHMARK_ONLY_LABELS = set(CASES["2025-04-11"]["models"][2:])
REGULAR_MODEL_LABELS = {"StormDiT", "风雷"}
BENCHMARK_LABEL_TO_MODEL = {label: model for model, label in BENCHMARK_MODEL_LABELS.items()}


def stormdit_directories() -> tuple[str, ...]:
    raw = os.getenv(
        "UPSTREAM_MODEL_DIRS",
        "external/湖北两个个例;external/湖北个例补充",
    )
    return tuple(item.strip() for item in raw.split(";") if item.strip())


def is_benchmark(models) -> bool:
    return bool(set(models) & BENCHMARK_ONLY_LABELS)


def model_family(models) -> str:
    """Return the one admissible data contract represented by model labels."""
    labels = set(models)
    if labels <= REGULAR_MODEL_LABELS:
        return "case"
    if labels <= BENCHMARK_ONLY_LABELS:
        return "benchmark"
    raise ValueError("业务个例模型与 Zero-shot Benchmark 模型使用不同资料，不能混合选择")


def regular_model_ids(labels) -> tuple[str, ...]:
    return tuple(MODEL_STORMDIT if label == "StormDiT" else MODEL_FENGLEI for label in labels)


def benchmark_model_ids(labels) -> tuple[str, ...]:
    return tuple(BENCHMARK_LABEL_TO_MODEL[label] for label in labels)


@lru_cache(maxsize=4)
def regular_catalog(case_id: str):
    case = CASES[case_id]
    return build_multimodel_catalog(
        stormdit_directories=stormdit_directories(),
        fenglei_directory=case["forecast_dir"],
        observation_directory=case["observation_dir"],
    )


@lru_cache(maxsize=1)
def benchmark_catalog():
    package = Path(os.getenv("BENCHMARK_PACKAGE_DIR", "external/delivery/apr11_wuhan_full"))
    fenglei = Path(os.getenv("FENGLEI_CASE_DIR", "external/20250411风雷预测资料"))
    return build_benchmark_catalog(package, fenglei_directory=fenglei)


def public_case_options() -> list[dict]:
    return [
        {
            "id": case["id"],
            "label": case["label"],
            "models": case["models"],
            "model_groups": {
                "case": [model for model in case["models"] if model in REGULAR_MODEL_LABELS],
                "benchmark": [model for model in case["models"] if model in BENCHMARK_ONLY_LABELS],
            },
            "default_bbox": list(DEFAULT_HUBEI_BBOX),
            "scopes": ["all", "custom", "single"],
        }
        for case in CASES.values()
    ]

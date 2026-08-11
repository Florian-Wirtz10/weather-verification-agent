"""Shared product and case configuration for the web API."""

from __future__ import annotations

from functools import lru_cache
import os
from threading import Lock

from data_io.benchmark import (
    BENCHMARK_MODEL_LABELS,
    build_benchmark_catalog,
    index_benchmark_stormdit_forecasts,
)
from data_io.forecast_sources import (
    MODEL_FENGLEI,
    MODEL_STORMDIT,
    build_multimodel_catalog,
)
from webapi.settings import DATA_ROOT, environment_path, project_path


DEFAULT_HUBEI_BBOX = (29.0, 33.5, 108.0, 116.5)
CASES = {
    "2025-04-11": {
        "id": "2025-04-11",
        "label": "2025-04-11 强对流个例",
        "forecast_dir": environment_path(
            "FENGLEI_CASE_DIR", DATA_ROOT / "cases/2025-04-11/forecasts/fenglei"
        ),
        "observation_dir": environment_path(
            "RADAR_OBS_DIR", DATA_ROOT / "cases/2025-04-11/observations/cref"
        ),
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
        "forecast_dir": environment_path(
            "FENGLEI_20260408_DIR", DATA_ROOT / "cases/2026-04-08/forecasts/fenglei"
        ),
        "observation_dir": environment_path(
            "RADAR_20260408_DIR", DATA_ROOT / "cases/2026-04-08/observations/cref"
        ),
        "models": ["StormDiT", "风雷"],
    },
}
BENCHMARK_ONLY_LABELS = set(CASES["2025-04-11"]["models"][2:])
REGULAR_MODEL_LABELS = {"StormDiT", "风雷"}
BENCHMARK_LABEL_TO_MODEL = {label: model for model, label in BENCHMARK_MODEL_LABELS.items()}

# ``lru_cache`` protects its own dictionary, but it does not coalesce concurrent
# misses.  Keep the cache lookup and a possible heavyweight catalog build inside
# one critical section so a cold start can never build multiple catalogs at once.
_catalog_build_lock = Lock()


def stormdit_directories() -> tuple[str, ...]:
    raw = os.getenv(
        "UPSTREAM_MODEL_DIRS",
        os.fspath(DATA_ROOT / "models/stormdit"),
    )
    return tuple(os.fspath(project_path(item.strip())) for item in raw.split(";") if item.strip())


def is_benchmark(models) -> bool:
    return bool(set(models) & BENCHMARK_ONLY_LABELS)


def model_family(models) -> str:
    """Choose regular-case or common-grid benchmark execution.

    Selecting any benchmark-only model moves the complete selection onto the
    benchmark package, where StormDiT and Fenglei remain available as business
    comparators on the same samples and native common lead window.
    """
    labels = set(models)
    if labels <= REGULAR_MODEL_LABELS:
        return "case"
    if labels & BENCHMARK_ONLY_LABELS and labels <= (
        REGULAR_MODEL_LABELS | BENCHMARK_ONLY_LABELS
    ):
        return "benchmark"
    raise ValueError("所选模型不在当前个例的可用模型目录中")


def regular_model_ids(labels) -> tuple[str, ...]:
    return tuple(MODEL_STORMDIT if label == "StormDiT" else MODEL_FENGLEI for label in labels)


def benchmark_model_ids(labels) -> tuple[str, ...]:
    return tuple(BENCHMARK_LABEL_TO_MODEL[label] for label in labels)


@lru_cache(maxsize=4)
def _regular_catalog_cached(case_id: str):
    case = CASES[case_id]
    catalog = build_multimodel_catalog(
        stormdit_directories=stormdit_directories(),
        fenglei_directory=case["forecast_dir"],
        observation_directory=case["observation_dir"],
    )
    if case_id != "2025-04-11":
        return catalog

    supplemental = index_benchmark_stormdit_forecasts(_benchmark_catalog_cached())
    stormdit = dict(catalog.forecasts[MODEL_STORMDIT])
    duplicates = set(stormdit).intersection(supplemental)
    if duplicates:
        raise ValueError(
            "StormDiT业务交付与公共基准样本存在重复起报/时效："
            f"{sorted(duplicates)[:3]}"
        )
    stormdit.update(supplemental)
    forecasts = dict(catalog.forecasts)
    forecasts[MODEL_STORMDIT] = stormdit
    return type(catalog)(
        observations=catalog.observations,
        forecasts=forecasts,
        stormdit_contract=f"{catalog.stormdit_contract}+benchmark-wuhan-v1",
    )


def regular_catalog(case_id: str):
    with _catalog_build_lock:
        return _regular_catalog_cached(case_id)


@lru_cache(maxsize=1)
def _benchmark_catalog_cached():
    package = environment_path(
        "BENCHMARK_PACKAGE_DIR",
        DATA_ROOT / "benchmarks/2025-04-11/wuhan-zero-shot",
    )
    fenglei = environment_path(
        "FENGLEI_CASE_DIR", DATA_ROOT / "cases/2025-04-11/forecasts/fenglei"
    )
    return build_benchmark_catalog(package, fenglei_directory=fenglei)


def benchmark_catalog():
    with _catalog_build_lock:
        return _benchmark_catalog_cached()


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


def public_data_metadata() -> dict:
    """Return the documented, non-sensitive data contract shown by the UI."""
    return {
        "llm": {
            "provider": "DeepSeek API",
            "model": "deepseek-chat",
            "role": "将自然语言解析为白名单检验参数，并基于结构化指标生成AI解读",
        },
        "products": [
            {
                "name": "StormDiT",
                "kind": "预报",
                "format": "NPZ",
                "grid": "640 × 720，0.03°",
                "clock": (
                    "业务交付为BJT整点起报；公共基准样本为每小时一次、固定在:36起报；"
                    "进入检验前统一转换为UTC"
                ),
                "cadence": (
                    "起报频率为1小时；预报输出间隔为6分钟。StormDiT自身提供"
                    "+6—+360分钟，当前业务联合检验使用+6—+180分钟"
                ),
                "domain": (
                    "业务交付为101.4—122.97°E、21.83—41.00°N；公共基准样本"
                    "仅覆盖112.38—116.19°E、28.70—32.51°N，混合统计自动使用公共区域"
                ),
            },
            {
                "name": "风雷",
                "kind": "预报",
                "format": "SWAN / BZ2",
                "grid": "约 0.01°，以文件头 nx、ny、起点和步长为准",
                "clock": "SWAN 文件头为 UTC；2026 V11 文件名为 BJT",
                "cadence": "起报频率和预报输出间隔均为6分钟；提供+6—+180分钟",
                "domain": "读取文件原生网格后按检验区域裁剪",
            },
            {
                "name": "CREF",
                "kind": "观测",
                "format": "NetCDF / MOC BIN",
                "grid": "约 0.01°，以坐标变量或二进制文件头为准",
                "clock": "2025-04-11 与 2026-04-08 交付时间均按 UTC 读取",
                "cadence": "观测间隔为6分钟；观测资料没有起报时刻",
                "domain": "作为统一雷达组合反射率观测基准",
            },
            {
                "name": "Zero-shot Benchmark",
                "kind": "基线预报",
                "format": "NPZ / CSV",
                "grid": "128 × 128，0.03°",
                "clock": "内部按 UTC 严格匹配，界面同时标注 BJT",
                "cadence": (
                    "公共样本每小时一次、固定在:36起报；预报输出间隔为6分钟。"
                    "ConvLSTM等为+6—+72分钟，DiffCast为+6—+120分钟，"
                    "StormDiT为+6—+360分钟；联合比较取最短公共时效"
                ),
                "domain": "武汉区域公共样本网格",
            },
        ],
        "cases": [
            {
                "id": "2025-04-11",
                "label": "2025-04-11 强对流个例",
                "availability": "StormDiT、风雷、CREF 与 Zero-shot Benchmark",
            },
            {
                "id": "2026-04-08",
                "label": "2026-04-08 强对流个例",
                "availability": "StormDiT、风雷与 CREF",
            },
        ],
        "rules": [
            "内部统一使用 UTC 建立索引，界面以 BJT 为主并附 UTC。",
            "预报有效时间必须与 CREF 观测时刻严格相同，不进行时间插值。",
            "空间检验在公共地理区域和双方有限值掩膜上进行；需要时仅将预报确定性重采样到观测网格。",
            "双模型比较只使用 StormDiT、风雷与 CREF 三者共同存在的样本。",
            "6分钟描述的是观测或预报帧间隔；只有风雷是每6分钟起报，StormDiT和公共基准样本均为每小时一次起报。",
        ],
    }

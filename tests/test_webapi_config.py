from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

from webapi import config


def test_regular_catalog_coalesces_concurrent_cold_start(monkeypatch):
    calls = 0
    calls_lock = Lock()
    catalog = object()

    def build_catalog(**_kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return catalog

    monkeypatch.setattr(config, "build_multimodel_catalog", build_catalog)
    config._regular_catalog_cached.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(config.regular_catalog, ["2025-04-11"] * 8))

        assert calls == 1
        assert all(result is catalog for result in results)
    finally:
        config._regular_catalog_cached.cache_clear()


def test_benchmark_catalog_coalesces_concurrent_cold_start(monkeypatch):
    calls = 0
    calls_lock = Lock()
    catalog = object()

    def build_catalog(*_args, **_kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return catalog

    monkeypatch.setattr(config, "build_benchmark_catalog", build_catalog)
    config._benchmark_catalog_cached.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _index: config.benchmark_catalog(), range(8)))

        assert calls == 1
        assert all(result is catalog for result in results)
    finally:
        config._benchmark_catalog_cached.cache_clear()

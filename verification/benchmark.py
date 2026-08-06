"""Deterministic verification for the delivered zero-shot benchmark package."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from data_io.benchmark import (
    BENCHMARK_MODEL_FRAMES,
    BenchmarkCatalog,
    load_benchmark_arrays,
)


def common_horizon_minutes(models) -> int:
    selected = tuple(dict.fromkeys(str(model) for model in models))
    if not selected:
        raise ValueError("select at least one benchmark model")
    unknown = set(selected).difference(BENCHMARK_MODEL_FRAMES)
    if unknown:
        raise ValueError(f"unsupported benchmark models: {sorted(unknown)}")
    return min(BENCHMARK_MODEL_FRAMES[model] * 6 for model in selected)


def benchmark_available_samples(catalog, models, *, lead_range=None):
    """Return samples containing every selected model at every requested lead."""
    models = tuple(dict.fromkeys(str(model) for model in models))
    horizon = common_horizon_minutes(models)
    start, end = lead_range or (6, horizon)
    leads = tuple(range(int(start), int(end) + 1, 6))
    available = []
    for sample in catalog.samples:
        if "Fenglei" in models and not all(
            (sample.analysis_time_utc, lead) in catalog.fenglei_forecasts for lead in leads
        ):
            continue
        available.append(sample)
    return tuple(available)


def _safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else float("nan")


def _fraction_field(binary, window: int):
    """Return a zero-padded square-neighbourhood fraction field."""
    if window not in {1, 3, 5}:
        raise ValueError("FSS window must be 1, 3 or 5 pixels")
    values = np.asarray(binary, dtype=np.float64)
    if window == 1:
        return values
    radius = window // 2
    padded = np.pad(values, radius, mode="constant", constant_values=0)
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    sums = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return sums / float(window * window)


def _frame_row(forecast, truth, threshold, window_sizes):
    forecast = np.asarray(forecast, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.isfinite(forecast) & np.isfinite(truth)
    if not np.all(valid):
        # The delivery contract contains no NaNs. Keep an explicit guard so a
        # malformed future package cannot silently change the published sample set.
        raise ValueError("benchmark arrays must contain only finite values")
    forecast_yes = forecast >= threshold
    truth_yes = truth >= threshold
    hits = int(np.count_nonzero(forecast_yes & truth_yes))
    misses = int(np.count_nonzero(~forecast_yes & truth_yes))
    false_alarms = int(np.count_nonzero(forecast_yes & ~truth_yes))
    correct_negatives = int(np.count_nonzero(~forecast_yes & ~truth_yes))
    difference = forecast - truth
    row = {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "valid_count": int(forecast.size),
        "absolute_error_sum": float(np.abs(difference).sum(dtype=np.float64)),
        "squared_error_sum": float(np.square(difference).sum(dtype=np.float64)),
    }
    row.update(_scores(row))
    row["mae"] = row["absolute_error_sum"] / row["valid_count"]
    row["rmse"] = math.sqrt(row["squared_error_sum"] / row["valid_count"])
    for window in window_sizes:
        forecast_fraction = _fraction_field(forecast_yes, window)
        truth_fraction = _fraction_field(truth_yes, window)
        numerator = float(np.square(forecast_fraction - truth_fraction).sum(dtype=np.float64))
        denominator = float(
            (np.square(forecast_fraction) + np.square(truth_fraction)).sum(dtype=np.float64)
        )
        row[f"fss_{window}_numerator"] = numerator
        row[f"fss_{window}_denominator"] = denominator
        row[f"fss_w{window}"] = 1.0 - _safe_divide(numerator, denominator)
    return row


def _scores(row):
    hits = row["hits"]
    misses = row["misses"]
    false_alarms = row["false_alarms"]
    return {
        "csi": _safe_divide(hits, hits + misses + false_alarms),
        "pod": _safe_divide(hits, hits + misses),
        "far": _safe_divide(false_alarms, hits + false_alarms),
        "bias_score": _safe_divide(hits + false_alarms, hits + misses),
    }


def run_benchmark(
    catalog: BenchmarkCatalog,
    models,
    *,
    sample_ids=None,
    thresholds=(20.0, 35.0, 45.0),
    lead_range=(6, 72),
    fss_windows=(1, 3, 5),
    progress_callback=None,
):
    """Evaluate selected models on strict common samples and native lead times."""
    models = tuple(dict.fromkeys(str(model) for model in models))
    thresholds = tuple(dict.fromkeys(float(value) for value in thresholds))
    if not models:
        raise ValueError("select at least one benchmark model")
    if not thresholds or any(not 0 <= value <= 75 for value in thresholds):
        raise ValueError("benchmark thresholds must be between 0 and 75 dBZ")
    start, end = map(int, lead_range)
    if start < 6 or start % 6 or end < start or end % 6:
        raise ValueError("benchmark lead range must use positive 6-minute steps")
    horizon = common_horizon_minutes(models)
    if end > horizon:
        raise ValueError(f"selected models only share native forecasts through +{horizon} minutes")
    selected_ids = None if sample_ids is None else {int(value) for value in sample_ids}
    eligible_samples = benchmark_available_samples(catalog, models, lead_range=(start, end))
    samples = [
        sample
        for sample in eligible_samples
        if selected_ids is None or sample.sample_id in selected_ids
    ]
    if not samples:
        raise ValueError("no benchmark samples were selected")
    if selected_ids is not None and selected_ids != {sample.sample_id for sample in samples}:
        missing = sorted(selected_ids.difference(sample.sample_id for sample in samples))
        raise ValueError(f"unknown benchmark sample IDs: {missing}")

    rows = []
    leads = list(range(start, end + 1, 6))
    total = len(samples)
    for sample_index, sample in enumerate(samples, start=1):
        arrays = load_benchmark_arrays(sample, models, catalog=catalog, leads=leads)
        truth = arrays["truth"]
        for model in models:
            forecast = arrays["forecasts"][model]
            for lead in leads:
                frame_index = lead // 6 - 1
                for threshold in thresholds:
                    row = _frame_row(
                        forecast[frame_index],
                        truth[frame_index],
                        threshold,
                        fss_windows,
                    )
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "analysis_time_utc": sample.analysis_time_utc,
                            "analysis_time_bjt": sample.analysis_time_bjt,
                            "model": model,
                            "lead_minutes": lead,
                            "threshold_dbz": threshold,
                            **row,
                        }
                    )
        if progress_callback is not None:
            progress_callback(sample_index, total, sample.sample_id)
    per_sample = pd.DataFrame(rows)
    summary = aggregate_benchmark_rows(per_sample, fss_windows=fss_windows)
    return {
        "context": {
            "models": list(models),
            "thresholds_dbz": list(thresholds),
            "lead_range_minutes": [start, end],
            "common_horizon_minutes": horizon,
            "sample_policy": "strict common 128x128 CREF patch",
        },
        "coverage": {
            "sample_count": len(samples),
            "pair_count": len(samples) * len(models) * len(leads),
            "lead_count": len(leads),
        },
        "per_sample": per_sample,
        "summary": summary,
    }


def aggregate_benchmark_rows(rows: pd.DataFrame, *, fss_windows=(1, 3, 5)):
    """Pool contingency/error/FSS components before recomputing scores."""
    group_columns = ["model", "lead_minutes", "threshold_dbz"]
    sum_columns = [
        "hits",
        "misses",
        "false_alarms",
        "correct_negatives",
        "valid_count",
        "absolute_error_sum",
        "squared_error_sum",
    ] + [
        f"fss_{window}_{component}"
        for window in fss_windows
        for component in ("numerator", "denominator")
    ]
    grouped = rows.groupby(group_columns, as_index=False, sort=True)[sum_columns].sum()
    records = []
    for row in grouped.to_dict("records"):
        row.update(_scores(row))
        row["mae"] = _safe_divide(row["absolute_error_sum"], row["valid_count"])
        row["rmse"] = math.sqrt(_safe_divide(row["squared_error_sum"], row["valid_count"]))
        for window in fss_windows:
            ratio = _safe_divide(
                row[f"fss_{window}_numerator"],
                row[f"fss_{window}_denominator"],
            )
            row[f"fss_w{window}"] = 1.0 - ratio
        records.append(row)
    return pd.DataFrame(records).sort_values(group_columns).reset_index(drop=True)


def summarize_benchmark_period(rows: pd.DataFrame, *, fss_windows=(1, 3, 5)):
    """Pool every selected sample and lead into one row per model/threshold."""
    sum_columns = [
        "hits",
        "misses",
        "false_alarms",
        "correct_negatives",
        "valid_count",
        "absolute_error_sum",
        "squared_error_sum",
    ] + [
        f"fss_{window}_{component}"
        for window in fss_windows
        for component in ("numerator", "denominator")
    ]
    grouped = rows.groupby(["model", "threshold_dbz"], as_index=False, sort=True)[sum_columns].sum()
    records = []
    for row in grouped.to_dict("records"):
        row.update(_scores(row))
        row["mae"] = _safe_divide(row["absolute_error_sum"], row["valid_count"])
        row["rmse"] = math.sqrt(_safe_divide(row["squared_error_sum"], row["valid_count"]))
        for window in fss_windows:
            row[f"fss_w{window}"] = 1.0 - _safe_divide(
                row[f"fss_{window}_numerator"],
                row[f"fss_{window}_denominator"],
            )
        records.append(row)
    return pd.DataFrame(records).sort_values(["model", "threshold_dbz"]).reset_index(drop=True)


def benchmark_frame(catalog, sample_id, models, lead_minutes):
    """Load one already aligned spatial comparison frame."""
    sample = next((item for item in catalog.samples if item.sample_id == int(sample_id)), None)
    if sample is None:
        raise ValueError(f"unknown benchmark sample ID: {sample_id}")
    lead_minutes = int(lead_minutes)
    if lead_minutes < 6 or lead_minutes % 6:
        raise ValueError("benchmark spatial lead must be a positive multiple of 6")
    if lead_minutes > common_horizon_minutes(models):
        raise ValueError("benchmark spatial lead exceeds a selected model's native horizon")
    arrays = load_benchmark_arrays(
        sample,
        models,
        catalog=catalog,
        leads=(lead_minutes,),
    )
    index = lead_minutes // 6 - 1
    return {
        "sample": sample,
        "lat": arrays["lat"],
        "lon": arrays["lon"],
        "truth": arrays["truth"][index],
        "forecasts": {model: values[index] for model, values in arrays["forecasts"].items()},
        "lead_minutes": lead_minutes,
    }

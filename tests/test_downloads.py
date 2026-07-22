from datetime import datetime
from io import BytesIO
import json
from zipfile import ZipFile

import matplotlib.pyplot as plt

from reports.downloads import build_case_download_artifacts, figure_to_png


def _download_inputs():
    first, first_ax = plt.subplots()
    first_ax.plot([6, 12], [0.2, 0.1])
    second, second_ax = plt.subplots()
    second_ax.imshow([[1, 2], [3, 4]])
    result = {
        "context": {
            "case_id": "fenglei-refmosaic-202504110212",
            "product": "Fenglei V1.0 REFMOSAIC",
            "variable": "radar composite reflectivity",
            "units": "dBZ",
            "init_time": datetime(2025, 4, 11, 2, 12),
            "bbox": [28.0, 35.0, 108.0, 118.0],
            "source_path": "/secret/private/fenglei",
            "api_key": "do-not-export",
        },
        "coverage": {
            "requested_range": [0, 60],
            "selected_leads": [6, 12],
            "missing_leads": [],
        },
        "figures": {"view_01": first, "view_02": second},
        "figure_metadata": {
            "view_01": {"type": "score_time_series", "metric": "csi"},
            "view_02": {"type": "reflectivity_error", "lead": 12},
        },
    }
    request = {
        "schema_version": "1",
        "intent": "verify_case",
        "filters": {
            "thresholds_dbz": [35],
            "lead_range_minutes": {"start": 0, "end": 60},
            "data_path": "/secret/from/request",
        },
        "views": [{"type": "score_time_series", "metric": "csi"}],
        "unmatched_requests": ["暂不支持 Taylor 图 /secret/free-text"],
        "notes": "/secret/free-text-note",
    }
    rows = [
        {
            "lead_minutes": 6,
            "valid_time_utc": datetime(2025, 4, 11, 2, 18),
            "threshold_dbz": 35,
            "csi": 0.2,
            "source_path": "/secret/from/metric-row",
        }
    ]
    facts = {
        "context": {"case_id": "fenglei-refmosaic-202504110212"},
        "requested_metrics": ["csi"],
        "source_path": "/secret/from/facts",
    }
    return result, request, rows, facts


def test_figure_to_png_returns_real_png_bytes():
    figure, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])

    data = figure_to_png(figure)

    assert data.startswith(b"\x89PNG\r\n\x1a\n")


def test_download_bundle_contains_images_csv_report_and_safe_metadata():
    result, request, rows, facts = _download_inputs()

    artifacts = build_case_download_artifacts(
        result=result,
        request=request,
        metric_rows=rows,
        facts=facts,
        deterministic_summary="CSI 从 0.2 降至 0.1。",
        ai_insight="客观事实：CSI下降。",
        execution_status="partial",
    )

    assert len(artifacts["images"]) == 2
    assert artifacts["images"]["view_01"]["file_name"].endswith(
        "_01_score_time_series_csi.png"
    )
    assert artifacts["images"]["view_02"]["file_name"].endswith(
        "_02_reflectivity_error_012min.png"
    )
    assert artifacts["metrics"]["data"].startswith(b"\xef\xbb\xbf")
    assert "thr35dbz" in artifacts["base_name"]
    assert "_partial_" in artifacts["base_name"]
    assert "_req" in artifacts["base_name"]
    csv_text = artifacts["metrics"]["data"].decode("utf-8-sig")
    assert "case_id,init_time_utc,lat_min,lat_max,lon_min,lon_max" in csv_text
    assert "2025-04-11T02:12:00Z" in csv_text
    assert "source_path" not in csv_text
    assert "/secret/from/metric-row" not in csv_text

    bundle = artifacts["bundle"]["data"]
    assert b"/secret" not in bundle
    assert b"/secret/private/fenglei" not in bundle
    assert b"do-not-export" not in bundle
    with ZipFile(BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        assert any(name.endswith("_01_score_time_series_csi.png") for name in names)
        assert any(name.endswith("_02_reflectivity_error_012min.png") for name in names)
        assert any(name.startswith("tables/") and name.endswith("_metrics.csv") for name in names)
        assert any(name.startswith("reports/") and name.endswith("_summary.md") for name in names)
        assert {
            "metadata/execution_request.json",
            "metadata/facts.json",
            "metadata/manifest.json",
        } <= names
        manifest = archive.read("metadata/manifest.json").decode("utf-8")
        assert "source_path" not in manifest
        assert "api_key" not in manifest
        manifest_json = json.loads(manifest)
        assert manifest_json["execution"] == {
            "status": "partial",
            "unmatched_request_count": 1,
        }
        request_json = archive.read("metadata/execution_request.json").decode("utf-8")
        assert "/secret/from/request" not in request_json
        assert "/secret/free-text-note" not in request_json
        facts_json = archive.read("metadata/facts.json").decode("utf-8")
        assert "/secret/from/facts" not in facts_json
        report_name = next(name for name in names if name.startswith("reports/"))
        report = archive.read(report_name).decode("utf-8")
        assert "确定性事实摘要" in report
        assert "AI 解读" in report
        assert "执行状态：部分执行" in report
        assert "本次有 1 项请求超出当前能力" in report
        assert "/secret/free-text" not in report


def test_download_bundle_exports_valid_animation_gif():
    result, request, rows, facts = _download_inputs()
    result["animations"] = {"animation_01": b"GIF89a" + b"test"}
    result["animation_metadata"] = {
        "animation_01": {
            "type": "reflectivity_animation",
            "lead_range_minutes": {"start": 6, "end": 24},
            "fps": 2,
            "linked_metric": "csi",
        }
    }
    request["presentation"] = {"layout": "animation_focus"}
    request["views"].append(result["animation_metadata"]["animation_01"])

    artifacts = build_case_download_artifacts(
        result=result,
        request=request,
        metric_rows=rows,
        facts=facts,
        deterministic_summary="测试。",
        ai_insight=None,
    )
    assert artifacts["animations"]["animation_01"]["file_name"].endswith(".gif")
    assert "CSI" in artifacts["animations"]["animation_01"]["label"]
    with ZipFile(BytesIO(artifacts["bundle"]["data"])) as archive:
        assert any(name.startswith("animations/") and name.endswith(".gif") for name in archive.namelist())


def test_download_bundle_accepts_threshold_specific_rate_figure_metadata():
    result, request, rows, facts = _download_inputs()
    plt.close(result["figures"]["view_02"])
    result["figures"] = {"view_01": result["figures"]["view_01"]}
    result["figure_metadata"] = {
        "view_01": {"type": "forecast_observed_rate", "threshold_dbz": 15.0}
    }
    request["views"] = [{"type": "forecast_observed_rate"}]

    artifacts = build_case_download_artifacts(
        result=result,
        request=request,
        metric_rows=rows,
        facts=facts,
        deterministic_summary="测试。",
        ai_insight=None,
    )

    image = artifacts["images"]["view_01"]
    assert "15 dBZ" in image["label"]
    assert "15dbz-rate-trajectory" in image["file_name"]


def test_download_bundle_accepts_combined_rate_threshold_metadata():
    result, request, rows, facts = _download_inputs()
    plt.close(result["figures"]["view_02"])
    result["figures"] = {"view_01": result["figures"]["view_01"]}
    result["figure_metadata"] = {
        "view_01": {
            "type": "forecast_observed_rate",
            "threshold_mode": "combined",
            "thresholds_dbz": [10.0, 20.0, 35.0],
        }
    }
    request["views"] = [
        {"type": "forecast_observed_rate", "threshold_mode": "combined"}
    ]

    artifacts = build_case_download_artifacts(
        result=result,
        request=request,
        metric_rows=rows,
        facts=facts,
        deterministic_summary="测试。",
        ai_insight=None,
    )

    image = artifacts["images"]["view_01"]
    assert "10、20、35 dBZ" in image["label"]
    assert "threshold-rate-comparison" in image["file_name"]


def test_downloads_can_reuse_pngs_and_request_fingerprint_avoids_name_collision():
    result, request, rows, facts = _download_inputs()
    first = build_case_download_artifacts(
        result=result,
        request=request,
        metric_rows=rows,
        facts=facts,
        deterministic_summary="summary",
        ai_insight=None,
    )
    original_pngs = {
        key: item["data"] for key, item in first["images"].items()
    }
    changed_request = {
        **request,
        "views": [{"type": "score_time_series", "metric": "ets"}],
    }
    result["figures"] = {}

    second = build_case_download_artifacts(
        result=result,
        request=changed_request,
        metric_rows=rows,
        facts=facts,
        deterministic_summary="summary",
        ai_insight="new insight",
        rendered_images=first["images"],
    )

    assert second["base_name"] != first["base_name"]
    assert {key: item["data"] for key, item in second["images"].items()} == original_pngs
    assert b"new insight" in second["summary"]["data"]


def test_download_export_rejects_unknown_figure_metadata():
    result, request, rows, facts = _download_inputs()
    result["figure_metadata"]["view_01"] = {"type": "../../escape", "metric": "csi"}

    try:
        build_case_download_artifacts(
            result=result,
            request=request,
            metric_rows=rows,
            facts=facts,
            deterministic_summary="summary",
            ai_insight=None,
        )
    except ValueError as exc:
        assert "unsupported export view type" in str(exc)
    else:
        raise AssertionError("unknown metadata must fail closed")

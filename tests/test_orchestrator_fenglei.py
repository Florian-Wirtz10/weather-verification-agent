"""End-to-end test of run_fenglei_case_verification against small synthetic
fixtures (never the private external/ sample data)."""

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

import agent.orchestrator as orchestrator_module
from agent import (
    AmbiguousCaseError,
    DataContractError,
    discover_fenglei_cases,
    run_fenglei_case_verification,
)
from tests.test_real_data_io import _make_cref_nc, _make_swan_mosaic_bytes


def test_run_fenglei_case_verification_end_to_end(tmp_path, monkeypatch):
    fenglei_dir = tmp_path / "fenglei"
    radar_dir = tmp_path / "radar"
    fenglei_dir.mkdir()
    radar_dir.mkdir()

    nx, ny = 6, 6
    lat = 30.0 - 0.01 * np.arange(ny)  # north -> south, matches StartLat=30 in the fixture
    lat = lat[::-1]  # ascending, matching what read_swan_mosaic will produce
    lon = 110.0 + 0.01 * np.arange(nx)

    for lead_minutes in (6, 12):
        # a simple echo blob, shifted slightly by lead time so scores differ per lead
        raw = np.zeros((ny, nx), dtype="u1")
        raw[2:4, 2 + lead_minutes // 6 : 5] = 140  # 140*0.5-33 = 37 dBZ
        blob = _make_swan_mosaic_bytes(
            nx=nx,
            ny=ny,
            cell_bytes=1,
            scale=0.5,
            offset=-33.0,
            values=raw,
            header_overrides={"interval": lead_minutes},
        )
        path = fenglei_dir / f"REFMOSAIC-202504110212-{lead_minutes:03d}MIN.bz2"
        path.write_bytes(blob)

        # matching obs: same blob location, no lead-shift (perfect obs at valid time)
        obs_data = np.zeros((ny, nx), dtype="f4")
        obs_data[2:4, 2:5] = 40.0
        valid_time_utc = f"2025-04-11 02:{12 + lead_minutes:02d}:00"
        _make_cref_nc(
            radar_dir / f"CREF_with_time_{lead_minutes}.nc",
            lat=lat.astype("f4"),
            lon=lon.astype("f4"),
            data=obs_data,
            valid_time_str=valid_time_utc,
        )

    result = run_fenglei_case_verification(
        fenglei_dir=fenglei_dir,
        radar_dir=radar_dir,
        bbox=(29.9, 30.1, 109.9, 110.1),
        thresholds=(20,),
        lead_range=(0, 12),
        views=[
            {"type": "score_time_series", "metric": "csi"},
            {"type": "score_time_series", "metric": "ets"},
            {"type": "forecast_observed_rate"},
            {"type": "reflectivity_comparison", "lead": "first"},
            {"type": "reflectivity_error", "lead": 12},
        ],
    )

    assert result["lead_minutes"] == [6, 12]
    assert len(result["per_lead"]) == 2
    assert result["per_lead"][0]["categorical"][20]["csi"] > 0
    assert list(result["figures"]) == [
        "view_01",
        "view_02",
        "view_03",
        "view_04",
        "view_05",
    ]
    assert result["figure_metadata"] == {
        "view_01": {"type": "score_time_series", "metric": "csi"},
        "view_02": {"type": "score_time_series", "metric": "ets"},
        "view_03": {"type": "forecast_observed_rate"},
        "view_04": {"type": "reflectivity_comparison", "lead": 6},
        "view_05": {"type": "reflectivity_error", "lead": 12},
    }
    assert result["context"]["init_time"].isoformat() == "2025-04-11T02:12:00"
    assert result["coverage"]["selected_leads"] == [6, 12]
    assert result["coverage"]["missing_leads"] == []
    assert result["per_lead"][0]["contingency"][20]["valid_count"] > 0

    open_figures_before_animation = set(plt.get_fignums())
    animated = run_fenglei_case_verification(
        fenglei_dir=fenglei_dir,
        radar_dir=radar_dir,
        bbox=(29.9, 30.1, 109.9, 110.1),
        thresholds=(20,),
        lead_range=(0, 12),
        views=[
            {
                "type": "reflectivity_animation",
                "lead_range_minutes": {"start": 6, "end": 12},
                "fps": 2,
            }
        ],
    )
    assert animated["figures"] == {}
    assert animated["animation_metadata"] == {
        "animation_01": {
            "type": "reflectivity_animation",
            "lead_range_minutes": {"start": 6, "end": 12},
            "leads": [6, 12],
            "fps": 2,
        }
    }
    assert animated["animations"]["animation_01"].startswith(b"GIF89a")
    assert set(plt.get_fignums()) == open_figures_before_animation

    linked = run_fenglei_case_verification(
        fenglei_dir=fenglei_dir,
        radar_dir=radar_dir,
        bbox=(29.9, 30.1, 109.9, 110.1),
        thresholds=(20,),
        lead_range=(0, 12),
        views=[
            {"type": "score_time_series", "metric": "csi"},
            {
                "type": "reflectivity_animation",
                "lead_range_minutes": {"start": 6, "end": 12},
                "fps": 2,
            },
        ],
        presentation={"strategy": "linked_score_animation", "metric": "csi"},
    )
    assert linked["figures"] == {}
    assert linked["presentation"]["resolved_strategy"] == "linked_score_animation"
    assert linked["animation_metadata"]["animation_01"]["linked_metric"] == "csi"
    assert linked["animations"]["animation_01"].startswith(b"GIF89a")
    with Image.open(BytesIO(linked["animations"]["animation_01"])) as linked_gif:
        assert int(getattr(linked_gif, "n_frames", 1)) == 2
        assert linked_gif.width <= 1_200
        assert linked_gif.height > linked_gif.width / 2

    cases = discover_fenglei_cases(fenglei_dir, radar_dir)
    assert len(cases) == 1
    assert cases[0]["available_leads"] == [6, 12]

    decoded_paths = []
    original_reader = orchestrator_module.read_swan_mosaic

    def tracking_reader(path):
        decoded_paths.append(path)
        return original_reader(path)

    monkeypatch.setattr(orchestrator_module, "read_swan_mosaic", tracking_reader)
    filtered = run_fenglei_case_verification(
        fenglei_dir=fenglei_dir,
        radar_dir=radar_dir,
        bbox=(29.9, 30.1, 109.9, 110.1),
        thresholds=(20,),
        lead_range=(6, 6),
    )
    assert filtered["lead_minutes"] == [6]
    assert list(filtered["figures"]) == ["view_01"]
    assert len(decoded_paths) == 1
    assert "006MIN" in decoded_paths[0].name

    with pytest.raises(DataContractError, match="at most 6 spatial views"):
        run_fenglei_case_verification(
            fenglei_dir=fenglei_dir,
            radar_dir=radar_dir,
            bbox=(29.9, 30.1, 109.9, 110.1),
            thresholds=(20,),
            views=[
                {"type": "reflectivity_error", "lead": 6}
                for _ in range(7)
            ],
        )


def test_run_fenglei_case_requires_exact_timestamp_match(tmp_path):
    fenglei_dir = tmp_path / "fenglei"
    radar_dir = tmp_path / "radar"
    fenglei_dir.mkdir()
    radar_dir.mkdir()

    raw = np.zeros((2, 2), dtype="u1")
    blob = _make_swan_mosaic_bytes(
        nx=2, ny=2, cell_bytes=1, scale=0.5, offset=-33.0, values=raw
    )
    (fenglei_dir / "REFMOSAIC-202504110212-006MIN.bz2").write_bytes(blob)
    _make_cref_nc(
        radar_dir / "CREF_with_time_mismatch.nc",
        lat=np.array([29.99, 30.0], dtype="f4"),
        lon=np.array([110.0, 110.01], dtype="f4"),
        data=np.zeros((2, 2), dtype="f4"),
        valid_time_str="2025-04-11 10:19:00",
    )

    with pytest.raises(ValueError, match="exactly time-matched"):
        run_fenglei_case_verification(
            fenglei_dir=fenglei_dir,
            radar_dir=radar_dir,
            bbox=(29.9, 30.1, 109.9, 110.1),
            thresholds=(20,),
        )


def test_run_fenglei_case_never_mixes_initializations(tmp_path):
    fenglei_dir = tmp_path / "fenglei"
    radar_dir = tmp_path / "radar"
    fenglei_dir.mkdir()
    radar_dir.mkdir()
    raw = np.zeros((2, 2), dtype="u1")
    lat = np.array([29.99, 30.0], dtype="f4")
    lon = np.array([110.0, 110.01], dtype="f4")

    for hour in (2, 3):
        blob = _make_swan_mosaic_bytes(
            nx=2,
            ny=2,
            cell_bytes=1,
            scale=0.5,
            offset=-33.0,
            values=raw,
            header_overrides={"hour": hour, "minute": 12, "interval": 6},
        )
        (fenglei_dir / f"REFMOSAIC-init-{hour}-006MIN.bz2").write_bytes(blob)
        _make_cref_nc(
            radar_dir / f"CREF_with_time_{hour}.nc",
            lat=lat,
            lon=lon,
            data=np.zeros((2, 2), dtype="f4"),
            valid_time_str=f"2025-04-11 {hour:02d}:18:00",
        )

    with pytest.raises(AmbiguousCaseError, match="select one forecast initialization"):
        run_fenglei_case_verification(
            fenglei_dir=fenglei_dir,
            radar_dir=radar_dir,
            bbox=(29.9, 30.1, 109.9, 110.1),
            thresholds=(20,),
        )

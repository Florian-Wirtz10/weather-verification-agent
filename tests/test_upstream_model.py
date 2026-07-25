from datetime import datetime

import numpy as np
import pytest

from data_io.upstream_model import (
    UpstreamModelContract,
    read_upstream_model_forecast,
    read_upstream_model_forecasts,
    read_upstream_model_metadata,
)


@pytest.fixture
def miniature_contract():
    return UpstreamModelContract(
        ny=2,
        nx=3,
        history_count=2,
        prediction_count=3,
        step_minutes=6,
        north_lat=41.0,
        west_lon=101.4,
        resolution_degrees=0.03,
        name="test-contract",
    )


def _write_npz(path):
    predicted = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    history = predicted[:2] + 0.5
    np.savez(
        path,
        predicted_radar=predicted,
        history_frames=history,
        start_time=np.asarray("2025-04-11 08:00:00"),
        history_times=np.asarray(["202504110754", "202504110800"]),
        num_history=np.asarray(2),
        num_prediction=np.asarray(3),
    )
    return predicted


def test_metadata_converts_bjt_to_utc_and_builds_future_leads(
    tmp_path, miniature_contract
):
    path = tmp_path / "forecast.npz"
    _write_npz(path)

    metadata = read_upstream_model_metadata(path, contract=miniature_contract)

    assert metadata.init_time_bjt == datetime(2025, 4, 11, 8)
    assert metadata.init_time_utc == datetime(2025, 4, 11, 0)
    assert metadata.available_leads == (6, 12, 18)
    assert metadata.contract_name == "test-contract"


def test_reader_selects_future_frame_and_normalizes_latitude(
    tmp_path, miniature_contract
):
    path = tmp_path / "forecast.npz"
    predicted = _write_npz(path)

    field = read_upstream_model_forecast(
        path, 6, contract=miniature_contract
    )

    # Two history/reconstruction frames precede the first future frame.
    np.testing.assert_array_equal(field.data, predicted[2][::-1])
    np.testing.assert_allclose(field.lat, [40.97, 41.0])
    np.testing.assert_allclose(field.lon, [101.4, 101.43, 101.46])
    assert field.init_time == datetime(2025, 4, 11, 0)
    assert field.valid_time == datetime(2025, 4, 11, 0, 6)
    assert field.lead_minutes == 6


def test_reader_rejects_non_native_lead(tmp_path, miniature_contract):
    path = tmp_path / "forecast.npz"
    _write_npz(path)

    with pytest.raises(ValueError, match="native step"):
        read_upstream_model_forecast(path, 7, contract=miniature_contract)


def test_reader_loads_multiple_future_leads(tmp_path, miniature_contract):
    path = tmp_path / "forecast.npz"
    predicted = _write_npz(path)

    fields = read_upstream_model_forecasts(
        path, (6, 18), contract=miniature_contract
    )

    assert list(fields) == [6, 18]
    np.testing.assert_array_equal(fields[6].data, predicted[2][::-1])
    np.testing.assert_array_equal(fields[18].data, predicted[4][::-1])

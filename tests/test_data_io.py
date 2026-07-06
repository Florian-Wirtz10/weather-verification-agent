import numpy as np

from data_io.synthetic import generate_synthetic_fields


def test_shape_and_nonnegativity():
    forecast, obs = generate_synthetic_fields(shape=(16, 24), seed=1)

    assert forecast.shape == (16, 24)
    assert obs.shape == (16, 24)
    assert np.all(forecast >= 0)
    assert np.all(obs >= 0)


def test_reproducible_with_seed():
    forecast1, obs1 = generate_synthetic_fields(shape=(8, 8), seed=42)
    forecast2, obs2 = generate_synthetic_fields(shape=(8, 8), seed=42)

    np.testing.assert_array_equal(forecast1, forecast2)
    np.testing.assert_array_equal(obs1, obs2)


def test_different_seeds_differ():
    forecast1, _ = generate_synthetic_fields(shape=(8, 8), seed=1)
    forecast2, _ = generate_synthetic_fields(shape=(8, 8), seed=2)

    assert not np.array_equal(forecast1, forecast2)

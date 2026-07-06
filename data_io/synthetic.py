"""Synthetic forecast/observation field generation.

Phase 1 stand-in for real gridded precipitation products: produces a pair
of correlated but non-identical 2D fields so downstream verification code
has something non-trivial to score.
"""

import numpy as np


def generate_synthetic_fields(shape=(64, 64), seed=None):
    """Generate a (forecast, obs) pair of 2D non-negative precipitation fields (mm).

    obs is derived from a shared latent field plus independent noise for
    both forecast and obs, so the two are correlated but not identical.
    """
    rng = np.random.default_rng(seed)

    latent = rng.gamma(shape=2.0, scale=3.0, size=shape)

    forecast = latent + rng.normal(loc=0.0, scale=1.5, size=shape)
    obs = latent + rng.normal(loc=0.0, scale=1.5, size=shape)

    forecast = np.clip(forecast, 0.0, None)
    obs = np.clip(obs, 0.0, None)

    return forecast, obs

"""Keep pyplot state isolated between tests that create scientific figures."""

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_test_figures():
    yield
    plt.close("all")

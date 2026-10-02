# Unit tests for the PSI drift metric.
import numpy as np
from monitor import psi


# Same distribution gives PSI near 0.
def test_psi_same_distribution_is_near_zero():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert psi(a, b) < 0.05


# A clear shift gives PSI above the alert level.
def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)
    assert psi(a, b) > 0.25

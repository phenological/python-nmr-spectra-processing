"""Test that fixtures are working correctly."""

import numpy as np


def test_ppm_fixture(ppm):
    """Test ppm fixture loads correctly."""
    assert ppm.shape == (40806,)
    assert ppm.dtype == np.float64


def test_avo3_fixture(avo3):
    """Test avo3 fixture loads correctly."""
    assert avo3.shape == (3, 40806)
    assert avo3.dtype == np.float64


def test_avo3_tsp_fixture(avo3_tsp, ppm_tsp):
    """Test TSP region fixtures."""
    assert avo3_tsp.shape[0] == 3
    assert avo3_tsp.shape[1] == len(ppm_tsp)
    assert len(ppm_tsp) == 174  # Expected TSP region size


def test_avo3_rand_shift_fixture(avo3_rand_shift, avo3):
    """Test randomly shifted fixture."""
    assert avo3_rand_shift.shape == avo3.shape
    # Third spectrum should be identical to original
    np.testing.assert_array_equal(avo3_rand_shift[2], avo3[2])


def test_synthetic_gaussian(synthetic_gaussian):
    """Test synthetic Gaussian generator."""
    x, y = synthetic_gaussian(center=50, width=10, length=100)
    assert len(x) == 100
    assert len(y) == 100
    # Peak should be at center
    assert np.argmax(y) == 50


def test_synthetic_baseline(synthetic_baseline):
    """Test synthetic baseline generator."""
    x, y = synthetic_baseline(length=1000, order=2)
    assert len(x) == 1000
    assert len(y) == 1000

"""Pytest configuration and fixtures for nmr-spectra-processing tests."""

import numpy as np
import pytest
from pathlib import Path


# Path to test data
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def ppm():
    """Chemical shift scale (40806 points)."""
    return np.load(DATA_DIR / "ppm.npy")


@pytest.fixture
def avo3():
    """Three avocado NMR spectra (3 x 40806).

    First three methanol samples from avocado NMR dataset (ANPC, 2022).
    """
    return np.load(DATA_DIR / "avo3.npy")


@pytest.fixture
def ppm_tsp():
    """Chemical shift scale for TSP region."""
    return np.load(DATA_DIR / "ppm_tsp.npy")


@pytest.fixture
def avo3_tsp():
    """Avocado spectra cropped to TSP region (-0.02 to 0.02 ppm)."""
    return np.load(DATA_DIR / "avo3tsp.npy")


@pytest.fixture
def ppm_ala():
    """Chemical shift scale for alanine region."""
    return np.load(DATA_DIR / "ppm_ala.npy")


@pytest.fixture
def avo3_ala():
    """Avocado spectra cropped to alanine region (1.47 to 1.52 ppm)."""
    return np.load(DATA_DIR / "avo3Ala.npy")


@pytest.fixture
def ppm_start_to_ala():
    """Chemical shift scale from start to alanine."""
    return np.load(DATA_DIR / "ppm_start_to_ala.npy")


@pytest.fixture
def avo3_start_to_ala():
    """Avocado spectra from start to alanine (end=1.52 ppm)."""
    return np.load(DATA_DIR / "avo3StartToAla.npy")


@pytest.fixture
def ppm_sucr_to_end():
    """Chemical shift scale from sucrose to end."""
    return np.load(DATA_DIR / "ppm_sucr_to_end.npy")


@pytest.fixture
def avo3_sucr_to_end():
    """Avocado spectra from sucrose to end (start=5.4 ppm)."""
    return np.load(DATA_DIR / "avo3SucrToEnd.npy")


@pytest.fixture
def avo3_rand_shift():
    """Randomly shifted avocado spectra for alignment testing.

    - Spectrum 1: shifted right by 7 points
    - Spectrum 2: shifted left by 10 points
    - Spectrum 3: no shift (original)
    """
    return np.load(DATA_DIR / "avo3RandShift.npy")


@pytest.fixture
def synthetic_gaussian():
    """Generate synthetic Gaussian peak for testing.

    Returns:
        tuple: (x, y) where x is the axis and y is the Gaussian peak
    """
    def _make_gaussian(center=50, width=10, amplitude=1.0, length=100):
        x = np.arange(length)
        y = amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))
        return x, y
    return _make_gaussian


@pytest.fixture
def synthetic_baseline():
    """Generate synthetic baseline for testing baseline correction.

    Returns:
        function: Function to generate polynomial baseline + noise
    """
    def _make_baseline(length=1000, order=2, noise_level=0.01):
        x = np.linspace(0, 1, length)
        # Polynomial baseline
        coeffs = np.random.randn(order + 1)
        baseline = np.polyval(coeffs, x)
        # Add noise
        noise = np.random.randn(length) * noise_level
        return x, baseline + noise
    return _make_baseline

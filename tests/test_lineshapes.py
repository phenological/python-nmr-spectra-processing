"""Tests for the canonical lineshape definitions."""

import numpy as np
import pytest

from nmr_spectra_processing import (
    area_from_height,
    height_from_area,
    measure_fwhm,
    multiplet,
    pseudo_voigt_fwhm,
    voigt,
    voigt_area,
    voigt_height,
    voigt_profile,
    voigt_with_satellites,
)


@pytest.fixture
def axis():
    return np.linspace(-15, 15, 8192)


# ── parameterisations ─────────────────────────────────────────────────────────

class TestVoigtParameterisations:
    def test_height_peak_equals_height(self, axis):
        # exact at the centre; on a grid the sampled max is within one step
        assert np.isclose(voigt_height(2.5, 7.5, 2.5, 0.1, 0.2), 7.5, rtol=1e-9)
        y = voigt_height(axis, height=7.5, center=2.5, sigma=0.1, gamma=0.2)
        assert np.isclose(y.max(), 7.5, rtol=1e-3)
        assert abs(axis[np.argmax(y)] - 2.5) < 0.05

    def test_area_integral_equals_area(self, axis):
        dx = axis[1] - axis[0]
        y = voigt_area(axis, area=5.0, center=0.0, sigma=0.05, gamma=0.05)
        assert abs(np.sum(y) * dx - 5.0) < 0.05

    def test_profile_unit_peak(self, axis):
        assert np.isclose(voigt_profile(0.0, 0.0, 0.1, 0.1), 1.0, rtol=1e-9)
        y = voigt_profile(axis, center=0.0, sigma=0.1, gamma=0.1)
        assert np.isclose(y.max(), 1.0, rtol=1e-3)

    def test_area_from_height_matches_integral(self):
        x = np.arange(-20, 20, 0.001)
        h, sigma, gamma = 3.0, 0.05, 0.05
        y = voigt_height(x, h, 0.0, sigma, gamma)
        assert np.isclose(np.trapezoid(y, x), area_from_height(h, sigma, gamma),
                          rtol=5e-3)

    def test_gaussian_limit_area(self):
        h, sigma, gamma = 5.0, 0.1, 1e-7
        expected = h * sigma * np.sqrt(2 * np.pi)
        assert np.isclose(area_from_height(h, sigma, gamma), expected, rtol=1e-4)


class TestConversions:
    def test_roundtrip(self):
        h, sigma, gamma = 4.2, 0.02, 0.07
        area = area_from_height(h, sigma, gamma)
        assert np.isclose(height_from_area(area, sigma, gamma), h, rtol=1e-10)


# ── composite shapes ──────────────────────────────────────────────────────────

class TestMultiplet:
    def test_doublet_positions_and_heights(self):
        x = np.linspace(-2, 2, 4001)
        y = multiplet(x, height=6.0, center=0.0, sigma=0.02, gamma=0.01,
                      offsets=(-0.3, 0.3), heights_rel=(1.0, 1.0))
        left, right = y[x < 0], y[x > 0]
        assert abs(x[x < 0][np.argmax(left)] - (-0.3)) < 0.01
        assert abs(x[x > 0][np.argmax(right)] - 0.3) < 0.01
        assert np.isclose(left.max(), right.max(), rtol=1e-6)


class TestSatellites:
    def test_side_peaks_added(self):
        x = np.linspace(-2, 2, 4001)
        common = {"height": 10.0, "center": 0.0, "sigma": 0.01, "gamma": 0.005}
        no_sat = voigt_with_satellites(x, f_sat=0.0, delta_sat=0.0,
                                       sigma_sat=0.01, gamma_sat=0.005, **common)
        with_sat = voigt_with_satellites(x, f_sat=0.1, delta_sat=0.5,
                                         sigma_sat=0.01, gamma_sat=0.005, **common)
        assert with_sat.max() >= no_sat.max()
        assert np.isclose(with_sat.max(), no_sat.max(), rtol=1e-3)
        assert with_sat[np.abs(x - 0.5) < 0.05].max() > 0.5 * 10.0 * 0.1


# ── width helpers ─────────────────────────────────────────────────────────────

class TestWidths:
    def test_pseudo_voigt_fwhm_matches_numeric(self):
        x = np.linspace(-5, 5, 20001)
        sigma, gamma = 0.08, 0.05
        y = voigt_height(x, 1.0, 0.0, sigma, gamma)
        assert np.isclose(measure_fwhm(x, y, 0.0), pseudo_voigt_fwhm(sigma, gamma),
                          rtol=2e-2)

    def test_measure_fwhm_gaussian(self):
        x = np.linspace(-3, 3, 6001)
        sigma = 0.2
        y = np.exp(-0.5 * (x / sigma) ** 2)
        assert np.isclose(measure_fwhm(x, y, 0.0), 2.3548 * sigma, rtol=1e-3)


# ── backward compatibility ────────────────────────────────────────────────────

class TestBackCompat:
    def test_voigt_alias_is_area_normalised(self, axis):
        y1 = voigt(axis, 5.0, 0.0, 0.05, 0.05)
        y2 = voigt_area(axis, 5.0, 0.0, 0.05, 0.05)
        assert np.allclose(y1, y2)

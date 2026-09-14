"""Tests for reference deconvolution and lineshape manipulation."""

import numpy as np
import pytest
from nmr_spectra_processing import (
    broaden,
    measure_fwhm,
    nmr_to_voigt,
    ref_deconv,
    spectrum_from_peaks,
    voigt,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def axis():
    return np.linspace(-15, 15, 8192)


@pytest.fixture
def single_peak_obs(axis):
    """Broad single Lorentzian peak (fwhm_L=0.8, fwhm_G=0.0)."""
    return spectrum_from_peaks(axis, [[0.0, 1.0, 0.8, 0.0]])


@pytest.fixture
def multi_peak_obs(axis):
    """Multiple overlapping peaks with uniform broadening."""
    peaks = [
        [-4.0, 5.0, 0.8, 0.1],
        [ 0.0, 8.0, 0.8, 0.1],
        [ 4.0, 6.0, 0.8, 0.1],
    ]
    return spectrum_from_peaks(axis, peaks)


# ── voigt ─────────────────────────────────────────────────────────────────────

class TestVoigt:
    def test_peak_at_center(self, axis):
        y = voigt(axis, amplitude=1.0, center=2.5, sigma=0.1, gamma=0.2)
        assert abs(axis[np.argmax(y)] - 2.5) < 0.05

    def test_area_normalised(self, axis):
        # Use narrow sigma/gamma so Voigt tails are well within ±15
        dx = axis[1] - axis[0]
        y = voigt(axis, amplitude=5.0, center=0.0, sigma=0.05, gamma=0.05)
        assert abs(np.sum(y) * dx - 5.0) < 0.05

    def test_pure_lorentzian_limit(self, axis):
        """Very small sigma → approaches pure Lorentzian."""
        gamma = 0.5
        y = voigt(axis, amplitude=1.0, center=0.0, sigma=1e-6, gamma=gamma)
        fwhm = measure_fwhm(axis, y, 0.0)
        assert abs(fwhm - 2 * gamma) < 0.05

    def test_positive_values(self, axis):
        y = voigt(axis, amplitude=1.0, center=0.0, sigma=0.2, gamma=0.3)
        assert np.all(y >= 0)


# ── nmr_to_voigt ─────────────────────────────────────────────────────────────

class TestNmrToVoigt:
    def test_returns_four_params(self):
        result = nmr_to_voigt(cs=1.5, area=2.0, fwhm_L=0.4, fwhm_G=0.2)
        assert len(result) == 4

    def test_center_preserved(self):
        _, center, _, _ = nmr_to_voigt(cs=3.7, area=1.0, fwhm_L=0.3, fwhm_G=0.1)
        assert center == 3.7

    def test_area_preserved(self):
        amplitude, _, _, _ = nmr_to_voigt(cs=0.0, area=4.2, fwhm_L=0.3, fwhm_G=0.1)
        assert amplitude == 4.2

    def test_gamma_from_fwhm_L(self):
        _, _, _, gamma = nmr_to_voigt(cs=0.0, area=1.0, fwhm_L=1.0, fwhm_G=0.0)
        assert abs(gamma - 0.5) < 1e-10

    def test_zero_fwhm_G_clamped(self):
        """sigma must remain positive even when fwhm_G=0."""
        _, _, sigma, _ = nmr_to_voigt(cs=0.0, area=1.0, fwhm_L=0.5, fwhm_G=0.0)
        assert sigma > 0


# ── spectrum_from_peaks ───────────────────────────────────────────────────────

class TestSpectrumFromPeaks:
    def test_output_length(self, axis):
        y = spectrum_from_peaks(axis, [[0.0, 1.0, 0.5, 0.1]])
        assert len(y) == len(axis)

    def test_single_peak_position(self, axis):
        y = spectrum_from_peaks(axis, [[3.0, 1.0, 0.5, 0.1]])
        assert abs(axis[np.argmax(y)] - 3.0) < 0.05

    def test_multiple_peaks_sum(self, axis):
        y1 = spectrum_from_peaks(axis, [[-5.0, 1.0, 0.3, 0.0]])
        y2 = spectrum_from_peaks(axis, [[ 5.0, 1.0, 0.3, 0.0]])
        y_both = spectrum_from_peaks(axis, [
            [-5.0, 1.0, 0.3, 0.0],
            [ 5.0, 1.0, 0.3, 0.0],
        ])
        np.testing.assert_allclose(y_both, y1 + y2)

    def test_empty_peaks(self, axis):
        y = spectrum_from_peaks(axis, [])
        np.testing.assert_array_equal(y, np.zeros_like(axis))


# ── measure_fwhm ──────────────────────────────────────────────────────────────

class TestMeasureFwhm:
    def test_lorentzian_fwhm(self, axis):
        fwhm_L = 0.6
        y = spectrum_from_peaks(axis, [[2.0, 1.0, fwhm_L, 0.0]])
        fwhm = measure_fwhm(axis, y, x_center=2.0)
        assert abs(fwhm - fwhm_L) < 0.02

    def test_broader_peak(self, axis):
        y_narrow = spectrum_from_peaks(axis, [[0.0, 1.0, 0.3, 0.0]])
        y_broad  = spectrum_from_peaks(axis, [[0.0, 1.0, 1.2, 0.0]])
        fwhm_narrow = measure_fwhm(axis, y_narrow, 0.0)
        fwhm_broad  = measure_fwhm(axis, y_broad,  0.0)
        assert fwhm_broad > fwhm_narrow

    def test_center_offset(self, axis):
        fwhm_L = 0.5
        center = -6.0
        y = spectrum_from_peaks(axis, [[center, 1.0, fwhm_L, 0.0]])
        fwhm = measure_fwhm(axis, y, x_center=center)
        assert abs(fwhm - fwhm_L) < 0.02

    def test_returns_nan_for_truncated_peak(self):
        """Peak so close to axis edge that one half-max crossing is missing."""
        x = np.linspace(0, 5, 2048)  # axis starts at 0
        # Peak centered at x=0 (left edge) — left crossing is outside axis
        y = spectrum_from_peaks(x, [[0.0, 1.0, 2.0, 0.0]])
        fwhm = measure_fwhm(x, y, x_center=0.0)
        assert np.isnan(fwhm)


# ── ref_deconv ────────────────────────────────────────────────────────────────

class TestRefDeconv:
    def test_output_length(self, axis, single_peak_obs):
        result = ref_deconv(axis, single_peak_obs, fwhm_L_obs=0.8,
                            fwhm_L_target=0.2)
        assert len(result) == len(axis)

    def test_sharpening_reduces_fwhm(self, axis, single_peak_obs):
        fwhm_obs = measure_fwhm(axis, single_peak_obs, x_center=0.0)
        y_sharp = ref_deconv(axis, single_peak_obs, fwhm_L_obs=0.8,
                             fwhm_L_target=0.2, lb=0.1)
        fwhm_sharp = measure_fwhm(axis, y_sharp, x_center=0.0)
        assert fwhm_sharp < fwhm_obs

    def test_output_fwhm_close_to_target(self, axis):
        """With noiseless data and lb=0, output FWHM ≈ fwhm_L_target."""
        fwhm_obs    = 0.8
        fwhm_target = 0.25
        lb          = 0.10
        y_obs = spectrum_from_peaks(axis, [[0.0, 1.0, fwhm_obs, 0.0]])
        y_out = ref_deconv(axis, y_obs, fwhm_L_obs=fwhm_obs,
                           fwhm_L_target=fwhm_target, lb=lb)
        fwhm_out = measure_fwhm(axis, y_out, x_center=0.0)
        expected = fwhm_target + lb
        assert abs(fwhm_out - expected) < 0.10  # within 0.10 ppm

    def test_broadening_increases_fwhm(self, axis, single_peak_obs):
        """fwhm_L_target > fwhm_L_obs should broaden the peak."""
        fwhm_obs = measure_fwhm(axis, single_peak_obs, x_center=0.0)
        y_broad = ref_deconv(axis, single_peak_obs, fwhm_L_obs=0.8,
                             fwhm_L_target=1.5)
        fwhm_broad = measure_fwhm(axis, y_broad, x_center=0.0)
        assert fwhm_broad > fwhm_obs

    def test_peak_position_preserved(self, axis):
        """Peak center should not shift significantly."""
        center = 3.0
        y_obs = spectrum_from_peaks(axis, [[center, 1.0, 0.8, 0.1]])
        y_out = ref_deconv(axis, y_obs, fwhm_L_obs=0.8, fwhm_L_target=0.2, lb=0.15)
        center_out = axis[np.argmax(y_out)]
        assert abs(center_out - center) < 0.1

    def test_area_approximately_preserved(self, axis):
        """Total area should be roughly conserved after sharpening."""
        dx = axis[1] - axis[0]
        y_obs = spectrum_from_peaks(axis, [[0.0, 5.0, 0.8, 0.1]])
        y_out = ref_deconv(axis, y_obs, fwhm_L_obs=0.8, fwhm_L_target=0.2, lb=0.2)
        area_obs = np.sum(y_obs) * dx
        area_out = np.sum(y_out) * dx
        # Allow 20% tolerance (taper + Wiener filter cause some area loss)
        assert abs(area_out - area_obs) / area_obs < 0.20

    def test_noisy_spectrum_sharpens(self, axis):
        """Reference deconvolution should sharpen even with added noise."""
        rng = np.random.default_rng(42)
        y_obs = spectrum_from_peaks(axis, [[0.0, 1.0, 0.8, 0.1]])
        y_noisy = y_obs + rng.normal(0, 0.003, len(axis))
        fwhm_before = measure_fwhm(axis, y_noisy, x_center=0.0)
        y_sharp = ref_deconv(axis, y_noisy, fwhm_L_obs=0.8, fwhm_L_target=0.2, lb=0.2)
        fwhm_after = measure_fwhm(axis, y_sharp, x_center=0.0)
        assert fwhm_after < fwhm_before

    def test_overlapping_peaks_resolved(self, axis):
        """Sharpening should help resolve overlapping peaks."""
        # Two peaks 0.4 ppm apart — very overlapping when fwhm_L=0.8
        peaks = [[-0.2, 1.0, 0.8, 0.0], [0.2, 1.0, 0.8, 0.0]]
        y_obs = spectrum_from_peaks(axis, peaks)
        y_sharp = ref_deconv(axis, y_obs, fwhm_L_obs=0.8, fwhm_L_target=0.15, lb=0.1)
        # In the sharpened spectrum the minimum between the two peaks should
        # be lower relative to the maxima than in the observed spectrum
        center_idx = np.argmin(np.abs(axis))
        valley_obs   = y_obs[center_idx]   / np.max(y_obs)
        valley_sharp = y_sharp[center_idx] / np.max(y_sharp)
        assert valley_sharp < valley_obs


# ── broaden ───────────────────────────────────────────────────────────────────

class TestBroaden:
    def test_output_length(self, axis, single_peak_obs):
        result = broaden(axis, single_peak_obs, extra_fwhm_L=0.5)
        assert len(result) == len(axis)

    def test_broadening_increases_fwhm(self, axis):
        y = spectrum_from_peaks(axis, [[0.0, 1.0, 0.3, 0.0]])
        fwhm_before = measure_fwhm(axis, y, x_center=0.0)
        y_broad = broaden(axis, y, extra_fwhm_L=0.5)
        fwhm_after = measure_fwhm(axis, y_broad, x_center=0.0)
        assert fwhm_after > fwhm_before

    def test_peak_position_preserved(self, axis):
        center = -3.0
        y = spectrum_from_peaks(axis, [[center, 1.0, 0.3, 0.0]])
        y_broad = broaden(axis, y, extra_fwhm_L=0.6)
        center_out = axis[np.argmax(y_broad)]
        assert abs(center_out - center) < 0.1

    def test_more_broadening_wider(self, axis):
        y = spectrum_from_peaks(axis, [[0.0, 1.0, 0.3, 0.0]])
        y_b1 = broaden(axis, y, extra_fwhm_L=0.3)
        y_b2 = broaden(axis, y, extra_fwhm_L=0.9)
        fwhm1 = measure_fwhm(axis, y_b1, x_center=0.0)
        fwhm2 = measure_fwhm(axis, y_b2, x_center=0.0)
        assert fwhm2 > fwhm1

    def test_area_conserved(self, axis):
        """Convolution should conserve total spectral area."""
        dx = axis[1] - axis[0]
        y = spectrum_from_peaks(axis, [[0.0, 3.0, 0.3, 0.0]])
        y_broad = broaden(axis, y, extra_fwhm_L=0.4)
        area_before = np.sum(y) * dx
        area_after  = np.sum(y_broad) * dx
        assert abs(area_after - area_before) / area_before < 0.05

    def test_extra_fwhm_G_also_broadens(self, axis):
        y = spectrum_from_peaks(axis, [[0.0, 1.0, 0.3, 0.0]])
        y_broad = broaden(axis, y, extra_fwhm_L=0.0, extra_fwhm_G=0.5)
        fwhm_before = measure_fwhm(axis, y, x_center=0.0)
        fwhm_after  = measure_fwhm(axis, y_broad, x_center=0.0)
        assert fwhm_after > fwhm_before

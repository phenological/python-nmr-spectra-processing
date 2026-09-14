"""Tests for Voigt reference deconvolution."""

import numpy as np
import pytest

from nmr_spectra_processing import measure_fwhm, ref_deconv_voigt
from nmr_spectra_processing.core.lineshapes import voigt_height


@pytest.fixture
def axis():
    return np.linspace(-15, 15, 8192)


class TestRefDeconvVoigt:
    def test_identity_when_target_equals_obs(self, axis):
        sigma_obs, gamma_obs = 0.2, 0.15
        y = voigt_height(axis, 10.0, 0.0, sigma_obs, gamma_obs)
        out = ref_deconv_voigt(axis, y, sigma_obs, gamma_obs)
        # target defaults to observed -> near-identity on a clean Voigt
        assert np.corrcoef(y, out)[0, 1] > 0.999

    def test_sharpening_reduces_fwhm(self, axis):
        sigma_obs, gamma_obs = 0.3, 0.25
        y = voigt_height(axis, 10.0, 0.0, sigma_obs, gamma_obs)
        out = ref_deconv_voigt(axis, y, sigma_obs, gamma_obs,
                               sigma_target=sigma_obs / 2, gamma_target=gamma_obs / 2)
        assert measure_fwhm(axis, out, 0.0) < measure_fwhm(axis, y, 0.0)

    def test_broadening_increases_fwhm(self, axis):
        sigma_obs, gamma_obs = 0.15, 0.1
        y = voigt_height(axis, 10.0, 0.0, sigma_obs, gamma_obs)
        out = ref_deconv_voigt(axis, y, sigma_obs, gamma_obs,
                               sigma_target=sigma_obs * 2, gamma_target=gamma_obs * 2)
        assert measure_fwhm(axis, out, 0.0) > measure_fwhm(axis, y, 0.0)

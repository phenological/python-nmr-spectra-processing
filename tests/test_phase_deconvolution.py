"""Tests for phase-only reference deconvolution and cross-region phase transfer."""

import numpy as np
import pytest
from scipy.special import wofz

from nmr_spectra_processing import (
    apply_phase_angles,
    compute_phase_angles,
    ref_deconv_from_peak,
)

_SQRT2 = np.sqrt(2.0)


def phased_peak(x, center, sigma, gamma, phi):
    """A Voigt peak with a phase error ``phi`` mixing in the dispersive part.

    phi = 0 gives the pure absorptive (symmetric) Voigt; phi != 0 adds the
    antisymmetric dispersive component, making the peak asymmetric.
    """
    z = ((x - center) + 1j * gamma) / (sigma * _SQRT2)
    return np.real(np.exp(-1j * phi) * wofz(z))


def asymmetry(x, y, center):
    """Signed left/right imbalance, ~0 for a symmetric absorptive peak."""
    left = y[x < center].sum()
    right = y[x > center].sum()
    return (right - left) / max(np.abs(y).sum(), 1e-30)


@pytest.fixture
def axis():
    return np.linspace(-5, 5, 4096)


class TestPhaseOnly:
    def test_symmetrises_distorted_peak(self, axis):
        c, s, g, phi = 0.0, 0.1, 0.1, 0.5
        distorted = phased_peak(axis, c, s, g, phi)
        corrected = ref_deconv_from_peak(axis, distorted, distorted, c, s, g,
                                         lb=0.0, snr_override=1e6)
        assert abs(asymmetry(axis, corrected, c)) < 0.3 * abs(asymmetry(axis, distorted, c))

    def test_identity_when_reference_ideal(self, axis):
        c, s, g = 0.0, 0.1, 0.1
        ideal = phased_peak(axis, c, s, g, 0.0)  # pure absorptive
        corrected = ref_deconv_from_peak(axis, ideal, ideal, c, s, g,
                                         lb=0.0, snr_override=1e6)
        assert np.corrcoef(ideal, corrected)[0, 1] > 0.99

    def test_preserves_integral(self, axis):
        c, s, g, phi = 0.0, 0.1, 0.1, 0.5
        distorted = phased_peak(axis, c, s, g, phi)
        corrected = ref_deconv_from_peak(axis, distorted, distorted, c, s, g,
                                         snr_override=1e6)
        assert np.isclose(np.trapezoid(corrected, axis),
                          np.trapezoid(distorted, axis), rtol=5e-2)

    def test_empty_reference_is_noop(self, axis):
        c, s, g = 0.0, 0.1, 0.1
        spec = phased_peak(axis, c, s, g, 0.5)
        out = ref_deconv_from_peak(axis, spec, np.zeros_like(axis), c, s, g)
        assert np.allclose(out, spec)


class TestCrossAxisTransfer:
    def test_transfer_reduces_asymmetry(self):
        c, s, g, phi = 0.0, 0.1, 0.1, 0.5
        x_ref = np.linspace(-5, 5, 4096)
        ref_peak = phased_peak(x_ref, c, s, g, phi)
        angles, t = compute_phase_angles(x_ref, ref_peak, c, s, g)

        x_b = np.linspace(-5, 5, 2048)  # different, coarser axis
        spec_b = phased_peak(x_b, c, s, g, phi)
        corrected = apply_phase_angles(x_b, spec_b, angles, t, snr_override=1e6)

        assert abs(asymmetry(x_b, corrected, c)) < 0.5 * abs(asymmetry(x_b, spec_b, c))

    def test_empty_reference_gives_zero_angles(self):
        x_ref = np.linspace(-5, 5, 1024)
        angles, _ = compute_phase_angles(x_ref, np.zeros_like(x_ref), 0.0, 0.1, 0.1)
        assert np.allclose(angles, 0.0)

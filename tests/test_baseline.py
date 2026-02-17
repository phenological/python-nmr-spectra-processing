"""Tests for baseline correction functions."""

import numpy as np
import pytest
from nmr_spectra_processing.core import baseline_correction


class TestBaselineCorrectionSingle:
    """Tests for baseline correction on single spectrum."""

    def test_baseline_correction_polynomial_baseline(self):
        """Test baseline correction on spectrum with polynomial baseline."""
        # Create x-axis
        x = np.linspace(0, 10, 1000)

        # Create polynomial baseline
        baseline = 0.1 * x**2 - 0.5 * x + 2.0

        # Add peaks
        spectrum = baseline.copy()
        spectrum += 5.0 * np.exp(-((x - 3.0) ** 2) / 0.1)
        spectrum += 3.0 * np.exp(-((x - 7.0) ** 2) / 0.1)

        # Apply baseline correction
        corrected = baseline_correction(spectrum, lam=1e6, p=0.01)

        # After correction, baseline should be close to zero
        # Check regions without peaks (0-2 ppm, 4-6 ppm, 8-10 ppm)
        blank_regions = (x < 2.0) | ((x > 4.0) & (x < 6.0)) | (x > 8.0)
        baseline_level = np.mean(corrected[blank_regions])

        # Baseline should be near zero
        assert abs(baseline_level) < 0.5

    def test_baseline_correction_default_parameters(self):
        """Test baseline correction with default parameters."""
        # Create spectrum with baseline drift
        x = np.linspace(0, 10, 500)
        baseline = 0.05 * x + 1.0
        spectrum = baseline + np.exp(-((x - 5.0) ** 2) / 0.5)

        corrected = baseline_correction(spectrum)

        # Should return same length
        assert len(corrected) == len(spectrum)
        # Should be different from original
        assert not np.allclose(corrected, spectrum)

    def test_baseline_correction_preserves_peaks(self):
        """Test that baseline correction preserves peak positions."""
        # Create spectrum with baseline and peak
        x = np.linspace(0, 10, 1000)
        baseline = 0.1 * x + 1.0
        peak = 10.0 * np.exp(-((x - 5.0) ** 2) / 0.1)
        spectrum = baseline + peak

        corrected = baseline_correction(spectrum, lam=1e5, p=0.001)

        # Peak position should be approximately preserved (within 1-2 points due to smoothing)
        original_peak_idx = np.argmax(spectrum)
        corrected_peak_idx = np.argmax(corrected)

        assert abs(original_peak_idx - corrected_peak_idx) <= 2

    def test_baseline_correction_different_lambda(self):
        """Test effect of lambda parameter on smoothness."""
        # Create spectrum with noisy baseline
        np.random.seed(42)
        x = np.linspace(0, 10, 500)
        baseline = 0.1 * x + np.random.randn(500) * 0.1
        spectrum = baseline + 5.0 * np.exp(-((x - 5.0) ** 2) / 0.5)

        # Small lambda (less smooth)
        corrected_small = baseline_correction(spectrum, lam=1e3, p=0.01)

        # Large lambda (more smooth)
        corrected_large = baseline_correction(spectrum, lam=1e7, p=0.01)

        # Both should correct baseline but with different smoothness
        assert len(corrected_small) == len(spectrum)
        assert len(corrected_large) == len(spectrum)

    def test_baseline_correction_different_p(self):
        """Test effect of asymmetry parameter p."""
        x = np.linspace(0, 10, 500)
        baseline = 0.1 * x + 1.0
        spectrum = baseline + 5.0 * np.exp(-((x - 5.0) ** 2) / 0.5)

        # Different p values
        corrected_p001 = baseline_correction(spectrum, lam=1e5, p=0.001)
        corrected_p01 = baseline_correction(spectrum, lam=1e5, p=0.01)
        corrected_p05 = baseline_correction(spectrum, lam=1e5, p=0.05)

        # All should work but with different asymmetry
        assert len(corrected_p001) == len(spectrum)
        assert len(corrected_p01) == len(spectrum)
        assert len(corrected_p05) == len(spectrum)


class TestBaselineCorrectionMatrix:
    """Tests for baseline correction on multiple spectra."""

    def test_baseline_correction_matrix(self):
        """Test baseline correction on matrix of spectra."""
        # Create 3 spectra with different baselines
        x = np.linspace(0, 10, 500)
        spectra = np.zeros((3, 500))

        for i in range(3):
            baseline = 0.1 * (i + 1) * x + (i + 1) * 0.5
            peak = 5.0 * np.exp(-((x - 5.0) ** 2) / 0.5)
            spectra[i] = baseline + peak

        corrected = baseline_correction(spectra, lam=1e5, p=0.01)

        assert corrected.shape == spectra.shape
        # Each spectrum should be corrected
        for i in range(3):
            assert not np.allclose(corrected[i], spectra[i])

    def test_baseline_correction_matrix_preserves_shape(self):
        """Test that matrix shape is preserved."""
        spectra = np.random.randn(5, 1000) + 2.0  # Add constant baseline

        corrected = baseline_correction(spectra, lam=1e5, p=0.01)

        assert corrected.shape == (5, 1000)

    def test_baseline_correction_matrix_independent(self):
        """Test that each spectrum is corrected independently."""
        x = np.linspace(0, 10, 500)
        spectra = np.zeros((3, 500))

        # Create very different spectra
        spectra[0] = 0.1 * x + 1.0  # Linear baseline
        spectra[1] = 0.01 * x**2 + 0.5  # Quadratic baseline
        spectra[2] = 5.0 * np.exp(-((x - 5.0) ** 2) / 0.5) + 2.0  # Peak + constant

        corrected = baseline_correction(spectra, lam=1e6, p=0.01)

        assert corrected.shape == spectra.shape
        # All should be corrected
        assert not np.allclose(corrected, spectra)


class TestBaselineCorrectionEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_baseline_correction_constant_spectrum(self):
        """Test baseline correction on constant spectrum."""
        spectrum = np.ones(500) * 2.0

        corrected = baseline_correction(spectrum, lam=1e5, p=0.01)

        # Should remove constant baseline
        assert len(corrected) == len(spectrum)
        # Result should be close to zero
        assert np.mean(np.abs(corrected)) < 0.5

    def test_baseline_correction_zero_spectrum(self):
        """Test baseline correction on zero spectrum."""
        spectrum = np.zeros(500)

        corrected = baseline_correction(spectrum, lam=1e5, p=0.01)

        # Should remain zero
        np.testing.assert_allclose(corrected, spectrum, atol=1e-10)

    def test_baseline_correction_negative_lambda_raises(self):
        """Test that negative lambda raises error."""
        spectrum = np.random.randn(100)

        with pytest.raises(ValueError, match="lam must be positive"):
            baseline_correction(spectrum, lam=-1000)

    def test_baseline_correction_invalid_p_raises(self):
        """Test that invalid p raises error."""
        spectrum = np.random.randn(100)

        # p must be between 0 and 1
        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            baseline_correction(spectrum, p=1.5)

        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            baseline_correction(spectrum, p=-0.1)

    def test_baseline_correction_invalid_niter_raises(self):
        """Test that invalid niter raises error."""
        spectrum = np.random.randn(100)

        with pytest.raises(ValueError, match="niter must be at least 1"):
            baseline_correction(spectrum, niter=0)

    def test_baseline_correction_unknown_method_raises(self):
        """Test that unknown method raises error."""
        spectrum = np.random.randn(100)

        with pytest.raises(ValueError, match="Unknown method"):
            baseline_correction(spectrum, method="unknown")

    def test_baseline_correction_3d_array_raises(self):
        """Test that 3D array raises error."""
        spectra = np.random.randn(5, 10, 100)

        with pytest.raises(ValueError, match="Spectra must be 1D or 2D"):
            baseline_correction(spectra)

    def test_baseline_correction_type_coercion(self):
        """Test that non-array input is coerced."""
        spectrum = [1.0, 2.0, 3.0, 4.0, 5.0]

        corrected = baseline_correction(spectrum, lam=1e3, p=0.01)

        assert isinstance(corrected, np.ndarray)
        assert len(corrected) == 5


class TestBaselineCorrectionRealWorld:
    """Tests with real-world scenarios."""

    def test_baseline_correction_with_avo3_fixtures(self, avo3):
        """Test baseline correction on avo3 data."""
        corrected = baseline_correction(avo3, lam=1e6, p=0.01)

        assert corrected.shape == avo3.shape
        # All values should be finite
        assert np.all(np.isfinite(corrected))

    def test_baseline_correction_reduces_baseline_drift(self):
        """Test that baseline correction reduces drift."""
        # Create spectrum with significant drift
        x = np.linspace(0, 10, 1000)
        baseline = 0.5 * x  # Strong linear drift
        peak = 3.0 * np.exp(-((x - 5.0) ** 2) / 0.2)
        spectrum = baseline + peak

        corrected = baseline_correction(spectrum, lam=1e6, p=0.001)

        # Check that ends of spectrum are closer to zero after correction
        end_mean_before = np.mean(np.abs(spectrum[:100]))
        end_mean_after = np.mean(np.abs(corrected[:100]))

        # Correction should reduce baseline level
        assert end_mean_after < end_mean_before

    def test_baseline_correction_with_multiple_peaks(self):
        """Test baseline correction with multiple peaks."""
        x = np.linspace(0, 10, 1000)
        baseline = 0.1 * x**2 - x + 2.0

        # Add multiple peaks
        spectrum = baseline.copy()
        for center in [2.0, 4.0, 6.0, 8.0]:
            spectrum += 2.0 * np.exp(-((x - center) ** 2) / 0.1)

        corrected = baseline_correction(spectrum, lam=1e6, p=0.01)

        # Peak positions should be preserved
        original_peaks = np.where((spectrum[1:-1] > spectrum[:-2]) &
                                   (spectrum[1:-1] > spectrum[2:]))[0] + 1
        corrected_peaks = np.where((corrected[1:-1] > corrected[:-2]) &
                                    (corrected[1:-1] > corrected[2:]))[0] + 1

        # Should find approximately same number of peaks
        assert abs(len(original_peaks) - len(corrected_peaks)) <= 2

    def test_baseline_correction_different_niter(self):
        """Test effect of number of iterations."""
        x = np.linspace(0, 10, 500)
        baseline = 0.1 * x + 1.0
        spectrum = baseline + 5.0 * np.exp(-((x - 5.0) ** 2) / 0.5)

        # Few iterations
        corrected_2 = baseline_correction(spectrum, lam=1e5, p=0.01, niter=2)

        # Many iterations
        corrected_20 = baseline_correction(spectrum, lam=1e5, p=0.01, niter=20)

        # Both should work but may differ slightly
        assert len(corrected_2) == len(spectrum)
        assert len(corrected_20) == len(spectrum)

    def test_baseline_correction_with_noise(self):
        """Test baseline correction on noisy spectrum."""
        np.random.seed(42)
        x = np.linspace(0, 10, 1000)
        baseline = 0.1 * x + 1.0
        peak = 5.0 * np.exp(-((x - 5.0) ** 2) / 0.5)
        noise = np.random.randn(1000) * 0.1
        spectrum = baseline + peak + noise

        corrected = baseline_correction(spectrum, lam=1e6, p=0.01)

        # Should handle noise gracefully
        assert np.all(np.isfinite(corrected))
        # Peak should still be visible
        assert np.max(corrected) > 2.0

    def test_baseline_correction_idempotence(self):
        """Test that applying correction twice gives diminishing changes."""
        x = np.linspace(0, 10, 500)
        baseline = 0.1 * x + 1.0
        spectrum = baseline + 5.0 * np.exp(-((x - 5.0) ** 2) / 0.5)

        # First correction
        corrected_once = baseline_correction(spectrum, lam=1e6, p=0.01)

        # Second correction
        corrected_twice = baseline_correction(corrected_once, lam=1e6, p=0.01)

        # Change from first correction should be much larger than second
        change_first = np.mean(np.abs(spectrum - corrected_once))
        change_second = np.mean(np.abs(corrected_once - corrected_twice))

        # Second change should be much smaller (most baseline already removed)
        assert change_second < change_first / 2

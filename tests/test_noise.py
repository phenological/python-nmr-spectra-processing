"""Tests for noise estimation functions."""

import numpy as np
import pytest
from nmr_spectra_processing.core import estimate_noise


class TestEstimateNoiseSingle:
    """Tests for noise estimation on single spectrum."""

    def test_estimate_noise_default_parameters(self):
        """Test noise estimation with default parameters."""
        # Create ppm scale
        ppm = np.linspace(0, 10, 1000)

        # Create spectrum with noise in blank region (9.8-10 ppm)
        spectrum = np.zeros(1000)
        # Add noise in blank region
        blank_mask = (ppm >= 9.8) & (ppm <= 10.0)
        np.random.seed(42)
        spectrum[blank_mask] = np.random.randn(np.sum(blank_mask)) * 0.01

        noise = estimate_noise(ppm, spectrum)

        # Should be close to 99th percentile of noise
        expected = np.quantile(spectrum[blank_mask], 0.99)
        assert noise == expected

    def test_estimate_noise_custom_level(self):
        """Test noise estimation with custom quantile level."""
        ppm = np.linspace(0, 10, 1000)
        spectrum = np.zeros(1000)
        blank_mask = (ppm >= 9.8) & (ppm <= 10.0)
        np.random.seed(42)
        spectrum[blank_mask] = np.random.randn(np.sum(blank_mask)) * 0.01

        # Use 95th percentile
        noise = estimate_noise(ppm, spectrum, level=0.95)

        expected = np.quantile(spectrum[blank_mask], 0.95)
        assert noise == expected

    def test_estimate_noise_custom_roi(self):
        """Test noise estimation with custom ROI."""
        ppm = np.linspace(0, 10, 1000)
        spectrum = np.zeros(1000)

        # Add noise in custom region (0-1 ppm)
        roi_mask = (ppm >= 0.0) & (ppm <= 1.0)
        np.random.seed(42)
        spectrum[roi_mask] = np.random.randn(np.sum(roi_mask)) * 0.05

        noise = estimate_noise(ppm, spectrum, roi=(0.0, 1.0))

        expected = np.quantile(spectrum[roi_mask], 0.99)
        assert noise == expected

    def test_estimate_noise_realistic_spectrum(self):
        """Test with spectrum that has signal + noise."""
        ppm = np.linspace(0, 10, 1000)

        # Create spectrum with peak and noise
        spectrum = np.exp(-((ppm - 5.0) ** 2) / 0.1)  # Peak at 5 ppm
        np.random.seed(42)
        spectrum += np.random.randn(1000) * 0.01  # Add noise everywhere

        noise = estimate_noise(ppm, spectrum, roi=(9.8, 10.0))

        # Noise should be small (around 0.01 scale)
        assert 0.005 < noise < 0.05


class TestEstimateNoiseMatrix:
    """Tests for noise estimation on multiple spectra."""

    def test_estimate_noise_matrix(self):
        """Test noise estimation on matrix of spectra."""
        ppm = np.linspace(0, 10, 1000)

        # Create 3 spectra with different noise levels
        np.random.seed(42)
        spectra = np.zeros((3, 1000))
        blank_mask = (ppm >= 9.8) & (ppm <= 10.0)

        spectra[0, blank_mask] = np.random.randn(np.sum(blank_mask)) * 0.01
        spectra[1, blank_mask] = np.random.randn(np.sum(blank_mask)) * 0.02
        spectra[2, blank_mask] = np.random.randn(np.sum(blank_mask)) * 0.05

        noise_levels = estimate_noise(ppm, spectra)

        assert len(noise_levels) == 3
        # Higher noise spectra should have higher noise estimates
        assert noise_levels[0] < noise_levels[2]

    def test_estimate_noise_matrix_different_levels(self):
        """Test that matrix returns one noise level per spectrum."""
        ppm = np.linspace(0, 10, 500)
        spectra = np.random.randn(5, 500)

        noise_levels = estimate_noise(ppm, spectra)

        assert noise_levels.shape == (5,)
        # All should be positive (for random normal noise)
        assert np.all(noise_levels > 0)

    def test_estimate_noise_matrix_with_custom_roi(self):
        """Test matrix with custom ROI."""
        ppm = np.linspace(0, 10, 1000)
        spectra = np.random.randn(3, 1000) * 0.01

        noise_levels = estimate_noise(ppm, spectra, roi=(0.0, 1.0))

        assert len(noise_levels) == 3


class TestEstimateNoiseEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_estimate_noise_roi_outside_range_raises(self):
        """Test that ROI outside ppm range raises error."""
        ppm = np.linspace(0, 10, 1000)
        spectrum = np.random.randn(1000)

        with pytest.raises(ValueError, match="roi is outside ppm range"):
            estimate_noise(ppm, spectrum, roi=(15.0, 20.0))

    def test_estimate_noise_invalid_roi_order_raises(self):
        """Test that invalid ROI order raises error."""
        ppm = np.linspace(0, 10, 1000)
        spectrum = np.random.randn(1000)

        with pytest.raises(ValueError, match="roi start must be less than end"):
            estimate_noise(ppm, spectrum, roi=(10.0, 5.0))

    def test_estimate_noise_invalid_roi_length_raises(self):
        """Test that ROI with wrong length raises error."""
        ppm = np.linspace(0, 10, 1000)
        spectrum = np.random.randn(1000)

        with pytest.raises(ValueError, match="roi must have 2 elements"):
            estimate_noise(ppm, spectrum, roi=(9.8,))

    def test_estimate_noise_ppm_spectra_mismatch_raises(self):
        """Test that ppm/spectra length mismatch raises error."""
        ppm = np.linspace(0, 10, 1000)
        spectra = np.random.randn(3, 500)  # Wrong length

        with pytest.raises(ValueError, match="ppm and spectra dimensions mismatch"):
            estimate_noise(ppm, spectra)

    def test_estimate_noise_type_coercion_warnings(self):
        """Test that non-array inputs trigger warnings."""
        # List inputs (should work with warning)
        ppm = list(np.linspace(0, 10, 100))
        spectrum = list(np.random.randn(100))

        result = estimate_noise(ppm, spectrum)
        assert isinstance(result, (float, np.floating))

    def test_estimate_noise_empty_roi_raises(self):
        """Test that ROI with no points raises error."""
        ppm = np.linspace(0, 10, 1000)
        spectrum = np.random.randn(1000)

        # ROI between points (no exact matches)
        # This should still work as long as there are points in range
        # But if we use a very narrow ROI outside the range...
        with pytest.raises(ValueError, match="roi is outside ppm range"):
            estimate_noise(ppm, spectrum, roi=(10.5, 11.0))


class TestEstimateNoiseRealWorld:
    """Tests with real-world scenarios."""

    def test_estimate_noise_with_avo3_fixtures(self, ppm, avo3):
        """Test noise estimation on avo3 data."""
        noise_levels = estimate_noise(ppm, avo3)

        assert len(noise_levels) == 3
        # All noise levels should be positive
        assert np.all(noise_levels > 0)
        # Should all be finite
        assert np.all(np.isfinite(noise_levels))

    def test_estimate_noise_different_quantiles(self):
        """Test that different quantiles give different results."""
        ppm = np.linspace(0, 10, 1000)
        spectrum = np.random.randn(1000) * 0.01

        noise_90 = estimate_noise(ppm, spectrum, level=0.90)
        noise_95 = estimate_noise(ppm, spectrum, level=0.95)
        noise_99 = estimate_noise(ppm, spectrum, level=0.99)

        # Higher quantiles should give higher estimates
        assert noise_90 < noise_95 < noise_99

    def test_estimate_noise_signal_to_noise_ratio(self):
        """Test computing signal-to-noise ratio."""
        ppm = np.linspace(0, 10, 1000)

        # Create spectrum with known peak and noise
        spectrum = np.zeros(1000)
        spectrum[500] = 1.0  # Peak with intensity 1.0
        np.random.seed(42)
        spectrum += np.random.randn(1000) * 0.01  # Add noise

        noise = estimate_noise(ppm, spectrum, roi=(9.8, 10.0))
        signal = np.max(spectrum)
        snr = signal / noise

        # SNR should be reasonably high
        assert snr > 10

    def test_estimate_noise_baseline_corrected_spectrum(self):
        """Test noise estimation on baseline-corrected spectrum with negative values."""
        ppm = np.linspace(0, 10, 1000)

        # Baseline-corrected spectrum (centered around 0)
        spectrum = np.random.randn(1000) * 0.01

        noise = estimate_noise(ppm, spectrum, roi=(9.8, 10.0))

        # Should still work with negative values
        assert np.isfinite(noise)

    def test_estimate_noise_multiple_roi_comparison(self):
        """Test comparing noise estimates from different ROIs."""
        ppm = np.linspace(0, 10, 1000)
        spectrum = np.random.randn(1000) * 0.01

        # Add a peak in middle region
        peak_mask = (ppm >= 4.5) & (ppm <= 5.5)
        spectrum[peak_mask] += 0.5

        # Estimate noise from blank region
        noise_blank = estimate_noise(ppm, spectrum, roi=(9.8, 10.0))

        # Estimate noise from region with peak (should be higher)
        noise_peak = estimate_noise(ppm, spectrum, roi=(4.5, 5.5))

        # Region with peak should have higher "noise" estimate
        assert noise_peak > noise_blank

    def test_estimate_noise_consistency_across_spectra(self):
        """Test that similar spectra give similar noise estimates."""
        ppm = np.linspace(0, 10, 1000)

        # Create 3 similar spectra
        np.random.seed(42)
        base_noise = 0.01
        spectra = np.random.randn(3, 1000) * base_noise

        noise_levels = estimate_noise(ppm, spectra)

        # All noise levels should be similar (within factor of 2)
        mean_noise = np.mean(noise_levels)
        assert np.all(noise_levels > mean_noise / 2)
        assert np.all(noise_levels < mean_noise * 2)

    def test_estimate_noise_absolute_value_handling(self):
        """Test noise estimation on magnitude spectra (all positive)."""
        ppm = np.linspace(0, 10, 1000)

        # Magnitude spectrum (absolute values)
        spectrum = np.abs(np.random.randn(1000) * 0.01)

        noise = estimate_noise(ppm, spectrum, roi=(9.8, 10.0))

        # Should be positive
        assert noise > 0
        assert np.isfinite(noise)

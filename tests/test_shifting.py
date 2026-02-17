"""Tests for shifting functions."""

import numpy as np
import pytest
from nmr_spectra_processing.core import shift_series, shift_spectra


class TestShiftSeries:
    """Tests for discrete shift_series function."""

    def test_shift_right_with_zeros(self):
        """Test shifting right with zero padding."""
        x = np.arange(10)
        shifted = shift_series(x, shift=3, padding="zeroes")

        assert len(shifted) == 10
        # First 3 should be zeros
        np.testing.assert_array_equal(shifted[:3], [0, 0, 0])
        # Rest should be first 7 elements of x
        np.testing.assert_array_equal(shifted[3:], x[:7])

    def test_shift_left_with_zeros(self):
        """Test shifting left with zero padding."""
        x = np.arange(10)
        shifted = shift_series(x, shift=-2, padding="zeroes")

        assert len(shifted) == 10
        # First 8 should be elements 2-9 of x
        np.testing.assert_array_equal(shifted[:8], x[2:])
        # Last 2 should be zeros
        np.testing.assert_array_equal(shifted[8:], [0, 0])

    def test_shift_zero_returns_copy(self):
        """Test that zero shift returns a copy."""
        x = np.array([1, 2, 3, 4, 5])
        shifted = shift_series(x, shift=0)

        np.testing.assert_array_equal(shifted, x)
        assert shifted is not x  # Should be a copy

    def test_shift_circular(self):
        """Test shifting with circular padding."""
        x = np.arange(10)
        # Shift right by 3 with circular padding
        shifted = shift_series(x, shift=3, padding="circular")

        assert len(shifted) == 10
        # Should be [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
        expected = np.array([7, 8, 9, 0, 1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(shifted, expected)

    def test_shift_sampling(self):
        """Test shifting with sampling padding."""
        np.random.seed(42)
        x = np.arange(100)
        shifted = shift_series(x, shift=5, padding="sampling")

        assert len(shifted) == 100
        # Last 95 should be first 95 elements
        np.testing.assert_array_equal(shifted[5:], x[:95])
        # First 5 should be sampled from last 1/15th
        assert all(93 <= val <= 99 for val in shifted[:5])

    def test_shift_negative_large(self):
        """Test large negative shift."""
        x = np.arange(20)
        shifted = shift_series(x, shift=-10, padding="zeroes")

        assert len(shifted) == 20
        # First 10 should be last 10 of x
        np.testing.assert_array_equal(shifted[:10], x[10:])
        # Last 10 should be zeros
        np.testing.assert_array_equal(shifted[10:], np.zeros(10))

    def test_shift_type_coercion(self):
        """Test that list input is coerced."""
        x = [1, 2, 3, 4, 5]
        shifted = shift_series(x, shift=1, padding="zeroes")

        assert isinstance(shifted, np.ndarray)
        assert len(shifted) == 5


class TestShiftSpectra:
    """Tests for continuous shift_spectra function."""

    def test_shift_single_spectrum_ppm(self):
        """Test shifting single spectrum in ppm."""
        ppm = np.linspace(0, 10, 100)
        # Create a Gaussian peak at 5.0 ppm
        spectrum = np.exp(-((ppm - 5.0) ** 2) / 0.1)

        # Shift by 1.0 ppm (positive = shift right to higher ppm)
        shifted = shift_spectra(ppm, spectrum, shift=1.0)

        assert len(shifted) == len(spectrum)
        # Peak should now be at 6.0 ppm (shifted right)
        original_peak_idx = np.argmax(spectrum)
        shifted_peak_idx = np.argmax(shifted)
        # Peak should move right by ~10 indices (1.0 ppm out of 10 range = 10 points)
        assert abs((shifted_peak_idx - original_peak_idx) - 10) < 2

    def test_shift_single_spectrum_hertz(self):
        """Test shifting single spectrum in Hz."""
        ppm = np.linspace(0, 10, 100)
        spectrum = np.exp(-((ppm - 5.0) ** 2) / 0.1)

        # Shift by 600 Hz at 600 MHz = 1 ppm
        shifted = shift_spectra(ppm, spectrum, shift=600.0, hertz=True, SF=600.0)

        # Should be equivalent to 1 ppm shift (right)
        assert len(shifted) == len(spectrum)
        original_peak_idx = np.argmax(spectrum)
        shifted_peak_idx = np.argmax(shifted)
        assert abs((shifted_peak_idx - original_peak_idx) - 10) < 2

    def test_shift_multiple_spectra_same_shift(self):
        """Test shifting multiple spectra with same shift."""
        ppm = np.linspace(0, 10, 100)
        spectra = np.random.randn(5, 100)

        shifted = shift_spectra(ppm, spectra, shift=0.5)

        assert shifted.shape == spectra.shape
        # Each spectrum should be shifted
        for i in range(5):
            # Verify that shifted spectrum is different from original
            assert not np.allclose(shifted[i], spectra[i])

    def test_shift_multiple_spectra_different_shifts(self):
        """Test shifting multiple spectra with different shifts."""
        ppm = np.linspace(0, 10, 100)
        spectra = np.random.randn(3, 100)
        shifts = np.array([0.0, 0.5, 1.0])

        shifted = shift_spectra(ppm, spectra, shift=shifts)

        assert shifted.shape == spectra.shape
        # First spectrum should be unchanged (shift=0)
        np.testing.assert_allclose(shifted[0], spectra[0], rtol=1e-10)

    def test_shift_recycling_shift(self):
        """Test that shift parameter is recycled."""
        ppm = np.linspace(0, 10, 100)
        spectra = np.random.randn(6, 100)
        shifts = np.array([0.1, 0.2])  # Only 2 shifts for 6 spectra

        shifted = shift_spectra(ppm, spectra, shift=shifts)

        assert shifted.shape == spectra.shape
        # Shifts should be recycled: [0.1, 0.2, 0.1, 0.2, 0.1, 0.2]

    def test_shift_recycling_SF(self):
        """Test that SF parameter is recycled."""
        ppm = np.linspace(0, 10, 100)
        spectra = np.random.randn(4, 100)
        SF_values = np.array([600.0, 800.0])  # Only 2 SF values for 4 spectra

        shifted = shift_spectra(ppm, spectra, shift=600.0, hertz=True, SF=SF_values)

        assert shifted.shape == spectra.shape

    def test_shift_zero_no_change(self):
        """Test that zero shift doesn't change spectrum."""
        ppm = np.linspace(0, 10, 100)
        spectrum = np.sin(ppm)

        shifted = shift_spectra(ppm, spectrum, shift=0.0)

        np.testing.assert_allclose(shifted, spectrum, rtol=1e-10)

    def test_shift_interpolation_methods(self):
        """Test different interpolation methods."""
        ppm = np.linspace(0, 10, 100)
        spectrum = np.exp(-((ppm - 5.0) ** 2) / 0.1)

        # Test different methods
        for method in ["linear", "cubic", "quadratic"]:
            shifted = shift_spectra(ppm, spectrum, shift=0.5, method=method)
            assert len(shifted) == len(spectrum)

    def test_shift_preserves_shape_matrix(self):
        """Test that matrix shape is preserved."""
        ppm = np.linspace(0, 10, 50)
        spectra = np.random.randn(10, 50)

        shifted = shift_spectra(ppm, spectra, shift=0.1)

        assert shifted.shape == (10, 50)

    def test_shift_with_avo3_fixtures(self, ppm, avo3):
        """Test shifting with real avo3 data."""
        # Shift first spectrum by 0.01 ppm
        shifted = shift_spectra(ppm, avo3[0], shift=0.01)

        assert len(shifted) == len(avo3[0])
        # Should be different from original
        assert not np.allclose(shifted, avo3[0])


class TestShiftingEdgeCases:
    """Tests for edge cases in shifting functions."""

    def test_shift_series_single_element(self):
        """Test shift_series with single element."""
        x = np.array([42.0])
        shifted = shift_series(x, shift=1, padding="zeroes")

        assert len(shifted) == 1
        assert shifted[0] == 0.0  # Original element shifted out

    def test_shift_spectra_single_point(self):
        """Test shift_spectra with single point spectrum."""
        ppm = np.array([5.0])
        spectrum = np.array([1.0])

        # Should handle gracefully (though not very useful)
        shifted = shift_spectra(ppm, spectrum, shift=0.1)
        assert len(shifted) == 1

    def test_shift_negative_vs_positive(self):
        """Test that negative and positive shifts are inverse."""
        x = np.arange(20)

        # Shift right then left should approximate original
        shifted_right = shift_series(x, shift=5, padding="circular")
        shifted_back = shift_series(shifted_right, shift=-5, padding="circular")

        np.testing.assert_array_equal(shifted_back, x)

    def test_shift_spectra_out_of_bounds_handling(self):
        """Test that out-of-bounds interpolation is handled (filled with zeros)."""
        ppm = np.linspace(0, 10, 100)
        spectrum = np.ones(100)

        # Large shift that moves everything out of bounds
        shifted = shift_spectra(ppm, spectrum, shift=20.0)

        # Should be mostly zeros
        assert np.mean(shifted) < 0.1

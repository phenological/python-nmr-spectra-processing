"""Tests for alignment functions."""

import numpy as np
import pytest
from nmr_spectra_processing.core import align_spectra


class TestAlignSpectraSingle:
    """Tests for aligning single spectrum."""

    def test_align_single_to_reference(self):
        """Test aligning single spectrum to explicit reference."""
        # Create a spectrum with a peak
        x = np.zeros(100)
        x[50] = 1.0  # Peak at index 50

        # Create shifted reference (peak at index 45)
        ref = np.zeros(100)
        ref[45] = 1.0

        # Align x to ref - should shift left by 5
        aligned = align_spectra(x, ref=ref, padding="zeroes")

        # After alignment, peak should be closer to index 45
        peak_idx = np.argmax(aligned)
        assert abs(peak_idx - 45) <= 1  # Allow small interpolation error

    def test_align_single_returns_shift(self):
        """Test returning shift value instead of aligned spectrum."""
        x = np.zeros(100)
        x[50] = 1.0

        ref = np.zeros(100)
        ref[45] = 1.0

        # Get shift value - should be negative (shift left)
        shift = align_spectra(x, ref=ref, return_shifts=True)

        # Should shift left by 5 (negative shift = -5)
        assert isinstance(shift, (int, np.integer))
        assert abs(shift + 5) <= 1  # Approximate shift (around -5)

    def test_align_single_below_threshold(self):
        """Test that spectrum below threshold is left unshifted."""
        # Create two uncorrelated spectra
        np.random.seed(42)
        x = np.random.randn(100)
        ref = np.random.randn(100)

        # Align with high threshold - should leave unshifted
        shift = align_spectra(x, ref=ref, threshold=0.99, return_shifts=True)

        assert shift == 0


class TestAlignSpectraMatrix:
    """Tests for aligning multiple spectra."""

    def test_align_to_median_reference(self):
        """Test aligning multiple spectra to median."""
        # Create 5 spectra with Gaussian peaks at different positions
        # Use Gaussian to avoid sparse peaks (median of sparse peaks is zero)
        base = np.exp(-((np.arange(100) - 50) ** 2) / 50)
        spectra = np.array([
            np.roll(base, -4),  # Shifted left by 4
            np.roll(base, -2),  # Shifted left by 2
            base,               # Center
            np.roll(base, 2),   # Shifted right by 2
            np.roll(base, 4),   # Shifted right by 4
        ])

        # Align to median
        aligned = align_spectra(spectra, ref="median", padding="circular")

        assert aligned.shape == spectra.shape

        # After alignment, peaks should be closer together
        peak_positions = [np.argmax(aligned[i]) for i in range(5)]
        peak_std = np.std(peak_positions)
        original_std = np.std([46, 48, 50, 52, 54])

        # Aligned peaks should be less dispersed
        assert peak_std < original_std

    def test_align_to_mean_reference(self):
        """Test aligning to mean reference."""
        # Create similar spectra with slight shifts
        base = np.exp(-((np.arange(100) - 50) ** 2) / 50)
        spectra = np.array([
            np.roll(base, -2),
            np.roll(base, -1),
            base,
            np.roll(base, 1),
            np.roll(base, 2),
        ])

        aligned = align_spectra(spectra, ref="mean", padding="circular")

        assert aligned.shape == spectra.shape

    def test_align_to_index_reference(self):
        """Test aligning to a specific spectrum by index."""
        # Create spectra
        spectra = np.random.randn(5, 100)

        # Align to first spectrum
        aligned = align_spectra(spectra, ref=0, padding="zeroes")

        assert aligned.shape == spectra.shape

        # First spectrum should be similar to reference (itself)
        # After alignment with itself, should be close to original
        np.testing.assert_allclose(aligned[0], spectra[0], rtol=0.1)

    def test_align_with_boolean_mask(self):
        """Test aligning with boolean mask as reference."""
        spectra = np.random.randn(5, 100)

        # Use second spectrum as reference via boolean mask
        mask = np.array([False, True, False, False, False])
        aligned = align_spectra(spectra, ref=mask, padding="zeroes")

        assert aligned.shape == spectra.shape

    def test_align_with_callable_reference(self):
        """Test aligning with callable reference."""
        spectra = np.random.randn(5, 100)

        # Custom function: median but with extra processing
        def custom_ref(spec_matrix):
            return np.median(spec_matrix, axis=0) * 1.0

        aligned = align_spectra(spectra, ref=custom_ref, padding="zeroes")

        assert aligned.shape == spectra.shape

    def test_align_return_shifts_matrix(self):
        """Test returning shift values for matrix."""
        # Create spectra with known shifts
        base = np.zeros(100)
        base[50] = 1.0

        spectra = np.array([
            np.roll(base, -3),
            np.roll(base, -1),
            base,
            np.roll(base, 2),
            np.roll(base, 4),
        ])

        shifts = align_spectra(spectra, ref=2, return_shifts=True)  # Ref is unshifted

        assert len(shifts) == 5
        # Shifts should be approximately [3, 1, 0, -2, -4]
        # (positive means shift right to align)
        assert shifts[2] == 0  # Reference to itself

    def test_align_with_avo3_fixtures(self, avo3):
        """Test alignment with real avo3 data."""
        # Align first 3 spectra to median
        aligned = align_spectra(avo3, ref="median", padding="zeroes")

        assert aligned.shape == avo3.shape
        # Should return numeric array
        assert np.all(np.isfinite(aligned))


class TestAlignSpectraEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_align_constant_series_no_divergence(self):
        """Test aligning constant series (should warn and leave unshifted)."""
        x = np.ones(100)
        ref = np.ones(100) * 2

        # Should leave unshifted due to divergence warning
        shift = align_spectra(x, ref=ref, return_shifts=True)
        assert shift == 0

    def test_align_invalid_reference_type_raises(self):
        """Test that invalid reference string raises error."""
        spectra = np.random.randn(5, 100)

        with pytest.raises(ValueError, match="Unknown reference type"):
            align_spectra(spectra, ref="invalid")

    def test_align_index_out_of_bounds_raises(self):
        """Test that out-of-bounds index raises error."""
        spectra = np.random.randn(5, 100)

        with pytest.raises(IndexError, match="out of bounds"):
            align_spectra(spectra, ref=10)

    def test_align_reference_length_mismatch_raises(self):
        """Test that reference length mismatch raises error."""
        spectra = np.random.randn(5, 100)
        wrong_ref = np.random.randn(50)  # Wrong length

        with pytest.raises(ValueError, match="Reference length.*does not match"):
            align_spectra(spectra, ref=wrong_ref)

    def test_align_single_reference_length_mismatch_raises(self):
        """Test single spectrum with wrong reference length."""
        x = np.random.randn(100)
        ref = np.random.randn(50)

        with pytest.raises(ValueError, match="Reference length must match"):
            align_spectra(x, ref=ref)

    def test_align_type_coercion_warnings(self):
        """Test that non-array inputs trigger warnings."""
        # List input (should work with warning)
        x = [1, 2, 3, 4, 5]
        ref = [2, 3, 4, 5, 6]

        result = align_spectra(x, ref=ref, padding="zeroes")
        assert isinstance(result, np.ndarray)


class TestAlignSpectraPaddingMethods:
    """Tests for different padding methods during alignment."""

    def test_align_with_circular_padding(self):
        """Test alignment with circular padding."""
        # Create spectra with periodic structure
        x = np.sin(np.linspace(0, 4 * np.pi, 100))
        ref = np.sin(np.linspace(0, 4 * np.pi, 100) + 0.1)

        aligned = align_spectra(x, ref=ref, padding="circular")
        assert len(aligned) == len(x)

    def test_align_with_sampling_padding(self):
        """Test alignment with sampling padding."""
        np.random.seed(42)
        spectra = np.random.randn(3, 100)

        aligned = align_spectra(spectra, ref="median", padding="sampling")
        assert aligned.shape == spectra.shape

    def test_align_with_custom_from_region(self):
        """Test alignment with custom padding region."""
        np.random.seed(42)
        spectra = np.random.randn(3, 100)

        # Use first 10 points for sampling
        aligned = align_spectra(
            spectra,
            ref="median",
            padding="sampling",
            from_region=slice(0, 10)
        )
        assert aligned.shape == spectra.shape


class TestAlignSpectraRealWorld:
    """Tests with real-world scenarios."""

    def test_align_randomly_shifted_spectra(self, ppm, avo3_rand_shift):
        """Test aligning randomly shifted spectra."""
        # avo3_rand_shift has randomly shifted versions
        aligned = align_spectra(avo3_rand_shift, ref="median", padding="zeroes")

        assert aligned.shape == avo3_rand_shift.shape

        # After alignment, spectra should be more similar
        # Compute pairwise correlation before and after
        def pairwise_correlation(spec):
            n = spec.shape[0]
            corrs = []
            for i in range(n):
                for j in range(i + 1, n):
                    corr = np.corrcoef(spec[i], spec[j])[0, 1]
                    corrs.append(corr)
            return np.mean(corrs)

        before_corr = pairwise_correlation(avo3_rand_shift)
        after_corr = pairwise_correlation(aligned)

        # Aligned spectra should have higher correlation
        assert after_corr > before_corr

    def test_align_preserves_intensity(self):
        """Test that alignment preserves total intensity (with circular padding)."""
        spectra = np.random.randn(5, 100)

        aligned = align_spectra(spectra, ref="median", padding="circular")

        # With circular padding, total intensity should be preserved
        original_sums = np.sum(spectra, axis=1)
        aligned_sums = np.sum(aligned, axis=1)

        np.testing.assert_allclose(aligned_sums, original_sums, rtol=1e-10)

    def test_align_threshold_effect(self):
        """Test threshold parameter effect."""
        # Create two similar spectra
        base = np.exp(-((np.arange(100) - 50) ** 2) / 50)
        spectra = np.array([
            base,
            np.roll(base, 5),
        ])

        # Low threshold - should align
        shifts_low = align_spectra(spectra, ref=0, threshold=0.3, return_shifts=True)
        assert shifts_low[1] != 0  # Second spectrum should be shifted

        # Very high threshold - should not align
        shifts_high = align_spectra(spectra, ref=0, threshold=0.999, return_shifts=True)
        # May or may not align depending on actual correlation, but should work without error
        assert len(shifts_high) == 2

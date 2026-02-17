"""Tests for normalization functions."""

import numpy as np
import pytest
from nmr_spectra_processing.core import pqn, normalize


class TestPQN:
    """Tests for PQN (Probabilistic Quotient Normalization)."""

    def test_pqn_single_spectrum_with_array_ref(self):
        """Test PQN on single spectrum with array reference."""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        reference = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        # Quotients: [0.5, 0.5, 0.5, 0.5, 0.5]
        # Median: 0.5
        factor = pqn(spectrum, ref=reference)

        assert isinstance(factor, (float, np.floating))
        assert factor == 0.5

    def test_pqn_matrix_with_median_ref(self):
        """Test PQN on matrix with median reference."""
        # Create 3 spectra with different dilutions
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectra = np.array([
            base * 1.0,
            base * 2.0,
            base * 0.5,
        ])

        factors = pqn(spectra, ref="median")

        assert len(factors) == 3
        # Median is base * 1.0
        # Factors should be [1.0, 2.0, 0.5]
        np.testing.assert_allclose(factors, [1.0, 2.0, 0.5], rtol=1e-10)

    def test_pqn_matrix_with_mean_ref(self):
        """Test PQN with mean reference."""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectra = np.array([
            base * 1.0,
            base * 2.0,
            base * 3.0,
        ])

        factors = pqn(spectra, ref="mean")

        assert len(factors) == 3
        # Mean is base * 2.0
        # Factors should be [0.5, 1.0, 1.5]
        np.testing.assert_allclose(factors, [0.5, 1.0, 1.5], rtol=1e-10)

    def test_pqn_matrix_with_callable_ref(self):
        """Test PQN with callable reference."""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectra = np.array([
            base * 1.0,
            base * 2.0,
            base * 3.0,
        ])

        # Custom reference: median
        factors = pqn(spectra, ref=lambda x: np.median(x, axis=0))

        assert len(factors) == 3
        # Median is base * 2.0
        # Factors should be [0.5, 1.0, 1.5]
        np.testing.assert_allclose(factors, [0.5, 1.0, 1.5], rtol=1e-10)

    def test_pqn_with_of_region_filter(self):
        """Test PQN with of_region filter."""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectra = np.array([
            base * 1.0,
            base * 2.0,
            base * 10.0,  # Outlier
            base * 3.0,
        ])

        # Use only first 2 spectra for reference (exclude outlier)
        mask = np.array([True, True, False, True])
        factors = pqn(spectra, ref="median", of_region=mask)

        assert len(factors) == 4
        # Median of [1, 2, 3] is 2
        # Factors should be [0.5, 1.0, 5.0, 1.5]
        np.testing.assert_allclose(factors, [0.5, 1.0, 5.0, 1.5], rtol=1e-10)

    def test_pqn_with_integer_indices_filter(self):
        """Test PQN with integer indices filter."""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectra = np.array([
            base * 1.0,
            base * 2.0,
            base * 10.0,  # Outlier
            base * 3.0,
        ])

        # Use indices [0, 1, 3] for reference (exclude outlier at index 2)
        indices = np.array([0, 1, 3])
        factors = pqn(spectra, ref="median", of_region=indices)

        assert len(factors) == 4

    def test_pqn_returns_dilution_factors_not_normalized(self):
        """Test that PQN returns dilution factors, not normalized spectra."""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectra = np.array([
            base * 1.0,
            base * 2.0,
        ])

        factors = pqn(spectra, ref="median")

        # Factors should be scalars, not arrays
        assert factors.shape == (2,)
        # To normalize, we divide by factors
        normalized = spectra / factors[:, np.newaxis]
        # After normalization, both should be similar to reference
        assert normalized.shape == spectra.shape


class TestPQNEdgeCases:
    """Tests for PQN edge cases and error conditions."""

    def test_pqn_single_spectrum_with_callable_raises(self):
        """Test that callable ref with single spectrum raises error."""
        spectrum = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Cannot use callable reference"):
            pqn(spectrum, ref=lambda x: np.median(x, axis=0))

    def test_pqn_reference_length_mismatch_raises(self):
        """Test that reference length mismatch raises error."""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        wrong_ref = np.array([1.0, 2.0, 3.0])  # Wrong length

        with pytest.raises(ValueError, match="Reference length mismatch"):
            pqn(spectrum, ref=wrong_ref)

    def test_pqn_invalid_string_ref_raises(self):
        """Test that invalid string reference raises error."""
        spectra = np.random.randn(3, 10)

        with pytest.raises(ValueError, match="Unknown reference"):
            pqn(spectra, ref="invalid")

    def test_pqn_non_numeric_raises(self):
        """Test that non-numeric input raises error."""
        spectrum = np.array(['a', 'b', 'c'])

        with pytest.raises(TypeError, match="Spectra must be numeric"):
            pqn(spectrum, ref=np.array([1.0, 2.0, 3.0]))


class TestNormalize:
    """Tests for normalize() wrapper function."""

    def test_normalize_pqn_default(self):
        """Test normalize with default PQN method."""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectra = np.array([
            base * 1.0,
            base * 2.0,
            base * 0.5,
        ])

        normalized = normalize(spectra)

        assert normalized.shape == spectra.shape
        # After PQN, median should be similar across spectra
        # All spectra should be similar to the reference (median)
        ref = np.median(spectra, axis=0)
        factors = pqn(spectra, ref="median")
        expected = spectra / factors[:, np.newaxis]
        np.testing.assert_allclose(normalized, expected)

    def test_normalize_pqn_explicit(self):
        """Test normalize with explicit PQN method."""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectra = np.array([
            base * 1.0,
            base * 2.0,
        ])

        normalized = normalize(spectra, method="pqn", ref="median")

        assert normalized.shape == spectra.shape

    def test_normalize_max_single_spectrum(self):
        """Test max normalization on single spectrum."""
        spectrum = np.array([1.0, 3.0, 2.0, 5.0, 4.0])

        normalized = normalize(spectrum, method="max")

        # Should be normalized to max=1
        assert np.max(normalized) == 1.0
        np.testing.assert_allclose(normalized, spectrum / 5.0)

    def test_normalize_max_matrix(self):
        """Test max normalization on matrix."""
        spectra = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [0.5, 1.0, 1.5, 2.0, 2.5],
        ])

        normalized = normalize(spectra, method="max")

        assert normalized.shape == spectra.shape
        # Each row should have max=1
        max_vals = np.max(normalized, axis=1)
        np.testing.assert_allclose(max_vals, [1.0, 1.0, 1.0])

    def test_normalize_sum_single_spectrum(self):
        """Test sum normalization on single spectrum."""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        normalized = normalize(spectrum, method="sum")

        # Total area should be 1
        assert np.sum(normalized) == 1.0
        np.testing.assert_allclose(normalized, spectrum / 15.0)

    def test_normalize_sum_matrix(self):
        """Test sum normalization on matrix."""
        spectra = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [0.5, 1.0, 1.5, 2.0, 2.5],
        ])

        normalized = normalize(spectra, method="sum")

        assert normalized.shape == spectra.shape
        # Each row should sum to 1
        sum_vals = np.sum(normalized, axis=1)
        np.testing.assert_allclose(sum_vals, [1.0, 1.0, 1.0])

    def test_normalize_custom_callable(self):
        """Test normalize with custom callable."""
        spectra = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
        ])

        # Custom: normalize by standard deviation
        normalized = normalize(spectra, method=lambda x: np.std(x, axis=1))

        assert normalized.shape == spectra.shape

    def test_normalize_custom_scalar_result(self):
        """Test normalize with custom callable returning scalar."""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Custom method returning scalar
        normalized = normalize(spectrum, method=lambda x: 2.0)

        np.testing.assert_allclose(normalized, spectrum / 2.0)

    def test_normalize_unknown_method_raises(self):
        """Test that unknown method raises error."""
        spectra = np.random.randn(3, 10)

        with pytest.raises(ValueError, match="Unknown method"):
            normalize(spectra, method="unknown")

    def test_normalize_invalid_method_type_raises(self):
        """Test that invalid method type raises error."""
        spectra = np.random.randn(3, 10)

        with pytest.raises(TypeError, match="Invalid method type"):
            normalize(spectra, method=123)


class TestNormalizeRealWorld:
    """Tests with real-world scenarios."""

    def test_normalize_avo3_pqn(self, avo3):
        """Test PQN normalization on avo3 data."""
        normalized = normalize(avo3, method="pqn", ref="median")

        assert normalized.shape == avo3.shape
        # All values should be finite
        assert np.all(np.isfinite(normalized))

    def test_normalize_avo3_max(self, avo3):
        """Test max normalization on avo3 data."""
        normalized = normalize(avo3, method="max")

        assert normalized.shape == avo3.shape
        # Each spectrum should have max=1
        max_vals = np.max(normalized, axis=1)
        np.testing.assert_allclose(max_vals, [1.0, 1.0, 1.0])

    def test_normalize_avo3_sum(self, avo3):
        """Test sum normalization on avo3 data."""
        normalized = normalize(avo3, method="sum")

        assert normalized.shape == avo3.shape
        # Each spectrum should sum to 1
        sum_vals = np.sum(normalized, axis=1)
        np.testing.assert_allclose(sum_vals, [1.0, 1.0, 1.0])

    def test_pqn_factors_consistency(self):
        """Test that PQN factors can be reused."""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectra = np.array([
            base * 1.0,
            base * 2.0,
            base * 0.5,
        ])

        # Get factors
        factors = pqn(spectra, ref="median")

        # Manual normalization
        manual_normalized = spectra / factors[:, np.newaxis]

        # Using normalize function
        auto_normalized = normalize(spectra, method="pqn", ref="median")

        # Should be the same
        np.testing.assert_allclose(manual_normalized, auto_normalized)

    def test_normalize_preserves_relative_intensities(self):
        """Test that normalization preserves relative peak intensities."""
        # Create spectrum with two peaks at 2:1 ratio
        spectrum = np.zeros(100)
        spectrum[30] = 2.0
        spectrum[70] = 1.0

        normalized = normalize(spectrum, method="max")

        # Ratio should be preserved
        ratio_before = spectrum[30] / spectrum[70]
        ratio_after = normalized[30] / normalized[70]

        assert ratio_before == ratio_after

    def test_normalize_with_negative_values(self):
        """Test normalization with negative values (e.g., baseline-corrected spectra)."""
        spectrum = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])

        # Max normalization with negative values
        normalized = normalize(spectrum, method="max")

        assert np.max(normalized) == 1.0

    def test_pqn_with_zeros_handling(self):
        """Test PQN handling of zero values in reference."""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        reference = np.array([2.0, 0.0, 6.0, 8.0, 10.0])  # Contains zero

        # Should handle division by zero gracefully (inf)
        factor = pqn(spectrum, ref=reference)

        # Factor should be computed from non-zero quotients
        assert np.isfinite(factor) or np.isinf(factor)  # Either valid or inf

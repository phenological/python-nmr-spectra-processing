"""Tests for padding function."""

import numpy as np
import pytest
from nmr_spectra_processing.core import pad_series


class TestPaddingZeroes:
    """Tests for padding with zeros."""

    def test_pad_right_with_zeros(self):
        """Test padding on the right with zeros."""
        x = np.arange(10)
        padded = pad_series(x, n=3, side=1, method="zeroes")

        assert len(padded) == 13
        np.testing.assert_array_equal(padded[:10], x)
        np.testing.assert_array_equal(padded[10:], [0, 0, 0])

    def test_pad_left_with_zeros(self):
        """Test padding on the left with zeros."""
        x = np.arange(10)
        padded = pad_series(x, n=5, side=-1, method="zeroes")

        assert len(padded) == 15
        np.testing.assert_array_equal(padded[:5], [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(padded[5:], x)

    def test_pad_both_with_zeros(self):
        """Test padding on both sides with zeros."""
        x = np.arange(10)
        padded = pad_series(x, n=2, side=0, method="zeroes")

        assert len(padded) == 14
        np.testing.assert_array_equal(padded[:2], [0, 0])
        np.testing.assert_array_equal(padded[2:12], x)
        np.testing.assert_array_equal(padded[12:], [0, 0])

    def test_pad_zeros_default_parameters(self):
        """Test padding with default parameters (right, zeros)."""
        x = np.array([1.0, 2.0, 3.0])
        padded = pad_series(x, n=2)

        assert len(padded) == 5
        np.testing.assert_array_equal(padded[:3], x)
        np.testing.assert_array_equal(padded[3:], [0, 0])


class TestPaddingCircular:
    """Tests for circular padding."""

    def test_pad_right_circular(self):
        """Test circular padding on the right."""
        x = np.arange(10)
        padded = pad_series(x, n=3, side=1, method="circular")

        assert len(padded) == 13
        np.testing.assert_array_equal(padded[:10], x)
        np.testing.assert_array_equal(padded[10:], [0, 1, 2])

    def test_pad_left_circular(self):
        """Test circular padding on the left."""
        x = np.arange(10)
        padded = pad_series(x, n=4, side=-1, method="circular")

        assert len(padded) == 14
        np.testing.assert_array_equal(padded[:4], [6, 7, 8, 9])
        np.testing.assert_array_equal(padded[4:], x)

    def test_pad_circular_full_wrap(self):
        """Test circular padding with n equal to length."""
        x = np.array([1, 2, 3, 4, 5])
        padded = pad_series(x, n=5, side=1, method="circular")

        assert len(padded) == 10
        np.testing.assert_array_equal(padded[:5], x)
        np.testing.assert_array_equal(padded[5:], x)

    def test_pad_circular_n_exceeds_length_raises(self):
        """Test that circular padding with n > length raises error."""
        x = np.arange(5)

        with pytest.raises(ValueError, match="n cannot exceed series length"):
            pad_series(x, n=10, side=1, method="circular")


class TestPaddingSampling:
    """Tests for sampling padding."""

    def test_pad_sampling_default_region(self):
        """Test sampling padding with default region (last 1/15th)."""
        np.random.seed(42)
        x = np.arange(100)
        padded = pad_series(x, n=5, side=1, method="sampling")

        assert len(padded) == 105
        np.testing.assert_array_equal(padded[:100], x)
        # Sampled values should be from last 1/15th (indices 93-99)
        assert all(93 <= val <= 99 for val in padded[100:])

    def test_pad_sampling_left(self):
        """Test sampling padding on the left."""
        np.random.seed(42)
        x = np.arange(100)
        padded = pad_series(x, n=3, side=-1, method="sampling")

        assert len(padded) == 103
        np.testing.assert_array_equal(padded[3:], x)
        # First 3 should be sampled from last 1/15th
        assert all(93 <= val <= 99 for val in padded[:3])

    def test_pad_sampling_both(self):
        """Test sampling padding on both sides."""
        np.random.seed(42)
        x = np.arange(50)
        padded = pad_series(x, n=2, side=0, method="sampling")

        assert len(padded) == 54
        np.testing.assert_array_equal(padded[2:52], x)

    def test_pad_sampling_custom_region_slice(self):
        """Test sampling with custom region as slice."""
        np.random.seed(42)
        x = np.arange(100)
        # Sample from first 10 elements
        padded = pad_series(x, n=5, side=1, method="sampling",
                           from_region=slice(0, 10))

        assert len(padded) == 105
        # Sampled values should be from 0-9
        assert all(0 <= val <= 9 for val in padded[100:])

    def test_pad_sampling_custom_region_bool_mask(self):
        """Test sampling with custom boolean mask."""
        np.random.seed(42)
        x = np.arange(20)
        # Sample from middle region (indices 5-15)
        mask = np.zeros(20, dtype=bool)
        mask[5:15] = True

        padded = pad_series(x, n=4, side=1, method="sampling",
                           from_region=mask)

        assert len(padded) == 24
        # Sampled values should be from 5-14
        assert all(5 <= val <= 14 for val in padded[20:])

    def test_pad_sampling_custom_region_indices(self):
        """Test sampling with custom integer indices."""
        np.random.seed(42)
        x = np.arange(50)
        # Sample from specific indices
        indices = np.array([10, 20, 30, 40])

        padded = pad_series(x, n=6, side=1, method="sampling",
                           from_region=indices)

        assert len(padded) == 56
        # Sampled values should be from the specified indices
        assert all(val in [10, 20, 30, 40] for val in padded[50:])

    def test_pad_sampling_invalid_region_uses_default(self):
        """Test that invalid region falls back to default."""
        np.random.seed(42)
        x = np.arange(100)

        # Invalid boolean mask (wrong length)
        mask = np.array([True, False, True])
        padded = pad_series(x, n=5, side=1, method="sampling",
                           from_region=mask)

        # Should still work (using default region)
        assert len(padded) == 105


class TestPaddingEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_pad_single_element(self):
        """Test padding single element array."""
        x = np.array([42.0])
        padded = pad_series(x, n=3, side=1, method="zeroes")

        assert len(padded) == 4
        assert padded[0] == 42.0
        np.testing.assert_array_equal(padded[1:], [0, 0, 0])

    def test_pad_n_zero(self):
        """Test padding with n=0 returns original."""
        x = np.arange(10)
        padded = pad_series(x, n=0, side=1, method="zeroes")

        np.testing.assert_array_equal(padded, x)

    def test_pad_invalid_side_raises(self):
        """Test that invalid side value raises error."""
        x = np.arange(10)

        with pytest.raises(ValueError, match="side must be -1, 0, or 1"):
            pad_series(x, n=5, side=2, method="zeroes")

    def test_pad_unknown_method_raises(self):
        """Test that unknown method raises error."""
        x = np.arange(10)

        with pytest.raises(ValueError, match="Unknown method"):
            pad_series(x, n=5, side=1, method="unknown")

    def test_pad_type_coercion_list_input(self):
        """Test that list input is coerced to numpy array."""
        x = [1, 2, 3, 4, 5]
        padded = pad_series(x, n=2, side=1, method="zeroes")

        assert isinstance(padded, np.ndarray)
        assert len(padded) == 7

    def test_pad_float_n_converted_to_int(self):
        """Test that float n is converted to int."""
        x = np.arange(10)
        padded = pad_series(x, n=3.7, side=1, method="zeroes")

        # Should use n=3 (truncated)
        assert len(padded) == 13

    def test_pad_preserves_dtype(self):
        """Test that padding preserves data type."""
        x = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        padded = pad_series(x, n=2, side=1, method="zeroes")

        assert padded.dtype == np.float32

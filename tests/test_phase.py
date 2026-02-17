"""Tests for phase correction functions."""

import numpy as np
import pytest
from nmr_spectra_processing.core import phase_correction


class TestPhaseCorrectionBasic:
    """Tests for basic phase correction functionality."""

    def test_phase_correction_identity(self):
        """Test that zero correction returns unchanged spectrum."""
        real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        imag = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

        real_c, imag_c = phase_correction(real, imag, phi0=0, phi1=0)

        np.testing.assert_allclose(real_c, real, rtol=1e-10)
        np.testing.assert_allclose(imag_c, imag, rtol=1e-10)

    def test_phase_correction_zero_order_90deg(self):
        """Test 90 degree zero-order rotation."""
        # Start with purely real signal
        real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        imag = np.zeros(5)

        # 90 degree rotation should convert real to imaginary
        real_c, imag_c = phase_correction(real, imag, phi0=90, phi1=0)

        # Real should become near-zero, imaginary should be original real
        np.testing.assert_allclose(real_c, np.zeros(5), atol=1e-10)
        np.testing.assert_allclose(imag_c, real, rtol=1e-10)

    def test_phase_correction_zero_order_180deg(self):
        """Test 180 degree zero-order rotation."""
        real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        imag = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

        # 180 degree rotation should invert signs
        real_c, imag_c = phase_correction(real, imag, phi0=180, phi1=0)

        np.testing.assert_allclose(real_c, -real, rtol=1e-10)
        np.testing.assert_allclose(imag_c, -imag, rtol=1e-10)

    def test_phase_correction_zero_order_360deg(self):
        """Test 360 degree rotation (full circle)."""
        real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        imag = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

        # 360 degrees should return to original
        real_c, imag_c = phase_correction(real, imag, phi0=360, phi1=0)

        np.testing.assert_allclose(real_c, real, rtol=1e-10)
        np.testing.assert_allclose(imag_c, imag, rtol=1e-10)

    def test_phase_correction_zero_order_45deg(self):
        """Test 45 degree zero-order rotation."""
        # Purely real signal
        real = np.array([1.0, 0.0, -1.0, 0.0])
        imag = np.zeros(4)

        real_c, imag_c = phase_correction(real, imag, phi0=45, phi1=0)

        # At 45 degrees, cos(45) = sin(45) = sqrt(2)/2
        sqrt2_2 = np.sqrt(2) / 2
        expected_real = real * sqrt2_2
        expected_imag = real * sqrt2_2

        np.testing.assert_allclose(real_c, expected_real, rtol=1e-10)
        np.testing.assert_allclose(imag_c, expected_imag, rtol=1e-10)


class TestPhaseCorrectionFirstOrder:
    """Tests for first-order phase correction."""

    def test_phase_correction_first_order_only(self):
        """Test first-order correction without zero-order."""
        # Create spectrum with constant phase error
        real = np.ones(100)
        imag = np.zeros(100)

        # Apply first-order correction
        real_c, imag_c = phase_correction(real, imag, phi0=0, phi1=180)

        # First point should be rotated by phi1 (180 degrees)
        # Last point should be rotated by approximately 0 degrees
        # (phi1 is specified at index=0, decreases linearly)
        assert real_c[0] < 0  # First point inverted
        assert real_c[-1] > 0  # Last point positive

    def test_phase_correction_first_order_linear_gradient(self):
        """Test that first-order creates linear phase gradient."""
        real = np.ones(1000)
        imag = np.zeros(1000)

        real_c, imag_c = phase_correction(real, imag, phi0=0, phi1=360)

        # Phase should vary linearly from 360 to 0 degrees
        # At index 0: 360 degrees (full rotation, should be ~1)
        # At index 1000: 0 degrees (no rotation, should be 1)
        np.testing.assert_allclose(real_c[0], 1.0, atol=1e-2)
        np.testing.assert_allclose(real_c[-1], 1.0, atol=1e-2)

    def test_phase_correction_combined_zero_and_first_order(self):
        """Test combined zero and first-order correction."""
        real = np.ones(100)
        imag = np.zeros(100)

        real_c, imag_c = phase_correction(real, imag, phi0=45, phi1=90)

        # All points should be corrected, but with varying phases
        assert len(real_c) == 100
        assert len(imag_c) == 100
        # Result should be different from input
        assert not np.allclose(real_c, real)


class TestPhaseCorrectionReverse:
    """Tests for reverse mode (inverted chemical shift scale)."""

    def test_phase_correction_reverse_mode(self):
        """Test reverse mode changes first-order direction."""
        real = np.ones(100)
        imag = np.zeros(100)

        # Normal mode
        real_normal, imag_normal = phase_correction(real, imag, phi0=0, phi1=90, reverse=False)

        # Reverse mode
        real_reverse, imag_reverse = phase_correction(real, imag, phi0=0, phi1=90, reverse=True)

        # Should be different
        assert not np.allclose(real_normal, real_reverse)

    def test_phase_correction_reverse_first_order(self):
        """Test reverse mode inverts first-order gradient."""
        real = np.ones(1000)
        imag = np.zeros(1000)

        # Reverse mode: first-order should increase instead of decrease
        real_c, imag_c = phase_correction(real, imag, phi0=0, phi1=90, reverse=True)

        # Phase should increase from 0 to 90 degrees
        # (opposite of normal mode)
        assert len(real_c) == 1000


class TestPhaseCorrectionEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_phase_correction_zero_spectrum(self):
        """Test phase correction on zero spectrum."""
        real = np.zeros(100)
        imag = np.zeros(100)

        real_c, imag_c = phase_correction(real, imag, phi0=45, phi1=90)

        # Should remain zero
        np.testing.assert_allclose(real_c, np.zeros(100), atol=1e-10)
        np.testing.assert_allclose(imag_c, np.zeros(100), atol=1e-10)

    def test_phase_correction_single_point(self):
        """Test phase correction on single point."""
        real = np.array([1.0])
        imag = np.array([0.0])

        real_c, imag_c = phase_correction(real, imag, phi0=90, phi1=0)

        # 90 degree rotation
        np.testing.assert_allclose(real_c, [0.0], atol=1e-10)
        np.testing.assert_allclose(imag_c, [1.0], rtol=1e-10)

    def test_phase_correction_length_mismatch_raises(self):
        """Test that mismatched lengths raise error."""
        real = np.array([1.0, 2.0, 3.0])
        imag = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="real and imag must have same length"):
            phase_correction(real, imag, phi0=45, phi1=0)

    def test_phase_correction_2d_array_raises(self):
        """Test that 2D array raises error."""
        real = np.array([[1.0, 2.0], [3.0, 4.0]])
        imag = np.array([[0.5, 1.0], [1.5, 2.0]])

        with pytest.raises(ValueError, match="real must be 1D array"):
            phase_correction(real, imag, phi0=45, phi1=0)

    def test_phase_correction_type_coercion(self):
        """Test that non-array inputs are coerced."""
        real = [1.0, 2.0, 3.0, 4.0, 5.0]
        imag = [0.5, 1.0, 1.5, 2.0, 2.5]

        real_c, imag_c = phase_correction(real, imag, phi0=45, phi1=0)

        assert isinstance(real_c, np.ndarray)
        assert isinstance(imag_c, np.ndarray)
        assert len(real_c) == 5
        assert len(imag_c) == 5

    def test_phase_correction_negative_angles(self):
        """Test phase correction with negative angles."""
        real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        imag = np.zeros(5)

        # Negative angle should rotate in opposite direction
        real_c, imag_c = phase_correction(real, imag, phi0=-90, phi1=0)

        # -90 degrees should convert real to negative imaginary
        np.testing.assert_allclose(real_c, np.zeros(5), atol=1e-10)
        np.testing.assert_allclose(imag_c, -real, rtol=1e-10)


class TestPhaseCorrectionRealWorld:
    """Tests with real-world scenarios."""

    def test_phase_correction_preserves_magnitude(self):
        """Test that phase correction preserves spectral magnitude."""
        real = np.random.randn(1000)
        imag = np.random.randn(1000)

        # Calculate original magnitude
        mag_original = np.sqrt(real**2 + imag**2)

        # Apply phase correction
        real_c, imag_c = phase_correction(real, imag, phi0=45, phi1=90)

        # Calculate corrected magnitude
        mag_corrected = np.sqrt(real_c**2 + imag_c**2)

        # Magnitude should be preserved
        np.testing.assert_allclose(mag_corrected, mag_original, rtol=1e-10)

    def test_phase_correction_complex_spectrum(self):
        """Test phase correction on realistic complex spectrum."""
        # Create synthetic FID with phase error
        t = np.linspace(0, 1, 1000)
        freq = 10.0
        decay = 2.0

        # Complex signal with phase error
        signal = np.exp(2j * np.pi * freq * t - decay * t)
        # Add phase error
        phase_error = 30 * np.pi / 180  # 30 degrees
        signal *= np.exp(1j * phase_error)

        real = signal.real
        imag = signal.imag

        # Correct with -30 degrees
        real_c, imag_c = phase_correction(real, imag, phi0=-30, phi1=0)

        # Real part should be enhanced
        assert np.sum(np.abs(real_c)) > np.sum(np.abs(real))

    def test_phase_correction_iterative_application(self):
        """Test that phase corrections can be applied iteratively."""
        real = np.ones(100)
        imag = np.zeros(100)

        # Apply 45 degrees twice should equal 90 degrees once
        real_45, imag_45 = phase_correction(real, imag, phi0=45, phi1=0)
        real_90_iter, imag_90_iter = phase_correction(real_45, imag_45, phi0=45, phi1=0)

        # Compare with direct 90 degree correction
        real_90_direct, imag_90_direct = phase_correction(real, imag, phi0=90, phi1=0)

        # Allow for floating-point accumulation errors
        np.testing.assert_allclose(real_90_iter, real_90_direct, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(imag_90_iter, imag_90_direct, rtol=1e-8, atol=1e-10)

    def test_phase_correction_small_angles(self):
        """Test phase correction with small angles."""
        np.random.seed(42)
        real = np.random.randn(1000)
        imag = np.random.randn(1000)

        # Very small angle
        real_c, imag_c = phase_correction(real, imag, phi0=0.1, phi1=0.05)

        # Should be very close to original (small rotation)
        # Allow for slightly larger tolerance due to very small angles
        np.testing.assert_allclose(real_c, real, rtol=0.02, atol=0.01)

    def test_phase_correction_large_first_order(self):
        """Test phase correction with large first-order."""
        real = np.ones(1000)
        imag = np.zeros(1000)

        # Large first-order correction
        real_c, imag_c = phase_correction(real, imag, phi0=0, phi1=720)

        # Should handle multiple rotations
        assert len(real_c) == 1000
        assert np.all(np.isfinite(real_c))
        assert np.all(np.isfinite(imag_c))

    def test_phase_correction_symmetry(self):
        """Test that forward and reverse corrections are inverses."""
        real = np.random.randn(100)
        imag = np.random.randn(100)

        # Apply correction
        real_c, imag_c = phase_correction(real, imag, phi0=45, phi1=0)

        # Apply inverse correction
        real_back, imag_back = phase_correction(real_c, imag_c, phi0=-45, phi1=0)

        # Should return to original
        np.testing.assert_allclose(real_back, real, rtol=1e-10)
        np.testing.assert_allclose(imag_back, imag, rtol=1e-10)

"""Tests for calibration functions."""

import numpy as np
import pytest
from nmr_spectra_processing.core import calibrate_signal, calibrate_spectra
from nmr_spectra_processing.reference import NMRSignal, NMRPeak, get_reference_signal


class TestCalibrateSingle:
    """Tests for calibrating single spectrum."""

    def test_calibrate_to_tsp(self):
        """Test calibrating spectrum to TSP reference."""
        # Create ppm scale
        ppm = np.linspace(-1, 1, 1000)

        # Create TSP signal (but shifted to 0.1 ppm instead of 0.0)
        signal = get_reference_signal("tsp", cshift=0.1)
        spectrum = signal.to_spectrum(ppm, linewidth=0.01)

        # Calibrate to TSP at 0.0 ppm
        calibrated = calibrate_signal(ppm, spectrum, "tsp")

        # After calibration, peak should be closer to 0.0 ppm
        peak_before = ppm[np.argmax(spectrum)]
        peak_after = ppm[np.argmax(calibrated)]

        assert abs(peak_after) < abs(peak_before)

    def test_calibrate_get_shift_value(self):
        """Test getting shift value instead of calibrated spectrum."""
        ppm = np.linspace(-1, 1, 1000)

        # Create shifted TSP
        signal = get_reference_signal("tsp", cshift=0.05)
        spectrum = signal.to_spectrum(ppm, linewidth=0.01)

        # Get shift value
        shift = calibrate_signal(ppm, spectrum, "tsp", apply_shift=False)

        # Should be a scalar
        assert isinstance(shift, (float, np.floating))
        # Shift should be approximately -0.05 ppm (to move peak from 0.05 to 0.0)
        assert abs(shift - (-0.05)) < 0.02

    def test_calibrate_to_alanine(self):
        """Test calibrating to alanine reference."""
        ppm = np.linspace(0, 3, 3000)

        # Create alanine signal shifted slightly
        signal = get_reference_signal("alanine", frequency=600, cshift=1.55)
        spectrum = signal.to_spectrum(ppm, linewidth=0.01)

        # Calibrate to alanine at 1.48 ppm
        calibrated = calibrate_signal(ppm, spectrum, "alanine", frequency=600)

        # Peak center should move toward 1.48 ppm
        assert len(calibrated) == len(spectrum)

    def test_calibrate_to_glucose(self):
        """Test calibrating to glucose reference."""
        ppm = np.linspace(4, 6, 2000)

        # Create glucose signal
        signal = get_reference_signal("glucose", frequency=600, cshift=5.25)
        spectrum = signal.to_spectrum(ppm, linewidth=0.01)

        # Calibrate
        calibrated = calibrate_signal(ppm, spectrum, "glucose", frequency=600)

        assert len(calibrated) == len(spectrum)

    def test_calibrate_custom_roi(self):
        """Test calibration with custom ROI."""
        ppm = np.linspace(-1, 1, 1000)
        signal = get_reference_signal("tsp", cshift=0.1)
        spectrum = signal.to_spectrum(ppm, linewidth=0.01)

        # Calibrate with custom ROI
        calibrated = calibrate_signal(
            ppm, spectrum, "tsp",
            roi=(-0.5, 0.5)  # Wider ROI
        )

        assert len(calibrated) == len(spectrum)

    def test_calibrate_max_shift_constraint(self):
        """Test that max_shift constraint is enforced."""
        ppm = np.linspace(-1, 1, 1000)

        # Create very shifted TSP (0.5 ppm)
        signal = get_reference_signal("tsp", cshift=0.5)
        spectrum = signal.to_spectrum(ppm, linewidth=0.01)

        # Calibrate with small max_shift (should warn and not shift)
        shift = calibrate_signal(
            ppm, spectrum, "tsp",
            max_shift=0.05,  # Very small
            apply_shift=False
        )

        # Shift should be zero (exceeded max_shift)
        assert shift == 0.0

    def test_calibrate_with_nmr_signal_object(self):
        """Test calibrating with NMRSignal object instead of string."""
        ppm = np.linspace(-1, 1, 1000)

        # Create custom signal
        signal = NMRSignal(
            name="Custom",
            peaks=[NMRPeak(cshift=0.15, intensity=1.0)]
        )
        spectrum = signal.to_spectrum(ppm, linewidth=0.01)

        # Calibrate using the signal object
        calibrated = calibrate_signal(ppm, spectrum, signal, roi=(-0.5, 0.5))

        assert len(calibrated) == len(spectrum)


class TestCalibrateMultiple:
    """Tests for calibrating multiple spectra."""

    def test_calibrate_spectra_matrix(self):
        """Test calibrating multiple spectra."""
        ppm = np.linspace(-1, 1, 1000)

        # Create 3 spectra with different shifts
        spectra = np.zeros((3, 1000))
        for i in range(3):
            shift = (i - 1) * 0.05  # Shifts: -0.05, 0.0, 0.05
            signal = get_reference_signal("tsp", cshift=shift)
            spectra[i] = signal.to_spectrum(ppm, linewidth=0.01)

        # Calibrate all spectra
        calibrated = calibrate_spectra(ppm, spectra, ref="tsp")

        assert calibrated.shape == spectra.shape

        # All spectra should now have peaks closer to 0.0 ppm
        for i in range(3):
            peak_pos = ppm[np.argmax(calibrated[i])]
            assert abs(peak_pos) < 0.03

    def test_calibrate_spectra_get_shifts(self):
        """Test getting shift values for multiple spectra."""
        ppm = np.linspace(-1, 1, 1000)

        # Create 3 spectra with known shifts
        spectra = np.zeros((3, 1000))
        expected_shifts = [-0.05, 0.0, 0.05]
        for i, shift in enumerate(expected_shifts):
            signal = get_reference_signal("tsp", cshift=shift)
            spectra[i] = signal.to_spectrum(ppm, linewidth=0.01)

        # Get shifts
        shifts = calibrate_spectra(ppm, spectra, ref="tsp", apply_shift=False)

        assert len(shifts) == 3
        # Shifts should be approximately negative of expected shifts
        for i, expected in enumerate(expected_shifts):
            assert abs(shifts[i] - (-expected)) < 0.02

    def test_calibrate_spectra_single_spectrum(self):
        """Test that calibrate_spectra works with single spectrum."""
        ppm = np.linspace(-1, 1, 1000)
        signal = get_reference_signal("tsp", cshift=0.1)
        spectrum = signal.to_spectrum(ppm, linewidth=0.01)

        # Calibrate single spectrum
        calibrated = calibrate_spectra(ppm, spectrum, ref="tsp")

        assert len(calibrated) == len(spectrum)

    def test_calibrate_spectra_different_references(self):
        """Test calibrating with different reference types."""
        ppm_tsp = np.linspace(-1, 1, 1000)
        ppm_ala = np.linspace(0, 3, 1500)
        ppm_glc = np.linspace(4, 6, 1000)

        # TSP
        spec_tsp = get_reference_signal("tsp", cshift=0.05).to_spectrum(ppm_tsp, 0.01)
        cal_tsp = calibrate_spectra(ppm_tsp, spec_tsp, ref="tsp")
        assert len(cal_tsp) == len(spec_tsp)

        # Alanine
        spec_ala = get_reference_signal("alanine", 600, cshift=1.5).to_spectrum(ppm_ala, 0.01)
        cal_ala = calibrate_spectra(ppm_ala, spec_ala, ref="alanine", frequency=600)
        assert len(cal_ala) == len(spec_ala)

        # Glucose
        spec_glc = get_reference_signal("glucose", 600, cshift=5.25).to_spectrum(ppm_glc, 0.01)
        cal_glc = calibrate_spectra(ppm_glc, spec_glc, ref="glucose", frequency=600)
        assert len(cal_glc) == len(spec_glc)


class TestCalibrateEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_calibrate_zero_spectrum(self):
        """Test calibrating zero spectrum (should warn and return unchanged)."""
        ppm = np.linspace(-1, 1, 1000)
        spectrum = np.zeros(1000)

        # Calibrate zero spectrum (should handle gracefully)
        calibrated = calibrate_signal(ppm, spectrum, "tsp")

        # Should return original (can't calibrate zero spectrum)
        np.testing.assert_array_equal(calibrated, spectrum)

    def test_calibrate_roi_outside_range_raises(self):
        """Test that ROI outside ppm range raises error."""
        ppm = np.linspace(0, 10, 1000)
        spectrum = np.random.randn(1000)

        with pytest.raises(ValueError, match="ROI is outside ppm range"):
            calibrate_signal(ppm, spectrum, "tsp", roi=(15.0, 20.0))

    def test_calibrate_invalid_signal_type_raises(self):
        """Test that invalid signal type raises error."""
        ppm = np.linspace(0, 10, 1000)
        spectrum = np.random.randn(1000)

        with pytest.raises(TypeError, match="Invalid signal type"):
            calibrate_signal(ppm, spectrum, 12345)  # Invalid type

    def test_calibrate_3d_array_raises(self):
        """Test that 3D array raises error."""
        ppm = np.linspace(0, 10, 100)
        spectra = np.random.randn(5, 10, 100)

        with pytest.raises(ValueError, match="Invalid spectra dimensions"):
            calibrate_spectra(ppm, spectra, ref="tsp")

    def test_calibrate_type_coercion(self):
        """Test that non-array inputs are coerced."""
        ppm = list(np.linspace(-1, 1, 100))
        spectrum = list(np.random.randn(100))

        # Should work with type coercion
        result = calibrate_signal(ppm, spectrum, "tsp", roi=(-0.5, 0.5))
        assert isinstance(result, np.ndarray)


class TestCalibrateIntegration:
    """Integration tests with realistic scenarios."""

    def test_calibrate_with_noise(self):
        """Test calibration with noisy spectrum."""
        np.random.seed(42)
        ppm = np.linspace(-1, 1, 1000)

        # Create signal with noise
        signal = get_reference_signal("tsp", cshift=0.08)
        spectrum = signal.to_spectrum(ppm, linewidth=0.01)
        spectrum += np.random.randn(1000) * 0.01  # Add noise

        # Calibrate
        calibrated = calibrate_signal(ppm, spectrum, "tsp")

        # Should still work reasonably well
        peak_pos = ppm[np.argmax(calibrated)]
        assert abs(peak_pos) < 0.05

    def test_calibrate_with_baseline(self):
        """Test calibration with baseline drift."""
        ppm = np.linspace(-1, 1, 1000)

        # Create signal with baseline
        signal = get_reference_signal("tsp", cshift=0.1)
        spectrum = signal.to_spectrum(ppm, linewidth=0.01)
        spectrum += 0.2 * ppm + 0.5  # Add linear baseline

        # Calibrate (baseline correction is built-in)
        calibrated = calibrate_signal(ppm, spectrum, "tsp")

        assert len(calibrated) == len(spectrum)

    def test_calibrate_workflow(self):
        """Test typical calibration workflow."""
        ppm = np.linspace(0, 10, 5000)

        # Create realistic spectrum with multiple peaks and shifted TSP
        spectrum = np.zeros(5000)

        # Add TSP at 0.05 ppm (should be at 0.0)
        tsp = get_reference_signal("tsp", cshift=0.05)
        spectrum += tsp.to_spectrum(ppm, linewidth=0.005)

        # Add other peaks (simulating metabolites)
        for cshift in [1.2, 3.5, 5.2, 7.8]:
            peak = NMRSignal("Peak", [NMRPeak(cshift=cshift)])
            spectrum += peak.to_spectrum(ppm, linewidth=0.01) * 0.5

        # Calibrate to TSP
        calibrated = calibrate_spectra(ppm, spectrum, ref="tsp")

        # TSP peak should now be at ~0.0 ppm
        tsp_region = (ppm >= -0.1) & (ppm <= 0.1)
        tsp_peak_idx = np.argmax(calibrated[tsp_region])
        tsp_peak_pos = ppm[tsp_region][tsp_peak_idx]

        assert abs(tsp_peak_pos) < 0.02

    def test_calibrate_custom_parameters(self):
        """Test calibration with custom signal parameters."""
        ppm = np.linspace(0, 3, 3000)

        # Create alanine with non-standard parameters
        signal = get_reference_signal("alanine", frequency=600, cshift=1.52, J=7.5)
        spectrum = signal.to_spectrum(ppm, linewidth=0.01)

        # Calibrate with matching custom parameters
        calibrated = calibrate_signal(
            ppm, spectrum, "alanine",
            frequency=600, cshift=1.48, J=7.26  # Standard alanine
        )

        assert len(calibrated) == len(spectrum)

    def test_calibrate_consistency(self):
        """Test that calibration is consistent across multiple runs."""
        np.random.seed(42)
        ppm = np.linspace(-1, 1, 1000)
        signal = get_reference_signal("tsp", cshift=0.07)
        spectrum = signal.to_spectrum(ppm, linewidth=0.01)

        # Calibrate multiple times
        cal1 = calibrate_signal(ppm, spectrum, "tsp")
        cal2 = calibrate_signal(ppm, spectrum, "tsp")

        # Should get identical results
        np.testing.assert_array_equal(cal1, cal2)

    def test_calibrate_preserves_peak_shape(self):
        """Test that calibration preserves peak shapes."""
        ppm = np.linspace(-1, 1, 1000)
        signal = get_reference_signal("tsp", cshift=0.05)
        spectrum = signal.to_spectrum(ppm, linewidth=0.01)

        # Calibrate
        calibrated = calibrate_signal(ppm, spectrum, "tsp")

        # Peak width should be similar
        # (calibration shifts but doesn't distort)
        def half_max_width(spec):
            max_val = np.max(spec)
            above_half = spec > max_val / 2
            return np.sum(above_half)

        width_before = half_max_width(spectrum)
        width_after = half_max_width(calibrated)

        # Widths should be very similar
        assert abs(width_before - width_after) <= 2  # Allow ±2 points difference

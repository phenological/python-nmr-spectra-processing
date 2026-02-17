"""Tests for NMR signal reference classes."""

import numpy as np
import pytest
from nmr_spectra_processing.reference import (
    NMRPeak,
    NMRSignal,
    get_reference_signal,
    create_custom_signal,
)


class TestNMRPeak:
    """Tests for NMRPeak dataclass."""

    def test_nmr_peak_creation(self):
        """Test creating NMRPeak with defaults."""
        peak = NMRPeak(cshift=1.5)

        assert peak.cshift == 1.5
        assert peak.J == 0.0
        assert peak.multiplicity == 1
        assert peak.intensity == 1.0

    def test_nmr_peak_custom_values(self):
        """Test creating NMRPeak with custom values."""
        peak = NMRPeak(cshift=2.3, J=7.5, multiplicity=2, intensity=0.8)

        assert peak.cshift == 2.3
        assert peak.J == 7.5
        assert peak.multiplicity == 2
        assert peak.intensity == 0.8


class TestNMRSignal:
    """Tests for NMRSignal dataclass."""

    def test_nmr_signal_creation(self):
        """Test creating NMRSignal."""
        peaks = [
            NMRPeak(cshift=1.0, intensity=1.0),
            NMRPeak(cshift=2.0, intensity=0.5)
        ]
        signal = NMRSignal(name="Test", peaks=peaks)

        assert signal.name == "Test"
        assert len(signal.peaks) == 2
        assert signal.peaks[0].cshift == 1.0
        assert signal.peaks[1].cshift == 2.0

    def test_to_spectrum_lorentzian(self):
        """Test generating spectrum with Lorentzian lineshape."""
        peak = NMRPeak(cshift=5.0, intensity=1.0)
        signal = NMRSignal(name="SinglePeak", peaks=[peak])

        ppm = np.linspace(0, 10, 1000)
        spectrum = signal.to_spectrum(ppm, linewidth=0.5, lineshape="lorentzian")

        # Check that spectrum has correct length
        assert len(spectrum) == len(ppm)

        # Peak should be at maximum at cshift=5.0
        peak_idx = np.argmax(spectrum)
        assert abs(ppm[peak_idx] - 5.0) < 0.02

        # Maximum intensity should be 1.0 (at peak center)
        assert spectrum[peak_idx] > 0.9

    def test_to_spectrum_gaussian(self):
        """Test generating spectrum with Gaussian lineshape."""
        peak = NMRPeak(cshift=3.0, intensity=1.0)
        signal = NMRSignal(name="SinglePeak", peaks=[peak])

        ppm = np.linspace(0, 10, 1000)
        spectrum = signal.to_spectrum(ppm, linewidth=0.5, lineshape="gaussian")

        # Peak should be at cshift
        peak_idx = np.argmax(spectrum)
        assert abs(ppm[peak_idx] - 3.0) < 0.02

    def test_to_spectrum_pseudovoigt(self):
        """Test generating spectrum with pseudo-Voigt lineshape."""
        peak = NMRPeak(cshift=7.0, intensity=1.0)
        signal = NMRSignal(name="SinglePeak", peaks=[peak])

        ppm = np.linspace(0, 10, 1000)
        spectrum = signal.to_spectrum(ppm, linewidth=0.5, lineshape="pseudovoigt")

        # Peak should be at cshift
        peak_idx = np.argmax(spectrum)
        assert abs(ppm[peak_idx] - 7.0) < 0.02

    def test_to_spectrum_multiple_peaks(self):
        """Test generating spectrum with multiple peaks."""
        peaks = [
            NMRPeak(cshift=2.0, intensity=1.0),
            NMRPeak(cshift=5.0, intensity=0.5),
            NMRPeak(cshift=8.0, intensity=1.5)
        ]
        signal = NMRSignal(name="MultiplePeaks", peaks=peaks)

        ppm = np.linspace(0, 10, 1000)
        spectrum = signal.to_spectrum(ppm, linewidth=0.3, lineshape="lorentzian")

        # Should have multiple peaks
        # Find local maxima (peaks)
        local_max = (spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])
        num_peaks = np.sum(local_max)

        # Should find approximately 3 peaks (may vary slightly due to resolution)
        assert 2 <= num_peaks <= 4

    def test_to_spectrum_custom_linewidth(self):
        """Test effect of linewidth parameter."""
        peak = NMRPeak(cshift=5.0, intensity=1.0)
        signal = NMRSignal(name="SinglePeak", peaks=[peak])

        ppm = np.linspace(0, 10, 1000)

        # Narrow linewidth
        spectrum_narrow = signal.to_spectrum(ppm, linewidth=0.1, lineshape="lorentzian")

        # Wide linewidth
        spectrum_wide = signal.to_spectrum(ppm, linewidth=1.0, lineshape="lorentzian")

        # Narrow peak should have more concentrated intensity (higher at edges relative to center)
        # Compare half-maximum widths: narrow peak drops faster
        half_max_narrow = np.sum(spectrum_narrow > 0.5)
        half_max_wide = np.sum(spectrum_wide > 0.5)

        # Narrow peak should have fewer points above half-maximum
        assert half_max_narrow < half_max_wide

    def test_to_spectrum_unknown_lineshape_raises(self):
        """Test that unknown lineshape raises error."""
        peak = NMRPeak(cshift=5.0)
        signal = NMRSignal(name="Test", peaks=[peak])
        ppm = np.linspace(0, 10, 100)

        with pytest.raises(ValueError, match="Unknown lineshape"):
            signal.to_spectrum(ppm, lineshape="invalid")


class TestGetReferenceSignal:
    """Tests for get_reference_signal function."""

    def test_get_tsp_reference(self):
        """Test getting TSP reference."""
        tsp = get_reference_signal("tsp")

        assert tsp.name == "TSP"
        assert len(tsp.peaks) == 1
        assert tsp.peaks[0].cshift == 0.0
        assert tsp.peaks[0].multiplicity == 1

    def test_get_tsp_custom_shift(self):
        """Test TSP with custom chemical shift."""
        tsp = get_reference_signal("tsp", cshift=0.1)

        assert tsp.peaks[0].cshift == 0.1

    def test_get_alanine_reference(self):
        """Test getting alanine reference."""
        ala = get_reference_signal("alanine", frequency=600)

        assert ala.name == "Alanine"
        assert len(ala.peaks) == 2  # Doublet
        # Default chemical shift: 1.48 ppm
        avg_shift = (ala.peaks[0].cshift + ala.peaks[1].cshift) / 2
        assert abs(avg_shift - 1.48) < 0.001

    def test_get_alanine_custom_parameters(self):
        """Test alanine with custom parameters."""
        ala = get_reference_signal("alanine", frequency=600, cshift=1.5, J=8.0)

        # Check average chemical shift
        avg_shift = (ala.peaks[0].cshift + ala.peaks[1].cshift) / 2
        assert abs(avg_shift - 1.5) < 0.001

        # Check splitting (J in ppm)
        splitting = abs(ala.peaks[1].cshift - ala.peaks[0].cshift)
        expected_splitting = 8.0 / 600  # J / frequency
        assert abs(splitting - expected_splitting) < 0.0001

    def test_get_glucose_reference(self):
        """Test getting glucose reference."""
        glc = get_reference_signal("glucose", frequency=600)

        assert glc.name == "Glucose"
        assert len(glc.peaks) == 2  # Doublet
        # Default chemical shift: 5.223 ppm
        avg_shift = (glc.peaks[0].cshift + glc.peaks[1].cshift) / 2
        assert abs(avg_shift - 5.223) < 0.001

    def test_get_serum_reference(self):
        """Test getting serum reference (alias for glucose)."""
        serum = get_reference_signal("serum", frequency=600)

        assert serum.name == "Glucose"
        assert len(serum.peaks) == 2

    def test_get_reference_case_insensitive(self):
        """Test that reference names are case-insensitive."""
        tsp1 = get_reference_signal("TSP")
        tsp2 = get_reference_signal("tsp")
        tsp3 = get_reference_signal("TsP")

        assert tsp1.name == tsp2.name == tsp3.name

    def test_get_reference_unknown_raises(self):
        """Test that unknown reference raises error."""
        with pytest.raises(ValueError, match="Unknown reference"):
            get_reference_signal("invalid")

    def test_get_reference_different_frequencies(self):
        """Test that frequency affects coupling splitting."""
        # Alanine at different field strengths
        ala_600 = get_reference_signal("alanine", frequency=600, J=7.26)
        ala_800 = get_reference_signal("alanine", frequency=800, J=7.26)

        # Splitting in ppm should be different
        split_600 = abs(ala_600.peaks[1].cshift - ala_600.peaks[0].cshift)
        split_800 = abs(ala_800.peaks[1].cshift - ala_800.peaks[0].cshift)

        # Higher frequency = smaller splitting in ppm
        assert split_600 > split_800


class TestCreateCustomSignal:
    """Tests for create_custom_signal function."""

    def test_create_custom_signal_simple(self):
        """Test creating custom signal with single peak."""
        signal = create_custom_signal("Custom", peak_positions=[3.5])

        assert signal.name == "Custom"
        assert len(signal.peaks) == 1
        assert signal.peaks[0].cshift == 3.5
        assert signal.peaks[0].intensity == 1.0

    def test_create_custom_signal_multiple_peaks(self):
        """Test creating custom signal with multiple peaks."""
        signal = create_custom_signal(
            "Triplet",
            peak_positions=[3.4, 3.5, 3.6],
            intensities=[1.0, 2.0, 1.0]
        )

        assert signal.name == "Triplet"
        assert len(signal.peaks) == 3
        assert signal.peaks[0].cshift == 3.4
        assert signal.peaks[1].cshift == 3.5
        assert signal.peaks[2].cshift == 3.6
        assert signal.peaks[0].intensity == 1.0
        assert signal.peaks[1].intensity == 2.0
        assert signal.peaks[2].intensity == 1.0

    def test_create_custom_signal_default_intensities(self):
        """Test creating signal with default intensities."""
        signal = create_custom_signal("Doublet", peak_positions=[2.0, 2.1])

        assert len(signal.peaks) == 2
        assert signal.peaks[0].intensity == 1.0
        assert signal.peaks[1].intensity == 1.0

    def test_create_custom_signal_length_mismatch_raises(self):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError, match="same length"):
            create_custom_signal(
                "Invalid",
                peak_positions=[1.0, 2.0, 3.0],
                intensities=[1.0, 2.0]
            )


class TestSignalIntegration:
    """Integration tests combining signal creation and spectrum generation."""

    def test_tsp_spectrum_generation(self):
        """Test generating TSP spectrum."""
        tsp = get_reference_signal("tsp")
        ppm = np.linspace(-1, 1, 1000)
        spectrum = tsp.to_spectrum(ppm, linewidth=0.01, lineshape="lorentzian")

        # Peak should be at 0.0 ppm
        peak_idx = np.argmax(spectrum)
        assert abs(ppm[peak_idx]) < 0.01

        # Spectrum should be mostly near zero except at peak
        assert np.sum(spectrum > 0.1) < 100  # Only narrow region has intensity

    def test_alanine_doublet_separation(self):
        """Test that alanine doublet has correct separation."""
        ala = get_reference_signal("alanine", frequency=600, J=7.26)
        ppm = np.linspace(0, 3, 3000)  # High resolution
        spectrum = ala.to_spectrum(ppm, linewidth=0.01, lineshape="lorentzian")

        # Find peaks
        local_max = (spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])
        peak_indices = np.where(local_max)[0] + 1
        peak_positions = ppm[peak_indices]

        # Should find 2 peaks
        assert len(peak_positions) == 2

        # Separation should be J/frequency
        separation = abs(peak_positions[1] - peak_positions[0])
        expected_separation = 7.26 / 600
        assert abs(separation - expected_separation) < 0.01

    def test_glucose_doublet_generation(self):
        """Test glucose doublet spectrum."""
        glc = get_reference_signal("glucose", frequency=600)
        ppm = np.linspace(4, 6, 2000)
        spectrum = glc.to_spectrum(ppm, linewidth=0.01, lineshape="lorentzian")

        # Should have 2 peaks
        local_max = (spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])
        num_peaks = np.sum(local_max)

        assert num_peaks == 2

    def test_custom_multiplet_spectrum(self):
        """Test custom multiplet spectrum."""
        # Create quartet (1:3:3:1 pattern)
        signal = create_custom_signal(
            "Quartet",
            peak_positions=[3.0, 3.01, 3.02, 3.03],
            intensities=[1.0, 3.0, 3.0, 1.0]
        )

        ppm = np.linspace(2.5, 3.5, 5000)
        spectrum = signal.to_spectrum(ppm, linewidth=0.005, lineshape="lorentzian")

        # Central peaks should be higher than outer peaks
        center_idx = np.argmax(spectrum)
        center_value = spectrum[center_idx]

        # Edges should have lower intensity
        edge_value = np.mean([spectrum[0], spectrum[-1]])
        assert center_value > edge_value * 2

    def test_lineshape_comparison(self):
        """Test different lineshapes produce different spectra."""
        peak = NMRPeak(cshift=5.0)
        signal = NMRSignal("Test", peaks=[peak])
        ppm = np.linspace(0, 10, 1000)

        spec_lorentz = signal.to_spectrum(ppm, linewidth=0.5, lineshape="lorentzian")
        spec_gauss = signal.to_spectrum(ppm, linewidth=0.5, lineshape="gaussian")
        spec_voigt = signal.to_spectrum(ppm, linewidth=0.5, lineshape="pseudovoigt")

        # All should have peaks at same position
        assert abs(ppm[np.argmax(spec_lorentz)] - 5.0) < 0.02
        assert abs(ppm[np.argmax(spec_gauss)] - 5.0) < 0.02
        assert abs(ppm[np.argmax(spec_voigt)] - 5.0) < 0.02

        # But spectra should be different
        assert not np.allclose(spec_lorentz, spec_gauss)
        assert not np.allclose(spec_lorentz, spec_voigt)
        assert not np.allclose(spec_gauss, spec_voigt)

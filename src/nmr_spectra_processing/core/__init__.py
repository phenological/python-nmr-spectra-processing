"""Core processing functions for NMR spectra."""

from nmr_spectra_processing.core.alignment import align_spectra
from nmr_spectra_processing.core.baseline import baseline_correction
from nmr_spectra_processing.core.calibration import calibrate_signal, calibrate_spectra
from nmr_spectra_processing.core.deconvolution import (
    broaden,
    measure_fwhm,
    nmr_to_voigt,
    ref_deconv,
    spectrum_from_peaks,
    voigt,
)
from nmr_spectra_processing.core.noise import estimate_noise
from nmr_spectra_processing.core.normalization import normalize, pqn
from nmr_spectra_processing.core.padding import pad_series
from nmr_spectra_processing.core.phase import phase_correction
from nmr_spectra_processing.core.shifting import shift_series, shift_spectra

__all__ = [
    "align_spectra",
    "baseline_correction",
    "broaden",
    "calibrate_signal",
    "calibrate_spectra",
    "estimate_noise",
    "measure_fwhm",
    "nmr_to_voigt",
    "normalize",
    "pad_series",
    "phase_correction",
    "pqn",
    "ref_deconv",
    "shift_series",
    "shift_spectra",
    "spectrum_from_peaks",
    "voigt",
]

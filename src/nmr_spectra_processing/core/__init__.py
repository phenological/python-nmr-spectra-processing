"""Core processing functions for NMR spectra."""

from nmr_spectra_processing.core.alignment import align_spectra
from nmr_spectra_processing.core.baseline import baseline_correction
from nmr_spectra_processing.core.calibration import calibrate_signal, calibrate_spectra
from nmr_spectra_processing.core.noise import estimate_noise
from nmr_spectra_processing.core.normalization import normalize, pqn
from nmr_spectra_processing.core.padding import pad_series
from nmr_spectra_processing.core.phase import phase_correction
from nmr_spectra_processing.core.shifting import shift_series, shift_spectra

__all__ = [
    "align_spectra",
    "baseline_correction",
    "calibrate_signal",
    "calibrate_spectra",
    "estimate_noise",
    "normalize",
    "pad_series",
    "phase_correction",
    "pqn",
    "shift_series",
    "shift_spectra",
]

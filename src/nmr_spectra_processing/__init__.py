"""
nmr-spectra-processing: Process Fourier-transformed NMR spectra

Python migration of the R package nmr.spectra.processing (v0.1.6).
Provides tools for alignment, calibration, normalization, baseline correction,
and phase correction of NMR spectra.
"""

from nmr_spectra_processing.version import __version__

# Core processing functions
from nmr_spectra_processing.core import (
    align_spectra,
    baseline_correction,
    broaden,
    calibrate_signal,
    calibrate_spectra,
    estimate_noise,
    measure_fwhm,
    nmr_to_voigt,
    normalize,
    pad_series,
    phase_correction,
    pqn,
    ref_deconv,
    shift_series,
    shift_spectra,
    spectrum_from_peaks,
    voigt,
)

# Utility functions
from nmr_spectra_processing.utils import (
    crop_region,
    get_indices,
    get_top_spectra,
)

# Reference signals
from nmr_spectra_processing.reference import (
    NMRPeak,
    NMRSignal,
    get_reference_signal,
    create_custom_signal,
)

__all__ = [
    # Version
    "__version__",
    # Core processing
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
    # Utilities
    "crop_region",
    "get_indices",
    "get_top_spectra",
    # Reference signals
    "NMRPeak",
    "NMRSignal",
    "get_reference_signal",
    "create_custom_signal",
]

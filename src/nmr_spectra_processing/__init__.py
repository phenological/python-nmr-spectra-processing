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
    area_from_height,
    baseline_correction,
    broaden,
    calibrate_signal,
    calibrate_spectra,
    estimate_noise,
    height_from_area,
    measure_fwhm,
    multiplet,
    nmr_to_voigt,
    normalize,
    pad_series,
    phase_correction,
    pqn,
    pseudo_voigt_fwhm,
    ref_deconv,
    shift_series,
    shift_spectra,
    spectrum_from_peaks,
    voigt,
    voigt_area,
    voigt_height,
    voigt_profile,
    voigt_with_satellites,
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
    "area_from_height",
    "baseline_correction",
    "broaden",
    "calibrate_signal",
    "calibrate_spectra",
    "estimate_noise",
    "height_from_area",
    "measure_fwhm",
    "multiplet",
    "nmr_to_voigt",
    "normalize",
    "pad_series",
    "phase_correction",
    "pqn",
    "pseudo_voigt_fwhm",
    "ref_deconv",
    "shift_series",
    "shift_spectra",
    "spectrum_from_peaks",
    "voigt",
    "voigt_area",
    "voigt_height",
    "voigt_profile",
    "voigt_with_satellites",
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

"""Core processing functions for NMR spectra."""

from nmr_spectra_processing.core.alignment import align_spectra
from nmr_spectra_processing.core.baseline import baseline_correction
from nmr_spectra_processing.core.calibration import calibrate_signal, calibrate_spectra
from nmr_spectra_processing.core.deconvolution import (
    broaden,
    nmr_to_voigt,
    ref_deconv,
    spectrum_from_peaks,
)
from nmr_spectra_processing.core.lineshapes import (
    area_from_height,
    height_from_area,
    measure_fwhm,
    multiplet,
    pseudo_voigt_fwhm,
    voigt,
    voigt_area,
    voigt_height,
    voigt_profile,
    voigt_with_satellites,
)
from nmr_spectra_processing.core.noise import estimate_noise
from nmr_spectra_processing.core.normalization import normalize, pqn
from nmr_spectra_processing.core.padding import pad_series
from nmr_spectra_processing.core.phase import phase_correction
from nmr_spectra_processing.core.shifting import shift_series, shift_spectra

__all__ = [
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
]

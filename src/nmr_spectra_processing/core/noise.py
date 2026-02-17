"""
Noise estimation functions for NMR spectra.

Provides methods to estimate noise levels from blank spectral regions.
"""

from typing import Optional, Tuple, Union
import numpy as np
from nmr_spectra_processing.utils.logger import warning, error


def estimate_noise(
    ppm: np.ndarray,
    spectra: np.ndarray,
    level: float = 0.99,
    roi: Tuple[float, float] = (9.8, 10.0)
) -> Union[float, np.ndarray]:
    """
    Estimate noise level from a blank spectral region.

    Migrated from R function noiseLevel().

    Args:
        ppm: Chemical shift scale (1D array)
        spectra: Spectrum intensities (1D array) or matrix with spectra in rows
        level: Quantile level for noise estimation (default 0.99).
               Specified as probability in [0, 1].
        roi: Region of reference (start, end) in ppm for noise estimation.
             Default: (9.8, 10.0) - typical blank region.

    Returns:
        Noise level estimate (scalar for vector, array for matrix)

    Notes:
        - Takes a 'blank' region of the spectrum as reference
        - Estimates noise as the specified quantile of intensities in that region
        - Default region (9.8-10 ppm) is typically blank for 1H NMR
        - For matrix input, returns noise estimate for each spectrum

    Examples:
        >>> # Estimate noise from single spectrum
        >>> noise = estimate_noise(ppm, spectrum)
        >>> # Estimate noise from multiple spectra
        >>> noise_levels = estimate_noise(ppm, spectra)
        >>> # Use custom ROI and quantile
        >>> noise = estimate_noise(ppm, spectrum, level=0.95, roi=(9.5, 10.0))
    """
    # Type checking for ppm
    if not isinstance(ppm, np.ndarray):
        warning(
            "Non-numeric argument ppm being cast as numpy array. "
            "Unpredictable results will follow if casting fails.",
            prefix="nmr_spectra_processing::estimate_noise"
        )
        ppm = np.asarray(ppm)

    if not np.issubdtype(ppm.dtype, np.number):
        warning(
            "Non-numeric argument ppm being cast as numeric. "
            "Unpredictable results will follow if casting fails.",
            prefix="nmr_spectra_processing::estimate_noise"
        )
        ppm = ppm.astype(float)

    # Validate ROI
    if not isinstance(roi, (tuple, list, np.ndarray)) or len(roi) != 2:
        error(
            "Invalid roi - must be tuple/list of 2 elements (start, end)",
            prefix="nmr_spectra_processing::estimate_noise"
        )
        raise ValueError("roi must have 2 elements")

    roi_start, roi_end = roi
    if roi_start >= roi_end:
        error(
            "Invalid roi - start must be less than end",
            prefix="nmr_spectra_processing::estimate_noise"
        )
        raise ValueError("roi start must be less than end")

    if roi_start >= np.max(ppm) or roi_end <= np.min(ppm):
        error(
            "roi outside of ppm range",
            prefix="nmr_spectra_processing::estimate_noise"
        )
        raise ValueError("roi is outside ppm range")

    # Create filter for ROI
    roi_mask = (ppm >= roi_start) & (ppm <= roi_end)

    if not np.any(roi_mask):
        error(
            "No points found in specified roi",
            prefix="nmr_spectra_processing::estimate_noise"
        )
        raise ValueError("roi contains no data points")

    # Convert spectra to array first if needed
    if not isinstance(spectra, np.ndarray):
        warning(
            "Argument spectra being cast as numpy array. "
            "Unpredictable results may follow if casting fails.",
            prefix="nmr_spectra_processing::estimate_noise"
        )
        spectra = np.asarray(spectra)

    # Handle vector input (single spectrum)
    if spectra.ndim == 1:
        if not np.issubdtype(spectra.dtype, np.number):
            warning(
                "Non-numeric argument spectra being cast as numeric. "
                "Unpredictable results will follow if casting fails.",
                prefix="nmr_spectra_processing::estimate_noise"
            )
            spectra = spectra.astype(float)

        # Compute quantile in ROI
        roi_intensities = spectra[roi_mask]
        return np.quantile(roi_intensities, level)

    # Handle matrix input
    if spectra.ndim != 2:
        # Try to convert to matrix
        spectra = np.atleast_2d(spectra)

    if not np.issubdtype(spectra.dtype, np.number):
        error(
            "Expected spectra to be a numeric array or matrix",
            prefix="nmr_spectra_processing::estimate_noise"
        )
        raise TypeError("Spectra must be numeric")

    # Validate that ppm length matches spectra columns
    if len(ppm) != spectra.shape[1]:
        error(
            f"ppm length {len(ppm)} does not match spectra width {spectra.shape[1]}",
            prefix="nmr_spectra_processing::estimate_noise"
        )
        raise ValueError("ppm and spectra dimensions mismatch")

    # Compute quantile for each spectrum in ROI
    roi_spectra = spectra[:, roi_mask]
    noise_levels = np.quantile(roi_spectra, level, axis=1)

    return noise_levels

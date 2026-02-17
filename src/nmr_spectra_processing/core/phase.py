"""
Phase correction functions for NMR spectra.

Provides zero-order and first-order phase correction for complex NMR data.
"""

from typing import Tuple
import numpy as np
from nmr_spectra_processing.utils.logger import warning


def phase_correction(
    real: np.ndarray,
    imag: np.ndarray,
    phi0: float = 0.0,
    phi1: float = 0.0,
    reverse: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply phase correction to complex NMR spectrum.

    Migrated from R function phaseCorrection().

    Args:
        real: Real part of spectrum (1D array)
        imag: Imaginary part of spectrum (1D array)
        phi0: Zero-order phase correction in degrees (default 0.0)
        phi1: First-order phase correction in degrees (default 0.0)
        reverse: If True, reverse direction of first-order correction.
                Used when chemical shift scale is inverted (decreases left to right).
                Default: False.

    Returns:
        Tuple of (real_corrected, imag_corrected)

    Notes:
        - Zero-order (phi0): Uniform phase rotation applied to all points
        - First-order (phi1): Linear phase correction, increases across spectrum
        - phi1 is specified at index=0
        - Uses incremental rotation for computational efficiency
        - Phase is specified in degrees and converted to radians internally

    Examples:
        >>> # Apply zero-order correction only
        >>> real_c, imag_c = phase_correction(real, imag, phi0=45)
        >>> # Apply both zero and first-order
        >>> real_c, imag_c = phase_correction(real, imag, phi0=30, phi1=180)
        >>> # For inverted scale
        >>> real_c, imag_c = phase_correction(real, imag, phi0=30, phi1=180, reverse=True)
    """
    # Type checking
    if not isinstance(real, np.ndarray):
        warning(
            "Argument real being cast to numpy array. "
            "Unpredictable results may follow if casting fails.",
            prefix="nmr_spectra_processing::phase_correction"
        )
        real = np.asarray(real)

    if not isinstance(imag, np.ndarray):
        warning(
            "Argument imag being cast to numpy array. "
            "Unpredictable results may follow if casting fails.",
            prefix="nmr_spectra_processing::phase_correction"
        )
        imag = np.asarray(imag)

    if not np.issubdtype(real.dtype, np.number):
        warning(
            "Argument real being cast as numeric. "
            "Unpredictable results may follow if casting fails.",
            prefix="nmr_spectra_processing::phase_correction"
        )
        real = real.astype(float)

    if not np.issubdtype(imag.dtype, np.number):
        warning(
            "Argument imag being cast as numeric. "
            "Unpredictable results may follow if casting fails.",
            prefix="nmr_spectra_processing::phase_correction"
        )
        imag = imag.astype(float)

    # Validate dimensions
    if real.ndim != 1:
        raise ValueError("real must be 1D array")
    if imag.ndim != 1:
        raise ValueError("imag must be 1D array")
    if len(real) != len(imag):
        raise ValueError("real and imag must have same length")

    # Convert degrees to radians
    phi0_rad = phi0 * np.pi / 180
    phi1_rad = phi1 * np.pi / 180

    # Calculate incremental rotation parameters
    lng = len(real)
    delta = phi1_rad / lng
    first_angle = phi0_rad

    # Handle reverse mode (for inverted chemical shift scale)
    if not reverse:
        delta = -delta
        first_angle = first_angle + phi1_rad

    # Incremental rotation parameters (for efficiency)
    # These avoid computing sin/cos at every point
    alpha = 2 * np.sin(delta / 2) ** 2
    beta = np.sin(delta)
    cos_theta = np.cos(first_angle)
    sin_theta = np.sin(first_angle)

    # Initialize output arrays
    real_corrected = np.zeros_like(real)
    imag_corrected = np.zeros_like(imag)

    # Apply phase correction with incremental rotation
    for i in range(lng):
        # Apply rotation to current point
        real_corrected[i] = real[i] * cos_theta - imag[i] * sin_theta
        imag_corrected[i] = imag[i] * cos_theta + real[i] * sin_theta

        # Incrementally update rotation angle for next point
        # This is more efficient than computing sin/cos for each angle
        tmp_cos = cos_theta - (alpha * cos_theta + beta * sin_theta)
        tmp_sin = sin_theta - (alpha * sin_theta - beta * cos_theta)
        cos_theta = tmp_cos
        sin_theta = tmp_sin

    return real_corrected, imag_corrected

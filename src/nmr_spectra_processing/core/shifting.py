"""
Shifting functions for NMR spectra.

Provides discrete and continuous shifting methods for spectral data.
"""

from typing import Optional, Union
import numpy as np
from scipy.interpolate import interp1d
from nmr_spectra_processing.core.padding import pad_series
from nmr_spectra_processing.utils.logger import warning


def shift_series(
    x: np.ndarray,
    shift: int,
    padding: str = "sampling",
    from_region: Optional[Union[np.ndarray, slice]] = None
) -> np.ndarray:
    """
    Shift a series by a given number of points (discrete shift).

    Migrated from R function shiftSeries().

    Args:
        x: Series to be shifted (1D array)
        shift: Number of points to shift. Negative shifts left, positive shifts right.
        padding: Padding method for empty extremes ("zeroes", "circular", "sampling")
        from_region: For "sampling" method - region to sample from (see pad_series)

    Returns:
        Shifted series (same length as input)

    Notes:
        - Positive shift moves series to the right
        - Negative shift moves series to the left
        - Uses pad_series to fill empty regions
        - Default padding: "sampling" from last 1/15th points

    Examples:
        >>> x = np.arange(10)
        >>> # Shift right by 3 (elements 0-2 become padding)
        >>> shifted = shift_series(x, shift=3, padding="zeroes")
        >>> # Shift left by 2 (elements 8-9 become padding)
        >>> shifted = shift_series(x, shift=-2, padding="zeroes")
    """
    # Type checking
    if not isinstance(x, np.ndarray):
        warning(
            "Argument x being cast to numpy array. "
            "Unpredictable results will follow if casting fails.",
            prefix="nmr_spectra_processing::shift_series"
        )
        x = np.asarray(x)

    if not np.issubdtype(x.dtype, np.number):
        warning(
            "Argument x being cast as numeric. "
            "Unpredictable results may follow if casting to numeric array fails.",
            prefix="nmr_spectra_processing::shift_series"
        )
        x = x.astype(float)

    # Determine direction and magnitude
    direction = np.sign(shift)
    abs_shift = int(abs(shift))

    if abs_shift == 0:
        return x.copy()

    # Pad on the opposite side of the shift direction
    # If shifting right (direction=1), pad on left (side=-1)
    # If shifting left (direction=-1), pad on right (side=1)
    side = -int(direction)

    # Set up default from_region if not provided
    if from_region is None and padding == "sampling":
        start_idx = int(len(x) * 14 / 15)
        from_region = slice(start_idx, len(x))

    padded = pad_series(x, n=abs_shift, side=side, method=padding, from_region=from_region)

    # Extract the shifted portion
    if direction == 1:
        # Shifted right: take first len(x) elements
        return padded[:len(x)]
    else:
        # Shifted left: take last len(x) elements (skip first abs_shift)
        return padded[abs_shift:]


def shift_spectra(
    ppm: np.ndarray,
    spectra: np.ndarray,
    shift: Union[float, np.ndarray],
    hertz: bool = False,
    SF: Union[float, np.ndarray] = 600.0,
    method: str = "cubic"
) -> np.ndarray:
    """
    Shift spectra by a given frequency using interpolation (continuous shift).

    Migrated from R function shiftSpectra().

    Args:
        ppm: Chemical shift scale (1D array)
        spectra: Intensity data (1D vector or 2D matrix with spectra in rows)
        shift: Frequency to shift each spectrum by (scalar or array)
        hertz: If True, shift is in Hz; if False (default), shift is in ppm
        SF: Spectrometer field strength in MHz (for Hz to ppm conversion)
        method: Interpolation method ("linear", "cubic", "quadratic")
                Default "cubic" corresponds to R's "spline"

    Returns:
        Shifted spectra (same shape as input)

    Notes:
        - Works by adding shift to ppm scale, then interpolating at original ppm
        - shift and SF are recycled to match number of spectra
        - Positive shift moves peaks to higher ppm (right)
        - Negative shift moves peaks to lower ppm (left)

    Examples:
        >>> ppm = np.linspace(0, 10, 1000)
        >>> spectrum = np.sin(ppm)
        >>> # Shift by 0.1 ppm
        >>> shifted = shift_spectra(ppm, spectrum, shift=0.1)
        >>> # Shift multiple spectra by different amounts
        >>> spectra = np.random.randn(5, 1000)
        >>> shifts = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        >>> shifted = shift_spectra(ppm, spectra, shift=shifts)
    """
    # Helper function to shift a single spectrum
    def _shift_single_spectrum(ppm_scale, y, shift_amount, interp_method):
        """Shift single spectrum using interpolation."""
        # Shift the ppm scale
        ppm_shifted = ppm_scale + shift_amount

        # For single-point or very small arrays, use linear interpolation
        # (cubic/quadratic require at least 4/3 points respectively)
        if len(ppm_scale) < 4 and interp_method in ["cubic", "quadratic"]:
            interp_method = "linear"

        # Interpolate at original ppm points
        interpolator = interp1d(
            ppm_shifted,
            y,
            kind=interp_method,
            bounds_error=False,
            fill_value=0.0  # Fill out-of-bounds with zero
        )
        return interpolator(ppm_scale)

    # Handle vector input (single spectrum)
    if spectra.ndim == 1:
        # Convert to ppm if needed
        if hertz:
            if isinstance(SF, np.ndarray):
                SF = SF[0]
            if isinstance(shift, np.ndarray):
                shift = shift[0]
            shift_ppm = shift / SF
        else:
            if isinstance(shift, np.ndarray):
                shift_ppm = shift[0]
            else:
                shift_ppm = shift

        return _shift_single_spectrum(ppm, spectra, shift_ppm, method)

    # Handle matrix input
    if not isinstance(spectra, np.ndarray):
        warning(
            "Argument spectra being cast to numpy array. "
            "Unpredictable results may follow if casting fails.",
            prefix="nmr_spectra_processing::shift_spectra"
        )
        spectra = np.asarray(spectra)

    if spectra.ndim != 2:
        # Try to convert to matrix
        spectra = np.atleast_2d(spectra)

    num_spectra = spectra.shape[0]

    # Recycle shift and SF to match number of spectra
    if isinstance(shift, (int, float)):
        shift = np.array([shift])
    else:
        shift = np.asarray(shift)

    if isinstance(SF, (int, float)):
        SF = np.array([SF])
    else:
        SF = np.asarray(SF)

    # Tile to match number of spectra
    if len(shift) < num_spectra:
        num_repeats = (num_spectra // len(shift)) + 1
        shift = np.tile(shift, num_repeats)[:num_spectra]

    if len(SF) < num_spectra:
        num_repeats = (num_spectra // len(SF)) + 1
        SF = np.tile(SF, num_repeats)[:num_spectra]

    # Convert to ppm if needed
    if hertz:
        shift_ppm = shift / SF
    else:
        shift_ppm = shift

    # Shift each spectrum
    shifted_spectra = np.zeros_like(spectra)
    for i in range(num_spectra):
        shifted_spectra[i] = _shift_single_spectrum(
            ppm, spectra[i], shift_ppm[i], method
        )

    return shifted_spectra

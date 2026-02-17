"""
Utility functions for spectrum selection.

Functions for finding spectra with highest/lowest intensities.
"""

from typing import Optional, Tuple, Union
import numpy as np
from nmr_spectra_processing.utils.regions import crop_region, get_indices
from nmr_spectra_processing.utils.logger import warning, error


def get_top_spectra(
    ppm: np.ndarray,
    spectra: np.ndarray,
    n: int = 10,
    cshift: Optional[float] = None,
    roi: Optional[Tuple[float, float]] = None,
    bottom: bool = False,
    return_indices: bool = False
) -> Union[np.ndarray, np.ndarray]:
    """
    Get n spectra with highest (or lowest) intensity at a specific shift or range.

    Migrated from R function top().

    Args:
        ppm: Chemical shift scale
        spectra: Intensity matrix (spectra in rows)
        n: Number of spectra to return (default: 10)
        cshift: Specific chemical shift value to query
        roi: Chemical shift range (start, end) to query
        bottom: If True, return spectra with lowest intensity (default: False)
        return_indices: If True, return row indices instead of spectra (default: False)

    Returns:
        If return_indices is True: Array of row indices
        Otherwise: Matrix of selected spectra (n × spectrum_length)

    Notes:
        - If cshift is provided, it takes priority over roi
        - If neither cshift nor roi is provided, uses full spectrum range
        - n is capped at the number of available spectra
        - For cshift: returns spectra with highest intensity at that point
        - For roi: returns spectra with highest max intensity within that range

    Examples:
        >>> ppm = np.linspace(0, 10, 1000)
        >>> spectra = np.random.randn(100, 1000)
        >>> # Get 5 spectra with highest intensity at 5.0 ppm
        >>> top5 = get_top_spectra(ppm, spectra, n=5, cshift=5.0)
        >>> # Get indices of 10 spectra with lowest intensity in TSP region
        >>> indices = get_top_spectra(ppm, spectra, n=10,
        ...                           roi=(-0.02, 0.02), bottom=True, return_indices=True)
    """
    # Type checking with warnings
    if not isinstance(ppm, np.ndarray):
        warning(
            "Argument ppm being cast to numpy array. "
            "Unpredictable results may follow if casting fails.",
            prefix="nmr_spectra_processing::get_top_spectra"
        )
        ppm = np.asarray(ppm)

    if not isinstance(spectra, np.ndarray):
        warning(
            "Argument spectra being cast to numpy array. "
            "Unpredictable results may follow if casting fails.",
            prefix="nmr_spectra_processing::get_top_spectra"
        )
        spectra = np.asarray(spectra)

    # Ensure spectra is 2D
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)

    if not np.issubdtype(spectra.dtype, np.number):
        error(
            "Expected spectra to be a numeric array",
            prefix="nmr_spectra_processing::get_top_spectra"
        )
        raise TypeError("spectra must be numeric")

    # Cap n at number of available spectra
    n = min(n, spectra.shape[0])

    # Determine indices based on cshift or roi
    if cshift is not None:
        # Query at specific chemical shift
        idx_at_shift = get_indices(ppm, cshift)
        intensities = spectra[:, idx_at_shift]
        sorted_indices = np.argsort(intensities)
    else:
        # Query within region (default: full range)
        if roi is None:
            roi = (-np.inf, np.inf)
        mask = crop_region(ppm, roi=roi)

        # Get max intensity within region for each spectrum
        max_intensities = np.max(spectra[:, mask], axis=1)
        sorted_indices = np.argsort(max_intensities)

    # Get top n indices (reverse if not bottom)
    if not bottom:
        selected_indices = sorted_indices[-n:][::-1]  # Highest values, descending
    else:
        selected_indices = sorted_indices[:n]  # Lowest values, ascending

    if return_indices:
        return selected_indices
    else:
        return spectra[selected_indices, :]

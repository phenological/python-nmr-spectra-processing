"""
Utility functions for region selection and indexing.

Functions for cropping spectral regions and finding indices of chemical shifts.
"""

from typing import Optional, Tuple, Union
import numpy as np


def crop_region(
    ppm: np.ndarray,
    start: Optional[float] = None,
    end: Optional[float] = None,
    roi: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Create a boolean mask to crop a spectral region.

    Migrated from R function crop().

    Args:
        ppm: Chemical shift scale
        start: Lower limit of the region (default: -inf)
        end: Upper limit of the region (default: +inf)
        roi: Tuple of (start, end) limits. Takes priority over start/end

    Returns:
        Boolean mask for elements of ppm within the specified region

    Notes:
        - If roi is provided, it takes priority over start/end
        - If start is given but not end, upper limit is max(ppm)
        - If end is given but not start, lower limit is min(ppm)

    Examples:
        >>> ppm = np.linspace(0, 10, 100)
        >>> mask = crop_region(ppm, start=2.0, end=5.0)
        >>> cropped_ppm = ppm[mask]
    """
    if roi is not None:
        start, end = roi

    # Set defaults if not specified
    if start is None:
        start = -np.inf
    if end is None:
        end = np.inf

    return (ppm >= start) & (ppm <= end)


def get_indices(ppm: np.ndarray, *values: float) -> Union[int, np.ndarray]:
    """
    Get indices of chemical shifts closest to query values.

    Migrated from R function getI().

    Args:
        ppm: Chemical shift scale
        *values: One or more query chemical shift values

    Returns:
        Index (int) if single value, or array of indices if multiple values

    Notes:
        Finds the element(s) of the chemical shift scale closest to the
        given value(s). Useful for getting approximate intensity at a
        given chemical shift.

    Examples:
        >>> ppm = np.linspace(0, 10, 1000)
        >>> idx = get_indices(ppm, 5.0)
        >>> # Get indices for multiple values
        >>> indices = get_indices(ppm, 1.48, 5.22, 7.26)
    """
    if len(values) == 0:
        raise ValueError("At least one query value is required")

    indices = []
    for value in values:
        idx = np.argmin(np.abs(ppm - value))
        indices.append(idx)

    if len(indices) == 1:
        return indices[0]
    return np.array(indices)

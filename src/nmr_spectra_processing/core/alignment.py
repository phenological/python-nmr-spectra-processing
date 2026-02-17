"""
Alignment functions for NMR spectra.

Provides cross-correlation-based alignment to reference spectra.
"""

from typing import Optional, Union, Callable
import numpy as np
from scipy.signal import correlate
from nmr_spectra_processing.core.shifting import shift_series
from nmr_spectra_processing.utils.logger import warning, error


def align_spectra(
    spectra: np.ndarray,
    ref: Union[str, int, np.ndarray, Callable] = "median",
    threshold: float = 0.6,
    return_shifts: bool = False,
    padding: str = "zeroes",
    from_region: Optional[Union[np.ndarray, slice]] = None,
    **kwargs
) -> np.ndarray:
    """
    Align spectra to a reference using cross-correlation.

    Migrated from R function alignSeries().

    Args:
        spectra: Series to align (1D array) or matrix with spectra in rows
        ref: Reference specification. Can be:
            - "median" (default): Use median of spectra
            - "mean": Use mean of spectra
            - int: Row index to use as reference
            - array: Literal reference spectrum
            - bool array: Logical filter selecting reference row
            - callable: Function called with spectra, returns reference
        threshold: Minimum correlation for alignment (default 0.6).
                  Below this, series left unshifted.
        return_shifts: If True, return shift values instead of aligned spectra
        padding: Padding method for shift_series ("zeroes", "circular", "sampling")
        from_region: Region for sampling padding (see shift_series)
        **kwargs: Additional arguments (reserved for future use)

    Returns:
        Aligned spectra (same shape as input) or shift values if return_shifts=True

    Notes:
        - Uses cross-correlation to find optimal alignment
        - Spectra with correlation < threshold are left unshifted
        - Positive shifts move right, negative move left
        - For matrix input, aligns all rows to the reference

    Examples:
        >>> # Align to median
        >>> aligned = align_spectra(spectra, ref="median")
        >>> # Align to first spectrum
        >>> aligned = align_spectra(spectra, ref=0)
        >>> # Get shift values only
        >>> shifts = align_spectra(spectra, ref="median", return_shifts=True)
    """
    # Helper function to align single spectrum
    def _align_single(x, ref_spec, threshold, apply_shift, padding, from_region):
        """Align single spectrum to reference."""
        # Compute cross-correlation
        correlation = correlate(x, ref_spec, mode='same', method='auto')

        # Find lag at maximum correlation
        max_idx = np.argmax(correlation)
        max_corr = correlation[max_idx]

        # Check if correlation diverged (e.g., constant series)
        if max_corr == 0 or not np.isfinite(max_corr):
            warning(
                "Cross-correlation diverged, may be due to non-converging input "
                "(i.e., convex or constant series). Series left unshifted.",
                prefix="nmr_spectra_processing::align_spectra"
            )
            lag = 0
        else:
            # Compute lag from index (center is at len(x)//2)
            lag = max_idx - len(x) // 2

            # Normalize correlation to [0, 1] range for threshold comparison
            # scipy correlate doesn't normalize, so we do it manually
            norm_corr = max_corr / (np.sqrt(np.sum(x**2) * np.sum(ref_spec**2)) + 1e-10)

            if norm_corr < threshold:
                warning(
                    f"Cross-correlation {norm_corr:.3f} lower than threshold {threshold}. "
                    "Series left unshifted.",
                    prefix="nmr_spectra_processing::align_spectra"
                )
                lag = 0

        if apply_shift:
            # Shift series by negative lag (opposite direction of correlation lag)
            return shift_series(x, -lag, padding=padding, from_region=from_region)
        else:
            return -lag

    # Convert to array first if needed
    if not isinstance(spectra, np.ndarray):
        warning(
            "Argument spectra being cast to numpy array. "
            "Unpredictable results may follow if casting fails.",
            prefix="nmr_spectra_processing::align_spectra"
        )
        spectra = np.asarray(spectra)

    # Handle vector input (single spectrum)
    if spectra.ndim == 1:
        if not np.issubdtype(spectra.dtype, np.number):
            warning(
                "Non-numeric argument spectra being cast as numeric. "
                "Unpredictable results may follow if casting fails.",
                prefix="nmr_spectra_processing::align_spectra"
            )
            spectra = spectra.astype(float)

        # Validate reference
        if not isinstance(ref, np.ndarray):
            warning(
                "Expected numeric reference. "
                "Non-numeric argument ref being cast as numeric.",
                prefix="nmr_spectra_processing::align_spectra"
            )
            ref = np.asarray(ref)

        if len(ref) != len(spectra):
            error(
                "Expected a numeric reference of the same length as the input series",
                prefix="nmr_spectra_processing::align_spectra"
            )
            raise ValueError("Reference length must match spectra length")

        return _align_single(
            spectra, ref, threshold, not return_shifts, padding, from_region
        )

    # Handle matrix input
    if spectra.ndim != 2:
        # Try to convert to matrix
        spectra = np.atleast_2d(spectra)

    if not np.issubdtype(spectra.dtype, np.number):
        error(
            "Expected spectra to be a numeric array or matrix",
            prefix="nmr_spectra_processing::align_spectra"
        )
        raise TypeError("Spectra must be numeric")

    # Parse reference and build reference spectrum
    ref_spec = None

    if callable(ref):
        # Call function with spectra
        ref_spec = ref(spectra)

    elif isinstance(ref, str):
        # "median" or "mean"
        if ref == "median":
            ref_spec = np.median(spectra, axis=0)
        elif ref == "mean":
            ref_spec = np.mean(spectra, axis=0)
        else:
            error(
                "Invalid character ref: must be 'median' (default), 'mean', or a function.",
                prefix="nmr_spectra_processing::align_spectra"
            )
            raise ValueError(f"Unknown reference type: {ref}")

    elif isinstance(ref, (int, np.integer)):
        # Row index
        if ref < spectra.shape[0]:
            ref_spec = spectra[ref]
        else:
            error(
                "ref index not found in input matrix",
                prefix="nmr_spectra_processing::align_spectra"
            )
            raise IndexError(f"Index {ref} out of bounds for {spectra.shape[0]} spectra")

    elif isinstance(ref, np.ndarray):
        if ref.dtype == bool:
            # Boolean indexing
            ref_spec = spectra[ref]
            if ref_spec.ndim == 2 and ref_spec.shape[0] == 1:
                ref_spec = ref_spec[0]
        else:
            # Numeric array - literal reference
            if len(ref.shape) == 0:
                # Scalar - treat as index
                ref_spec = spectra[int(ref)]
            elif len(ref) == 1:
                # Single element - treat as index
                ref_spec = spectra[int(ref[0])]
            else:
                # Array - literal reference
                ref_spec = ref

    # Validate reference
    if ref_spec is None or not isinstance(ref_spec, np.ndarray):
        error(
            "Invalid reference",
            prefix="nmr_spectra_processing::align_spectra"
        )
        raise ValueError("Could not parse reference")

    if len(ref_spec) != spectra.shape[1]:
        error(
            "Invalid reference - length mismatch",
            prefix="nmr_spectra_processing::align_spectra"
        )
        raise ValueError(f"Reference length {len(ref_spec)} does not match spectra width {spectra.shape[1]}")

    # Align each spectrum
    results = []
    for i in range(spectra.shape[0]):
        result = _align_single(
            spectra[i], ref_spec, threshold, not return_shifts, padding, from_region
        )
        results.append(result)

    result_array = np.array(results)

    # If returning shifts, result is 1D; if returning spectra, result is 2D
    return result_array

"""
Normalization functions for NMR spectra.

Provides PQN (Probabilistic Quotient Normalization) and other normalization methods.
"""

from typing import Optional, Union, Callable
import numpy as np
from nmr_spectra_processing.utils.logger import error


def pqn(
    spectra: np.ndarray,
    ref: Union[str, Callable, np.ndarray] = "median",
    of_region: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute PQN (Probabilistic Quotient Normalization) dilution factors.

    Migrated from R function pqn().

    Args:
        spectra: Spectrum (1D array) or matrix with spectra in rows
        ref: Reference for PQN. Can be:
            - "median" (default): Use median of spectra
            - "mean": Use mean of spectra
            - callable: Function to compute reference from spectra
            - array: Literal reference spectrum
        of_region: Boolean mask selecting spectra rows to use for reference calculation.
                   Only used when ref is callable. Default: use all spectra.

    Returns:
        Dilution factors (1D array for matrix, scalar for vector).
        To normalize, divide each spectrum by its dilution factor.

    Notes:
        - Implements Probabilistic Quotient Normalization (doi:10.1021/ac051632c)
        - Returns dilution factors, NOT normalized spectra
        - For matrix input, returns one factor per spectrum
        - Dilution factor = median(spectrum / reference)

    Examples:
        >>> # Get dilution factors
        >>> factors = pqn(spectra, ref="median")
        >>> # Apply normalization
        >>> normalized = spectra / factors[:, np.newaxis]

        >>> # Or use normalize() wrapper
        >>> normalized = normalize(spectra, method="pqn")
    """
    # Type checking
    if not isinstance(spectra, np.ndarray):
        spectra = np.asarray(spectra)

    if not np.issubdtype(spectra.dtype, np.number):
        error(
            "Non-numeric spectrum",
            prefix="nmr_spectra_processing::pqn"
        )
        raise TypeError("Spectra must be numeric")

    # Handle callable reference
    if callable(ref):
        if spectra.ndim != 2:
            error(
                "Refuses to pqn a single spectrum with a reference "
                "extracted from the same spectrum. "
                "Please contact the developer if you think that would make sense.",
                prefix="nmr_spectra_processing::pqn"
            )
            raise ValueError("Cannot use callable reference with single spectrum")

        # Apply filter if provided
        if of_region is None:
            ref_spec = ref(spectra)
        else:
            if isinstance(of_region, np.ndarray) and of_region.dtype == bool:
                # Boolean mask
                ref_spec = ref(spectra[of_region, :])
            else:
                # Assume integer indices
                ref_spec = ref(spectra[of_region, :])

    # Handle string reference
    elif isinstance(ref, str):
        if spectra.ndim != 2:
            error(
                "String reference requires matrix input",
                prefix="nmr_spectra_processing::pqn"
            )
            raise ValueError("String reference requires matrix input")

        # Apply filter if provided
        if of_region is None:
            spectra_for_ref = spectra
        else:
            if isinstance(of_region, np.ndarray) and of_region.dtype == bool:
                spectra_for_ref = spectra[of_region, :]
            else:
                spectra_for_ref = spectra[of_region, :]

        if ref == "median":
            ref_spec = np.median(spectra_for_ref, axis=0)
        elif ref == "mean":
            ref_spec = np.mean(spectra_for_ref, axis=0)
        else:
            error(
                f"Unknown string reference: {ref}. Use 'median' or 'mean'.",
                prefix="nmr_spectra_processing::pqn"
            )
            raise ValueError(f"Unknown reference: {ref}")

    # Handle numeric array reference
    elif isinstance(ref, np.ndarray):
        ref_spec = ref

    else:
        error(
            "Invalid reference",
            prefix="nmr_spectra_processing::pqn"
        )
        raise TypeError("Reference must be callable, string, or array")

    # Validate reference length
    n = spectra.shape[-1]  # Last dimension is spectrum length
    if len(ref_spec) != n:
        error(
            f"Reference length {len(ref_spec)} does not match spectrum length {n}.",
            prefix="nmr_spectra_processing::pqn"
        )
        raise ValueError("Reference length mismatch")

    # Compute dilution factors
    if spectra.ndim == 2:
        # Matrix: compute factor for each row
        quotients = spectra / ref_spec[np.newaxis, :]
        dilution_factors = np.median(quotients, axis=1)
        return dilution_factors
    else:
        # Vector: single dilution factor
        quotients = spectra / ref_spec
        return np.median(quotients)


def normalize(
    spectra: np.ndarray,
    method: Union[str, Callable] = "pqn",
    **kwargs
) -> np.ndarray:
    """
    Normalize spectra using specified method.

    Migrated from R function normalize().

    Args:
        spectra: Spectrum (1D array) or matrix with spectra in rows
        method: Normalization method. Can be:
            - "pqn" (default): Probabilistic Quotient Normalization
            - "max": Normalize to maximum intensity
            - "sum": Normalize to total area (sum)
            - callable: Custom normalization function
        **kwargs: Additional arguments passed to normalization method

    Returns:
        Normalized spectra (same shape as input)

    Notes:
        - Convenience wrapper that divides spectra by normalization coefficients
        - For "max" and "sum", applies row-wise for matrix input
        - For "pqn", passes kwargs to pqn() function
        - For callable, function should return coefficients to divide by

    Examples:
        >>> # PQN normalization (default)
        >>> normalized = normalize(spectra)
        >>> # Max intensity normalization
        >>> normalized = normalize(spectra, method="max")
        >>> # Total area normalization
        >>> normalized = normalize(spectra, method="sum")
        >>> # Custom method
        >>> normalized = normalize(spectra, method=lambda x: np.std(x, axis=1))
    """
    # Type checking
    if not isinstance(spectra, np.ndarray):
        spectra = np.asarray(spectra)

    if not np.issubdtype(spectra.dtype, np.number):
        error(
            "Non-numeric spectrum",
            prefix="nmr_spectra_processing::normalize"
        )
        raise TypeError("Spectra must be numeric")

    # Handle string methods
    if isinstance(method, str):
        if method == "pqn":
            # Use pqn function
            factors = pqn(spectra, **kwargs)
            if spectra.ndim == 2:
                return spectra / factors[:, np.newaxis]
            else:
                return spectra / factors

        elif method == "max":
            # Max intensity normalization
            if spectra.ndim == 2:
                max_vals = np.max(spectra, axis=1)
                return spectra / max_vals[:, np.newaxis]
            else:
                return spectra / np.max(spectra)

        elif method == "sum":
            # Total area normalization
            if spectra.ndim == 2:
                sum_vals = np.sum(spectra, axis=1)
                return spectra / sum_vals[:, np.newaxis]
            else:
                return spectra / np.sum(spectra)

        else:
            error(
                f"Unknown normalization method: {method}",
                prefix="nmr_spectra_processing::normalize"
            )
            raise ValueError(f"Unknown method: {method}")

    # Handle callable method
    elif callable(method):
        # Apply custom method
        coefficients = method(spectra)
        if spectra.ndim == 2:
            # Ensure coefficients are correct shape
            if np.isscalar(coefficients):
                return spectra / coefficients
            elif len(coefficients) == spectra.shape[0]:
                return spectra / coefficients[:, np.newaxis]
            else:
                error(
                    "Custom method returned coefficients of wrong shape",
                    prefix="nmr_spectra_processing::normalize"
                )
                raise ValueError("Coefficient shape mismatch")
        else:
            return spectra / coefficients

    else:
        error(
            "Method must be string or callable",
            prefix="nmr_spectra_processing::normalize"
        )
        raise TypeError("Invalid method type")

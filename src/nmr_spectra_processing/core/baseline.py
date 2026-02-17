"""
Baseline correction functions for NMR spectra.

Provides asymmetric least squares (ALS) baseline correction.
"""

from typing import Union
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from nmr_spectra_processing.utils.logger import warning, error


def baseline_correction(
    spectra: np.ndarray,
    method: str = "als",
    lam: float = 1e5,
    p: float = 0.001,
    niter: int = 10
) -> np.ndarray:
    """
    Baseline correction using Asymmetric Least Squares (ALS).

    Migrated from R function baselineCorrection() which wraps ptw::baseline.corr().

    Args:
        spectra: Spectrum intensities (1D array) or matrix with spectra in rows
        method: Baseline correction method. Currently only "als" supported.
        lam: Smoothness parameter (lambda). Larger values = smoother baseline.
             Typical range: 1e5 to 1e7. Default: 1e5.
        p: Asymmetry parameter. Smaller values = more baseline.
           Typical range: 0.001 to 0.1. Default: 0.001.
        niter: Number of iterations. Default: 10.

    Returns:
        Baseline-corrected spectra (same shape as input)

    Notes:
        - Implements Asymmetric Least Squares smoothing
        - Based on Eilers & Boelens (2005) "Baseline Correction with
          Asymmetric Least Squares Smoothing"
        - The algorithm iteratively fits a smooth baseline with asymmetric weights
        - Points above baseline get low weight (p), points below get high weight (1-p)
        - This allows the baseline to follow the lower envelope of the spectrum

    Examples:
        >>> # Correct single spectrum
        >>> corrected = baseline_correction(spectrum)
        >>> # Correct with custom parameters
        >>> corrected = baseline_correction(spectrum, lam=1e6, p=0.01)
        >>> # Correct multiple spectra
        >>> corrected = baseline_correction(spectra)
    """
    # Type checking
    if not isinstance(spectra, np.ndarray):
        warning(
            "Argument spectra being cast as numpy array. "
            "Unpredictable results may follow if casting fails.",
            prefix="nmr_spectra_processing::baseline_correction"
        )
        spectra = np.asarray(spectra)

    if not np.issubdtype(spectra.dtype, np.number):
        warning(
            "Argument spectra being cast as numeric. "
            "Unpredictable results may follow if casting fails.",
            prefix="nmr_spectra_processing::baseline_correction"
        )
        spectra = spectra.astype(float)

    # Validate method
    if method != "als":
        error(
            f"Unknown baseline correction method: {method}. Only 'als' is supported.",
            prefix="nmr_spectra_processing::baseline_correction"
        )
        raise ValueError(f"Unknown method: {method}")

    # Validate parameters
    if lam <= 0:
        error(
            "lam (lambda) must be positive",
            prefix="nmr_spectra_processing::baseline_correction"
        )
        raise ValueError("lam must be positive")

    if not (0 < p < 1):
        error(
            "p (asymmetry) must be between 0 and 1",
            prefix="nmr_spectra_processing::baseline_correction"
        )
        raise ValueError("p must be between 0 and 1")

    if niter < 1:
        error(
            "niter must be at least 1",
            prefix="nmr_spectra_processing::baseline_correction"
        )
        raise ValueError("niter must be at least 1")

    # Define ALS baseline estimation function
    def _als_baseline(y, lam, p, niter):
        """
        Asymmetric Least Squares baseline estimation.

        Args:
            y: 1D array of intensities
            lam: Smoothness parameter
            p: Asymmetry parameter
            niter: Number of iterations

        Returns:
            Estimated baseline
        """
        L = len(y)

        # Difference matrix (2nd derivative for smoothness penalty)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
        D = lam * D.dot(D.transpose())  # D'D

        # Initialize weights
        w = np.ones(L)

        # Iteratively reweighted least squares
        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = (W + D).tocsr()  # Convert to CSR for efficient solving
            z = spsolve(Z, w * y)

            # Update weights based on residuals
            # Points above baseline (y > z) get low weight (p)
            # Points below baseline (y <= z) get high weight (1-p)
            w = p * (y > z) + (1 - p) * (y <= z)

        return z

    # Handle vector input (single spectrum)
    if spectra.ndim == 1:
        baseline = _als_baseline(spectra, lam, p, niter)
        return spectra - baseline

    # Handle matrix input (multiple spectra)
    elif spectra.ndim == 2:
        corrected = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            baseline = _als_baseline(spectra[i], lam, p, niter)
            corrected[i] = spectra[i] - baseline
        return corrected

    else:
        error(
            "Spectra must be 1D or 2D array",
            prefix="nmr_spectra_processing::baseline_correction"
        )
        raise ValueError("Spectra must be 1D or 2D")

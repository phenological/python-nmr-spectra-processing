"""
Padding functions for NMR spectra.

Provides methods to pad series with zeros, circular wrapping, or sampled values.
"""

from typing import Optional, Union
import numpy as np
from nmr_spectra_processing.utils.logger import warning, error


def pad_series(
    x: np.ndarray,
    n: int,
    side: int = 1,
    method: str = "zeroes",
    from_region: Optional[Union[np.ndarray, slice]] = None
) -> np.ndarray:
    """
    Pad a series on either extreme with specified number of points.

    Migrated from R function pad().

    Args:
        x: Series to be padded (1D array)
        n: Number of points to add
        side: Where to pad (-1: left, 0: both, 1: right/default)
        method: Padding method ("zeroes", "circular", "sampling")
        from_region: For "sampling" method - boolean mask or integer indices
                    specifying region to sample from. Default: last 1/15th points

    Returns:
        Padded series

    Notes:
        - "zeroes": Pads with 0 values
        - "circular": Wraps end to start (or vice versa). n must be <= len(x)
        - "sampling": Randomly samples with replacement from specified region

    Examples:
        >>> x = np.arange(10)
        >>> # Pad right with zeros
        >>> padded = pad_series(x, n=3, side=1, method="zeroes")
        >>> # Pad both sides with sampling
        >>> padded = pad_series(x, n=5, side=0, method="sampling")
        >>> # Circular padding on left
        >>> padded = pad_series(x, n=2, side=-1, method="circular")
    """
    # Type checking
    if not isinstance(x, np.ndarray):
        warning(
            "Argument x being cast to numpy array. "
            "Unpredictable results will follow if casting fails.",
            prefix="nmr_spectra_processing::pad_series"
        )
        x = np.asarray(x)

    if not np.issubdtype(x.dtype, np.number):
        warning(
            "Argument x being cast as numeric. "
            "Unpredictable results may follow if casting to numeric array fails.",
            prefix="nmr_spectra_processing::pad_series"
        )
        x = x.astype(float)

    # Validate side parameter
    if side not in (-1, 0, 1):
        error(
            "Wrong side specification: use -1 for left, 1 for right, and 0 for both",
            prefix="nmr_spectra_processing::pad_series"
        )
        raise ValueError("side must be -1, 0, or 1")

    # Convert n to integer
    n = int(n)

    # Method: zeroes
    if method == "zeroes":
        zeros = np.zeros(n, dtype=x.dtype)
        if side == -1:
            return np.concatenate([zeros, x])
        elif side == 0:
            return np.concatenate([zeros, x, zeros])
        else:  # side == 1
            return np.concatenate([x, zeros])

    # Method: sampling
    elif method == "sampling":
        # Set up sampling region
        if from_region is None:
            # Default: last 1/15th points
            start_idx = int(len(x) * 14 / 15)
            from_region = slice(start_idx, len(x))

        # Validate from_region
        if isinstance(from_region, np.ndarray):
            if from_region.dtype == bool:
                # Boolean mask
                if len(from_region) != len(x):
                    warning(
                        "Invalid argument value for from_region, "
                        "switching to default last 1/15 points for sampling",
                        prefix="nmr_spectra_processing::pad_series"
                    )
                    start_idx = int(len(x) * 14 / 15)
                    sample_pool = x[start_idx:]
                else:
                    # Use boolean indexing
                    sample_pool = x[from_region]
            else:
                # Integer indices
                try:
                    sample_pool = x[from_region.astype(int)]
                except (IndexError, ValueError):
                    warning(
                        "Invalid indices in from_region, "
                        "switching to default last 1/15 points",
                        prefix="nmr_spectra_processing::pad_series"
                    )
                    start_idx = int(len(x) * 14 / 15)
                    sample_pool = x[start_idx:]
        elif isinstance(from_region, slice):
            sample_pool = x[from_region]
        else:
            warning(
                "Invalid from_region type, switching to default last 1/15 points",
                prefix="nmr_spectra_processing::pad_series"
            )
            start_idx = int(len(x) * 14 / 15)
            sample_pool = x[start_idx:]

        # Sample with replacement
        sampled = np.random.choice(sample_pool, size=n, replace=True)

        if side == -1:
            return np.concatenate([sampled, x])
        elif side == 0:
            sampled_right = np.random.choice(sample_pool, size=n, replace=True)
            return np.concatenate([sampled, x, sampled_right])
        else:  # side == 1
            return np.concatenate([x, sampled])

    # Method: circular
    elif method == "circular":
        N = len(x)

        # Validate n for circular padding
        if n > N:
            error(
                "n greater than series length not allowed in circular padding",
                prefix="nmr_spectra_processing::pad_series"
            )
            raise ValueError("n cannot exceed series length for circular padding")

        # Circular padding on both ends doesn't make sense; default to right
        if side == -1:
            # Pad left: take last n elements
            return np.concatenate([x[N-n:], x])
        else:  # side == 1 or side == 0 (both → right by default)
            # Pad right: take first n elements
            return np.concatenate([x, x[:n]])

    else:
        error(
            f"Unknown padding method: {method}. Use 'zeroes', 'circular', or 'sampling'",
            prefix="nmr_spectra_processing::pad_series"
        )
        raise ValueError(f"Unknown method: {method}")

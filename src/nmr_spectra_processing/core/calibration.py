"""
Calibration functions for NMR spectra.

Provides reference-based spectral calibration using cross-correlation alignment.
"""

from typing import Union, Optional, Tuple
import numpy as np
from nmr_spectra_processing.core.alignment import align_spectra
from nmr_spectra_processing.core.baseline import baseline_correction
from nmr_spectra_processing.core.shifting import shift_series
from nmr_spectra_processing.reference import NMRSignal, get_reference_signal
from nmr_spectra_processing.utils.logger import warning, error


def calibrate_signal(
    ppm: np.ndarray,
    spectrum: np.ndarray,
    signal: Union[str, NMRSignal],
    roi: Optional[Tuple[float, float]] = None,
    max_shift: float = 1/3,
    threshold: float = 0.2,
    lam: float = 1e3,
    frequency: float = 600.0,
    apply_shift: bool = True,
    padding: str = "zeroes",
    from_region: Optional[Union[np.ndarray, slice]] = None,
    **kwargs
) -> Union[np.ndarray, float]:
    """
    Calibrate single spectrum to reference signal.

    Args:
        ppm: Chemical shift scale (1D array)
        spectrum: Spectrum intensities (1D array)
        signal: Reference signal - either string ("tsp", "alanine", "glucose", "serum")
                or NMRSignal object
        roi: Region of interest (start, end) in ppm for alignment.
             Default: auto-determined from signal type.
        max_shift: Maximum allowed shift in ppm (default 0.1)
        threshold: Minimum correlation for alignment (default 0.2)
        lam: Lambda parameter for baseline correction (default 1e3)
        frequency: Spectrometer frequency in MHz (default 600)
        apply_shift: If True, return shifted spectrum; if False, return shift value
        padding: Padding method for shifting ("zeroes", "circular", "sampling")
        from_region: Region for sampling padding
        **kwargs: Additional arguments (e.g., cshift, J for custom parameters)

    Returns:
        Shifted spectrum (if apply_shift=True) or shift value in ppm (if False)

    Examples:
        >>> # Calibrate to TSP
        >>> calibrated = calibrate_signal(ppm, spectrum, "tsp")
        >>> # Get shift value only
        >>> shift = calibrate_signal(ppm, spectrum, "tsp", apply_shift=False)
        >>> # Calibrate to custom alanine position
        >>> calibrated = calibrate_signal(ppm, spectrum, "alanine", cshift=1.5)
    """
    # Type checking and conversion
    if not isinstance(ppm, np.ndarray):
        warning(
            "Argument ppm being cast to numpy array",
            prefix="nmr_spectra_processing::calibrate_signal"
        )
        ppm = np.asarray(ppm)

    if not isinstance(spectrum, np.ndarray):
        warning(
            "Argument spectrum being cast to numpy array",
            prefix="nmr_spectra_processing::calibrate_signal"
        )
        spectrum = np.asarray(spectrum)

    # Parse signal
    if isinstance(signal, str):
        signal_obj = get_reference_signal(signal, frequency=frequency, **kwargs)
    elif isinstance(signal, NMRSignal):
        signal_obj = signal
    else:
        error(
            "signal must be string or NMRSignal object",
            prefix="nmr_spectra_processing::calibrate_signal"
        )
        raise TypeError("Invalid signal type")

    # Determine ROI if not provided
    if roi is None:
        if isinstance(signal, str):
            signal_lower = signal.lower()
            if signal_lower == "tsp":
                # Widen ROI to handle larger miscalibrations
                roi = (-0.15, 0.15)
            elif signal_lower == "serum":
                roi = (5.2, 5.4)
            else:
                # Default: ±0.5 ppm around signal center
                center = np.mean([peak.cshift for peak in signal_obj.peaks])
                roi = (center - 0.5, center + 0.5)
        else:
            # For NMRSignal objects, use ±0.5 ppm around center
            center = np.mean([peak.cshift for peak in signal_obj.peaks])
            roi = (center - 0.5, center + 0.5)

    # Apply ROI filter
    roi_mask = (ppm >= roi[0]) & (ppm <= roi[1])
    if not np.any(roi_mask):
        error(
            "ROI contains no data points",
            prefix="nmr_spectra_processing::calibrate_signal"
        )
        raise ValueError("ROI is outside ppm range or too narrow")

    ppm_roi = ppm[roi_mask]
    spectrum_roi = spectrum[roi_mask]

    # Baseline correction with small lambda
    spectrum_bc = baseline_correction(spectrum_roi, lam=lam)

    # Normalize spectrum
    max_val = np.max(np.abs(spectrum_bc))
    if max_val > 0:
        spectrum_norm = spectrum_bc / max_val
    else:
        warning(
            "Spectrum has zero intensity in ROI, cannot calibrate",
            prefix="nmr_spectra_processing::calibrate_signal"
        )
        if apply_shift:
            return spectrum
        else:
            return 0.0

    # Generate reference spectrum from signal
    ref_spectrum = signal_obj.to_spectrum(ppm_roi, linewidth=0.01, lineshape="lorentzian")

    # Normalize reference
    max_ref = np.max(ref_spectrum)
    if max_ref > 0:
        ref_spectrum = ref_spectrum / max_ref

    # Align spectrum to reference
    # align_spectra expects 2D array, so reshape if needed
    spectrum_2d = spectrum_norm[np.newaxis, :] if spectrum_norm.ndim == 1 else spectrum_norm
    shift_points = align_spectra(
        spectrum_2d,
        ref=ref_spectrum,
        threshold=threshold,
        return_shifts=True,
        padding=padding,
        from_region=from_region
    )

    # Extract scalar shift (align_spectra returns array)
    shift_points = shift_points[0] if isinstance(shift_points, np.ndarray) else shift_points

    # Convert shift from points to ppm
    if len(ppm_roi) > 1:
        delta_ppm_per_point = ppm_roi[1] - ppm_roi[0]
        shift_ppm = shift_points * delta_ppm_per_point
    else:
        shift_ppm = 0.0

    # Check max_shift constraint
    if abs(shift_ppm) > max_shift:
        warning(
            f"Shift {shift_ppm:.4f} ppm exceeds max_shift {max_shift} ppm. "
            "Setting shift to zero.",
            prefix="nmr_spectra_processing::calibrate_signal"
        )
        shift_ppm = 0.0

    if apply_shift:
        # Shift the full spectrum
        shift_points_full = int(round(shift_ppm / (ppm[1] - ppm[0])))
        return shift_series(spectrum, shift_points_full, padding=padding, from_region=from_region)
    else:
        return shift_ppm


def calibrate_spectra(
    ppm: np.ndarray,
    spectra: np.ndarray,
    ref: Union[str, NMRSignal] = "tsp",
    frequency: float = 600.0,
    max_shift: float = 1/3,
    threshold: float = 0.2,
    roi: Optional[Tuple[float, float]] = None,
    lam: float = 1e3,
    apply_shift: bool = True,
    padding: str = "zeroes",
    from_region: Optional[Union[np.ndarray, slice]] = None,
    **kwargs
) -> Union[np.ndarray, np.ndarray]:
    """
    Calibrate spectra to reference signal.

    Args:
        ppm: Chemical shift scale (1D array)
        spectra: Spectrum intensities (1D for single, 2D matrix for multiple spectra)
        ref: Reference signal - "tsp", "alanine", "glucose", "serum", or NMRSignal
        frequency: Spectrometer frequency in MHz (default 600)
        max_shift: Maximum allowed shift in ppm (default 0.1)
        threshold: Minimum correlation for alignment (default 0.2)
        roi: Region of interest for alignment (default: auto-determined)
        lam: Lambda for baseline correction (default 1e3)
        apply_shift: If True, return shifted spectra; if False, return shifts
        padding: Padding method ("zeroes", "circular", "sampling")
        from_region: Region for sampling padding
        **kwargs: Additional args (cshift, J for custom signal parameters)

    Returns:
        Calibrated spectra (same shape as input) or shift values

    Notes:
        - TSP: Singlet at 0.0 ppm
        - Alanine: Doublet at 1.48 ppm (J=7.26 Hz)
        - Glucose: Doublet at 5.223 ppm (J=3.63 Hz)
        - Serum: Alias for glucose with extended shift range

    Examples:
        >>> # Calibrate to TSP
        >>> calibrated = calibrate_spectra(ppm, spectra, ref="tsp")
        >>> # Calibrate to custom alanine
        >>> calibrated = calibrate_spectra(ppm, spectra, ref="alanine", cshift=1.5)
        >>> # Get shifts only
        >>> shifts = calibrate_spectra(ppm, spectra, ref="tsp", apply_shift=False)
    """
    # Type checking
    if not isinstance(ppm, np.ndarray):
        warning(
            "Argument ppm being cast to numpy array",
            prefix="nmr_spectra_processing::calibrate_spectra"
        )
        ppm = np.asarray(ppm)

    if not isinstance(spectra, np.ndarray):
        warning(
            "Argument spectra being cast to numpy array",
            prefix="nmr_spectra_processing::calibrate_spectra"
        )
        spectra = np.asarray(spectra)

    # Handle single spectrum
    if spectra.ndim == 1:
        return calibrate_signal(
            ppm, spectra, ref, roi=roi, max_shift=max_shift,
            threshold=threshold, lam=lam, frequency=frequency,
            apply_shift=apply_shift, padding=padding,
            from_region=from_region, **kwargs
        )

    # Handle multiple spectra
    if spectra.ndim == 2:
        results = []
        for i in range(spectra.shape[0]):
            result = calibrate_signal(
                ppm, spectra[i], ref, roi=roi, max_shift=max_shift,
                threshold=threshold, lam=lam, frequency=frequency,
                apply_shift=apply_shift, padding=padding,
                from_region=from_region, **kwargs
            )
            results.append(result)

        return np.array(results)

    else:
        error(
            "Spectra must be 1D or 2D array",
            prefix="nmr_spectra_processing::calibrate_spectra"
        )
        raise ValueError("Invalid spectra dimensions")

"""
NMR signal reference definitions.

Provides data structures and predefined signals for spectral calibration.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class NMRPeak:
    """
    Single NMR peak.

    Attributes:
        cshift: Chemical shift in ppm
        J: Coupling constant in Hz (default 0.0)
        multiplicity: Peak multiplicity - 1=singlet, 2=doublet, etc. (default 1)
        intensity: Relative peak intensity (default 1.0)
    """
    cshift: float
    J: float = 0.0
    multiplicity: int = 1
    intensity: float = 1.0


@dataclass
class NMRSignal:
    """
    Collection of NMR peaks representing a signal.

    Attributes:
        name: Signal name (e.g., "TSP", "Alanine")
        peaks: List of NMRPeak objects
    """
    name: str
    peaks: List[NMRPeak]

    def to_spectrum(
        self,
        ppm: np.ndarray,
        linewidth: float = 0.5,
        lineshape: str = "lorentzian"
    ) -> np.ndarray:
        """
        Generate synthetic spectrum from peaks.

        Args:
            ppm: Chemical shift scale (1D array)
            linewidth: Peak linewidth in Hz (default 0.5)
            lineshape: Peak shape - "lorentzian", "gaussian", or "pseudovoigt" (default "lorentzian")

        Returns:
            Synthetic spectrum with peaks at specified positions

        Notes:
            - Lorentzian: Classic NMR lineshape (1 / (1 + x^2))
            - Gaussian: exp(-x^2)
            - Pseudo-Voigt: Mix of Lorentzian and Gaussian
        """
        spectrum = np.zeros_like(ppm)

        for peak in self.peaks:
            # Calculate offset from peak center
            offset = ppm - peak.cshift

            # Generate lineshape
            if lineshape == "lorentzian":
                # Lorentzian: 1 / (1 + (2*offset/width)^2)
                line = 1.0 / (1.0 + (2.0 * offset / linewidth) ** 2)
            elif lineshape == "gaussian":
                # Gaussian: exp(-(2*offset/width)^2)
                line = np.exp(-((2.0 * offset / linewidth) ** 2))
            elif lineshape == "pseudovoigt":
                # Pseudo-Voigt: 0.5 * Lorentzian + 0.5 * Gaussian
                lorentz = 1.0 / (1.0 + (2.0 * offset / linewidth) ** 2)
                gauss = np.exp(-((2.0 * offset / linewidth) ** 2))
                line = 0.5 * lorentz + 0.5 * gauss
            else:
                raise ValueError(f"Unknown lineshape: {lineshape}")

            # Add to spectrum with intensity scaling
            spectrum += peak.intensity * line

        return spectrum


def get_reference_signal(
    ref: str,
    frequency: float = 600.0,
    **kwargs
) -> NMRSignal:
    """
    Get predefined reference signal.

    Args:
        ref: Reference type - "tsp", "alanine", "glucose", or "serum"
        frequency: Spectrometer frequency in MHz (for Hz→ppm conversion)
        **kwargs: Custom parameters (cshift, J, etc.)

    Returns:
        NMRSignal object with predefined peaks

    Notes:
        - TSP: Trimethylsilylpropanoic acid (singlet at 0.0 ppm)
        - Alanine: Doublet (default at 1.48 ppm, J=7.26 Hz)
        - Glucose: Alpha-glucose anomeric proton doublet (default at 5.223 ppm, J=3.63 Hz)
        - Serum: Alias for glucose with extended shift range

    Examples:
        >>> # Get TSP reference
        >>> tsp = get_reference_signal("tsp")
        >>> # Get alanine with custom chemical shift
        >>> ala = get_reference_signal("alanine", frequency=600, cshift=1.5)
        >>> # Generate spectrum
        >>> ppm = np.linspace(0, 10, 1000)
        >>> spectrum = tsp.to_spectrum(ppm)
    """
    ref = ref.lower()

    if ref == "tsp":
        # TSP: singlet at 0.0 ppm (or custom)
        cshift = kwargs.get("cshift", 0.0)
        intensity = kwargs.get("intensity", 1.0)

        return NMRSignal(
            name="TSP",
            peaks=[NMRPeak(cshift=cshift, J=0.0, multiplicity=1, intensity=intensity)]
        )

    elif ref == "alanine":
        # Alanine: doublet (CH3 group)
        # Default: 1.48 ppm, J = 7.26 Hz
        cshift = kwargs.get("cshift", 1.48)
        J = kwargs.get("J", 7.26)
        intensity = kwargs.get("intensity", 1.0)

        # Convert J from Hz to ppm
        J_ppm = J / frequency

        # Create doublet (two peaks split by J)
        return NMRSignal(
            name="Alanine",
            peaks=[
                NMRPeak(cshift=cshift - J_ppm / 2, J=J, multiplicity=2, intensity=intensity),
                NMRPeak(cshift=cshift + J_ppm / 2, J=J, multiplicity=2, intensity=intensity)
            ]
        )

    elif ref == "glucose":
        # Glucose: alpha-glucose anomeric proton doublet
        # Default: 5.223 ppm, J = 3.63 Hz
        cshift = kwargs.get("cshift", 5.223)
        J = kwargs.get("J", 3.63)
        intensity = kwargs.get("intensity", 1.0)

        # Convert J from Hz to ppm
        J_ppm = J / frequency

        # Create doublet
        return NMRSignal(
            name="Glucose",
            peaks=[
                NMRPeak(cshift=cshift - J_ppm / 2, J=J, multiplicity=2, intensity=intensity),
                NMRPeak(cshift=cshift + J_ppm / 2, J=J, multiplicity=2, intensity=intensity)
            ]
        )

    elif ref == "serum":
        # Serum: alias for glucose (commonly used in serum NMR)
        return get_reference_signal("glucose", frequency=frequency, **kwargs)

    else:
        raise ValueError(
            f"Unknown reference: {ref}. "
            f"Available references: 'tsp', 'alanine', 'glucose', 'serum'"
        )


def create_custom_signal(
    name: str,
    peak_positions: List[float],
    intensities: Optional[List[float]] = None
) -> NMRSignal:
    """
    Create custom NMR signal from peak positions.

    Args:
        name: Signal name
        peak_positions: List of chemical shifts in ppm
        intensities: Optional list of relative intensities (default: all 1.0)

    Returns:
        NMRSignal object

    Examples:
        >>> # Create triplet at 3.5 ppm with 1:2:1 intensities
        >>> signal = create_custom_signal(
        ...     "Triplet",
        ...     peak_positions=[3.45, 3.50, 3.55],
        ...     intensities=[1.0, 2.0, 1.0]
        ... )
    """
    if intensities is None:
        intensities = [1.0] * len(peak_positions)

    if len(peak_positions) != len(intensities):
        raise ValueError("peak_positions and intensities must have same length")

    peaks = [
        NMRPeak(cshift=pos, intensity=intensity)
        for pos, intensity in zip(peak_positions, intensities)
    ]

    return NMRSignal(name=name, peaks=peaks)

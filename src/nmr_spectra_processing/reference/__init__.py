"""Reference signals and data for NMR calibration."""

from nmr_spectra_processing.reference.signals import (
    NMRPeak,
    NMRSignal,
    get_reference_signal,
    create_custom_signal,
)

__all__ = [
    "NMRPeak",
    "NMRSignal",
    "get_reference_signal",
    "create_custom_signal",
]

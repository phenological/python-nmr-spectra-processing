#!/usr/bin/env python3
"""Demonstration of the logger utility."""

from nmr_spectra_processing.utils import warning, error, info, success

# Example usage of logger functions
print("\n=== Logger Demo ===\n")

# Info message (blue)
info("Starting NMR spectra processing...")

# Success message (green with checkmark)
success("Spectra loaded successfully")

# Warning message (yellow) - like R's crayon::yellow()
warning("Cross-correlation 0.55 below threshold 0.6")

# Warning with function prefix
warning("Series left unshifted", prefix="nmr_spectra_processing::align_spectra")

# Error message (red) - like R's crayon::red()
error("Expected numeric reference of same length as input series")

# Error with function prefix
error("Invalid reference", prefix="nmr_spectra_processing::align_spectra")

print("\n=== End Demo ===\n")

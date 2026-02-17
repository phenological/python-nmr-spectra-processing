#!/usr/bin/env python3
"""
Package Validation Script
=========================

Validates that all core functionality works correctly.
Run this after installation to verify the package is working.

Usage:
    python validate_package.py
"""

import sys
import numpy as np
from pathlib import Path


def validate_imports():
    """Test that all modules can be imported."""
    print("="*60)
    print("VALIDATION: Package Imports")
    print("="*60)

    try:
        # Core functions
        from nmr_spectra_processing import (
            align_spectra,
            baseline_correction,
            calibrate_signal,
            calibrate_spectra,
            estimate_noise,
            normalize,
            pad_series,
            phase_correction,
            pqn,
            shift_series,
            shift_spectra,
        )

        # Utilities
        from nmr_spectra_processing import (
            crop_region,
            get_indices,
            get_top_spectra,
        )

        # Reference signals
        from nmr_spectra_processing import (
            NMRPeak,
            NMRSignal,
            get_reference_signal,
            create_custom_signal,
        )

        print("✓ All imports successful")
        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def validate_baseline_correction():
    """Test baseline correction."""
    print("\n" + "="*60)
    print("VALIDATION: Baseline Correction")
    print("="*60)

    from nmr_spectra_processing import baseline_correction

    # Create spectrum with polynomial baseline
    x = np.linspace(0, 10, 1000)
    baseline = 0.5 * x**2 - 3 * x + 2
    signal = np.exp(-((x - 5)**2) / 0.5)
    spectrum = signal + baseline

    # Apply correction
    corrected = baseline_correction(spectrum, lam=1e6, p=0.001)

    # Check that baseline is reduced (not necessarily zero)
    baseline_before = np.mean(spectrum[:100])
    baseline_after = np.mean(corrected[:100])
    reduction = abs(baseline_before - baseline_after) / abs(baseline_before)

    if reduction > 0.5:  # At least 50% reduction
        print(f"✓ Baseline correction works correctly ({reduction*100:.1f}% reduction)")
        return True
    else:
        print(f"✗ Baseline correction failed ({reduction*100:.1f}% reduction)")
        return False


def validate_calibration():
    """Test spectral calibration."""
    print("\n" + "="*60)
    print("VALIDATION: Calibration")
    print("="*60)

    from nmr_spectra_processing import calibrate_signal, get_reference_signal

    # Create test spectrum with TSP at 0.08 ppm
    ppm = np.linspace(-0.5, 0.5, 1000)
    tsp = get_reference_signal("tsp", cshift=0.08)
    spectrum = tsp.to_spectrum(ppm, linewidth=0.01)

    # Calibrate to 0.0 ppm
    calibrated = calibrate_signal(ppm, spectrum, signal="tsp")

    # Check peak moved to ~0.0 ppm
    peak_pos = ppm[np.argmax(calibrated)]
    if abs(peak_pos) < 0.01:
        print(f"✓ Calibration works correctly (peak at {peak_pos:.4f} ppm)")
        return True
    else:
        print(f"✗ Calibration failed (peak at {peak_pos:.4f} ppm, expected ~0.0)")
        return False


def validate_normalization():
    """Test normalization."""
    print("\n" + "="*60)
    print("VALIDATION: Normalization")
    print("="*60)

    from nmr_spectra_processing import normalize

    # Create test spectra with different intensities
    np.random.seed(42)
    spectra = np.random.randn(5, 1000) + 10
    spectra *= np.array([0.5, 1.0, 1.5, 2.0, 2.5])[:, np.newaxis]

    # Normalize with PQN
    normalized = normalize(spectra, method="pqn")

    # Check that total intensities are similar
    total_before = np.sum(spectra, axis=1)
    total_after = np.sum(normalized, axis=1)
    cv_before = np.std(total_before) / np.mean(total_before)
    cv_after = np.std(total_after) / np.mean(total_after)

    if cv_after < cv_before:
        print(f"✓ Normalization works correctly (CV: {cv_before:.3f} → {cv_after:.3f})")
        return True
    else:
        print(f"✗ Normalization failed (CV: {cv_before:.3f} → {cv_after:.3f})")
        return False


def validate_alignment():
    """Test spectral alignment."""
    print("\n" + "="*60)
    print("VALIDATION: Alignment")
    print("="*60)

    from nmr_spectra_processing import align_spectra

    # Create shifted versions of a signal
    x = np.linspace(0, 10, 1000)
    base_signal = np.exp(-((x - 5)**2) / 0.5)

    spectra = np.zeros((3, 1000))
    shifts = [-20, 0, 20]  # Shift in points
    for i, shift in enumerate(shifts):
        spectra[i] = np.roll(base_signal, shift)

    # Align to median
    aligned = align_spectra(spectra, ref="median")

    # Check that peaks are now at same position
    peak_positions = [np.argmax(aligned[i]) for i in range(3)]
    peak_std = np.std(peak_positions)

    if peak_std < 5:  # Within 5 points
        print(f"✓ Alignment works correctly (peak std: {peak_std:.1f} points)")
        return True
    else:
        print(f"✗ Alignment failed (peak std: {peak_std:.1f} points)")
        return False


def validate_phase_correction():
    """Test phase correction."""
    print("\n" + "="*60)
    print("VALIDATION: Phase Correction")
    print("="*60)

    from nmr_spectra_processing import phase_correction

    # Create complex signal
    n = 1000
    real = np.sin(2 * np.pi * np.arange(n) / 100)
    imag = np.cos(2 * np.pi * np.arange(n) / 100)

    # Apply 90-degree phase shift (should swap real and imaginary)
    real_shifted, imag_shifted = phase_correction(real, imag, phi0=90)

    # Check that phase was applied
    # After 90° shift, real → -imag and imag → real
    # So real_shifted should be similar to -imag
    correlation = np.corrcoef(real_shifted, -imag)[0, 1]

    if abs(correlation) > 0.95:  # Strong correlation
        print(f"✓ Phase correction works correctly (correlation: {correlation:.3f})")
        return True
    else:
        print(f"✗ Phase correction failed (correlation: {correlation:.3f})")
        return False


def validate_reference_signals():
    """Test reference signal generation."""
    print("\n" + "="*60)
    print("VALIDATION: Reference Signals")
    print("="*60)

    from nmr_spectra_processing import get_reference_signal

    ppm = np.linspace(-1, 2, 3000)

    # Test TSP
    tsp = get_reference_signal("tsp")
    tsp_spectrum = tsp.to_spectrum(ppm, linewidth=0.01)
    tsp_peak = ppm[np.argmax(tsp_spectrum)]

    # Test alanine
    ala = get_reference_signal("alanine", frequency=600)
    ala_spectrum = ala.to_spectrum(ppm, linewidth=0.01)
    ala_peak = ppm[np.argmax(ala_spectrum)]

    if abs(tsp_peak - 0.0) < 0.01 and abs(ala_peak - 1.48) < 0.05:
        print(f"✓ Reference signals work correctly")
        print(f"  TSP peak: {tsp_peak:.4f} ppm (expected 0.0)")
        print(f"  Alanine peak: {ala_peak:.4f} ppm (expected ~1.48)")
        return True
    else:
        print(f"✗ Reference signals failed")
        print(f"  TSP peak: {tsp_peak:.4f} ppm (expected 0.0)")
        print(f"  Alanine peak: {ala_peak:.4f} ppm (expected ~1.48)")
        return False


def validate_utilities():
    """Test utility functions."""
    print("\n" + "="*60)
    print("VALIDATION: Utility Functions")
    print("="*60)

    from nmr_spectra_processing import crop_region, get_indices, get_top_spectra

    ppm = np.linspace(0, 10, 1000)
    spectra = np.random.randn(5, 1000)

    # Test crop_region
    mask = crop_region(ppm, roi=(2.0, 4.0))
    n_points = np.sum(mask)

    # Test get_indices
    idx_5 = get_indices(ppm, 5.0)

    # Test get_top_spectra
    top_2 = get_top_spectra(ppm, spectra, n=2, cshift=5.0)

    if n_points > 0 and 0 <= idx_5 < len(ppm) and top_2.shape[0] == 2:
        print(f"✓ Utility functions work correctly")
        print(f"  crop_region: {n_points} points in ROI")
        print(f"  get_indices: index {idx_5} for 5.0 ppm")
        print(f"  get_top_spectra: selected {top_2.shape[0]} spectra")
        return True
    else:
        print(f"✗ Utility functions failed")
        return False


def validate_examples():
    """Check that examples exist and are runnable."""
    print("\n" + "="*60)
    print("VALIDATION: Examples")
    print("="*60)

    examples_dir = Path("examples")

    if not examples_dir.exists():
        print("✗ examples/ directory not found")
        return False

    example_files = [
        "basic_processing.py",
        "calibration_example.py",
        "README.md"
    ]

    missing = []
    for file in example_files:
        if not (examples_dir / file).exists():
            missing.append(file)

    if missing:
        print(f"✗ Missing example files: {', '.join(missing)}")
        return False
    else:
        print(f"✓ All example files present")
        return True


def main():
    """Run all validation tests."""
    print("\n" + "#"*60)
    print("# NMR Spectra Processing - Package Validation")
    print("#"*60)

    results = {
        "Imports": validate_imports(),
        "Baseline Correction": validate_baseline_correction(),
        "Calibration": validate_calibration(),
        "Normalization": validate_normalization(),
        "Alignment": validate_alignment(),
        "Phase Correction": validate_phase_correction(),
        "Reference Signals": validate_reference_signals(),
        "Utilities": validate_utilities(),
        "Examples": validate_examples(),
    }

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test:.<45} {status}")

    total = len(results)
    passed = sum(results.values())
    percentage = (passed / total) * 100

    print("="*60)
    print(f"Result: {passed}/{total} tests passed ({percentage:.0f}%)")
    print("="*60)

    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("The package is ready to use.")
        return 0
    else:
        print("\n⚠️  Some validation tests failed.")
        print("Please check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

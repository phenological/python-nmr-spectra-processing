#!/usr/bin/env python3
"""
NMR Spectral Calibration Examples
==================================

This example demonstrates various calibration approaches for NMR spectra:
1. Calibration to TSP (trimethylsilylpropanoic acid)
2. Calibration to alanine doublet
3. Calibration to glucose anomeric peak
4. Custom signal calibration
5. Batch calibration with shift tracking

Requirements:
- numpy
- nmr_spectra_processing package installed

"""

import numpy as np
from pathlib import Path

from nmr_spectra_processing import calibrate_signal, calibrate_spectra
from nmr_spectra_processing.reference import (
    get_reference_signal,
    NMRSignal,
    NMRPeak,
    create_custom_signal,
)


def example_1_tsp_calibration():
    """Example 1: Calibration to TSP reference (singlet at 0.0 ppm)."""
    print("\n" + "="*60)
    print("EXAMPLE 1: TSP Calibration")
    print("="*60)

    # Create synthetic spectrum with TSP shifted to 0.08 ppm
    ppm = np.linspace(-1, 1, 2000)

    print("\nCreating test spectrum with TSP at 0.08 ppm (should be at 0.0)...")
    tsp_signal = get_reference_signal("tsp", cshift=0.08)
    spectrum = tsp_signal.to_spectrum(ppm, linewidth=0.01)

    # Find peak position before calibration
    peak_before = ppm[np.argmax(spectrum)]
    print(f"  Peak position before calibration: {peak_before:.4f} ppm")

    # Calibrate to TSP at 0.0 ppm
    print("\nCalibrating to TSP reference (0.0 ppm)...")
    spectrum_calibrated = calibrate_signal(
        ppm,
        spectrum,
        signal="tsp",        # Use predefined TSP reference
        frequency=600.0,     # Spectrometer frequency (MHz)
        max_shift=0.3,       # Maximum allowed shift (ppm)
        threshold=0.2,       # Correlation threshold
    )

    # Find peak position after calibration
    peak_after = ppm[np.argmax(spectrum_calibrated)]
    print(f"  Peak position after calibration:  {peak_after:.4f} ppm")
    print(f"  Shift applied: {peak_after - peak_before:.4f} ppm")

    # Get shift value without applying it
    print("\nGetting shift value only (not applying)...")
    shift_value = calibrate_signal(
        ppm,
        spectrum,
        signal="tsp",
        apply_shift=False    # Return shift value instead of calibrated spectrum
    )
    print(f"  Calculated shift: {shift_value:.4f} ppm")

    return ppm, spectrum_calibrated


def example_2_alanine_calibration():
    """Example 2: Calibration to alanine doublet (1.48 ppm)."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Alanine Calibration")
    print("="*60)

    # Create synthetic spectrum with alanine at 1.52 ppm (shifted from 1.48)
    ppm = np.linspace(0, 3, 3000)

    print("\nCreating test spectrum with alanine at 1.52 ppm (should be at 1.48)...")
    alanine_signal = get_reference_signal(
        "alanine",
        frequency=600.0,
        cshift=1.52,         # Shifted position
        J=7.26               # Coupling constant (Hz)
    )
    spectrum = alanine_signal.to_spectrum(ppm, linewidth=0.01)

    # Find doublet center before calibration
    ala_region = (ppm >= 1.4) & (ppm <= 1.6)
    peak_before = ppm[ala_region][np.argmax(spectrum[ala_region])]
    print(f"  Peak position before calibration: {peak_before:.4f} ppm")

    # Calibrate to alanine at 1.48 ppm
    print("\nCalibrating to alanine reference (1.48 ppm)...")
    spectrum_calibrated = calibrate_signal(
        ppm,
        spectrum,
        signal="alanine",
        frequency=600.0,
        cshift=1.48,         # Target position
        J=7.26               # Coupling constant
    )

    # Find doublet center after calibration
    peak_after = ppm[ala_region][np.argmax(spectrum_calibrated[ala_region])]
    print(f"  Peak position after calibration:  {peak_after:.4f} ppm")
    print(f"  Shift applied: {peak_after - peak_before:.4f} ppm")

    return ppm, spectrum_calibrated


def example_3_glucose_calibration():
    """Example 3: Calibration to glucose anomeric peak (5.223 ppm)."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Glucose Calibration")
    print("="*60)

    # Create synthetic spectrum with glucose at 5.25 ppm
    ppm = np.linspace(4, 6, 2000)

    print("\nCreating test spectrum with glucose at 5.25 ppm (should be at 5.223)...")
    glucose_signal = get_reference_signal(
        "glucose",
        frequency=600.0,
        cshift=5.25,
        J=3.63
    )
    spectrum = glucose_signal.to_spectrum(ppm, linewidth=0.01)

    peak_before = ppm[np.argmax(spectrum)]
    print(f"  Peak position before calibration: {peak_before:.4f} ppm")

    # Calibrate to glucose
    print("\nCalibrating to glucose reference (5.223 ppm)...")
    spectrum_calibrated = calibrate_signal(
        ppm,
        spectrum,
        signal="glucose",
        frequency=600.0
    )

    peak_after = ppm[np.argmax(spectrum_calibrated)]
    print(f"  Peak position after calibration:  {peak_after:.4f} ppm")
    print(f"  Shift applied: {peak_after - peak_before:.4f} ppm")

    # Alternative: Use "serum" alias (same as glucose)
    print("\nNote: 'serum' is an alias for 'glucose' calibration")
    print("  calibrate_signal(ppm, spectrum, signal='serum', ...)")

    return ppm, spectrum_calibrated


def example_4_custom_signal_calibration():
    """Example 4: Calibration to custom signal."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Signal Calibration")
    print("="*60)

    ppm = np.linspace(0, 5, 5000)

    # Create a custom triplet pattern (e.g., ethanol CH2 group)
    # Triplet with 1:2:1 intensity ratio at 3.65 ppm
    print("\nCreating custom signal (triplet at 3.65 ppm)...")
    custom_signal = create_custom_signal(
        name="Ethanol_CH2",
        peak_positions=[3.64, 3.65, 3.66],  # Positions separated by J/frequency
        intensities=[1.0, 2.0, 1.0]          # 1:2:1 ratio
    )

    # Create test spectrum with shifted triplet at 3.70 ppm
    print("Creating test spectrum with triplet at 3.70 ppm...")
    shifted_signal = create_custom_signal(
        name="Shifted_triplet",
        peak_positions=[3.69, 3.70, 3.71],
        intensities=[1.0, 2.0, 1.0]
    )
    spectrum = shifted_signal.to_spectrum(ppm, linewidth=0.01)

    # Calibrate using the custom signal as reference
    print("\nCalibrating to custom signal...")
    spectrum_calibrated = calibrate_signal(
        ppm,
        spectrum,
        signal=custom_signal,  # Pass NMRSignal object directly
        roi=(3.5, 3.8)        # Define region of interest
    )

    # Check calibration
    roi_mask = (ppm >= 3.5) & (ppm <= 3.8)
    peak_before = ppm[roi_mask][np.argmax(spectrum[roi_mask])]
    peak_after = ppm[roi_mask][np.argmax(spectrum_calibrated[roi_mask])]

    print(f"  Peak position before: {peak_before:.4f} ppm")
    print(f"  Peak position after:  {peak_after:.4f} ppm")
    print(f"  Shift applied: {peak_after - peak_before:.4f} ppm")

    return ppm, spectrum_calibrated


def example_5_batch_calibration():
    """Example 5: Batch calibration with shift tracking."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Calibration")
    print("="*60)

    # Create multiple spectra with different miscalibrations
    ppm = np.linspace(-0.5, 0.5, 1000)
    n_spectra = 5

    print(f"\nCreating {n_spectra} spectra with different TSP shifts...")

    # Known shifts for testing
    true_shifts = [-0.08, -0.04, 0.0, 0.04, 0.08]
    spectra = np.zeros((n_spectra, len(ppm)))

    for i, shift in enumerate(true_shifts):
        tsp = get_reference_signal("tsp", cshift=shift)
        spectra[i] = tsp.to_spectrum(ppm, linewidth=0.01)
        print(f"  Spectrum {i+1}: TSP at {shift:+.2f} ppm")

    # Get shift values for all spectra (without applying)
    print("\nCalculating calibration shifts for all spectra...")
    calculated_shifts = calibrate_spectra(
        ppm,
        spectra,
        ref="tsp",
        apply_shift=False    # Return shifts instead of calibrated spectra
    )

    print("\nCalibration shifts:")
    print("  Spectrum  |  True Shift  |  Calculated  |  Difference")
    print("  " + "-"*55)
    for i, (true, calc) in enumerate(zip(true_shifts, calculated_shifts)):
        diff = calc - (-true)  # Calibration shift is negative of position
        print(f"     {i+1}     |   {true:+.4f}    |   {calc:+.4f}    |   {diff:+.4f}")

    # Apply calibration to all spectra
    print("\nApplying calibration to all spectra...")
    spectra_calibrated = calibrate_spectra(
        ppm,
        spectra,
        ref="tsp",
        apply_shift=True
    )

    # Verify calibration success
    print("\nVerifying calibration results:")
    for i in range(n_spectra):
        peak_pos = ppm[np.argmax(spectra_calibrated[i])]
        print(f"  Spectrum {i+1}: TSP now at {peak_pos:+.4f} ppm (target: 0.0)")

    return ppm, spectra_calibrated


def example_6_calibration_parameters():
    """Example 6: Effect of calibration parameters."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Calibration Parameters")
    print("="*60)

    ppm = np.linspace(-0.3, 0.3, 600)

    # Create spectrum with TSP at 0.15 ppm
    tsp = get_reference_signal("tsp", cshift=0.15)
    spectrum = tsp.to_spectrum(ppm, linewidth=0.01)

    print("\nTest spectrum: TSP at 0.15 ppm")
    print("\nTesting different parameter settings...")

    # Test 1: max_shift constraint
    print("\n1. Effect of max_shift constraint:")

    for max_shift in [0.05, 0.10, 0.20]:
        shift = calibrate_signal(
            ppm, spectrum, "tsp",
            max_shift=max_shift,
            apply_shift=False
        )
        status = "✓" if shift != 0.0 else "✗ (rejected)"
        print(f"   max_shift={max_shift:.2f}: shift={shift:+.4f} ppm {status}")

    # Test 2: threshold parameter
    print("\n2. Effect of correlation threshold:")

    # Add noise to make correlation weaker
    noisy_spectrum = spectrum + np.random.randn(len(spectrum)) * 0.05

    for threshold in [0.1, 0.3, 0.5]:
        try:
            shift = calibrate_signal(
                ppm, noisy_spectrum, "tsp",
                threshold=threshold,
                apply_shift=False
            )
            print(f"   threshold={threshold:.1f}: shift={shift:+.4f} ppm")
        except Exception as e:
            print(f"   threshold={threshold:.1f}: Failed ({type(e).__name__})")

    # Test 3: ROI width
    print("\n3. Effect of ROI width:")

    for roi in [(-0.05, 0.05), (-0.10, 0.10), (-0.20, 0.20)]:
        shift = calibrate_signal(
            ppm, spectrum, "tsp",
            roi=roi,
            apply_shift=False
        )
        roi_width = roi[1] - roi[0]
        print(f"   ROI width={roi_width:.2f}: shift={shift:+.4f} ppm")

    print("\nKey insights:")
    print("  - max_shift prevents excessive corrections")
    print("  - threshold filters low-quality alignments")
    print("  - Wider ROI allows detection of larger shifts")


def save_calibration_report(output_dir="output"):
    """Generate a calibration report with all examples."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    report_file = output_path / "calibration_report.txt"

    print(f"\nSaving calibration report to {report_file}...")

    with open(report_file, "w") as f:
        f.write("NMR Spectral Calibration Report\n")
        f.write("="*60 + "\n\n")
        f.write("This report summarizes calibration capabilities:\n\n")
        f.write("1. TSP Calibration (0.0 ppm singlet)\n")
        f.write("   - Standard reference for aqueous samples\n")
        f.write("   - Narrow linewidth, singlet pattern\n\n")
        f.write("2. Alanine Calibration (1.48 ppm doublet)\n")
        f.write("   - Useful for biological samples\n")
        f.write("   - Doublet pattern with J=7.26 Hz\n\n")
        f.write("3. Glucose Calibration (5.223 ppm doublet)\n")
        f.write("   - Anomeric proton reference\n")
        f.write("   - Alternative: use 'serum' alias\n\n")
        f.write("4. Custom Signal Calibration\n")
        f.write("   - Define any multiplet pattern\n")
        f.write("   - Specify peak positions and intensities\n\n")
        f.write("5. Batch Processing\n")
        f.write("   - Calibrate multiple spectra at once\n")
        f.write("   - Track calibration shifts\n\n")
        f.write("Parameters:\n")
        f.write("  - max_shift: Maximum allowed shift (default: 0.333 ppm)\n")
        f.write("  - threshold: Correlation threshold (default: 0.2)\n")
        f.write("  - roi: Region of interest for alignment\n")
        f.write("  - lam: Baseline correction parameter (default: 1e3)\n")

    print(f"  Report saved: {report_file}")


def main():
    """Run all calibration examples."""
    print("="*60)
    print("NMR Spectral Calibration Examples")
    print("="*60)

    try:
        # Run all examples
        example_1_tsp_calibration()
        example_2_alanine_calibration()
        example_3_glucose_calibration()
        example_4_custom_signal_calibration()
        example_5_batch_calibration()
        example_6_calibration_parameters()

        # Save report
        save_calibration_report()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Basic NMR Spectra Processing Pipeline
======================================

This example demonstrates a complete processing workflow for 1D NMR spectra:
1. Load data
2. Baseline correction
3. Calibration to reference signal
4. Normalization (PQN)
5. Spectral alignment

Requirements:
- numpy
- nmr_spectra_processing package installed

"""

import numpy as np
from pathlib import Path

# Import processing functions
from nmr_spectra_processing import (
    baseline_correction,
    calibrate_spectra,
    normalize,
    align_spectra,
    estimate_noise,
)

# Optional: For creating synthetic test data
from nmr_spectra_processing.reference import get_reference_signal, NMRSignal, NMRPeak


def create_synthetic_data():
    """
    Create synthetic NMR spectra for demonstration.

    Returns:
        ppm: Chemical shift scale (1D array)
        spectra: Matrix of spectra (shape: n_spectra x n_points)
    """
    print("Creating synthetic NMR spectra...")

    # Create ppm scale (10 to 0 ppm, typical for 1H NMR)
    ppm = np.linspace(10, 0, 5000)

    # Create 10 synthetic spectra with slight variations
    n_spectra = 10
    spectra = np.zeros((n_spectra, len(ppm)))

    # Define metabolite peaks (simplified)
    metabolite_peaks = [
        # (chemical shift, intensity range, linewidth)
        (0.05, (0.8, 1.2), 0.005),   # TSP reference (should be at 0.0)
        (1.33, (0.3, 0.5), 0.01),    # Lactate
        (2.05, (0.1, 0.3), 0.015),   # Acetate
        (3.21, (0.4, 0.6), 0.012),   # Choline
        (5.23, (0.2, 0.4), 0.008),   # Glucose anomeric
        (7.85, (0.1, 0.2), 0.01),    # Aromatic
    ]

    np.random.seed(42)  # For reproducibility

    for i in range(n_spectra):
        # Add metabolite peaks with random variations
        for cshift, (int_min, int_max), linewidth in metabolite_peaks:
            # Random intensity variation
            intensity = np.random.uniform(int_min, int_max)

            # Small random shift to simulate misalignment (±0.01 ppm)
            shift_variation = np.random.uniform(-0.01, 0.01)

            # Create peak
            signal = NMRSignal(
                name=f"Peak_{cshift}",
                peaks=[NMRPeak(cshift=cshift + shift_variation, intensity=intensity)]
            )
            spectra[i] += signal.to_spectrum(ppm, linewidth=linewidth)

        # Add baseline drift (polynomial)
        baseline_drift = np.poly1d([0.001, -0.01, 0.05])(ppm)
        spectra[i] += baseline_drift

        # Add noise
        noise_level = 0.01
        spectra[i] += np.random.randn(len(ppm)) * noise_level

    print(f"  Created {n_spectra} spectra with {len(ppm)} points each")
    print(f"  Chemical shift range: {ppm[-1]:.2f} to {ppm[0]:.2f} ppm")

    return ppm, spectra


def process_spectra(ppm, spectra):
    """
    Complete processing pipeline for NMR spectra.

    Parameters:
        ppm: Chemical shift scale (1D array)
        spectra: Matrix of spectra (shape: n_spectra x n_points)

    Returns:
        processed: Fully processed spectra matrix
    """
    print("\n" + "="*60)
    print("NMR SPECTRA PROCESSING PIPELINE")
    print("="*60)

    # Step 1: Estimate noise level
    print("\n1. Estimating noise level...")
    noise_levels = estimate_noise(ppm, spectra, level=0.99, roi=(9.5, 10.0))
    print(f"   Mean noise level: {np.mean(noise_levels):.4f}")
    print(f"   Noise range: {np.min(noise_levels):.4f} - {np.max(noise_levels):.4f}")

    # Step 2: Baseline correction
    print("\n2. Applying baseline correction (ALS method)...")
    spectra_bc = baseline_correction(
        spectra,
        method="als",
        lam=1e5,      # Smoothness parameter (larger = smoother)
        p=0.001,      # Asymmetry parameter (smaller = more baseline)
        niter=10      # Number of iterations
    )
    print("   Baseline correction applied")

    # Step 3: Calibration to TSP reference
    print("\n3. Calibrating to TSP reference (0.0 ppm)...")
    spectra_cal = calibrate_spectra(
        ppm,
        spectra_bc,
        ref="tsp",           # Calibrate to TSP singlet
        frequency=600.0,     # Spectrometer frequency (MHz)
        max_shift=0.15,      # Maximum allowed shift (ppm)
        threshold=0.2,       # Correlation threshold
    )

    # Check calibration success
    tsp_region = (ppm >= -0.05) & (ppm <= 0.05)
    tsp_peaks = [ppm[tsp_region][np.argmax(spec[tsp_region])] for spec in spectra_cal]
    print(f"   TSP peak positions after calibration:")
    print(f"   Mean: {np.mean(tsp_peaks):.4f} ppm (target: 0.0 ppm)")
    print(f"   Std:  {np.std(tsp_peaks):.4f} ppm")

    # Step 4: Normalization (PQN)
    print("\n4. Applying Probabilistic Quotient Normalization (PQN)...")
    spectra_norm = normalize(
        spectra_cal,
        method="pqn",       # PQN normalization
        ref="median"        # Use median as reference
    )
    print("   PQN normalization applied")

    # Compare total intensities before and after normalization
    total_before = np.sum(spectra_cal, axis=1)
    total_after = np.sum(spectra_norm, axis=1)
    print(f"   Total intensity before normalization: {np.mean(total_before):.2f} ± {np.std(total_before):.2f}")
    print(f"   Total intensity after normalization:  {np.mean(total_after):.2f} ± {np.std(total_after):.2f}")

    # Step 5: Spectral alignment
    print("\n5. Aligning spectra to median reference...")
    spectra_aligned = align_spectra(
        spectra_norm,
        ref="median",        # Align to median spectrum
        threshold=0.6,       # Correlation threshold
        padding="zeroes"     # Padding method for shifting
    )
    print("   Spectral alignment complete")

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)

    return spectra_aligned


def save_results(ppm, spectra_raw, spectra_processed, output_dir="output"):
    """Save processing results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nSaving results to {output_dir}/...")

    # Save as NumPy arrays
    np.save(output_path / "ppm.npy", ppm)
    np.save(output_path / "spectra_raw.npy", spectra_raw)
    np.save(output_path / "spectra_processed.npy", spectra_processed)

    print(f"  Saved: ppm.npy ({ppm.shape})")
    print(f"  Saved: spectra_raw.npy ({spectra_raw.shape})")
    print(f"  Saved: spectra_processed.npy ({spectra_processed.shape})")

    # Save summary statistics
    with open(output_path / "summary.txt", "w") as f:
        f.write("NMR Spectra Processing Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of spectra: {spectra_raw.shape[0]}\n")
        f.write(f"Number of points per spectrum: {spectra_raw.shape[1]}\n")
        f.write(f"Chemical shift range: {ppm[-1]:.2f} to {ppm[0]:.2f} ppm\n")
        f.write(f"\nRaw spectra statistics:\n")
        f.write(f"  Mean intensity: {np.mean(spectra_raw):.4f}\n")
        f.write(f"  Std intensity:  {np.std(spectra_raw):.4f}\n")
        f.write(f"\nProcessed spectra statistics:\n")
        f.write(f"  Mean intensity: {np.mean(spectra_processed):.4f}\n")
        f.write(f"  Std intensity:  {np.std(spectra_processed):.4f}\n")

    print(f"  Saved: summary.txt")


def main():
    """Main execution function."""
    print("="*60)
    print("NMR Spectra Processing Example")
    print("="*60)

    # Option 1: Create synthetic data
    print("\nGenerating synthetic data for demonstration...")
    ppm, spectra_raw = create_synthetic_data()

    # Option 2: Load real data (uncomment to use)
    # print("\nLoading real data...")
    # ppm = np.load("path/to/ppm.npy")
    # spectra_raw = np.load("path/to/spectra.npy")
    # print(f"  Loaded {spectra_raw.shape[0]} spectra with {spectra_raw.shape[1]} points")

    # Process spectra
    spectra_processed = process_spectra(ppm, spectra_raw)

    # Save results
    save_results(ppm, spectra_raw, spectra_processed)

    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)
    print("\nTo visualize results, you can use matplotlib:")
    print("  import matplotlib.pyplot as plt")
    print("  plt.plot(ppm, spectra_processed.T)")
    print("  plt.xlabel('Chemical Shift (ppm)')")
    print("  plt.ylabel('Intensity')")
    print("  plt.gca().invert_xaxis()  # NMR convention")
    print("  plt.show()")


if __name__ == "__main__":
    main()

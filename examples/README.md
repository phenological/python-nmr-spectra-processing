# NMR Spectra Processing Examples

This directory contains example scripts demonstrating the usage of the `nmr-spectra-processing` package.

## Running the Examples

Make sure the package is installed first:

```bash
# From the repository root
pip install -e .

# Or if using a virtual environment
source .venv/bin/activate
pip install -e .
```

Then run any example:

```bash
python examples/basic_processing.py
python examples/calibration_example.py
```

## Available Examples

### 1. `basic_processing.py` - Complete Processing Pipeline

Demonstrates a full NMR spectra processing workflow:

- **Data generation**: Creates synthetic NMR spectra with metabolite peaks
- **Noise estimation**: Estimates noise level from blank spectral regions
- **Baseline correction**: Removes baseline drift using ALS algorithm
- **Calibration**: Aligns chemical shift scale to TSP reference (0.0 ppm)
- **Normalization**: Applies Probabilistic Quotient Normalization (PQN)
- **Spectral alignment**: Aligns spectra to median reference

**Output**:
- `output/ppm.npy` - Chemical shift scale
- `output/spectra_raw.npy` - Raw spectra matrix
- `output/spectra_processed.npy` - Fully processed spectra
- `output/summary.txt` - Processing summary statistics

**Usage**:
```bash
python examples/basic_processing.py
```

To use with your own data, modify the `main()` function:
```python
# Replace synthetic data generation with:
ppm = np.load("path/to/your/ppm.npy")
spectra_raw = np.load("path/to/your/spectra.npy")
```

### 2. `calibration_example.py` - Spectral Calibration

Demonstrates various calibration approaches:

**Example 1: TSP Calibration**
- Calibrates to TSP singlet at 0.0 ppm
- Standard reference for aqueous samples
- Shows both calibrated spectrum and shift value extraction

**Example 2: Alanine Calibration**
- Calibrates to alanine doublet at 1.48 ppm
- Useful for biological/metabolomic samples
- Demonstrates doublet pattern recognition

**Example 3: Glucose Calibration**
- Calibrates to glucose anomeric peak at 5.223 ppm
- Alternative to TSP for certain sample types
- Shows 'serum' alias usage

**Example 4: Custom Signal Calibration**
- Creates custom multiplet patterns (e.g., triplet)
- Shows how to define custom reference signals
- Demonstrates NMRSignal object usage

**Example 5: Batch Calibration**
- Calibrates multiple spectra simultaneously
- Tracks calibration shifts for all spectra
- Verifies calibration success

**Example 6: Parameter Effects**
- Tests max_shift constraint
- Tests correlation threshold
- Tests ROI width effects
- Demonstrates parameter tuning

**Output**:
- `output/calibration_report.txt` - Summary of calibration methods

**Usage**:
```bash
python examples/calibration_example.py
```

## Output Directory

Both examples create an `output/` directory (relative to where they're run) containing:

- **NumPy arrays** (`.npy` files) - Processed data that can be loaded with `np.load()`
- **Text reports** (`.txt` files) - Human-readable summaries
- **Plots** (if matplotlib is installed and visualization code is uncommented)

## Visualization

The examples save processed data but don't generate plots by default. To visualize results:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load processed data
ppm = np.load("output/ppm.npy")
spectra = np.load("output/spectra_processed.npy")

# Plot all spectra
plt.figure(figsize=(12, 6))
plt.plot(ppm, spectra.T, alpha=0.7)
plt.xlabel("Chemical Shift (ppm)")
plt.ylabel("Intensity")
plt.title("Processed NMR Spectra")
plt.gca().invert_xaxis()  # NMR convention: high to low ppm
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Customizing Examples

### Adjusting Processing Parameters

In `basic_processing.py`, modify parameters in the `process_spectra()` function:

```python
# Baseline correction
spectra_bc = baseline_correction(
    spectra,
    lam=1e5,      # Increase for smoother baseline
    p=0.001,      # Decrease for more aggressive baseline removal
    niter=10      # Increase for better convergence
)

# Calibration
spectra_cal = calibrate_spectra(
    ppm, spectra_bc,
    ref="tsp",           # Try: "alanine", "glucose", "serum"
    max_shift=0.15,      # Adjust based on expected miscalibration
    threshold=0.2,       # Lower for noisier data
)

# Normalization
spectra_norm = normalize(
    spectra_cal,
    method="pqn",        # Try: "max", "sum", or custom function
    ref="median"         # Or: "mean", or array
)
```

### Using Real Data

For real NMR data, you typically need:

1. **Chemical shift scale** (`ppm`): 1D array of ppm values
2. **Spectra matrix** (`spectra`): 2D array (n_spectra × n_points)

Example data loading:

```python
# From Bruker format (requires nmrglue)
import nmrglue as ng
dic, data = ng.bruker.read("path/to/experiment")
ppm = ng.bruker.create_ppm_scale(dic)
spectra = data  # Adjust dimensions as needed

# From CSV
import pandas as pd
df = pd.read_csv("spectra.csv")
ppm = df["ppm"].values
spectra = df.iloc[:, 1:].T.values  # Transpose if needed

# From NumPy
ppm = np.load("ppm.npy")
spectra = np.load("spectra.npy")
```

## Common Issues

### 1. Import errors
**Problem**: `ImportError: cannot import name 'baseline_correction'`

**Solution**: Install the package in editable mode:
```bash
pip install -e .
```

### 2. Low correlation warnings
**Problem**: `Cross-correlation diverged` warnings during alignment

**Solution**: This typically happens with very noisy or flat spectra. Try:
- Increase noise threshold
- Apply more aggressive baseline correction
- Use smaller spectral region for alignment

### 3. NaN values after normalization
**Problem**: `RuntimeWarning: invalid value encountered in divide`

**Solution**: This happens when the reference spectrum has zero values. Try:
- Use a different normalization region (avoid baseline regions)
- Apply baseline correction before normalization
- Use a different normalization method

## Next Steps

After running these examples:

1. **Explore the test suite** (`tests/`) for more usage patterns
2. **Read the documentation** (when Phase 2 is complete)
3. **Try with your own data** - modify examples for your specific needs
4. **Experiment with parameters** - tune settings for your data characteristics

## Further Reading

- **PQN Normalization**: Dieterle et al., Anal. Chem. 2006, doi:10.1021/ac051632c
- **Baseline Correction (ALS)**: Eilers & Boelens, Baseline Correction with Asymmetric Least Squares Smoothing (2005)
- **Cross-correlation Alignment**: Standard signal processing technique

## Support

For issues or questions:
- Open an issue on GitHub
- Check the test files for additional usage examples
- See the R package documentation for algorithm details

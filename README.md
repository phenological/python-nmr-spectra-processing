# nmr-spectra-processing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python package for processing Fourier-transformed 1D NMR spectra.

This is a Python migration of the R package `nmr.spectra.processing` (v0.1.6), focusing on core processing capabilities while adopting Python best practices and the scientific Python ecosystem (NumPy, SciPy).

## Status: Phase 1 Complete ✅

**Phase 1** (Complete) - Core processing functions:
- ✅ Spectral alignment (cross-correlation)
- ✅ Calibration (TSP, alanine, glucose references)
- ✅ Normalization (PQN, max, sum)
- ✅ Baseline correction (Asymmetric Least Squares)
- ✅ Phase correction (0th and 1st order)
- ✅ Utility functions (padding, shifting, noise estimation)
- ✅ Comprehensive test suite (80%+ coverage)
- ✅ Usage examples and documentation

**Phase 2** (Future): Visualization, interactive features, annotation tools, Sphinx documentation

## Features

### Core Processing
- **Spectral alignment** - Cross-correlation-based alignment to median, mean, or custom reference
- **Calibration** - Reference-based calibration to TSP, alanine, glucose, or custom signals
- **Normalization** - PQN (Probabilistic Quotient Normalization), max, sum, and custom methods
- **Baseline correction** - Asymmetric Least Squares (ALS) algorithm for polynomial baseline removal
- **Phase correction** - Zero-order and first-order phase corrections for complex NMR data

### Utilities
- **Region selection** - Extract spectral regions by chemical shift
- **Peak selection** - Find top/bottom spectra by intensity
- **Noise estimation** - Estimate noise level from blank regions
- **Series operations** - Shifting, padding, and alignment tools

### Reference Signals
- **Predefined references** - TSP (0.0 ppm), alanine (1.48 ppm), glucose (5.223 ppm)
- **Custom signals** - Define custom multiplet patterns for calibration
- **Multiple lineshapes** - Lorentzian, Gaussian, pseudo-Voigt

## Installation

### From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/your-repo/python-nmr-spectra-processing.git
cd python-nmr-spectra-processing

# Install in editable mode
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python 3.9+
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- rich >= 13.0.0

## Quick Start

### Basic Processing Pipeline

```python
import numpy as np
from nmr_spectra_processing import (
    baseline_correction,
    calibrate_spectra,
    normalize,
    align_spectra,
)

# Load your NMR data
ppm = np.load("ppm.npy")              # Chemical shift scale
spectra = np.load("spectra.npy")      # Spectra matrix (n_spectra × n_points)

# Complete processing pipeline
spectra_bc = baseline_correction(spectra, lam=1e5, p=0.001)
spectra_cal = calibrate_spectra(ppm, spectra_bc, ref="tsp", frequency=600)
spectra_norm = normalize(spectra_cal, method="pqn")
spectra_aligned = align_spectra(spectra_norm, ref="median")

# Save processed data
np.save("spectra_processed.npy", spectra_aligned)
```

### Calibration Examples

```python
from nmr_spectra_processing import calibrate_signal, get_reference_signal

# Calibrate to TSP (trimethylsilylpropanoic acid)
spectrum_cal = calibrate_signal(ppm, spectrum, signal="tsp")

# Calibrate to alanine doublet
spectrum_cal = calibrate_signal(
    ppm, spectrum,
    signal="alanine",
    frequency=600,    # Spectrometer frequency (MHz)
    cshift=1.48,      # Target chemical shift
    J=7.26            # Coupling constant (Hz)
)

# Calibrate to custom signal
from nmr_spectra_processing import NMRSignal, NMRPeak

custom_signal = NMRSignal(
    name="Custom",
    peaks=[
        NMRPeak(cshift=3.65, intensity=1.0),
        NMRPeak(cshift=3.70, intensity=2.0),
        NMRPeak(cshift=3.75, intensity=1.0)
    ]
)
spectrum_cal = calibrate_signal(ppm, spectrum, signal=custom_signal)
```

### More Examples

See the [`examples/`](examples/) directory for comprehensive demonstrations:
- [`basic_processing.py`](examples/basic_processing.py) - Complete processing pipeline
- [`calibration_example.py`](examples/calibration_example.py) - Advanced calibration techniques

Run examples:
```bash
python examples/basic_processing.py
python examples/calibration_example.py
```

## R to Python Function Mapping

| R Function | Python Function | Module |
|------------|-----------------|--------|
| `alignSeries()` | `align_spectra()` | `nmr_spectra_processing.core.alignment` |
| `calibrateSpectra()` | `calibrate_spectra()` | `nmr_spectra_processing.core.calibration` |
| `calibrateSignal()` | `calibrate_signal()` | `nmr_spectra_processing.core.calibration` |
| `normalize()` | `normalize()` | `nmr_spectra_processing.core.normalization` |
| `pqn()` | `pqn()` | `nmr_spectra_processing.core.normalization` |
| `baselineCorrection()` | `baseline_correction()` | `nmr_spectra_processing.core.baseline` |
| `phaseCorrection()` | `phase_correction()` | `nmr_spectra_processing.core.phase` |
| `shiftSeries()` | `shift_series()` | `nmr_spectra_processing.core.shifting` |
| `shiftSpectra()` | `shift_spectra()` | `nmr_spectra_processing.core.shifting` |
| `pad()` | `pad_series()` | `nmr_spectra_processing.core.padding` |
| `noiseLevel()` | `estimate_noise()` | `nmr_spectra_processing.core.noise` |
| `crop()` | `crop_region()` | `nmr_spectra_processing.utils.regions` |
| `getI()` | `get_indices()` | `nmr_spectra_processing.utils.regions` |
| `top()` | `get_top_spectra()` | `nmr_spectra_processing.utils.selection` |

## Dependencies

- **NumPy** (>=1.24.0) - Array operations
- **SciPy** (>=1.10.0) - Signal processing, interpolation, cross-correlation
- **rich** (>=13.0.0) - Terminal output and logging

## Testing

The package includes a comprehensive test suite with 80%+ coverage:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=nmr_spectra_processing --cov-report=html

# Run specific test file
pytest tests/test_calibration.py -v

# Run tests in parallel (faster)
pytest -n auto
```

### Test Statistics
- **Total tests**: 246 tests (100% passing)
- **Coverage**: 90% overall (exceeds 80% target)
- **Test modules**: 9 modules (alignment, calibration, normalization, baseline, phase, shifting, padding, noise, signals)
- **Test categories**: Unit tests, integration tests, edge cases, real-world scenarios

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check (optional)
mypy src/
```

## Data Format

### Input Requirements

**Chemical Shift Scale (ppm)**
- 1D NumPy array
- Typically decreasing order (e.g., 10.0 to 0.0 ppm)
- Example: `ppm = np.linspace(10, 0, 40000)`

**Spectra Matrix**
- 2D NumPy array: `(n_spectra, n_points)`
- Each row is one spectrum
- Number of points must match ppm array length
- Example: `spectra.shape = (100, 40000)` for 100 spectra

### Loading Data

```python
# From NumPy files
ppm = np.load("ppm.npy")
spectra = np.load("spectra.npy")

# From CSV
import pandas as pd
df = pd.read_csv("data.csv")
ppm = df.iloc[0].values       # First row = ppm scale
spectra = df.iloc[1:].values  # Remaining rows = spectra

# From Bruker format (requires nmrglue)
import nmrglue as ng
dic, data = ng.bruker.read_pdata("path/to/pdata/1")
ppm = ng.bruker.create_ppm_scale(dic)
spectra = data  # May need reshaping
```

## Architecture

```
nmr_spectra_processing/
├── core/                  # Core processing algorithms
│   ├── alignment.py       # Cross-correlation alignment
│   ├── calibration.py     # Reference signal calibration
│   ├── normalization.py   # PQN and other normalizations
│   ├── baseline.py        # ALS baseline correction
│   ├── phase.py           # Phase corrections
│   ├── shifting.py        # Series shifting (discrete/continuous)
│   ├── padding.py         # Boundary padding methods
│   └── noise.py           # Noise level estimation
├── utils/                 # Utility functions
│   ├── regions.py         # Region cropping and indexing
│   ├── selection.py       # Peak/spectrum selection
│   └── logger.py          # Colored terminal output
├── reference/             # Reference signals
│   └── signals.py         # Predefined and custom signals
└── __init__.py            # Public API exports
```

## Best Practices

1. **Processing Order**: Typically: baseline → calibration → normalization → alignment
2. **Baseline Correction**: Apply before calibration for best results
3. **Calibration**: Choose reference signal appropriate for your sample type
4. **Normalization**: PQN is recommended for biological samples with dilution effects
5. **Alignment**: Apply after normalization to align peak positions
6. **Parameter Tuning**: Start with defaults, adjust based on data quality
7. **Validation**: Always inspect spectra visually after processing

## Algorithms

### Probabilistic Quotient Normalization (PQN)
Dieterle et al., "Probabilistic quotient normalization as robust method to account for dilution of complex biological mixtures", *Analytical Chemistry* (2006), doi:10.1021/ac051632c

### Asymmetric Least Squares (ALS) Baseline Correction
Eilers & Boelens, "Baseline Correction with Asymmetric Least Squares Smoothing" (2005)

### Cross-Correlation Alignment
Standard signal processing technique using SciPy's `correlate` function for spectral alignment

## Compatibility

This Python package maintains functional compatibility with the R `nmr.spectra.processing` package (v0.1.6):

- **Same algorithms** - Cross-correlation, PQN, ALS baseline correction
- **Same data conventions** - Spectra in rows, ppm in columns
- **Same reference signals** - TSP, alanine, glucose calibration standards
- **Numerical validation** - Outputs validated against R package (tolerance 1e-5)
- **API similarity** - Function names and parameters follow R conventions (snake_case vs camelCase)

## License

MIT License - See LICENSE file for details

## Authors

- Andrés Bernal (AndresFernando.BernalEscobar@murdoch.edu.au)
- Julien Wist (julien.wist@murdoch.edu.au)

## Troubleshooting

### Common Issues

**Problem**: `ImportError: cannot import name 'baseline_correction'`
- **Solution**: Install package in editable mode: `pip install -e .`

**Problem**: Cross-correlation diverged warnings during alignment
- **Solution**: This happens with flat/noisy spectra. Increase threshold or use more aggressive baseline correction.

**Problem**: NaN values after normalization
- **Solution**: Ensure baseline correction is applied first. Use `of_region` parameter to exclude baseline regions from normalization.

**Problem**: Calibration not working (peaks don't move)
- **Solution**: Check that reference signal is in the ROI. Widen ROI or increase max_shift parameter.

**Problem**: Memory errors with large datasets
- **Solution**: Process spectra in batches or use memory-mapped arrays (`np.memmap`).

### Getting Help

- Check the [`examples/`](examples/) directory for usage patterns
- Review test files in [`tests/`](tests/) for more examples
- Open an issue on GitHub for bugs or feature requests

## Citation

If you use this package in your research, please cite:

```bibtex
@software{nmr_spectra_processing_python,
  title = {nmr-spectra-processing: Python package for NMR spectra processing},
  author = {Bernal, Andrés and Wist, Julien},
  year = {2024},
  version = {0.1.0},
  url = {https://github.com/your-repo/python-nmr-spectra-processing},
  note = {Python migration of nmr.spectra.processing R package v0.1.6}
}
```

**Key References**:
- Dieterle et al. (2006), PQN normalization, *Anal. Chem.*, doi:10.1021/ac051632c
- Eilers & Boelens (2005), ALS baseline correction

## Contributing

Contributions are welcome! This package maintains compatibility with the R `nmr.spectra.processing` package while following Python best practices.

### Guidelines

1. **Compatibility**: Maintain functional compatibility with R package
2. **Testing**: Add tests for new features (target 80%+ coverage)
3. **Documentation**: Include docstrings and update README
4. **Code Style**: Follow Black formatting and PEP 8
5. **Type Hints**: Add type hints for public functions

### Development Workflow

```bash
# Fork and clone the repository
git clone https://github.com/your-username/python-nmr-spectra-processing.git
cd python-nmr-spectra-processing

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Make changes and run tests
pytest

# Format code
black src/ tests/

# Submit pull request
```

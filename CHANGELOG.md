# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-02-17

### Added - Phase 1 Complete

#### Core Processing Functions
- **Spectral Alignment** (`align_spectra`)
  - Cross-correlation based alignment
  - Multiple reference types: median, mean, index, array, callable
  - Configurable threshold and padding methods
  - Returns aligned spectra or shift values

- **Calibration** (`calibrate_signal`, `calibrate_spectra`)
  - Predefined references: TSP (0.0 ppm), alanine (1.48 ppm), glucose (5.223 ppm)
  - Custom signal support via NMRSignal objects
  - ROI-based alignment for accurate calibration
  - Max shift constraint for safety
  - Batch calibration with shift tracking

- **Normalization** (`normalize`, `pqn`)
  - Probabilistic Quotient Normalization (PQN)
  - Max, sum, and custom normalization methods
  - Region-specific normalization support
  - Returns normalized spectra or dilution factors

- **Baseline Correction** (`baseline_correction`)
  - Asymmetric Least Squares (ALS) algorithm
  - Configurable smoothness (lambda) and asymmetry (p)
  - Handles single spectra and matrices
  - Efficient sparse matrix implementation

- **Phase Correction** (`phase_correction`)
  - Zero-order (phi0) phase correction
  - First-order (phi1) phase correction
  - Reverse mode support
  - Incremental rotation for efficiency

- **Series Operations** (`shift_series`, `shift_spectra`)
  - Discrete shifting by integer points
  - Continuous shifting with interpolation
  - Hz-to-ppm conversion
  - Multiple padding methods

- **Padding** (`pad_series`)
  - Three padding methods: zeroes, circular, sampling
  - Configurable padding side (left, right, both)
  - Custom sampling regions

- **Noise Estimation** (`estimate_noise`)
  - Quantile-based noise estimation
  - Configurable blank regions
  - Batch processing support

#### Utility Functions
- **Region Operations** (`crop_region`, `get_indices`)
  - Boolean masking for spectral regions
  - Index lookup by chemical shift values
  - ROI extraction

- **Spectrum Selection** (`get_top_spectra`)
  - Select top/bottom spectra by intensity
  - Region-specific selection
  - Returns spectra or indices

- **Logging** (rich-based colored output)
  - Warning, error, info, success messages
  - Prefix support for module identification
  - Replaces R's crayon package

#### Reference Signals
- **Signal Classes** (`NMRPeak`, `NMRSignal`)
  - Dataclass-based signal representation
  - Peak properties: chemical shift, coupling, multiplicity, intensity
  - Spectrum generation with multiple lineshapes

- **Predefined References** (`get_reference_signal`)
  - TSP: singlet at 0.0 ppm
  - Alanine: doublet at 1.48 ppm (J=7.26 Hz)
  - Glucose: doublet at 5.223 ppm (J=3.63 Hz)
  - Serum: alias for glucose

- **Custom Signals** (`create_custom_signal`)
  - Define arbitrary multiplet patterns
  - Custom peak positions and intensities

- **Lineshapes**
  - Lorentzian (default)
  - Gaussian
  - Pseudo-Voigt

#### Testing
- **246 comprehensive tests** (100% passing)
- **90% code coverage** (exceeds 80% target)
- **Test categories**:
  - Unit tests for individual functions
  - Integration tests for workflows
  - Edge case handling
  - Real-world scenarios with avo3 dataset
- **Fixtures**: avo3 NMR dataset, synthetic data generators

#### Documentation
- **README.md**: Complete usage guide
- **examples/**: Two comprehensive example scripts
  - `basic_processing.py`: Complete processing pipeline
  - `calibration_example.py`: Advanced calibration techniques
- **examples/README.md**: Detailed example documentation
- **API documentation**: Docstrings for all public functions
- **Migration guide**: R to Python function mapping

#### Package Structure
- Modern Python packaging with `pyproject.toml`
- Editable installation support
- Development dependencies included
- Type hints throughout codebase
- Black code formatting
- Ruff linting configuration

### Technical Details

#### Algorithms Implemented
- Cross-correlation alignment (scipy.signal.correlate)
- Probabilistic Quotient Normalization (Dieterle et al., 2006)
- Asymmetric Least Squares baseline correction (Eilers & Boelens, 2005)
- Cubic spline interpolation for continuous shifting
- Incremental rotation for efficient phase correction

#### Dependencies
- NumPy >= 1.24.0 (array operations)
- SciPy >= 1.10.0 (signal processing, interpolation)
- rich >= 13.0.0 (terminal output)

#### Compatibility
- Python 3.9+
- Functional compatibility with R nmr.spectra.processing v0.1.6
- Same algorithms and data conventions
- Numerical validation (tolerance 1e-5)

### Migration Notes from R Package

#### Naming Conventions
- R camelCase → Python snake_case
- `alignSeries()` → `align_spectra()`
- `calibrateSpectra()` → `calibrate_spectra()`
- `baselineCorrection()` → `baseline_correction()`
- `phaseCorrection()` → `phase_correction()`

#### Data Conventions
- Maintained: Spectra in rows, measurements in columns
- NumPy arrays instead of R matrices/data.frames
- Boolean masks instead of integer indices

#### Functional Changes
- Type coercion with warnings (not silent)
- Explicit error messages
- Colored terminal output via rich (replaces crayon)
- Optional type hints for better IDE support

#### Not Included in Phase 1
- Visualization functions (smatplot, sannotate)
- Interactive features (nmrium)
- Peak deconvolution (unidec)
- Plotting and annotation tools

These features are deferred to Phase 2 to focus on core processing capabilities.

---

## [Unreleased]

### Planned for Phase 2
- Visualization module (matplotlib-based)
- Fast plotting with automatic binning
- Spectral annotation tools
- Interactive visualization (plotly/dash)
- Sphinx documentation
- PyPI publication
- Additional test data and benchmarks

---

## References

- Dieterle, F., et al. (2006). "Probabilistic quotient normalization as robust method to account for dilution of complex biological mixtures." *Analytical Chemistry*, 78(13), 4281-4290. doi:10.1021/ac051632c

- Eilers, P. H., & Boelens, H. F. (2005). "Baseline correction with asymmetric least squares smoothing." Unpublished manuscript.

- Original R package: `nmr.spectra.processing` v0.1.6
  - Authors: Andrés Bernal, Julien Wist
  - Repository: https://github.com/phenological/nmr-spectra-processing

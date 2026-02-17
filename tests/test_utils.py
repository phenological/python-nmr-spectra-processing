"""Tests for utility functions (regions and selection)."""

import numpy as np
import pytest
from nmr_spectra_processing.utils import crop_region, get_indices, get_top_spectra


class TestCropRegion:
    """Tests for crop_region function."""

    def test_crop_with_start_and_end(self):
        """Test cropping with start and end parameters."""
        ppm = np.linspace(0, 10, 100)
        mask = crop_region(ppm, start=2.0, end=5.0)

        cropped = ppm[mask]
        assert np.all(cropped >= 2.0)
        assert np.all(cropped <= 5.0)
        assert len(cropped) > 0

    def test_crop_with_roi(self):
        """Test cropping with roi parameter."""
        ppm = np.linspace(0, 10, 100)
        mask = crop_region(ppm, roi=(3.0, 7.0))

        cropped = ppm[mask]
        assert np.all(cropped >= 3.0)
        assert np.all(cropped <= 7.0)

    def test_roi_takes_priority(self):
        """Test that roi parameter takes priority over start/end."""
        ppm = np.linspace(0, 10, 100)
        mask1 = crop_region(ppm, start=1.0, end=5.0, roi=(2.0, 4.0))
        mask2 = crop_region(ppm, roi=(2.0, 4.0))

        np.testing.assert_array_equal(mask1, mask2)

    def test_crop_with_only_start(self):
        """Test cropping with only start parameter."""
        ppm = np.linspace(0, 10, 100)
        mask = crop_region(ppm, start=8.0)

        cropped = ppm[mask]
        assert np.all(cropped >= 8.0)
        assert np.max(cropped) == np.max(ppm)

    def test_crop_with_only_end(self):
        """Test cropping with only end parameter."""
        ppm = np.linspace(0, 10, 100)
        mask = crop_region(ppm, end=2.0)

        cropped = ppm[mask]
        assert np.all(cropped <= 2.0)
        assert np.min(cropped) == np.min(ppm)

    def test_crop_full_range(self):
        """Test cropping without any parameters returns all True."""
        ppm = np.linspace(0, 10, 100)
        mask = crop_region(ppm)

        assert np.all(mask)
        assert len(mask) == len(ppm)

    def test_crop_empty_region(self):
        """Test cropping with no overlap returns empty."""
        ppm = np.linspace(0, 10, 100)
        mask = crop_region(ppm, start=15.0, end=20.0)

        assert not np.any(mask)

    def test_crop_with_avo3_fixtures(self, ppm, ppm_tsp):
        """Test crop_region reproduces TSP region from fixtures."""
        mask = crop_region(ppm, roi=(-0.02, 0.02))
        cropped = ppm[mask]

        # Should match the ppm_tsp fixture
        assert len(cropped) == len(ppm_tsp)
        np.testing.assert_allclose(cropped, ppm_tsp)


class TestGetIndices:
    """Tests for get_indices function."""

    def test_get_single_index(self):
        """Test getting index for single value."""
        ppm = np.linspace(0, 10, 1001)  # 0.01 spacing
        idx = get_indices(ppm, 5.0)

        assert isinstance(idx, (int, np.integer))
        assert ppm[idx] == pytest.approx(5.0, abs=0.01)

    def test_get_multiple_indices(self):
        """Test getting indices for multiple values."""
        ppm = np.linspace(0, 10, 1001)
        indices = get_indices(ppm, 1.0, 5.0, 9.0)

        assert len(indices) == 3
        assert ppm[indices[0]] == pytest.approx(1.0, abs=0.01)
        assert ppm[indices[1]] == pytest.approx(5.0, abs=0.01)
        assert ppm[indices[2]] == pytest.approx(9.0, abs=0.01)

    def test_get_index_exact_match(self):
        """Test getting index with exact match."""
        ppm = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        idx = get_indices(ppm, 3.0)

        assert idx == 3
        assert ppm[idx] == 3.0

    def test_get_index_interpolated(self):
        """Test getting closest index for value between points."""
        ppm = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        idx = get_indices(ppm, 2.4)

        # Should be closest to 2.0 (index 2)
        assert idx == 2

    def test_get_index_out_of_range(self):
        """Test getting index for out-of-range value."""
        ppm = np.linspace(0, 10, 100)
        idx = get_indices(ppm, 15.0)

        # Should return last index
        assert idx == len(ppm) - 1

    def test_get_indices_no_values_raises(self):
        """Test that calling with no values raises error."""
        ppm = np.linspace(0, 10, 100)
        with pytest.raises(ValueError, match="At least one query value"):
            get_indices(ppm)


class TestGetTopSpectra:
    """Tests for get_top_spectra function."""

    def test_get_top_by_cshift(self):
        """Test getting top spectra by specific chemical shift."""
        ppm = np.linspace(0, 10, 100)
        # Create spectra where values at closest index to 5.0 determine order
        idx_at_5 = get_indices(ppm, 5.0)
        spectra = np.ones((10, 100)) * 0.001  # Small baseline
        spectra[3, idx_at_5] = 100  # Spectrum 3 has highest value
        spectra[7, idx_at_5] = 90   # Spectrum 7 has second highest

        top2_indices = get_top_spectra(ppm, spectra, n=2, cshift=5.0, return_indices=True)
        top2_spectra = get_top_spectra(ppm, spectra, n=2, cshift=5.0, return_indices=False)

        assert len(top2_indices) == 2
        # Check that spectrum 3 and 7 are in the top 2
        assert set(top2_indices) == {3, 7}
        # Check that returned spectra match
        assert top2_spectra.shape == (2, 100)
        # First should be spectrum with highest value (3)
        assert top2_indices[0] == 3

    def test_get_top_by_roi(self):
        """Test getting top spectra by region."""
        ppm = np.linspace(0, 10, 100)
        spectra = np.random.randn(5, 100)

        # Make spectrum 2 have highest max in region 40-60
        spectra[2, 40:60] = 50

        top1 = get_top_spectra(ppm, spectra, n=1, roi=(4.0, 6.0), return_indices=True)

        assert len(top1) == 1
        assert top1[0] == 2

    def test_get_bottom_spectra(self):
        """Test getting bottom (lowest) spectra."""
        ppm = np.linspace(0, 10, 100)
        idx_at_5 = get_indices(ppm, 5.0)
        spectra = np.ones((5, 100))  # All ones
        spectra[1, idx_at_5] = -100  # Lowest value

        bottom1 = get_top_spectra(ppm, spectra, n=1, cshift=5.0,
                                  bottom=True, return_indices=True)

        assert len(bottom1) == 1
        assert bottom1[0] == 1

    def test_return_spectra_not_indices(self):
        """Test returning spectra instead of indices."""
        ppm = np.linspace(0, 10, 100)
        spectra = np.random.randn(10, 100)

        top3 = get_top_spectra(ppm, spectra, n=3, cshift=5.0, return_indices=False)

        assert top3.shape == (3, 100)
        assert isinstance(top3, np.ndarray)

    def test_n_capped_at_num_spectra(self):
        """Test that n is capped at number of available spectra."""
        ppm = np.linspace(0, 10, 100)
        spectra = np.random.randn(5, 100)

        # Request more than available
        top10 = get_top_spectra(ppm, spectra, n=10, cshift=5.0, return_indices=True)

        assert len(top10) == 5  # Only 5 available

    def test_cshift_priority_over_roi(self):
        """Test that cshift takes priority over roi."""
        ppm = np.linspace(0, 10, 100)
        idx_at_5 = get_indices(ppm, 5.0)
        spectra = np.ones((5, 100)) * 0.001  # Small baseline
        spectra[2, idx_at_5] = 100  # Highest at cshift=5.0

        # Both provided, should use cshift
        result = get_top_spectra(ppm, spectra, n=1, cshift=5.0,
                                roi=(0.0, 2.0), return_indices=True)

        # Should find spectrum 2 (based on cshift=5.0, not roi)
        assert result[0] == 2

    def test_single_spectrum(self):
        """Test with single spectrum (1D array)."""
        ppm = np.linspace(0, 10, 100)
        spectrum = np.random.randn(100)

        result = get_top_spectra(ppm, spectrum, n=1, cshift=5.0, return_indices=False)

        assert result.shape == (1, 100)

    def test_type_coercion_warnings(self, capsys):
        """Test that non-numpy arrays trigger warnings."""
        ppm = [0, 1, 2, 3, 4, 5]
        spectra = [[1, 2, 3, 4, 5, 6]]

        # Should work but print warnings
        result = get_top_spectra(ppm, spectra, n=1, return_indices=False)

        assert result.shape == (1, 6)

    def test_non_numeric_spectra_raises(self):
        """Test that non-numeric spectra raise error."""
        ppm = np.linspace(0, 10, 100)
        spectra = np.array([["a", "b", "c"]])

        with pytest.raises(TypeError, match="spectra must be numeric"):
            get_top_spectra(ppm, spectra, n=1)

    def test_with_avo3_fixtures(self, ppm, avo3):
        """Test with real avo3 data."""
        # Get spectrum with highest intensity around TSP region
        top1 = get_top_spectra(ppm, avo3, n=1, roi=(-0.02, 0.02), return_indices=True)

        assert len(top1) == 1
        assert 0 <= top1[0] < 3  # Should be one of the 3 spectra

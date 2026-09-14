"""
Canonical NMR lineshape definitions.

This module is the single home for peak-shape maths in the phenological Python
NMR stack. Other packages (fitting, dashboards) import shapes from here rather
than redefining them, so a Voigt means the same thing everywhere.

Two parameterisations of the same Voigt profile are provided:

* :func:`voigt_area` — normalised so its integral equals ``area``. This is the
  historical ``deconvolution.voigt`` (kept as an alias for backward
  compatibility).
* :func:`voigt_height` — scaled so its peak value equals ``height``. This is
  what least-squares fitters use.

:func:`area_from_height` / :func:`height_from_area` convert between the two.
"""

from collections.abc import Sequence

import numpy as np
from scipy.special import wofz

_SQRT2 = np.sqrt(2.0)
_SQRT2PI = np.sqrt(2.0 * np.pi)
_SQRT2LN2 = np.sqrt(2.0 * np.log(2.0))
_MIN_SIGMA = 1e-12


# ── Voigt profile ───────────────────────────────────────────────────────────

def _wofz_peak(sigma: float, gamma: float) -> float:
    """Re(w(i gamma / (sigma sqrt2))) — the unscaled Voigt value at its centre."""
    sigma = max(float(sigma), _MIN_SIGMA)
    return float(np.real(wofz(1j * gamma / (sigma * _SQRT2))))


def voigt_profile(x: np.ndarray, center: float, sigma: float,
                  gamma: float) -> np.ndarray:
    """Voigt profile normalised to unit peak height.

    Parameters
    ----------
    x      : frequency axis
    center : peak center
    sigma  : Gaussian width parameter (sigma = FWHM_G / (2 sqrt(2 ln 2)))
    gamma  : Lorentzian half-width at half-maximum (= FWHM_L / 2)
    """
    sigma = max(float(sigma), _MIN_SIGMA)
    z = ((x - center) + 1j * gamma) / (sigma * _SQRT2)
    return np.real(wofz(z)) / _wofz_peak(sigma, gamma)


def voigt_area(x: np.ndarray, area: float, center: float,
               sigma: float, gamma: float) -> np.ndarray:
    """Voigt profile normalised so its integral equals ``area``.

    Identical to the historical ``deconvolution.voigt``.

    Parameters
    ----------
    x      : frequency axis
    area   : peak area (integral)
    center : peak center
    sigma  : Gaussian width parameter (sigma = FWHM_G / (2 sqrt(2 ln 2)))
    gamma  : Lorentzian half-width at half-maximum (= FWHM_L / 2)
    """
    sigma = max(float(sigma), _MIN_SIGMA)
    z = ((x - center) + 1j * gamma) / (sigma * _SQRT2)
    return area * np.real(wofz(z)) / (sigma * _SQRT2PI)


def voigt(x: np.ndarray, amplitude: float, center: float,
          sigma: float, gamma: float) -> np.ndarray:
    """Backward-compatible alias of :func:`voigt_area`.

    Kept with its historical parameter name ``amplitude`` (= peak area) so
    existing callers of the old ``deconvolution.voigt`` keep working. New code
    should call :func:`voigt_area` or :func:`voigt_height` directly.
    """
    return voigt_area(x, amplitude, center, sigma, gamma)


def voigt_height(x: np.ndarray, height: float, center: float,
                 sigma: float, gamma: float) -> np.ndarray:
    """Voigt profile scaled so its peak equals ``height``.

    Parameters
    ----------
    x      : frequency axis
    height : peak height (value at the center)
    center : peak center
    sigma  : Gaussian width parameter
    gamma  : Lorentzian half-width at half-maximum
    """
    return height * voigt_profile(x, center, sigma, gamma)


def area_from_height(height: float, sigma: float, gamma: float) -> float:
    """Analytic integral of a height-parameterised Voigt.

    area = height * sigma * sqrt(2 pi) / Re(w(i gamma / (sigma sqrt2))).
    """
    sigma = max(float(sigma), _MIN_SIGMA)
    return float(height) * sigma * _SQRT2PI / _wofz_peak(sigma, gamma)


def height_from_area(area: float, sigma: float, gamma: float) -> float:
    """Inverse of :func:`area_from_height`."""
    sigma = max(float(sigma), _MIN_SIGMA)
    return float(area) * _wofz_peak(sigma, gamma) / (sigma * _SQRT2PI)


# ── Composite shapes ──────────────────────────────────────────────────────────

def multiplet(x: np.ndarray, height: float, center: float, sigma: float,
              gamma: float, offsets: Sequence[float],
              heights_rel: Sequence[float]) -> np.ndarray:
    """Sum of Voigt lines sharing one lineshape at annotation-fixed offsets.

    Parameters
    ----------
    x           : frequency axis
    height      : height of the tallest line (heights_rel max should be 1)
    center      : multiplet center; each line sits at ``center + offset``
    sigma, gamma : shared Voigt widths for every line
    offsets     : per-line position offsets from ``center``
    heights_rel : per-line relative heights (e.g. [1, 2, 1] for a triplet)
    """
    y = np.zeros_like(np.asarray(x, dtype=float))
    for offset, h_rel in zip(offsets, heights_rel):
        y = y + voigt_height(x, height * h_rel, center + offset, sigma, gamma)
    return y


def voigt_with_satellites(x: np.ndarray, height: float, center: float,
                          sigma: float, gamma: float, f_sat: float,
                          delta_sat: float, sigma_sat: float,
                          gamma_sat: float) -> np.ndarray:
    """Main Voigt plus two symmetric satellites (the TMS-style model).

    Satellites are added only when ``f_sat > 0``; each has height
    ``f_sat * height`` and sits at ``center +/- delta_sat``.
    """
    y = voigt_height(x, height, center, sigma, gamma)
    if f_sat and f_sat > 0:
        h_sat = f_sat * height
        y = y + voigt_height(x, h_sat, center - delta_sat, sigma_sat, gamma_sat)
        y = y + voigt_height(x, h_sat, center + delta_sat, sigma_sat, gamma_sat)
    return y


# ── Width helpers ───────────────────────────────────────────────────────────

def pseudo_voigt_fwhm(sigma: float, gamma: float) -> float:
    """Full width at half maximum of a Voigt via the pseudo-Voigt approximation.

    Uses the Olivero-Longbothum combination of the Gaussian FWHM
    (2 sqrt(2 ln 2) sigma) and the Lorentzian FWHM (2 gamma).
    """
    f_l = 2.0 * float(gamma)
    f_g = 2.0 * _SQRT2LN2 * float(sigma)
    return 0.5346 * f_l + np.sqrt(0.2166 * f_l * f_l + f_g * f_g)


def measure_fwhm(x: np.ndarray, y: np.ndarray, x_center: float) -> float:
    """Measure the full-width at half-maximum of a peak by linear interpolation.

    Parameters
    ----------
    x        : frequency axis (uniform spacing)
    y        : spectrum intensities
    x_center : approximate peak center (used to find the peak maximum)

    Returns
    -------
    FWHM in the same units as x, or nan if the half-maximum crossings could
    not be found (peak too close to the edge or too broad).
    """
    dx = x[1] - x[0]
    idx = int(np.argmin(np.abs(x - x_center)))
    half = y[idx] / 2.0
    left = right = None

    for i in range(idx, 0, -1):
        if y[i - 1] <= half:
            frac = (y[i] - half) / max(y[i] - y[i - 1], 1e-30)
            left = (i - frac) * dx + x[0]
            break
    for i in range(idx, len(x) - 1):
        if y[i + 1] <= half:
            frac = (y[i] - half) / max(y[i] - y[i + 1], 1e-30)
            right = (i + frac) * dx + x[0]
            break

    return (right - left) if (left is not None and right is not None) else float("nan")

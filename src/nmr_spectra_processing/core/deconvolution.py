"""
Reference deconvolution and lineshape manipulation for NMR spectra.

Theory
------
The observed spectrum S_obs = S_true * R_obs  (* = convolution).
In the pseudo-time (FID) domain this becomes multiplication:

    Ŝ_obs(t) = Ŝ_true(t) · R̂_obs(t)

Reference deconvolution reshapes the lineshape by multiplying with the ratio:

    Ŝ_out(t) = Ŝ_obs(t) · R̂_target(t) / R̂_obs(t)

For a pure Lorentzian, R̂(t) = exp(-π·LW·t), so the ratio is:

    R̂_target / R̂_obs = exp(+π·(LW_obs - LW_target)·t)

Numerically the FID from FFT(spectrum) extends to t_max = 1/(2Δx) where only
noise exists. That noise gets amplified by exp(+π·ΔLW·t_max) → catastrophic.

Fix: Wiener-type weight that smoothly suppresses the noise-dominated FID tail.

    w(t) = (SNR · exp(-π·LW_obs·t))² / (1 + (SNR · exp(-π·LW_obs·t))²)

  ≈ 1 while signal >> noise,  → 0 in the noise tail.

Net filter:  h(t) = exp(+π·(LW_obs - LW_target - lb)·t) · w(t)
  → exp(-π·LW_target·t) in signal region (recovers target lineshape)
  → 0 in noise region   (no amplification)

lb (same units as x) adds a small extra Lorentzian broadening for robustness.
"""

from typing import List, Sequence
import numpy as np
from scipy.special import wofz

_SQRT2LN2 = np.sqrt(2.0 * np.log(2.0))


# ── Voigt profile ─────────────────────────────────────────────────────────────

def voigt(x: np.ndarray, amplitude: float, center: float,
          sigma: float, gamma: float) -> np.ndarray:
    """
    Area-normalised Voigt profile.

    Parameters
    ----------
    x         : frequency axis
    amplitude : peak area (integral)
    center    : peak center
    sigma     : Gaussian width parameter (sigma = FWHM_G / (2√(2 ln 2)))
    gamma     : Lorentzian half-width at half-maximum (= FWHM_L / 2)

    Returns
    -------
    Voigt profile evaluated at x
    """
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def nmr_to_voigt(cs: float, area: float, fwhm_L: float,
                 fwhm_G: float) -> tuple:
    """
    Convert NMR peak parameters to Voigt profile parameters.

    Parameters
    ----------
    cs     : chemical shift (center)
    area   : peak area
    fwhm_L : Lorentzian full-width at half-maximum
    fwhm_G : Gaussian full-width at half-maximum

    Returns
    -------
    (amplitude, center, sigma, gamma) for use with voigt()
    """
    gamma = fwhm_L / 2.0
    sigma = max(fwhm_G / (2.0 * _SQRT2LN2), 1e-9)
    return area, cs, sigma, gamma


def spectrum_from_peaks(x: np.ndarray,
                        peaks: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Build a spectrum from a list of Voigt peaks.

    Parameters
    ----------
    x     : frequency axis (uniform spacing)
    peaks : list of [center, area, fwhm_L, fwhm_G] entries

    Returns
    -------
    Summed Voigt spectrum evaluated at x
    """
    y = np.zeros_like(x, dtype=float)
    for p in peaks:
        y += voigt(x, *nmr_to_voigt(*p))
    return y


# ── Reference deconvolution ───────────────────────────────────────────────────

def ref_deconv(x: np.ndarray, spectrum: np.ndarray,
               fwhm_L_obs: float, fwhm_L_target: float,
               lb: float = 0.0) -> np.ndarray:
    """
    Sharpen (or broaden) a spectrum by reference deconvolution.

    The function reshapes the Lorentzian component of the lineshape from
    ``fwhm_L_obs`` to ``fwhm_L_target + lb`` using Fourier-domain division
    stabilised by a Wiener filter.  The Wiener weight suppresses the
    noise-dominated tail of the pseudo-FID, preventing noise amplification.

    An edge cosine taper is applied before the FFT to reduce spectral leakage
    from the non-zero boundaries of Lorentzian tails.

    Parameters
    ----------
    x             : uniform frequency axis (ppm, Hz, …)
    spectrum      : observed spectrum S(x)
    fwhm_L_obs    : Lorentzian FWHM of the current lineshape (same units as x)
    fwhm_L_target : Lorentzian FWHM of the desired lineshape.
                    Use < fwhm_L_obs for sharpening, > fwhm_L_obs for
                    broadening. Set to 0.0 for maximum sharpening (the lb
                    parameter then controls the effective output FWHM).
    lb            : Extra Lorentzian broadening added after deconvolution
                    (same units as x). Acts as regularisation; the effective
                    output FWHM_L ≈ fwhm_L_target + lb. Default 0.0.

    Returns
    -------
    s_out : spectrum with reshaped Lorentzian lineshape, same length as x

    Notes
    -----
    * For sharpening, a non-zero ``lb`` is recommended for robustness against
      noise (typical value: 0.1–0.3 × fwhm_L_obs).
    * The spectrum must have uniform x-spacing.
    * Zero-padding (4×) is used internally to avoid circular-convolution
      wrap-around artefacts.

    Examples
    --------
    >>> import numpy as np
    >>> from nmr_spectra_processing import ref_deconv, spectrum_from_peaks
    >>> x = np.linspace(-10, 10, 4096)
    >>> peaks = [[0.0, 1.0, 0.8, 0.1]]   # [center, area, fwhm_L, fwhm_G]
    >>> y_obs = spectrum_from_peaks(x, peaks)
    >>> y_sharp = ref_deconv(x, y_obs, fwhm_L_obs=0.8, fwhm_L_target=0.2)
    """
    dx = x[1] - x[0]
    n = len(x)

    # Edge taper: Lorentzian tails are non-zero at x = ±x_max, causing
    # spectral leakage that gets amplified by the deconvolution filter.
    # Apply a cosine taper over the outer 15% on each side to suppress this.
    taper_n = max(1, int(n * 0.15))
    taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper_n) / taper_n))
    win = np.ones(n)
    win[:taper_n] = taper
    win[-taper_n:] = taper[::-1]
    spec_tapered = spectrum * win

    # Zero-pad to avoid remaining circular-convolution wrap-around.
    n_pad = n * 4
    t = np.fft.rfftfreq(n_pad, d=dx)

    S = np.fft.rfft(spec_tapered, n=n_pad) * dx

    # SNR estimation from the pseudo-FID
    area = np.abs(S[0])
    fft_noise = np.std(np.abs(S[n_pad // 4:]))
    # Floor at 1e-6 × area: prevents snr → ∞ for noiseless data, while still
    # allowing snr ~ 1e6 (Wiener weight decays to 0 at t ≳ ln(1e6)/(π·LW_obs))
    snr = area / max(fft_noise, 1e-6 * area)

    # Wiener weight
    fid = np.exp(-np.pi * fwhm_L_obs * t)
    snr_t = snr * fid
    weight = snr_t**2 / (1.0 + snr_t**2)

    # Deconvolution ratio + lb apodisation
    ratio = np.exp(np.pi * (fwhm_L_obs - fwhm_L_target - lb) * t)

    s_out = np.fft.irfft(S * ratio * weight, n=n_pad) / dx
    return s_out[:n]


# ── Broadening ────────────────────────────────────────────────────────────────

def broaden(x: np.ndarray, spectrum: np.ndarray,
            extra_fwhm_L: float, extra_fwhm_G: float = 0.0) -> np.ndarray:
    """
    Broaden a spectrum by convolving with an extra Voigt kernel.

    No deconvolution is involved — always numerically stable.  Use this
    when you need to add broadening without the stability constraints of
    reference deconvolution.

    Parameters
    ----------
    x           : uniform frequency axis (ppm, Hz, …)
    spectrum    : input spectrum S(x)
    extra_fwhm_L : Lorentzian FWHM of the broadening kernel (same units as x)
    extra_fwhm_G : Gaussian FWHM of the broadening kernel (default 0.0)

    Returns
    -------
    Broadened spectrum, same length as x

    Examples
    --------
    >>> import numpy as np
    >>> from nmr_spectra_processing import broaden, spectrum_from_peaks
    >>> x = np.linspace(-10, 10, 4096)
    >>> peaks = [[0.0, 1.0, 0.2, 0.1]]
    >>> y = spectrum_from_peaks(x, peaks)
    >>> y_broad = broaden(x, y, extra_fwhm_L=0.5)
    """
    dx = x[1] - x[0]
    n = len(x)
    # Kernel centered at x=0; ifftshift moves it to index 0 for correct DFT
    kernel = voigt(x, *nmr_to_voigt(0.0, 1.0, extra_fwhm_L, extra_fwhm_G))
    K = np.fft.rfft(np.fft.ifftshift(kernel)) * dx
    S = np.fft.rfft(spectrum) * dx
    return np.fft.irfft(S * K, n=n) / dx


# ── FWHM measurement ──────────────────────────────────────────────────────────

def measure_fwhm(x: np.ndarray, y: np.ndarray, x_center: float) -> float:
    """
    Measure the full-width at half-maximum of a peak by linear interpolation.

    Parameters
    ----------
    x        : frequency axis
    y        : spectrum intensities
    x_center : approximate peak center (used to find the peak maximum)

    Returns
    -------
    FWHM in the same units as x, or nan if the half-maximum crossings
    could not be found (peak too close to the edge or too broad).

    Examples
    --------
    >>> import numpy as np
    >>> from nmr_spectra_processing import measure_fwhm, spectrum_from_peaks
    >>> x = np.linspace(-5, 5, 2048)
    >>> peaks = [[0.0, 1.0, 0.5, 0.0]]
    >>> y = spectrum_from_peaks(x, peaks)
    >>> fwhm = measure_fwhm(x, y, x_center=0.0)
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

    return (right - left) if (left is not None and right is not None) else float('nan')

"""
Phase-only reference deconvolution and cross-region phase transfer.

Where :func:`~nmr_spectra_processing.core.deconvolution.ref_deconv` and
:func:`~nmr_spectra_processing.core.deconvolution.ref_deconv_voigt` reshape the
*linewidth* of a spectrum, the functions here correct only the *phase* (the
asymmetric, dispersive component) without changing linewidth, satellite
amplitudes, or integral.

Method
------
The observed pseudo-FID phase is the sum of a carrier phase and an asymmetry
phase. Using the ideal Voigt pseudo-FID of the reference peak as a carrier
reference isolates just the asymmetry::

    phase_correction = exp(i * (angle(fid_ideal) - angle(fid_obs)))

Correcting only this angle difference removes the asymmetry without the
circular-shift artefact that correcting the full observed phase would cause.

Cross-region transfer
----------------------
:func:`compute_phase_angles` derives the correction from a reference peak on
its own axis; :func:`apply_phase_angles` applies it to a spectrum on a possibly
different axis by interpolating onto the target pseudo-time grid. This lets a
correction measured on, say, a TMS reference be applied to another region.
"""

import numpy as np
from scipy.interpolate import interp1d

from nmr_spectra_processing.core.deconvolution import _make_taper
from nmr_spectra_processing.core.lineshapes import voigt_area


def _wiener_weight(fid_mag: np.ndarray, n_pad: int) -> np.ndarray:
    """Wiener weight from a (normalised) pseudo-FID magnitude envelope.

    The SNR is estimated from the tail of ``fid_mag`` (where only noise
    remains), so the weight -> 1 while signal dominates and -> 0 in the noise
    tail, preventing amplification.
    """
    area = float(fid_mag[0])
    fft_noise = np.std(fid_mag[n_pad // 4:])
    snr = area / max(fft_noise, 1e-6 * area)
    snr_t = snr * fid_mag
    return snr_t**2 / (1.0 + snr_t**2)


def compute_phase_angles(x_ref: np.ndarray, peak_spectrum: np.ndarray,
                         center: float, sigma_target: float,
                         gamma_target: float) -> tuple:
    """Compute asymmetry phase angles from a reference peak.

    Parameters
    ----------
    x_ref         : uniform axis of the reference region
    peak_spectrum : windowed reference peak (excludes satellites/neighbours)
    center        : fitted reference peak center
    sigma_target  : Gaussian width of the ideal reference Voigt
    gamma_target  : Lorentzian half-width of the ideal reference Voigt

    Returns
    -------
    (phase_angles, t) where ``phase_angles = angle(fid_ideal) - angle(fid_obs)``
    and ``t`` is the corresponding pseudo-time axis. Apply the result with
    :func:`apply_phase_angles`, which can retarget it to a different axis.
    """
    dx = x_ref[1] - x_ref[0]
    n = len(x_ref)
    n_pad = n * 4
    win = _make_taper(n)

    t = np.fft.rfftfreq(n_pad, d=dx)
    s_ref = np.fft.rfft(peak_spectrum * win, n=n_pad) * dx

    # Empty / all-zero reference window: no phase information -> no-op correction.
    if np.abs(s_ref[0]) < 1e-30:
        return np.zeros_like(t), t

    fid_obs = s_ref / np.abs(s_ref[0])
    ideal = voigt_area(x_ref, 1.0, center, sigma_target, gamma_target)
    s_ideal = np.fft.rfft(ideal * win, n=n_pad) * dx
    fid_ideal = s_ideal / np.abs(s_ideal[0])

    return np.angle(fid_ideal) - np.angle(fid_obs), t


def apply_phase_angles(x: np.ndarray, spectrum: np.ndarray,
                       phase_angles: np.ndarray, t_src: np.ndarray,
                       lb: float = 0.0, snr_override: float = None) -> np.ndarray:
    """Apply a pre-computed phase correction to a spectrum on any axis.

    The ``phase_angles`` / ``t_src`` pair (from :func:`compute_phase_angles`)
    may have been derived from a different region; the angles are linearly
    interpolated onto this spectrum's pseudo-time grid before application.

    Parameters
    ----------
    x            : uniform axis of ``spectrum``
    spectrum     : spectrum to be phase-corrected
    phase_angles : phase-angle array from :func:`compute_phase_angles`
    t_src        : pseudo-time axis for ``phase_angles``
    lb           : Lorentzian regularisation broadening (same units as x)
    snr_override : override the auto Wiener SNR (set high, e.g. 1e4, to disable
                   Wiener broadening)

    Returns
    -------
    Phase-corrected spectrum, same length as x.
    """
    dx = x[1] - x[0]
    n = len(x)
    n_pad = n * 4
    win = _make_taper(n)

    S = np.fft.rfft(spectrum * win, n=n_pad) * dx
    t = np.fft.rfftfreq(n_pad, d=dx)

    interp = interp1d(t_src, phase_angles, kind="linear",
                      bounds_error=False, fill_value=0.0)
    phase_corr = np.exp(1j * interp(t))

    fid_mag = np.abs(S) / max(np.abs(S[0]), 1e-30)
    if snr_override is not None:
        snr_t = snr_override * fid_mag
        weight = snr_t**2 / (1.0 + snr_t**2)
    else:
        weight = _wiener_weight(fid_mag * np.abs(S[0]), n_pad)

    lb_apo = np.exp(-np.pi * lb * t)
    s_out = np.fft.irfft(S * phase_corr * weight * lb_apo, n=n_pad) / dx
    return s_out[:n]


def ref_deconv_from_peak(x: np.ndarray, spectrum: np.ndarray,
                         peak_spectrum: np.ndarray, center: float,
                         sigma_target: float, gamma_target: float,
                         lb: float = 0.0,
                         snr_override: float = None) -> np.ndarray:
    """Phase-only reference deconvolution using a measured reference peak.

    Corrects the asymmetric (dispersive) component of ``spectrum`` using the
    phase of the measured ``peak_spectrum`` relative to an ideal Voigt, without
    changing lineshape, linewidth, or satellite amplitudes. This is
    :func:`compute_phase_angles` + :func:`apply_phase_angles` on a single axis.

    Parameters
    ----------
    x             : uniform axis (reference peak and spectrum share it here)
    spectrum      : full observed spectrum to correct
    peak_spectrum : windowed reference peak (excludes satellites)
    center        : fitted reference peak center
    sigma_target  : Gaussian width of the ideal reference Voigt
    gamma_target  : Lorentzian half-width of the ideal reference Voigt
    lb            : Lorentzian regularisation broadening (same units as x)
    snr_override  : override the auto Wiener SNR (set high to disable
                    Wiener broadening)

    Returns
    -------
    Phase-corrected spectrum, same length as x.
    """
    dx = x[1] - x[0]
    n = len(x)
    n_pad = n * 4
    win = _make_taper(n)

    S = np.fft.rfft(spectrum * win, n=n_pad) * dx
    s_ref = np.fft.rfft(peak_spectrum * win, n=n_pad) * dx
    t = np.fft.rfftfreq(n_pad, d=dx)

    # Empty / all-zero reference window: nothing to correct against -> no-op.
    if np.abs(s_ref[0]) < 1e-30:
        return np.asarray(spectrum, dtype=float)

    fid_obs = s_ref / np.abs(s_ref[0])
    ideal = voigt_area(x, 1.0, center, sigma_target, gamma_target)
    s_ideal = np.fft.rfft(ideal * win, n=n_pad) * dx
    fid_ideal = s_ideal / np.abs(s_ideal[0])

    phase_correction = np.exp(1j * (np.angle(fid_ideal) - np.angle(fid_obs)))

    if snr_override is not None:
        snr_t = snr_override * np.abs(fid_obs)
        weight = snr_t**2 / (1.0 + snr_t**2)
    else:
        weight = _wiener_weight(np.abs(fid_obs), n_pad)

    lb_apo = np.exp(-np.pi * lb * t)
    s_out = np.fft.irfft(S * phase_correction * weight * lb_apo, n=n_pad) / dx
    return s_out[:n]

"""
Reference Deconvolution Example
================================
Demonstrates lineshape sharpening, broadening, and FWHM measurement using
the nmr_spectra_processing deconvolution module.

Scenarios covered
-----------------
1. Sharpening to match true linewidth  (noisy spectrum, realistic lb)
2. Sharpening beyond true linewidth    (noiseless spectrum, small lb)
3. Broadening via reference deconvolution
4. Broadening via direct convolution   (broaden — always stable)
5. FWHM report at the reference peak
"""

import numpy as np
import matplotlib.pyplot as plt
from nmr_spectra_processing import (
    broaden,
    measure_fwhm,
    ref_deconv,
    spectrum_from_peaks,
)

# ── Simulation parameters ─────────────────────────────────────────────────────
rng = np.random.default_rng(42)
x   = np.linspace(-15, 15, 8192)

base_L, base_G = 0.20, 0.08        # true Voigt parameters
extra_L        = 0.60              # uniform extra Lorentzian broadening
obs_L          = base_L + extra_L  # 0.80 — observed FWHM_L

# True narrow peaks: [center, area, fwhm_L, fwhm_G]
true_peaks = [
    [-8.0,  5.0, base_L, base_G],
    [-4.5, 10.0, base_L, base_G],
    [-1.5,  8.0, base_L, base_G],  # overlapping pair
    [ 0.0,  6.0, base_L, base_G],  #   ^
    [ 3.0, 12.0, base_L, base_G],  # ← reference peak
    [ 6.5,  7.0, base_L, base_G],  # overlapping triplet
    [ 7.5,  5.0, base_L, base_G],  #   ^
    [ 8.5,  9.0, base_L, base_G],  #   ^
]

obs_peaks   = [[cs, a, obs_L, g] for cs, a, _, g in true_peaks]
y_true      = spectrum_from_peaks(x, true_peaks)
y_obs_clean = spectrum_from_peaks(x, obs_peaks)
y_obs       = y_obs_clean + rng.normal(0, 0.003, len(x))

# ── 1. Sharpen to match true linewidth (from noisy observed) ──────────────────
lb_match = 0.25
y_match  = ref_deconv(x, y_obs, obs_L, base_L, lb=lb_match)

# ── 2. Sharpen beyond true linewidth (noiseless — high SNR regime) ────────────
lb_sharp = 0.08
y_sharp  = ref_deconv(x, y_obs_clean, obs_L, fwhm_L_target=0.0, lb=lb_sharp)

# ── 3. Broaden via reference deconvolution (fwhm_L_target > fwhm_L_obs) ───────
y_deconv_broad = ref_deconv(x, y_obs, obs_L, fwhm_L_target=obs_L + 0.70)

# ── 4. Broaden via direct convolution (always numerically stable) ─────────────
y_broad = broaden(x, y_obs, extra_fwhm_L=0.90)

# ── 5. FWHM report ────────────────────────────────────────────────────────────
ref_c = 3.0
print("FWHM at reference peak (x = 3.0):")
print(f"  {'Label':<34}  {'Measured':>8}  {'Expected':>10}")
for label, y, exp_fwhm in [
    ("True",                    y_true,         base_L),
    ("Observed (noisy)",        y_obs,          obs_L),
    (f"Match true (lb={lb_match})",   y_match,  base_L + lb_match),
    (f"Sharp (lb={lb_sharp}, noiseless)", y_sharp, lb_sharp),
    ("Deconv-broaden",          y_deconv_broad, obs_L + 0.70),
    ("Broaden (convolution)",   y_broad,        obs_L + 0.90),
]:
    mfwhm = measure_fwhm(x, y, ref_c)
    print(f"  {label:<34}  {mfwhm:>8.3f}  {exp_fwhm:>10.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(6, 1, figsize=(13, 18), sharex=True,
                         gridspec_kw={'hspace': 0.50})
fig.suptitle("Reference Deconvolution — Voigt peaks", fontsize=14,
             fontweight="bold")

specs = [
    (y_true,         f"True  (fwhm_L={base_L}, fwhm_G={base_G})",                        "C2"),
    (y_obs,          f"Observed  (fwhm_L={obs_L}, +{extra_L} broadening, noise added)",   "C0"),
    (y_match,        f"Match true  (ref_deconv, lb={lb_match}, eff. fwhm_L≈{base_L+lb_match:.2f})", "C1"),
    (y_sharp,        f"Sharper than true  (ref_deconv, lb={lb_sharp}, noiseless input)",  "C4"),
    (y_deconv_broad, f"Broadened  (ref_deconv, fwhm_L_target={obs_L+0.70:.2f})",          "C5"),
    (y_broad,        f"Broadened  (broaden/convolution, extra_fwhm_L=0.90)",              "C3"),
]

for ax, (y, label, color) in zip(axes, specs):
    ax.plot(x, y, color=color, lw=1.2)
    ax.axvline(ref_c, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("Intensity", fontsize=9)
    ax.set_title(label, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(x[0], x[-1])

axes[-1].set_xlabel("Chemical Shift (ppm)")
plt.tight_layout()
plt.savefig("deconvolution_example.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved → deconvolution_example.png")

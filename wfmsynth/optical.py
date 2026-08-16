"""
wfmsynth.optical — optical-link primitives (400G/800G optical PAM4 and friends).

Optical intensity-modulated / direct-detection links have their own realism dimension that
electrical models don't capture:

  * extinction ratio (ER) — the modulator never fully extinguishes, so the low level carries
    nonzero power; ER = P_high / P_low.
  * relative intensity noise (RIN) — laser noise whose std is PROPORTIONAL to optical power.
  * shot noise — photon-counting (Poisson) noise whose variance is proportional to power
    (this is also the logged electrical shot-noise term, #30).
  * chromatic dispersion — frequency-dependent group delay that SPREADS a pulse over fibre.
  * multipath interference (MPI) — a delayed, attenuated copy (double reflection) beats in.

Signals here are optical INTENSITY (power ≥ 0). `to_optical` maps a bipolar electrical
waveform into intensity with a finite ER. numpy/scipy only.
"""
from __future__ import annotations

import numpy as np


def to_optical(x, er_db=10.0, p_avg=1.0):
    """Map a bipolar electrical waveform to optical INTENSITY (power ≥ 0) with a finite
    extinction ratio ``er_db`` = 10·log10(P_high/P_low). Output is scaled to mean power
    ``p_avg``. Real modulators never fully extinguish, so the low level carries power."""
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    d = (x - lo) / ((hi - lo) + 1e-12)          # 0..1
    r = 10 ** (-er_db / 10.0)                    # P_low / P_high
    power = d * (1.0 - r) + r                    # in [r, 1]
    return power * (p_avg / (power.mean() + 1e-12))


def rin_noise(power, rin_db_per_hz=-140.0, bw_hz=1e10, rng=None):
    """Relative intensity noise: additive noise with std PROPORTIONAL to instantaneous power.
    ``rin_db_per_hz`` integrated over ``bw_hz`` sets the relative noise level."""
    rng = rng or np.random.default_rng()
    power = np.asarray(power, float)
    sigma = np.sqrt(10 ** (rin_db_per_hz / 10.0) * bw_hz) * power
    return power + rng.standard_normal(len(power)) * sigma


def shot_noise(power, photons_per_unit=1e4, rng=None):
    """Photon-counting (Poisson) shot noise: variance PROPORTIONAL to optical power. Higher
    ``photons_per_unit`` (more photons per unit power) lowers the relative noise. Gaussian
    approximation of the Poisson count."""
    rng = rng or np.random.default_rng()
    p = np.clip(np.asarray(power, float), 0.0, None)
    mean = photons_per_unit * p
    return (mean + np.sqrt(mean + 1e-12) * rng.standard_normal(len(p))) / photons_per_unit


def chromatic_dispersion(x, strength=20.0):
    """Chromatic dispersion as a quadratic all-pass phase in frequency — frequency-dependent
    group delay that SPREADS a pulse. ``strength`` scales the accumulated dispersion (∝ D·L)."""
    x = np.asarray(x, float)
    n = len(x)
    f = np.fft.rfftfreq(n)
    H = np.exp(-1j * strength * (f / (f.max() + 1e-12)) ** 2 * 2 * np.pi)
    return np.fft.irfft(np.fft.rfft(x) * H, n)


def laser_chirp(power, alpha=2.0, grid=None):
    """Transient laser chirp: the instantaneous optical FREQUENCY excursion during intensity
    transitions, Δf = (alpha/4π)·d(ln P)/dt. It occurs at the edges (zero on a steady level)
    and scales with the linewidth-enhancement factor ``alpha``; combined with dispersion it
    causes transition-dependent pulse distortion. Returns the frequency deviation (Hz if a
    grid is given, else per-sample)."""
    p = np.clip(np.asarray(power, float), 1e-9, None)
    dt = grid.dt if grid is not None else 1.0
    return (alpha / (4 * np.pi)) * np.gradient(np.log(p)) / dt


def mpi(power, delay_samples, reflectivity=0.05):
    """Multipath interference: add a delayed, attenuated copy (a double-reflection ghost)."""
    p = np.asarray(power, float)
    d = int(delay_samples)
    ghost = np.zeros_like(p)
    if d < len(p):
        ghost[d:] = p[:len(p) - d]
    return p + reflectivity * ghost

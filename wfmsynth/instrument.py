"""
wfmsynth.instrument — front-end / digitizer models. A real high-speed scope is a
TIME-INTERLEAVED ADC: M sub-ADC cores sample in round-robin, and the mismatch between
them is much of what makes a measured capture look measured. Perfectly white noise on a
perfectly uniform grid is exactly the "too clean" signature a model trained on synthetic
learns to rely on and then breaks on real data.

`interleave_adc` injects per-core **gain**, **offset** and **timing-skew** mismatch,
which produce the characteristic interleave spurs: offset mismatch -> tones at k*fs/M
(input-independent); gain/skew mismatch -> image tones mirrored about fs/(2M), i.e. at
k*fs/M +/- f_in. Set all mismatch to zero for an ideal ADC (no spurs). numpy/scipy only.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as _sig


def interleave_adc(x, m_cores=4, gain_mm=0.0, offset_mm=0.0, skew_mm=0.0,
                   rng=None, mismatch=None):
    """Pass x through an M-way time-interleaved ADC with per-core mismatch.

      gain_mm    per-core gain error, std (fractional, e.g. 0.01 = 1%)
      offset_mm  per-core offset, std as a fraction of the signal span
      skew_mm    per-core sampling-time skew, std in SAMPLES (sub-sample)

    Cores are assigned round-robin: sample i is taken by core i % m_cores. Draw the M
    per-core errors once (seed via rng), or pass `mismatch=(g, o, s)` explicitly for a
    reproducible fixed pattern. Returns the digitized array (same length)."""
    x = np.asarray(x, float)
    n = len(x)
    span = np.ptp(x) + 1e-9
    core = np.arange(n) % m_cores
    if mismatch is not None:
        g, o, s = (np.asarray(v, float) for v in mismatch)
    else:
        rng = rng or np.random.default_rng()
        g = rng.normal(0.0, gain_mm, m_cores)
        o = rng.normal(0.0, offset_mm, m_cores) * span
        s = rng.normal(0.0, skew_mm, m_cores)
    y = x.copy()
    if skew_mm > 0 or (mismatch is not None and np.any(s)):
        xi = np.arange(n)                          # sub-sample timing skew per core
        for c in range(m_cores):
            idx = np.where(core == c)[0]
            y[idx] = np.interp(idx + s[c], xi, x)
    y = y * (1.0 + g[core]) + o[core]              # per-core gain + offset
    return y


def shaped_noise_floor(n, rms=0.01, shape="pink", rng=None):
    """A frequency-shaped noise floor (real front ends are not flat). shape in
    {'white','pink','blue'} — pink ~ 1/sqrt(f), blue ~ sqrt(f). Returns length-n noise
    with the requested RMS. Add to a signal to model a coloured floor."""
    rng = rng or np.random.default_rng()
    w = rng.standard_normal(n)
    if shape != "white":
        W = np.fft.rfft(w)
        f = np.arange(W.shape[0], dtype=float); f[0] = 1.0
        W = W / np.sqrt(f) if shape == "pink" else W * np.sqrt(f)
        w = np.fft.irfft(W, n)
    return (rms / (w.std() + 1e-12)) * w


def clip_adc(x, full_scale=1.0):
    """Hard-clip to +/- full_scale (a real ADC saturates at full scale). Returns
    (clipped_signal, clipped_mask) so downstream tools can exclude saturated samples."""
    x = np.asarray(x, float)
    mask = np.abs(x) >= full_scale
    return np.clip(x, -full_scale, full_scale), mask

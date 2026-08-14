"""
wfmsynth.impairments — physically-grounded impairments (a "fault library") plus
label-preserving domain randomization, applied to any waveform on the physics grid.

    apply_impairment(name, x, rng) -> waveform with one named impairment applied
    domain_randomize(x, rng)       -> waveform with capture-condition nuisances applied

`IMPAIRMENTS` is the vocabulary. Domain randomization (structured-randomization in the
Tobin-2017 sense) is deliberately kept sub-threshold vs the impairment magnitudes so it
stays label-preserving — it models capture-condition variation (gain, DC, bandwidth,
noise, quantization, ambient coupling), not a fault. numpy/scipy only.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as _sig

from . import physics as P

IMPAIRMENTS = ["lossy", "reflect", "jitter", "pj", "dcd", "glitch", "spike",
               "dropout", "dc_offset", "clip", "noise_burst", "attenuate",
               "crosstalk", "baseline_wander"]
IMP_INDEX = {k: i for i, k in enumerate(IMPAIRMENTS)}


def apply_impairment(imp, x, rng):
    """Apply one named impairment (see IMPAIRMENTS) to waveform x on the P.N grid."""
    N = P.N
    if imp == "lossy":
        return P.lossy_channel(x, length_in=rng.uniform(4, 14), tand=rng.uniform(0.01, 0.025), causal=True)
    if imp == "reflect":
        return P.multi_reflection(x, td_frac=rng.uniform(0.06, 0.2),
                                  gamma_s=rng.uniform(0.2, 0.45), gamma_l=rng.uniform(0.2, 0.5))
    if imp == "jitter":
        return P.inject_jitter(x, sigma_rj=rng.uniform(2, 8), rng=rng)
    if imp == "pj":
        return P.inject_jitter(x, a_pj=rng.uniform(3, 10), f_pj=rng.uniform(3, 9), rng=rng)
    if imp == "dcd":
        return P.inject_jitter(x, dcd=rng.uniform(3, 12), rng=rng)
    if imp == "glitch":
        y = x.copy(); t0 = rng.integers(int(0.1*N), int(0.9*N)); w = int(rng.uniform(0.005, 0.02)*N)
        y[t0:t0+w] += rng.uniform(0.5, 1.2) * np.ptp(x) * rng.choice([-1, 1]); return y
    if imp == "spike":
        y = x.copy(); t0 = rng.integers(int(0.1*N), int(0.9*N))
        tt = np.arange(int(0.01*N)); de = np.exp(-tt/1.5) - np.exp(-tt/0.4); de /= de.max()+1e-9
        y[t0:t0+len(tt)] += rng.uniform(0.6, 1.3)*np.ptp(x)*rng.choice([-1, 1])*de; return y
    if imp == "dropout":
        y = x.copy(); t0 = rng.integers(int(0.2*N), int(0.7*N)); w = int(rng.uniform(0.03, 0.12)*N)
        y[t0:t0+w] = x[t0]; return y
    if imp == "dc_offset":
        return x + rng.uniform(0.05, 0.25)*np.ptp(x)*rng.choice([-1, 1])
    if imp == "clip":
        lo, hi = np.percentile(x, [8, 92]); hi = lo + rng.uniform(0.6, 0.9)*(hi-lo); return np.clip(x, None, hi)
    if imp == "noise_burst":
        y = x.copy(); t0 = rng.integers(int(0.2*N), int(0.7*N)); w = int(rng.uniform(0.05, 0.15)*N)
        y[t0:t0+w] += rng.normal(0, 0.15*np.ptp(x), w); return y
    if imp == "attenuate":
        return x * rng.uniform(0.3, 0.7)
    if imp == "crosstalk":                                   # aggressor (NEXT/FEXT) coupling
        aggr = P.nrz(n_ui=int(rng.integers(16, 64)), seed=int(rng.integers(1, 1_000_000)))
        return P.crosstalk(x, aggr, coupling=rng.uniform(0.08, 0.25),
                           kind="fext" if rng.uniform() < 0.5 else "next", td_frac=rng.uniform(0.02, 0.12))
    if imp == "baseline_wander":                            # AC-coupling droop
        lf = x - P.ac_couple(x, fc_frac=rng.uniform(0.006, 0.02))
        return x - rng.uniform(0.6, 1.0) * lf
    return x


def mix_at_constant_power(components, weights, total_rms):
    """Combine several impairment components in declared proportions while holding the
    **total** impairment power fixed — the natural API for "same SNR, different noise
    character" datasets, which separates *how much* impairment there is from *what kind*.

      components  list of arrays (equal length), each an impairment realization
      weights     relative power proportion per component (need not sum to 1)
      total_rms   the RMS the combined result is scaled to, regardless of the mix

    Each component is normalized to unit RMS then scaled by sqrt(w_i) (w normalized to sum
    1), so the powers add in quadrature; the result is then rescaled to exactly ``total_rms``
    so total impairment power is invariant across a mixing sweep while the character (which
    components dominate) moves. Returns the combined array."""
    comps = [np.asarray(c, float) for c in components]
    w = np.asarray(weights, float)
    if w.min() < 0:
        raise ValueError("weights must be non-negative")
    w = w / (w.sum() + 1e-300)
    rms = lambda a: np.sqrt(np.mean(a ** 2)) + 1e-12
    out = np.zeros_like(comps[0])
    for c, wi in zip(comps, w):
        out = out + np.sqrt(wi) * (c / rms(c))                  # unit-power, quadrature weight
    return total_rms * out / rms(out)                           # exact total power, any mix


def domain_randomize(x, rng):
    """Label-preserving capture-condition nuisances: gain, DC, analog bandwidth
    roll-off, AWGN at randomized SNR, 1/f (pink) noise, ambient crosstalk, partial
    baseline wander, ADC quantization. All kept sub-threshold vs the impairment
    magnitudes, so this never masquerades as a labeled fault."""
    N = P.N
    span0 = np.ptp(x) + 1e-9
    # gain + small DC (capture-condition variation, well below the dc_offset/attenuate faults)
    x = x * rng.uniform(0.85, 1.2) + rng.uniform(-0.03, 0.03) * span0
    # mild analog-front-end bandwidth roll-off (not aggressive enough to mimic 'lossy')
    if rng.uniform() < 0.5:
        x = _sig.sosfiltfilt(_sig.bessel(4, rng.uniform(0.2, 0.45), output="sos"), x)
    # global AWGN at a mild randomized SNR
    sig_rms = x.std() + 1e-9
    x = x + rng.normal(0, sig_rms / (10 ** (rng.uniform(20, 45) / 20)), N)
    # low-level 1/f (pink) component
    if rng.uniform() < 0.4:
        w = rng.standard_normal(N)
        X = np.fft.rfft(w); f = np.arange(X.shape[0]); f[0] = 1
        pink = np.fft.irfft(X / np.sqrt(f), N)
        x = x + rng.uniform(0.004, 0.02) * span0 * pink / (pink.std() + 1e-9)
    # mild ambient crosstalk (real links always carry some aggressor coupling)
    if rng.uniform() < 0.4:
        aggr = P.nrz(n_ui=int(rng.integers(16, 64)), seed=int(rng.integers(1, 1_000_000)))
        x = P.crosstalk(x, aggr, coupling=rng.uniform(0.006, 0.02),
                        kind="fext" if rng.uniform() < 0.5 else "next", td_frac=rng.uniform(0.02, 0.10))
    # partial AC-coupling baseline wander (reads as capture-condition, not a droop fault)
    if rng.uniform() < 0.4:
        lf = x - P.ac_couple(x, fc_frac=rng.uniform(0.002, 0.006))
        x = x - rng.uniform(0.15, 0.45) * lf
    # ADC quantization at a randomized bit depth
    if rng.uniform() < 0.5:
        bits = rng.integers(8, 13); lsb = span0 / (2 ** bits)
        x = lsb * np.round(x / lsb)
    return x

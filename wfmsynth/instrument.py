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
                   rng=None, mismatch=None, offset_v=None):
    """Pass x through an M-way time-interleaved ADC with per-core mismatch.

      gain_mm    per-core gain error, std (fractional, e.g. 0.01 = 1%)
      offset_mm  per-core offset, std as a fraction of the signal span
      offset_v   per-core offset, std in ABSOLUTE amplitude units (volts). A real
                 converter's offset error is a property of the ADC, not the signal: it
                 stays put when the signal shrinks (which is when it matters most).
                 Takes precedence over offset_mm when given.
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
        o_std = offset_v if offset_v is not None else offset_mm * span   # absolute vs fraction
        o = rng.normal(0.0, o_std, m_cores)
        s = rng.normal(0.0, skew_mm, m_cores)
    y = x.copy()
    if skew_mm > 0 or (mismatch is not None and np.any(s)):
        xi = np.arange(n)                          # sub-sample timing skew per core
        for c in range(m_cores):
            idx = np.where(core == c)[0]
            y[idx] = np.interp(idx + s[c], xi, x)
    y = y * (1.0 + g[core]) + o[core]              # per-core gain + offset
    return y


def scope_bandwidth(x, grid, bw_hz, kind="bessel", order=4):
    """The acquisition front-end's finite analog bandwidth — a real scope is band-limited and
    rolls off high frequencies. ``kind='bessel'`` (flat group delay, like a real scope) or
    ``'gaussian'``, with the −3 dB point at ``bw_hz``. Part of "what the scope records"."""
    x = np.asarray(x, float)
    wn = min(bw_hz / (grid.fs / 2.0), 0.99)
    if kind == "gaussian":
        f = np.fft.rfftfreq(len(x))
        H = np.exp(-0.5 * (f / (wn / 2.0 + 1e-12)) ** 2)
        return np.fft.irfft(np.fft.rfft(x) * H, len(x))
    return _sig.sosfiltfilt(_sig.bessel(order, wn, output="sos"), x)


def probe_loading(x, grid, c_load_f=0.5e-12, r_source=50.0):
    """A passive probe's input capacitance LOADS the node it measures — an RC low-pass with a
    pole at 1/(2·pi·R·C) that attenuates high frequency. Real probes perturb the DUT."""
    fc = 1.0 / (2 * np.pi * r_source * c_load_f)
    return scope_bandwidth(x, grid, fc, kind="bessel", order=1)


def timebase_jitter(x, grid, rms_ps=0.5, rng=None):
    """Sample-clock / timebase jitter — each sample is taken at a slightly wrong time, which
    smears the eye HORIZONTALLY. Adds independent per-sample timing error (RMS in ps)."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, float)
    n = len(x)
    dev = (rms_ps * 1e-12 * grid.fs) * rng.standard_normal(n)
    idx = np.arange(n)
    return np.interp(idx - dev, idx, x, left=x[0], right=x[-1])


def quantize_adc(x, enob=6.0, full_scale=None):
    """Quantise to a finite-ENOB ADC lattice — arguably the single most characteristic
    thing an ADC does, and previously reachable only inside the full deep_capture pipeline.

      enob        effective number of bits -> ~2**enob levels across the range
      full_scale  +/- range of the lattice; None takes it from the signal's peak

    Returns the quantised array; each sample moves by at most half an LSB. Pairs with
    clip_adc (clip first so out-of-range samples land on the top code, not beyond it)."""
    x = np.asarray(x, float)
    fs = float(np.max(np.abs(x))) + 1e-12 if full_scale is None else float(full_scale)
    lsb = 2.0 * fs / 2 ** enob
    return np.round(x / lsb) * lsb


def digitize(x, grid=None, interleave=None, clip_full_scale=None, enob=None,
             noise_floor=None, rng=None):
    """Compose the ADC stages in the physically correct order and return ``(y, info)``.

    Order — all of it AFTER the channel and the additive impairment: additive noise floor
    -> interleave mismatch (at the sampling instant) -> hard clip (at the ADC input) ->
    quantise (last). Getting this order wrong is silent: quantising before the noise, or
    clipping after quantising, yields a plausible waveform with the wrong noise floor. It
    lives here once so a caller cannot get it wrong.

      noise_floor       kwargs for shaped_noise_floor, e.g. {"rms": 1e-3, "shape": "pink"}
      interleave        kwargs for interleave_adc, e.g. {"m_cores": 4, "offset_v": 1e-3}
      clip_full_scale   hard-clip level (absolute); None to skip. Also sets the quantiser range.
      enob              effective bits for the final quantiser; None to skip

    ``info`` records the applied settings and the clipped-sample mask fraction — feeding
    provenance (#5) and measured ground truth (#8). ``grid`` is accepted for API symmetry;
    amplitude stages need no rate conversion."""
    x = np.asarray(x, float)
    rng = rng or np.random.default_rng()
    info = {}
    if noise_floor:
        x = x + shaped_noise_floor(len(x), rng=rng, **noise_floor)
        info["noise_floor"] = dict(noise_floor)
    if interleave:
        x = interleave_adc(x, rng=rng, **interleave)
        info["interleave"] = dict(interleave)
    if clip_full_scale is not None:
        x, mask = clip_adc(x, clip_full_scale)
        info["clip_full_scale"] = float(clip_full_scale)
        info["clipped_fraction"] = float(mask.mean())
    if enob is not None:
        x = quantize_adc(x, enob=enob, full_scale=clip_full_scale)
        info["enob"] = float(enob)
    return x, info


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

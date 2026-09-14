"""Bandlimited interpolation at arbitrary real sample positions.

Sub-sample displacement is the mechanism behind jitter, duty-cycle distortion, intra-pair skew,
a free-running sample clock and a fractional reflection delay, so how it is interpolated is
physics rather than convenience. Linear interpolation is cheap and wrong in a way that scales
with how sharp the content is. MEASURED against the kernel here, on band-limited content, for a
half-sample shift:

    samples per symbol      4        10       32
    linear error        1.52 %    0.30 %   0.024 %      of peak-to-peak

On a link graded on a 6 mV eye out of 800 mV -- 0.75 % -- the coarse case is twice the quantity
being measured.

This module holds the kernel so `physics` and `instrument` can both use it without one importing
the other. `instrument` re-exports the names it published first.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import i0 as _i0

HALF_WIDTH = 32          # taps each side; the kernel is 2*half_width long
CUTOFF = 0.45            # kernel cutoff as a fraction of fs
BETA = 13.0              # Kaiser beta
PASSBAND_FRAC = 0.386    # MEASURED: exact to about -130 dB below this, useless above it


def resample_at(x, src, half_width=HALF_WIDTH, cutoff=CUTOFF, beta=BETA, chunk=8192):
    """``out[k] = x(src[k])`` for arbitrary real positions `src`, by Kaiser-windowed sinc.

    Normalised per output sample so the DC gain is exactly 1, and linear in `x` (`src` does not
    depend on it), so it obeys superposition.

    Exact to about -130 dB out to ``PASSBAND_FRAC = 0.386*fs``, which is the Kaiser passband edge
    for this length, and USELESS above it. Content above the passband is the one way to misuse
    this: pass a bigger `half_width` (the transition narrows as 1/half_width) or resample from a
    finer grid.

    ENDS: an output sample within `half_width` of either end has no kernel support there, and the
    indices are CLAMPED (a hold at ``x[0]`` / ``x[-1]``). That is exact for a record whose ends are
    a settled quiescent line -- which is what `Signal.lead_in` renders and discards -- and an edge
    artefact over `half_width` samples for a record that starts mid-pattern.
    """
    x = np.asarray(x, dtype=float)
    src = np.asarray(src, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("nothing to resample")
    out = np.empty(src.size, dtype=float)
    taps = np.arange(-half_width + 1, half_width + 1)
    for start in range(0, src.size, chunk):
        s = src[start:start + chunk]
        base = np.floor(s).astype(np.int64)
        frac = s - base
        # offsets of each tap from the requested position
        dt = frac[:, None] - taps[None, :]
        w = np.sinc(2.0 * cutoff * dt) * _kaiser(dt, half_width, beta)
        idx = np.clip(base[:, None] + taps[None, :], 0, n - 1)
        num = (w * x[idx]).sum(axis=1)
        den = w.sum(axis=1)
        out[start:start + chunk] = num / np.where(den == 0.0, 1.0, den)
    return out


def _kaiser(dt, half_width, beta):
    """The Kaiser window evaluated at each tap's offset, zero outside the kernel."""
    r = dt / half_width
    inside = np.abs(r) <= 1.0
    arg = np.sqrt(np.clip(1.0 - r * r, 0.0, None))
    return np.where(inside, _i0(beta * arg) / _i0(beta), 0.0)


def shift(x, samples, half_width=HALF_WIDTH, fill="hold"):
    """Delay `x` by `samples` (a single real number).

    A whole-sample delay is taken exactly rather than through the kernel, so an integer request
    stays bit-exact and cannot pick up interpolation error it does not need.

    `fill` says what lies outside the record, and it is a physical choice rather than a detail.
    "hold" repeats the end sample, which is exact for a record whose ends are a settled quiescent
    line -- what `Signal.lead_in` renders and discards. `fill=0.0` says nothing is there, which is
    what an ECHO needs: the reflection of a record that had not started yet has not arrived, and
    holding the first sample would invent a precursor that the line never carried.
    """
    x = np.asarray(x, dtype=float)
    d = float(samples)
    if d == 0.0:
        return x.copy()
    zero = not isinstance(fill, str)
    if float(d).is_integer():
        k = int(d)
        y = np.zeros_like(x) if zero else np.empty_like(x)
        if k > 0:
            if not zero:
                y[:min(k, x.size)] = x[0]
            if k < x.size:
                y[k:] = x[:x.size - k]
        else:
            k = -k
            if k < x.size:
                y[:x.size - k] = x[k:]
            if not zero:
                y[max(x.size - k, 0):] = x[-1]
        return y
    if zero:
        # Pad ASYMMETRICALLY. Only the side the signal is coming FROM is unknown -- for a delay
        # that is the time before the record started, and `fill` is what was there. The far end
        # is padded by holding, because the record's own last sample is the best statement about
        # what the line was doing then; filling it would roll the tail off and invent an ending.
        pad = half_width + int(math.ceil(abs(d))) + 1
        if d > 0:
            xp = np.concatenate([np.full(pad, float(fill)), x, np.full(pad, x[-1])])
        else:
            xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, float(fill))])
        src = np.arange(pad, pad + x.size, dtype=float) - d
        return resample_at(xp, src, half_width=half_width)
    return resample_at(x, np.arange(x.size, dtype=float) - d, half_width=half_width)


def displace(x, dev, half_width=HALF_WIDTH):
    """Displace each sample of `x` by its own `dev[k]` samples -- the jitter/DCD mechanism.

    `dev` is positive for a LATER edge, matching the `np.interp(idx - dev, ...)` convention this
    replaced.
    """
    x = np.asarray(x, dtype=float)
    dev = np.asarray(dev, dtype=float)
    if dev.size == 1:
        return shift(x, float(dev), half_width=half_width)
    if dev.shape != x.shape:
        raise ValueError(f"dev is {dev.shape}, x is {x.shape}")
    if not np.any(dev):
        return x.copy()
    return resample_at(x, np.arange(x.size, dtype=float) - dev, half_width=half_width)

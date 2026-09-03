"""
wfmsynth.rx — receiver-side equalization (companions to the transmit-side FFE, #11).

A real high-speed receiver equalizes the channel before slicing. Two standard blocks:

  * CTLE — a Continuous-Time Linear Equalizer: an analog high-frequency-peaking filter (a
    zero below its poles) that boosts the attenuated high frequencies to flatten the channel.
    Linear, memoryless of decisions, applied to the waveform.
  * DFE — a Decision-Feedback Equalizer: cancels post-cursor ISI by subtracting weighted PAST
    DECISIONS from the current sample before slicing. Nonlinear (it feeds back sliced
    symbols), so it can remove a sharp discrete post-cursor a linear EQ only smears.

CTLE is composable in the waveform chain (`Signal.ctle(...)`); DFE operates on the per-symbol
samples a receiver has already recovered (pair it with `measure.sample_at_phase`).
"""
from __future__ import annotations

import numpy as np
from scipy import signal as _sig

_PAM4 = np.array([-1.0, -1 / 3, 1 / 3, 1.0])


def ctle(x, grid, fz_ghz, fp1_ghz, fp2_ghz, dc_gain=1.0):
    """Continuous-time linear equalizer: ``H(s) = dc_gain·(1 + s/ωz) / ((1+s/ωp1)(1+s/ωp2))``
    with the zero below the poles, so it peaks at high frequency and flattens a lossy channel.
    ``dc_gain`` is the exact DC gain; the corner frequencies are in GHz. Applied to the
    waveform via a bilinear-transformed digital filter."""
    wz, wp1, wp2 = (2 * np.pi * f * 1e9 for f in (fz_ghz, fp1_ghz, fp2_ghz))
    num = dc_gain * np.array([1.0 / wz, 1.0])
    den = np.polymul([1.0 / wp1, 1.0], [1.0 / wp2, 1.0])
    b, a = _sig.bilinear(num, den, fs=grid.fs)
    return _sig.lfilter(b, a, np.asarray(x, float))


def ffe(x, taps, tap_spacing, pre=0):
    """Receiver feed-forward equalizer: a linear FIR with ``taps`` spaced ``tap_spacing`` samples apart,
    ``pre`` of them pre-cursor. Unlike CTLE (a fixed analog shape) this is an arbitrary-tap FIR the RX
    trains to the channel. ``tap_spacing = samples_per_ui`` is a T-spaced (baud-rate) FFE; ``= spb//2``
    is a fractionally (T/2) spaced FFE — the usual receiver form, insensitive to sampling phase.
    Applied to the WAVEFORM (post-channel), companion to the transmit-side ``tx_ffe``."""
    x = np.asarray(x, float); taps = np.asarray(taps, float); n = len(x); y = np.zeros(n)
    for k, c in enumerate(taps):
        d = int(round((k - pre) * tap_spacing))
        if d >= 0:
            y[d:] += c * x[:n - d]
        elif -d < n:
            y[:d] += c * x[-d:]
    return y


def dfe(samples, taps, levels=_PAM4):
    """Decision-feedback equalizer over per-symbol ``samples``. For each symbol it subtracts
    ``taps · [previous decisions]`` (the post-cursor estimate) before slicing to the nearest
    level. Returns ``(equalized, decisions)``. ``taps[j]`` is the post-cursor weight at lag
    ``j+1``; set them to the channel's post-cursor to cancel it exactly."""
    samples = np.asarray(samples, float)
    taps = np.asarray(taps, float)
    levels = np.asarray(levels, float)
    eq = np.empty_like(samples)
    dec = np.empty_like(samples)
    hist = np.zeros(len(taps))
    for k in range(len(samples)):
        c = samples[k] - np.dot(taps, hist)
        d = levels[int(np.argmin(np.abs(levels - c)))]
        eq[k] = c
        dec[k] = d
        if len(hist):
            hist = np.roll(hist, 1)
            hist[0] = d
    return eq, dec

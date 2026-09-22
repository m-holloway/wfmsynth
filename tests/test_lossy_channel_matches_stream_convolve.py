"""GitHub #56: `lossy_channel`'s whole-record path and `stream.stream_convolve` (the bounded-
memory path for a deep-memory record) must agree at the record's HEAD -- a causal channel
cannot carry energy backwards from the end of an acquisition to its start, so the correct
answer is the LINEAR convolution, not the circular one.

Not reproducible against the current source (`linear=True` is already `lossy_channel`'s
default, and `tests/test_linear_convolution.py` covers the underlying `apply_transfer`
mechanism in detail) -- this pins the exact framing the issue used, so a future default change
cannot quietly reopen it.
"""
from __future__ import annotations

import numpy as np

import wfmsynth as ws
from wfmsynth import physics as P
from wfmsynth.stream import channel_fir, stream_convolve


def test_lossy_channel_head_matches_the_streamed_linear_reference_not_the_circular_one():
    n = 1 << 14
    g = ws.Grid(fs=256e9, baud=28e9, n=n)
    x = P.nrz(n_ui=n // 8, seed=3, n=n, tr_frac=0.35, causal=True)

    h = channel_fir(lambda i: P.lossy_channel(i, loss_db=9.0, loss_at_ghz=53.0, grid=g,
                                              causal=True), n_taps=2048)
    whole = P.lossy_channel(x, loss_db=9.0, loss_at_ghz=53.0, grid=g, causal=True)
    streamed = stream_convolve(x, h)
    circ = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(h, n), n=n)

    span = np.ptp(x)
    head = slice(0, 2048)
    err_vs_linear = np.abs(whole[head] - streamed[head]).max() / span
    err_vs_circular = np.abs(whole[head] - circ[head]).max() / span
    assert err_vs_linear < 0.01, f"whole-record path disagrees with stream_convolve: {err_vs_linear:.4f}"
    assert err_vs_circular > 0.1, "the two references should differ sharply at the head"


def test_lossy_channel_default_is_linear_not_circular():
    n = 1 << 13
    g = ws.Grid(fs=256e9, baud=28e9, n=n)
    x = P.nrz(n_ui=n // 8, seed=5, n=n, tr_frac=0.35, causal=True)
    kw = dict(loss_db=9.0, loss_at_ghz=53.0, grid=g, causal=True)
    default = P.lossy_channel(x, **kw)
    linear = P.lossy_channel(x, linear=True, **kw)
    circular = P.lossy_channel(x, linear=False, **kw)
    assert np.array_equal(default, linear)
    assert not np.array_equal(default, circular)

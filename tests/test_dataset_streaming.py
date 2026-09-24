"""`dataset` writes each record into the output array as it is rendered.

It used to collect every waveform in a list first and then stack them, which held the whole
dataset TWICE -- once at float64 in the list and once at float32 in the array -- so the case it
handled worst was a dataset large enough to be worth building. Streaming is bit-identical
because nothing about the rendering order changes, and therefore nothing about any RNG draw
changes; these tests pin both halves of that claim.
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

L = 1 << 14


def _build(rng):
    """Note `carrier(seed=)`, not just `Signal(seed=)`. `Signal(seed=)` does not seed the
    pattern phase, so a builder that varied only it would hand back n bit-identical records --
    which is exactly what `test_the_seed_still_determines_the_whole_dataset` caught here."""
    s = int(rng.integers(1 << 30))
    g = Grid(fs=80e9, baud=10e9, n=L)
    return (Signal(seed=s, grid=g)
            .carrier("nrz", n_ui=L >> 3, pattern="prbs13", causal=True, seed=s)
            .lossy(length_in=6.0, tand=0.02, causal=True))


def _collected(build, n, seed=0):
    """The form this replaced: render everything, then stack."""
    rng = np.random.default_rng(seed)
    sigs = [build(rng) for _ in range(n)]
    waves = [s.waveform() for s in sigs]
    ln = len(waves[0])
    X = np.empty((n, ln), np.float32)
    for i, w in enumerate(waves):
        X[i] = w[:ln] if len(w) >= ln else np.pad(w, (0, ln - len(w)))
    return X, [s.recipe() for s in sigs]


def test_streaming_is_bit_identical_to_collecting():
    """The whole safety argument in one line: same order, same draws, same records."""
    a, ra = _collected(_build, 8)
    b, rb = ws.dataset(_build, 8)
    assert np.array_equal(a, b)
    assert ra == rb


def test_every_record_still_replays_from_its_own_recipe():
    X, recipes = ws.dataset(_build, 4)
    for i, r in enumerate(recipes):
        assert np.allclose(Signal.from_recipe(r).waveform()[:X.shape[1]], X[i], atol=0)


def test_the_seed_still_determines_the_whole_dataset():
    a, _ = ws.dataset(_build, 4, seed=7)
    b, _ = ws.dataset(_build, 4, seed=7)
    c, _ = ws.dataset(_build, 4, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_peak_memory_is_close_to_the_array_rather_than_a_multiple_of_it():
    """The reason for the change. Peak allocation is a deterministic function of the code, so
    this is a real bound and not a noisy one -- but the bar is set well clear of the measured
    1.2x so that it gates a regression to the 3x form without being brittle."""
    n = 24
    tracemalloc.start()
    X, _ = ws.dataset(_build, n)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    array_bytes = X.nbytes
    assert peak < 2.0 * array_bytes, (
        f"peak {peak/1e6:.1f} MB against a {array_bytes/1e6:.1f} MB array "
        f"= {peak/array_bytes:.2f}x; the collected form was 3.0x")


def test_an_empty_dataset_is_refused_rather_than_crashing_on_an_index():
    """It used to raise IndexError from `waves[0]`, which names nothing."""
    with pytest.raises(ValueError, match="at least 1"):
        ws.dataset(_build, 0)

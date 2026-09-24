"""`Signal.stored_grid(x)` describes the record `waveform()` returned, not the one it was built on.

`Signal.grid` is the SYNTHESIS grid. An `acquire` stores at its own rate and
`digitize(n_out=)` resamples, so after either the synthesis grid no longer describes the record
in hand -- and handing it to a measurement raises nothing, because `measure` only reads
`samples_per_ui` off it. The fold then happens at the wrong symbol period and returns a
plausible number. That was gotcha 17 in the skill file: a documented footgun with no code
behind it.

The reason this takes the rendered record instead of being a property is that it CHECKS. A grid
it cannot justify from the ops is a wrong timebase travelling silently into a measurement,
which is the exact failure it exists to prevent, so it raises instead.
"""
from __future__ import annotations

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth.acquire import AcquisitionProfile
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

N = 1 << 14
FS, BAUD = 80e9, 10e9


def _synth():
    return Grid(fs=FS, baud=BAUD, n=N)


def _chain(**prof_kw):
    prof = AcquisitionProfile(sample_rate_hz=FS / 2, record_length=N // 2,
                              input_bandwidth_hz=25e9, enob=7.0, **prof_kw)
    return (Signal(seed=1, grid=_synth())
            .carrier("nrz", n_ui=N >> 3, pattern="prbs13", causal=True)
            .lossy(length_in=6.0, tand=0.02, causal=True)
            .acquire(prof))


def test_it_recovers_the_grid_the_acquisition_actually_stored_at():
    sig = _chain()
    x = sig.waveform()
    got = sig.stored_grid(x)
    want = Grid(fs=FS / 2, baud=BAUD, n=len(x))
    assert got.fs == want.fs
    assert got.n == want.n == len(x)
    assert got.samples_per_ui == want.samples_per_ui


def test_the_measurement_it_fixes_was_silently_wrong():
    """The whole point, stated as the number. The stale grid returns a plausible figure with no
    warning; the stored grid returns the one an explicitly-built grid returns."""
    sig = _chain()
    x = sig.waveform()
    stale = ws.eye_height(x, sig.grid, levels=2)
    fixed = ws.eye_height(x, sig.stored_grid(x), levels=2)
    truth = ws.eye_height(x, Grid(fs=FS / 2, baud=BAUD, n=len(x)), levels=2)
    assert fixed == truth
    assert abs(stale - truth) / truth > 0.02, (
        "the stale grid no longer disagrees, so this test is no longer measuring anything")


def test_a_chain_that_does_not_resample_gets_its_own_grid_back():
    """The common case must be a no-op, or nobody will use it by default."""
    sig = (Signal(seed=1, grid=_synth())
           .carrier("nrz", n_ui=N >> 3, pattern="prbs13", causal=True)
           .lossy(length_in=6.0, tand=0.02, causal=True))
    x = sig.waveform()
    got = sig.stored_grid(x)
    assert (got.fs, got.n, got.samples_per_ui) == (
        sig.grid.fs, sig.grid.n, sig.grid.samples_per_ui)


def test_decimation_moves_the_rate_as_well_as_the_length():
    """`decimation` keeps every k-th sample, so the stored rate falls by the same factor. A
    grid that took the length but not the rate would be wrong in the ratio that matters."""
    sig = _chain(decimation=dict(mode="sample", depth=2048))
    x = sig.waveform()
    got = sig.stored_grid(x)
    assert got.n == 2048
    assert got.fs == pytest.approx((FS / 2) * 2048 / (N // 2))


def test_peak_hold_has_no_timebase_and_is_refused():
    """A peak-hold acquisition returns a (2, depth) min/max pair -- two channels, not a
    waveform on a timebase. Inventing a grid for it would be worse than refusing."""
    sig = _chain(decimation=dict(mode="peak_hold", depth=2048))
    with pytest.raises(ValueError, match="peak_hold"):
        sig.stored_grid(np.zeros((2, 2048)))


def test_a_length_it_cannot_justify_is_refused_naming_the_fix():
    """The safety net. If the ops imply a different length than the record has, then this does
    not understand the chain, and it must not state a rate either."""
    sig = _chain()
    with pytest.raises(ValueError, match="does not model|derived"):
        sig.stored_grid(np.zeros(99))


def test_it_accepts_a_length_as_well_as_a_record():
    sig = _chain()
    x = sig.waveform()
    assert sig.stored_grid(len(x)).fs == sig.stored_grid(x).fs


def test_digitize_n_out_is_a_resample_and_moves_the_rate():
    """`digitize(n_out=)` goes through `resample_poly`, so the record spans the same time at a
    different rate -- unlike `sample_clock(n_out=)`, which truncates."""
    sig = (Signal(seed=1, grid=_synth())
           .carrier("nrz", n_ui=N >> 3, pattern="prbs13", causal=True)
           .digitize(bits=8, full_scale=1.2, n_out=N // 4))
    x = sig.waveform()
    got = sig.stored_grid(x)
    assert got.n == len(x) == N // 4
    assert got.fs == pytest.approx(FS / 4)


def test_sample_clock_n_out_truncates_so_the_rate_does_not_move():
    """The library's own gotcha 1: `n_out` on `sample_clock` is a record LENGTH, and passing it
    truncates rather than resampling. The rate must therefore stay put."""
    sig = (Signal(seed=1, grid=_synth())
           .carrier("nrz", n_ui=N >> 3, pattern="prbs13", causal=True)
           .sample_clock(ppm=50.0, n_out=N // 4))
    x = sig.waveform()
    got = sig.stored_grid(x)
    assert got.n == len(x) == N // 4
    assert got.fs == FS

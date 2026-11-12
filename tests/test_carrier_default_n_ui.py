"""`carrier`'s `n_ui` defaults to 32 symbols stretched over the record, whatever the grid says.

The default is not changed: every record that relied on it would move. What is added is one
RuntimeWarning for the case that is silently not what was asked -- the grid STATES a symbol
rate, `n_ui` was left out, and the 32 symbols come out at a different rate -- naming both
rates and the value that agrees. It must stay silent wherever the record and the grid already
agree or the caller said what they meant.
"""
import warnings

import numpy as np
import pytest

import wfmsynth as ws
import wfmsynth.physics as P
from wfmsynth import Grid, Signal

G = Grid(fs=64e9, baud=1e9, n=1 << 16)            # 64 samples/UI, 1024 UI


def _caught(build):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = build()
    return out, [str(c.message) for c in w if "n_ui was not stated" in str(c.message)]


def test_the_disagreement_is_said_once_with_both_rates_and_the_fix():
    x, msgs = _caught(lambda: Signal(seed=1, grid=G).carrier("nrz", pattern="clock").waveform())
    assert len(msgs) == 1
    m = msgs[0]
    assert "3.125e+07 Bd" in m and "baud=1e+09" in m           # what it is, what the grid says
    assert "2048 samples per UI" in m and "64 samples per UI" in m
    assert "n_ui=1024 matches the grid" in m
    # and the claim in it is true of the samples: an edge every 2048 samples, not every 64
    hi = x > 0
    assert set(np.diff(np.flatnonzero(hi[1:] != hi[:-1])).tolist()) == {2048}


def test_the_default_output_is_not_changed():
    """The same 32 symbols as ever, and stating them is the same record without the warning."""
    x, msgs = _caught(lambda: Signal(seed=1, grid=G).carrier("nrz").waveform())
    y, none = _caught(lambda: Signal(seed=1, grid=G).carrier("nrz", n_ui=32).waveform())
    assert msgs and not none
    assert np.array_equal(x, y) and np.array_equal(x, P.nrz(n_ui=32, n=G.n))


@pytest.mark.parametrize("build", [
    lambda: Signal(seed=1, grid=G).carrier("nrz", n_ui=1024),                 # stated, agrees
    lambda: Signal(seed=1, grid=G).carrier("nrz", n_ui=100),                  # stated: their call
    lambda: Signal(seed=1, grid=Grid(fs=64e9, n=4096)).carrier("nrz"),        # no baud stated
    lambda: Signal(seed=1, grid=Grid(fs=64e9, baud=1e9, n=2048)).carrier("nrz"),   # 32 IS right
    lambda: Signal(seed=1, grid=Grid(fs=64e9, baud=1e9, n=2048)).carrier("pam4"),
    lambda: Signal(seed=1).carrier("nrz"),                                    # no grid at all
    lambda: Signal(seed=1, grid=G).carrier("sine", f_hz=1e9),                 # not symbols
    lambda: Signal(seed=1, grid=G).carrier("square", f_hz=1e9),
    lambda: Signal(seed=1, grid=G).symbols([1.0, -1.0] * 16),                 # count is stated
    lambda: Signal(seed=1, grid=G).coded("8b10b", prbs=7, n_bits=800),
])
def test_silent_where_the_record_and_the_grid_agree_or_the_caller_said(build):
    _, msgs = _caught(lambda: build().waveform())
    assert msgs == []


def test_a_rate_that_does_not_fit_the_record_is_said_without_inventing_a_count():
    g = Grid(fs=64e9, baud=1.5e9, n=1000)                     # 23.4375 UI
    _, msgs = _caught(lambda: Signal(seed=1, grid=g).carrier("pam4").waveform())
    assert len(msgs) == 1 and "23.4375 UI" in msgs[0] and "matches the grid" not in msgs[0]


def test_it_reaches_a_sub_chain_and_a_channel_which_nothing_else_tells():
    """A sub-chain and a `Channels` chain are written `Signal()` and take the parent's grid."""
    lane = Signal().carrier("nrz", pattern="clock")
    _, msgs = _caught(lambda: Signal(seed=1, grid=G).carrier("nrz", n_ui=1024)
                      .crosstalk(aggressor=lane, coupling=0.1).waveform())
    assert len(msgs) == 1 and "n_ui=1024" in msgs[0]
    _, msgs = _caught(lambda: ws.Channels(
        {"clk": Signal().carrier("nrz", pattern="clock"),
         "dat": Signal().carrier("nrz", n_ui=1024)}, seed=1, grid=G).waveforms())
    assert len(msgs) == 1

"""`method=` has to reach the composable API and survive the recipe, or the block path is
unreachable from the only interface most callers use -- and worse, a record rendered with it
would not replay.

The two paths differ by about 1e-5 of peak-to-peak, which `test_byte_identity.py` classifies as
a behaviour change rather than round-off, so `method` is part of the recipe exactly the way a
physical parameter is: a recipe rendered on the block path must replay on the block path.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS, BAUD = 80e9, 10e9
N = 1 << 17


def _chain(method=None, op="lossy"):
    g = Grid(fs=FS, baud=BAUD, n=N)
    s = Signal(seed=1, grid=g).carrier("nrz", n_ui=N // 8, pattern="prbs13",
                                       tr_frac=0.4, causal=True)
    kw = {} if method is None else {"method": method}
    if op == "lossy":
        return s.lossy(length_in=8.0, tand=0.02, causal=True, **kw)
    return s.cascade([{"line": {"length_in": 4.0}}, {"disc": {"gamma": 0.06}},
                      {"line": {"length_in": 4.0}}], **kw)


@pytest.mark.parametrize("op", ["lossy", "cascade"])
def test_method_reaches_the_op_and_changes_the_render(op):
    """If `method` were silently dropped -- the defect #57 and #63 were both instances of --
    the two renders would be identical and the option would be decorative."""
    assert not np.array_equal(_chain(None, op).waveform(),
                              _chain("overlap", op).waveform())


@pytest.mark.parametrize("op", ["lossy", "cascade"])
def test_the_block_path_is_the_same_channel_to_the_stated_tolerance(op):
    """It must be a different way of computing the SAME convolution, not a different filter."""
    a = _chain(None, op).waveform()
    b = _chain("overlap", op).waveform()
    assert np.abs(a - b).max() < 2e-3 * np.ptp(a)


@pytest.mark.parametrize("op", ["lossy", "cascade"])
def test_method_round_trips_through_the_recipe(op):
    sig = _chain("overlap", op)
    x = sig.waveform()
    assert sig.recipe()["ops"][-1]["method"] == "overlap"
    assert np.array_equal(x, Signal.from_recipe(sig.recipe()).waveform())


@pytest.mark.parametrize("op", ["lossy", "cascade"])
def test_omitting_method_records_nothing_and_renders_as_before(op):
    """The compatibility bar: an existing recipe gains no key, so it hashes exactly as it did."""
    sig = _chain(None, op)
    assert "method" not in sig.recipe()["ops"][-1]
    assert np.array_equal(sig.waveform(), _chain("fft", op).waveform())


def test_an_unknown_method_is_refused_by_the_op_rather_than_ignored():
    with pytest.raises(ValueError, match="unknown method"):
        _chain("bogus").waveform()

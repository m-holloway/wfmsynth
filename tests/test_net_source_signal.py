"""A `net` driver's `{"ops": Signal}` takes the Signal's ops, and says when it drops the rest.

A driver's chain is part of the record: it draws under the record's seed and renders on the
record's grid, so the Signal's own seed and grid are not used. The generic sub-chain path
refuses a Signal whose seed or grid would be dropped; `net` has always accepted it, and
existing calls render, so their output is not changed. What changes is that a seed or grid
that DIFFERS from the record's -- a record that is silently not what was written -- warns,
naming what is ignored and the fix. The record's own seed and grid, and the defaults, stay
silent.
"""
import warnings

import numpy as np
import pytest

from wfmsynth import Grid, Signal

G = Grid(fs=1e8, baud=1e6, n=4096)
NET = dict(nodes=["bus"], rails={"vdd": 3.3}, probe="bus",
           elements=[{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": 885.0},
                     {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 2e-9}])


def _chain(**kw):
    return Signal(**kw).carrier("nrz", n_ui=40, jitter=dict(rj=0.4))


def _net(source, seed=5):
    return Signal(seed=seed, grid=G).net(**NET, drivers=[
        {"id": "host", "node": "bus", "ohm": 60.0, "state": "driving",
         "source": {"ops": source, "volts": [0.0, 3.3]}}])


def _caught(build):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = build()
    return out, [str(c.message) for c in w if "IGNORED" in str(c.message)]


@pytest.mark.parametrize("kw", [
    dict(),                              # nothing stated: the usual way to write one
    dict(grid=G),                        # the record's own grid, as the docs write it
    dict(seed=5, grid=G),                # the record's own seed
    dict(seed=0, grid=Grid(fs=1e8, baud=1e6, n=4096)),      # equal, not the same object
])
def test_silent_when_what_is_dropped_is_the_records_own_or_a_default(kw):
    sig, msgs = _caught(lambda: _net(_chain(**kw)))
    assert msgs == []
    assert np.array_equal(sig.waveform(), _net(_chain()).waveform())


def test_a_different_seed_warns_that_it_is_ignored_and_the_record_is_unchanged():
    sig, msgs = _caught(lambda: _net(_chain(seed=7, grid=G)))
    assert len(msgs) == 1
    m = msgs[0]
    assert "driver 'host'" in m and "seed=7" in m and "seed (5)" in m
    assert "Leave the seed off" in m and "'from': that_signal.recipe()" in m
    # the output of the call is what it always was: the record's seed, not the Signal's
    assert np.array_equal(sig.waveform(), _net(_chain()).waveform())
    assert sig.to_json() == _net(_chain()).to_json()
    # and the claim in the warning is true: that Signal by itself is a different waveform
    own = _chain(seed=7, grid=G).waveform()
    assert not np.array_equal(sig.subchain("net/0/host"), own)


def test_a_different_grid_warns_that_it_is_ignored_and_the_record_is_unchanged():
    other = Grid(fs=2e8, baud=1e6, n=8192)
    sig, msgs = _caught(lambda: _net(_chain(grid=other)))
    assert len(msgs) == 1
    m = msgs[0]
    assert "fs=2e+08 Hz" in m and "n=8192" in m and "fs=1e+08 Hz" in m and "n=4096" in m
    assert "nothing is" in m and "Build it on this record's grid" in m
    assert np.array_equal(sig.waveform(), _net(_chain()).waveform())
    assert len(sig.subchain("net/0/host")) == 4096


def test_both_dropped_is_two_warnings():
    _, msgs = _caught(lambda: _net(_chain(seed=7, grid=Grid(fs=2e8, baud=1e6, n=8192))))
    assert len(msgs) == 2 and "seed=7" in msgs[0] and "grid is IGNORED" in msgs[1]


def test_the_fix_the_warning_names_embeds_the_signal_as_itself():
    own = _chain(seed=7, grid=G)
    sig, msgs = _caught(lambda: Signal(seed=5, grid=G).net(**NET, drivers=[
        {"id": "host", "node": "bus", "ohm": 60.0, "state": "driving",
         "source": {"from": own.recipe(), "volts": [0.0, 3.3]}}]))
    assert msgs == []
    assert np.array_equal(sig.subchain("net/0/host"), own.waveform())


def test_the_generic_sub_chain_path_still_refuses_the_same_thing():
    with pytest.raises(ValueError, match="seed=7.*would be ignored"):
        Signal(seed=5, grid=G).carrier("nrz", n_ui=40).crosstalk(
            aggressor=_chain(seed=7, grid=G), coupling=0.1)

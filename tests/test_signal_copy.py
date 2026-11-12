"""`Signal.copy()` is how a chain is branched.

A builder method mutates the Signal and returns it -- that is kept, because statement-style
code (`sig.lossy(...)` on a line of its own) depends on it. The cost is that reusing a base
chain silently builds ONE chain with every variant's ops. `copy()` is the one-line answer: it
renders and digests as the original does, and is independent afterwards.
"""
import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Grid, Signal

G = Grid(fs=64e9, baud=16e9, n=4096)


def _base():
    return (Signal(seed=4, grid=G).carrier("nrz", n_ui=1024, jitter=dict(rj=0.3))
            .probe(r_source=ws.tol(50, 10), c_load_f=1e-12)
            .crosstalk(aggressor=Signal().carrier("nrz", n_ui=1024, seed=9), coupling=0.1))


def test_the_trap_copy_exists_for():
    base = Signal(seed=1, grid=G).carrier("nrz", n_ui=1024)
    a = base.lossy(loss_db=3, loss_at_ghz=8)
    b = base.rx_noise(rms=0.01)
    assert a is b is base and [o["op"] for o in base.ops] == ["carrier", "lossy", "rx_noise"]


def test_a_copy_renders_and_digests_as_its_original():
    base = _base()
    c = base.copy()
    assert c is not base and c == base
    assert c.to_json() == base.to_json() and c.sha256() == base.sha256()
    assert np.array_equal(c.waveform(), base.waveform())
    assert c.roles() == base.roles() and c.grid is base.grid
    assert c.ops[1]["r_source"] == base.ops[1]["r_source"]      # the part as drawn, not redrawn
    assert np.array_equal(c.contrast("jitter/0", seed=3), base.contrast("jitter/0", seed=3))


def test_a_copy_is_independent_afterwards_in_both_directions():
    base = _base()
    before = base.to_json()
    x = base.waveform()
    short = base.copy().lossy(loss_db=3, loss_at_ghz=8)
    long = base.copy().lossy(loss_db=12, loss_at_ghz=8)
    assert base.to_json() == before and len(base.ops) == 3
    assert [len(s.ops) for s in (short, long)] == [4, 4]
    assert short.ops[-1]["loss_db"] == 3 and long.ops[-1]["loss_db"] == 12
    assert not np.array_equal(short.waveform(), long.waveform())
    # deep, not shallow: a parameter changed inside one does not show in the other
    c = base.copy()
    c.ops[0]["jitter"]["rj"] = 0.9
    c.ops[2]["aggressor"]["ops"][0]["seed"] = 5
    c.annotate(node="TP2")
    assert base.to_json() == before
    base.rx_noise(rms=0.02)                                    # ... and the other way round
    assert len(c.ops) == 3 and np.array_equal(base.copy().waveform(), base.waveform())
    assert not np.array_equal(base.waveform(), x)


def test_a_copy_keeps_the_lead_in_and_a_channel_view_stays_a_view():
    s = Signal(seed=2, grid=G, lead_in=True).carrier("nrz", n_ui=1024).lossy(
        loss_db=6, loss_at_ghz=8, causal=True)
    assert s.copy().lead_in is True and np.array_equal(s.copy().waveform(), s.waveform())
    rec = ws.Channels({"a": Signal().carrier("nrz", n_ui=1024).rx_noise(rms=0.01)},
                      seed=3, grid=G)
    view = rec.signal("a").copy()
    assert np.array_equal(view.waveform(), rec.waveforms()["a"])
    with pytest.raises(ValueError, match="channel 'a' of a several-channel record"):
        view.recipe()


def test_a_several_channel_record_does_not_have_the_trap():
    """`Channels` has no builder methods, and it takes the chains it is given as data: a
    Signal changed after the record was made does not reach the record. So it needs no copy."""
    lane = Signal().carrier("nrz", n_ui=1024)
    rec = ws.Channels({"x": lane, "y": lane}, seed=1, grid=G)
    digest = rec.sha256()
    lane.lossy(loss_db=6, loss_at_ghz=8)
    assert rec.sha256() == digest and not hasattr(rec, "copy")

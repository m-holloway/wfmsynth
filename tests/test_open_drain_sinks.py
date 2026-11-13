"""Any number of sinks on an open-drain line (#102): `open_drain_line(sinks=[...])` and the
Signal op's `others=`. The one- and two-sink forms are pinned by test_open_drain_second_sink.py;
here the N-sink form is checked against the same closed-form dividers."""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth import physics as P
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS, RP, CB, VDD = 1e9, 1000.0, 200e-12, 3.3
R = [20.0, 45.0, 80.0, 150.0]
N = 20000


def _on(*ks):
    return [(np.full(N, k in ks), r) for k, r in enumerate(R)]


@pytest.mark.parametrize("k", range(4))
def test_each_sink_alone_settles_to_its_own_divider(k):
    y = P.open_drain_line(None, FS, RP, CB, v_dd=VDD, sinks=_on(k))
    assert y[-1] == pytest.approx(VDD * R[k] / (R[k] + RP), rel=1e-6)


def test_simultaneous_pulls_use_the_parallel_resistance():
    y = P.open_drain_line(None, FS, RP, CB, v_dd=VDD, sinks=_on(0, 2, 3))
    par = 1.0 / sum(1.0 / R[k] for k in (0, 2, 3))
    assert y[-1] == pytest.approx(VDD * par / (par + RP), rel=1e-6)
    assert y[-1] == pytest.approx(P.open_drain_low_v(RP, [R[0], R[2], R[3]], VDD))


def test_one_and_two_sinks_match_the_legacy_forms():
    rng = np.random.default_rng(3)
    a, b = rng.random(4096) < 0.4, rng.random(4096) < 0.4
    one = P.open_drain_line(a, FS, RP, CB, v_dd=VDD, r_sink_ohm=25.0)
    two = P.open_drain_line(a, FS, RP, CB, v_dd=VDD, r_sink_ohm=25.0, sink_b_on=b,
                            r_sink_b_ohm=90.0)
    assert np.abs(P.open_drain_line(None, FS, RP, CB, v_dd=VDD, sinks=[(a, 25.0)]) - one).max() < 1e-12
    got = P.open_drain_line(None, FS, RP, CB, v_dd=VDD, sinks=[(a, 25.0), (b, 90.0)])
    assert np.abs(got - two).max() < 1e-12


def test_rise_after_the_last_release_is_pullup_times_bus_capacitance():
    m = np.zeros(N, bool)
    m[:2000] = True
    n2 = np.zeros(N, bool)
    n2[500:2500] = True                       # released last, at sample 2500
    y = P.open_drain_line(None, FS, RP, CB, v_dd=VDD, sinks=[(m, 20.0), (n2, 60.0)])
    low = P.open_drain_low_v(RP, [20.0, 60.0], VDD)
    t = np.arange(N - 2500) / FS
    want = VDD + (y[2499] - VDD) * np.exp(-t / (RP * CB))
    assert np.abs(y[2500:] - want).max() < 5e-3 * (VDD - low)


def test_sinks_refusals_are_named():
    z = np.zeros(100, bool)
    with pytest.raises(ValueError, match="not both"):
        P.open_drain_line(z, FS, RP, CB, sinks=[(z, 20.0)])
    with pytest.raises(ValueError, match=r"sinks\[1\].*positive"):
        P.open_drain_line(None, FS, RP, CB, sinks=[(z, 20.0), (z, 0.0)])
    with pytest.raises(ValueError, match="same length"):
        P.open_drain_line(None, FS, RP, CB, sinks=[(z, 20.0), (z[:50], 20.0)])
    with pytest.raises(ValueError, match="non-empty"):
        P.open_drain_line(None, FS, RP, CB, sinks=[])


# ---------------------------------------------------------------- the op
G = Grid(fs=FS, n=8192)


def _bus(**kw):
    return (Signal(seed=1, grid=G).carrier("nrz", n_ui=32, seed=3)
            .open_drain(r_pullup_ohm=RP, c_bus_f=CB, v_dd=VDD, r_sink_ohm=25.0, **kw))


def test_others_round_trips_and_the_recipe_records_them():
    sig = _bus(second=dict(kind="nrz", n_ui=32, seed=9), r_sink_b_ohm=90.0,
               others=[dict(source=dict(kind="nrz", n_ui=32, seed=11), r_sink_ohm=60.0),
                       dict(source=Signal(grid=G).carrier("nrz", n_ui=32, seed=12))])
    rec = sig.recipe()
    assert rec["ops"][-1]["others"][0]["r_sink_ohm"] == 60.0
    assert np.array_equal(Signal.from_recipe(rec).waveform(), sig.waveform())


def test_others_adds_the_sinks_the_two_sink_form_would_resolve():
    spec = lambda s: dict(kind="nrz", n_ui=32, seed=s)
    a = _bus(second=spec(9), r_sink_b_ohm=90.0).waveform()
    b = _bus(second=spec(9), r_sink_b_ohm=90.0,
             others=[dict(source=spec(11), r_sink_ohm=60.0)]).waveform()
    assert np.abs(a - b).max() > 0.05                    # a third pull is visible
    # others alone, in place of second, is the same line as second
    c = _bus(second=spec(9), r_sink_b_ohm=90.0).waveform()
    d = _bus(others=[dict(source=spec(9), r_sink_ohm=90.0)]).waveform()
    assert np.abs(c - d).max() < 1e-9


def test_others_roles_and_namespaces():
    sig = _bus(others=[dict(source=dict(kind="nrz", n_ui=32, seed=11, jitter=dict(rj=1e-12))),
                       dict(source=Signal(grid=G).carrier("nrz", n_ui=32, seed=12))])
    assert any("open_drain_other" in r for r in sig.roles())
    assert "open_drain/1/others/1/source" in sig.subchains()


def test_others_refusals_name_the_entry_and_suggest():
    spec = dict(kind="nrz", n_ui=32, seed=11)
    with pytest.raises(ValueError, match=r"others\[0\].*'r_sink_ohm'"):
        _bus(others=[dict(source=spec, r_sink_ohms=60.0)])
    with pytest.raises(ValueError, match=r"others\[1\].*'source'"):
        _bus(others=[dict(source=spec), dict(sourse=spec)])
    with pytest.raises(ValueError, match=r"others\[0\] needs source"):
        _bus(others=[dict(r_sink_ohm=60.0)])
    with pytest.raises(ValueError, match="positive"):
        _bus(others=[dict(source=spec, r_sink_ohm=-1.0)])
    with pytest.raises(ValueError, match="non-empty list"):
        _bus(others=spec)
    with pytest.raises(ValueError, match="non-empty list"):
        _bus(others=[])


def test_open_drain_low_v_refuses_a_bad_resistance_by_name():
    for bad in (0.0, -5.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match=r"r_sink_ohms\[1\]"):
            P.open_drain_low_v(4.7e3, [60.0, bad])
        with pytest.raises(ValueError, match="r_pullup_ohm"):
            P.open_drain_low_v(bad, [60.0])
    assert P.open_drain_low_v(4.7e3, []) == pytest.approx(3.3)


def test_an_others_source_of_another_length_names_the_entry_and_the_fix():
    import wfmsynth as ws
    g = ws.Grid(fs=2e9, n=4000)
    a = ws.Signal(grid=g).carrier("pulse", amp=-1.0, offset=1.0, t_start_s=1e-6, width_s=1e-6,
                                  tr_s=20e-9)
    with pytest.raises(ValueError, match=r"others\[0\].*2000.*4000.*same grid"):
        a.open_drain(r_pullup_ohm=4.7e3, c_bus_f=1e-10,
                     others=[dict(source={"kind": "nrz", "n": 2000},
                                  r_sink_ohm=60.0)]).waveform()

"""GitHub #72, follow-on: drivers whose output follows their own signal chain.

Tier 1 lets a driver's STATE follow a carrier; its output was still a fixed Thevenin level. On
a net shared by several drivers each party has its own levels, its own edge rate, its own
timing impairments and its own path to the node, and adding their contributions by hand outside
the solve is where a level held by a driver that has let go gets carried into the next owner's
baseline.

`solve(sources={id: volts})` makes a driver's level a waveform, and `Signal.net` carries that as
a nested spec per driver. The claim is still EXACTNESS, so these tests check it the same three
independent ways `test_net.py` does:

  * the reduction: a constant source must be a constant `volts` BIT-FOR-BIT, not nearly;
  * against a CLOSED FORM -- the one-driver case is exponential stepping on its own target,
    which is the method `physics.open_drain_line` is written in;
  * against an INDEPENDENT integrator, from the node equations by hand and stepped with RK45 --
    a different formulation and a different method, so agreement is evidence not a shared bug.

And the behaviour the request was actually about: a released driver contributing NOTHING,
whatever it was holding.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from wfmsynth import net as NET
from wfmsynth import scenario as SC
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS = 1e9
DT = 1.0 / FS


def _one_node(drivers, elements=(), rails=None, farad=100e-12):
    return {"nodes": ["bus"], "rails": dict(rails or {}),
            "elements": [{"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": farad},
                         *elements],
            "drivers": list(drivers)}


# ------------------------------------------------------------------- the reduction to tier 1
def test_a_constant_source_is_a_constant_volts_bit_for_bit():
    """Not "to 1e-15". A source that never moves must take the SAME code path as a declared
    level, or every record that happens to idle picks up a different last bit from the one the
    tier-1 solver gave it, and nothing in a digest would say why."""
    n = 4000
    net = _one_node([{"id": "d", "node": "bus", "ohm": 50.0, "state": "driving"}],
                    elements=[{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": 1000.0}],
                    rails={"vdd": 3.3})
    declared = {**net, "drivers": [{**net["drivers"][0], "volts": 0.4}]}
    a = NET.solve(declared, fs=FS, n=n, probes="bus")
    b = NET.solve(net, fs=FS, n=n, probes="bus", sources={"d": np.full(n, 0.4)})
    assert np.array_equal(a, b)


def test_an_interval_that_does_not_move_is_still_bit_for_bit():
    """The routing is per INTERVAL, not per record: a source that is flat while the driver
    happens to be on must still reproduce the declared-level answer exactly."""
    n = 3000
    net = _one_node([{"id": "d", "node": "bus", "ohm": 50.0, "state": "hiz"}],
                    elements=[{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": 1000.0}],
                    rails={"vdd": 3.3})
    sched = [{"t_s": 1e-6, "id": "d", "state": "driving"},
             {"t_s": 2e-6, "id": "d", "state": "hiz"}]
    src = np.full(n, 0.25)
    src[:1000] = np.linspace(-5.0, 5.0, 1000)          # moves only while the driver is OFF
    declared = {**net, "drivers": [{**net["drivers"][0], "volts": 0.25}]}
    assert np.array_equal(NET.solve(declared, fs=FS, n=n, probes="bus", schedule=sched),
                          NET.solve(net, fs=FS, n=n, probes="bus", schedule=sched,
                                    sources={"d": src}))


# ------------------------------------------------------ closed form: one driver, its own chain
def test_one_driver_equals_its_own_chain_through_the_same_r_and_c():
    """With one driver and one capacitor the net IS a first-order RC fed by the driver, and
    `physics.open_drain_line`'s method -- exponential stepping on a target held across each
    sample -- gives it in one line. Same answer, written two ways."""
    n, rd, c = 2000, 90.0, 47e-12
    t = np.arange(n) * DT
    s = 1.25 + 1.0 * np.sin(2 * np.pi * 3e6 * t) + 0.3 * np.sin(2 * np.pi * 11e6 * t)
    got = NET.solve(_one_node([{"id": "d", "node": "bus", "ohm": rd, "state": "driving"}],
                              farad=c), fs=FS, n=n, probes="bus", sources={"d": s})

    alpha = np.exp(-DT / (rd * c))
    v, want = s[0], np.zeros(n)                        # settled at the first sample's level
    for k in range(n):
        v = s[k] + (v - s[k]) * alpha
        want[k] = v
    err = np.abs(got - want).max()
    assert err < 1e-14 * np.ptp(want), f"{err:.2e} against the scalar closed form"


# ------------------------------------------- an independent integrator, by hand, other method
def test_it_agrees_with_an_independent_integrator_on_a_moving_source():
    """Two coupled nodes, cross-coupling capacitance, a pull-up, and a driver whose level
    moves every sample. Written from the node equations and stepped with RK45."""
    n = 600
    rab, rpb, rd, vdd = 220.0, 1500.0, 90.0, 2.5
    ca, cb, cab = 47e-12, 68e-12, 5e-12
    spec = {"nodes": ["a", "b"], "rails": {"vdd": vdd},
            "elements": [{"id": "rab", "kind": "r", "a": "a", "b": "b", "ohm": rab},
                         {"id": "rpb", "kind": "r", "a": "b", "b": "vdd", "ohm": rpb},
                         {"id": "ca", "kind": "c", "a": "a", "b": "gnd", "farad": ca},
                         {"id": "cb", "kind": "c", "a": "b", "b": "gnd", "farad": cb},
                         {"id": "cab", "kind": "c", "a": "a", "b": "b", "farad": cab}],
            "drivers": [{"id": "d", "node": "a", "ohm": rd, "state": "driving"}]}
    t = np.arange(n) * DT
    s = 1.25 + 1.0 * np.sin(2 * np.pi * 3e6 * t) + 0.3 * np.sin(2 * np.pi * 11e6 * t)
    got = NET.solve(spec, fs=FS, n=n, probes=["a", "b"], sources={"d": s})

    C = np.array([[ca + cab, -cab], [-cab, cb + cab]])
    G = np.array([[1 / rab + 1 / rd, -1 / rab], [-1 / rab, 1 / rab + 1 / rpb]])
    Cinv = np.linalg.inv(C)
    v = np.linalg.solve(G, np.array([s[0] / rd, vdd / rpb]))
    ref = np.zeros((n, 2))
    for k in range(n):
        bk = np.array([s[k] / rd, vdd / rpb])          # held across the sample: a sampled source
        r = solve_ivp(lambda _t, y: Cinv @ (bk - G @ y), (0.0, DT), v,
                      rtol=1e-12, atol=1e-14, method="RK45")
        v = r.y[:, -1]
        ref[k] = v
    for i, nm in enumerate(("a", "b")):
        err = np.abs(got[nm] - ref[:, i]).max()
        assert err < 1e-11, f"node {nm}: {err:.2e} against the hand-integrated reference"


# ------------------------------------------------------- what the request was actually about
def _handover(levels, n=24000, t_hold=2e-6, gap=2e-6, ohm=50.0):
    """`len(levels)` drivers taking the bus in turn, each letting go while holding a level that
    is not the idle one. The bus leaks to gnd, so idle is 0 V."""
    ids = [f"d{i}" for i in range(len(levels))]
    net = _one_node([{"id": i, "node": "bus", "ohm": ohm, "state": "hiz"} for i in ids],
                    elements=[{"id": "leak", "kind": "r", "a": "bus", "b": "gnd", "ohm": 1000.0}])
    t = np.arange(n) * DT
    sched, sources, gaps = [], {}, []
    for k, (i, lv) in enumerate(zip(ids, levels)):
        t0 = k * (t_hold + gap)
        sched += [{"t_s": t0, "id": i, "state": "driving"},
                  {"t_s": t0 + t_hold, "id": i, "state": "hiz"}]
        gaps.append(NET.sample_at(t0 + t_hold + gap, FS) - 1)
        # a real driver: moving while it holds, and NOT at the idle level when it lets go
        sources[i] = lv * (1.0 + 0.25 * np.sin(2 * np.pi * 1.5e6 * t))
    return net, sched, sources, gaps


def test_a_released_driver_contributes_nothing_whatever_it_was_holding():
    """The bug the request names. A driver that has let go is out of the stamp, so its level is
    ABSENT rather than inherited -- which means the record cannot depend on it at all. Changing
    every released driver's source by volts must leave the samples bit-identical."""
    levels = [3.0, 1.2, 2.4, 0.6]
    net, sched, src_a, _ = _handover(levels)
    _, _, src_b, _ = _handover([-9.0, 7.5, -4.5, 11.0])
    n = len(next(iter(src_a.values())))
    on = {}
    for i in src_a:
        k0 = next(NET.sample_at(e["t_s"], FS) for e in sched
                  if e["id"] == i and e["state"] == "driving")
        k1 = next(NET.sample_at(e["t_s"], FS) for e in sched
                  if e["id"] == i and e["state"] == "hiz")
        on[i] = (k0, k1)
    mixed = {i: np.where((np.arange(n) >= on[i][0]) & (np.arange(n) < on[i][1]),
                         src_a[i], src_b[i]) for i in src_a}
    assert np.array_equal(NET.solve(net, fs=FS, n=n, probes="bus", schedule=sched,
                                    sources=src_a),
                          NET.solve(net, fs=FS, n=n, probes="bus", schedule=sched,
                                    sources=mixed))


def test_alternating_drivers_leave_no_baseline_drift():
    """Four drivers in turn, each holding a different non-idle level when it lets go. The gap
    is twenty time constants, so the bus must be back at its own idle level at the end of every
    one of them -- the same number each time, not one held offset further along."""
    levels = [3.0, 1.2, 2.4, 0.6]
    net, sched, sources, gaps = _handover(levels)
    n = len(next(iter(sources.values())))
    v = NET.solve(net, fs=FS, n=n, probes="bus", schedule=sched, sources=sources)
    idle = np.array([v[k] for k in gaps])
    assert np.abs(idle).max() < 1e-7, f"the bus idles at {idle} V, not at its own 0 V"
    assert v[: gaps[0]].max() > 2.9, "nothing was driven at all"


# ----------------------------------------------------------------- the op, labels, the recipe
def _g(n=24000):
    return Grid(fs=FS, baud=None, n=n)


def _two_party(**source_kw):
    g = _g()
    host = {"id": "host", "node": "bus", "ohm": 60.0, "state": "hiz",
            "source": {"kind": "cmos", "v_lo": 0.0, "v_hi": 3.3, "f_hz": 1.5e6, "tr_s": 20e-9,
                       "volts": True, **source_kw}}
    tgt = {"id": "tgt", "node": "bus", "ohm": 30.0, "state": "hiz",
           "source": {"kind": "cmos", "v_lo": 0.0, "v_hi": 1.8, "f_hz": 2.5e6, "tr_s": 20e-9,
                      "volts": True}}
    holds = [SC.Hold("host", 1e-6, 5e-6, who_drives="host"),
             SC.Hold("tgt", 8e-6, 14e-6, who_drives="tgt")]
    agents = {"host": SC.Agent(during={"host": "driving"}, after={"host": "hiz"}),
              "tgt": SC.Agent(during={"tgt": "driving"}, after={"tgt": "hiz"})}
    sig = (Signal(seed=4, grid=g)
           .net(nodes=["bus"],
                elements=[{"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 100e-12},
                          {"id": "leak", "kind": "r", "a": "bus", "b": "gnd", "ohm": 1000.0}],
                drivers=[host, tgt], holds=holds, agents=agents, probe="bus"))
    return sig, holds, g


def test_each_drivers_hold_intervals_come_out_as_labels():
    """The hold list is the stimulus and the label at once, and with per-driver sources the
    label is now checkable against the levels: each window carries the driver that produced it,
    and each driver's own rail shows through its own output resistance into the bus leak."""
    sig, holds, g = _two_party()
    x = sig.waveform()
    win = SC.hold_windows(holds, g.fs, g.n)
    by = {w["who"]: x[w["start"]:w["stop"]] for w in win}
    assert by["host"].max() == pytest.approx(3.3 * 1000.0 / 1060.0, abs=0.01)   # 60 ohm out
    assert by["tgt"].max() == pytest.approx(1.8 * 1000.0 / 1030.0, abs=0.01)    # 30 ohm out
    assert by["tgt"].max() < by["host"].max() - 1.0, "the two drivers are not distinguishable"
    quiet = x[win[0]["stop"] + 2000:win[1]["start"] - 10]
    assert np.abs(quiet).max() < 1e-3, "the bus did not go idle between the two parties"


def test_a_driver_source_round_trips_through_the_recipe():
    sig, _, _ = _two_party()
    x = sig.waveform()
    r = sig.recipe()
    json.dumps(r)                                        # provenance only if it serialises
    assert np.array_equal(Signal.from_recipe(r).waveform(), x)
    assert Signal.from_recipe(r).sha256() == sig.sha256()


def test_a_signal_is_accepted_as_a_source_chain_and_stored_as_its_ops():
    g = _g(4000)
    drv = (Signal(grid=g).carrier("cmos", v_lo=0.0, v_hi=1.8, f_hz=2e6, tr_s=10e-9)
           .nonlinearity(compression=0.08))
    sig = (Signal(seed=1, grid=g)
           .net(nodes=["bus"],
                elements=[{"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 200e-12},
                          {"id": "leak", "kind": "r", "a": "bus", "b": "gnd", "ohm": 2000.0}],
                drivers=[{"id": "d", "node": "bus", "ohm": 40.0, "state": "driving",
                          "source": {"ops": drv, "volts": True}}],
                probe="bus"))
    stored = sig.recipe()["ops"][0]["drivers"][0]["source"]["ops"]
    assert [o["op"] for o in stored] == ["carrier", "nonlinearity"]
    json.dumps(sig.recipe())
    assert np.array_equal(Signal.from_recipe(sig.recipe()).waveform(), sig.waveform())


def test_the_net_can_be_the_first_op_when_every_driver_brings_its_own_source():
    """Nothing has to arrive at a net all of whose drivers carry their own chain; the length
    comes from the grid. A decorative carrier in front of it would be a lie about the record."""
    g = _g(4000)
    sig = Signal(seed=2, grid=g).net(
        nodes=["bus"],
        elements=[{"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 200e-12},
                  {"id": "leak", "kind": "r", "a": "bus", "b": "gnd", "ohm": 2000.0}],
        drivers=[{"id": "d", "node": "bus", "ohm": 40.0, "state": "driving",
                  "source": {"kind": "cmos", "v_lo": 0.0, "v_hi": 1.8, "f_hz": 2e6,
                             "tr_s": 10e-9, "volts": True}}],
        probe="bus")
    x = sig.waveform()
    assert len(x) == g.n and x.max() > 1.7


def test_from_chain_makes_the_incoming_waveform_this_drivers_level():
    g = _g(4000)
    sig = (Signal(seed=5, grid=g).carrier("nrz", n_ui=40, tr_frac=0.3)
           .net(nodes=["bus"],
                elements=[{"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 200e-12},
                          {"id": "leak", "kind": "r", "a": "bus", "b": "gnd", "ohm": 5000.0}],
                drivers=[{"id": "d", "node": "bus", "ohm": 50.0, "state": "driving",
                          "source": {"from": "chain", "volts": [0.0, 1.8]}}],
                probe="bus"))
    x = sig.waveform()
    assert x.min() > -0.05 and 1.7 < x.max() <= 1.85


def test_two_drivers_with_the_same_spec_do_not_share_a_draw():
    """Role names are namespaced by driver id, so the same spec on two drivers is two
    independent devices -- and both still move with the record's one seed."""
    g = _g(2000)

    def build(seed, probe):
        spec = {"kind": "nrz", "n_ui": 20, "tr_frac": 0.3, "jitter": {"rj": 0.05},
                "volts": [0.0, 1.8]}
        return Signal(seed=seed, grid=g).net(
            nodes=["a", "b"],
            elements=[{"id": "ca", "kind": "c", "a": "a", "b": "gnd", "farad": 200e-12},
                      {"id": "cb", "kind": "c", "a": "b", "b": "gnd", "farad": 200e-12},
                      {"id": "la", "kind": "r", "a": "a", "b": "gnd", "ohm": 5000.0},
                      {"id": "lb", "kind": "r", "a": "b", "b": "gnd", "ohm": 5000.0}],
            drivers=[{"id": "p", "node": "a", "ohm": 50.0, "state": "driving",
                      "source": dict(spec)},
                     {"id": "q", "node": "b", "ohm": 50.0, "state": "driving",
                      "source": dict(spec)}],
            probe=probe)

    xa, xb = build(7, "a").waveform(), build(7, "b").waveform()
    assert not np.array_equal(xa, xb), \
        "two drivers with the same spec drew the same jitter -- one device, written twice"
    assert np.array_equal(xa, build(7, "a").waveform())        # and still deterministic
    assert not np.array_equal(xa, build(8, "a").waveform()), "the record ignores Signal(seed=)"


# ------------------------------------------------------------------------------ refusals
def _d(**kw):
    d = {"id": "d", "node": "bus", "ohm": 50.0, "state": "driving"}
    d.update(kw)
    return d


def test_a_source_without_volts_is_refused_because_a_carrier_is_not_volts():
    g = _g(512)
    sig = Signal(seed=1, grid=g).net(
        nodes=["bus"],
        elements=[{"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 20e-12},
                  {"id": "lk", "kind": "r", "a": "bus", "b": "gnd", "ohm": 5000.0}],
        drivers=[_d(source={"kind": "nrz", "n_ui": 8})], probe="bus")
    with pytest.raises(ValueError, match="must say `volts`"):
        sig.waveform()


@pytest.mark.parametrize("src,match", [
    ({"kind": "nrz", "n_ui": 8, "volts": True}, "normalised"),
    ({"kind": "cmos", "v_lo": 0.0, "v_hi": 1.8, "f_hz": 1e6, "volts": [0.0, 1.8]},
     "already in volts"),
    ({"kind": "nrz", "n_ui": 8, "volts": [1.0, 1.0]}, "no swing"),
    ({"kind": "nrz", "n_ui": 8, "volts": "yes"}, r"volts='yes'"),
    ({"volts": True}, "exactly one of"),
    ({"kind": "nrz", "n_ui": 8, "ops": [], "volts": True}, "exactly one of"),
    ({"from": "later", "volts": True}, "only value is"),
    ({"from": "chain", "volts": [0.0, 1.8]}, "first op"),
])
def test_a_malformed_source_is_refused_naming_what_is_wrong(src, match):
    g = _g(512)
    sig = Signal(seed=1, grid=g).net(
        nodes=["bus"],
        elements=[{"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 20e-12},
                  {"id": "lk", "kind": "r", "a": "bus", "b": "gnd", "ohm": 5000.0}],
        drivers=[_d(source=src)], probe="bus")
    with pytest.raises(ValueError, match=match):
        sig.waveform()


def test_a_source_chain_that_changes_the_record_length_is_refused():
    g = _g(512)
    sig = Signal(seed=1, grid=g).net(
        nodes=["bus"],
        elements=[{"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 20e-12},
                  {"id": "lk", "kind": "r", "a": "bus", "b": "gnd", "ohm": 5000.0}],
        drivers=[_d(source={"ops": [{"op": "carrier", "kind": "cmos", "v_lo": 0.0,
                                     "v_hi": 1.8, "f_hz": 1e6, "tr_s": 1e-8},
                                    {"op": "digitize", "n_out": 256}],
                            "volts": True})], probe="bus")
    with pytest.raises(ValueError, match="SAME grid"):
        sig.waveform()


def test_a_source_in_the_net_description_points_at_the_solver_argument():
    """`net.solve` takes rendered arrays; the description has to stay JSON for the recipe."""
    with pytest.raises(ValueError, match="sources="):
        NET.solve(_one_node([_d(source={"kind": "nrz"})]), fs=FS, n=8, probes="bus")


def test_sources_naming_an_element_says_a_resistor_has_no_level():
    net = _one_node([_d()], elements=[{"id": "lk", "kind": "r", "a": "bus", "b": "gnd",
                                       "ohm": 1000.0}])
    with pytest.raises(ValueError, match="ELEMENT"):
        NET.solve(net, fs=FS, n=8, probes="bus", sources={"lk": np.zeros(8)})


def test_a_driver_with_both_volts_and_a_source_is_refused():
    with pytest.raises(ValueError, match="silently ignored"):
        NET.solve(_one_node([_d(volts=0.0)]), fs=FS, n=8, probes="bus",
                  sources={"d": np.zeros(8)})


def test_a_scheduled_src_against_a_rendered_source_is_refused():
    with pytest.raises(ValueError, match="Both are its Thevenin level"):
        NET.solve(_one_node([_d()]), fs=FS, n=8, probes="bus",
                  sources={"d": np.zeros(8)},
                  schedule=[{"t_s": 2e-9, "id": "d", "src": 1.0}])


@pytest.mark.parametrize("arr,match", [
    (np.zeros(7), "samples"),
    (np.array([0.0] * 7 + [np.nan]), "not finite"),
])
def test_a_source_that_is_not_a_level_per_sample_is_refused(arr, match):
    with pytest.raises(ValueError, match=match):
        NET.solve(_one_node([_d()]), fs=FS, n=8, probes="bus", sources={"d": arr})


@pytest.mark.parametrize("bad,match", [
    ({"id": "d", "node": "bus", "ohms": 50.0}, "unknown key 'ohms'"),
    ({"id": "d", "node": "bus", "ohm": 50.0, "level": 1.0}, "unknown key 'level'"),
    ({"id": "d", "volts": 0.0, "ohm": 50.0}, "no 'node'"),
])
def test_a_misspelled_driver_key_is_refused_rather_than_ignored(bad, match):
    """A description is a dict, so an invented key would otherwise be accepted and do nothing."""
    with pytest.raises(ValueError, match=match):
        NET.solve(_one_node([bad]), fs=FS, n=8, probes="bus")


@pytest.mark.parametrize("bad,match", [
    ({"id": "x", "kind": "c", "a": "bus", "b": "gnd", "farad": 1e-12, "state": "open"},
     r"c 'x' carries 'state'.*CHARGE is the state"),
    ({"id": "x", "kind": "r", "a": "bus", "b": "gnd", "farad": 1e-12},
     r"element 'x' has unknown key 'farad'"),
    ({"id": "x", "a": "bus", "b": "gnd", "ohm": 1e3}, r"no 'kind'"),
])
def test_a_misspelled_element_key_is_refused_per_kind(bad, match):
    """Per KIND, because the traps differ: a capacitor is never switched out of the stamp --
    its charge is the state the solve carries across every switch -- so a `state` on one is
    honoured by nobody, and a resistor given a `farad` has no resistance to stamp."""
    with pytest.raises(ValueError, match=match):
        NET.solve(_one_node([_d(volts=0.0)], elements=[bad]), fs=FS, n=8, probes="bus")


def test_an_annotation_key_on_a_driver_is_allowed():
    """The recipe convention: anything beginning with an underscore is annotation."""
    NET.solve(_one_node([_d(volts=0.0, _prov={"part": "BSS138"})]), fs=FS, n=8, probes="bus")

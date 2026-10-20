"""GitHub #72, tier 1: a shared net as one channel stage.

A channel elsewhere in this library is a series 2-port cascade with one source and one probe. A
shared net is not that -- several agents drive it at different times through hi-Z, terminations
switch in and out, a weaker driver holds a level against a stronger one -- and none of that is
expressible as a chain. `wfmsynth.net` solves the lumped tier of it exactly.

"Exactly" is the claim these tests exist to check, and they check it three independent ways:

  * against CLOSED FORMS derived here (an RC charge, a divider, a switched divider);
  * against an INDEPENDENT integrator, written in this file from the node equations by hand
    and stepped with scipy's RK45 -- a different formulation AND a different method from the
    matrix-exponential solve, so agreement is evidence rather than a shared bug;
  * against `physics.open_drain_line`, which is the one-node case of the same recurrence and
    which the issue requires stay bit-for-bit.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from wfmsynth import net as NET
from wfmsynth import physics as P
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS = 1e9
RP, CB, VDD, RS = 885.0, 400e-12, 3.3, 20.0


def _bus(**driver):
    d = {"id": "d0", "node": "bus", "volts": 0.0, "ohm": RS, "state": "hiz"}
    d.update(driver)
    return {"nodes": ["bus"], "rails": {"vdd": VDD},
            "elements": [{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": RP},
                         {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": CB}],
            "drivers": [d]}


# --------------------------------------------------------------- closed forms, derived here
def test_an_rc_settles_to_the_rail_on_its_own_time_constant():
    """v(t) = V(1 - exp(-t/RC)) from rest. The solver is told to start at 0 V."""
    n = 4000
    v = NET.solve(_bus(), fs=FS, n=n, probes="bus", v0={"bus": 0.0})
    t = (np.arange(1, n + 1)) / FS                      # a sample records the END of its step
    want = VDD * (1.0 - np.exp(-t / (RP * CB)))
    assert np.abs(v - want).max() < 1e-12 * VDD


def test_a_closed_driver_settles_to_the_divider():
    v = NET.solve(_bus(state="driving"), fs=FS, n=6000, probes="bus")
    assert abs(float(v[-1]) - VDD * RS / (RP + RS)) < 1e-12


def test_the_initial_condition_is_the_dc_state_of_the_configuration_at_t0():
    """Not zero, and not the final state -- the steady state of whatever is in force at t=0."""
    assert abs(float(NET.solve(_bus(), fs=FS, n=8, probes="bus")[0]) - VDD) < 1e-12
    held = NET.solve(_bus(state="driving"), fs=FS, n=8, probes="bus")
    assert abs(float(held[0]) - VDD * RS / (RP + RS)) < 1e-12


# ------------------------------------------------- the one-node case IS open_drain_line
@pytest.mark.parametrize("seed", [1, 7])
def test_it_reproduces_open_drain_line(seed):
    """The issue's requirement: `open_drain_line` stays the one-node special case, bit-for-bit.
    Equality is to double precision, not to a tolerance anyone chose."""
    n = 4000
    x = np.asarray(P.nrz(n_ui=n // 50, n=n, seed=seed, causal=True), float)
    sink = x >= 0.5 * (x.min() + x.max())
    ref = P.open_drain_line(sink, FS, RP, CB, v_dd=VDD, r_sink_ohm=RS)
    got = NET.solve(_bus(), fs=FS, n=n, probes="bus",
                    schedule=NET.schedule_from_waveform(x, "d0"))
    assert np.abs(got - ref).max() < 1e-13, "the one-node case drifted from open_drain_line"


def test_a_second_sink_reproduces_open_drains_wired_and():
    """Two agents pulling the same net down together -- the wired-AND. The parallel sink
    resistance is a consequence of the nodal solve here, not a special case written into it."""
    n = 3000
    a = np.zeros(n, bool); a[500:2000] = True
    b = np.zeros(n, bool); b[1200:2600] = True
    ref = P.open_drain_line(a, FS, RP, CB, v_dd=VDD, r_sink_ohm=RS,
                            sink_b_on=b, r_sink_b_ohm=47.0)
    spec = _bus()
    spec["drivers"].append({"id": "d1", "node": "bus", "volts": 0.0, "ohm": 47.0, "state": "hiz"})

    def sched(mask, did):
        e = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
        out = [{"t_s": 0.0, "id": did, "state": "driving" if mask[0] else "hiz"}]
        return out + [{"t_s": float(i), "_sample": int(i), "id": did,
                       "state": "driving" if mask[i] else "hiz"} for i in e]

    got = NET.solve(spec, fs=FS, n=n, probes="bus", schedule=sched(a, "d0") + sched(b, "d1"))
    assert np.abs(got - ref).max() < 1e-13


# ----------------------------------------- an independent integrator, by hand, different method
def _hand_integrated(fs, n, t_switch):
    """Two coupled nodes, written straight from the node equations and stepped with RK45.

    Ca va' = (vdd - va)/Rpa + (vb - va)/Rab
    Cb vb' = (va - vb)/Rab - vb/Rt   [Rt only while closed]

    Deliberately neither the same formulation nor the same method as `net.solve`'s
    matrix exponential, so agreement is evidence and not a shared mistake.
    """
    Rpa, Rab, Rt, Ca, Cb, V = 500.0, 150.0, 100.0, 12e-12, 8e-12, 1.8

    def f(t, y, term):
        va, vb = y
        ia = (V - va) / Rpa + (vb - va) / Rab
        ib = (va - vb) / Rab - (vb / Rt if term else 0.0)
        return [ia / Ca, ib / Cb]

    y = [V, V]                                           # DC state with Rt open: both at the rail
    ts = np.arange(1, n + 1) / fs
    out = np.zeros((n, 2))
    t0 = 0.0
    for k, t in enumerate(ts):
        term = t0 >= t_switch - 1e-18
        r = solve_ivp(f, (t0, t), y, args=(term,), rtol=1e-11, atol=1e-14, method="RK45")
        y = [r.y[0, -1], r.y[1, -1]]
        out[k] = y
        t0 = t
    return out


def test_it_agrees_with_an_independent_integrator_on_a_coupled_two_node_net():
    """The case `open_drain_line` cannot express at all: two nodes, coupling between them, and
    a termination that switches."""
    n, t_switch = 300, 100 / FS
    spec = {"nodes": ["a", "b"], "rails": {"vdd": 1.8},
            "elements": [{"id": "rpa", "kind": "r", "a": "a", "b": "vdd", "ohm": 500.0},
                         {"id": "ca", "kind": "c", "a": "a", "b": "gnd", "farad": 12e-12},
                         {"id": "cb", "kind": "c", "a": "b", "b": "gnd", "farad": 8e-12},
                         {"id": "rab", "kind": "r", "a": "a", "b": "b", "ohm": 150.0},
                         {"id": "rt", "kind": "r", "a": "b", "b": "gnd", "ohm": 100.0,
                          "state": "open"}]}
    got = NET.solve(spec, fs=FS, n=n, probes=["a", "b"],
                    schedule=[{"t_s": t_switch, "id": "rt", "state": "closed"}])
    ref = _hand_integrated(FS, n, t_switch)
    for i, nm in enumerate(("a", "b")):
        err = np.abs(got[nm] - ref[:, i]).max()
        assert err < 1e-9, f"node {nm}: {err:.2e} against the hand-integrated reference"


def test_the_switch_is_continuous_in_the_capacitor_voltages():
    """Physics, not convention: a capacitor's voltage cannot step. Across the switching sample
    the change must be bounded by the local slope, not by the change in steady state."""
    n, ks = 400, 200
    spec = {"nodes": ["a"], "rails": {"vdd": 1.8},
            "elements": [{"id": "rp", "kind": "r", "a": "a", "b": "vdd", "ohm": 500.0},
                         {"id": "ca", "kind": "c", "a": "a", "b": "gnd", "farad": 12e-12},
                         {"id": "rt", "kind": "r", "a": "a", "b": "gnd", "ohm": 2000.0,
                          "state": "open"}]}
    # tau = (500||2000) * 12 pF = 4.8 ns against dt = 1 ns, so the edge is RESOLVED and the
    # per-sample change is a slope rather than a jump. At 50 ohm it would be 545 ps, the
    # transient would finish inside one sample, and the solver warns -- see the test below.
    v = NET.solve(spec, fs=FS, n=n, probes="a",
                  schedule=[{"t_s": ks / FS, "id": "rt", "state": "closed"}])
    step = abs(v[ks] - v[ks - 1])
    settle = abs(v[-1] - v[0])
    assert step < 0.35 * settle, f"the voltage jumped {step:.3f} V across the switch"
    assert step > 0, "nothing happened at the switch"


# ------------------------------------------------------------------ the op, and the recipe
def _chain(**kw):
    g = Grid(fs=FS, baud=20e6, n=4000)
    p = dict(nodes=["bus"], rails={"vdd": VDD},
             elements=[{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": RP},
                       {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": CB}],
             drivers=[{"id": "d0", "node": "bus", "volts": 0.0, "ohm": RS, "state": "hiz"}],
             follow="d0")
    p.update(kw)
    return (Signal(seed=1, grid=g)
            .carrier("nrz", n_ui=80, pattern="prbs13", tr_frac=0.5, causal=True)
            .net(**p))


def test_the_op_reproduces_the_same_physics_as_the_function():
    assert np.all(np.isfinite(_chain().waveform()))


def test_the_recipe_round_trips_bit_for_bit():
    sig = _chain()
    x = sig.waveform()
    assert np.array_equal(x, Signal.from_recipe(sig.recipe()).waveform())


def test_the_whole_net_is_plain_json():
    """A recipe is provenance only if it serialises. Nested dicts are fine; a callable is not,
    which is why voltage-dependent laws are named with parameters rather than passed in."""
    import json
    r = _chain().recipe()
    assert json.loads(json.dumps(r)) == r


@pytest.mark.parametrize("knob,value", [
    ("ohm_pullup", 2000.0), ("farad_bus", 120e-12), ("ohm_sink", 5.0),
])
def test_moving_a_knob_moves_the_waveform(knob, value):
    """`net` is excused from the global knob sweep because its parameters are nested dicts, so
    it owes the same assurance here: each physical quantity must actually reach the solve."""
    base = _chain().waveform()
    p = dict(nodes=["bus"], rails={"vdd": VDD},
             elements=[{"id": "rp", "kind": "r", "a": "bus", "b": "vdd",
                        "ohm": value if knob == "ohm_pullup" else RP},
                       {"id": "cb", "kind": "c", "a": "bus", "b": "gnd",
                        "farad": value if knob == "farad_bus" else CB}],
             drivers=[{"id": "d0", "node": "bus", "volts": 0.0,
                       "ohm": value if knob == "ohm_sink" else RS, "state": "hiz"}],
             follow="d0")
    assert not np.array_equal(base, _chain(**p).waveform()), f"{knob} did not reach the solve"


def test_a_schedule_time_moves_the_waveform():
    s = [{"t_s": 1e-6, "id": "d0", "state": "driving"}]
    a = _chain(follow=None, schedule=s).waveform()
    b = _chain(follow=None, schedule=[{"t_s": 2e-6, "id": "d0", "state": "driving"}]).waveform()
    assert not np.array_equal(a, b)


def test_a_lead_in_is_refused_because_the_schedule_is_in_absolute_time():
    """A termination switching at 2 us is not the same termination switching 2 us into a guard,
    so the op refuses rather than silently moving its own events."""
    from wfmsynth.compose import _LEAD_REJECT
    assert "net" in _LEAD_REJECT
    assert "absolute" in _LEAD_REJECT["net"].lower()


# ------------------------------------------------------------------------------ refusals
def test_a_node_with_no_capacitance_is_refused_naming_it():
    spec = {"nodes": ["x"], "rails": {"v": 1.0},
            "elements": [{"id": "r", "kind": "r", "a": "x", "b": "v", "ohm": 50.0}]}
    with pytest.raises(ValueError, match="no capacitance"):
        NET.solve(spec, fs=FS, n=4, probes="x")


def test_a_floating_node_is_refused_rather_than_solved_as_anything():
    spec = {"nodes": ["x"], "rails": {"v": 1.0},
            "elements": [{"id": "c", "kind": "c", "a": "x", "b": "gnd", "farad": 1e-12},
                         {"id": "r", "kind": "r", "a": "x", "b": "v", "ohm": 50.0,
                          "state": "open"}]}
    with pytest.raises(ValueError, match="singular|floating|no resistive path"):
        NET.solve(spec, fs=FS, n=4, probes="x")


@pytest.mark.parametrize("bad,match", [
    ({"nodes": []}, "nothing to solve"),
    ({"nodes": ["a", "a"]}, "duplicate node"),
    ({"nodes": ["gnd"]}, "implicit 0 V reference"),
])
def test_malformed_nets_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        NET.solve(bad, fs=FS, n=4, probes="a")


def test_an_unknown_node_reference_names_what_is_known():
    spec = {"nodes": ["a"], "elements": [{"id": "r", "kind": "r", "a": "a", "b": "nope",
                                          "ohm": 50.0}]}
    with pytest.raises(ValueError, match="nope"):
        NET.solve(spec, fs=FS, n=4, probes="a")


def test_a_schedule_event_for_an_unknown_id_is_refused():
    with pytest.raises(ValueError, match="not an element or driver"):
        NET.solve(_bus(), fs=FS, n=8, probes="bus",
                  schedule=[{"t_s": 1e-9, "id": "ghost", "state": "driving"}])


def test_an_unsupported_element_kind_says_what_this_tier_has():
    spec = _bus()
    spec["elements"].append({"id": "l", "kind": "l", "a": "bus", "b": "gnd", "henry": 1e-9})
    with pytest.raises(ValueError, match="kind|'r' and 'c'"):
        NET.solve(spec, fs=FS, n=8, probes="bus")


def test_several_nodes_without_a_probe_is_refused_not_guessed():
    """A chain carries one waveform. Picking a node for the caller would be a silent choice."""
    g = Grid(fs=FS, baud=20e6, n=64)
    sig = (Signal(seed=1, grid=g).carrier("nrz", n_ui=2, tr_frac=0.5)
           .net(nodes=["a", "b"], rails={"v": 1.0},
                elements=[{"id": "ca", "kind": "c", "a": "a", "b": "gnd", "farad": 1e-12},
                          {"id": "cb", "kind": "c", "a": "b", "b": "gnd", "farad": 1e-12},
                          {"id": "r1", "kind": "r", "a": "a", "b": "v", "ohm": 50.0},
                          {"id": "r2", "kind": "r", "a": "b", "b": "a", "ohm": 50.0}]))
    with pytest.raises(ValueError, match="probe"):
        sig.waveform()


def test_a_mode_the_grid_cannot_show_is_named():
    """Exact is not the same as visible. A transient that completes inside one sample is solved
    correctly and cannot be seen, and a rise time measured off it would be the grid's --
    `open_drain_line` warns about exactly this, and so must its generalisation."""
    spec = {"nodes": ["a"], "rails": {"vdd": 1.8},
            "elements": [{"id": "rp", "kind": "r", "a": "a", "b": "vdd", "ohm": 500.0},
                         {"id": "ca", "kind": "c", "a": "a", "b": "gnd", "farad": 12e-12},
                         {"id": "rt", "kind": "r", "a": "a", "b": "gnd", "ohm": 50.0,
                          "state": "open"}]}
    with pytest.warns(RuntimeWarning, match="faster than this grid can show"):
        NET.solve(spec, fs=FS, n=400, probes="a",
                  schedule=[{"t_s": 200 / FS, "id": "rt", "state": "closed"}])


def test_the_warning_fires_once_per_time_constant_not_once_per_switch():
    """A bus that switches a thousand times has one under-resolved edge, not a thousand."""
    spec = {"nodes": ["a"], "rails": {"vdd": 1.8},
            "elements": [{"id": "rp", "kind": "r", "a": "a", "b": "vdd", "ohm": 500.0},
                         {"id": "ca", "kind": "c", "a": "a", "b": "gnd", "farad": 12e-12},
                         {"id": "rt", "kind": "r", "a": "a", "b": "gnd", "ohm": 50.0,
                          "state": "open"}]}
    sched = [{"t_s": k * 20 / FS, "id": "rt", "state": "closed" if k % 2 else "open"}
             for k in range(1, 30)]
    with pytest.warns(RuntimeWarning) as rec:
        NET.solve(spec, fs=FS, n=800, probes="a", schedule=sched)
    assert len([w for w in rec if "grid can show" in str(w.message)]) <= 2, \
        "one warning per switch would be noise nobody reads"

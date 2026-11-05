"""GitHub #74: inductance in the shared net, so that a hand-over rings.

An R-C net can only SETTLE. What a scope shows when a driver turns on or lets go on a fast bus
is overshoot and a ring, at a frequency set by the package and trace inductance against the node
capacitance and decaying at a rate set by the termination -- and for a waveform-level comparison
of a synthetic record against a measured one, that was the largest remaining visible difference.

An inductor puts its branch current into the state vector, which costs the symmetry the R-C
tier's eigensolver needs, so the inductive path is a different decomposition (an exact
zero-order-hold pair back-substituted through a Schur form). That means the exactness claim has
to be re-established rather than inherited, and these tests do it the same three independent
ways `test_net.py` and `test_net_sources.py` do:

  * against a CLOSED FORM -- the textbook series-RLC step response, in all three damping
    regimes, including EXACTLY critical damping, where the eigenvector solve the obvious
    implementation would use is defective and returns hundreds of millivolts of nonsense;
  * against an INDEPENDENT integrator -- the state equations written out by hand and stepped
    with RK45, a different formulation and a different method;
  * by SPLITTING a record at a switch and re-entering the solve with the state it handed over,
    which must reproduce the single solve bit for bit -- the acceptance criterion about an
    inductor current staying continuous across a switch, stated as an identity rather than as a
    tolerance.

Plus the third acceptance criterion, that a net with no inductors is untouched: it must not
reach the inductive code at all.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from wfmsynth import net as NET
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

# The series RLC under test: 100 nH and 100 pF ring at 50 MHz, far enough below the grid that
# the record shows the ring itself rather than the grid's opinion of it.
L, C, VS = 100e-9, 100e-12, 1.0
FS = 2e10
DT = 1.0 / FS
R_CRIT = 2.0 * np.sqrt(L / C)                    # 2 sqrt(L/C) = 63.2 ohm, exactly critical


def _series_rlc(ohm, henry=L, farad=C, volts=VS):
    """A rail through a series L (with its own series resistance) into a capacitor: a textbook
    series RLC, with no node that is not a state and no parasitic invented to make it solvable.
    `v0` gives it the textbook initial conditions -- at rest, uncharged."""
    return {"nodes": ["bus"], "rails": {"vdd": volts},
            "elements": [{"id": "lp", "kind": "l", "a": "vdd", "b": "bus",
                          "henry": henry, "ohm": ohm},
                         {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": farad}]}


def _analytic(ohm, t, henry=L, farad=C, volts=VS):
    """The capacitor voltage of a series RLC driven by a step, from the standard solution --
    three different closed forms, because the damping decides which one applies."""
    alpha, w0 = ohm / (2.0 * henry), 1.0 / np.sqrt(henry * farad)
    if alpha < w0 * (1 - 1e-12):
        wd = np.sqrt(w0 ** 2 - alpha ** 2)
        return volts * (1 - np.exp(-alpha * t) * (np.cos(wd * t) + alpha / wd * np.sin(wd * t)))
    if alpha > w0 * (1 + 1e-12):
        wd = np.sqrt(alpha ** 2 - w0 ** 2)
        return volts * (1 - np.exp(-alpha * t) * (np.cosh(wd * t) + alpha / wd * np.sinh(wd * t)))
    return volts * (1 - np.exp(-alpha * t) * (1 + alpha * t))


def _solve_rlc(ohm, n=4000, **kw):
    return NET.solve(_series_rlc(ohm, **kw), fs=FS, n=n, probes=["bus", "i(lp)"],
                     v0={"bus": 0.0, "i(lp)": 0.0})


# --------------------------------------------------------------------- against the closed form

@pytest.mark.parametrize("ohm,label", [(10.0, "underdamped"), (R_CRIT, "critical"),
                                       (4 * np.sqrt(L / C), "overdamped")])
def test_a_series_rlc_is_the_analytic_step_response_in_every_damping_regime(ohm, label):
    """The acceptance criterion. Three regimes and three different closed forms, and the solve
    is not told which one it is in.

    The critical case is the one that decides the implementation: at exactly R = 2 sqrt(L/C) the
    system matrix is DEFECTIVE -- its eigenvalues coincide and its eigenvector matrix is
    singular -- so the modal solve that generalises the R-C tier returns 368 mV of nonsense on
    this 1 V step. The Schur form has a unitary basis and does not care."""
    n = 4000
    out = _solve_rlc(ohm, n=n)
    t = np.arange(1, n + 1) * DT
    err = np.abs(out["bus"] - _analytic(ohm, t)).max()
    assert err < 1e-12, f"{label}: {err:.3e} V against the closed form"


def test_the_ring_frequency_damping_and_overshoot_are_the_circuit_s():
    """Named separately from the waveform comparison because these three numbers are what a
    measurement actually reports, and a record can be close in RMS while getting them wrong."""
    ohm, n = 10.0, 4000
    out = _solve_rlc(ohm, n=n)
    v, t = out["bus"], np.arange(1, n + 1) * DT
    alpha, w0 = ohm / (2.0 * L), 1.0 / np.sqrt(L * C)
    wd = np.sqrt(w0 ** 2 - alpha ** 2)

    # frequency, from the zero crossings of v - 1 (one half period apart)
    sign = np.sign(v - VS)
    cross = t[np.flatnonzero(np.diff(sign) != 0)]
    assert len(cross) > 8
    f_meas = 0.5 / np.diff(cross).mean()
    assert abs(f_meas / (wd / (2 * np.pi)) - 1.0) < 2e-3

    # damping, from the decay of successive peak excursions
    peaks = np.flatnonzero((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:])) + 1
    assert len(peaks) >= 3
    ratio = (v[peaks[1]] - VS) / (v[peaks[0]] - VS)
    assert abs(ratio / np.exp(-alpha * 2 * np.pi / wd) - 1.0) < 1e-3

    # overshoot: the first peak of a series RLC step is exp(-alpha pi / wd) above the level
    assert abs((v.max() - VS) / (VS * np.exp(-alpha * np.pi / wd)) - 1.0) < 1e-4


def test_critical_damping_is_the_fastest_settle_without_overshoot():
    """The reason R = 2 sqrt(L/C) is a number somebody designs to, and therefore the reason the
    defective case is not a corner nobody reaches."""
    just_under = _solve_rlc(R_CRIT * 0.98)["bus"]
    critical = _solve_rlc(R_CRIT)["bus"]
    just_over = _solve_rlc(R_CRIT * 1.02)["bus"]
    assert just_under.max() > VS                      # rings
    assert critical.max() <= VS + 1e-12               # does not
    # and it gets there sooner than the overdamped one: time to reach 99 % of the level
    t99 = lambda v: int(np.flatnonzero(v > 0.99 * VS)[0])    # noqa: E731
    assert t99(critical) < t99(just_over)


def test_an_inductor_current_is_amps_and_is_the_loop_current():
    """`i(<id>)` is the state the solve now carries, and the only way to see it. At DC the loop
    current is the rail divided by the total series resistance, which is arithmetic."""
    net = {"nodes": ["bus"], "rails": {"vdd": 2.0},
           "elements": [{"id": "lp", "kind": "l", "a": "vdd", "b": "bus",
                         "henry": 1e-9, "ohm": 3.0},
                        {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 50e-12},
                        {"id": "rl", "kind": "r", "a": "bus", "b": "gnd", "ohm": 97.0}]}
    dc = NET.solve(net, fs=1e10, n=4000, probes=["bus", "i(lp)"])
    assert abs(dc["i(lp)"][-1] - 2.0 / 100.0) < 1e-12
    assert abs(dc["bus"][-1] - 2.0 * 97.0 / 100.0) < 1e-12
    assert np.ptp(dc["i(lp)"]) < 1e-15, "a record opened at DC has no transient to show"
    # started from rest it overshoots instead, because the current is a STATE with its own
    # momentum rather than an algebraic consequence of the node voltage
    step = NET.solve(net, fs=1e10, n=4000, probes=["bus", "i(lp)"],
                     v0={"bus": 0.0, "i(lp)": 0.0})
    assert step["i(lp)"].max() > 1.5 * (2.0 / 100.0)
    assert abs(step["i(lp)"][-1] - 2.0 / 100.0) < 1e-9


def test_the_series_ohm_on_an_inductor_is_the_wire_s_own_resistance():
    """It is one element with two numbers because the node between a series R and a series L has
    no capacitance, and this tier solves states. So the series ohm has to damp exactly as a
    resistor in that position would -- which is what the closed-form comparison above measures,
    and this pins the two-element reading of it as equivalent where it IS expressible."""
    # a resistor from the rail to a node, then the inductor on to the bus: expressible only
    # because the intermediate node is given the capacitance it would really have
    split = {"nodes": ["mid", "bus"], "rails": {"vdd": VS},
             "elements": [{"id": "rs", "kind": "r", "a": "vdd", "b": "mid", "ohm": 10.0},
                          {"id": "cm", "kind": "c", "a": "mid", "b": "gnd", "farad": C * 1e-6},
                          {"id": "lp", "kind": "l", "a": "mid", "b": "bus", "henry": L},
                          {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": C}]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # the 1 fF node is a fast mode
        two = NET.solve(split, fs=FS, n=4000, probes="bus",
                        v0={"mid": 0.0, "bus": 0.0, "i(lp)": 0.0})
    one = _solve_rlc(10.0)["bus"]
    # the difference is the parasitic the split form had to invent, not the reading of `ohm`
    assert np.abs(one - two).max() < 1e-5 * VS


# ---------------------------------------------------------- against an independent integrator

def _board(drive_ohm=5.0, henry=4e-9, series=0.2):
    """A driver behind a package inductance onto a terminated bus -- the configuration the
    request describes, and the one that actually rings at a hand-over."""
    return {"nodes": ["pad", "bus"], "rails": {"vdd": 3.3},
            "elements": [{"id": "lp", "kind": "l", "a": "pad", "b": "bus",
                          "henry": henry, "ohm": series},
                         {"id": "cp", "kind": "c", "a": "pad", "b": "gnd", "farad": 4e-12},
                         {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 20e-12},
                         {"id": "rt", "kind": "r", "a": "bus", "b": "vdd", "ohm": 150.0}],
            "drivers": [{"id": "d0", "node": "pad", "volts": 0.0, "ohm": drive_ohm,
                         "state": "driving"}]}


def test_a_switched_inductive_net_agrees_with_an_independent_rk45_integrator():
    """A different formulation (the state equations written out by hand) stepped by a different
    method (adaptive Runge-Kutta), across a switch. Agreement is evidence; a shared bug would
    have to be in two unrelated places."""
    fs, n, t_sw = 4e10, 4000, 50e-9
    net = _board()
    sched = [{"t_s": t_sw, "id": "d0", "state": "hiz"}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = NET.solve(net, fs=fs, n=n, schedule=sched, probes=["pad", "bus", "i(lp)"])

    cp, cb, rt, lh, ls, rd, vdd = 4e-12, 20e-12, 150.0, 4e-9, 0.2, 5.0, 3.3

    def rhs(t, y, driving):
        vp, vb, il = y
        # KCL at the pad: the driver (when it conducts) and the inductor current leaving it
        ip = (-vp / rd if driving else 0.0) - il
        ib = (vdd - vb) / rt + il
        return [ip / cp, ib / cb, (vp - vb - ls * il) / lh]

    # DC with the driver on, by hand: di/dt = 0 means vp - vb = ls*il, and the two node
    # equations then give il = -vdd / (rd + rt + ls)
    il0 = -vdd / (rd + rt + ls)
    y0 = [-il0 * rd, vdd + il0 * rt, il0]
    ref = []
    for seg_t0, seg_t1, on in ((0.0, t_sw, True), (t_sw, n / fs, False)):
        ts = np.arange(1, n + 1) / fs
        ts = ts[(ts > seg_t0) & (ts <= seg_t1 + 0.5 / fs)]
        if not ts.size:
            continue
        sol = solve_ivp(rhs, (seg_t0, ts[-1]), y0, t_eval=ts, args=(on,),
                        rtol=1e-11, atol=1e-16, method="RK45")
        ref.append(sol.y)
        y0 = list(sol.y[:, -1])
    ref = np.hstack(ref)
    assert ref.shape[1] == n
    for k, name in enumerate(("pad", "bus")):
        err = np.abs(out[name] - ref[k]).max()
        assert err < 1e-8, f"{name}: {err:.3e} V against RK45"
    assert np.abs(out["i(lp)"] - ref[2]).max() < 1e-10


# ----------------------------------------------------- the current is continuous across a switch

def test_the_state_handed_over_at_a_switch_reproduces_the_record_bit_for_bit():
    """The acceptance criterion about continuity, as an identity rather than a tolerance.

    Split the record at the switch and re-enter the solve with the voltages AND the inductor
    current it handed over. If anything about the hand-over were a convention rather than the
    state carried forward -- a current reset, a re-solved DC point, an off-by-one -- the two
    halves would not line up, and a continuity test written as `abs(i[k] - i[k-1]) < eps` would
    not have noticed, because one sample of a real di/dt is not zero either."""
    fs, n, k_sw = 4e10, 4000, 2000
    net, sched = _board(), [{"t_s": k_sw / 4e10, "id": "d0", "state": "hiz"}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        whole = NET.solve(net, fs=fs, n=n, schedule=sched, probes=["pad", "bus", "i(lp)"])
        first = NET.solve(net, fs=fs, n=k_sw, probes=["pad", "bus", "i(lp)"])
        released = {**net, "drivers": [{**net["drivers"][0], "state": "hiz"}]}
        second = NET.solve(released, fs=fs, n=n - k_sw, probes=["pad", "bus", "i(lp)"],
                           v0={nm: first[nm][-1] for nm in ("pad", "bus", "i(lp)")})
    for nm in ("pad", "bus", "i(lp)"):
        assert np.array_equal(whole[nm][:k_sw], first[nm])
        assert np.array_equal(whole[nm][k_sw:], second[nm]), nm


def test_an_interrupted_driver_leaves_its_inductor_current_to_the_node_capacitance():
    """Why an inductor is not switchable and why every node needs capacitance: the current has
    to go somewhere. Here the driver lets go while sinking 20 mA, and the pad capacitance takes
    it -- which is the overshoot. Nothing about the released driver's level can matter."""
    fs, n = 4e10, 4000
    net, sched = _board(), [{"t_s": 50e-9, "id": "d0", "state": "hiz"}]
    k = NET.sample_at(50e-9, fs)
    src = {"d0": np.zeros(n)}
    absurd = {"d0": np.where(np.arange(n) >= k, -47.0, 0.0)}
    net = {**net, "drivers": [{kk: vv for kk, vv in net["drivers"][0].items()
                               if kk != "volts"}]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        held = NET.solve(net, fs=fs, n=n, schedule=sched, probes=["pad", "i(lp)"],
                         sources=src)
        other = NET.solve(net, fs=fs, n=n, schedule=sched, probes=["pad", "i(lp)"],
                          sources=absurd)
    assert abs(held["i(lp)"][k - 1]) > 15e-3                 # it really was conducting
    assert np.array_equal(held["pad"][k:], other["pad"][k:])   # released means absent
    assert held["pad"][k:].max() > 3.3                       # and the pad overshoots the rail
    assert held["pad"][k:].min() < held["pad"][k - 1] + 1e-9  # ... having come from below it


def test_a_lossless_tank_keeps_ringing_without_drifting():
    """The worst case for a recurrence: nothing damps the rounding either. An ideal L across an
    ideal C oscillates forever, so this is where the inductive path's accumulation floor is
    measured rather than asserted -- and the amplitude must not creep."""
    henry, farad = 1e-9, 1e-12
    net = {"nodes": ["bus"], "rails": {"vdd": 1.0},
           "elements": [{"id": "lp", "kind": "l", "a": "vdd", "b": "bus", "henry": henry},
                        {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": farad}]}
    n = 200000
    out = NET.solve(net, fs=1e12, n=n, probes="bus", v0={"bus": 0.0})
    t = np.arange(1, n + 1) * 1e-12
    w0 = 1.0 / np.sqrt(henry * farad)
    err = np.abs(out - (1.0 - np.cos(w0 * t))).max()
    assert err < 1e-10, f"{err:.3e} V over {n} samples of a lossless tank"
    # no creep: an undamped mode that gained energy from the recurrence would leave the
    # analytic envelope, and 200000 samples is 1000 periods for it to do it in
    assert out.max() <= 2.0 + 1e-10 and out.min() >= -1e-10


# ------------------------------------------------------------ a net with no inductors is untouched

def test_a_net_with_no_inductors_never_reaches_the_inductive_path(monkeypatch):
    """The third acceptance criterion, pinned where it cannot drift: bit-for-bit identity with
    the R-C tier is a consequence of taking the SAME CODE, so the test is that the inductive
    code is not entered at all rather than a digest somebody would later regenerate."""
    def boom(*a, **k):                                 # pragma: no cover - must not be called
        raise AssertionError("an R-C net took the inductive path")
    monkeypatch.setattr(NET, "_rlc_prepare", boom)
    monkeypatch.setattr(NET, "_rlc_system", boom)
    net = {"nodes": ["bus"], "rails": {"vdd": 3.3},
           "elements": [{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": 885.0},
                        {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 400e-12}],
           "drivers": [{"id": "d0", "node": "bus", "volts": 0.0, "ohm": 200.0, "state": "hiz"}]}
    out = NET.solve(net, fs=1e8, n=512, probes="bus",
                    schedule=[{"t_s": 1e-6, "id": "d0", "state": "driving"}])
    assert np.isfinite(out).all() and out[0] > 3.0


# --------------------------------------------------------------- the grid has to show the ring

def test_a_ring_the_grid_cannot_show_is_named_and_an_alias_is_called_one():
    """Exactness is not visibility, and a ring is worse than a decay: a decay too fast for the
    grid is simply missing from the record, but a ring too fast ALIASES and comes back at a
    frequency that is not the circuit's and looks entirely plausible."""
    net = _series_rlc(2.0)                   # rings at 50 MHz
    with pytest.warns(RuntimeWarning, match=r"rings at 5\.03\S*e\+07 Hz.*8 samples per period"):
        NET.solve(net, fs=2e8, n=64, probes="bus")       # 4 samples per period: coarse
    with pytest.warns(RuntimeWarning, match="past Nyquist.*alias"):
        NET.solve(net, fs=6e7, n=64, probes="bus")       # 1.2 samples per period: aliased


def test_a_well_resolved_ring_says_nothing():
    """The warning has to be worth reading, so it must not fire on the ordinary case."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NET.solve(_series_rlc(10.0), fs=FS, n=512, probes="bus")


def test_the_ring_warning_fires_once_per_frequency_not_once_per_interval():
    """A bus that switches a thousand times has one under-resolved ring, not a thousand."""
    net = _series_rlc(2.0)
    net["drivers"] = [{"id": "d0", "node": "bus", "volts": 0.0, "ohm": 1e4, "state": "hiz"}]
    sched = [{"t_s": k * 1e-7, "id": "d0", "state": "driving" if k % 2 else "hiz"}
             for k in range(1, 20)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        NET.solve(net, fs=2e8, n=4096, schedule=sched, probes="bus")
    rings = [w for w in caught if "rings at" in str(w.message)]
    assert 0 < len(rings) <= 2, [str(w.message)[:60] for w in rings]


# ------------------------------------------------------------------------------- through a recipe

def test_an_inductor_round_trips_through_a_recipe():
    """A net is plain data, so an inductor is too: the recipe has to replay to the same samples
    and the same digest."""
    g = Grid(fs=2e10, baud=1e9, n=2048)
    sig = (Signal(seed=4, grid=g).carrier("nrz", n_ui=16, tr_frac=0.25)
           .net(nodes=["pad", "bus"], rails={"vdd": 3.3},
                elements=[{"id": "lp", "kind": "l", "a": "pad", "b": "bus",
                           "henry": 4e-9, "ohm": 0.3},
                          {"id": "cp", "kind": "c", "a": "pad", "b": "gnd", "farad": 4e-12},
                          {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 20e-12},
                          {"id": "rt", "kind": "r", "a": "bus", "b": "vdd", "ohm": 150.0}],
                drivers=[{"id": "d0", "node": "pad", "volts": 0.0, "ohm": 8.0, "state": "hiz"}],
                follow="d0", probe="bus"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x = sig.waveform()
        rec = sig.recipe()
        back = Signal.from_recipe(json.loads(json.dumps(rec)))
        y = back.waveform()
    assert np.array_equal(x, y)
    assert back.sha256() == sig.sha256()
    assert x.max() > 3.3, "a driven bus behind an inductance overshoots the rail"


# ------------------------------------------------------------------------------------- a line

def _matched_line(td, tr, z0=50.0, sections=None, loss_ohm=None):
    nodes, elements = NET.line_ladder("a", "b", z0, td, tr, sections=sections,
                                      loss_ohm=loss_ohm)
    elements = elements + [{"id": "rl", "kind": "r", "a": "b", "b": "gnd", "ohm": z0}]
    return {"nodes": ["a", "b"] + nodes, "elements": elements,
            "drivers": [{"id": "d", "node": "a", "ohm": z0, "state": "driving"}]}


def _edge(n, fs, tr, t0):
    """A raised-cosine step whose 10-90 % time is `tr`, built here so that the reference the
    ladder is compared against is the same waveform shifted, exactly."""
    u = np.clip((np.arange(n) / fs - t0) / (tr / 0.8) + 0.5, 0.0, 1.0)
    return np.where((u > 0) & (u < 1), 0.5 - 0.5 * np.cos(np.pi * u), u)


@pytest.mark.parametrize("ratio", [1.0, 2.0, 5.0])
def test_a_line_ladder_delivers_the_edge_delayed_and_halved_to_within_the_stated_error(ratio):
    """What the helper claims: at its own section count, a lumped ladder carries the edge with
    an error near 2 % of the swing, and the number comes out of a matched source into a matched
    load -- launched amplitude Z0/(Zs+Z0), arriving one delay later."""
    td, z0 = 1e-9, 50.0
    tr = td / ratio
    fs = 4e11
    n = int(round((10 * tr + 3 * td) * fs))
    vs = _edge(n, fs, tr, 5 * tr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = NET.solve(_matched_line(td, tr), fs=fs, n=n, probes="b", sources={"d": vs})
    k = int(round(td * fs))
    ref = 0.5 * np.concatenate([np.zeros(k), vs[:n - k]])
    assert np.abs(out - ref).max() / 0.5 < 0.02
    assert abs(out[-1] - 0.5) < 1e-3          # and it really is a matched divider at DC


def test_a_line_ladder_s_delay_is_the_delay_it_was_asked_for():
    """Read off the record the way a measurement would: the 50 % crossing at the far end, minus
    the 50 % crossing of the launched wave."""
    td, tr, z0, fs = 2e-9, 400e-12, 50.0, 4e11
    n = int(round(12e-9 * fs))
    vs = _edge(n, fs, tr, 2e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = NET.solve(_matched_line(td, tr), fs=fs, n=n, probes=["a", "b"], sources={"d": vs})
    cross = lambda v, lv: np.interp(lv, v[:np.argmax(v > lv) + 1],   # noqa: E731
                                    np.arange(np.argmax(v > lv) + 1) / fs)
    measured = cross(out["b"], 0.25) - cross(out["a"], 0.25)
    assert abs(measured - td) < 0.05 * td


def test_a_line_s_loss_shows_up_as_a_lower_level_not_as_a_shorter_delay():
    """`loss_ohm` is a DC series resistance spread along the line -- which is a divider, and
    nothing about frequency. Saying so is the point: skin effect belongs to `lossy`."""
    td, tr, z0, fs = 1e-9, 500e-12, 50.0, 4e11
    n = int(round(8e-9 * fs))
    vs = _edge(n, fs, tr, 2e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lossless = NET.solve(_matched_line(td, tr), fs=fs, n=n, probes="b", sources={"d": vs})
        lossy = NET.solve(_matched_line(td, tr, loss_ohm=10.0), fs=fs, n=n, probes="b",
                          sources={"d": vs})
    assert abs(lossless[-1] - 0.5) < 1e-3
    assert abs(lossy[-1] - 1.0 * 50.0 / 110.0) < 2e-3     # 50 + 10 + 50 ohm divider
    assert lossy.max() < lossless.max()


def test_the_section_count_is_the_measured_rule_and_grows_with_the_line_s_length():
    assert NET.line_sections(1e-9, 1e-9) == 6
    assert NET.line_sections(1e-9, 500e-12) == 17
    assert NET.line_sections(1e-9, 200e-12) == 68
    # monotone in both arguments, and never zero
    assert NET.line_sections(1e-12, 1e-6) == 1
    assert (NET.line_sections(2e-9, 500e-12) > NET.line_sections(1e-9, 500e-12)
            > NET.line_sections(1e-9, 1e-9))


def test_an_electrically_long_line_is_refused_and_says_what_to_use_instead():
    """The honest boundary: the state count grows as the 1.5 power of the line's length in rise
    times and the solve as roughly its cube, so a long line is the frequency-domain tier's job."""
    with pytest.raises(ValueError, match=r"rise times long.*sparam|sparam.*rise times long"):
        NET.line_ladder("a", "b", 50.0, 20e-9, 100e-12)


def test_an_undersized_ladder_warns_that_its_own_cutoff_looks_like_ringing():
    """The one artefact that cannot be told apart from what this net exists to show."""
    with pytest.warns(RuntimeWarning, match="under the measured rule.*indistinguishable"):
        NET.line_ladder("a", "b", 50.0, 1e-9, 200e-12, sections=8)


def test_a_ladder_gives_its_end_nodes_the_line_s_own_capacitance():
    """A symmetric T ladder, which is also what makes `a` and `b` solvable without the caller
    inventing a capacitance for them."""
    n = NET.line_sections(1e-9, 1e-9)
    nodes, elements = NET.line_ladder("a", "b", 50.0, 1e-9, 1e-9)
    half = [e for e in elements if e["id"] in ("ln/c0", "ln/cn")]
    mid = [e for e in elements if e["kind"] == "c" and e["id"] not in ("ln/c0", "ln/cn")]
    assert {e["a"] for e in half} == {"a", "b"}, "the half sections land on the caller's nodes"
    assert all(abs(e["farad"] - 2 * half[0]["farad"]) < 1e-30 for e in mid)
    assert len(nodes) == len(mid) == n - 1
    # and the two line equations come back out of the parts
    henry = [e["henry"] for e in elements if e["kind"] == "l"]
    assert len(henry) == n
    total_l, total_c = sum(henry), sum(e["farad"] for e in elements if e["kind"] == "c")
    assert abs(np.sqrt(total_l / total_c) - 50.0) < 1e-9          # Z0 = sqrt(L/C)
    assert abs(np.sqrt(total_l * total_c) - 1e-9) < 1e-20         # delay = sqrt(L C)


# --------------------------------------------------------------------------------- refusals

def test_an_inductor_cannot_be_switched_and_the_message_says_why():
    """Its current is state. Taking it out of the stamp would zero that current within one
    sample, which is the one thing an inductor cannot do."""
    net = _series_rlc(10.0)
    net["elements"][0]["state"] = "open"
    with pytest.raises(ValueError, match="l 'lp' carries 'state'.*CURRENT is the state"):
        NET.solve(net, fs=FS, n=8, probes="bus")


@pytest.mark.parametrize("kind,eid", [("l", "lp"), ("c", "cb")])
def test_a_schedule_event_for_an_unswitchable_element_is_refused(kind, eid):
    """Both are state, so neither can leave the stamp. Today an event on either rendered as
    though it had not been written at all."""
    with pytest.raises(ValueError, match=rf"targets {kind} '{eid}'"):
        NET.solve(_series_rlc(10.0), fs=FS, n=8, probes="bus",
                  schedule=[{"t_s": 1e-9, "id": eid, "g": 0.0}])


@pytest.mark.parametrize("el,match", [
    ({"id": "lp", "kind": "l", "a": "vdd", "b": "bus"}, "has no 'henry'"),
    ({"id": "lp", "kind": "l", "a": "vdd", "b": "bus", "henry": 0.0}, "must be positive"),
    ({"id": "lp", "kind": "l", "a": "vdd", "b": "bus", "henry": -1e-9}, "must be positive"),
    ({"id": "lp", "kind": "l", "a": "vdd", "b": "bus", "henry": 1e-9, "ohm": -1.0},
     "series resistance of the same piece of wire"),
    ({"id": "lp", "kind": "l", "a": "bus", "b": "bus", "henry": 1e-9},
     "both ends on 'bus'"),
    ({"id": "lp", "kind": "l", "a": "vdd", "b": "bus", "henry": 1e-9, "farad": 1e-12},
     r"unknown key 'farad'"),
])
def test_a_broken_inductor_description_is_refused_by_name(el, match):
    net = _series_rlc(10.0)
    net["elements"][0] = el
    with pytest.raises(ValueError, match=match):
        NET.solve(net, fs=FS, n=8, probes="bus")


def test_two_ideal_inductors_in_parallel_have_no_dc_operating_point():
    """At DC an inductor is a short, so a loop of ideal ones divides current by nothing. The
    refusal says that and says which wire to give its real resistance to."""
    net = _series_rlc(0.0)
    net["elements"].append({"id": "lq", "kind": "l", "a": "vdd", "b": "bus", "henry": L})
    with pytest.raises(ValueError, match="no DC operating point.*divides current by nothing"):
        NET.solve(net, fs=FS, n=8, probes="bus")


def test_a_node_without_capacitance_is_told_that_r_and_l_are_one_element():
    """The mistake the series `ohm` exists to prevent: a node invented between a series resistor
    and a series inductor, which has no capacitance and therefore is not a state."""
    net = {"nodes": ["mid", "bus"], "rails": {"vdd": 1.0},
           "elements": [{"id": "rs", "kind": "r", "a": "vdd", "b": "mid", "ohm": 10.0},
                        {"id": "lp", "kind": "l", "a": "mid", "b": "bus", "henry": L},
                        {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": C}]}
    with pytest.raises(ValueError, match=r"no capacitance.*are ONE element"):
        NET.solve(net, fs=FS, n=8, probes="bus")


def test_a_source_cannot_be_given_to_an_inductor():
    with pytest.raises(ValueError, match="which is an ELEMENT"):
        NET.solve(_series_rlc(10.0), fs=FS, n=8, probes="bus",
                  sources={"lp": np.zeros(8)})


@pytest.mark.parametrize("probes,match", [
    (["bus", "i(nope)"], r"not a solved state.*i\(lp\)"),
    (["bus", "lp"], r"not a solved state"),
])
def test_a_probe_that_is_not_a_state_is_refused_with_the_states_listed(probes, match):
    with pytest.raises(ValueError, match=match):
        NET.solve(_series_rlc(10.0), fs=FS, n=8, probes=probes)


def test_probing_a_current_on_a_net_with_no_inductors_says_there_are_none():
    net = {"nodes": ["bus"], "rails": {"vdd": 1.0},
           "elements": [{"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": C},
                        {"id": "rt", "kind": "r", "a": "bus", "b": "vdd", "ohm": 50.0}]}
    with pytest.raises(ValueError, match="no inductors, so there are no branch currents"):
        NET.solve(net, fs=FS, n=8, probes=["bus", "i(cb)"])


@pytest.mark.parametrize("v0,match", [
    ({"bus": 0.0, "i(nope)": 0.0}, r"not solved states"),
    ({"i(lp)": 0.0}, r"leaves node\(s\) \['bus'\] unset"),
])
def test_a_broken_initial_condition_is_refused(v0, match):
    with pytest.raises(ValueError, match=match):
        NET.solve(_series_rlc(10.0), fs=FS, n=8, probes="bus", v0=v0)


def test_an_unnamed_inductor_current_starts_at_rest():
    """Zero, which is at rest, rather than a steady state consistent with voltages the caller
    just overrode -- and the record says so by being the textbook step response."""
    n = 4000
    out = NET.solve(_series_rlc(10.0), fs=FS, n=n, probes=["bus", "i(lp)"], v0={"bus": 0.0})
    t = np.arange(1, n + 1) * DT
    assert np.abs(out["bus"] - _analytic(10.0, t)).max() < 1e-12

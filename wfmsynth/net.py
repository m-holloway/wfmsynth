"""
wfmsynth.net — a shared net: several drivers, switched terminations, one solve.

A channel in this library is otherwise a SERIES cascade with one source and one probe. A real
shared net is not that. Several agents drive it at different times and hand over through hi-Z;
terminations and pull-ups switch in and out on a schedule; a weaker driver holds a level against
a stronger one. None of that is expressible as a 2-port chain, and it is where open-drain buses,
turnaround and contention live.

WHAT A NET IS HERE
------------------
Nodes, elements between them, drivers onto them, and a schedule that switches things:

    net = {
      "nodes": ["bus"],                               # solved; "gnd" is the implicit reference
      "rails": {"vdd": 3.3},                          # ideal fixed voltages, never solved
      "elements": [
        {"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": 885.0},
        {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 400e-12},
      ],
      "drivers": [
        {"id": "d0", "node": "bus", "volts": 0.0, "ohm": 20.0, "state": "hiz"},
      ],
    }
    schedule = [{"t_s": 1e-6, "id": "d0", "state": "driving"}, ...]

An event says `state` (in or out), `g` (its conductance in siemens, where zero IS an open), or
`src` (a driver's Thevenin level in volts) -- any combination of the three. All of them leave
the network constant BETWEEN events, which is the one thing the closed form below needs. A
quantity that varies DURING an interval (`gate`, a turn-on slew; `relax`, settling toward
another strength) is refused by name rather than by silence, with what to do instead.

AN INDUCTOR, AND WHY A HAND-OVER RINGS
--------------------------------------
`{"kind": "l", "a": ..., "b": ..., "henry": ..., "ohm": ...}` is a package or trace inductance,
with `ohm` the series resistance of the same piece of wire (absent means ideal). It is what a
scope actually shows at a driver turning on or letting go: overshoot and a ring at a frequency
set by the inductance against the node capacitance, decaying at a rate set by the termination.
An RC net can only settle; it cannot overshoot, so a hand-over rendered without inductance is
the one place a synthetic record differs from a measured one by more than noise.

The series `ohm` is on the inductor rather than a separate `r` on purpose: the node BETWEEN a
series resistor and a series inductor has no capacitance, and this tier solves states (below),
so it would be refused. A bond wire is one piece of metal with both properties, so it is one
element with both numbers.

An inductor's CURRENT is a state variable, carried across every switch exactly as capacitor
charge is -- which is why it is not switchable and the schedule refuses it by name. Removing an
inductor from the stamp would take its current to zero within one sample, and that is the one
thing an inductor cannot do. Switch what is in SERIES with it (the driver, or an `r`): the
current then keeps flowing into the node capacitance, which is what rings on a real board, and
is why "every node has some capacitance" is a physical rule here rather than a solver's
convenience.

`solve(..., probes=["bus", "i(lp)"])` reads an inductor's current in AMPS, which is the only
way to see the state the solve now carries; `v0={"bus": 0.0, "i(lp)": 0.0}` sets it.

A DRIVER WHOSE LEVEL IS A WAVEFORM
----------------------------------
`sources={"d0": volts_per_sample}` gives a driver a per-sample Thevenin level instead of the
fixed `volts` in its description -- its own levels, its own edge rate, its own timing
impairments, one sample per record sample. On a shared net each party usually has a different
output stage and a different path to the node, and adding their contributions by hand outside
the solve is where a level held by a driver that has let go gets carried into the next owner's
baseline. Here it cannot: a released driver is out of the stamp, so its level is absent rather
than inherited, and what it was holding decays through the elements on the node.

This is still exact. Only `b` moves -- the network itself is constant between events, so the
eigenbasis is computed once per interval as before and the sampled source is carried through it
by the zero-order-hold recurrence in `_step_sources`, which is the analytic answer for a source
held across each sample (`physics.open_drain_line` holds its target the same way). An interval
whose sources happen not to move is routed back to the closed form, so a constant source array
reproduces a constant `volts` BIT-FOR-BIT rather than nearly.

The level is volts and the array is numbers, so it does not belong in the net description, which
has to stay JSON for a recipe. `Signal.net` carries each driver's `source` as a nested spec --
its own chain -- renders it, and passes the arrays here.

Several drivers may sit on one node. Each conducting one stamps its own conductance and its own
current, which is the parallel Thevenin combination, so a weaker driver holding a level against
a stronger one and a hand-over where both are briefly on are ordinary solves, not special cases.

Everything is plain data, so a net round-trips through a recipe and a schedule can be produced
by a protocol, traffic or fault generator without this module knowing about any of them.
`wfmsynth.scenario` is the layer that writes one from "who held the net when".

WHY THIS IS EXACT, AND WHY IT IS NOT A TIME-STEPPED INTEGRATION
---------------------------------------------------------------
Between two schedule events the network is CONSTANT, so the node voltages obey

    C v' = -G (v - v_ss),      v_ss = G^-1 b

which has a closed-form solution. Solving the symmetric generalised eigenproblem `G V = C V L`
once per configuration (both matrices are symmetric positive semi-definite for a passive RC
network) decouples it into independent first-order modes, so

    v(k dt) = v_ss + V ( exp(-L k dt) * z0 ),     z0 = V^T C (v(0) - v_ss)

is evaluated for every sample of the interval AT ONCE. That means:

  * no integration error -- this is the analytic solution, not a discretisation of it, so there
    is no step-size to choose and no accuracy that degrades with a longer record;
  * no guard time and no periodicity assumption, which a frequency-domain solve of a SWITCHED
    network needs, because switching makes it time-varying;
  * continuity across a switch is exact: the capacitor voltages at the end of one interval are
    the initial condition of the next, which is the physics and not a convention;
  * it is vectorised per interval rather than a per-sample Python loop.

`physics.open_drain_line` is the one-node case of exactly this recurrence, and
`tests/test_net.py` pins the two against each other.

WITH AN INDUCTOR IT IS STILL CLOSED-FORM, BY A DIFFERENT DECOMPOSITION
---------------------------------------------------------------------
An inductor adds its branch current to the state vector, and the resulting system matrix is no
longer symmetric -- the skew coupling between charge and flux is precisely what makes a mode
OSCILLATE rather than decay -- so `eigh` does not apply. Eigenvectors in general do not either:
at EXACTLY critical damping (`R = 2*sqrt(L/C)`, which is a damping network somebody designs on
purpose) the matrix is defective, its eigenvector matrix is singular, and a modal solve returns
hundreds of millivolts of nonsense. So the inductive path instead takes the exact zero-order-hold
pair for one sample (`Phi = expm(-W dt)` and its integral, both from one matrix exponential) and
back-substitutes them through a SCHUR form, whose basis is unitary and therefore well conditioned
for every damping. Each triangular row is then one first-order recurrence in `lfilter`, as in
`_step_sources`, with the rows below it already solved.

That is still the analytic answer, now accumulated one sample at a time rather than evaluated at
`k dt` in closed form, so it carries a rounding floor instead of none: measured 3.5e-14 V on a
1 V step for any mode that decays, and 8.4e-12 V over 262144 samples of a LOSSLESS tank, the
worst case there is, where nothing damps the accumulation. It buys exactness at critical damping
and it is identical in all three damping regimes. A net with no inductors does not take this
path at all and renders bit for bit as before.

WHAT THIS TIER DOES NOT DO, stated so nobody infers it
-------------------------------------------------------
Transmission lines and S-parameter blocks inside a net, sub-nets, voltage-dependent
(nonlinear) laws, and emitting driver/element intervals as events. A net whose elements are all
lumped R, L and C is solved exactly here; the frequency-domain tier earns its place when
distributed elements arrive, which is the case this one cannot represent at all.

A LINE between two nodes is reachable as a lumped ladder of the elements above --
`line_ladder()` builds one, sized by its own measured rule -- which is a real approximation with
a stated error rather than a distributed solve. It has a cutoff: a ladder passes an edge whose
rise time spans several sections and mangles one that does not, so the helper refuses a count
that cannot carry the edge you say you are sending.

A driver's SOURCE may be a waveform (above); its OUTPUT CONDUCTANCE still may not. A resistance
that varies within an interval is the one thing the closed form cannot absorb, and it is why
`gate` and `relax` are refused by name. An inductance is constant for the same reason.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import eigh, expm, schur
from scipy.signal import lfilter

GND = "gnd"

# A conducting element is in the network; an open one is removed entirely rather than given a
# large resistance. A "large resistance" is a number someone has to choose, it leaks current,
# and it puts a fast mode into the eigenproblem that has no physical counterpart.
_ON = ("closed", "driving", "on")
_OFF = ("open", "hiz", "off")

# What a schedule event may say. `state` switches a thing in or out; `g` sets its conductance in
# siemens (and therefore its state: a conductance of zero is an open); `src` sets a driver's
# Thevenin level in volts. All three leave the network piecewise-constant between events, which
# is what the closed-form solution needs.
_EVENT_KEYS = frozenset({"t_s", "id", "state", "g", "src", "_sample"})

# Named rather than lumped into "unknown key", because each is a real thing somebody will reach
# for and the reason it is absent is specific. Both describe a quantity that VARIES during an
# interval, which is exactly the assumption the closed form rests on.
_DEFERRED_KEYS = {
    "gate": "a turn-on slew is a conductance that varies DURING an interval, so the network is "
            "no longer constant between events and the closed form does not apply. Approximate "
            "it today with a short staircase of 'g' events -- which is a discretisation you "
            "chose and can see, not one the solver hid in a step size.",
    "relax": "settling toward another strength is the same thing: a time-varying element. The "
             "same staircase applies.",
}


# What an element and a driver description may say, per kind. A description is a dict, so a
# misspelled key would otherwise be accepted and do nothing: a driver written with `ohms=60`
# falls back to whatever the schedule set, and a capacitor written with a `state` is switched
# by nobody -- a capacitor is never removed from the stamp, because its charge is the state the
# solve carries across every switch. As in a recipe op, anything beginning with an underscore
# is annotation and is ignored.
_ELEMENT_KEYS = {"r": frozenset({"id", "kind", "a", "b", "ohm", "state"}),
                 "c": frozenset({"id", "kind", "a", "b", "farad"}),
                 "l": frozenset({"id", "kind", "a", "b", "henry", "ohm"})}

# Why a schedule cannot target one of these at all, said per kind rather than as "unknown id".
# Both are STATE: a capacitor's charge and an inductor's current are what the solve carries
# across a switch, so neither can be taken out of the stamp without a discontinuity. Today an
# event on either rendered as though it had not been written.
_UNSWITCHABLE = {
    "c": "a capacitor's CHARGE is the state this solve carries across every switch, so a "
         "capacitor is never removed from the stamp and there is nothing here for an event to "
         "set. Switch a resistor or a driver in series with it.",
    "l": "an inductor's CURRENT is the state this solve carries across every switch. 'g' is a "
         "conductance and an inductor has none, and switching it out would take its current to "
         "zero within one sample -- the one thing an inductor cannot do. Switch what is in "
         "SERIES with it (the driver, or an 'r'): the current then flows on into the node "
         "capacitance, which is what rings on a real board.",
}
_DRIVER_KEYS = frozenset({"id", "node", "volts", "ohm", "state"})


def _is_on(state):
    if state is None:
        return True
    s = str(state).lower()
    if s in _ON:
        return True
    if s in _OFF:
        return False
    raise ValueError(f"net: unknown state {state!r}; use one of {_ON + _OFF}")


# A time within this many samples of a sample boundary IS that boundary. A schedule is normally
# generated -- `t0 + i*dt` for bit i -- and `2.5e-6 * 1e8` is 250.00000000000003, so a bare
# `ceil` puts every other nominal edge one sample late depending on how the generator happened
# to accumulate. One sample of placement wobble that tracks float representation rather than
# the scenario is exactly the kind of difference nobody attributes correctly later.
_SNAP = 1e-9


def sample_at(t_s, fs):
    """The first sample index at or after `t_s`, snapped to a sample boundary it is within
    `_SNAP` of. `scenario.hold_windows` uses the same rule, so a hold's label window and the
    instant the solver actually switched are the same sample rather than neighbours."""
    k = float(t_s) * float(fs)
    return int(np.ceil(k - _SNAP * max(1.0, abs(k))))


def _conductance(e, ov, what):
    """Siemens for one element or driver: the schedule's override if it set one, otherwise
    `1/ohm` from the description. `ohm` is validated only where it is actually used, so a
    driver whose strength the schedule always sets need not carry a meaningful one."""
    if "g" in ov:
        g = float(ov["g"])
        if g < 0:
            raise ValueError(f"net: {what} {e['id']!r} was given g={g!r}; a conductance is "
                             f"non-negative. Use g=0 (or state='open') to disconnect it.")
        return g
    ohm = float(e["ohm"])
    if ohm <= 0:
        raise ValueError(f"net: {what} {e['id']!r} has ohm={ohm!r}; a resistance must be "
                         f"positive (use state='open'/'hiz' to disconnect it, or schedule g=0)")
    return 1.0 / ohm


def _check_inductor(e):
    """`henry` and the optional series `ohm`, checked where the description is read rather than
    where the matrix is stamped, so a net with an inductor it cannot solve is refused before any
    of it renders."""
    eid = e["id"]
    if "henry" not in e:
        raise ValueError(f"net: inductor {eid!r} has no 'henry'; an inductance in henries is "
                         f"what makes it an inductor (a few nanohenries for a bond wire, "
                         f"a few hundred picohenries per millimetre of trace)")
    h = float(e["henry"])
    if h <= 0:
        raise ValueError(f"net: inductor {eid!r} has henry={h!r}; an inductance must be "
                         f"positive. Zero henries is a wire: name both ends the same node "
                         f"instead, or give the wire the series 'ohm' it really has.")
    r = float(e.get("ohm", 0.0))
    if r < 0:
        raise ValueError(f"net: inductor {eid!r} has ohm={r!r}; on an inductor 'ohm' is the "
                         f"series resistance of the same piece of wire and is non-negative "
                         f"(absent or 0 means ideal). It is not a parallel resistor, and it "
                         f"is not switchable.")
    if e.get("a") == e.get("b"):
        raise ValueError(f"net: inductor {eid!r} has both ends on {e.get('a')!r}. A loop of "
                         f"inductance with nothing in it carries a current the solve cannot "
                         f"determine -- at DC an inductor is a short, so the loop divides "
                         f"current by nothing.")


def _check(net):
    """Validate the description and return (node_index, rails, ids). Every refusal names the
    thing it is refusing."""
    nodes = list(net.get("nodes", []))
    if not nodes:
        raise ValueError("net: 'nodes' is empty; there is nothing to solve for")
    if len(set(nodes)) != len(nodes):
        raise ValueError(f"net: duplicate node names in {nodes}")
    rails = dict(net.get("rails", {}))
    clash = sorted(set(nodes) & set(rails))
    if clash:
        raise ValueError(f"net: {clash} are listed as both solved nodes and fixed rails; a node "
                         f"is one or the other")
    if GND in nodes:
        raise ValueError(f"net: {GND!r} is the implicit 0 V reference and cannot be a solved node")
    known = set(nodes) | set(rails) | {GND}
    ids = {}
    for e, what in ([(e, "element") for e in net.get("elements", [])]
                    + [(d, "driver") for d in net.get("drivers", [])]):
        eid = e.get("id")
        if not eid:
            raise ValueError(f"net: every element and driver needs an 'id'; this one has none: {e}")
        if eid in ids:
            raise ValueError(f"net: duplicate id {eid!r}; ids address the schedule and must be unique")
        ids[eid] = e
        kind = None
        if what == "driver":
            if "node" not in e:
                raise ValueError(f"net: driver {eid!r} has no 'node'; a driver drives one node")
            ok = _DRIVER_KEYS
        else:
            if "kind" not in e:
                raise ValueError(f"net: element {eid!r} has no 'kind'; this tier has 'r', 'c' "
                                 f"and 'l'")
            kind = str(e["kind"]).lower()
            if kind not in _ELEMENT_KEYS:
                # Named HERE rather than when the element is first stamped: an element the tier
                # cannot represent is a broken description whether or not the schedule happens
                # to leave it open, and refusing it only when it conducts means a net with an
                # element parked open renders as though it were not there.
                raise ValueError(f"net: element {eid!r} has kind {e['kind']!r}; this tier "
                                 f"supports 'r', 'c' and 'l' only (see the module docstring "
                                 f"for what is deferred)")
            ok = _ELEMENT_KEYS[kind]
            if kind == "l":
                _check_inductor(e)
        for k in sorted(set(e) - ok):
            if k.startswith("_"):
                continue                     # annotation, as in a recipe op
            if k == "state" and kind in _UNSWITCHABLE:
                raise ValueError(f"net: {kind} {eid!r} carries 'state', but "
                                 f"{_UNSWITCHABLE[kind]}")
            if k == "source" and what == "driver":
                raise ValueError(
                    f"net: driver {eid!r} carries 'source' in its description. A per-sample "
                    f"level is an ARRAY of volts and the description has to stay JSON for a "
                    f"recipe, so pass it separately: solve(..., sources={{{eid!r}: volts}}). "
                    f"Through `Signal.net` a driver's `source` IS a nested spec and is rendered "
                    f"for you.")
            raise ValueError(f"net: {what} {eid!r} has unknown key {k!r}; it accepts {sorted(ok)}")
        for key in ("a", "b", "node"):
            if key in e and e[key] not in known:
                raise ValueError(f"net: {eid!r} references node {e[key]!r}, which is not a solved "
                                 f"node, a rail, or {GND!r}. Known: {sorted(known)}")
    return {nm: i for i, nm in enumerate(nodes)}, rails, ids


def _stamp(net, index, rails, state, over=None, sources=None):
    """Nodal G, C and b for ONE configuration, plus the per-sample injections.

    `state` maps id -> bool (conducting); `over` maps id -> the schedule's `g` / `src`
    overrides for it, empty when nothing set any. `sources` names the drivers whose level is a
    waveform: those contribute no constant term to `b` and are returned as
    `[(node_index, siemens, driver_id)]` for the caller to fold in sample by sample.

    A driver that is not conducting appears in neither, which is the whole point -- its level
    is absent from the solution rather than carried into it."""
    over = over or {}
    sources = sources or {}
    inj = []
    n = len(index)
    G, C, b = np.zeros((n, n)), np.zeros((n, n)), np.zeros(n)

    def add(mat, a, bn, g):
        ia, ib = index.get(a), index.get(bn)
        if ia is not None:
            mat[ia, ia] += g
        if ib is not None:
            mat[ib, ib] += g
        if ia is not None and ib is not None:
            mat[ia, ib] -= g
            mat[ib, ia] -= g

    for e in net.get("elements", []):
        eid, kind = e["id"], str(e["kind"]).lower()
        if kind == "c":
            add(C, e["a"], e["b"], float(e["farad"]))
            continue
        if kind == "l":
            # a branch CURRENT rather than a conductance, so it is not in G at all; `_inductors`
            # collects it once per solve, because an inductor is not switchable
            continue
        if not state.get(eid, True):
            continue
        ov = over.get(eid, {})
        if "src" in ov:
            raise ValueError(f"net: the schedule sets src={ov['src']!r} on element {eid!r}, but "
                             f"'src' is a DRIVER's Thevenin level and a resistor has none. "
                             f"Scheduling a resistor's strength is 'g'.")
        g = _conductance(e, ov, "element")
        if g == 0.0:
            continue
        add(G, e["a"], e["b"], g)
        # a rail on either end injects current into the solved end
        for near, far in ((e["a"], e["b"]), (e["b"], e["a"])):
            if near in index and far in rails:
                b[index[near]] += g * float(rails[far])

    # Several drivers on one node is the ordinary case, not a special one: each conducting one
    # adds its own conductance and its own current, which IS the parallel Thevenin combination.
    # That is what makes a weaker driver holding a level against a stronger one, and a hand-over
    # where both are briefly on, fall out of the solve rather than need a rule.
    for d in net.get("drivers", []):
        if not state.get(d["id"], True):
            continue
        ov = over.get(d["id"], {})
        g = _conductance(d, ov, "driver")
        if g == 0.0:
            continue
        i = index[d["node"]]
        G[i, i] += g
        # A hi-Z driver is removed from the stamp entirely, above -- so its level is not
        # "ignored", it is absent, and parking a released driver at some non-rest voltage
        # cannot leak into the solution.
        if d["id"] in sources:
            inj.append((i, g, d["id"]))      # its level is a waveform; folded in per sample
            continue
        b[i] += g * float(ov["src"] if "src" in ov else d["volts"])
    return G, C, b, inj


def _require_states(C, index):
    """Every node's voltage has to BE a state, which means every node needs capacitance."""
    # EXACT zero, not np.isclose. These are farads: a 400 pF bus capacitance is 4e-10, and
    # np.isclose's default atol of 1e-8 calls that zero -- which rejected the open-drain bus,
    # the very net this tier exists for. The matrix is stamped here, so a node untouched by any
    # capacitor has an exactly-zero row and there is no tolerance to get wrong.
    zero = np.flatnonzero(np.diag(C) == 0.0)
    if zero.size:
        names = [nm for nm, i in index.items() if i in zero]
        raise ValueError(
            f"net: node(s) {sorted(names)} have no capacitance, so their voltage is algebraically "
            f"determined rather than a state, and this tier solves states. Every real node has "
            f"some capacitance -- give it a small 'c' to gnd (a parasitic picofarad is usually "
            f"honest) -- or remove the node and wire through it. A series resistance and a "
            f"series inductance between the same two nodes are ONE element: give the 'l' an "
            f"'ohm' rather than inventing a node between them.")


def _modes(G, C, b, index):
    """(v_ss, eigenvalues, V, C) for one configuration -- the closed-form ingredients."""
    _require_states(C, index)
    if np.linalg.matrix_rank(G) < G.shape[0]:
        raise ValueError(
            "net: the conductance matrix is singular, so at least one node has no resistive path "
            "to a rail or to gnd in this configuration and its DC voltage is undefined. A node "
            "left floating by a schedule needs a leak resistor to somewhere.")
    v_ss = np.linalg.solve(G, b)
    lam, V = eigh(G, C)                 # G V = C V diag(lam); V^T C V = I
    return v_ss, lam, V, C


def _inductors(net, index, rails):
    """The inductor branches of a net, or None when it has none.

    `(ids, henry, series_ohm, A, e)`, where `A[node, k]` is +1 at inductor k's `a` end and -1 at
    its `b` end -- its current flows a -> b, so it LEAVES node a -- and `e[k]` carries the rail
    voltages at whichever of its ends are rails. An inductor is not switchable, so none of this
    depends on the schedule and it is built once per solve rather than once per interval."""
    ls = [e for e in net.get("elements", []) if str(e["kind"]).lower() == "l"]
    if not ls:
        return None
    A = np.zeros((len(index), len(ls)))
    e = np.zeros(len(ls))
    for k, el in enumerate(ls):
        for end, sign in ((el["a"], 1.0), (el["b"], -1.0)):
            if end in index:
                A[index[end], k] += sign
            elif end in rails:
                e[k] += sign * float(rails[end])
    return ([el["id"] for el in ls],
            np.array([float(el["henry"]) for el in ls], float),
            np.array([float(el.get("ohm", 0.0)) for el in ls], float), A, e)


def _rlc_system(G, C, b, ind):
    """`M x' = -K x + f` for one configuration of a net WITH inductors, where the state vector
    is `x = [node voltages; inductor currents]`.

    Nodal KCL gains the branch currents (`G v + A i = b`) and each inductor brings its own law
    (`L i' = v_a - v_b - R_s i`, which is `A^T v - R_s i + e`), so

        M = [[C, 0], [0, diag(henry)]]      K = [[G, A], [-A^T, diag(ohm)]]      f = [b, e]

    `M` is symmetric positive definite, which is the same statement as "every node has
    capacitance and every inductance is positive". `K` is NOT symmetric, and that asymmetry is
    the physics: it is the skew coupling between charge and flux, and it is what turns a pair of
    real decaying modes into a complex conjugate pair that rings."""
    ids, henry, rs, A, e = ind
    n, m = G.shape[0], len(ids)
    K = np.block([[G, A], [-A.T, np.diag(rs)]])
    M = np.block([[C, np.zeros((n, m))], [np.zeros((m, n)), np.diag(henry)]])
    return K, M, np.concatenate([b, e])


def _rlc_dc(K, f, index, ind):
    """The DC operating point of one configuration, and the refusal when it has none."""
    if np.linalg.matrix_rank(K) < K.shape[0]:
        raise ValueError(
            f"net: this configuration has no DC operating point, so there is nothing for the "
            f"transient to settle to. Either a node has no DC path to a rail or to {GND!r} -- a "
            f"node left floating by a schedule needs a leak resistor to somewhere -- or two or "
            f"more IDEAL inductors ({sorted(ind[0])}) sit in parallel or form a loop: at DC an "
            f"inductor is a short, so the loop divides current by nothing and there is no single "
            f"answer. Give one of them the series 'ohm' the wire really has.")
    return np.linalg.solve(K, f)


def _rlc_prepare(K, M, dt):
    """`(S, Q, Gam, lam)` for one configuration of a net with inductors: the exact zero-order-hold
    pair for one sample, in a basis that is well conditioned for every damping.

    `Phi = expm(-W dt)` with `W = M^-1 K` advances the state one sample, and
    `Gam = (integral of expm(-W u) du over one sample) M^-1` carries the forcing held across it.
    Both come out of ONE matrix exponential, by van Loan's identity

        expm([[-W dt, I dt], [0, 0]]) = [[Phi, integral], [0, I]]

    which also means `W` is never inverted: a configuration whose `K` is singular is refused by
    `_rlc_dc` for having no DC point, not by a linear algebra error here.

    The decomposition is a SCHUR form (`Phi = Q S Q^H`, `S` upper triangular, `Q` unitary) and
    not an eigendecomposition, which matters at exactly critical damping (`R = 2 sqrt(L/C)` --
    a damping network somebody designs on purpose). There `Phi` is defective: its eigenvalues
    coincide, its eigenvector matrix is singular, and a modal solve of the same circuit returns
    368 mV of nonsense on a 1 V step. `Q` is unitary for every matrix, so `cond(Q) == 1` and
    there is no regime where this degrades."""
    s = K.shape[0]
    W = np.linalg.solve(M, K)
    Z = np.zeros((s, s))
    E = expm(np.block([[-W * dt, np.eye(s) * dt], [Z, Z]]))
    Gam = np.linalg.solve(M.T, E[:s, s:].T).T          # (integral) @ inv(M), M symmetric
    S, Q = schur(E[:s, :s], output="complex")
    # the CONTINUOUS eigenvalues, for the resolution warning only: read off Phi they would come
    # back through a logarithm whose branch is exactly what aliasing destroys, so a ring too
    # fast for the grid would be reported at the wrong frequency -- the one thing the warning
    # exists to say.
    return S, Q, Gam, np.linalg.eigvals(W)


def _rlc_run(S, Q, Gam, F, x0, nsamp):
    """One interval of a net with inductors: `x[k+1] = Phi x[k] + Gam f[k]`, back-substituted
    through the Schur form so that each row is one scalar recurrence.

    In `y = Q^H x` the step is `y[k+1] = S y[k] + Q^H Gam f[k]` with `S` upper triangular, so
    row r depends only on rows BELOW it in the state vector (`r+1 ...`), which are already
    solved. Each row is then a first-order recurrence run by `lfilter` rather than by a Python
    loop over samples -- the same shape as `_step_sources`, one call per state instead of one
    per mode.

    `F` is the forcing, one column per sample, or a single column for an interval across which
    nothing moves (the ordinary case: it is broadcast). This is the analytic answer for a source
    held across each sample, accumulated sample by sample rather than evaluated at `k dt`, so it
    carries a rounding floor the RC path does not: measured 3.5e-14 V on a 1 V step for any mode
    that decays, and 8.4e-12 V over 262144 samples of a lossless tank, where nothing damps the
    accumulation at all."""
    s = S.shape[0]
    U = Q.conj().T @ (Gam @ F)                 # the forcing in the Schur basis
    y0 = Q.conj().T @ x0
    Y = np.empty((s, nsamp), complex)
    for r in range(s - 1, -1, -1):
        w = np.full(nsamp, U[r, 0]) if U.shape[1] == 1 else U[r].copy()
        if r + 1 < s:
            # `y_r[k+1]` takes the lower rows at sample k. Y holds samples 1..nsamp, so sample k
            # is y0 when k == 0 and Y[:, k-1] after that -- an off-by-one here is a quarter of a
            # percent of the swing and looks exactly like a plausible answer.
            w = w + S[r, r + 1:] @ np.concatenate([y0[r + 1:, None], Y[r + 1:, :-1]], axis=1)
        # zi holds S_rr * y_r[0], the previous sample's state: continuity across a switch is the
        # capacitor voltages AND the inductor currents carried forward, which is the physics.
        Y[r] = lfilter([1.0], [1.0, -S[r, r]], w, zi=np.array([S[r, r] * y0[r]]))[0]
    return np.real(Q @ Y).T


# A lumped ladder's own error, as a fraction of the swing, is set by how many sections an edge
# spans -- not by the section count alone, because phase error accumulates along the line. The
# measured rule is N = CEIL * (delay/rise time) ** EXP, which holds the error between 1.1 % and
# 1.9 % of the swing for a line half a rise time to ten rise times long;
# `tests/test_net_inductors.py` pins three points of that and `validate.py` one. Below the rule
# it degrades fast: about 5 % of the swing at half the count and 10 to 16 % at a quarter.
_LADDER_CEIL, _LADDER_EXP, _LADDER_MAX = 6.0, 1.5, 256


def line_sections(delay_s, tr_s):
    """How many L-C sections a lumped ladder needs to carry an edge of 10-90 % rise time `tr_s`
    down a line of one-way delay `delay_s`, holding the ladder's own error near 2 % of the
    swing. See `line_ladder`, which is where the number is used and the measurement is stated."""
    delay_s, tr_s = float(delay_s), float(tr_s)
    if delay_s <= 0 or tr_s <= 0:
        raise ValueError(f"net: a line needs a positive delay and a positive rise time; got "
                         f"delay_s={delay_s!r} tr_s={tr_s!r}")
    return max(1, int(np.ceil(_LADDER_CEIL * (delay_s / tr_s) ** _LADDER_EXP)))


def line_ladder(a, b, z0_ohm, delay_s, tr_s, sections=None, loss_ohm=None, prefix="ln"):
    """A lumped L-C ladder standing in for a line of impedance `z0_ohm` and one-way delay
    `delay_s` between nodes `a` and `b`. Returns ``(nodes, elements)`` to splice into a net.

    This is an APPROXIMATION, which the rest of this module is not, and the approximation is a
    cutoff: a ladder is a low-pass filter, so it carries an edge that spans several sections and
    mangles one that does not -- as its own overshoot, which is the one artefact indistinguishable
    from the inductive ringing the net is being built to show. So the edge is an argument rather
    than an afterthought. `tr_s` is the 10-90 % rise time of the fastest edge you intend to send
    down it, and the default section count comes from it:

        sections = ceil(6 * (delay_s / tr_s) ** 1.5)

    which is MEASURED, not a rule of thumb: against a matched source and load it holds the
    ladder's error between 1.1 % and 1.9 % of the swing for lines from half a rise time to ten
    rise times long. Below it the error grows quickly -- about 5 % of the swing at half the
    count, 10 to 16 % at a quarter -- so a smaller `sections` warns rather than renders
    quietly.

    The cost of that accuracy is states: the count grows as the 1.5 power of the line's length in
    rise times, and the per-interval solve as roughly its cube (0.07 s at five rise times, 0.45 s
    at ten, each time the schedule switches). Past `_LADDER_MAX` sections it is refused, because
    an electrically LONG line is what the frequency-domain ops are for -- `lossy`, `sparam`,
    `cascade` and `reflect` represent a line as a line, exactly and at any length, and the only
    thing they cannot do is sit inside a switched net.

    `loss_ohm` is the line's total series DC resistance, spread evenly along it. It is a DC
    resistance and nothing more: skin effect and dielectric loss are frequency-dependent and
    belong to `lossy`, which models them properly.

    The line's own capacitance appears as a half section at each end, so `a` and `b` come out
    with capacitance whether or not the caller gave them any -- which is also what makes them
    solvable nodes here."""
    z0, td = float(z0_ohm), float(delay_s)
    if z0 <= 0:
        raise ValueError(f"net: a line needs a positive impedance; got z0_ohm={z0_ohm!r}")
    rule = line_sections(td, tr_s)
    n = rule if sections is None else int(sections)
    if n < 1:
        raise ValueError(f"net: a ladder needs at least one section; got sections={sections!r}")
    if n > _LADDER_MAX:
        raise ValueError(
            f"net: this line is {td / float(tr_s):.3g} rise times long, so carrying its edge "
            f"takes {rule} ladder sections ({2 * rule} more states) and more than "
            f"{_LADDER_MAX} is refused: the solve cost grows as roughly the cube of the state "
            f"count, per switch. A line this long is what the frequency-domain ops are for -- "
            f"`lossy`, `sparam`, `cascade`, `reflect` -- which represent it exactly at any "
            f"length. Pass `sections=` explicitly if you mean to accept a coarser ladder.")
    if sections is not None and n < rule:
        warnings.warn(
            f"net: a {n}-section ladder for a line {td / float(tr_s):.3g} rise times long is "
            f"under the measured rule of {rule}. The ladder's own cutoff then shows as overshoot "
            f"and a softened edge -- about 5 % of the swing at half the rule's count and 10 to "
            f"16 % at a quarter -- and that overshoot is indistinguishable from the circuit's "
            f"own ring, which is what this net is for.", RuntimeWarning, stacklevel=2)
    # L and C per section from the two line equations: Z0 = sqrt(L/C), delay = sqrt(L*C).
    henry, farad = z0 * td / n, td / (z0 * n)
    ohm = (float(loss_ohm) / n) if loss_ohm else None
    nodes = [f"{prefix}/{k}" for k in range(1, n)]
    elements = []
    for k in range(n):
        el = {"id": f"{prefix}/l{k}", "kind": "l",
              "a": a if k == 0 else f"{prefix}/{k}",
              "b": b if k == n - 1 else f"{prefix}/{k + 1}", "henry": henry}
        if ohm is not None:
            el["ohm"] = ohm
        elements.append(el)
    # half a section's capacitance at each end -- a symmetric T ladder, and the reason `a` and
    # `b` need no capacitance of their own
    elements.append({"id": f"{prefix}/c0", "kind": "c", "a": a, "b": GND, "farad": farad / 2})
    elements.append({"id": f"{prefix}/cn", "kind": "c", "a": b, "b": GND, "farad": farad / 2})
    elements += [{"id": f"{prefix}/c{k}", "kind": "c", "a": f"{prefix}/{k}", "b": GND,
                  "farad": farad} for k in range(1, n)]
    return nodes, elements


def schedule_from_waveform(x, driver_id, on_when="high"):
    """Turn a waveform into schedule events for one driver, by thresholding at the MIDPOINT of
    its own excursion -- the same rule `physics.open_drain_line` applies to decide when a device
    sinks, so a net carrying one such driver reproduces that op.

    A waveform that never crosses its own midpoint (a DC level, any level) produces no events
    and the driver keeps its declared state. That is the open-drain behaviour too: a constant
    carrier does not drive the bus."""
    x = np.asarray(x, float)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return []
    on = x >= 0.5 * (lo + hi)
    if on_when != "high":
        on = ~on
    edges = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    out = [{"t_s": 0.0, "id": driver_id, "state": "driving" if on[0] else "hiz"}]
    out += [{"t_s": float(i), "id": driver_id, "state": "driving" if on[i] else "hiz",
             "_sample": int(i)} for i in edges]
    return out


def _check_event(ev, ids):
    """One schedule event, before anything is solved. A misspelled key would otherwise do
    nothing at all: the record renders, the digest is stable, and the switch never happened."""
    if "id" not in ev or "t_s" not in ev:
        raise ValueError(f"net: a schedule event needs 't_s' and 'id'; got {ev!r}")
    if ev["id"] not in ids:
        raise ValueError(f"net: schedule event at t={ev['t_s']} targets id {ev['id']!r}, which "
                         f"is not an element or driver. Known ids: {sorted(ids)}")
    kind = str(ids[ev["id"]].get("kind", "")).lower()
    if kind in _UNSWITCHABLE:
        raise ValueError(f"net: schedule event at t={ev['t_s']} targets {kind} {ev['id']!r}, but "
                         f"{_UNSWITCHABLE[kind]}")
    for k in sorted(set(ev) - _EVENT_KEYS):
        if k in _DEFERRED_KEYS:
            raise ValueError(f"net: event for {ev['id']!r} uses {k!r}, which this tier does not "
                             f"have: {_DEFERRED_KEYS[k]}")
        raise ValueError(f"net: event for {ev['id']!r} has unknown key {k!r}; the vocabulary is "
                         f"{sorted(_EVENT_KEYS - {'_sample'})}")
    if not set(ev) & {"state", "g", "src"}:
        raise ValueError(f"net: event for {ev['id']!r} at t={ev['t_s']} says nothing -- give it "
                         f"a 'state', a 'g' (siemens) or a 'src' (volts)")
    if "state" in ev and "g" in ev and _is_on(ev["state"]) != (float(ev["g"]) > 0.0):
        raise ValueError(f"net: event for {ev['id']!r} says state={ev['state']!r} and "
                         f"g={ev['g']!r}, which disagree. A conductance of zero IS an open; "
                         f"give one or the other.")


def _apply_event(ev, state, over):
    """Fold one event into the running configuration. `g` sets the strength AND the state,
    because a conductance of zero is an open and carrying both separately lets them disagree."""
    eid = ev["id"]
    if "g" in ev:
        g = float(ev["g"])
        if g < 0:
            raise ValueError(f"net: event for {eid!r} has g={g!r}; a conductance is non-negative")
        over.setdefault(eid, {})["g"] = g
        state[eid] = g > 0.0
    if "state" in ev:
        state[eid] = _is_on(ev["state"])
    if "src" in ev:
        over.setdefault(eid, {})["src"] = float(ev["src"])


def _step_sources(lam, V, Cm, cur, B, dt):
    """One interval whose network is constant but whose driver LEVELS move: the exact
    zero-order-hold solution, evaluated in the modal basis.

    Modal coordinates (v = V z, with V^T C V = I and V^T G V = diag(lam)) turn `C v' = -G v + b`
    into independent scalars `z' = -lam z + f`, where `f = V^T b`. Holding `f` across one sample
    -- which is what a SAMPLED source is, and the same hold `physics.open_drain_line` applies to
    its target -- integrates exactly to

        z[k+1] = exp(-lam dt) z[k] + (1 - exp(-lam dt))/lam * f[k]

    one first-order recurrence per mode, run by `lfilter` rather than by a Python loop over
    samples. There is no integration error and no step size to choose: `dt` is the grid, and the
    answer is the analytic response to the source the grid actually carries.

    For a CONSTANT `f` this is algebraically the closed form used for the rest of the module --
    same fixed point `f/lam`, same decay `exp(-lam dt)` per step -- so the two agree to rounding.
    `solve` routes a constant interval to the closed form anyway, so the reduction to tier 1 is
    bit-exact rather than merely close.
    """
    a = np.exp(-lam * dt)
    # -expm1(-x)/x rather than (1 - exp(-x))/x: at lam*dt around 1e-9 that subtraction throws
    # away most of the mantissa, and this gain is what carries the source into the answer.
    # lam is positive here (`_modes` refuses a singular G), so the guard is only against an
    # exact zero arriving from a degenerate configuration.
    safe = np.where(lam > 0.0, lam, 1.0)
    gain = np.where(lam > 0.0, -np.expm1(-lam * dt) / safe, dt)
    F = V.T @ B                                  # modal forcing, one column per sample
    z0 = V.T @ (Cm @ cur)                        # modal state at the interval's first instant
    Z = np.empty_like(F)
    for m in range(lam.size):
        # zi is the filter's stored state, which for a=[1, -alpha] holds alpha*y[-1]: the
        # previous interval's final voltage, so continuity across a switch is still the
        # capacitor voltages carried forward.
        Z[m] = lfilter([gain[m]], [1.0, -a[m]], F[m], zi=np.array([a[m] * z0[m]]))[0]
    return (V @ Z).T


def _warn_unresolved(lam, dt, seg, warned):
    """Name a mode the grid cannot show. The solve is EXACT either way -- this is the analytic
    solution sampled, not an integration -- but a transient that completes inside one sample is
    not visible in the record, and a rise time or slew measured off it would be the grid's.
    `physics.open_drain_line` gives the same warning for the same reason; this is its
    generalisation, with the time constants read straight off the eigenvalues.

    A RINGING mode (a complex eigenvalue, which only an inductor produces) needs the second
    check, and it is the more dangerous of the two: a decay the grid cannot show is simply
    missing from the record, but a ring the grid cannot show ALIASES, and comes back as a
    slower ring that looks entirely plausible and is not the circuit's. The library's own
    sizing rule is k >= 8 samples across the fastest feature, so that is the line here too,
    with the message saying which side of Nyquist the mode actually fell.

    Warned once per distinct time constant and once per distinct frequency, not once per
    interval: a bus that switches a thousand times has one under-resolved edge, not a thousand."""
    lam = np.asarray(lam)
    if np.iscomplexobj(lam):
        for f0 in np.unique(np.abs(lam.imag)[np.abs(lam.imag) > 0] / (2.0 * np.pi)):
            per_dt = 1.0 / (f0 * dt)                      # samples per ring period
            if per_dt >= 8.0:
                continue
            key = ("ring", round(float(f0), 6))
            if key in warned:
                continue
            warned.add(key)
            aliased = ("" if per_dt > 2.0 else
                       " It is past Nyquist, so the ring IN THE RECORD is an alias at some "
                       "other frequency entirely, not this one.")
            warnings.warn(
                f"net: an inductive mode rings at {f0:.4g} Hz, which this grid samples "
                f"{per_dt:.3g} times per period (dt = {dt:.4g} s). Overshoot and ring frequency "
                f"read off this record are the grid's below about 8 samples per period.{aliased} "
                f"Raise fs above {8.0 * f0:.4g} Sa/s, or damp the mode (more series resistance, "
                f"less inductance).",
                RuntimeWarning, stacklevel=3)
    tau = 1.0 / lam.real[lam.real > 0]
    for t in tau[tau < 2.0 * dt]:
        key = ("tau", round(float(t), 15))
        if key in warned:
            continue
        warned.add(key)
        warnings.warn(
            f"net: a mode with time constant {t:.4g} s is faster than this grid can show "
            f"(dt = {dt:.4g} s). The levels and the solve are exact -- it is the analytic "
            f"solution -- but that transient completes inside a sample, so a rise time or slew "
            f"measured off this record would be the grid's. Raise fs above {2.0 / t:.4g} Sa/s, "
            f"or slow the mode (a larger R or C on the node that switches).",
            RuntimeWarning, stacklevel=3)


def _prepare_sources(sources, ids, n):
    """Validate `sources` and return `{driver_id: float array of length n}`."""
    out = {}
    for did, arr in dict(sources or {}).items():
        did = str(did)
        if did not in ids:
            raise ValueError(f"net: sources names {did!r}, which is not an element or driver. "
                             f"Known ids: {sorted(ids)}")
        d = ids[did]
        if "node" not in d:
            raise ValueError(f"net: sources names {did!r}, which is an ELEMENT. A per-sample "
                             f"level is a DRIVER's Thevenin source; a resistor has none, and "
                             f"scheduling a resistor's strength is 'g'.")
        if "volts" in d:
            raise ValueError(f"net: driver {did!r} has both volts={d['volts']!r} and a rendered "
                             f"per-sample source. Both are its Thevenin level, so one would be "
                             f"silently ignored -- drop `volts`, the source is the level.")
        a = np.asarray(arr, dtype=float)
        if a.shape != (n,):
            raise ValueError(f"net: the source for {did!r} has shape {a.shape}, but this record "
                             f"is {n} samples. A driver's level is a waveform ON THIS GRID, one "
                             f"sample per record sample -- not a resampled or truncated one.")
        if not np.all(np.isfinite(a)):
            bad = int(np.flatnonzero(~np.isfinite(a))[0])
            raise ValueError(f"net: the source for {did!r} is not finite at sample {bad} "
                             f"({a[bad]!r}); a Thevenin level has to be a number of volts.")
        out[did] = a
    return out


def _state_columns(index, ind):
    """state name -> column. Node voltages in volts, then `i(<id>)` for each inductor in amps."""
    cols = dict(index)
    if ind is not None:
        for k, lid in enumerate(ind[0]):
            cols[f"i({lid})"] = len(index) + k
    return cols


def _initial(v0, index, ind):
    """The state vector `v0` asks for, or None. Every node has to be named -- a partial initial
    condition is a steady state for the nodes left out, mixed with an arbitrary one for the rest,
    and nothing about the record would say so. An inductor current not named starts at ZERO,
    which is at rest; it is named `i(<id>)`, as in `probes`."""
    if v0 is None:
        return None
    v0 = {str(k): v for k, v in dict(v0).items()}
    cols = _state_columns(index, ind)
    unknown = sorted(set(v0) - set(cols))
    if unknown:
        raise ValueError(f"net: v0 names {unknown}, which are not solved states. An initial "
                         f"condition is a node voltage in volts, or an inductor current in amps "
                         f"as 'i(<id>)'. This net has {sorted(cols)}.")
    missing = sorted(set(index) - set(v0))
    if missing:
        raise ValueError(f"net: v0 leaves node(s) {missing} unset. Give every node a voltage: a "
                         f"partial initial condition is the steady state for the nodes left out "
                         f"and something else for the rest, and nothing in the record says which "
                         f"is which. Inductor currents default to zero, which is at rest.")
    x0 = np.zeros(len(cols))
    for k, i in cols.items():
        if k in v0:
            x0[i] = float(v0[k])
    return x0


def solve(net, fs, n, schedule=(), probes=None, v0=None, sources=None):
    """Solve a switched lumped net on a uniform grid. Returns ``{node: array}``, or one array
    when `probes` names a single node. A probe named ``i(<inductor id>)`` reads that inductor's
    current in AMPS instead of a node voltage -- the state an inductive net carries across every
    switch, and the only way to see it.

    `fs` samples per second, `n` samples. Events landing on the same sample are applied as ONE
    configuration change, not in sequence, so a hand-over written as two events never passes
    through a half-state. A schedule event takes effect at the first sample
    whose time is at or after its `t_s`, and that sample is the FIRST one reflecting it: a
    sample records the voltage at the end of its own step. That is `physics.open_drain_line`'s
    convention, which makes the one-node case bit-for-bit (pinned in `tests/test_net.py`), and
    it means ``v[0]`` is one step past the initial condition rather than the initial condition.

    `sources` maps a driver id to a per-sample array of VOLTS, length `n`: that driver's
    Thevenin level is a waveform -- its own levels and its own edge rate -- rather than the
    fixed `volts` in its description, which it must then not carry. Still exact: only `b` moves,
    so the eigenbasis is computed once per interval as before and the sampled source is carried
    through it by a zero-order-hold recurrence (`_step_sources`). An interval across which no
    conducting source actually moves is routed to the closed form, so a constant source array
    reproduces a constant `volts` bit-for-bit.

    The initial condition is the DC steady state of whatever configuration is in force at t=0,
    unless `v0` gives node voltages -- with a source, the steady state of its FIRST sample, so
    a record does not open on an edge nobody asked for. `v0` must name every node; an inductor
    current it does not name (as ``i(<id>)``) starts at zero, which is at rest.

    An `{"kind": "l", ...}` element puts its branch current into the state vector and the solve
    takes its inductive path, which is closed-form in the same sense and exact at every damping
    (see the module docstring). A net with no inductors is untouched by it and renders bit for
    bit as before.
    """
    index, rails, ids = _check(net)
    fs, n = float(fs), int(n)
    if fs <= 0 or n <= 0:
        raise ValueError(f"net: fs and n must be positive, got fs={fs!r} n={n!r}")
    dt = 1.0 / fs

    ind = _inductors(net, index, rails)
    cols = _state_columns(index, ind)
    nst = len(cols)

    state = {}
    for e in list(net.get("elements", [])) + list(net.get("drivers", [])):
        if str(e.get("kind", "")).lower() not in _UNSWITCHABLE:
            state[e["id"]] = _is_on(e.get("state"))

    src = _prepare_sources(sources, ids, n)
    for ev in schedule or ():
        _check_event(ev, ids)
        if "src" in ev and ev["id"] in src:
            raise ValueError(
                f"net: driver {ev['id']!r} has a rendered per-sample source AND the schedule "
                f"sets src={ev['src']!r} on it at t={ev['t_s']}. Both are its Thevenin level. "
                f"Schedule its CONDUCTANCE instead ('g', or a state): a hand-over is a driver "
                f"leaving the stamp, not a driver changing level.")
    events = sorted(schedule or (), key=lambda ev: (ev.get("_sample", float(ev["t_s"]) * fs)))

    def _at(ev):
        # `_sample` is an exact sample index, used by waveform-derived schedules so that a
        # threshold crossing lands on the sample it crossed at rather than on a rounded time.
        k = ev["_sample"] if "_sample" in ev else sample_at(ev["t_s"], fs)
        return min(n, max(0, int(k)))

    starts = [0] + [_at(ev) for ev in events] + [n]

    x_init = _initial(v0, index, ind)
    v = np.zeros((n, nst))
    cur, warned, over = None, set(), {}
    for seg in range(len(starts) - 1):
        a, bnd = starts[seg], starts[seg + 1]
        if seg > 0:
            # Entering segment `seg` applies event seg-1. Events landing on the SAME sample
            # therefore all fold into the configuration before anything is solved -- the
            # `continue` below skips the empty interval but not the state change. That matters:
            # applied one at a time with a solve between, a hand-over would pass through a
            # moment where both drivers are off (or both on), and the record would carry a
            # transient worth a good fraction of the swing that no event asked for.
            _apply_event(events[seg - 1], state, over)
        if bnd <= a:
            continue                             # two events at the same sample: no interval
        G, C, b, inj = _stamp(net, index, rails, state, over, src)
        # The conducting source drivers' contribution to b, one column per sample. Their
        # conductance is part of G and is constant across the interval; only the level moves.
        S = None
        if inj:
            S = np.zeros((len(index), bnd - a))
            for i, g, did in inj:
                S[i] += g * src[did][a:bnd]
            if not np.any(S[:, 1:] != S[:, :1]):
                # nothing actually moves here -- a held level, an idle interval, or a source
                # that is constant throughout. Fold it into b and take the closed form, which
                # is what makes a constant source EQUAL a constant `volts` rather than round
                # to it. The comparison is exact on purpose: constant means identical bits.
                b, S = b + S[:, 0], None
        if ind is not None:
            # an inductor's branch current is one more state, and the system is no longer
            # symmetric, so this takes the zero-order-hold/Schur path instead of the modal
            # closed form. The RC net above is untouched by it, down to the last bit.
            _require_states(C, index)
            K, M, f = _rlc_system(G, C, b if S is None else b + S[:, 0], ind)
            Sch, Q, Gam, lam = _rlc_prepare(K, M, dt)
            _warn_unresolved(lam, dt, seg, warned)
            x_ss = _rlc_dc(K, f, index, ind)
            if cur is None:
                cur = x_ss if x_init is None else x_init
            F = f[:, None] if S is None else np.vstack(
                [b[:, None] + S, np.broadcast_to(ind[4][:, None], (len(ind[0]), bnd - a))])
            seg_v = _rlc_run(Sch, Q, Gam, F, cur, bnd - a)
            v[a:bnd] = seg_v
            cur = seg_v[-1]
            continue
        v_ss, lam, V, Cm = _modes(G, C, b if S is None else b + S[:, 0], index)
        _warn_unresolved(lam, dt, seg, warned)
        if cur is None:
            cur = x_init if x_init is not None else v_ss.copy()
        if S is None:
            # k starts at ONE: sample k records the voltage at the END of its own step, so a
            # schedule event is first visible on the sample it fires at. That is the convention
            # `physics.open_drain_line` already uses -- it advances then records -- and matching
            # it makes the one-node case bit-for-bit rather than off by one. It also means v[0]
            # is one step past the initial condition, not the initial condition itself.
            k = np.arange(1, bnd - a + 1)[:, None]
            z0 = V.T @ (Cm @ (cur - v_ss))
            seg_v = v_ss + (np.exp(-lam[None, :] * (k * dt)) * z0[None, :]) @ V.T
        else:
            seg_v = _step_sources(lam, V, Cm, cur, b[:, None] + S, dt)
        v[a:bnd] = seg_v
        # the state handed to the next configuration is this one evaluated one sample PAST its
        # last, which is the capacitor voltage at the switching instant. Continuity across a
        # switch is the physics, not a convention.
        cur = seg_v[-1]
    want = list(index) if probes is None else ([probes] if isinstance(probes, str) else list(probes))
    for p in want:
        if p not in cols:
            extra = ("" if ind is not None else
                     " This net has no inductors, so there are no branch currents to probe.")
            raise ValueError(f"net: probe {p!r} is not a solved state; have {sorted(cols)}. A "
                             f"node name reads volts and 'i(<inductor id>)' reads amps.{extra}")
    out = {p: v[:, cols[p]].copy() for p in want}
    return out[want[0]] if isinstance(probes, str) else out

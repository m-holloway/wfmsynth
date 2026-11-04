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

WHAT THIS TIER DOES NOT DO, stated so nobody infers it
-------------------------------------------------------
Transmission lines and S-parameter blocks inside a net, sub-nets, inductors, voltage-dependent
(nonlinear) laws, and emitting driver/element intervals as events. A net whose elements are all
lumped R and C is solved exactly here; the frequency-domain tier earns its place when
distributed elements arrive, which is the case this one cannot represent at all.

A driver's SOURCE may be a waveform (above); its OUTPUT CONDUCTANCE still may not. A resistance
that varies within an interval is the one thing the closed form cannot absorb, and it is why
`gate` and `relax` are refused by name.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import eigh
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
                 "c": frozenset({"id", "kind", "a", "b", "farad"})}
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
        if what == "driver":
            if "node" not in e:
                raise ValueError(f"net: driver {eid!r} has no 'node'; a driver drives one node")
            # an unknown kind is named by `_stamp`, which says what this tier supports, so the
            # key check falls back to the union rather than pre-empting it with a worse message
            ok = _DRIVER_KEYS
        else:
            if "kind" not in e:
                raise ValueError(f"net: element {eid!r} has no 'kind'; this tier has 'r' and 'c'")
            kind = str(e["kind"]).lower()
            if kind not in _ELEMENT_KEYS:
                # Named HERE rather than when the element is first stamped: an element the tier
                # cannot represent is a broken description whether or not the schedule happens
                # to leave it open, and refusing it only when it conducts means a net with an
                # inductor parked open renders as though the inductor were not there.
                raise ValueError(f"net: element {eid!r} has kind {e['kind']!r}; this tier "
                                 f"supports 'r' and 'c' only (see the module docstring for "
                                 f"what is deferred)")
            ok = _ELEMENT_KEYS[kind]
        for k in sorted(set(e) - ok):
            if k.startswith("_"):
                continue                     # annotation, as in a recipe op
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


def _modes(G, C, b, index):
    """(v_ss, eigenvalues, V, C) for one configuration -- the closed-form ingredients."""
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
            f"honest) -- or remove the node and wire through it.")
    if np.linalg.matrix_rank(G) < G.shape[0]:
        raise ValueError(
            "net: the conductance matrix is singular, so at least one node has no resistive path "
            "to a rail or to gnd in this configuration and its DC voltage is undefined. A node "
            "left floating by a schedule needs a leak resistor to somewhere.")
    v_ss = np.linalg.solve(G, b)
    lam, V = eigh(G, C)                 # G V = C V diag(lam); V^T C V = I
    return v_ss, lam, V, C


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

    Warned once per distinct time constant, not once per interval: a bus that switches a
    thousand times has one under-resolved edge, not a thousand."""
    tau = 1.0 / lam[lam > 0]
    for t in tau[tau < 2.0 * dt]:
        key = round(float(t), 15)
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


def solve(net, fs, n, schedule=(), probes=None, v0=None, sources=None):
    """Solve a switched lumped net on a uniform grid. Returns ``{node: array}``, or one array
    when `probes` names a single node.

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
    a record does not open on an edge nobody asked for.
    """
    index, rails, ids = _check(net)
    fs, n = float(fs), int(n)
    if fs <= 0 or n <= 0:
        raise ValueError(f"net: fs and n must be positive, got fs={fs!r} n={n!r}")
    dt = 1.0 / fs

    state = {}
    for e in list(net.get("elements", [])) + list(net.get("drivers", [])):
        if str(e.get("kind", "")).lower() != "c":
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

    v = np.zeros((n, len(index)))
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
        v_ss, lam, V, Cm = _modes(G, C, b if S is None else b + S[:, 0], index)
        _warn_unresolved(lam, dt, seg, warned)
        if cur is None:
            cur = np.array([float(v0[nm]) for nm in index], float) if v0 else v_ss.copy()
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
        if p not in index:
            raise ValueError(f"net: probe {p!r} is not a solved node; have {sorted(index)}")
    out = {p: v[:, index[p]].copy() for p in want}
    return out[want[0]] if isinstance(probes, str) else out

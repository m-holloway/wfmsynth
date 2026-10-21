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
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import eigh

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
    for e in list(net.get("elements", [])) + list(net.get("drivers", [])):
        eid = e.get("id")
        if not eid:
            raise ValueError(f"net: every element and driver needs an 'id'; this one has none: {e}")
        if eid in ids:
            raise ValueError(f"net: duplicate id {eid!r}; ids address the schedule and must be unique")
        ids[eid] = e
        for key in ("a", "b", "node"):
            if key in e and e[key] not in known:
                raise ValueError(f"net: {eid!r} references node {e[key]!r}, which is not a solved "
                                 f"node, a rail, or {GND!r}. Known: {sorted(known)}")
    return {nm: i for i, nm in enumerate(nodes)}, rails, ids


def _stamp(net, index, rails, state, over=None):
    """Nodal G, C and b for ONE configuration. `state` maps id -> bool (conducting); `over`
    maps id -> the schedule's `g` / `src` overrides for it, empty when nothing set any."""
    over = over or {}
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
        if kind != "r":
            raise ValueError(f"net: element {eid!r} has kind {e['kind']!r}; this tier supports "
                             f"'r' and 'c' only (see the module docstring for what is deferred)")
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
        b[i] += g * float(ov["src"] if "src" in ov else d["volts"])
    return G, C, b


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


def solve(net, fs, n, schedule=(), probes=None, v0=None):
    """Solve a switched lumped net on a uniform grid. Returns ``{node: array}``, or one array
    when `probes` names a single node.

    `fs` samples per second, `n` samples. Events landing on the same sample are applied as ONE
    configuration change, not in sequence, so a hand-over written as two events never passes
    through a half-state. A schedule event takes effect at the first sample
    whose time is at or after its `t_s`, and that sample is the FIRST one reflecting it: a
    sample records the voltage at the end of its own step. That is `physics.open_drain_line`'s
    convention, which makes the one-node case bit-for-bit (pinned in `tests/test_net.py`), and
    it means ``v[0]`` is one step past the initial condition rather than the initial condition.

    The initial condition is the DC steady state of whatever configuration is in force at t=0,
    unless `v0` gives node voltages.
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

    for ev in schedule or ():
        _check_event(ev, ids)
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
        G, C, b = _stamp(net, index, rails, state, over)
        v_ss, lam, V, Cm = _modes(G, C, b, index)
        _warn_unresolved(lam, dt, seg, warned)
        if cur is None:
            cur = np.array([float(v0[nm]) for nm in index], float) if v0 else v_ss.copy()
        # k starts at ONE: sample k records the voltage at the END of its own step, so a
        # schedule event is first visible on the sample it fires at. That is the convention
        # `physics.open_drain_line` already uses -- it advances then records -- and matching it
        # makes the one-node case bit-for-bit rather than off by one. It also means v[0] is one
        # step past the initial condition, not the initial condition itself.
        k = np.arange(1, bnd - a + 1)[:, None]
        z0 = V.T @ (Cm @ (cur - v_ss))
        seg_v = v_ss + (np.exp(-lam[None, :] * (k * dt)) * z0[None, :]) @ V.T
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

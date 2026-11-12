"""wfmsynth.states -- a level that DWELLS in one of a few states and jumps between them.

Two things real records do are this one mechanism. Burst (random telegraph) noise is a level
that sits at one value for a random time and then jumps; bursty traffic is a link that is idle
for an irregular gap and active for an irregular length. Both are a RENEWAL process over a few
states: how long each state lasts is drawn from a stated distribution, and which state follows
is drawn from a stated table. It is a source, `carrier("states", ...)`:

    Signal(seed=1, grid=g).carrier("states", dwell=2e-6)          # two states, 0 and 1
    Signal(seed=1, grid=g).carrier(
        "states", levels=[0.0, 0.4, 1.0],
        dwell=[5e-6, {"dist": "lognormal", "median_s": 1e-6, "sigma": 0.8},
               {"dist": "uniform", "lo_s": 2e-6, "hi_s": 6e-6}],
        transitions=[[0, 1, 0], [0, 0, 1], [1, 0, 0]], tr_s=50e-9)

    levels        the value of each state (default [0.0, 1.0], which a `gate` or a two-input
                  `switch` reads at its default threshold of 0.5)
    dwell         how long a state lasts, in seconds: one spec for every state, or one per
                  state. A number is the MEAN of an exponential (the memoryless dwell of
                  telegraph noise); otherwise one of
                      {"dist": "exponential", "mean_s": m}
                      {"dist": "lognormal", "median_s": m, "sigma": s}   s = std of ln(dwell)
                      {"dist": "uniform", "lo_s": a, "hi_s": b}
                      {"dist": "fixed", "t_s": T}
                      {"dist": "table", "t_s": [...], "weights": [...]}  measured dwells,
                                              redrawn as they are (weights default to equal)
    transitions   row i is the probability of each NEXT state on leaving state i. The diagonal
                  is zero -- staying is a longer dwell, and belongs in `dwell`. Left out, two
                  states alternate and more than two go to any other state equally.
    start         "stationary" (default): the record is a window on a process that was already
                  running -- the first state is drawn in proportion to the time the process
                  spends in each, and it is already part-way through that dwell. Or a state
                  index: the process ENTERS that state at t = 0 with a whole dwell ahead.
    tr_s          the 10-90 % time of each jump (`cmos`'s word for it). Left out it is four
                  samples (`DEFAULT_TR_SAMPLES`). `causal=True` shapes forward only, so a
                  jump starts at its time instead of being centred on it.

Each jump is at its own time to a fraction of a sample, as `square`'s edges are
(`physics._edge_train`): the hard step is delayed by its fraction before the edge shaper, so
dwells are not rounded to the grid. The time stated is where the jump is HALF WAY (with
`causal=True`, about where it starts): a hard step whose first new sample is `m` is half way
at `m - 0.5`, so the step is placed half a sample after its time. A dwell shorter than `tr_s` is a runt -- the level does not
arrive -- which is what a band limit does. A jump between samples rings by a fraction of its
own size, more the fewer samples the edge has (measured, worst case over where it falls: 7.3 %
at the 2-sample floor, 1.9 % at 4 samples, 1.1 % at 8), which is why the default is four.

THE RANDOMNESS is one role, `states/<op index>`: one uniform number per interval for which
state comes next and one for how long it lasts, each through the stated distribution's
quantile. So the same seed is the same process on a longer record or a faster grid, and
`contrast("states/0")` is another realisation with everything else held.

GROUND TRUTH. Every interval is a `state` event at the instant it is entered (`t_s`, between
samples), with `params` = {"state", "level", "dwell_s", "complete"}; `intervals()` reads them
back as a table. `complete` is False for an interval the record cut -- the one in force at
t = 0 under a stationary start, and the last -- whose `dwell_s` is the part inside the record
and so is NOT a draw from the stated distribution. `occupancy()` is the fraction of time the
process spends in each state.

USED DIRECTLY it is burst noise: add it with `combine`. As `gate(by=)` or `switch(select=)` it
is an activity schedule. Noise with a 1/f spectrum is many of these superposed, which is a
composition and not a second source: `wfmsynth.recipes.flicker`.

REFUSED, by name: a dwell given as a callable (a recipe is data -- tabulate it), a
self-transition, a row that does not sum to 1, a state whose mean dwell is under one sample, a
stationary start on a chain that has no single stationary state, a parameter of another kind
of carrier, a bare carrier spec (`aggressor=dict(kind="states")`, which has no event list --
write the sub-chain), and a lead-in.
"""
from __future__ import annotations

import numbers

import numpy as np
from scipy import special

from . import physics as P
from .events import Event

KEYS = ("levels", "dwell", "transitions", "start", "tr_s", "causal")
DISTS = {"exponential": ("mean_s",), "lognormal": ("median_s", "sigma"),
         "uniform": ("lo_s", "hi_s"), "fixed": ("t_s",), "table": ("t_s", "weights")}
DEFAULT_TR_SAMPLES = 4.0   # the edge when `tr_s` is not stated: see the ringing figures above
CHUNK = 256            # intervals drawn at a time. Fixed, so a longer record extends the same draws.
_ELSEWHERE = {
    "seed": "the process draws from the role states/<op index> of Signal(seed=)",
    "amp": "levels=[...] states each state's value", "offset": "levels=[...] states each "
    "state's value", "level": "levels=[...] (one value per state)",
    "tr_frac": "tr_s=<seconds> (there is no period for a fraction to be of)",
    "duty": "dwell=[<off>, <on>] (the duty is what the two dwells make it: `occupancy()`)",
    "f_hz": "dwell=<seconds> (a renewal process has dwell times, not a frequency)",
    "transition_s": "tr_s=<seconds>, the 10-90 % time, as on `cmos`",
    "rate_hz": "dwell=<mean seconds>, the reciprocal of the rate of leaving a state",
}


def _positive(v, who):
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or not np.isfinite(v) or v <= 0:
        raise ValueError(f"{who} is a positive, finite number, got {v!r}.")
    return float(v)


def _dwell(spec, who):
    """One dwell spec, checked, as a dict with every number a float."""
    if callable(spec):
        raise ValueError(
            f"{who} is a callable. A recipe is data, and a function cannot be written into "
            f"it or digested. Tabulate the distribution instead: "
            f"{{'dist': 'table', 't_s': [...], 'weights': [...]}}.")
    if isinstance(spec, numbers.Real) and not isinstance(spec, bool):
        return {"dist": "exponential", "mean_s": _positive(spec, f"{who} (a mean, in seconds)")}
    if not isinstance(spec, dict) or spec.get("dist") not in DISTS:
        raise ValueError(
            f"{who} is a number (the mean of an exponential dwell, in seconds) or a dict whose "
            f"'dist' is one of {sorted(DISTS)}; got {spec!r}.")
    dist = spec["dist"]
    extra = sorted(set(spec) - {"dist", *DISTS[dist]})
    missing = [k for k in DISTS[dist] if k not in spec and k != "weights"]
    if extra or missing:
        raise ValueError(
            f"{who}: a {dist!r} dwell takes {list(DISTS[dist])}"
            + (f"; it is missing {missing}" if missing else "")
            + (f"; it has {extra}, which nothing reads" if extra else "") + ".")
    if dist == "table":
        t = np.atleast_1d(np.asarray(spec["t_s"], float))
        w = np.ones(t.size) if spec.get("weights") is None else \
            np.atleast_1d(np.asarray(spec["weights"], float))
        if t.ndim != 1 or t.size < 1 or not np.all(np.isfinite(t)) or np.any(t <= 0):
            raise ValueError(f"{who}: 't_s' is a list of positive dwell times in seconds.")
        if w.shape != t.shape or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError(f"{who}: 'weights' is one non-negative number per entry of 't_s' "
                             f"({t.size}), not all zero; got {spec.get('weights')!r}.")
        return {"dist": dist, "t_s": t, "weights": w / w.sum()}
    out = {"dist": dist, **{k: _positive(spec[k], f"{who}: {k!r}") for k in DISTS[dist]}}
    if dist == "uniform" and not out["lo_s"] < out["hi_s"]:
        raise ValueError(f"{who}: a uniform dwell needs lo_s < hi_s, got {spec['lo_s']!r} and "
                         f"{spec['hi_s']!r}. One exact dwell is {{'dist': 'fixed', 't_s': ...}}.")
    return out


def mean_s(spec):
    """The mean of a dwell spec, in seconds."""
    d = _dwell(spec, "dwell")
    if d["dist"] == "exponential":
        return d["mean_s"]
    if d["dist"] == "lognormal":
        return d["median_s"] * float(np.exp(0.5 * d["sigma"] ** 2))
    if d["dist"] == "uniform":
        return 0.5 * (d["lo_s"] + d["hi_s"])
    if d["dist"] == "fixed":
        return d["t_s"]
    return float(np.dot(d["weights"], d["t_s"]))


def _pick(values, weights, u):
    cum = np.cumsum(weights)
    return values[np.minimum(np.searchsorted(cum, u * cum[-1], side="right"), len(values) - 1)]


def quantile(spec, u, biased=False):
    """The dwell at cumulative probability `u` (an array in [0, 1)), in seconds.

    `biased=True` is the LENGTH-BIASED distribution, density proportional to ``t * p(t)``:
    the dwell that a moment chosen at random falls in, which is longer than a dwell chosen at
    random. A stationary start draws the interval in force at t = 0 from it."""
    d = _dwell(spec, "dwell")
    u = np.asarray(u, float)
    if d["dist"] == "exponential":
        if biased:                       # a gamma of shape 2: its regularised integral, inverted
            return d["mean_s"] * special.gammaincinv(2.0, u)
        return -d["mean_s"] * np.log1p(-u)
    if d["dist"] == "lognormal":
        shift = d["sigma"] ** 2 if biased else 0.0
        return d["median_s"] * np.exp(shift + d["sigma"] * special.ndtri(u))
    if d["dist"] == "uniform":
        lo, hi = d["lo_s"], d["hi_s"]
        return np.sqrt(lo * lo + u * (hi * hi - lo * lo)) if biased else lo + u * (hi - lo)
    if d["dist"] == "fixed":
        return np.full(u.shape, d["t_s"])
    return _pick(d["t_s"], d["weights"] * d["t_s"] if biased else d["weights"], u)


def _matrix(transitions, k):
    """The transition table as a (k, k) array, or the default for `k` states."""
    if transitions is None:
        return (1.0 - np.eye(k)) / (k - 1)
    try:
        p = np.asarray(transitions, float)
    except (TypeError, ValueError):
        p = np.empty(0)
    if p.shape != (k, k) or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError(
            f"carrier('states'): transitions is a {k} x {k} table of probabilities, row i "
            f"being the chance of each next state on leaving state i; got {transitions!r}.")
    stay = np.flatnonzero(np.diag(p) != 0)
    if stay.size:
        i = int(stay[0])
        raise ValueError(
            f"carrier('states'): transitions[{i}][{i}] = {p[i, i]:g} keeps state {i} where it "
            f"is. That is a longer dwell, not a transition, and it would make the dwell of "
            f"state {i} a different distribution from the one stated: lengthen dwell[{i}] and "
            f"set the diagonal to zero.")
    off = np.flatnonzero(np.abs(p.sum(axis=1) - 1.0) > 1e-9)
    if off.size:
        i = int(off[0])
        raise ValueError(
            f"carrier('states'): transitions[{i}] sums to {p[i].sum():.9g}, and a state that "
            f"is left goes somewhere with probability 1. Each row must sum to 1.")
    return p


def _plan(levels=None, dwell=None, transitions=None, start="stationary", tr_s=None,
          causal=False, **unread):
    """The parameters checked -> ``(levels, [dwell spec], P, start)``. No grid needed."""
    if unread:
        raise ValueError(
            "carrier('states') does not read " + ", ".join(
                f"{k}=" + (f" -- {_ELSEWHERE[k]}" if k in _ELSEWHERE else "")
                for k in sorted(unread))
            + f". It takes {list(KEYS)}: a renewal process is its levels, how long each lasts "
              f"and which follows which.")
    lv = np.asarray([0.0, 1.0] if levels is None else levels, float)
    if lv.ndim != 1 or lv.size < 2 or not np.all(np.isfinite(lv)):
        raise ValueError(f"carrier('states'): levels is a list of at least two finite numbers, "
                         f"the value of each state; got {levels!r}. One level is "
                         f"carrier('dc', level=...).")
    if dwell is None:
        raise ValueError(
            "carrier('states') needs dwell=: how long a state lasts, in seconds. A number is "
            "the mean of an exponential dwell (dwell=2e-6); a list states one per state; a "
            f"dict names a distribution, one of {sorted(DISTS)}.")
    each = list(dwell) if isinstance(dwell, (list, tuple)) else [dwell] * lv.size
    if len(each) != lv.size:
        raise ValueError(f"carrier('states'): dwell has {len(each)} entries and there are "
                         f"{lv.size} levels. Give one dwell for all of them, or one each.")
    specs = [_dwell(d, f"carrier('states'): dwell[{i}]") for i, d in enumerate(each)]
    p = _matrix(transitions, lv.size)
    ok = isinstance(start, numbers.Integral) and not isinstance(start, bool) \
        and 0 <= start < lv.size
    if start != "stationary" and not ok:
        raise ValueError(
            f"carrier('states'): start is 'stationary' (the process was already running) or "
            f"the index of the state it enters at t = 0, 0 to {lv.size - 1}; got {start!r}.")
    if tr_s is not None:
        _positive(tr_s, "carrier('states'): tr_s (the 10-90 % time of a jump, seconds)")
    return lv, specs, p, start


def check(params):
    """Refuse a `carrier("states", ...)` that cannot be rendered: `params` is the op's own,
    or what the builder was handed."""
    _plan(**{k: v for k, v in params.items()
             if k not in ("op", "kind", "n") and not k.startswith("_")})


def occupancy(dwell, transitions=None, n_states=2):
    """The fraction of time the process spends in each state, in the long run.

    The chain of visits has a stationary distribution ``v`` (``v P = v``), and a visit to
    state i lasts `mean_s(dwell[i])` on average, so the time fractions are ``v * mean``,
    normalised. Raises when the chain has no single such distribution (a state that is never
    entered, or two groups of states that never reach each other)."""
    k = len(dwell) if isinstance(dwell, (list, tuple)) else int(n_states)
    _, specs, p, _ = _plan(levels=np.zeros(k), dwell=dwell, transitions=transitions)
    a = np.vstack([p.T - np.eye(k), np.ones((1, k))])
    if np.linalg.matrix_rank(a[:-1]) != k - 1:
        raise ValueError(
            "carrier('states'): these transitions have no single stationary state -- some "
            "states never reach the others -- so there is no one answer to where a process "
            "that was already running would be. State start=<the index it begins in>.")
    v = np.linalg.lstsq(a, np.concatenate([np.zeros(k), [1.0]]), rcond=None)[0]
    if np.any(v < 1e-12):
        raise ValueError(
            f"carrier('states'): state {int(np.argmin(v))} is never entered from another, so "
            f"a process that was already running is never in it. Remove it, give it a way "
            f"in, or state start=<index> to begin there.")
    t = v * np.asarray([mean_s(s) for s in specs])
    return t / t.sum()


def draw(rng, duration_s, specs, p, start):
    """One realisation over ``[0, duration_s)`` -> ``(state, t_enter_s, t_exit_s, whole)``.

    Arrays with one entry per interval that overlaps the record. `t_enter_s[0]` is 0 and the
    last `t_exit_s` is at or past `duration_s`; `whole[k]` says interval k's length
    ``t_exit_s - t_enter_s`` is a draw from its state's dwell (False for the interval a
    stationary start is already part-way through)."""
    k = len(specs)
    cum = np.cumsum(p, axis=1)
    # the last column a row can land in: rounding must not select a state of probability zero
    reach = [int(np.flatnonzero(row > 0)[-1]) for row in p]
    if start == "stationary":
        occ = occupancy([dict(s) for s in specs], p)
        u = rng.random(3)
        s0 = int(min(np.searchsorted(np.cumsum(occ), u[0], side="right"), k - 1))
        first = float(u[2] * quantile(specs[s0], u[1], biased=True))
    else:
        s0, first = int(start), None
    states, exits = [np.array([s0])], []
    t, s = 0.0, s0
    while True:
        u = rng.random((CHUNK, 2))
        if p.shape == (2, 2):
            nxt = (s + 1 + np.arange(CHUNK)) % 2
        else:
            nxt = np.empty(CHUNK, np.int64)
            for j in range(CHUNK):
                s = min(int(np.searchsorted(cum[s], u[j, 0], side="right")), reach[s])
                nxt[j] = s
        # row j holds: the dwell of the state in force (column 1), then where it goes (column 0)
        seq = np.concatenate([states[-1][-1:], nxt[:-1]])
        d = np.empty(CHUNK)
        for i in range(k):
            m = seq == i
            if m.any():
                d[m] = quantile(specs[i], u[m, 1])
        if first is not None and not exits:
            d[0] = first
        exits.append(t + np.cumsum(d))
        states.append(nxt)
        t, s = float(exits[-1][-1]), int(nxt[-1])
        if t >= duration_s:
            break
    t_exit = np.concatenate(exits)
    st = np.concatenate(states)
    keep = int(np.searchsorted(t_exit, duration_s, side="left")) + 1
    t_exit = t_exit[:keep]
    t_enter = np.concatenate([[0.0], t_exit[:-1]])
    whole = np.ones(keep, bool)
    whole[0] = first is None
    return st[:keep], t_enter, t_exit, whole


def train(n, tau, step, first):
    """A piecewise-constant level on `n` samples: `first`, then a jump of `step[k]` whose
    first new sample is the fractional sample `tau[k]`, each at its own time
    (`physics._edge_train`'s construction, for jumps of any size at any times)."""
    near = np.round(tau)
    tau = np.where(np.abs(tau - near) <= P.EDGE_SNAP_SAMPLES, near, tau)
    m = np.floor(tau).astype(np.int64)
    d = tau - m
    inside = m < n
    jumps = np.zeros(n)
    np.add.at(jumps, np.clip(m[inside], 0, None), step[inside])
    y = first + np.cumsum(jumps)
    frac = np.flatnonzero(d > 0)
    for a in range(0, frac.size, P.EDGE_BLOCK):
        sel = frac[a:a + P.EDGE_BLOCK]
        off, corr = P._delayed_step(d[sel])
        idx = m[sel][:, None] + off[None, :]
        val = step[sel][:, None] * corr
        ok = (idx >= 0) & (idx < n)
        np.add.at(y, idx[ok], val[ok])
    return y


def run(streams, grid, idx, n=None, **params):
    """The `carrier("states")` source: draw the intervals, render them, emit them."""
    if not isinstance(idx, int):
        raise ValueError(
            "carrier('states') in a bare carrier spec has nowhere to put its intervals, and a "
            "state process without its ground truth is a label nobody can check. Pass the "
            "sub-chain instead: Signal().carrier('states', ...).")
    lv, specs, p, start = _plan(**params)
    fs = getattr(grid, "fs", None)
    if not fs or not n:
        raise ValueError("carrier('states') states its dwells in seconds, so it needs the "
                         "record's sample rate and length: Signal(grid=Grid(fs=..., n=...)).")
    fs, n = float(fs), int(n)
    for i, s in enumerate(specs):
        if mean_s(s) * fs < 1.0:
            raise ValueError(
                f"carrier('states'): state {i} lasts {mean_s(s):.6g} s on average, which is "
                f"{mean_s(s) * fs:.3g} of a sample at fs={fs:g} Hz. The grid cannot show a "
                f"state it does not sample: raise Grid(fs=) above {1.0 / mean_s(s):.6g} Hz or "
                f"lengthen the dwell.")
    tr_s = params.get("tr_s")
    tr = DEFAULT_TR_SAMPLES if tr_s is None else P._resolve_abs_edge(float(tr_s) * fs)[0]
    st, t_in, t_out, whole = draw(streams.role(f"states/{idx}"), n / fs, specs, p, start)
    y = P._shape_edges(train(n, t_in[1:] * fs + 0.5, np.diff(lv[st]), lv[st[0]]), tr,
                       bool(params.get("causal", False)))
    sink = getattr(streams, "sink", None)
    if sink is not None:
        width = max(int(np.ceil(tr)), 1)
        for k in range(st.size):
            stop = min(float(t_out[k]), n / fs)
            sink.append(Event(
                sample=min(int(np.ceil(t_in[k] * fs - P.EDGE_SNAP_SAMPLES)), n - 1),
                kind="state", severity=0.0, t_s=float(t_in[k]), width=width,
                direction=float(np.sign(lv[st[k]] - lv[st[k - 1]])) if k else 0.0,
                placement=f"carrier/{idx}",
                params={"state": int(st[k]), "level": float(lv[st[k]]),
                        "dwell_s": stop - float(t_in[k]),
                        "complete": bool(whole[k] and t_out[k] <= n / fs)}))
    return y


def intervals(events, at=None):
    """The `state` events of an event list as a table: one dict per interval, in time order,
    with `t_start_s`, `t_stop_s`, `state`, `level` and `complete`.

    `at` picks one source when a record holds several: the namespace of the sub-chain it is
    in (`"gate/1/by"`, `"combine/1/others/0"`), or `""` for the record's own source."""
    out = [{"t_start_s": e.t_s, "t_stop_s": e.t_s + e.params["dwell_s"],
            "state": e.params["state"], "level": e.params["level"],
            "complete": e.params["complete"]}
           for e in events if e.kind == "state" and (at is None or e.at == at)]
    return sorted(out, key=lambda r: r["t_start_s"])

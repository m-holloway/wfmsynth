"""
wfmsynth.segments -- a segmented acquisition of an event that happens rarely: many
occurrences, each rendered on its own short grid, each a little different.

`trigger(n_segments=)` cuts windows out of ONE long render, which is right for a process that
runs all the time. A power-on, a surge, a relay closing is not such a process: between two of
them there is nothing but dead time, and rendering it costs everything (a thousand surges
100 ms apart at 1 GSa/s is 1e11 samples). Here each occurrence is rendered separately, on the
grid given, and triggered there with the same condition finding, arming, jitter and ground
truth as `trigger`:

    surge = (ws.Signal().carrier("step", t_step_s=2e-6, tr_s=50e-9, amp=0.5, offset=0.5)
             .gain(by=ws.tol(1.0, 5, per="segment"))            # each occurrence its own
             .gain(by=ws.trend("ramp", start=1.0, stop=0.9))      # and a slow sag across them
             .rx_noise(rms=0.01))
    seg = ws.Segments(surge, n=300, arrival=dict(dist="exponential", mean_s=0.1),
                      trigger=dict(pre_s=1e-6, post_s=4e-6, level=0.5), seed=1,
                      grid=ws.Grid(fs=1e9, n=8000))
    x, events = seg.realize()             # x.reshape(300, -1): one window per occurrence

THE OUTPUT is what `trigger(n_segments=n)` gives: the n windows laid end to end, and one
`Event(kind="trigger")` per window with the same fields -- `params.segment`,
`params.timestamp_s` (the crossing's time on the acquisition's clock), `jitter_s`, `start`
(the window's first sample on that clock), `level`, `condition`, `dropped` -- plus
`params.variant` when there are variants. An event the occurrence's chain placed is re-timed
into its window or dropped, as `trigger` does. Downstream code cannot tell how the segments
were made.

EACH OCCURRENCE is the chain rendered on `grid` (fs, and n samples: the occurrence's render),
starting at its arrival. The window is cut where the trigger first fires once `pre_s` (and
6 sigma of trigger jitter) has passed; everything before the window is the lead, rendered and
discarded, so put the event far enough in to let the chain settle. A render in which the
trigger does not fire, or fires too late to keep `post_s`, is refused naming the occurrence
and the render length that would do. `level` must be stated: a scope sets its level once for
the acquisition, and "50 % of each occurrence" would move it with the occurrence.

THE SCOPE'S SAMPLE CLOCK runs on through the dead time, so an occurrence does not start on a
sample. Its source's output (op 0) is delayed by the fraction of a sample its arrival falls
after one (`joins.delay`, band-limited, error stated by `joins.describe_delay`), which is what
spreads the trigger's sub-sample placement over the windows as a free-running scope's is.
What follows the source and acts the same at every instant (a channel, noise, the front end)
needs nothing more. An op that states its own instants (`events(on="times")`, a sub-chain held
by a later op) keeps them on the occurrence's render clock.

ARRIVAL is the time from one occurrence's start to the next, written as a `carrier("states")`
dwell is (`states.DISTS`): {"dist": "fixed", "t_s": T} is a period, and "exponential",
"lognormal", "uniform" and "table" are drawn under the role `arrival`. The first occurrence
arrives one interval after the acquisition starts. One that arrives before the previous
captured one's render is over is MISSED, as a scope still storing a segment misses it: it is
counted (`occurrences()[k]["missed"]`) and the next arrival is taken.

WHAT VARIES, and its role:
  arrival              the inter-arrival times
  variant              which chain each occurrence is (`variants=`, below)
  vary/<i>/<param>     a `ws.tol(..., per="segment")` value, one draw per occurrence
                       (`vary/<group>` when grouped)
  trigger              the trigger jitter, one draw per occurrence
  segment/<k>/<role>   occurrence k's own randomness (`segment/4/noise/2`)
  unit/...             a unit-to-unit part, fixed for the whole acquisition (`unit(seed)`)
`contrast(name)` re-rolls exactly one; `segment/*/<role>` names that role in every
occurrence. A `ws.trend` draws nothing and has no role: it is a cause, and
`without("trend/<i>/<param>")` (or "trend/*") is the same acquisition with it held at its
first value -- every draw, arrival and noise sample the same.

VARIANTS. `variants={"late": dict(p=0.02, chain=other)}` is another chain an occurrence is
with probability p (the main chain otherwise), drawn under `variant`; `dict(at=[17, 140],
chain=other)` states which occurrences instead. A variant's own parts are named under it
(`late/vary/2/by`, `late/unit/1/r_source`). It is a parameter here and not a `switch`:
`switch` selects between inputs WITHIN one record, sample by sample, by a signal; a variant
is one choice per occurrence, of which the record is made.

ONE RECIPE holds the acquisition -- seed, grid, n, arrival, trigger, the chains -- and
`Segments.from_recipe` replays it bit for bit with no other file. The values each occurrence
used are a function of that recipe alone: `occurrences()`, and `segment_params(recipe, op)`
in the style of `op_params`.

NOT HERE (BACKLOG "#97"): state carried from one occurrence to the next (a capacitor's
residual charge, a temperature), a trend that is another chain sampled at the arrivals, a
`Channels` record as the occurrence, and averaging.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

from . import domain as DOMAIN
from . import opkeys as OPKEYS
from . import physics as P
from . import states as STATES
from . import subchain as SUB
from . import tolerance as TOL
from . import trigger as TRIG
from .grid import Grid
from .streams import Streams

MAIN = "main"                      # the name the chain an occurrence usually is goes by
_NOT_HERE = {
    "n_segments": "n= of Segments is how many occurrences, each one window",
    "average": "averaging occurrences is not supported: average the rows of "
               "x.reshape(n, -1) yourself, knowing each is a different occurrence",
    "holdoff_s": "an occurrence that arrives during the previous one's render is missed and "
                 "counted; a holdoff would be a second rule for the same thing",
    "on": "the trigger is found on the occurrence itself; triggering one channel from "
          "another is a Channels record, not yet segmented (BACKLOG #97)",
}


def _arrival(spec):
    """The arrival spec, checked by the one dwell parser (`states`), and the JSON form kept."""
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        raise ValueError(
            f"Segments: arrival={spec!r} is a number, which could be a period or a mean. Say "
            f"which: dict(dist='fixed', t_s={spec!r}) is a period, dict(dist='exponential', "
            f"mean_s={spec!r}) a Poisson stream.")
    STATES._dwell(spec, "Segments: arrival")
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in spec.items()}


def _chain_ops(v, who, seed, grid, pre):
    """One chain as given -> its op list, its unit parts drawn under the record's seed."""
    from .compose import Signal
    if isinstance(v, dict) and set(v) == {"ops"}:
        v = v["ops"]
    if isinstance(v, Signal):
        if v.lead_in or v.lead_out is not None:
            raise ValueError(f"Segments: {who} is a Signal with a lead-in. The occurrence's "
                             f"own lead is the stretch before its window; drop the lead-in.")
        if v.seed not in (0, seed):
            raise ValueError(f"Segments: {who} is a Signal with seed={v.seed}, but every "
                             f"occurrence draws from the acquisition's seed ({seed}). Leave "
                             f"the seed off the chain and set Segments(seed=...).")
        if v.grid is not None and v.grid != grid:
            raise ValueError(f"Segments: {who} is a Signal on its own grid; every occurrence "
                             f"renders on the acquisition's. Build it with no grid.")
        v = TOL.stated(v.ops)
    if not isinstance(v, (list, tuple)) or not v:
        raise ValueError(f"Segments: {who} is a Signal or {{'ops': [...]}} with at least one "
                         f"op, got {type(v).__name__}.")
    ops = [TOL.resolve(i, dict(o), seed, pre=pre) for i, o in enumerate(v)]
    for i, o in enumerate(ops):
        OPKEYS.check(o.get("op"), o)
        if o["op"] in ("trigger", "use"):
            raise ValueError(
                f"Segments: {who} has `{o['op']}` at op {i}. "
                + ("The acquisition's trigger is Segments(trigger=...), one per occurrence."
                   if o["op"] == "trigger" else
                   "A `use` names a chain of a Channels record, and Segments renders one chain."))
    doms = DOMAIN.trace(ops, DOMAIN.grid_for(ops, grid))
    if any(d == DOMAIN.OPTICAL for d, _ in doms):
        raise ValueError(f"Segments: {who} holds an optical field. An occurrence is placed "
                         f"between the scope's samples by delaying its source, and a field's "
                         f"delay needs its laser's frame; not supported yet.")
    return ops


def _trigger(t):
    if not isinstance(t, dict):
        raise ValueError("Segments: trigger= is a dict of `trigger`'s parameters, e.g. "
                         "dict(pre_s=1e-6, post_s=4e-6, level=0.5).")
    for k in t:
        if k in _NOT_HERE:
            raise ValueError(f"Segments: trigger does not take {k}= here -- {_NOT_HERE[k]}.")
    q = TRIG.check(dict(t))
    if q["level"] is None:
        raise ValueError("Segments: trigger needs level=. A scope sets its level once for the "
                         "acquisition; 'midway between the extremes' of each occurrence would "
                         "move it with the occurrence's own amplitude.")
    return dict(t)


class Segments:
    """A segmented acquisition of `n` occurrences of `chain`. See the module docstring.

        Segments(chain, n, arrival, trigger, seed=0, grid=None, variants=None)
    """

    def __init__(self, chain, n, arrival, trigger, seed=0, grid=None, variants=None):
        self.seed = int(seed)
        if grid is None or not getattr(grid, "fs", None) or not getattr(grid, "n", None):
            raise ValueError("Segments: grid= is the grid each occurrence renders on, with its "
                             "rate and length: Grid(fs=..., n=...).")
        self.grid = grid
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"Segments: n is how many occurrences, a whole number >= 1, "
                             f"got {n!r}")
        self.n = int(n)
        self.arrival = _arrival(arrival)
        self.trigger = _trigger(trigger)
        self.chains = {MAIN: _chain_ops(chain, "the chain", self.seed, grid, "")}
        self.choose = {}                          # variant name -> {"p": ...} or {"at": [...]}
        for name, v in (variants or {}).items():
            if not isinstance(name, str) or not name or "/" in name or name == MAIN:
                raise ValueError(f"Segments: variant name {name!r} must be a non-empty string "
                                 f"without '/', and not {MAIN!r}: it prefixes its roles.")
            if not isinstance(v, dict) or "chain" not in v or len({"p", "at"} & set(v)) != 1 \
                    or set(v) - {"p", "at", "chain"}:
                raise ValueError(f"Segments: variant {name!r} is dict(p=<probability>, "
                                 f"chain=...) or dict(at=[occurrence indices], chain=...).")
            self.chains[name] = _chain_ops(v["chain"], f"variant {name!r}", self.seed, grid,
                                           name + "/")
            self.choose[name] = {k: v[k] for k in ("p", "at") if k in v}
        self._check_choice()

    def _check_choice(self):
        p = [c["p"] for c in self.choose.values() if "p" in c]
        if any(isinstance(x, bool) or not 0 < float(x) < 1 for x in p) or sum(p) >= 1:
            raise ValueError(f"Segments: each variant's p is a probability strictly between 0 "
                             f"and 1, and together under 1 (the main chain takes the rest); "
                             f"got {p}.")
        seen = {}
        for name, c in self.choose.items():
            for k in c.get("at", []):
                if isinstance(k, bool) or not isinstance(k, (int, np.integer)) \
                        or not 0 <= k < self.n:
                    raise ValueError(f"Segments: variant {name!r} at={c['at']!r} must hold "
                                     f"occurrence indices 0 to {self.n - 1}.")
                if k in seen:
                    raise ValueError(f"Segments: occurrence {k} is stated as both "
                                     f"{seen[k]!r} and {name!r}.")
                seen[int(k)] = name

    # ------------------------------------------------------------ what each occurrence is
    def _plan(self, st):
        """Every occurrence, before any is rendered: ``[{"segment", "arrival_s", "missed",
        "variant", "values", "ops"}]``. Draws from `st` once per role, so call it once per
        render."""
        n, fs = self.n, float(self.grid.fs)
        render_s = int(self.grid.n) / fs
        # the variant of each occurrence: stated, else drawn
        names = sorted(k for k, c in self.choose.items() if "p" in c)
        variant = [MAIN] * n
        if names:
            u = st.role("variant").random(n)
            cum = np.cumsum([self.choose[k]["p"] for k in names])
            for k in range(n):
                j = int(np.searchsorted(cum, u[k], side="right"))
                variant[k] = names[j] if j < len(names) else MAIN
        for name, c in self.choose.items():
            for k in c.get("at", []):
                variant[int(k)] = name
        # the arrivals, skipping one that lands inside the previous captured render
        spec = self.arrival
        rng = None if spec["dist"] == "fixed" else st.role("arrival")
        t, last, skipped, arrive, missed = 0.0, -math.inf, 0, [], []
        while len(arrive) < n:
            d = (np.full(STATES.CHUNK, float(spec["t_s"])) if rng is None
                 else STATES.quantile(spec, rng.random(STATES.CHUNK)))
            for dk in d:
                t += float(dk)
                if t - last < render_s:
                    skipped += 1
                    continue
                arrive.append(t)
                missed.append(skipped)
                last, skipped = t, 0
                if len(arrive) == n:
                    break
        # the per-occurrence values: one uniform number per occurrence per role
        drawn = {}
        out = []
        for k in range(n):
            ops, values = [], {}
            pre = "" if variant[k] == MAIN else variant[k] + "/"
            entries = {}
            for i, name, sp, loc, role in TOL.segment_entries(self.chains[variant[k]], pre):
                entries.setdefault(i, []).append((name, sp, loc, role))
            for i, op in enumerate(self.chains[variant[k]]):
                op = {kk: v for kk, v in op.items() if kk != "_segment"}
                for name, sp, loc, role in entries.get(i, []):
                    if role is None:
                        value = _trend_at(sp, k, n, arrive[k])
                    else:
                        if role not in drawn:
                            drawn[role] = st.role(role).random(n)
                        value = TOL.quantile(sp, float(drawn[role][k]))[0]
                    op = TOL._set(op, TOL._split(name), value)
                    values[loc] = value
                ops.append(op)
            occ = {"segment": k, "arrival_s": arrive[k], "missed": missed[k], "values": values,
                   "ops": ops}
            if self.choose:
                occ["variant"] = variant[k]
            out.append(occ)
        return out

    def occurrences(self, streams=None):
        """What each occurrence was, from the recipe alone and without rendering: a list of
        ``{"segment", "arrival_s", "missed", "values", ["variant"]}``. `values` maps each
        per-occurrence parameter's location (`vary/2/by`, `trend/1/by`) to the number that
        occurrence used; `missed` counts the arrivals skipped just before it."""
        return [{k: v for k, v in o.items() if k != "ops"}
                for o in self._plan(self._streams(streams))]

    def _streams(self, streams):
        return Streams(self.seed) if streams is None else Streams(streams.seed,
                                                                  streams._overrides)

    # ------------------------------------------------------------ rendering
    def _render(self, st, collect):
        from .compose import Signal
        from .events import EventList
        fs = float(self.grid.fs)
        trig = {k: v for k, v in self.trigger.items()}
        rows, events, dropped = [], [], 0
        jitter = st.role("trigger")
        n_win = None
        for occ in self._plan(st):
            k = occ["segment"]
            sig = Signal(seed=self.seed, grid=self.grid)
            sig.ops = occ["ops"]
            origin = math.floor(occ["arrival_s"] * fs)
            frac = occ["arrival_s"] * fs - origin
            if abs(frac - round(frac)) <= P.EDGE_SNAP_SAMPLES:
                origin, frac = origin + int(round(frac)), 0.0
            view = SUB.Namespaced(st, f"segment/{k}/")
            if collect:
                x, placed = sig._run(view, collect_events=True, _delay_s=frac / fs)
                placed = list(placed.events)
            else:
                x, placed = sig._run(view, _delay_s=frac / fs), None
            fs_now = sig.stored_grid(x).fs
            try:
                win = TRIG.apply(x, fs_now, rng=jitter, events=placed, placement="trigger",
                                 **trig)
            except ValueError as e:
                what = f"occurrence {k}" + (f" ({occ['variant']})" if self.choose else "")
                raise ValueError(f"Segments: {what}: " + str(e).replace(
                    " (n_segments x average)", "").replace(
                    "the long render", "this occurrence's render").replace(
                    "The long render", "This occurrence's render")) from None
            n_win = len(win)
            rows.append(win)
            if placed is None:
                continue
            shift = k * n_win
            at = int(round(origin * fs_now / fs))
            for e in placed:
                e.sample += shift
                if e.t_s is not None:
                    e.t_s = float(e.t_s + shift / fs_now)
                if e.kind == "trigger" and e.at == "trigger":
                    dropped += e.params["dropped"]
                    e.params = dict(e.params, segment=k,
                                    timestamp_s=float(origin / fs + e.params["timestamp_s"]),
                                    start=at + e.params["start"])
                    if self.choose:
                        e.params["variant"] = occ["variant"]
            events += placed
        x = np.concatenate(rows)
        if not collect:
            return x
        for e in events:
            if e.kind == "trigger" and e.at == "trigger":
                e.params["dropped"] = dropped
        return x, EventList(events, n=len(x), grid=self.grid)

    def waveform(self, streams=None):
        """The n windows laid end to end: ``waveform().reshape(n, -1)`` is one per row."""
        return self._render(self._streams(streams), False)

    def realize(self, streams=None):
        """``(waveform, EventList)``: the windows and every event in them, a `trigger` event
        per window carrying its occurrence's segment, timestamp and (with variants) variant."""
        return self._render(self._streams(streams), True)

    def stored_grid(self, x):
        """The `Grid` of what `waveform()` returned -- the windows at the rate the chain ends
        at, as `Signal.stored_grid` says for a `trigger`. Raises when `x` is not n windows."""
        from .compose import _stored_rate
        ops = self.chains[MAIN]
        fs = _stored_rate(ops, float(self.grid.fs), int(self.grid.n))[0]
        n = len(x) if hasattr(x, "__len__") else int(x)
        if n != self.n * TRIG.window_samples(self.trigger, fs):
            raise ValueError(f"Segments.stored_grid: {n} samples is not {self.n} windows of "
                             f"{TRIG.window_samples(self.trigger, fs)} at {fs:g} Sa/s.")
        dom, fc = DOMAIN.trace(ops, DOMAIN.grid_for(ops, self.grid))[-1]
        return Grid(fs=fs, baud=self.grid.baud, n=n, domain=dom, fc_hz=fc)

    # ------------------------------------------------------------ roles and siblings
    def _unit_roles(self):
        out = []
        for name, ops in self.chains.items():
            out += [r for r in TOL.roles(ops, "" if name == MAIN else name + "/")
                    if r not in out]
        return out

    def roles(self):
        """Exactly the randomness the acquisition draws from: `arrival`, `variant`, each
        `vary/...`, `trigger`, then `segment/<k>/<role>` for every occurrence's own, then the
        unit parts. Every name moves samples when re-rolled."""
        out = []
        if self.arrival["dist"] != "fixed":
            out.append("arrival")
        if any("p" in c for c in self.choose.values()):
            out.append("variant")
        for name, ops in self.chains.items():
            for *_, role in TOL.segment_entries(ops, "" if name == MAIN else name + "/"):
                if role is not None and role not in out:
                    out.append(role)
        if float(self.trigger.get("jitter_s", 0.0)):
            out.append("trigger")
        from .compose import Signal
        for occ in self._plan(Streams(self.seed)):
            sig = Signal(seed=self.seed, grid=self.grid)
            sig.ops = occ["ops"]
            out += [f"segment/{occ['segment']}/{r}" for r in sig.roles() if not TOL.is_role(r)]
        return out + self._unit_roles()

    def contrast(self, *factors, seed=None):
        """The same acquisition with ONLY the named roles re-rolled -> the windows end to end.
        `segment/*/<role>` is that role in every occurrence; `unit/*` every unit part
        (`unit(seed)` returns that unit as an acquisition). A name the acquisition does not
        draw from is refused."""
        roles = self.roles()
        names = []
        for f in factors:
            if f.startswith("segment/*/"):
                tail = f[len("segment/*/"):]
                hit = [r for r in roles if r.startswith("segment/") and
                       r.split("/", 2)[2] == tail]
                if not hit:
                    raise ValueError(f"Segments.contrast: no occurrence draws {tail!r}. "
                                     f"roles() lists what can be re-rolled.")
                names += hit
            else:
                names.append(f)
        st = Streams(self.seed).reroll(*names, seed=seed)
        units = self._unit_roles()
        seed_of = {k: v for k, v in st._overrides.items() if k == TOL.ANY or k in units}
        rec = self
        if seed_of:
            if not units:
                self.unit(0)                              # the refusal: nothing is toleranced
            rec = copy.copy(self)
            rec.chains = {name: TOL.redraw_ops(ops, seed_of, "" if name == MAIN else name + "/")
                          for name, ops in self.chains.items()}
        x = rec._render(st, False)
        idle = [k for k in st.idle(roles) if k not in seed_of]
        if idle:
            raise ValueError(f"Segments.contrast: {idle} names no randomness this acquisition "
                             f"draws, so the sibling would be this one again. roles() lists "
                             f"them.")
        from .compose import _warn_undrawn
        _warn_undrawn("Segments.contrast", st, roles)
        return x

    def unit(self, seed):
        """The same acquisition of ANOTHER UNIT: every `ws.tol` / `ws.corners` drawn once per
        unit is drawn again under `seed`; the per-occurrence draws, arrivals and noise -- all
        under the acquisition's seed -- stay. Several units each captured n times are
        `[seg.unit(s) for s in seeds]`."""
        if not self._unit_roles():
            raise ValueError("Segments.unit(): no part varies from unit to unit here, so there "
                             "is no unit to draw. Write one with ws.tol(...) (without per=).")
        new = copy.copy(self)
        new.chains = {name: [TOL.resolve(i, o, int(seed), pre="" if name == MAIN else name + "/")
                             for i, o in enumerate(TOL.stated(ops))]
                      for name, ops in self.chains.items()}
        return new

    def trends(self):
        """The location of every `ws.trend`, the names `without()` takes."""
        return [loc for name, ops in self.chains.items()
                for *_, loc, role in TOL.segment_entries(ops, "" if name == MAIN else name + "/")
                if role is None]

    def without(self, *trends):
        """The same acquisition with the named trends REMOVED -- each held at its first value
        (a ramp's `start`, a step's `before`) -- and nothing else changed: every arrival,
        variant, per-occurrence draw and noise sample is the one this acquisition has. That is
        a cause taken out, which is not a re-roll and so not `contrast`. `"trend/*"` is all."""
        have = self.trends()
        if not have:
            raise ValueError("Segments.without(): this acquisition has no ws.trend to remove.")
        want = have if "trend/*" in trends else list(trends)
        unknown = [t for t in want if t not in have]
        if unknown or not want:
            raise ValueError(f"Segments.without(): {unknown or 'nothing'} is not a trend here. "
                             f"It has: {have} (or 'trend/*' for all). A per-occurrence draw "
                             f"is re-rolled with contrast(), not removed.")
        new = copy.copy(self)
        new.chains = {}
        for name, ops in self.chains.items():
            ops = [dict(o) for o in ops]
            for i, pname, sp, loc, role in TOL.segment_entries(
                    ops, "" if name == MAIN else name + "/"):
                if loc in want:
                    rest = {k: v for k, v in ops[i]["_segment"].items() if k != pname}
                    ops[i] = TOL._set(ops[i], TOL._split(pname), TOL.placeholder(sp))
                    ops[i]["_segment"] = rest
                    if not rest:
                        del ops[i]["_segment"]
            new.chains[name] = ops
        return new

    # ------------------------------------------------------------ the recipe
    def recipe(self):
        """A JSON-serialisable dict of the whole acquisition, the ground truth for every
        occurrence: what each one drew is a function of it (`occurrences()`)."""
        from wfmsynth import __version__
        from .compose import Signal
        seg = {"n": self.n, "arrival": dict(self.arrival), "trigger": dict(self.trigger),
               "ops": [dict(o) for o in self.chains[MAIN]]}
        if self.choose:
            seg["variants"] = {name: {**self.choose[name],
                                      "ops": [dict(o) for o in self.chains[name]]}
                               for name in self.choose}
        return {"wfmsynth_version": __version__, "seed": self.seed,
                "grid": Signal(grid=self.grid).recipe()["grid"], "segments": seg}

    def to_json(self):
        """Canonical, key-sorted JSON of the recipe."""
        return json.dumps(self.recipe(), sort_keys=True)

    def sha256(self):
        """One content hash for the acquisition: equal hashes, equal samples."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_recipe(cls, r):
        """Rebuild the acquisition from its recipe; `.waveform()` reproduces it bit for bit."""
        from .compose import _warn_on_numeric_drift
        if "segments" not in r:
            raise ValueError("Segments.from_recipe: this recipe has no `segments`, so it is "
                             "one record: Signal.from_recipe (or Channels.from_recipe).")
        s = r["segments"]
        variants = {name: {**{k: v[k] for k in ("p", "at") if k in v}, "chain": {"ops": v["ops"]}}
                    for name, v in s.get("variants", {}).items()}
        rec = cls({"ops": s["ops"]}, n=s["n"], arrival=s["arrival"], trigger=s["trigger"],
                  seed=r["seed"], grid=Grid(**r["grid"]), variants=variants or None)
        _warn_on_numeric_drift(r.get("wfmsynth_version"),
                               [o for ops in rec.chains.values() for o in ops])
        return rec


def _trend_at(spec, k, n, t_s):
    """A trend's value at occurrence `k` of `n`, which arrived at `t_s`."""
    if spec["trend"] == "ramp":
        if "stop" in spec:
            return float(spec["start"] + (spec["stop"] - spec["start"]) * (k / (n - 1) if n > 1
                                                                            else 0.0))
        return float(spec["start"] + spec["rate_per_s"] * t_s)
    late = k >= spec["at"] if "at" in spec else t_s >= spec["at_s"]
    return spec["after"] if late else spec["before"]


def segment_params(recipe, op, which=None):
    """The parameters of one op as EACH occurrence of a `Segments` recipe rendered it -- the
    labelling route, as `op_params` is for one record:

        amp = [p["by"] for p in ws.segment_params(seg.recipe(), "gain", which=0)]

    One dict per occurrence, in order; None for an occurrence whose variant has no such op.
    Raises `KeyError` when no occurrence has it."""
    from .compose import op_params
    rec = Segments.from_recipe(recipe)
    out = []
    for occ in rec._plan(Streams(rec.seed)):
        try:
            p = op_params({"ops": occ["ops"]}, op, which)
        except KeyError:
            p = None
        out.append(p)
    if all(p is None for p in out):
        raise KeyError(f"no {op!r} op in any chain of this acquisition.")
    return out


__all__ = ["Segments", "segment_params", "MAIN"]

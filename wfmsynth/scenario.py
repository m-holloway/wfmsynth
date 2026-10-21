"""
wfmsynth.scenario — who held the net when, and what that does to it.

`net.solve` takes a schedule: a flat list of `{"t_s", "id", "state"}` against stable element and
driver ids. That is the right interface for a solver and the wrong one to write by hand. A bus
transaction is "the host held SDA low for this bit"; turning that into events means knowing that
the host's driver turns on a little before the nominal edge, that its output stage releases a
little after, that letting go rings unless somebody else grabbed the line in the meantime. Write
that per net and the protocol rules, the driver physics and the event timing end up in one
routine, and none of it transfers to the next net.

This module is the small layer between. It adds NO electrical physics -- it emits the same event
list a careful person would have written -- and it is pure data in, pure data out:

    holds  = [Hold("host/sda", 1.0e-6, 2.25e-6, bit=0), ...]   # who held the net, when
    agents = {"host/sda": Agent(during={"d_host": "driving"},  # what holding DOES
                                after={"d_host": "hiz"},
                                lead=20e-9, lag=35e-9)}
    events = schedule(holds, agents)                           # -> net.solve(schedule=...)

THE HOLD IS THE SCENARIO AND THE LABEL
--------------------------------------
One list says what to render and what the record contains, so they cannot disagree. `Hold`
carries arbitrary tags (`bit=0`, `field="addr"`, `byte=0x3c`), `hold_windows` turns the list into
the window dicts `events.label_windows` already consumes, and a fault is an edit to the holds --
which makes the edit its own label.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Protocol encoders, traffic and gap generators, fault injection and parameter priors. Those are
caller code: a plain class that returns a list of holds, with randomness drawn from the existing
role streams. `examples/i2c_bus.py` is a worked one, about forty lines, and it is an example
rather than a module because which bits a standard puts on a wire has a revision date and a
document behind it, while "an agent held the net and its driver turned on early" does not. That
is the same boundary `CONTRIBUTING.md` draws for patterns: mechanisms in the library, standards
knowledge in the caller's own code.
"""
from __future__ import annotations

import numpy as np

# `t0 + i*dt + dt` and `t0 + (i + 1)*dt` are the same instant and differ in the last bit, so two
# consecutive bit-times generated the obvious way do not compare equal. An exact `>=` would then
# refuse to merge them, and a bus idle for eight bits would release and re-acquire eight times --
# eight spurious edges, each one a transient the record did not earn. The slack is relative
# because these are seconds and a record can sit anywhere from nanoseconds to minutes.
_TOUCH_REL = 1e-12


def _touching(t1, t0, gap):
    """Does an interval ending at `t1` run into one starting at `t0`?"""
    return (t0 - t1) <= gap + _TOUCH_REL * max(abs(t0), abs(t1))


class Hold:
    """One agent occupying the net over ``[t0, t1)``, plus whatever tags label it.

    ``Hold("host/sda", 1e-6, 2e-6, bit=0, field="addr")``. The tags are free-form and travel
    into `hold_windows`, so the thing that produced the waveform is the thing that labels it.
    ``who`` names an `Agent`, which is what turns the hold into events.
    """
    __slots__ = ("who", "t0", "t1", "tag")

    def __init__(self, who, t0, t1, **tag):
        t0, t1 = float(t0), float(t1)
        if not t1 > t0:
            raise ValueError(f"Hold({who!r}): t1 must be after t0, got t0={t0!r} t1={t1!r}. A "
                             f"zero-length hold is an event, not a hold -- pass it to the net's "
                             f"`schedule=` directly.")
        self.who, self.t0, self.t1, self.tag = str(who), t0, t1, dict(tag)

    def to_dict(self):
        # Flat, like a recipe op. No tag can shadow the three fixed names: they are positional
        # parameters of __init__, so `Hold("a", 0, 1, who="b")` is already a TypeError from
        # Python itself and never reaches here.
        return {"who": self.who, "t0": self.t0, "t1": self.t1, **_jsonable(self.tag)}

    @classmethod
    def from_dict(cls, d):
        if isinstance(d, Hold):
            return d
        d = dict(d)
        return cls(d.pop("who"), d.pop("t0"), d.pop("t1"), **d)

    def __eq__(self, other):
        return isinstance(other, Hold) and self.to_dict() == other.to_dict()

    def __hash__(self):
        return hash((self.who, self.t0, self.t1, tuple(sorted(self.tag.items()))))

    def __repr__(self):
        tag = "".join(f", {k}={v!r}" for k, v in self.tag.items())
        return f"Hold({self.who!r}, {self.t0!r}, {self.t1!r}{tag})"


class Agent:
    """A party on the net, configured by behaviour rather than by a list of events.

    ``during``   element/driver id -> what it does while this agent holds, applied ``lead``
                 early. The turn-on row: its driver conducting, terminations on its own side.
    ``after``    the same, once it lets go, applied ``lag`` late. The idle row.
    ``lead``     seconds the ``during`` row precedes the hold. An output stage is already
                 conducting before the logic edge it is responding to.
    ``lag``      seconds the ``after`` row follows the hold, for the same reason in reverse.
    ``then``     extra timed rows after the release: ``(dt, id, state)``, or the dict form
                 ``{"dt":, "id":, "state":}``. This is where a release RESPONSE goes -- an
                 under-damped shelf as an output stage lets go, a termination settling.
    ``unless``   agent names whose holds suppress ``then``. A hand-over between two drivers
                 should not fire a release response: nobody released the net, it changed
                 hands. The agent's own holds count too, and the whole reaction is suppressed
                 together rather than entry by entry -- see `_suppressed` for why.

    A state is a string (``"driving"``, ``"hiz"``, ``"closed"``, ``"open"``) or a dict in the
    event vocabulary (``{"g": 0.05}`` for a conductance, ``{"src": 1.8}`` for a driver's level,
    or both together with a state). A bare number is refused: a conductance and a level are not
    distinguishable by type, and guessing wrong is silent.
    """
    __slots__ = ("during", "after", "lead", "lag", "then", "unless")

    def __init__(self, during=None, after=None, lead=0.0, lag=0.0, then=(), unless=()):
        self.during = {str(k): _payload(k, v) for k, v in dict(during or {}).items()}
        self.after = {str(k): _payload(k, v) for k, v in dict(after or {}).items()}
        self.lead, self.lag = float(lead), float(lag)
        if self.lead < 0 or self.lag < 0:
            raise ValueError(f"Agent: lead and lag are durations and cannot be negative, got "
                             f"lead={self.lead!r} lag={self.lag!r}. To place a row BEFORE the "
                             f"release rather than after it, shorten the hold.")
        self.then = tuple(_then_entry(t) for t in (then or ()))
        self.unless = tuple(str(u) for u in (unless or ()))

    def to_dict(self):
        d = {}
        if self.during:
            d["during"] = {k: dict(v) for k, v in self.during.items()}
        if self.after:
            d["after"] = {k: dict(v) for k, v in self.after.items()}
        if self.lead:
            d["lead"] = self.lead
        if self.lag:
            d["lag"] = self.lag
        if self.then:
            d["then"] = [dict(t) for t in self.then]
        if self.unless:
            d["unless"] = list(self.unless)
        return d

    @classmethod
    def from_dict(cls, d):
        if isinstance(d, Agent):
            return d
        return cls(**dict(d))

    def __eq__(self, other):
        return isinstance(other, Agent) and self.to_dict() == other.to_dict()

    def __repr__(self):
        return f"Agent({', '.join(f'{k}={v!r}' for k, v in self.to_dict().items())})"


def _jsonable(v):
    """numpy scalars are not JSON-serialisable, and a tag is usually computed. The recipe is
    provenance only if it survives `json.dumps`, so this is not cosmetic."""
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _payload(target, v):
    """One `during`/`after` value -> the event body that will be merged with `t_s` and `id`."""
    if isinstance(v, str):
        return {"state": v}
    if isinstance(v, dict):
        body = {k: _jsonable(x) for k, x in v.items()}
        unknown = sorted(set(body) - {"state", "g", "src"})
        if unknown:
            raise ValueError(f"Agent: row for {target!r} has key(s) {unknown}; the event "
                             f"vocabulary is 'state', 'g' (conductance, siemens) and 'src' (a "
                             f"driver's level, volts). See wfmsynth/net.py for what each means "
                             f"and which ones this solver tier can represent.")
        if not body:
            raise ValueError(f"Agent: row for {target!r} is empty; say what it does")
        return body
    raise ValueError(f"Agent: row for {target!r} is {v!r}. A bare number is ambiguous -- a "
                     f"conductance and a voltage are both floats -- so write {{'g': {v!r}}} or "
                     f"{{'src': {v!r}}}, or a state string such as 'driving'.")


def _then_entry(t):
    if isinstance(t, dict):
        d = dict(t)
        missing = sorted({"dt", "id"} - set(d))
        if missing:
            raise ValueError(f"Agent: a `then` entry needs {missing}; got {t!r}")
        dt, target = d.pop("dt"), d.pop("id")
        body = _payload(target, d)
    else:
        try:
            dt, target, state = t
        except (TypeError, ValueError):
            raise ValueError(f"Agent: a `then` entry is (dt, id, state) or "
                             f"{{'dt':, 'id':, 'state':}}; got {t!r}") from None
        body = _payload(target, state)
    dt = float(dt)
    if dt < 0:
        raise ValueError(f"Agent: a `then` entry fires AFTER the release, so dt cannot be "
                         f"negative; got {dt!r} for {target!r}")
    return {"dt": dt, "id": str(target), **body}


def _spans(holds, who, lead=0.0, lag=0.0, gap=0.0):
    """One agent's sorted, unioned `(t0, t1)` intervals, each widened by its lead and lag."""
    xs = sorted((h.t0 - lead, h.t1 + lag) for h in holds if h.who == who)
    out = []
    for a, b in xs:
        if out and _touching(out[-1][1], a, gap):
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def merge_holds(holds, gap=0.0):
    """Union each agent's touching holds, so a run of consecutive identical bits is one hold
    rather than a release and re-acquire at every bit boundary.

    `gap` widens what counts as touching, in seconds: two holds separated by less than it are
    one. Zero (the default) still merges holds that abut, including the float-noise case where
    one bit-time's end and the next one's start differ in the last bit.

    Merging is for the SCHEDULE, where tags play no part. The merged hold keeps only the tags
    its parts agreed on, key and value, and the parts are not recoverable from the result -- so
    label from the unmerged list (that is what `hold_windows` expects) and merge for the solve.
    """
    hs = [Hold.from_dict(h) for h in holds]
    out = []
    for who in dict.fromkeys(h.who for h in hs):          # first-seen order, deterministic
        mine = sorted((h for h in hs if h.who == who), key=lambda h: (h.t0, h.t1))
        for a, b in _spans(mine, who, gap=gap):
            parts = [h for h in mine if h.t0 < b and h.t1 > a]
            common = {k: v for k, v in parts[0].tag.items()
                      if all(k in p.tag and p.tag[k] == v for p in parts[1:])}
            out.append(Hold(who, a, b, **common))
    return sorted(out, key=lambda h: (h.t0, h.who))


def schedule(holds, agents, rest=True, gap=0.0):
    """Compile holds plus agent behaviour into a `net.solve` event list.

    Pure, deterministic and JSON-able: the same inputs give the same list, element for element,
    and nothing here touches a waveform or a grid. Returns `[{"t_s", "id", ...}]` sorted by
    time, with ties left in emission order -- which `net.solve` applies as ONE step anyway,
    because two events inside one sample are one configuration change and not a sequence.

    `rest=True` (the default) emits every agent's `after` row at t=0, so the net starts in the
    idle configuration whatever the drivers happen to declare in the net description. Without
    it a driver declared `state="driving"` would drive from the first sample until its agent's
    first hold ended, which renders, digests stably, and is not the scenario that was asked
    for. Pass `rest=False` when the declared states ARE the initial condition you want.

    `gap` is forwarded to `merge_holds`.
    """
    hs = [Hold.from_dict(h) for h in holds]
    ags = {str(k): Agent.from_dict(v) for k, v in dict(agents).items()}
    unknown = sorted({h.who for h in hs} - set(ags))
    if unknown:
        raise ValueError(f"scenario: hold(s) name agent(s) {unknown}, which are not configured. "
                         f"Known agents: {sorted(ags)}. An agent says what holding the net DOES; "
                         f"a hold with none would render as nothing at all.")
    # effective occupancy: when each agent is actually ON the net, lead and lag included. This
    # is what `unless` is tested against, and what gets merged -- two holds that do not touch
    # can still overlap once lead and lag widen them, and then the agent never let go.
    span = {nm: _spans(hs, nm, ags[nm].lead, ags[nm].lag, gap) for nm in ags}

    out = []
    for nm, ag in ags.items():                            # dict order: deterministic
        if rest:
            # The rest row is the agent LONG AFTER a release, which is `after` with the whole
            # release reaction applied on top -- not `after` alone. They differ whenever `then`
            # is a settling sequence: a translator that stays conducting for a while after the
            # line is let go has `after` = closed and `then` = open, and seeding the record
            # with `after` would start an idle bus with the switch shut.
            for tid, body in ag.after.items():
                out.append({"t_s": 0.0, "id": tid, **body})
            for th in sorted(ag.then, key=lambda t: t["dt"]):
                out.append({"t_s": 0.0, **{k: v for k, v in th.items() if k != "dt"}})
    for nm, ag in ags.items():
        for a, b in span[nm]:
            for tid, body in ag.during.items():
                out.append({"t_s": max(0.0, a), "id": tid, **body})
            for tid, body in ag.after.items():
                out.append({"t_s": max(0.0, b), "id": tid, **body})
            if _suppressed(nm, ag, b, span):
                continue
            for th in ag.then:
                out.append({"t_s": max(0.0, b + th["dt"]),
                            **{k: v for k, v in th.items() if k != "dt"}})
    return sorted(out, key=lambda ev: ev["t_s"])          # stable: ties keep emission order


def _suppressed(nm, ag, b, span):
    """Does the release reaction run at all, having let go at `b`?

    ALL of `then` or none of it, decided once. A reaction half-applied leaves the net in a
    configuration no row describes -- a two-step response whose second step is dropped parks
    the thing it was settling in its intermediate state, forever. So the test is over the
    reaction's whole window `[b, b + max dt]`: if the agent itself, or anyone it named in
    `unless`, is on the net at any point in it, none of the reaction is emitted and the `after`
    row stands. Which is why `after` has to be a complete idle row on its own, and not the
    first step of a sequence `then` finishes.

    The agent's own span is tested too, and is not vacuous: its CURRENT span ends at `b` (so it
    never suppresses itself at the instant it let go), but a hold it takes again before the
    reaction has finished would be silently undone by the tail of a reaction to the previous
    release.
    """
    if not ag.then:
        return False
    end = b + max(th["dt"] for th in ag.then)
    return any(s <= end and e > b
               for h in (nm, *ag.unless) for s, e in span.get(h, ()))


def hold_windows(holds, fs, n=None):
    """Holds -> the window dicts `events.label_windows` consumes, one per hold, in time order.

    Each window carries `who` and the hold's tags alongside `{i, start, stop, center}`.
    `label_windows` keys its rows by the same `i`, so join on it to get the hold's own fields
    back beside the measured ones -- it returns the fields it defines and drops the rest rather
    than guessing which extras are labels.

    Pass the UNMERGED holds: merging is for the solve, and a merged hold has only the tags its
    parts agreed on.
    """
    from .net import sample_at                # the one place the sample-boundary rule is written
    hs = sorted((Hold.from_dict(h) for h in holds), key=lambda h: (h.t0, h.who))
    fs = float(fs)
    out = []
    for i, h in enumerate(hs):
        lo, hi = sample_at(h.t0, fs), sample_at(h.t1, fs)
        if n is not None:
            lo, hi = max(0, lo), min(int(n), hi)
        if hi <= lo:
            continue
        out.append({"i": i, "start": lo, "stop": hi, "center": 0.5 * (lo + hi),
                    "who": h.who, "t0": h.t0, "t1": h.t1, **_jsonable(h.tag)})
    return out

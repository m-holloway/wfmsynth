"""
wfmsynth.subchain -- a sub-chain is a value.

Several ops take a SECOND signal: a crosstalk aggressor, a modulating message, another driver on
the net. Each used to take it in its own shape, and none could say "this other lane, after its
own channel". Here is the one form every such parameter takes, and the one resolver that renders
it:

    {"ops": [...]}            a chain that is PART OF THIS RECORD: rendered on this record's grid,
                              its randomness drawn from this record's seed under the namespace of
                              the parameter holding it (`crosstalk/2/aggressor/jitter/0`).
    {"from": <recipe>}        ANOTHER RECORD, as recorded: its own seed and its own grid (which
                              must have this record's sample rate and length). Bit-identical to
                              `Signal.from_recipe(recipe).waveform()`.
    a Signal                  the pleasant way to write the first form; the builder stores its
                              op list, so the recipe stays JSON and replays with no extra files.

    lane = Signal(grid=g).carrier("nrz", n_ui=512, seed=9).lossy(loss_db=6.0, loss_at_ghz=8.0)
    sig = Signal(seed=1, grid=g).carrier("nrz", n_ui=512).crosstalk(aggressor=lane, coupling=0.1)
    print(sig.summary())                               # the chain as a tree
    sig.subchain("crosstalk/1/aggressor")              # the aggressor exactly as it was coupled

A sub-chain is never resampled, truncated or stripped of an imaginary part to make it fit: one
that does not land on the record's own grid is refused by name. `Signal.roles()` and
`Signal.contrast()` reach inside under the same namespace, and an `events` op inside a sub-chain
surfaces in the parent's event list with `Event.at` set to that namespace.

FOR AN OP THAT TAKES A SIGNAL the whole integration is one line in its executor,

    aggr = SUB.signal(p, "aggressor", streams, grid, idx, n=len(x))

and nothing else: the namespace is `<op>/<idx>/<parameter>`, and the builder, `roles()`,
`summary()`, `subchain()` and the event list all find the parameter by its SHAPE (a dict with
`ops`, or with a recipe under `from`), so there is no table to register it in. `signals()` is the
same for a list parameter, and a list entry may hold one beside its own parameters
(`sequence`'s `segments[k]["chain"]`, namespace `sequence/<idx>/segments/<k>/chain`). The result is a REAL waveform on the record's own grid: an I/Q
sub-chain ends on `upconvert` (the domain rule, `wfmsynth.domain`, applies inside a sub-chain as
it does outside). A sub-chain starts in the domain ITS OWN ops need, whatever record it sits in
-- `Signal().carrier("sine", ...)` is a voltage inside an I/Q record -- and the parameter that
takes a PEER of an I/Q record (a `sequence` segment, a `combine` entry) asks for an envelope
instead with `domain="iq"`.

A TOLERANCE inside a sub-chain (`wfmsynth.tolerance`) belongs to the record it joins: a Signal's
resolved tolerances are restated on the way in and drawn under the outer seed, so `unit()` moves
the nested parts with everything else. `{"from": recipe}` is pinned, its unit included.
"""
from __future__ import annotations

import dataclasses
import json
import os
import warnings

import numpy as np

from . import domain as DOMAIN
from . import opkeys as OPKEYS
from . import tolerance as TOL
from .grid import Grid
from .streams import Streams


class Namespaced:
    """A `Streams` view that prefixes every role name.

    A nested chain asks for the same role names as the outer one -- `jitter/0`, `noise/1` --
    keyed by its own op index, so handing it the parent's `Streams` would give a sub-chain's
    jitter the parent op 0's draws. Prefixing keeps each sub-chain's randomness independent of
    its siblings and of the chain it sits in, while still deriving from the record's single
    seed, so one `Signal(seed=)` moves the whole record coherently and `Streams.reroll` can
    still name one factor inside it exactly (`reroll("crosstalk/2/aggressor/jitter/0")`).

    `sink` is where a render that is collecting events puts them; a sub-chain reads it off the
    view it was handed so that its own events reach the record's event list. `record` is the
    several-channel render this chain is part of, if any (`wfmsynth.channels`): it travels the
    same way, so a `use` op inside a sub-chain can still find the chain it names.
    """
    __slots__ = ("_st", "_pre", "sink", "record")

    def __init__(self, st, pre, sink=None):
        self._st, self._pre = st, pre
        self.sink = sink if sink is not None else getattr(st, "sink", None)
        self.record = getattr(st, "record", None)

    def role(self, name):
        return self._st.role(self._pre + str(name))


def _overrides(st, pre):
    """The re-rolls a `Streams` carries for roles under `pre`, with the prefix removed.

    A `{"from": recipe}` sub-chain has its own seed, so it gets its own `Streams` rather than a
    view of the parent's. These are what make `contrast("<namespace>/jitter/0")` reach it anyway.
    """
    while isinstance(st, Namespaced):
        st, pre = st._st, st._pre + pre
    return {k[len(pre):]: v for k, v in st._overrides.items() if k.startswith(pre)}


class _Asked:
    """The `asked` of a `{"from": recipe}` sub-chain's own `Streams`: each role it requests is
    noted on the record's, under the sub-chain's namespace."""

    def __init__(self, st, pre):
        while isinstance(st, Namespaced):
            st, pre = st._st, st._pre + pre
        self._to, self._pre = st.asked, pre

    def add(self, name):
        self._to.add(self._pre + name)


def is_subchain(v):
    """True for a value in the nested-signal form: `{"ops": [...]}` or `{"from": <recipe>}`."""
    return isinstance(v, dict) and (isinstance(v.get("ops"), (list, tuple))
                                    or isinstance(v.get("from"), dict))


# ------------------------------------------------------------------ on the way into a recipe
def _signal_ops(sig, parent, who):
    """A `Signal` used as a sub-chain -> its op list, refusing what the op list cannot carry."""
    if sig.lead_in or sig.lead_out is not None:
        raise ValueError(
            f"{who} is a Signal with a lead-in, which belongs to a whole record and not to an "
            f"op list. Pass {{'from': that_signal.recipe()}} to embed it as its own record, "
            f"lead-in included, or drop the lead-in.")
    if sig.seed not in (0, parent.seed):
        raise ValueError(
            f"{who} is a Signal with seed={sig.seed}, but a sub-chain given as a Signal becomes "
            f"part of this record and draws from THIS record's seed ({parent.seed}), so its own "
            f"seed would be ignored. Leave the seed off to make it part of this record, or pass "
            f"{{'from': that_signal.recipe()}} to embed it as its own record with its own seed.")
    # the same samples and rate is the same grid here: whether the sub-chain is a voltage or
    # an envelope is what its own ops say (`domain.grid_for`), not the record it joins
    if sig.grid is not None and parent.grid is not None and dataclasses.replace(
            sig.grid, domain=parent.grid.domain, fc_hz=parent.grid.fc_hz) == parent.grid:
        return TOL.stated(sig.ops)
    if sig.grid is not None and sig.grid != parent.grid:
        raise ValueError(
            f"{who} is a Signal on {_grid_words(sig.grid)}, but it would be rendered on this "
            f"record's grid, {_grid_words(parent.grid)}; nothing is resampled. Build it on this "
            f"record's grid (or with no grid). A lane at another symbol rate on the same sample "
            f"rate and length is {{'from': that_signal.recipe()}}.")
    return TOL.stated(sig.ops)        # its tolerances are drawn by the record it joins


def _grid_words(g):
    if g is None:
        return "no grid"
    return f"fs={g.fs:g} Hz, baud={'none' if g.baud is None else format(g.baud, 'g')}, n={g.n}"


def _flat(v, parent, who, level=0):
    # level 0: an op parameter; 1: an entry of a list parameter; 2: a value inside such an entry
    from .compose import Signal
    if isinstance(v, Signal):
        return {"ops": _signal_ops(v, parent, who)}
    if isinstance(v, dict):
        if isinstance(v.get("ops"), Signal):
            return {**v, "ops": _signal_ops(v["ops"], parent, who)}
        frm = v.get("from")
        if isinstance(frm, Signal):
            return {**v, "from": frm.recipe()}
        if isinstance(frm, (str, os.PathLike)):
            # A path would leave the record depending on a file its digest does not cover, so
            # the recipe is read HERE and embedded.
            try:
                with open(frm) as fh:
                    r = json.load(fh)
            except (OSError, ValueError):
                r = None
            if not (isinstance(r, dict) and "ops" in r and "seed" in r):
                raise ValueError(f"{who} has from={os.fspath(frm)!r}, which is not the path of "
                                 f"a saved recipe (JSON with 'ops' and 'seed', as "
                                 f"`Signal.to_json()` writes). `from` takes a recipe, a Signal "
                                 f"or such a path.")
            return {**v, "from": r}
        if level == 1 and "ops" not in v and "from" not in v:
            # a list entry that HOLDS a sub-chain beside its own parameters: a `sequence`
            # segment is {"chain": <sub-chain>, "duration_s": ...}
            new = {k: _flat(e, parent, f"{who}: `{k}`", 2) for k, e in v.items()}
            return new if any(new[k] is not v[k] for k in v) else v
        return v
    if isinstance(v, (list, tuple)) and level == 0:
        new = [_flat(e, parent, f"{who}[{i}]", 1) for i, e in enumerate(v)]
        return new if any(a is not b for a, b in zip(new, v)) else v
    return v


def flatten(op, params, parent):
    """An op's parameters on the way into a recipe, with every sub-chain made JSON.

    A `Signal`, `{"ops": Signal}`, `{"from": Signal}` and `{"from": <path>}` are convenient to
    write and are not provenance: a recipe has to serialise and to replay with no extra files.
    So they are resolved here, once, by `Signal._add`, for every op. Anything else -- including
    every parameter of every recipe that predates sub-chains -- passes through as the same
    object.
    """
    if op == "net":                 # a net's sub-chains sit inside `drivers`; it flattens them
        return params
    return {k: _flat(v, parent, f"{op}: `{k}`") for k, v in params.items()}


# ------------------------------------------------------------------ rendering
_NO_CHAIN = object()
_REQUIRED = object()


def _chain(spec, streams, grid, at, who):
    """``(Signal, streams)`` for a sub-chain spec: the chain to run and what it draws from."""
    from .compose import Signal, _EXEC
    if spec.get("ops") is not None:
        ops = [dict(o) for o in spec["ops"]]
        if not ops:
            raise ValueError(f"{who} has an empty `ops`, so it renders nothing.")
        for o in ops:
            if o.get("op") not in _EXEC:
                raise ValueError(f"{who} has op {o.get('op')!r} in its chain, which does not "
                                 f"exist. Known ops: {sorted(_EXEC)}")
            OPKEYS.check(o["op"], o)         # the same gate the builders go through
        sub = Signal(grid=DOMAIN.grid_for(ops, grid))   # a voltage or an envelope: its ops say
        sub.ops = ops
        return sub, Namespaced(streams, at + "/")
    r = spec["from"]
    if not (isinstance(r, dict) and "ops" in r and "seed" in r):
        raise ValueError(
            f"{who} has from={r!r}; it is a RECIPE -- the dict `Signal.recipe()` returns. A "
            f"path is read and embedded by the Signal builder so the record's digest covers "
            f"it; in a hand-written recipe embed it yourself with json.load(open(path)).")
    rg = r.get("grid")
    if rg and grid is not None and (rg["fs"] != grid.fs or rg["n"] != grid.n):
        raise ValueError(
            f"{who} is a record on fs={rg['fs']:g} Hz, n={rg['n']}, and this record is on "
            f"fs={grid.fs:g} Hz, n={grid.n}. A sub-chain is never resampled or cut to fit: "
            f"re-render that record on this grid (same fs and n; its baud may differ), or "
            f"resample it yourself and bring it in with `capture`.")
    sub = Signal.from_recipe(r)
    own = Streams(sub.seed, _overrides(streams, at + "/"))
    own.asked = _Asked(streams, at + "/")           # so the record knows what was drawn here
    return sub, own


def same_grid(y, n, who, sub=None, grid=None, domain=DOMAIN.REAL):
    """Refuse a rendered sub-chain that is not a waveform of `n` samples at the grid's rate, in
    the `domain` the parameter takes: a real voltage, or (`"iq"`) a complex envelope."""
    if domain == DOMAIN.IQ:
        if not np.iscomplexobj(y):
            raise ValueError(
                f"{who} renders a REAL waveform, and here it joins an I/Q record as a peer of "
                f"its envelope. Give an I/Q chain about the same centre frequency (one that "
                f"starts with `iq`, or ends on `downconvert`).")
        y = np.asarray(y, complex)
    else:
        if np.iscomplexobj(y):
            raise ValueError(
                f"{who} renders a COMPLEX waveform (an I/Q envelope or an optical field), and "
                f"taking it as a signal here would drop its imaginary part. End the sub-chain "
                f"on `upconvert` (I/Q) or on `photodetect` (optical).")
        y = np.asarray(y, float)
    if y.shape != (n,):
        raise ValueError(
            f"{who} rendered {y.shape[0] if y.ndim == 1 else y.shape} samples but this record "
            f"has {n}. A sub-chain is a waveform on the SAME grid as the record it joins and "
            f"is never resampled or truncated to fit -- give it a grid that agrees, and keep "
            f"the ops that resample or truncate a record (`acquire`, `digitize(n_out=)`, "
            f"`sample_clock`) outside it.")
    if sub is not None and grid is not None and sub.grid is not None and any(
            o["op"] == "acquire" or (o["op"] == "digitize" and o.get("n_out")) for o in sub.ops):
        fs = sub.stored_grid(y).fs
        if fs != grid.fs:
            raise ValueError(
                f"{who} stores its waveform at {fs:g} Sa/s (it ends in an `acquire` or a "
                f"`digitize(n_out=)`), and this record is at {grid.fs:g} Sa/s. The sample "
                f"counts agree but the rates do not, so it is on a different timebase; keep "
                f"the acquisition outside the sub-chain.")
    return y


def render(spec, streams, grid, at, n, who, chain=_NO_CHAIN, carrier=None, domain=DOMAIN.REAL,
           is_record=False):
    """One nested-signal spec -> a real waveform of `n` samples. The resolver.

    `domain="iq"` is for the parameter that takes a PEER of an I/Q record (a segment, a
    `combine` entry): the sub-chain must then render a complex envelope instead.

    `at` is the namespace (`crosstalk/2/aggressor`): the prefix of every role the sub-chain
    draws and the `Event.at` of every event it places. `who` names the parameter in a refusal.
    Those events are marked as NOT on the record (`Event.on_record`) -- they happened to the
    aggressor, the enable -- unless `is_record` says the sub-chain's samples are the record's
    (a `sequence` segment, a `switch` input).

    Two bodies are for the ops that had a form before this one existed. `carrier=` is a callable
    taking a bare carrier spec (`{"kind": "nrz", "n_ui": 64}`), which keeps that shorthand and
    the role names it always drew from; without it a bare carrier spec is refused. `chain=` is
    the waveform arriving at the op, which `{"from": "chain"}` then names (a `net` driver that
    IS the chain).
    """
    if not isinstance(spec, dict):
        raise ValueError(
            f"{who} is {type(spec).__name__}; it is a sub-chain -- {{'ops': [...]}}, "
            f"{{'from': <recipe>}}, or a Signal handed to the builder. An array does not belong "
            f"in a recipe; bring a measured waveform in with a `capture` op inside 'ops'.")
    frm, ops = spec.get("from"), spec.get("ops")
    if frm is None and ops is None:
        if carrier is None:
            raise ValueError(
                f"{who} needs exactly one of 'ops' (its own chain) or 'from' (a recipe); it "
                f"has keys {sorted(spec)}. One carrier is "
                f"{{'ops': [{{'op': 'carrier', 'kind': ...}}]}}, or Signal().carrier(...).")
        return np.asarray(carrier(spec), float)
    bodies = [nm for nm, v in (("from", frm), ("ops", ops), ("kind", spec.get("kind")))
              if v is not None]
    if len(bodies) != 1:
        raise ValueError(f"{who} needs exactly one of 'ops' (its own chain), 'from' (a recipe"
                         f"{', or the chain this op is in' if chain is not _NO_CHAIN else ''})"
                         f"{' or a carrier spec by its kind' if carrier else ''}; it has "
                         f"{bodies}.")
    extra = sorted(k for k in spec if k not in ("from", "ops"))
    if extra:
        raise ValueError(f"{who} is a sub-chain and takes only "
                         f"{'ops' if ops is not None else 'from'!r}; it also has {extra}, "
                         f"which nothing would read. Parameters go on the ops inside it.")
    if isinstance(frm, str) and chain is not _NO_CHAIN:
        if frm != "chain":
            raise ValueError(f"{who} has from={frm!r}; as a word the only value is 'chain', "
                             f"meaning the waveform arriving at this op. Another record is its "
                             f"recipe dict.")
        if chain is None:
            raise ValueError(f"{who} is from='chain', but this is the first op so no waveform "
                             f"has arrived. Put a carrier before it, or give it its own 'ops'.")
        return np.asarray(chain, float)
    sub, st = _chain(spec, streams, grid, at, who)
    if domain == DOMAIN.IQ and grid is not None and sub.ops:
        fc, own = getattr(grid, "fc_hz", None), DOMAIN.trace(sub.ops, sub.grid)[-1][1]
        if fc is not None and own is not None and own != fc:
            raise ValueError(
                f"{who} is an envelope about {own:g} Hz and the record it joins is about "
                f"{fc:g} Hz. Samples of two envelopes mean the same thing only about the SAME "
                f"centre; joined as they are, this one would land co-channel. Render it "
                f"about {fc:g} Hz and put freq_offset(f_hz={own - fc:g}) in its chain, which "
                f"places it where it is and checks that the grid has room for it.")
    sink = getattr(streams, "sink", None)
    if sink is None:
        y = sub._run(st)
    else:
        y, placed = sub._run(st, collect_events=True)
        for e in placed:
            e.at = at + ("/" + e.at if e.at else "")
            e.on_record = e.on_record and is_record
        sink.extend(placed.events)
    return same_grid(y, n, who, sub=sub, grid=grid, domain=domain)


def signal(p, name, streams, grid, idx, n=None, carrier=None, default=_REQUIRED,
           domain=DOMAIN.REAL):
    """The waveform held by op `p`'s parameter `name`: what an executor calls. ONE LINE.

        aggr = SUB.signal(p, "aggressor", streams, grid, idx, n=len(x))

    Returns a float array of exactly `n` samples (`grid.n` when `n` is omitted), or refuses by
    name. The namespace is `<op>/<idx>/<name>`. `default` is the spec used when the parameter is
    absent; without one an absent parameter is refused. `carrier` is the legacy carrier-spec
    shorthand, see `render`.
    """
    who = f"{p['op']}: `{name}`"
    spec = p.get(name)
    if spec is None:
        if default is _REQUIRED:
            raise ValueError(f"{p['op']} needs {name}=<a sub-chain>: a Signal, "
                             f"{{'ops': [...]}} or {{'from': <recipe>}}.")
        spec = default
    return render(spec, streams, grid, f"{p['op']}/{idx}/{name}", _length(n, grid, who), who,
                  carrier=carrier, domain=domain)


def signals(p, name, streams, grid, idx, n=None, domain=DOMAIN.REAL):
    """A LIST parameter of sub-chains -> a list of waveforms; namespaces `<op>/<idx>/<name>/<k>`."""
    who = f"{p['op']}: `{name}`"
    specs = p[name]
    if not isinstance(specs, (list, tuple)):
        raise ValueError(f"{who} is a list of sub-chains, one per signal; got "
                         f"{type(specs).__name__}.")
    n = _length(n, grid, who)
    return [render(s, streams, grid, f"{p['op']}/{idx}/{name}/{k}", n, f"{who}[{k}]",
                   domain=domain)
            for k, s in enumerate(specs)]


def _length(n, grid, who):
    if n is None and grid is not None:
        n = grid.n
    if n is None:
        raise ValueError(f"{who} is rendered on the record's grid and there is none -- pass "
                         f"Signal(grid=Grid(fs=..., n=...)).")
    return int(n)


# ------------------------------------------------------------------ finding them in a recipe
def walk(ops):
    """Yield ``(op_index, namespace, label, spec)`` for every sub-chain held directly by `ops`.

    Found by shape, so an op that takes a sub-chain needs no registration. A `net` holds its
    sub-chains inside its drivers, under the namespace it has always used (`net/<idx>/<id>`).
    """
    for i, o in enumerate(ops):
        if o.get("op") == "net":
            for d in o.get("drivers") or ():
                if isinstance(d, dict) and is_subchain(d.get("source")):
                    yield i, f"net/{i}/{d.get('id')}", f"driver {d.get('id')!r} source", d["source"]
            continue
        for k, v in o.items():
            if k.startswith("_"):
                continue
            if is_subchain(v):
                yield i, f"{o['op']}/{i}/{k}", k, v
            elif isinstance(v, (list, tuple)):
                for j, e in enumerate(v):
                    if is_subchain(e):
                        yield i, f"{o['op']}/{i}/{k}/{j}", f"{k}[{j}]", e
                    elif isinstance(e, dict):        # an entry holding one beside parameters
                        for kk, ee in e.items():
                            if is_subchain(ee):
                                own = ", ".join(f"{a}={_short(b)}" for a, b in e.items()
                                                if a != kk and not is_subchain(b))
                                yield (i, f"{o['op']}/{i}/{k}/{j}/{kk}",
                                       f"{k}[{j}].{kk}" + (f"  {own}" if own else ""), ee)


def child(spec, grid):
    """The `Signal` a sub-chain spec describes, for inspection: nothing is validated or run."""
    from .compose import Signal
    if isinstance(spec.get("ops"), (list, tuple)):
        s = Signal(grid=DOMAIN.grid_for(spec["ops"], grid))
        s.ops = [dict(o) for o in spec["ops"]]
        return s
    r = spec["from"]
    s = Signal(seed=r.get("seed", 0), grid=Grid(**r["grid"]) if r.get("grid") else None,
               lead_in=r.get("lead_in"), lead_out=r.get("lead_out"))
    s.ops = [dict(o) for o in r["ops"]]
    return s


def children(sig):
    """``{namespace: Signal}`` for every sub-chain in `sig`, nested ones included, in order."""
    out = {}
    for _i, at, _label, spec in walk(sig.ops):
        sub = child(spec, sig.grid)
        out[at] = sub
        out.update({f"{at}/{k}": v for k, v in children(sub).items()})
    return out


def roles(sig):
    """The re-rollable factors inside `sig`'s sub-chains, by their full role names."""
    return [f"{at}/{r}" for _i, at, _label, spec in walk(sig.ops)
            for r in child(spec, sig.grid).roles()]


def render_at(sig, path, streams=None):
    """The sub-chain at namespace `path`, rendered exactly as it is inside `sig`'s record."""
    st = streams if streams is not None else Streams(sig.seed)
    for _i, at, _label, spec in walk(sig.ops):
        if path == at or path.startswith(at + "/"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")      # the record's own render already warned
                sub, sst = _chain(spec, st, sig.grid, at, at)
            if path == at:
                return sub._run(sst)
            return render_at(sub, path[len(at) + 1:], sst)
    raise ValueError(f"no sub-chain at {path!r}. This signal holds: "
                     f"{list(children(sig)) or 'none'}")


def refuse_lead_in(ops):
    """A record with a lead-in cannot hold a sub-chain: say so rather than re-time it."""
    for _i, at, _label, _spec in walk(ops):
        raise ValueError(
            f"lead_in: the sub-chain at {at!r} is rendered on the record's grid, and a lead-in "
            f"lengthens that grid without extending the sub-chain's own pattern, which would "
            f"re-time it. Render the record without a lead-in, or give the guard to the "
            f"sub-chain's own record and embed it with {{'from': recipe}}.")


# ------------------------------------------------------------------ reading a recipe
def _short(v):
    """One parameter value on one line: its repr when that is short, its size when it is not."""
    if hasattr(v, "item") and getattr(v, "ndim", 1) == 0:
        v = v.item()
    r = repr(v)
    if isinstance(v, float) and len(r) > 8:
        return f"{v:.6g}"        # 500000000.0, 3.0000000000000004e-09: six figures read better
    if len(r) <= 48:
        return r
    if isinstance(v, dict):
        return "{" + ", ".join(str(k) for k in v) + "}"
    if hasattr(v, "__len__") and not isinstance(v, str):
        return f"[{len(v)} values]"
    return r[:45] + "..."


def _held(v):
    return is_subchain(v) or (isinstance(v, (list, tuple)) and any(
        is_subchain(e) or (isinstance(e, dict) and any(is_subchain(c) for c in e.values()))
        for e in v))


def _tree(sig, pre, indent):
    subs = {}
    for i, at, label, spec in walk(sig.ops):
        subs.setdefault(i, []).append((at, label, spec))
    wi = len(str(max(len(sig.ops) - 1, 0)))
    wo = max((len(str(o.get("op"))) for o in sig.ops), default=0)
    lines = []
    for i, o in enumerate(sig.ops):
        part = {k: TOL._role(i, k, e, pre) for k, e in o.get("_unit", {}).items()}
        shown = ", ".join(f"{k}={_short(v)}" + (f" ({part[k]})" if k in part else "")
                          for k, v in o.items()
                          if k != "op" and not k.startswith("_") and not _held(v))
        lines.append(f"{indent}{i:>{wi}}  {str(o.get('op')):<{wo}}  {shown}".rstrip())
        for at, label, spec in subs.get(i, ()):
            ops = spec.get("ops") or ()
            if len(ops) == 1 and ops[0].get("op") == "use":
                # a named chain of the record, as it stands: its ops are under its own name
                lines.append(f"{indent}{'':>{wi}}    {label}  = [{ops[0].get('name')}]")
                continue
            sub = child(spec, sig.grid)
            own = (f"  its own record: seed={sub.seed}" + (
                f", {_grid_words(sub.grid)}" if sub.grid is not None else "")
                   if "from" in spec else "")
            lines.append(f"{indent}{'':>{wi}}    {label}  [{pre}{at}]{own}")
            lines += _tree(sub, f"{pre}{at}/", indent + " " * (wi + 6))
    return lines


def tree(sig):
    """`sig`'s chain as text: one op per line, each sub-chain indented under the parameter that
    holds it and headed by its namespace -- the prefix of its role names in `roles()` and the
    argument `subchain()` takes."""
    head = f"Signal  seed={sig.seed}  " + _grid_words(sig.grid)
    if sig.lead_in:
        head += f"  lead_in={sig.lead_in!r}"
    return "\n".join([head] + _tree(sig, "", "  "))

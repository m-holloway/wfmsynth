"""
wfmsynth.channels -- several correlated channels from one recipe.

A clock and its data, the two legs of a pair, four lanes on one supply: these are ONE record,
and rendering them as separate recipes that happen to share a seed leaves their correlation to
convention. Here one recipe holds several NAMED CHAINS, and a chain refers to another BY NAME
with the one op `use`:

    pll = Signal().timing(phase_noise=dict(rms_ps=2.0))            # declared once
    rec = ws.Channels({"clk":  Signal().carrier("square", f_hz=5e9).use("pll"),
                       "data": Signal().carrier("nrz", n_ui=512).use("pll")},
                      shared={"pll": pll}, seed=1, grid=g)
    x = rec.waveforms()                       # {"clk": array, "data": array}

    ws.Channels({"p": Signal().carrier("nrz", n_ui=512),             # the two legs of a pair
                 "n": Signal().use("p").law("poly", coeffs=[0.0, -1.0])}, seed=1, grid=g)

`channels` are what `waveforms()` returns; `shared` are chains that exist only to be used. Both
are plain chains, and `use(name)` means one of two things depending on what the named chain is:

    a SIGNAL  (it starts with a source)   `use` is that chain's waveform. It goes first in a
                                          chain, or inside any parameter that takes a signal:
                                          `.crosstalk(aggressor=Signal().use("tx1"))`.
    a STAGE   (it starts with anything    `use` applies that chain's ops to the waveform
              else)                       arriving: a timing source, a supply, a channel model
                                          that several lanes go through.

WHY THE CORRELATION HOLDS. Every chain draws its randomness under ITS OWN NAME -- `pll/timing/0`,
`data/jitter/0` -- wherever it is used from, and each use starts those streams from the
beginning. So two channels that use `pll` are displaced by the same samples of the same
realisation, and `contrast("pll/timing/0")` moves every channel that uses it and nothing else.
Names survive serialisation, where the identity of a Python object does not: the recipe
(`recipe()`, one `sha256()` for all channels) is

    {"seed": 1, "grid": {...}, "channels": {"clk": {"ops": [...]}, "data": {"ops": [...]}},
     "shared": {"pll": {"ops": [...]}}}

and `Channels.from_recipe` replays it bit-identically with no other file.

A TOLERANCED PART (`ws.tol`, `ws.corners`) is drawn under the record's seed and its chain's
name, so `data/unit/3/r_source` and `clk/unit/3/r_source` are two parts; give both the same
`group=` and they are one (`unit/<group>`): the same resistor network, the same process
corner. `rec.unit(seed)` is the same design as another unit, under the same noise.

Two lanes that are EACH OTHER'S crosstalk cannot name each other -- that is a cycle, and it is
refused naming it. Physically each couples what the other LAUNCHED, so declare the launched
signals once and couple those:

    shared = {"tx0": Signal().carrier("nrz", n_ui=512, seed=1),
              "tx1": Signal().carrier("nrz", n_ui=512, seed=2)}
    lanes = {"lane0": Signal().use("tx0").crosstalk(aggressor=Signal().use("tx1"), coupling=0.1),
             "lane1": Signal().use("tx1").crosstalk(aggressor=Signal().use("tx0"), coupling=0.1)}
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json

import numpy as np

from . import domain as DOMAIN
from . import opkeys as OPKEYS
from . import subchain as SUB
from . import tolerance as TOL
from .grid import Grid
from .streams import Made, Streams


def _name_ok(name, what):
    if not isinstance(name, str) or not name or "/" in name or "*" in name:
        raise ValueError(f"Channels: {what} name {name!r} must be a non-empty string without "
                         f"'/' or '*': it is the prefix of that chain's role names "
                         f"(`{name}/jitter/0`).")


def _ops_of(v, name, rec):
    """One named chain as given -> its op list, refusing what a named chain cannot carry."""
    from .compose import Signal, _EXEC
    who = f"Channels: chain {name!r}"
    if isinstance(v, dict) and set(v) == {"ops"}:
        v = v["ops"]
    if isinstance(v, Signal):
        if v.lead_in or v.lead_out is not None:
            raise ValueError(f"{who} is a Signal with a lead-in, which lengthens one chain's "
                             f"render and not the others'. Drop the lead-in.")
        if v.seed not in (0, rec.seed):
            raise ValueError(f"{who} is a Signal with seed={v.seed}, but every chain of a "
                             f"record draws from the RECORD's seed ({rec.seed}) under its own "
                             f"name, so that seed would be ignored. Leave the seed off the "
                             f"chain and set Channels(seed=...).")
        if v.grid is not None and v.grid != rec.grid:
            raise ValueError(f"{who} is a Signal on its own grid, but every chain of a record "
                             f"is rendered on the record's. Build it with no grid, "
                             f"Signal().carrier(...), and set Channels(grid=...).")
        v = TOL.stated(v.ops)                # its tolerances are drawn by the record it joins
    if not isinstance(v, (list, tuple)):
        raise ValueError(f"{who} is {type(v).__name__}; a chain is a Signal or "
                         f"{{'ops': [...]}}.")
    ops = [dict(o) for o in v]
    if not ops:
        raise ValueError(f"{who} has no ops, so it renders nothing.")
    for i, o in enumerate(ops):
        if o.get("op") not in _EXEC:
            raise ValueError(f"{who} has op {o.get('op')!r} at {i}, which does not exist. "
                             f"Known ops: {sorted(_EXEC)}")
        # a toleranced value -> its number, drawn under the record's seed and this chain's
        # name (a recipe's values are already numbers and pass through untouched)
        ops[i] = TOL.resolve(i, o, rec.seed, pre=name + "/")
        OPKEYS.check(o["op"], ops[i])        # the same gate the builders go through
    return ops


def _uses(ops):
    """The names an op list refers to, sub-chains written inside it included."""
    for o in ops:
        if o.get("op") == "use":
            yield o.get("name")
    for _i, _at, _label, spec in SUB.walk(ops):
        if isinstance(spec.get("ops"), (list, tuple)):
            yield from _uses(spec["ops"])


class Channels:
    """Several named chains rendered as one record. See the module docstring.

        Channels(channels, shared=None, seed=0, grid=None)

    `channels` and `shared` are `{name: Signal}` (or `{name: {"ops": [...]}}`, the form the
    recipe stores). The Signals are written with no seed and no grid: the record has one of
    each. Channels are returned in the order given.
    """

    def __init__(self, channels, shared=None, seed=0, grid=None):
        self.seed, self.grid = int(seed), grid
        if not isinstance(channels, dict) or not channels:
            raise ValueError("Channels: `channels` is a non-empty {name: Signal}. One channel "
                             "alone is a Signal.")
        shared = dict(shared or {})
        both = sorted(set(channels) & set(shared))
        if both:
            raise ValueError(f"Channels: {both} is in both `channels` and `shared`. A name is "
                             f"one chain: list it under `channels` to get its waveform back, "
                             f"under `shared` if it exists only to be used.")
        for name in channels:
            _name_ok(name, "channel")
        for name in shared:
            _name_ok(name, "shared chain")
        self.names = list(channels)
        self.shared = list(shared)
        self.chains = {name: _ops_of(v, name, self) for name, v in {**channels, **shared}.items()}
        self._check_references()
        for name in self.names:
            if not self.is_signal(name):
                raise ValueError(
                    f"Channels: channel {name!r} starts with "
                    f"`{self.chains[name][0]['op']}`, which acts on a waveform and has none. "
                    f"A channel starts with a source (carrier, symbols, coded, capture, iq) "
                    f"or with use(<a chain that does>); a chain meant to be applied to "
                    f"others belongs under `shared`.")
        for name in self.names:                 # refuse the wrong domain where it is written
            flat = self._flat(name)
            DOMAIN.trace(flat, DOMAIN.grid_for(flat, self.grid))

    # ------------------------------------------------------------ what refers to what
    def _check_references(self):
        refs = {}
        for name, ops in self.chains.items():
            refs[name] = list(dict.fromkeys(_uses(ops)))
            for r in refs[name]:
                if r not in self.chains:
                    raise ValueError(f"Channels: chain {name!r} uses {r!r}, which is not a "
                                     f"chain of this record. It has: {list(self.chains)}")
        state = {}

        def visit(name, path):
            if state.get(name) == "done":
                return
            if name in path:
                loop = path[path.index(name):] + [name]
                raise ValueError(
                    f"Channels: {' -> '.join(loop)} is a cycle -- each of these needs "
                    f"another finished before it can start. Declare what they have in common "
                    f"once under `shared` and let each use that: two lanes that are each "
                    f"other's crosstalk each couple the signal the other LAUNCHED "
                    f"(shared={{'tx0': ..., 'tx1': ...}}), not its received channel.")
            for r in refs[name]:
                visit(r, path + [name])
            state[name] = "done"

        for name in self.chains:
            visit(name, [])

    def is_signal(self, name):
        """True when the chain produces a waveform by itself (it starts with a source, with a
        `net`, or with `use` of a chain that does); False when it is a stage to apply."""
        from .compose import op_kind
        first = self.chains[name][0]
        if first["op"] == "use":
            return self.is_signal(first["name"])
        return op_kind(first["op"]) == "source" or first["op"] == "net"

    def _flat(self, name):
        """The chain's ops with every `use` replaced by the ops it stands for, in order."""
        out = []
        for o in self.chains[name]:
            out += self._flat(o["name"]) if o["op"] == "use" else [o]
        return out

    def expand(self, name):
        """``(is_signal, ops)`` of a named chain, its own `use`s replaced: what
        `domain.trace` needs to follow a `use`."""
        return self.is_signal(name), self._flat(name)

    def _signal(self, name, like=None, dom=None):
        """The chain as a Signal on the record's grid, in the domain ITS ops need: a record may
        hold an I/Q channel and the real supply that scales it. A stage is put in the domain
        of the waveform `like` that it is being applied to: `dom` where the chain using it
        knows (a complex waveform is an envelope or an optical field), else what `like` and
        the stage's own ops say."""
        from .compose import Signal
        grid = DOMAIN.grid_for(self._flat(name), self.grid)
        if like is not None and grid is not None:
            dom = dom or DOMAIN.held(like, grid)[0]
            if dom != DOMAIN.of(grid):
                grid = dataclasses.replace(grid, domain=dom, fc_hz=None)
        s = Signal(seed=self.seed, grid=grid)
        s.ops = self.chains[name]
        return s

    # ------------------------------------------------------------ rendering
    def waveforms(self, streams=None):
        """Render every channel -> ``{name: samples}``, in the order the channels were given.
        Deterministic given the recipe. `streams` re-rolls selected roles as it does for a
        Signal; `contrast()` wraps the common case."""
        r = _Render(self, streams)
        return {name: r.signal(name)[0] for name in self.names}

    def waveform(self, name, streams=None):
        """One chain's waveform, exactly as `waveforms()` has it. `name` may also be a SHARED
        chain that is a signal -- the bit stream both legs carry, the aggressor that was
        coupled -- so what the channels have in common is ground truth too."""
        if name not in self.chains:
            raise ValueError(f"Channels: no chain {name!r}. This record has: "
                             f"{list(self.chains)}")
        if not self.is_signal(name):
            raise ValueError(f"Channels: {name!r} is a stage -- it starts with "
                             f"`{self.chains[name][0]['op']}` and acts on the waveform of "
                             f"whichever chain uses it -- so it has no waveform of its own.")
        return _Render(self, streams).signal(name)[0]

    def signal(self, name):
        """Channel (or shared signal chain) `name` as a `Signal` that renders EXACTLY as it
        does in this record -- the record's seed, the chain's own role names, its `use`s
        resolved here -- so everything a Signal can say about itself is available for one
        channel of several:

            two.signal("a").transmitted()         # the symbols channel a sent
            two.signal("a").rf_report()           # what each impairment came to, shared ones too
            two.signal("a").lo_phase()            # the shared oscillator's phase: a label
            rec.signal("data").subchain("crosstalk/2/aggressor")

        also `realize()`, `summary()`, `stored_grid(x)`, `filter_delays()`, `delay_errors()`.
        It is a view, not a record of its own: `recipe()`, `contrast()` and `unit()` are the
        record's, and are refused here saying so."""
        if name not in self.chains:
            raise ValueError(f"Channels: no chain {name!r}. This record has: "
                             f"{list(self.chains)}")
        if not self.is_signal(name):
            raise ValueError(f"Channels: {name!r} is a stage -- it starts with "
                             f"`{self.chains[name][0]['op']}` and acts on the waveform of "
                             f"whichever chain uses it -- so it is not a signal by itself.")
        s = self._signal(name)
        s.ops, s.within = list(s.ops), (self, name)
        return s

    def realize(self, streams=None):
        """One pass -> ``(waveforms, events)``, both ``{name: ...}``: each channel's samples
        and its `EventList`. An event placed inside a chain this channel uses carries that
        chain's name in `Event.at`."""
        r = _Render(self, streams, collect=True)
        done = {name: r.signal(name) for name in self.names}
        return {k: v[0] for k, v in done.items()}, {k: v[1] for k, v in done.items()}

    def stored_grid(self, name, x):
        """The `Grid` describing channel `name`'s returned samples `x` (see
        `Signal.stored_grid`): each channel may end in its own acquisition."""
        from .compose import Signal
        flat = self._flat(name)
        s = Signal(seed=self.seed, grid=DOMAIN.grid_for(flat, self.grid))
        s.ops = flat
        return s.stored_grid(x)

    # ------------------------------------------------------------ units
    def unit(self, seed):
        """The same design as ANOTHER UNIT: every toleranced parameter (`ws.tol`,
        `ws.corners`) of every chain drawn again under `seed`, and nothing else changed -- the
        record's seed, so its noise and jitter, stays. As `Signal.unit`.

        An ungrouped part belongs to its chain (`data/unit/3/r_source`); parts given the same
        `group=` are ONE draw across the whole record (`unit/<group>`), which is how two
        channels say they share a part or a process corner."""
        if not self._unit_roles():
            raise ValueError("Channels.unit(): this record has no toleranced parameter, so "
                             "there is no unit to draw. Write one with ws.tol(...) or "
                             "ws.corners(...).")
        new = copy.copy(self)
        new.chains = {name: [TOL.resolve(i, o, int(seed), pre=name + "/")
                             for i, o in enumerate(TOL.stated(ops))]
                      for name, ops in self.chains.items()}
        return new

    def _unit_roles(self):
        out = []
        for name, ops in self.chains.items():
            out += [r for r in TOL.roles(ops, name + "/") if r not in out]
        return out

    # ------------------------------------------------------------ roles
    def roles(self):
        """Every re-rollable factor in the record, as `<chain name>/<role>` -- the arguments
        `contrast()` takes. A shared chain's roles are listed once, under its own name,
        however many channels use it. Toleranced parts follow: `<chain name>/unit/...`, or
        `unit/<group>` for a part several chains share (see `unit`)."""
        return [f"{name}/{r}" for name in self.chains for r in self._signal(name).roles()
                if not TOL.is_role(r)] + self._unit_roles()

    def contrast(self, *factors, seed=None):
        """Sibling waveforms with ONLY the named roles re-rolled -> ``{name: samples}``.
        Re-rolling a shared chain's role moves every channel that uses it; every channel that
        does not is bit-identical. A toleranced part's role (or `"unit/*"` for all of them)
        draws that part again, as it does on a Signal; `unit(seed)` returns the other unit as
        a record, with its values in the recipe. Every role `roles()` lists is accepted, and a
        name that matches no randomness in the record is refused."""
        st = Streams(self.seed).reroll(*factors, seed=seed)
        units = self._unit_roles()
        seed_of = {k: v for k, v in st._overrides.items() if k == TOL.ANY or k in units}
        rec = self
        if seed_of:
            if not units:
                self.unit(0)                              # the refusal: nothing is toleranced
            rec = copy.copy(self)
            rec.chains = {name: TOL.redraw_ops(ops, seed_of, name + "/")
                          for name, ops in self.chains.items()}
        out = rec.waveforms(streams=st)
        idle = [k for k in st.idle(self.roles()) if k not in seed_of]
        if idle:
            raise ValueError(f"Channels.contrast: {idle} names no randomness this record "
                             f"draws, so the sibling would be this record again. A role is "
                             f"`<chain name>/<role>`; roles() lists them: {self.roles()}")
        from .compose import _warn_undrawn
        _warn_undrawn("Channels.contrast", st, self.roles())
        return out

    # ------------------------------------------------------------ the recipe
    def recipe(self):
        """A JSON-serialisable dict of the whole record: seed, grid, and every chain by name.
        This is the ground truth for all the channels at once."""
        from wfmsynth import __version__
        from .compose import Signal
        r = {"wfmsynth_version": __version__, "seed": self.seed,
             "channels": {n: {"ops": [dict(o) for o in self.chains[n]]} for n in self.names}}
        if self.shared:
            r["shared"] = {n: {"ops": [dict(o) for o in self.chains[n]]} for n in self.shared}
        if self.grid is not None:
            r["grid"] = Signal(grid=self.grid).recipe()["grid"]
        return r

    def to_json(self):
        """Canonical, key-sorted JSON of the recipe."""
        return json.dumps(self.recipe(), sort_keys=True)

    def sha256(self):
        """ONE content hash for all the channels: equal hashes mean equal samples in every
        channel."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_recipe(cls, r):
        """Rebuild the record from its recipe; `.waveforms()` reproduces every channel
        bit-for-bit."""
        from .compose import _warn_on_numeric_drift
        if "channels" not in r:
            raise ValueError("Channels.from_recipe: this recipe has no `channels`, so it is "
                             "one Signal. Signal.from_recipe(recipe).waveform() renders it.")
        rec = cls(r["channels"], shared=r.get("shared"), seed=r["seed"],
                  grid=Grid(**r["grid"]) if r.get("grid") else None)
        _warn_on_numeric_drift(r.get("wfmsynth_version"),
                               [o for ops in rec.chains.values() for o in ops])
        return rec

    def summary(self):
        """The record as text: each chain under its name, one op per line (sub-chains
        indented as in `Signal.summary`). The name is the prefix of the chain's roles."""
        lines = [f"Channels  seed={self.seed}  " + SUB._grid_words(self.grid)]
        for kind, names in (("channel", self.names), ("shared", self.shared)):
            for name in names:
                sig = self._signal(name)
                lines.append(f"  {kind} [{name}]" + SUB._starts(sig))
                lines += SUB._tree(sig, f"{name}/", "    ", expand=self.expand)
        return "\n".join(lines)

    # ------------------------------------------------------------ export
    def write_hdf5(self, path, **kw):
        """Write every channel to ONE HDF5 file (`wfmsynth.hdf5.write_hdf5`; `kw` passes
        through): `Channel 1..N` in channel order, each carrying its name, and the one recipe
        with its digest. `hdf5.read_recipe(path)` gives the recipe back and
        `hdf5.channel_names(path)` the ``{number: name}`` map.

        The file has one time axis, so channels stored at different rates are refused."""
        from . import hdf5 as H5
        x = self.waveforms()
        rates = {name: self.stored_grid(name, x[name]).fs for name in self.names}
        if len(set(rates.values())) != 1:
            raise ValueError(f"Channels.write_hdf5: the channels are stored at different "
                             f"rates {rates}, and the file has one time axis. Acquire them "
                             f"at one rate, or write the groups of equal rate to separate "
                             f"files.")
        return H5.write_hdf5(path, {k + 1: x[n] for k, n in enumerate(self.names)},
                             fs=rates[self.names[0]],
                             names={k + 1: n for k, n in enumerate(self.names)},
                             recipe=self.recipe(), **kw)


class _Render:
    """One render of a record: the seed and re-rolls every chain draws under, and the signal
    chains already rendered, so a chain used from several places is ONE waveform."""

    def __init__(self, rec, streams, collect=False):
        self.rec, self.collect = rec, collect
        self.seed = rec.seed if streams is None else streams.seed
        self.overrides = {} if streams is None else dict(streams._overrides)
        self.asked = set() if streams is None else streams.asked    # one for the whole record
        self.made = Made() if streams is None else streams.made
        self.done = {}

    def view(self, name, sink=None):
        # fresh streams per use: a role's draws are then a function of (seed, chain name,
        # role) alone, not of how many channels used the chain before this one
        st = Streams(self.seed, self.overrides)
        st.asked = self.asked
        st.made = self.made
        v = SUB.Namespaced(st, name + "/", sink)
        v.record = self
        return v

    def signal(self, name):
        """``(waveform, EventList or None)`` of a signal chain, rendered once."""
        if name not in self.done:
            sig = self.rec._signal(name)
            self.done[name] = (sig._run(self.view(name), collect_events=True) if self.collect
                               else (sig._run(self.view(name)), None))
        return self.done[name]


def use(x, name, streams, dom=None, carrier=None):
    """The `use` op: the chain `name` of the record being rendered, at this point in a chain.
    `dom` is the domain the waveform `x` is in, and `carrier` what a field is about."""
    r = getattr(streams, "record", None)
    if r is None:
        raise ValueError(
            f"use({name!r}) names a chain of the several-channel record this chain belongs "
            f"to, and it is being rendered alone (or inside a {{'from': recipe}} sub-chain, "
            f"which is its own record). Render it as part of "
            f"ws.Channels({{...}}, shared={{{name!r}: ...}}).")
    rec = r.rec
    if name not in rec.chains:
        raise ValueError(f"use({name!r}): no such chain in this record. It has: "
                         f"{list(rec.chains)}")
    sink = getattr(streams, "sink", None)
    first = rec.chains[name][0]["op"]
    if rec.is_signal(name):
        if x is not None:
            raise ValueError(
                f"use({name!r}): {name!r} is a whole signal (it starts with `{first}`), and "
                f"a waveform has already arrived here, which it would discard. Start the "
                f"chain with use({name!r}), or hand it to the parameter that takes a signal "
                f"-- crosstalk(aggressor=Signal().use({name!r})).")
        y, placed = r.signal(name)
        y = y.copy()                         # the one render is shared by every chain using it
        placed = None if placed is None else placed.events
    else:
        if x is None:
            raise ValueError(
                f"use({name!r}): {name!r} is a stage (it starts with `{first}`), which acts "
                f"on a waveform, and none has arrived: it is the first op here. Put a source "
                f"before it.")
        sig = rec._signal(name, like=x, dom=dom)
        if sink is None:
            return sig._run(r.view(name), _x=x, _carrier=carrier)
        # the stage sees the events already on the waveform it is handed, so a `trigger` in
        # it re-times them onto its windows (and drops the rest) along with the samples
        before = {id(e) for e in sink}
        y, placed = sig._run(r.view(name), collect_events=True, _x=x, _events=sink,
                             _carrier=carrier)
        sink[:] = [e for e in placed.events if id(e) in before]
        placed = [e for e in placed.events if id(e) not in before]
    if sink is not None and placed is not None:
        sink.extend(dataclasses.replace(e, at=name + ("/" + e.at if e.at else ""))
                    for e in placed)
    return y

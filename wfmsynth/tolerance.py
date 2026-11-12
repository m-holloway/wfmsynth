"""
wfmsynth.tolerance -- unit-to-unit variation: parameters drawn from stated tolerances and corners.

A seed re-rolls noise and patterns. It does not make a second BOARD: the resistor is still
exactly 50 ohms and the edge still exactly 300 ps, so a thousand records are a thousand captures
of one unit, and a model trained on them has never seen the population it will be used on. Real
parts vary within a tolerance and a process within its corners, and that variation is a
different thing from noise -- it is fixed for a unit and differs between units.

Write the tolerance where the number would go:

    import wfmsynth as ws
    s = (ws.Signal(seed=3, grid=g)
         .carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=1e6, tr_s=2e-9)
         .probe(r_source=ws.tol(1e3, 5), c_load_f=ws.tol(100e-12, 10, dist="normal")))

It is drawn ONCE, when the op is added, and the op then holds the NUMBER -- so the recipe
states what this unit's resistor actually is, every other part of the library sees an ordinary
parameter, and a recipe replays without drawing anything. What it was drawn from is kept beside
it under the op's `_unit` annotation, which is what lets the same design be drawn again:

    s.recipe()["ops"][1]["r_source"]         # a float: this unit's resistor, the ground truth
    s.unit(8)                                # the same design as another unit (a Signal)
    s.contrast("unit/*")                     # another unit, the SAME noise
    s.contrast("noise/2")                    # the same unit, other noise -- as it always was

THE FORMS. A toleranced value is plain data, so it can be written in a JSON recipe as well as
passed to a builder; `tol()` and `corners()` only build these dicts and check them.

    {"nom": 50.0, "tol_pct": 5}                      50 ohms +/- 5 %
    {"nom": 0.0, "tol_abs": 5e-12}                   +/- 5 ps about zero, where a percentage of
                                                     the nominal would vary nothing
    {"nom": ..., "tol_pct": ..., "dist": D}          D is one of:
        "uniform"   anywhere in nom +/- tol, equally likely                        (the default)
        "normal"    a Gaussian with tol = 3 SIGMA, TRUNCATED at +/- tol. Nothing is ever drawn
                    outside the tolerance -- a part outside it was rejected at incoming
                    inspection -- and the realised standard deviation is 0.9866 * tol / 3.
        "corner"    exactly nom - tol or nom + tol, equally likely: a worst-case pair
    {"corners": {"slow": 12.0, "typ": 10.0, "fast": 8.0}}       one of the named values, equally
                                                     likely; add "pick": "slow" to state it

A number drawn from a tolerance is a float. A corner keeps the value it was given, so a
parameter that has to be an integer takes corners, and a tolerance on one (`WHOLE_NUMBERS`:
`n_ui`, `bits`, `order`, ...) is refused.

ONE DRAW PER ROLE. Each toleranced parameter has a role, `unit/<op index>/<parameter>`, and its
value is its distribution's quantile at one uniform number that is a pure function of
(seed, role) -- independent of every other role, of the order ops were added, and of every
noise role. Parameters that vary TOGETHER say so with `"group": "process"`: they share the role
`unit/process` and so the same number, which is what a process corner is (the slow lot has the
slow edge AND the high output resistance) and what two resistors in one network are.

WHAT IS REFUSED, because each would otherwise be a silent no-op or a wrong label:
a nominal or a corner that is not a real number; a negative tolerance, or a percentage of 100
or more (it reaches zero or changes sign -- name the values with corners); a percentage of a
zero nominal; a tolerance inside a tolerance; a hand-written nested op list that arrives with
tolerances already resolved; `unit()` or a `unit/...` re-roll on a record that has none.

INSIDE A SUB-CHAIN. A chain nested in an op (`crosstalk(aggressor=...)`, a `net` driver's
source; see `wfmsynth.subchain`) is part of the unit, so its tolerances belong to the OUTER
record: one written in a nested op's dict, or carried in by a `Signal` used as the sub-chain, is
drawn under the outer seed with the role `<the sub-chain's namespace>/unit/<k>/<parameter>`
(`crosstalk/1/aggressor/unit/1/loss_db`, beside that sub-chain's `.../rx_noise/2`), and
`unit()` moves it with everything else. `{"from": recipe}` is another record pinned as rendered,
and so is its unit.

PER OCCURRENCE. A tolerance drawn once per unit is fixed for every capture of that unit. A
value that changes from one OCCURRENCE of an event to the next -- a surge's amplitude, a
relay's bounce -- is written the same way with `per="segment"`, and one that moves steadily
across the occurrences -- a supply sagging, a part warming -- is a `trend`:

    ws.tol(1.0, 5, per="segment")                     drawn again for every occurrence
    ws.trend("ramp", start=1.0, stop=0.9)             first occurrence 1.0, last 0.9
    ws.trend("ramp", start=1.0, rate_per_s=-1e-3)     at the occurrence's arrival time
    ws.trend("step", before=1.0, after=0.8, at=150)   from occurrence 150 on (or at_s=)

Both exist only across the occurrences of `ws.Segments` (`wfmsynth.segments`): the op holds
the nominal (a trend's starting value) as a placeholder, the spec is kept beside it under
`_segment`, and a chain that holds one is REFUSED when it is rendered alone, since no single
number is the truth for it. A segment draw nests inside the unit: the unit's values stay put
for the whole acquisition and the segment's are drawn under it, role
`vary/<op index>/<parameter>` (or `vary/<group>`), one uniform number per occurrence. A part
that varies from unit to unit AND from occurrence to occurrence is two factors on two ops --
`gain(by=ws.tol(1, 10))` then `gain(by=ws.tol(1, 2, per="segment"))` -- each with its own
stated distribution, not a tolerance inside a tolerance.

WHAT THIS OVERLAPS. `dataset(build, n)` already randomises anything: `build(rng)` draws a knob
and the drawn number is baked into the recipe. That stays the general tool, and `unit_dataset`
is built on it. What it cannot say is what the number was drawn FROM, or which other records are
the same unit -- the two things recorded here. `impairments.domain_randomize` and
`grammar.sample` randomise capture conditions on a bare array with one shared generator and
record nothing; they are the legacy route and are not a population of units. `sweep` is
deterministic and unrelated.
"""
from __future__ import annotations

import dataclasses
import difflib
import itertools
import math
import numbers

import numpy as np
from scipy.special import ndtr, ndtri

from .streams import _role_seed

DISTS = ("uniform", "normal", "corner")
NORMAL_SIGMAS = 3.0            # "normal": the tolerance is this many sigma, and the draw is truncated there
_TOL_KEYS = frozenset({"nom", "tol_pct", "tol_abs", "dist", "group", "per"})
_CORNER_KEYS = frozenset({"corners", "pick", "group", "per"})
_MARK = ("nom", "tol_pct", "tol_abs", "corners", "trend")   # any of these: a value still to draw
PER = ("unit", "segment")                             # the draw levels a tolerance names
TRENDS = {"ramp": (("start", "stop"), ("start", "rate_per_s")),
          "step": (("before", "after", "at"), ("before", "after", "at_s"))}
ANY = "unit/*"                                        # every unit role at once
# Parameters that are COUNTS. A tolerance draws a float, which each of these would truncate or
# render as a different thing than was drawn (8.86 bits is 8); they take named corners.
WHOLE_NUMBERS = frozenset({
    "bits", "order", "n_ui", "n", "n_out", "n_bits", "n_segments", "average", "n_bounce", "prbs",
    "pre", "count", "every", "m_cores", "record_length", "depth"})


# ---- the two forms ----------------------------------------------------------------------------
def tol(nom, tol_pct=None, *, tol_abs=None, dist="uniform", group=None, per="unit"):
    """A toleranced number: `tol(50, 5)` is 50 +/- 5 %, `tol(0.0, tol_abs=5e-12)` is +/- 5 ps.

    `dist` is "uniform" (default), "normal" (tol = 3 sigma, truncated at the tolerance) or
    "corner" (one of the two extremes). `group` names parameters that are drawn together.
    `per="segment"` draws it again for every occurrence of a `ws.Segments` acquisition
    instead of once per unit. Returns the plain dict a recipe carries, checked."""
    spec = {"nom": nom}
    if tol_pct is not None:
        spec["tol_pct"] = tol_pct
    if tol_abs is not None:
        spec["tol_abs"] = tol_abs
    if dist != "uniform":
        spec["dist"] = dist
    if group is not None:
        spec["group"] = group
    if per != "unit":
        spec["per"] = per
    _check(spec, "tol()")
    return spec


def corners(values, pick=None, group=None, per="unit"):
    """A named corner set: `corners({"slow": 12.0, "typ": 10.0, "fast": 8.0})` is one of the
    three, equally likely. `pick="slow"` states the corner instead of drawing it; `group` names
    parameters that share one draw: sets with the same names take the SAME corner.
    `per="segment"` draws the corner again for every occurrence of a `ws.Segments`."""
    if not isinstance(values, dict):
        raise ValueError(f"corners(): takes {{name: value}}, e.g. corners({{'slow': 12.0, "
                         f"'fast': 8.0}}), got {type(values).__name__}. The names are what the "
                         f"recipe records as the corner each unit drew.")
    spec = {"corners": dict(values)}
    if pick is not None:
        spec["pick"] = pick
    if group is not None:
        spec["group"] = group
    if per != "unit":
        spec["per"] = per
    _check(spec, "corners()")
    return spec


def trend(kind, **params):
    """A value that moves steadily across the occurrences of a `ws.Segments` acquisition -- a
    drift, deterministic, with no role: nothing about it is drawn.

        trend("ramp", start=1.0, stop=0.9)              linear in the occurrence's index
        trend("ramp", start=1.0, rate_per_s=-2e-3)      linear in its arrival time
        trend("step", before=1.0, after=0.8, at=150)    from occurrence 150 on
        trend("step", before=1.0, after=0.8, at_s=2.5)  from the first arriving at 2.5 s on

    Its location is `trend/<op index>/<parameter>`; `Segments.without("trend/...")` is the
    same acquisition with the trend held at its first value. Returns the plain dict."""
    spec = {"trend": kind, **params}
    _check(spec, "trend()")
    return spec


def _check_trend(spec, where):
    kind = spec["trend"]
    if kind not in TRENDS:
        raise ValueError(f"{where}: a trend is one of {sorted(TRENDS)}, got {kind!r}")
    keys = set(spec) - {"trend"}
    forms = TRENDS[kind]
    if not any(keys == set(f) for f in forms):
        raise ValueError(
            f"{where}: a {kind!r} trend takes "
            + " or ".join("{" + ", ".join(f) + "}" for f in forms)
            + f"; got {sorted(keys)}. A trend is deterministic, so it has no dist or group.")
    for k in keys - {"at"}:
        if not _real(spec[k]):
            raise ValueError(f"{where}: {k}={spec[k]!r} is not a real number.")
    if "at" in spec and (isinstance(spec["at"], bool) or not isinstance(spec["at"], numbers.Integral)
                         or spec["at"] < 1):
        raise ValueError(f"{where}: at= is the index of the first occurrence after the step, a "
                         f"whole number >= 1, got {spec['at']!r}. In seconds it is at_s=.")
    if "at_s" in spec and spec["at_s"] <= 0:
        raise ValueError(f"{where}: at_s= is a time after the acquisition starts, > 0, got "
                         f"{spec['at_s']!r}.")


def _real(v):
    return (isinstance(v, numbers.Real) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def _check(spec, where):
    """Refuse a toleranced value that cannot be drawn honestly. `where` names it in the message."""
    if "trend" in spec:
        return _check_trend(spec, where)
    if spec.get("per", "unit") not in PER:
        raise ValueError(f"{where}: per={spec['per']!r} is one of {list(PER)}: drawn once per "
                         f"unit (the default), or again for every occurrence of a ws.Segments.")
    keys = _CORNER_KEYS if "corners" in spec else _TOL_KEYS
    unknown = sorted(k for k in spec if k not in keys)
    if unknown:
        near = difflib.get_close_matches(unknown[0], sorted(keys), n=1, cutoff=0.6)
        raise ValueError(
            f"{where}: a toleranced value does not take {unknown}"
            + (f" -- did you mean {near[0]!r}?" if near else "")
            + f". It is {{'nom', 'tol_pct' or 'tol_abs', 'dist', 'group', 'per'}} or "
              f"{{'corners', 'pick', 'group', 'per'}}.")
    group = spec.get("group")
    if group is not None and (not isinstance(group, str) or not group or "/" in group
                              or group == "*" or group.isdigit()):
        raise ValueError(f"{where}: group must be a plain name such as 'process' (no '/', not "
                         f"'*', not a number -- those are how the other roles are written), "
                         f"got {group!r}")
    if "corners" in spec:
        cs = spec["corners"]
        if not isinstance(cs, dict) or not cs:
            raise ValueError(f"{where}: corners must be a non-empty {{name: value}} dict, "
                             f"got {cs!r}")
        bad = sorted(str(k) for k, v in cs.items() if not _real(v))
        if bad:
            raise ValueError(
                f"{where}: corner(s) {bad} are not real numbers. A corner set varies a NUMERIC "
                f"parameter; to vary a kind or a topology, build the records separately with "
                f"dataset(build, n).")
        if not all(isinstance(k, str) for k in cs):
            raise ValueError(f"{where}: corner names must be strings, got {sorted(map(repr, cs))}")
        if "pick" in spec and spec["pick"] not in cs:
            raise ValueError(f"{where}: pick={spec['pick']!r} is not one of the corners "
                             f"{sorted(cs)}")
        return
    if "nom" not in spec:
        raise ValueError(f"{where}: a tolerance needs 'nom', the value it is a tolerance ON")
    nom = spec["nom"]
    if isinstance(nom, dict):
        raise ValueError(f"{where}: the nominal is itself a toleranced value. A tolerance inside "
                         f"a tolerance has no single stated distribution; state the one you mean.")
    if not _real(nom):
        raise ValueError(
            f"{where}: nom={nom!r} is not a real number. A tolerance varies a NUMERIC "
            f"parameter; to vary a kind or a topology, build the records separately with "
            f"dataset(build, n).")
    has_pct, has_abs = "tol_pct" in spec, "tol_abs" in spec
    if has_pct == has_abs:
        raise ValueError(f"{where}: give exactly one of tol_pct (percent of nom) or tol_abs "
                         f"(same units as nom)")
    t = spec["tol_pct"] if has_pct else spec["tol_abs"]
    if not _real(t) or t < 0:
        raise ValueError(f"{where}: the tolerance must be a real number >= 0, got {t!r}")
    if has_pct and t >= 100:
        raise ValueError(
            f"{where}: tol_pct={t!r} reaches zero or changes the sign of the value. Name the "
            f"values you mean with corners, or state the spread with tol_abs.")
    if has_pct and float(nom) == 0.0 and t > 0:
        raise ValueError(
            f"{where}: {t!r} % of a nominal of 0 is 0, so nothing would vary. A parameter whose "
            f"nominal is zero (a skew, an offset) takes tol_abs, in the parameter's own units.")
    dist = spec.get("dist", "uniform")
    if dist not in DISTS:
        raise ValueError(f"{where}: dist={dist!r} is not one of {list(DISTS)}")


# ---- one draw ---------------------------------------------------------------------------------
def _u(seed, role):
    """The one uniform number in [0, 1) that belongs to (seed, role). The same derivation as
    `Streams.role`, taken fresh rather than from a `Streams` cache: a cached generator has
    advanced by the second parameter of a group, and a group is ONE number."""
    ss = np.random.SeedSequence([int(seed), _role_seed(role)])
    return float(np.random.default_rng(ss).random())


_P_LO, _P_HI = float(ndtr(-NORMAL_SIGMAS)), float(ndtr(NORMAL_SIGMAS))


def quantile(spec, u):
    """The value of a toleranced parameter at the uniform number `u` in [0, 1) -- its stated
    distribution's inverse CDF, which IS the statement of the distribution. Returns
    `(value, corner_name_or_None)`."""
    if "corners" in spec:
        cs = spec["corners"]
        names = sorted(cs)               # sorted, so JSON's key order cannot change the draw
        name = spec["pick"] if "pick" in spec else names[min(int(u * len(names)), len(names) - 1)]
        return cs[name], name
    nom = float(spec["nom"])
    half = (abs(nom) * float(spec["tol_pct"]) / 100.0 if "tol_pct" in spec
            else float(spec["tol_abs"]))
    dist = spec.get("dist", "uniform")
    if dist == "uniform":
        return nom + half * (2.0 * u - 1.0), None
    if dist == "corner":
        return (nom - half if u < 0.5 else nom + half), None
    z = float(ndtri(_P_LO + u * (_P_HI - _P_LO)))
    return nom + (half / NORMAL_SIGMAS) * z, None


def located(i, name, op=None, pre="", level="unit"):
    """Where a toleranced parameter sits, written the way every other role is. In the chain
    itself: `unit/<op index>/<parameter>`. Inside a sub-chain: that sub-chain's namespace,
    then the same -- `crosstalk/1/aggressor/unit/1/loss_db` beside
    `crosstalk/1/aggressor/rx_noise/2`. `pre` is the chain's name in a several-channel
    record (`clk/`). `level` is the first word: "unit", or "vary" / "trend" for a value
    that changes from one occurrence of a `ws.Segments` to the next."""
    path = _split(name)
    if op is None or "ops" not in path:
        return f"{pre}{level}/{i}/{name}"
    ns, k = [], 0
    while "ops" in path[k:]:
        j = path.index("ops", k)
        held = path[k:j]
        if op.get("op") == "net" and held[:1] == ("drivers",):     # as `subchain.walk` names it
            ns.append(f"net/{i}/{op['drivers'][held[1]].get('id')}")
        else:
            ns.append(f"{op.get('op')}/{i}/" + "/".join(map(str, held)))
        op, i, k = _get(op, path[k:j + 2]), path[j + 1], j + 2
    return f"{pre}{'/'.join(ns)}/{level}/{i}/" + ".".join(map(str, path[k:]))


def is_role(name):
    """True for the role of a toleranced part, wherever it sits."""
    return name.startswith("unit/") or "/unit/" in name


def _role(i, path, spec, pre="", op=None):
    """An ungrouped part's role is where it sits (`located`); a GROUP belongs to the whole
    record -- that is how two channels say they share a part."""
    g = spec.get("group")
    return f"unit/{g}" if g is not None else located(i, path, op, pre)


def _draw(spec, seed, role):
    """A resolved value and the `_unit` entry that says what it was drawn from."""
    value, name = quantile(spec, _u(seed, role))
    entry = {**spec, "seed": int(seed)}
    if name is not None:
        entry["corner"] = name
    return value, entry


# ---- finding and replacing them in an op ------------------------------------------------------
def is_toleranced(v):
    """True for the dict `tol()` or `corners()` returns: a value still to be drawn."""
    return isinstance(v, dict) and any(k in v for k in _MARK)


def _find(v, path, out, where):
    if isinstance(v, dict):
        if path and is_toleranced(v):
            out.append(path)
            return
        for k, c in v.items():
            if k == "from" and isinstance(c, dict):
                continue      # ANOTHER record, pinned as rendered: its unit is its own
            if k in ("_unit", "_segment") and path:
                raise ValueError(
                    f"{where}: a nested chain at {'.'.join(map(str, path))!r} arrives with its "
                    f"own resolved tolerances, drawn under its own seed, which this record's "
                    f"unit/* roles could not re-roll. Write the tolerance in place in the "
                    f"nested op's dict, or pass the chain as a Signal, whose tolerances this "
                    f"record then draws.")
            if isinstance(k, str) and k.startswith("_"):
                continue                                  # annotation, not a parameter
            _find(c, path + (k,), out, where)
    elif isinstance(v, (list, tuple)):
        for j, c in enumerate(v):
            if isinstance(c, (dict, list, tuple)):
                _find(c, path + (j,), out, where)


def _set(v, path, value):
    """A copy of `v` with the value at `path` replaced; only the containers on the path are
    copied, so everything else in the op is the same object it was."""
    k = path[0]
    new = value if len(path) == 1 else _set(v[k], path[1:], value)
    if isinstance(v, dict):
        return {**v, k: new}
    out = list(v)
    out[k] = new
    return tuple(out) if isinstance(v, tuple) else out


def _get(v, path):
    for k in path:
        v = v[k]
    return v


def _split(path):
    return tuple(int(k) if k.isdigit() else k for k in path.split("."))


def resolve(i, params, seed, pre=""):
    """An op's parameters with every toleranced value drawn -- called on the way INTO a recipe.

    `i` is the op's index (it names the role), `seed` the record's, `pre` the chain's name in
    a several-channel record (see `_role`). With no toleranced value
    the SAME dict comes back untouched, so a record that uses none is byte-identical to what it
    always was. Otherwise each is replaced by its number and described under `_unit`. A value
    that changes per occurrence (`per="segment"`, a `trend`) is replaced by its placeholder
    (`placeholder`) and kept under `_segment`, for `ws.Segments` to resolve occurrence by
    occurrence."""
    paths = []
    _find(params, (), paths, f"op {i}")
    if not paths:
        return params
    unit = dict(params.get("_unit", {}))
    seg = dict(params.get("_segment", {}))
    for path in paths:
        if any(isinstance(k, str) and ("." in k or k.isdigit()) for k in path):
            raise ValueError(
                f"op {i}: a toleranced value under the key {'/'.join(map(str, path))!r} cannot "
                f"be addressed (a role is written with dots and list indices). Rename the key.")
        name = ".".join(map(str, path))
        spec = _get(params, path)
        _check(spec, f"op {i}, parameter {name!r}")
        if path[-1] in WHOLE_NUMBERS and ("nom" in spec or spec.get("trend") == "ramp"):
            raise ValueError(
                f"op {i}, parameter {name!r}: {path[-1]} is a whole number, and a "
                f"{'ramp' if 'trend' in spec else 'tolerance'} gives a fraction of one, which "
                f"would be recorded and then truncated. Name the values it can take: "
                f"ws.corners({{'lo': ..., 'hi': ...}}).")
        if per_segment(spec):
            seg[name] = spec
            params = _set(params, path, placeholder(spec))
            continue
        value, unit[name] = _draw(spec, seed, _role(i, name, spec, pre, params))
        params = _set(params, path, value)
    out = {**params, "_unit": unit} if unit else dict(params)
    if seg:
        out["_segment"] = seg
    return out


def per_segment(spec):
    """True for a value that changes from one occurrence of a `ws.Segments` to the next."""
    return "trend" in spec or spec.get("per") == "segment"


def placeholder(spec):
    """The number an op holds for a per-occurrence value: the nominal, a corner set's first
    corner by name (or its `pick`), a trend's first value. It is the design's statement, not
    any occurrence's truth, which is why a chain holding one is refused when rendered alone."""
    if "trend" in spec:
        return spec["start"] if spec["trend"] == "ramp" else spec["before"]
    if "corners" in spec:
        return spec["corners"][spec.get("pick", sorted(spec["corners"])[0])]
    return spec["nom"]


def refuse_per_segment(ops, who="this record"):
    """Refuse a chain that still holds a per-occurrence value: rendered alone it has no one
    number to render, and the placeholder would be a silent stand-in."""
    for i, op in enumerate(ops):
        for name, spec in op.get("_segment", {}).items():
            what = (f"follows ws.trend({spec['trend']!r}, ...)" if "trend" in spec
                    else "is drawn per occurrence (ws.tol(..., per='segment'))")
            raise ValueError(
                f"{op.get('op')} (op {i}): {name} {what}, which has a value only across the "
                f"occurrences of ws.Segments, and {who} is rendered alone. Hand the chain to "
                f"ws.Segments(chain, n=, arrival=, trigger=...), or write a number (a "
                f"unit-to-unit part is ws.tol(...) without per=).")


def segment_entries(ops, pre=""):
    """``(op index, parameter, spec, location, role)`` for every per-occurrence value in an op
    list. `location` names the parameter (`vary/2/by`, `trend/1/by`); `role` is the
    randomness a segment draw comes from (`vary/2/by`, or `vary/<group>` shared), None for a
    trend."""
    for i, op in enumerate(ops):
        for name, spec in op.get("_segment", {}).items():
            if "trend" in spec:
                yield i, name, spec, located(i, name, op, pre, "trend"), None
                continue
            loc = located(i, name, op, pre, "vary")
            g = spec.get("group")
            yield i, name, spec, loc, (f"vary/{g}" if g is not None else loc)


def stated(ops):
    """An op list with every resolved value put back as the tolerance it was drawn from.

    This is what a `Signal` becomes when it is used INSIDE another record (a sub-chain, a net
    driver's source): the outer record's `resolve` then draws each one under its own seed and
    its own role, so `unit()` and `contrast("unit/*")` move the nested parts with the rest of
    the unit. An op with no toleranced value comes back as a plain copy."""
    out = []
    for op in ops:
        new = {k: v for k, v in op.items() if k not in ("_unit", "_segment")}
        for name, entry in op.get("_unit", {}).items():
            spec = {k: v for k, v in entry.items() if k not in ("seed", "corner")}
            new = _set(new, _split(name), spec)
        for name, spec in op.get("_segment", {}).items():
            new = _set(new, _split(name), spec)
        out.append(new)
    return out


def resolve_ops(ops, seed):
    """`resolve` over a recipe's op list -- so a recipe may be written with tolerances in place
    and `Signal.from_recipe` draws them under the recipe's own seed. A recipe whose values are
    already resolved passes through as the same objects, and replays bit-identically."""
    return [resolve(i, o, seed) for i, o in enumerate(ops)]


# ---- reading them back ------------------------------------------------------------------------
def _entries(ops, pre=""):
    for i, op in enumerate(ops):
        for name, entry in op.get("_unit", {}).items():
            yield i, name, entry, _role(i, name, entry, pre, op)


def roles(ops, pre=""):
    """The unit roles in an op list, in order, each once -- what `Signal.roles()` appends."""
    out = []
    for _i, _name, _entry, role in _entries(ops, pre):
        if role not in out:
            out.append(role)
    return out


def _chains(recipe):
    """``(chain name or None, role prefix, ops)`` for each chain of a recipe: the one of a
    Signal's, or every named chain of a `Channels` record's."""
    if "segments" in recipe:
        seg = recipe["segments"]
        return [(None, "", seg["ops"])] + [(name, name + "/", v["ops"])
                                           for name, v in seg.get("variants", {}).items()]
    if "channels" not in recipe:
        return [(None, "", recipe["ops"])]
    named = {**recipe["channels"], **recipe.get("shared", {})}
    return [(name, name + "/", c["ops"]) for name, c in named.items()]


def resolved(recipe):
    """Every toleranced parameter in a recipe, as a list of dicts in op order:

        {"role", "op", "index", "param", "value", "seed", <the stated tolerance>, ["corner"]}

    `value` is read from the parameter itself, so this is the ground truth and not a copy of it.
    The labelling route for a unit dataset: `[r["value"] for r in resolved(recipe)]`. For a
    several-channel recipe each also says which `chain` it is in."""
    out = []
    for chain, pre, ops in _chains(recipe):
        for i, name, entry, role in _entries(ops, pre):
            out.append({"role": role, **({} if chain is None else {"chain": chain}),
                        "op": ops[i]["op"], "index": i, "param": name,
                        "value": _get(ops[i], _split(name)), **entry})
    return out


def unit_of(recipe):
    """The seed this record's unit was drawn under -- the GROUP KEY for a split that holds out
    whole units: records with the same `unit_of` are captures of one unit.

    Raises when the record has no toleranced parameter (it is not a unit of anything) or when
    its parameters were drawn under different seeds (a partial re-roll: it is a mixture of two
    units, and filing it under either would leak one into the other's split)."""
    seeds = sorted({e["seed"] for _c, _pre, ops in _chains(recipe)
                    for _i, _n, e, _r in _entries(ops)})
    if not seeds:
        raise ValueError("unit_of: this recipe has no toleranced parameter, so it is not a "
                         "unit of a population. Write one with ws.tol(...) or ws.corners(...).")
    if len(seeds) > 1:
        raise ValueError(f"unit_of: this record's toleranced parameters were drawn under "
                         f"{len(seeds)} different seeds {seeds}, so it is not one unit. "
                         f"sig.unit(seed) draws all of them together.")
    return seeds[0]


# ---- drawing a design again -------------------------------------------------------------------
def redraw_ops(ops, seed_of, pre=""):
    """An op list with each unit role in `seed_of` (role -> seed; `ANY` matches all) drawn
    again. Only the toleranced values move. `pre` is the chain's name in a several-channel
    record (see `_role`)."""
    ops = list(ops)
    for i, name, entry, role in list(_entries(ops, pre)):
        seed = seed_of.get(role, seed_of.get(ANY))
        if seed is None:
            continue
        spec = {k: v for k, v in entry.items() if k not in ("seed", "corner")}
        value, new = _draw(spec, seed, role)
        ops[i] = _set(ops[i], _split(name), value)
        ops[i] = {**ops[i], "_unit": {**ops[i]["_unit"], name: new}}
    return ops


def _redraw(sig, seed_of, where):
    """A sibling Signal with each unit role in `seed_of` drawn again (`redraw_ops`); every
    other parameter, the seed and the grid are the same."""
    known = roles(sig.ops)
    if not known:
        raise ValueError(f"{where}: this record has no toleranced parameter, so there is no "
                         f"unit to draw. Write one with ws.tol(...) or ws.corners(...).")
    unknown = sorted(r for r in seed_of if r != ANY and r not in known)
    if unknown:
        raise ValueError(f"{where}: no unit role {unknown} in this record. It has: {known} "
                         f"(or '{ANY}' for all of them).")
    return dataclasses.replace(sig, ops=redraw_ops(sig.ops, seed_of))


def as_unit(sig, seed):
    """`Signal.unit(seed)`: the same design as the unit drawn under `seed`."""
    return _redraw(sig, {ANY: int(seed)}, "unit()")


def redrawn(sig, streams):
    """The record `streams` asks for: `sig` itself unless a `unit/...` role is re-rolled, in
    which case the sibling with those parameters drawn again. This is how `contrast("unit/*")`
    and `waveform(streams=...)` honour a unit role -- a toleranced value is resolved into the
    recipe, so re-rolling it is a different record rather than a different draw inside one."""
    # a sub-chain is run with a prefixed VIEW of the record's streams (subchain.Namespaced),
    # which has no overrides of its own: its tolerances were resolved, and are re-rolled, as
    # parameters of the outer op it sits in
    overrides = getattr(streams, "_overrides", {})
    seed_of = {k: v for k, v in overrides.items() if is_role(k)}
    if not seed_of:
        return sig
    same = sorted({r for _i, _n, e, r in _entries(sig.ops)
                   if seed_of.get(r, seed_of.get(ANY)) == e["seed"]})
    if same:
        raise ValueError(
            f"re-roll: {same} would be drawn under the seed they already have, so the 'other' "
            f"unit would be this one. Pass contrast(..., seed=<another seed>).")
    return _redraw(sig, seed_of, "re-roll")


# ---- N units x M captures ---------------------------------------------------------------------
def unit_dataset(build, n_units, n_captures, seed=0):
    """`n_units` units x `n_captures` captures of each, so a model can be tested on units it
    has never seen. Returns `(X, recipes)` exactly as `dataset` does -- it IS `dataset`, with
    each record's toleranced parameters drawn under its unit's seed instead of its own -- in
    unit-major order: rows `u * n_captures ... (u + 1) * n_captures - 1` are unit `u`.

        def build(rng):                                    # one CAPTURE: vary what a capture varies
            return (ws.Signal(seed=int(rng.integers(1 << 30)), grid=g)
                    .carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=1e6, tr_s=2e-9)
                    .probe(r_source=ws.tol(1e3, 5), c_load_f=100e-12, noise_rms=2e-3))

        X, recipes = ws.unit_dataset(build, n_units=20, n_captures=8)
        unit = np.array([ws.unit_of(r) for r in recipes])        # the group key, from the recipe
        test = np.isin(unit, np.unique(unit)[:4])                # hold out four whole units

    Split on `unit_of`, never on row position: a random split puts captures of one unit on both
    sides and the test score then measures memory of the unit.

    `build(rng)` must return the same design each call and vary something per capture (a
    `Signal(seed=)`, a pattern seed). It may return a `ws.Channels`: `X` is then
    ``{name: (n, L)}`` as from `dataset`, and `unit_of` reads the same key off its recipe. Captures of one unit that come out identical are refused:
    they would be one record counted `n_captures` times."""
    from .compose import dataset
    n_units, n_captures = int(n_units), int(n_captures)
    if n_units < 1 or n_captures < 1:
        raise ValueError(f"unit_dataset: n_units and n_captures must be at least 1, got "
                         f"{n_units} and {n_captures}")
    # the units' seeds come from their own stream, so asking for more captures per unit, or
    # changing what `build` draws, never changes which units these are
    useeds = np.random.default_rng([int(seed), _role_seed("unit")]).integers(
        1 << 53, size=n_units)
    k = itertools.count()
    seen = set()

    def one(rng):
        j = next(k)
        s = build(rng).unit(int(useeds[j // n_captures]))        # a Signal or a Channels
        h = s.sha256()
        if h in seen:
            raise ValueError(
                f"unit_dataset: capture {j % n_captures} of unit {j // n_captures} is "
                f"identical to an earlier record. build(rng) must vary something per capture "
                f"-- e.g. Signal(seed=int(rng.integers(1 << 30))) -- or every capture of a "
                f"unit is the same record.")
        seen.add(h)
        return s

    return dataset(one, n_units * n_captures, seed)


__all__ = ["tol", "corners", "trend", "quantile", "resolve", "resolve_ops", "roles", "resolved",
           "unit_of", "as_unit", "redrawn", "unit_dataset", "DISTS", "NORMAL_SIGMAS", "ANY"]

"""
wfmsynth.patterns — a NAME -> symbol-generator registry, and the generic mechanisms.

THE DESIGN DECISION, and everything below follows from it: **mechanisms live in this library,
standards knowledge does not.** A PRBS is a polynomial, a seed and a length; that is physics and
it ships here (`lfsr`, and `prbs<order>` for every polynomial in `physics.PRBS_TAPS`). *Which*
pattern a given standard names for a given lane rate, and what its published conformance numbers
are, is a fact that changes with the revision of a document — it is the caller's to state, by
registering a name with a `source=` clause that cites the clause. A physics library that hard-coded
those facts would be wrong the day a document was revised, and wrong silently.

    import wfmsynth.patterns as PAT

    PAT.resolve("prbs13", length=8191)                 # symbols from a shipped mechanism
    PAT.resolve("lfsr", taps=(10, 7), length=1023)     # any polynomial, no registration needed

    @PAT.pattern("my_link_test", levels=2, period=2048,
                 source="<standards designation, clause and table>")
    def _my_link_test(length, seed=1):                 # named, so a recipe can carry it
        ...

THE RECIPE CONTRACT (this is the part to read). A recipe records the pattern's **NAME and its
RESOLVED PARAMETERS, both**:

    {"op": "symbols",
     "pattern": {"name": "prbs13", "generator": "lfsr", "levels": 2, "period": 8191,
                 "params": {"taps": [13, 12, 2, 1], "seed": 9, "length": 4096}}}

Name alone (`{"pattern": "some_name"}`) cannot be replayed by anyone who does not have the
registry entry. Resolved parameters alone (a bare polynomial) can be replayed and cannot be
*read* — nobody can tell which pattern it is meant to be, or diff it against a document. Both
makes the JSON self-describing to a human AND replayable by a consumer who has this library and
nothing else: `replay` renders the block from the generic mechanism it names when the registry
entry is absent. `describe` records only the registry's own resolution plus the arguments the
caller actually passed — never a generator default, because the block is a content address and
restating a default would move the address of every record carrying the pattern the day that
default changed.

USER EXTENSION, WITHOUT LAMBDAS. A bare callable in a recipe cannot be serialised, hashed or
replayed, which breaks the property the whole archive rests on: a record that rebuilds itself
from its own JSON. So there are exactly two extension routes, and both round-trip:

  1. `@pattern(name, ...)` registers a NAMED generator and records a hash of its source in the
     block. A consumer missing the entry, or holding a different version of it, is told
     `needs pattern 'x' @ <hash>` instead of quietly rendering different samples under the same
     name. The hash covers user code only — a shipped entry is identified by `wfmsynth.__version__`
     instead, because hashing library source would churn the content address of every record on an
     unrelated edit. (Judgement. It is wrong if the library's generators ever become caller-editable
     in place; then they need hashes too.)
  2. `embed=True` on the authoring side writes the resolved SYMBOLS into the recipe as data. That
     needs no generator code at all on the consumer's side, and it is the only route that keeps
     working for a pattern whose generator the consumer will never have. It costs recipe size —
     one number per symbol — which is why it is opt-in rather than the default.

`physics.arbitrary(fn=...)` is the counter-example: it takes a callable, so a recipe carrying it
cannot be replayed from JSON. It is documented as such in its own docstring and the reproducible
route is here.
"""
from __future__ import annotations

import hashlib
import inspect
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import physics as P


class PatternUnavailable(LookupError):
    """The block names a pattern this process cannot build: no registry entry, no generic
    mechanism, no embedded symbols. The message names what is needed, including the source hash
    when the block carries one, so the consumer can go and get exactly that."""


class PatternMismatch(ValueError):
    """A registry entry with this NAME exists and is not the one the block was built with (its
    source hash differs). Refusing is the point: rendering a different sequence under the name the
    recipe asked for is the defect the hash exists to make impossible."""


# --------------------------------------------------------------- the generic mechanisms
#
# Two, and they are chosen because between them they cover the shape of every test pattern that is
# not a bespoke sequence: a maximal-length (or not) shift-register sequence, and a repeated block.
# Both are pure functions of their recorded parameters, so a block naming one is replayable by
# anyone holding this library — which is what makes the registry entry optional rather than load-
# bearing.
#
# EVERY GENERATOR TAKES `length` (the symbol count) as a parameter. That is the whole calling
# contract, it is checked at registration, and it is what lets a recipe carry one number for "how
# long" regardless of which generator produced the symbols.

def lfsr_bits(taps, length, seed=1, phase=0):
    """Raw LFSR bits (0/1) for an arbitrary feedback polynomial — a thin pass-through to
    `physics.lfsr`, which is the single bit engine `physics.prbs` also runs on. Kept here so the
    registry's mechanism and the kernel's PRBS cannot drift apart: there is only one loop."""
    return P.lfsr(taps, length, seed, phase)


def lfsr(length, taps, seed=1, phase=0):
    """Binary symbols (±1, one per UI) from an arbitrary-polynomial LFSR: the general mechanism
    behind every `prbs<order>` name and behind any polynomial a document specifies that this
    library does not happen to tabulate.

    `taps` are 1-based exponents of the feedback polynomial (`(7, 6)` is x^7 + x^6 + 1); the
    register width is `max(taps)`. `seed` is the initial state, which for a maximal-length
    polynomial IS the starting position in the sequence (see `physics.prbs`). Mapping is
    1 -> +1, 0 -> -1, matching `physics.carrier_symbols('nrz', ...)`.

    Primitivity is not checked — a non-primitive polynomial gives a short cycle rather than an
    error. Measure the period of what comes back (`measure.pattern_period`) if the polynomial did
    not come from a document.
    """
    return np.where(lfsr_bits(taps, length, seed, phase) > 0, 1.0, -1.0)


def block_repeat(block, length=None, repeat=None):
    """A fixed block of symbols, repeated: the other shape real test patterns come in (a word, a
    frame, a sequence of idle and data symbols) and the mechanism a caller reaches for when a
    document prints the sequence rather than a polynomial.

    Give `length` (symbols out, truncated mid-block if it does not divide) or `repeat` (whole
    repetitions). `block` is a list of LEVELS, not bits, so it carries any number of levels — a
    quaternary or ternary word needs no separate mechanism.
    """
    b = np.asarray(block, float).ravel()
    if b.size == 0:
        raise ValueError("block_repeat needs at least one symbol in `block`")
    if length is None and repeat is None:
        raise TypeError("block_repeat needs `length` (symbols) or `repeat` (whole repetitions)")
    n = int(b.size * int(repeat)) if length is None else int(length)
    return np.tile(b, int(np.ceil(n / b.size)))[:n]


# --------------------------------------------------------------- the registry
@dataclass(frozen=True)
class Pattern:
    """A registry entry. `fn(length=..., **params) -> symbols` is the generator; everything else
    is what makes a block self-describing.

    period    symbols before the sequence repeats, or None when it does not repeat inside any
              realistic record (a long PRBS) — the number an analyser pattern-locks to.
    levels    how many levels the sequence carries (2 for binary, 4 for quaternary). A consumer
              reads this to know which carrier the symbols belong on, and crossing them is the
              error `physics.carrier_symbols` already refuses to coerce.
    source    the provenance of the SEQUENCE, as a standards designation and clause — the one
              field that makes a name auditable against the document it came from. Left empty by
              the entries this library ships, because naming the standard that specifies a given
              polynomial is exactly the knowledge that does not belong in a physics library.
    marker    the alignment key: a short subsequence of LEVELS that occurs once per period, so a
              consumer can find where in the pattern a capture starts (`find_marker`). Optional,
              and declared by whoever registers the name, because which word a document designates
              is their knowledge, not this library's.
    generator the generic mechanism this entry resolves to, when it resolves to one — the field
              that makes the entry optional on the consumer's side.
    params    the parameters the ENTRY fixes (a polynomial, a block), which the block records
              alongside the caller's own arguments.
    digest    sha256 of the generator's source, first 12 hex — recorded for user code only.
    """
    name: str
    fn: Callable
    period: Optional[int] = None
    levels: Optional[int] = None
    source: Optional[str] = None
    marker: Optional[tuple] = None
    generator: Optional[str] = None
    params: dict = field(default_factory=dict)
    digest: Optional[str] = None


REGISTRY: dict[str, Pattern] = {}

# The mechanisms a consumer is guaranteed to have, keyed by the name a block records in
# `generator`. This is the whole reason a block without its registry entry still renders, so it
# holds ONLY functions that ship in this library and are pure in their recorded parameters.
MECHANISMS: dict[str, Callable] = {"lfsr": lfsr, "block_repeat": block_repeat}

_DEF_RE = re.compile(r"^\s*(async\s+)?def\b", re.M)


def source_digest(fn):
    """sha256 of a generator's source text, first 12 hex, or None when the source is unavailable
    (a built-in, a C function, a REPL definition). Decorator lines above the `def` are stripped:
    they carry the entry's METADATA, and a `source=` citation being corrected is not the generator
    changing. Unavailable source is recorded as no digest rather than as a guess — a digest that
    cannot be recomputed on the consumer's side would fail every comparison."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    m = _DEF_RE.search(src)
    if m:
        src = src[m.start():]
    return hashlib.sha256(inspect.cleandoc(src).encode("utf-8")).hexdigest()[:12]


def register(name, fn, period=None, levels=None, source=None, marker=None,
             generator=None, params=None, hash_source=False, replace=False):
    """Register `name` -> `fn`. Returns the `Pattern`.

    `fn` must take `length` (the symbol count); everything else it takes must be recordable in
    JSON, because a recipe replays it as `fn(**params)`. Pass `generator=`/`params=` when the
    entry is a shipped mechanism with fixed parameters (that is what makes the entry optional on
    a consumer's side); pass `hash_source=True` for caller-owned code, which `@pattern` does.

    Re-registering an existing name raises. A silently shadowed name is the same defect as a
    changed generator — the recipe still says one thing and the samples are another — so the
    overwrite has to be asked for with `replace=True`.
    """
    if not callable(fn):
        raise ValueError(f"pattern {name!r}: a generator must be callable")
    if name in REGISTRY and not replace:
        raise ValueError(f"pattern {name!r} is already registered (period="
                         f"{REGISTRY[name].period}, source={REGISTRY[name].source!r}); pass "
                         f"replace=True to overwrite it deliberately")
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):                      # unintrospectable: take it on trust
        sig = None
    if sig is not None and "length" not in sig.parameters and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        raise ValueError(f"pattern {name!r}: a generator must take `length` (the symbol count) -- "
                         f"got {tuple(sig.parameters)}. A recipe records one number for how long, "
                         f"whichever generator produced the symbols")
    entry = Pattern(name=name, fn=fn, period=None if period is None else int(period),
                    levels=None if levels is None else int(levels), source=source,
                    marker=None if marker is None else tuple(float(v) for v in marker),
                    generator=generator, params=dict(params or {}),
                    digest=source_digest(fn) if hash_source else None)
    REGISTRY[name] = entry
    return entry


def pattern(name, **meta):
    """Decorator: register a NAMED generator and hash its source, so a recipe using it is
    self-describing and a mismatch is named rather than silent.

        @pattern("my_link_test", levels=2, period=2048, source="<designation, clause>")
        def _my_link_test(length, seed=1):
            return PAT.block_repeat(WORD, length=length)

    The point of the decorator over a bare callable in a recipe: the recipe carries the NAME and
    the HASH, so a consumer without the code is told what it needs instead of getting a silent
    mismatch. `meta` is the `register` keywords (period, levels, source, marker, replace).
    Returns the function unchanged, so it stays directly callable and testable.
    """
    def deco(fn):
        register(name, fn, hash_source=True, **meta)
        return fn
    return deco


def unregister(name):
    """Remove a name. Returns the entry, or None. (A registry that can only grow cannot be
    tested against the consumer-without-the-entry case, which is the case that matters.)"""
    return REGISTRY.pop(name, None)


def get(name):
    """The `Pattern` entry, or raise `PatternUnavailable` naming what is registered."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise PatternUnavailable(
            f"no pattern {name!r} is registered; registered: {', '.join(names())}. A pattern a "
            f"standard names is registered by the caller -- see wfmsynth.patterns.pattern") from None


def names():
    """Registered names, sorted."""
    return sorted(REGISTRY)


# --------------------------------------------------------------- resolve / describe / replay
def resolve(name, **kw):
    """Symbols for a registered pattern: `resolve('prbs13', length=8191, seed=9)`.

    `length` (the symbol count) is required — by the calling contract every generator takes it,
    and a pattern with no length is not a sequence yet. Any other keyword goes to the generator,
    on top of the parameters the entry itself fixes."""
    entry = get(name)
    params = _params(entry, kw)
    if "length" not in params:
        raise TypeError(f"resolve({name!r}) needs length= (the number of symbols)")
    return np.asarray(entry.fn(**params), float)


def describe(name, **kw):
    """The BLOCK a recipe records: the name, the resolved parameters, and the metadata that makes
    the JSON readable on its own. See the module docstring for the contract.

    Only the entry's own resolution and the caller's own arguments are recorded — no generator
    default is restated, because the block is part of a content address."""
    entry = get(name)
    params = _params(entry, kw)
    if "length" not in params:
        raise TypeError(f"describe({name!r}) needs length= (the number of symbols)")
    block = {"name": name, "params": _jsonable(params)}
    if entry.generator:
        block["generator"] = entry.generator
    for key, val in (("levels", entry.levels), ("period", entry.period),
                     ("source", entry.source), ("digest", entry.digest)):
        if val is not None:
            block[key] = val
    if entry.marker is not None:
        block["marker"] = [float(v) for v in entry.marker]
    return block


def replay(block):
    """Symbols from a recorded block — the consumer's side of the contract. Three routes, in the
    order that needs least from the consumer:

      1. embedded `symbols`: data, no code, always works;
      2. the registry entry for `name`, hash-checked against the block when both carry one;
      3. the generic `generator` the block names, which ships in this library — so a block whose
         registry entry is absent still renders exactly the sequence it recorded.

    When none of the three is available the error NAMES what is missing, hash included, rather
    than falling back to something else."""
    if block.get("symbols") is not None:
        return np.asarray(block["symbols"], float)
    name = block.get("name")
    params = dict(block.get("params") or {})
    entry = REGISTRY.get(name)
    if entry is not None:
        want = block.get("digest")
        if want and want != entry.digest:
            raise PatternMismatch(
                f"needs pattern {name!r} @ {want}; the entry registered here is "
                f"@ {entry.digest or 'unhashed'}. Same name, different generator -- install the "
                f"one the recipe was built with, or replay a recipe with embedded symbols")
        return np.asarray(entry.fn(**params), float)
    gen = block.get("generator")
    if gen in MECHANISMS:
        return np.asarray(MECHANISMS[gen](**params), float)
    raise PatternUnavailable(
        f"needs pattern {name!r} @ {block.get('digest') or 'unhashed'}: it is not registered here "
        f"and its block names no generic mechanism ({gen!r}). Register that name, or rebuild the "
        f"recipe with embedded symbols (Signal.pattern(..., embed=True)), which needs no code")


def _params(entry, kw):
    """The entry's fixed parameters with the caller's on top. The caller wins: an entry that fixes
    a polynomial still lets a caller override it, and the recipe records what was actually used."""
    return {**entry.params, **kw}


def _jsonable(params):
    """Numpy scalars and arrays out, JSON-native values in — a recipe that carries an np.int64 is
    not a recipe, and the failure shows up at serialization time far from its cause."""
    out = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            out[k] = [_scalar(x) for x in v.ravel()]
        elif isinstance(v, (list, tuple)):
            out[k] = [_scalar(x) for x in v]
        else:
            out[k] = _scalar(v)
    return out


def _scalar(v):
    if isinstance(v, (bool, str)) or v is None:
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return v


# --------------------------------------------------------------- statistics and alignment
def symbol_stats(symbols):
    """Transition density and per-level probability of a symbol sequence.

    The instrument for checking a registered pattern against a document's PUBLISHED conformance
    numbers — which is how a caller establishes that the sequence they registered is the one the
    standard designates, rather than assuming it. Calibrated in the tests on a maximal-length
    PRBS, whose values are closed form (density 0.5, levels equiprobable to within one symbol per
    period).

    Transition density counts CHANGES between adjacent symbols over the n-1 adjacent pairs — the
    quantity a standard quotes for a multilevel sequence, where a change of one level and a change
    of three both count as one transition."""
    s = np.asarray(symbols, float)
    if s.size == 0:
        raise ValueError("symbol_stats needs at least one symbol")
    lv, counts = np.unique(s, return_counts=True)
    dens = float(np.count_nonzero(np.diff(s) != 0) / (s.size - 1)) if s.size > 1 else 0.0
    return {"transition_density": dens,
            "levels": int(lv.size),
            "level_probability": {float(a): float(c / s.size) for a, c in zip(lv, counts)},
            "n": int(s.size)}


def find_marker(symbols, marker):
    """Indices where `marker` (a subsequence of LEVELS) starts in `symbols`.

    A marker that occurs once per period is an alignment key: it says where in the pattern a
    capture begins, which is what makes a repeating sequence usable as position ground truth.
    Exact equality on levels — these are transmitted symbols, not measured ones; align a MEASURED
    stream with `measure.align_symbols` first."""
    s = np.asarray(symbols, float)
    m = np.asarray(marker, float).ravel()
    if m.size == 0 or m.size > s.size:
        return np.empty(0, dtype=int)
    win = np.lib.stride_tricks.sliding_window_view(s, m.size)
    return np.flatnonzero((win == m).all(axis=1))


# --------------------------------------------------------------- what this library ships
#
# Mechanisms, and the polynomials the kernel already tabulates. Nothing here names a standard: the
# tap sets come from `physics.PRBS_TAPS`, so an order added to that table becomes a registered
# pattern by construction and cannot drift out of sync with it, and WHICH document specifies a
# given polynomial for a given link is the caller's `source=` to state.
register("lfsr", lfsr, levels=2, generator="lfsr",
         source=None)                      # any polynomial; period depends on primitivity
register("block_repeat", block_repeat, generator="block_repeat", source=None)
for _order, _taps in sorted(P.PRBS_TAPS.items()):
    register(f"prbs{_order}", lfsr, levels=2, period=(1 << _order) - 1,
             generator="lfsr", params={"taps": list(_taps)})
# DELIBERATELY NOT REGISTERED HERE: the kernel's quaternary sequences (`physics.prbs13q`,
# `physics.prbs31q`) and the pattern names `physics.carrier_symbols` takes. Those are sequences a
# STANDARD designates -- the entry that names one belongs to whoever can cite the clause it comes
# from and keep the citation current, which is the caller. `wfmsynth.standard_patterns` is that
# caller for the sequences this library has been able to cite (SSPRQ, the Clause 72 pattern, the
# compliance patterns): importing that module registers them, and nothing here imports it, so this
# module and the kernel below it stay document-free. Registering one is three lines, and the
# generator must take `length` (the calling contract), so a kernel function with a different
# parameter name gets a small named adapter rather than a lambda:
#
#     def _q_sequence(length, seed=1):          # named, so the recipe can carry it and hash it
#         return P.prbs13q(length, seed)
#     PAT.register("<the standard's designation>", _q_sequence, levels=4, period=8191,
#                  source="<designation, clause>")
#
# The 1010... clock, as a repeated block rather than a special case: one transition per UI, no
# runs, no low-frequency content -- the deliberate ISI-free contrast to a long PRBS. Identical to
# `physics.clock_pattern`, asserted in the tests.
register("clock", block_repeat, levels=2, period=2, generator="block_repeat",
         params={"block": [1.0, -1.0]})

# Package-level aliases. Inside this module `register`/`resolve` are the right names; at the top
# of the package they would be ambiguous, so `wfmsynth.register_pattern(...)` is what is exported.
# `PATTERNS` IS `REGISTRY` -- the same dict under the naming the package's other tables use
# (IMPAIRMENTS, MECHANISMS, PATHOLOGIES), not a copy, so a registration through either is visible
# in both.
register_pattern = register
resolve_pattern = resolve
describe_pattern = describe
replay_pattern = replay
PATTERNS = REGISTRY

__all__ = ["Pattern", "PatternUnavailable", "PatternMismatch", "REGISTRY", "MECHANISMS",
           "register", "pattern", "unregister", "get", "names", "resolve", "describe", "replay",
           "lfsr", "lfsr_bits", "block_repeat", "symbol_stats", "find_marker", "source_digest",
           "register_pattern", "resolve_pattern", "describe_pattern", "replay_pattern", "PATTERNS"]

#!/usr/bin/env python3
"""Replay a recipe document back into waveforms, using this library and nothing else.

    python replay.py RECIPE.json OUT_DIR [--only ID] [--dry-run] [--no-verify]

WHAT A RECIPE IS, AND WHY THIS FILE IS ENOUGH TO REBUILD AN ARCHIVE
-------------------------------------------------------------------
A record's ground truth is not its samples — it is the ordered list of composer ops that
produced them, plus the grid and the seed. `wfmsynth.Signal` executes exactly that list, so
rebuilding a record needs no builder, no authoring layer and no per-archive script: it needs a
document, this file, and `wfmsynth` at the commit the document pins. That is the whole point.
A replay path that lives with one archive can only rebuild that archive; this one can rebuild
any document of the shape below.

    {"records": {"<record id>": {"ops": [...], "grid": {...}, "seed": 0,
                                 "values_sha256": "<hex>"}}}

`records` may be omitted and the top level be the id -> record mapping itself. `grid`, `seed`,
`lead_in`, `lead_out` and `values_sha256` are each optional; an absent key means the kernel's
own default, which is deliberate — restating a default here would freeze today's value into
every record this tool touches.

THE CHAIN, AND WHERE EACH OP SITS IN IT
---------------------------------------
Everything in a recipe is one link of one chain. Read it in this order and each op's place is
obvious:

  BITS        a binary sequence — a PRBS of stated order, or a standard's payload. Not yet a
              signal: no levels, no time. PRBS order is the single most consequential choice
              in a recipe, because channel ISI is a function of pattern history (PRBS7 repeats
              every 127 bits and renders a lossy channel systematically more open than the
              lab does; compliance work uses PRBS31 for exactly that reason).

  CODING      line coding / scrambling: 8b/10b-style running-disparity block inversion, or the
              64b/66b self-synchronous scrambler (`wfmsynth.coding`). This is what bounds DC
              wander and run length on a real link, and it is why an AC-coupled real capture
              does not baseline-wander the way a raw-PRBS render does.

  SYMBOLS     bits grouped into per-UI symbols and mapped to levels: one bit per symbol for
              NRZ, two for PAM4 with a Gray map. The symbol sequence is where a STANDARD test
              pattern lives (PRBS13Q is a quaternary sequence, not a binary one dressed up),
              and it is what an instrument pattern-locks to.
                -> ops: `carrier` (kind + a pattern in the kernel's own vocabulary) and
                   `symbols`, which is the SAME op whether its symbols arrive as a literal
                   list (a coded or scrambled stream) or as a registry `pattern` block naming
                   a standard sequence. A block records the name AND the parameters it
                   resolved to, so a record renders from the block where the registry entry is
                   installed, from the generic mechanism the block names where it is not, and
                   from embedded symbols with no generator code at all. See PATTERNS below.

  LEVELS      symbol levels laid onto the sample grid at the UI boundaries, with edges shaped
              to a finite rise time and, if asked, transmitter jitter applied at the edge
              TIMES rather than to the shaped samples. Still inside the `carrier`/`symbols` op:
              one op spans symbols -> levels, because the edge times and the levels are the
              same array.

  WAVEFORM    what the transmitter, the package and the board do to that ideal drive. Shaping
              first (`tx_ffe`, `de_emphasis`, `nonlinearity`, `dcd`, `ssc`, `timing`,
              `intra_pair_skew`, `events`), then supply (`supply_coupling`, `drift`), then the
              channel (`lossy`, `sparam`, `cascade`, `reflect`, `resonant_reflect`,
              `crosstalk`, `crosstalk_matrix`, `ac_couple`; `eo`/`fiber`/`edfa`/`optical_mpi`
              for an optical span). These are physics: causal, frequency-dependent, and the
              only place loss and reflections may enter.

  INSTRUMENT  what OBSERVES the waveform, which is not part of it: `probe` (loading + its own
              noise), then the receiver or scope front end (`ctle`, `dfe`, `rx_ffe`, `agc`,
              `rx_noise`, `tia`, `photodetect`, `sample_clock`, `scope`, `timebase`), then the
              converter (`digitize`) and finally `store`/`acquire` — the integer-code export
              that puts a real record on a lattice and gives it its noise floor.

A record's op order is the chain's order. The replayer does not reorder anything: it executes
the document as written, because the document is the answer, not a hint.

PATTERNS: WHAT A DOCUMENT NEEDS INSTALLED, AND WHAT IT CARRIES
--------------------------------------------------------------
A `symbols` op's pattern block is resolved by the composer, not here — a second implementation
of the same lookup could disagree with the first, and two disagreeing resolutions of one name
are worse than one. What this file does is make the DEPENDENCY visible, because it is the only
thing that can make an otherwise self-contained document unreplayable. Three cases, in the
order that needs least from the consumer:

  embedded symbols     the block carries the sequence. No generator code, no registry: it
                       replays anywhere this library runs. Report column says "embedded".
  a generic mechanism  the block names one that ships here (an LFSR with a recorded
                       polynomial, a repeated block) and renders from its own parameters, so
                       the registry entry is convenience, not a dependency.
  a caller's generator the block names code that must be INSTALLED, at the generator hash the
                       block recorded. A record like this is flagged in the report even on a
                       run where it passed, because that is the run on which the reader can
                       still do something about it. A name resolving to a different generator
                       is an error naming both hashes, never different samples under the same
                       name.

WHAT IT VERIFIES
----------------
For each record it recomputes a digest over the RENDERED VALUES and compares it against the
digest the document carries. Everything is reported by RECORD ID — never by row position,
because a row number is only meaningful inside the one table it was read from, and the whole
reason to replay is to compare across stores that may not agree on order.

A tool that exits 0 having checked nothing is worse than a tool that fails, so: an empty
selection is an error, an `--only` id that is not in the document is an error, and a run that
verified no digest at all is an error unless `--no-verify` says in as many words that only the
renders were wanted.

A record may declare `values_digest_alg`; a value this replayer does not implement is an ERROR
for that record, not a skipped check, for the same reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import wfmsynth as ws
except ImportError:                                   # running from a checkout, not an install
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import wfmsynth as ws


# ------------------------------------------------------------------ the values digest
# The algorithm id goes INTO the preimage. A digest whose algorithm changed must be
# distinguishable from a digest whose data changed: with the id hashed in, a document digested
# under v1 and recomputed under a future v2 disagrees in a way a tool can attribute to the
# algorithm, instead of reporting MISMATCH and sending someone to hunt corruption that is not
# there.
VALUES_DIGEST_ALG = "wfmsynth-values-sha256-v1"
_BLOCK = 1 << 20            # digest in blocks so a deep-memory record needs no second copy


def values_digest(x, dtype: str = "<f8") -> str:
    """sha256 over a rendered record's values, under algorithm `wfmsynth-values-sha256-v1`.

    Preimage: ``f"{VALUES_DIGEST_ALG}|{size}|{dtype.str}|"`` then the values as contiguous
    little-endian bytes of `dtype`. The length and dtype are in the header because otherwise a
    truncated record and a narrower one could collide with a full float64 one.

    Byte order is PINNED little-endian rather than native so the digest is a property of the
    record and not of the host that computed it. `dtype` is the dtype the archive STORES: a
    store holding float32 must be digested as float32, or a consumer reading the file and a
    consumer replaying the recipe compute two different numbers for the same record and one of
    them is wrong for no physical reason.
    """
    dt = np.dtype(dtype)
    a = np.ascontiguousarray(np.asarray(x)).astype(dt, copy=False)
    h = hashlib.sha256()
    h.update(f"{VALUES_DIGEST_ALG}|{a.size}|{dt.str}|".encode())
    for lo in range(0, a.size, _BLOCK):
        h.update(np.ascontiguousarray(a[lo:lo + _BLOCK]).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ patterns
# A recipe may name a standard test SEQUENCE rather than spelling it out: the source op carries a
# pattern BLOCK -- the name, plus the parameters the registry resolved it to (the polynomial, the
# block, the seed, the length) and a hash of the generator. The composer resolves that block when
# it executes the op (`wfmsynth.patterns.replay`), so this replayer does not resolve patterns
# itself; resolving them here would be a SECOND implementation of the same lookup, and two
# implementations of a resolution that could disagree are worse than one.
#
# What is this file's job is making the dependency VISIBLE, because it is the one thing that can
# make an otherwise self-contained document unreplayable:
#   * a block with embedded `symbols` needs no code at all -- the always-works route;
#   * a block naming a generic mechanism renders from the parameters it recorded, registry or not;
#   * a block that needs a registered entry renders only where that entry is installed, at the
#     same generator hash.
# So the report says which sequence each record carries, and a run that depends on an installed
# entry says so rather than leaving the reader to find out on another machine.
def pattern_label(ops):
    """A short description of the SEQUENCE a record's source op carries, for the report.

    Three shapes, because a source can say where its symbols came from in three ways: a carrier
    op naming a pattern in the kernel's own vocabulary, a symbols op carrying a registry block,
    or a symbols op carrying the symbols themselves."""
    for op in ops:
        if op.get("op") == "carrier":
            return f"{op.get('kind', '?')}:{op.get('pattern') or 'default'}"
        if op.get("op") == "symbols":
            b = op.get("pattern")
            if isinstance(b, dict) and b.get("name"):
                lab = b["name"]
                if b.get("digest"):
                    lab += f"@{b['digest']}"
                return lab + (" embedded" if op.get("symbols") is not None else "")
            if op.get("symbols") is not None:
                return f"symbols[{len(op['symbols'])}]"
    return ""


def needs_installed_pattern(ops):
    """True when a record can only be rendered where a registry ENTRY is installed — no embedded
    symbols and no generic mechanism named in the block. This is the one dependency a document
    cannot carry, so it is worth reporting even on a run where every digest matched, because it
    is what will fail on the next machine."""
    for op in ops:
        if op.get("op") != "symbols" or op.get("symbols") is not None:
            continue
        b = op.get("pattern")
        if isinstance(b, dict) and b.get("name") and not b.get("generator"):
            return True
    return False


# ------------------------------------------------------------------ rendering
def _grid(spec):
    if spec is None:
        return None
    # A null and an absent key mean the same thing: the kernel's default. `Signal.recipe()`
    # writes `"baud": null` for an unbound symbol rate, and passing that through would hand Grid
    # a None where it wants a number.
    kw = {k: v for k, v in spec.items() if v is not None}
    if kw.get("segments"):
        # JSON has no tuples and Grid is frozen on them. Note that `Signal.recipe()` does not
        # emit segments, so a segmented grid is something the document's author added.
        kw["segments"] = tuple(tuple(s) for s in kw["segments"])
    return ws.Grid(**kw)


def render(record):
    """Rebuild one record's samples from its recipe. Only the keys the record states are
    passed to `Signal`, so every unstated knob keeps the kernel's default."""
    kw = {}
    grid = _grid(record.get("grid"))
    if grid is not None:
        kw["grid"] = grid
    for k in ("seed", "lead_in", "lead_out"):
        if k in record and record[k] is not None:
            kw[k] = record[k]
    sig = ws.Signal(**kw)
    sig.ops = [dict(o) for o in record["ops"]]
    return sig.waveform()


# ------------------------------------------------------------------ the document
def _check_record_id(rid):
    """Record ids name output files, and a document is untrusted input. A '/' is allowed
    because real ids are hierarchical, but nothing that could leave the output tree is."""
    if not isinstance(rid, str) or not rid:
        raise ValueError(f"record id must be a non-empty string; got {rid!r}")
    if "\\" in rid or "\x00" in rid or rid.startswith("/") or ":" in rid:
        raise ValueError(f"record id {rid!r} is not usable as a path")
    for seg in rid.split("/"):
        if seg in ("", ".", ".."):
            raise ValueError(f"record id {rid!r} escapes the output tree")


def load_document(path):
    """Read a recipe document -> {record id: record}. Accepts either the wrapped form
    (``{"records": {...}}``) or a bare id -> record mapping."""
    with open(path) as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: a recipe document must be a JSON object")
    recs = doc["records"] if isinstance(doc.get("records"), dict) else doc
    # A key that is not record-shaped is an error rather than something to skip: skipping is how
    # a mis-keyed document gets checked zero times and still exits 0.
    bad = [k for k, v in recs.items() if not (isinstance(v, dict) and isinstance(v.get("ops"), list))]
    if bad and recs is doc:
        bad = [k for k in bad if k not in ("wfmsynth_version", "records", "meta")]
    if bad:
        raise ValueError(f"{path}: these keys are not records with an 'ops' list: {sorted(bad)}")
    for rid in recs:
        _check_record_id(rid)
    return {rid: recs[rid] for rid in sorted(recs)}


# ------------------------------------------------------------------ the run
@dataclass
class Result:
    record_id: str                      # the ONLY identity in this report. Never a row index.
    status: str                         # ok | rendered | MISMATCH | ERROR
    n: int = 0
    digest: Optional[str] = None
    expected: Optional[str] = None
    detail: str = ""
    path: Optional[str] = None
    pattern: str = ""                   # which SEQUENCE this record carries (see pattern_label)
    needs_entry: bool = False           # renderable only where a registry entry is installed


@dataclass
class Report:
    results: list = field(default_factory=list)
    n_verified: int = 0
    require_verification: bool = True

    @property
    def exit_code(self):
        if any(r.status in ("MISMATCH", "ERROR") for r in self.results):
            return 1
        if not self.results:
            return 3
        if self.require_verification and self.n_verified == 0:
            return 3
        return 0

    def lines(self):
        w = max((len(r.record_id) for r in self.results), default=9)
        p = max((len(r.pattern) for r in self.results), default=7)
        out = [f"{'record id'.ljust(w)}  {'status':<9} {'samples':>10}  "
               f"{'pattern'.ljust(p)}  digest"]
        for r in self.results:                       # already in record-id order
            d = (r.digest or "")[:16]
            out.append(f"{r.record_id.ljust(w)}  {r.status:<9} {r.n:>10}  "
                       f"{r.pattern.ljust(p)}  {d}" + (f"   {r.detail}" if r.detail else ""))
        n = len(self.results)
        out.append(f"{n} record(s), {self.n_verified} digest(s) verified, "
                   f"{sum(1 for r in self.results if r.status in ('MISMATCH', 'ERROR'))} failed")
        n_entry = sum(1 for r in self.results if r.needs_entry)
        if n_entry:
            # Not a failure here — it failed nowhere on THIS machine, which is exactly why it is
            # worth saying. The document is only as portable as the registry entries it needs.
            out.append(f"NOTE: {n_entry} record(s) render only where the named registry entry is "
                       f"installed at the recorded generator hash. Rebuilding them with embedded "
                       f"symbols makes the document replayable with no generator code.")
        if self.exit_code == 3 and self.results:
            out.append("NOTHING WAS VERIFIED: no record carried a digest under "
                       f"{VALUES_DIGEST_ALG}. Pass --no-verify if renders alone were wanted.")
        elif not self.results:
            out.append("NOTHING WAS SELECTED: zero records to replay.")
        return "\n".join(out)

    def to_json(self):
        return {"values_digest_alg": VALUES_DIGEST_ALG,
                "wfmsynth_version": ws.__version__,
                "n_records": len(self.results), "n_verified": self.n_verified,
                "exit_code": self.exit_code,
                "records": {r.record_id: {"status": r.status, "n": r.n, "digest": r.digest,
                                          "expected": r.expected, "detail": r.detail,
                                          "path": r.path, "pattern": r.pattern,
                                          "needs_registry_entry": r.needs_entry}
                            for r in self.results}}


def replay(records, out_dir=None, only=None, dry_run=False, require_verification=True):
    """Render, verify and (unless `dry_run`) write every selected record."""
    if only:
        missing = [rid for rid in only if rid not in records]
        if missing:
            # The classic silent pass: a mistyped id selects nothing and the run "succeeds".
            raise ValueError(f"--only named record id(s) that are not in this document: "
                             f"{sorted(missing)}")
        records = {rid: records[rid] for rid in sorted(only)}
    rep = Report(require_verification=require_verification)
    for rid, rec in records.items():                 # sorted by id in load_document
        ops = rec.get("ops") or []
        label, needs = pattern_label(ops), needs_installed_pattern(ops)
        try:
            alg = rec.get("values_digest_alg", VALUES_DIGEST_ALG)
            if alg != VALUES_DIGEST_ALG:
                # Cannot verify, so cannot pass. Reporting this as "no digest" would turn an
                # unverifiable record into a silently unchecked one.
                raise ValueError(f"record declares digest algorithm {alg!r}; this replayer "
                                 f"implements {VALUES_DIGEST_ALG!r} only")
            x = render(rec)
            got = values_digest(x, rec.get("values_dtype", "<f8"))
            want = rec.get("values_sha256")
            if want is None:
                res = Result(rid, "rendered", len(x), got)
            elif got == want:
                res = Result(rid, "ok", len(x), got, want)
                rep.n_verified += 1
            else:
                res = Result(rid, "MISMATCH", len(x), got, want,
                             f"expected {want[:16]}")
            # A MISMATCH is still written out. The first question after a mismatch is always
            # "how do they differ", and that needs the record that was actually produced.
            if not dry_run and out_dir is not None:
                p = os.path.join(out_dir, *rid.split("/")) + ".npy"
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, x)
                res.path = p
        except Exception as exc:                     # one bad record must not hide the rest
            # A missing or changed registry entry arrives here as PatternUnavailable /
            # PatternMismatch, whose message already names the pattern and its hash. Reporting it
            # against the RECORD ID is what makes it actionable in a document of thousands.
            res = Result(rid, "ERROR", detail=f"{type(exc).__name__}: {exc}")
        res.pattern, res.needs_entry = label, needs
        rep.results.append(res)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Replay a recipe document into waveforms and verify them by digest.")
    ap.add_argument("recipe", help="recipe document (JSON): id -> {ops, grid, ...}")
    ap.add_argument("out_dir", nargs="?", help="directory for the rendered .npy records "
                                              "(omit only with --dry-run)")
    ap.add_argument("--only", action="append", metavar="ID",
                    help="replay just this record id; repeatable")
    ap.add_argument("--dry-run", action="store_true",
                    help="render and verify, write nothing")
    ap.add_argument("--no-verify", action="store_true",
                    help="accept a document that carries no digests (renders only). Without "
                         "this, a run that verified nothing FAILS.")
    a = ap.parse_args(argv)
    if a.out_dir is None and not a.dry_run:
        ap.error("OUT_DIR is required unless --dry-run is given")
    try:
        records = load_document(a.recipe)
        rep = replay(records, out_dir=a.out_dir, only=a.only, dry_run=a.dry_run,
                     require_verification=not a.no_verify)
    except (OSError, ValueError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(rep.lines())
    if not a.dry_run and a.out_dir is not None:
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, "replay.report.json"), "w") as fh:
            json.dump(rep.to_json(), fh, indent=2, sort_keys=True)
    return rep.exit_code


if __name__ == "__main__":
    sys.exit(main())

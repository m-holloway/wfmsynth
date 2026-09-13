"""
This library ships publicly and on its own. A gate that keeps it that way.

Two properties, both easy to break by accident in a comment and neither visible in any other test:

  SELF-CONTAINED. Nothing here may reference a sibling project by name -- not in code, not in a
  docstring, not in a path, not in an environment variable's default. A reader with only this
  repository has to be able to follow every reference. A cross-reference to a package they cannot
  see is a dead end, and one that appears in an ENV VAR DEFAULT is worse, because it encodes one
  machine's directory layout as a fallback.

  VENDOR-NEUTRAL. No test-equipment maker, instrument model number, or customer. Standards bodies
  and standard designations are exactly what SHOULD be cited -- "IEEE 802.3 Clause 120.5.11.2.3",
  "PCI Express Base Specification Rev 4.0 Table 8-7" -- and a document is citable without naming
  whoever sells the scope it was measured on.

Why a test rather than a review habit: this repository already carries cited numbers from a dozen
standards, and the natural way to write a comment about where a figure came from is to name the
place you got it. That instinct is right for a standard and wrong for a vendor or a sibling package,
and the difference is invisible at the moment of writing. So it is checked.

Scope is the tracked tree, so the gate covers scratch under `spikes/` as well as the library. A file
that genuinely needs an exception gets one in ALLOW, with a reason.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Sibling projects and internal tooling. A name here is not secret -- it is simply not resolvable by
# someone holding only this repository, which is what makes it a dead reference.
SIBLING = ("wfmplan", "wfmsynth_demo", "build_multibus", "build_corpus")

# Test-equipment makers. Naming a standards body is encouraged; naming who sells the instrument is
# not, because it dates the document and reads as an endorsement.
VENDOR = ("keysight", "tektronix", "lecroy", "teledyne", "anritsu", "rohde", "granite river",
          "agilent", "advantest", "viavi")

# Files this gate does not read: binary, generated, or this file itself -- which necessarily contains
# every string it forbids.
SKIP_SUFFIX = (".png", ".jpg", ".pdf", ".s2p", ".s4p", ".npz", ".npy", ".zip", ".ipynb")
SKIP_NAME = ("test_public_hygiene.py",)

# Exceptions, each with the reason it is one. Empty is the goal.
ALLOW: dict[str, str] = {}


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        p = ROOT / line
        if p.name in SKIP_NAME or p.suffix.lower() in SKIP_SUFFIX or not p.is_file():
            continue
        yield line, p


def _hits(text, needles):
    low = text.lower()
    return sorted({n for n in needles if n in low})


def _offending_lines(p, needles):
    out = []
    for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
        got = _hits(line, needles)
        if got:
            out.append(f"    {i}: {line.strip()[:110]}   <- {got}")
    return out


def test_the_gate_can_actually_read_the_tree():
    """Calibrate before trusting: a gate that silently reads nothing passes forever. Two independent
    facts, so a broken `git ls-files` or a wrong root cannot look like a clean tree."""
    files = list(tracked_files())
    assert len(files) > 50, f"only {len(files)} tracked files readable -- the gate is not scanning"
    assert any(str(n).endswith("wfmsynth/physics.py") for n, _ in files), \
        "the library's own physics module is not in the scan"


def test_no_file_references_a_sibling_project():
    bad = {}
    for name, p in tracked_files():
        if name in ALLOW:
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if _hits(text, SIBLING):
            bad[name] = _offending_lines(p, SIBLING)
    assert not bad, (
        "these files name a sibling project, which a reader holding only this repository cannot "
        "resolve:\n" + "\n".join(f"  {k}\n" + "\n".join(v) for k, v in sorted(bad.items())))


def test_no_file_names_a_test_equipment_vendor():
    bad = {}
    for name, p in tracked_files():
        if name in ALLOW:
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if _hits(text, VENDOR):
            bad[name] = _offending_lines(p, VENDOR)
    assert not bad, (
        "these files name a test-equipment maker. Cite the DOCUMENT -- body, designation, clause, "
        "table -- which is citable without naming who sells the instrument:\n"
        + "\n".join(f"  {k}\n" + "\n".join(v) for k, v in sorted(bad.items())))


def test_no_environment_default_encodes_a_local_directory_layout():
    """An env var whose DEFAULT is a path under someone's home directory makes one machine's layout
    the fallback. It works there and nowhere else, and it is the quietest way a private path ships."""
    pat = re.compile(r"environ\.get\(\s*['\"][^'\"]+['\"]\s*,\s*[^)]*Path\.home\(\)")
    bad = []
    for name, p in tracked_files():
        if p.suffix != ".py" or name in ALLOW:
            continue
        text = p.read_text(errors="replace")
        for m in pat.finditer(text):
            bad.append(f"  {name}: {text[m.start():m.start()+100].splitlines()[0]}")
    assert not bad, ("an environment default resolves into a home directory:\n" + "\n".join(bad)
                     + "\n  Require the variable, or default to something relative.")


def test_the_allow_list_is_empty_or_every_entry_carries_a_reason():
    """An exception without a stated reason becomes permanent. A short, honest reason is the price."""
    for name, why in ALLOW.items():
        assert isinstance(why, str) and len(why) > 20, f"{name}: give a real reason, got {why!r}"


@pytest.mark.parametrize("needles,label", [(SIBLING, "sibling"), (VENDOR, "vendor")])
def test_the_gate_would_catch_a_reintroduction(tmp_path, needles, label):
    """Negative fixture: prove the matcher fires. A gate whose detector is broken reports a clean
    tree, which is indistinguishable from a clean tree."""
    probe = f"# a comment mentioning {needles[0]} in passing\n"
    assert _hits(probe, needles) == [needles[0]], f"{label} matcher does not fire"
    assert _hits("# IEEE 802.3 Clause 120.5.11.2.3, Table 8-7\n", needles) == [], \
        "the matcher flags a standards citation, which is what SHOULD be here"

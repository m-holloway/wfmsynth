"""The benchmark gate must protect CI without crying wolf on it.

Peak memory is deterministic on one platform+python+numpy and NOT portable across them. The
first time CI actually ran this gate it failed on `op:reflect` -- peak 1.6 MB against a 1.0 MB
baseline, +50 %, where the baseline said exactly 2.00x the record and the Linux runner measured
3.05x. One whole extra record-sized temporary, from a different numpy. Nothing had regressed;
the gate was comparing a macOS measurement against a Linux one, and it blocked two dependabot
PRs doing it.

A tolerance wide enough to absorb 50 % is wide enough to miss a real doubling, so the fix is to
compare like with like: gate memory only within one environment, and always gate the SCALING
RATIO, which divides the machine out and is portable by construction. That split is what these
tests pin -- including the half that must keep failing.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("bench", ROOT / "tools" / "benchmark.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

HERE = bench.environment()
OTHER = {"platform": "notaplatform", "python": "0.0", "numpy": "0.0.0"}

SIZES = (65536, 131072)


def _case(peak_a, peak_b, scale=1.9, t=0.001):
    return {"65536": {"peak_mb": peak_a, "time_s": t},
            "131072": {"peak_mb": peak_b, "time_s": t}, "scale": scale}


def _base(peak_a, peak_b, t=0.001):
    return {"65536": {"peak_mb": peak_a, "time_s": t},
            "131072": {"peak_mb": peak_b, "time_s": t}}


def test_memory_is_gated_when_the_environment_matches():
    bad = bench.check({"op:x": _case(10.0, 20.0)}, {"op:x": _base(1.0, 2.0)},
                      SIZES, base_env=HERE)
    assert any("peak memory" in b for b in bad), "a 10x memory regression went unreported"


def test_memory_is_NOT_gated_across_environments():
    """The false positive that blocked CI. It must still be printed -- silence would hide a
    real regression -- but it may not fail the build."""
    bad = bench.check({"op:x": _case(10.0, 20.0)}, {"op:x": _base(1.0, 2.0)},
                      SIZES, base_env=OTHER)
    assert not [b for b in bad if "peak memory" in b]


def test_complexity_is_gated_EVEN_across_environments():
    """The half that must never degrade. t(2n)/t(n) divides the machine out, so an O(n log n)
    stage turning O(n**2) shows up identically everywhere -- this is what actually protects CI
    once memory stops being comparable."""
    bad = bench.check({"op:x": _case(1.0, 2.0, scale=3.9)}, {"op:x": _base(1.0, 2.0)},
                      SIZES, base_env=OTHER)
    assert any("COMPLEXITY" in b for b in bad)


def test_a_missing_environment_is_treated_as_not_comparable():
    """A baseline predating the environment field must not be assumed to match."""
    bad = bench.check({"op:x": _case(10.0, 20.0)}, {"op:x": _base(1.0, 2.0)},
                      SIZES, base_env=None)
    assert not [b for b in bad if "peak memory" in b]


def test_numpy_patch_version_does_not_break_comparability():
    """numpy moves often and rarely moves allocations; platform and python minor are what
    decide. Requiring an exact numpy match would silently disable the memory gate for the
    maintainer the first time they upgraded."""
    other_numpy = dict(HERE, numpy="0.0.0")
    bad = bench.check({"op:x": _case(10.0, 20.0)}, {"op:x": _base(1.0, 2.0)},
                      SIZES, base_env=other_numpy)
    assert any("peak memory" in b for b in bad)


def test_the_committed_baseline_records_its_environment():
    """A baseline that does not say where it came from cannot be compared safely -- which is
    exactly how this defect shipped."""
    d = json.loads((ROOT / "tools" / "benchmark_baseline.json").read_text())
    env = d.get("environment")
    assert env, "the committed baseline has no environment stanza"
    assert set(env) >= {"platform", "python", "numpy"}, env

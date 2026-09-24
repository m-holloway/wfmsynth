"""Every shipped example actually runs.

`examples/` is the documented learning path -- `examples/README.md` walks a newcomer through it
in order -- so an example that raises is worse than no example: it is the first thing someone
tries, and it tells them the library is broken.

This caught a real one. `confounder_sweep.py` passed `ws.eye_height` to `hold_constant` as a
bare callable, and `eye_height` had since grown a required `levels` argument, so the example
died on the library's OWN most-documented gotcha ("eye_height requires levels and has no
default"). Nothing noticed, because nothing ran the examples.

Each one runs in a SUBPROCESS, which is the same path a reader takes. Running them in-process
would share imports and module state between them and stop testing what the README promises.
The whole file costs about nine seconds.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

# Demos: run with no arguments, print something, exit 0.
DEMOS = [
    "analog_instrument.py",
    "clock_recovery.py",
    "confounder_sweep.py",
    "events.py",
    "ground_truth.py",
    "non_integer_sps.py",
    "provenance.py",
    "quickstart.py",
    "realistic_scenario.py",
    "sim_to_real.py",
    "touchstone_channel.py",
    "two_rate_acquisition.py",
]

# Command-line TOOLS, not demos: they take arguments, so running them bare is a usage error
# rather than a failure. `--help` still exercises the import and the argument parser, which is
# where this kind of file rots -- a renamed symbol at import time fails exactly the same way.
CLI_TOOLS = [
    "fill_zarr.py",
    "replay.py",
]


def _run(name, args=()):
    return subprocess.run([sys.executable, str(EXAMPLES / name), *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=300,
                          env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin",
                               "HOME": str(Path.home())})


def test_every_example_is_classified():
    """The property that keeps this file honest as the project grows: a NEW example cannot be
    silently left out of the check, because the two lists must cover the directory exactly."""
    on_disk = {p.name for p in EXAMPLES.glob("*.py")}
    listed = set(DEMOS) | set(CLI_TOOLS)
    assert on_disk == listed, (
        f"examples/ and this file disagree.\n"
        f"  not covered by a test: {sorted(on_disk - listed)}\n"
        f"  listed but missing:    {sorted(listed - on_disk)}")


@pytest.mark.parametrize("name", DEMOS)
def test_the_demo_runs_clean(name):
    r = _run(name)
    assert r.returncode == 0, (
        f"{name} exited {r.returncode}\n--- stderr ---\n{r.stderr[-2000:]}")
    assert r.stdout.strip(), f"{name} ran but printed nothing -- a demo should show its result"


# Examples that still trip the library's own `k >= 8` sizing rule, with the MEASURED debt.
#
# All of them take the default `tr_frac=0.15` -- a fast, realistic transmitter edge -- on a grid
# of 2.3 to 10 samples/UI. Those two facts cannot both hold: resolving a 15 %-of-UI edge with 8
# samples needs ~53 samples/UI. So the rule and the default are not in tension with each other,
# they jointly demand a high sample rate, and these examples chose runtime instead.
#
# That is real debt and it is written down rather than filtered away, because the library tells
# users that below k = 8 a measurement is measuring the grid -- and `ground_truth.py` and
# `sim_to_real.py` measure things. The cost column is what fixing each one would take at
# tr_frac=0.5; the two cheap ones are the place to start.
#
# NOTHING MAY BE ADDED HERE without a measured k and a cost. A new example must satisfy the
# rule; this list is for the ones that predate the gate.
UNDERSAMPLED = {
    "confounder_sweep.py":     "k=0.60 at 4.0 samples/UI; 4.0x fs to fix",
    "events.py":               "k=1.50 at 10.0 samples/UI; 1.6x fs to fix -- cheapest",
    "ground_truth.py":         "k=0.60 at 4.0 samples/UI; 4.0x fs to fix (and it MEASURES)",
    "provenance.py":           "k=0.34 at 2.3 samples/UI; 7.0x fs to fix -- worst",
    "realistic_scenario.py":   "k=0.60 at 4.0 samples/UI; 4.0x fs to fix",
    "sim_to_real.py":          "k=0.60 at 4.0 samples/UI; 4.0x fs to fix (and it MEASURES)",
    "touchstone_channel.py":   "k=0.60 at 4.0 samples/UI; 4.0x fs to fix",
    "two_rate_acquisition.py": "k=1.20 at 8.0 samples/UI; 2.0x fs to fix -- cheap",
}


@pytest.mark.parametrize("name", [d for d in DEMOS if d not in UNDERSAMPLED])
def test_the_demo_does_not_warn(name):
    """Exiting 0 is not the bar. An example that WARNS is telling a newcomer, in the project's
    own voice, that the project's own code is set up wrong -- and it is the first code they run.

    This caught the flagship README example and `quickstart.py` both breaking the library's
    headline `k >= 8` rule: the README asked for 0.6 samples across an edge and was clamped to
    2.0, and quickstart sat at exactly 2.0, silent only because it landed precisely on the
    floor. Both now satisfy the rule and say why, so the first example anyone reads TEACHES it.

    pyproject's `filterwarnings` silences the clamp warning under pytest, deliberately, because
    many fast tests use a coarse grid on purpose. It does not reach these subprocesses -- which
    is right, since a reader running the example gets the unfiltered output."""
    r = _run(name)
    noisy = [ln for ln in r.stderr.splitlines()
             if "Warning:" in ln and "DeprecationWarning" not in ln]
    assert not noisy, (
        f"{name} warns when a reader runs it:\n  " + "\n  ".join(noisy[:5]))


@pytest.mark.parametrize("name", sorted(UNDERSAMPLED))
def test_the_undersampled_list_is_still_accurate(name):
    """The debt list has to stay honest in BOTH directions. An entry that has since been fixed
    must be removed, or the list becomes a place where resolved problems go to look unresolved
    -- and then nobody trusts it enough to work through it."""
    r = _run(name)
    assert any("clamped" in ln for ln in r.stderr.splitlines()), (
        f"{name} no longer warns -- remove it from UNDERSAMPLED ({UNDERSAMPLED[name]})")


@pytest.mark.parametrize("name", CLI_TOOLS)
def test_the_cli_tool_imports_and_describes_itself(name):
    """`--help` must work WITHOUT the example's optional dependencies -- someone deciding
    whether to install zarr should be able to read what the tool does first."""
    r = _run(name, ["--help"])
    assert r.returncode == 0, (
        f"{name} --help exited {r.returncode}\n--- stderr ---\n{r.stderr[-2000:]}")
    assert "usage:" in r.stdout.lower()


def test_a_missing_optional_dependency_is_explained_not_traced():
    """An optional dependency that is absent should produce advice, not a traceback. The
    traceback names `import zarr` on some line of a file the reader did not write."""
    zarr = pytest.importorskip  # noqa: F841  (documenting intent; we want the ABSENT case)
    try:
        import zarr as _z  # noqa: F401
        pytest.skip("zarr is installed, so the missing-dependency path cannot be exercised")
    except ImportError:
        pass
    r = _run("fill_zarr.py", ["seed.zarr", "recipes.json"])
    assert "pip install zarr" in (r.stdout + r.stderr)
    assert "Traceback" not in r.stderr

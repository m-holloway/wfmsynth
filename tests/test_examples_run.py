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

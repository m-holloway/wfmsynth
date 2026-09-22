"""GitHub #54: `__version__` and `pyproject.toml` were two independently-stated strings and had
already drifted (0.39.1 vs 0.40.0), so `Signal.recipe()` stamped provenance with a version that
did not match the installed package, and `importlib.metadata.version("wfmsynth")` disagreed with
`wfmsynth.__version__` from the same install. `__version__` must now be DERIVED from
`pyproject.toml`, not a second literal that can drift from it.
"""
from __future__ import annotations

import re
from pathlib import Path

import wfmsynth as ws

REPO = Path(__file__).resolve().parent.parent


def _pyproject_version():
    text = (REPO / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert m, "pyproject.toml has no top-level version = \"...\" line"
    return m.group(1)


def test_dunder_version_matches_pyproject_toml():
    assert ws.__version__ == _pyproject_version()


def test_dunder_version_is_not_a_second_hardcoded_literal():
    """The fix has to be structural, not a one-time sync: `__version__` may not appear as a
    literal string assignment in wfmsynth/__init__.py's source at all -- it must be computed."""
    src = (REPO / "wfmsynth" / "__init__.py").read_text()
    assert not re.search(r'^__version__\s*=\s*"[0-9]', src, re.M), (
        "__version__ is assigned a literal version string again; it must be derived "
        "(e.g. from importlib.metadata or by reading pyproject.toml), see GitHub #54")


def test_recipe_stamps_the_same_version_the_package_reports():
    g = ws.Grid(fs=1e9, n=256)
    sig = ws.Signal(seed=1, grid=g).carrier("sine", f_hz=1e6)
    assert sig.recipe()["wfmsynth_version"] == ws.__version__ == _pyproject_version()

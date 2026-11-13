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


_STALE_INSTALL = (
    "\n\n`__version__` should follow the source tree's pyproject.toml (GitHub #101); if it "
    "does not, this file is not being imported from the checkout.")


def test_dunder_version_matches_pyproject_toml():
    assert ws.__version__ == _pyproject_version(), (
        f"__version__ is {ws.__version__!r} but pyproject.toml says "
        f"{_pyproject_version()!r}." + _STALE_INSTALL)


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
    assert sig.recipe()["wfmsynth_version"] == ws.__version__ == _pyproject_version(), (
        f"recipe stamps {sig.recipe()['wfmsynth_version']!r}, __version__ is "
        f"{ws.__version__!r}, pyproject says {_pyproject_version()!r}." + _STALE_INSTALL)


def _fake_tree(tmp_path, pyproject):
    pkg = tmp_path / "wfmsynth"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    if pyproject is not None:
        (tmp_path / "pyproject.toml").write_text(pyproject)
    return str(pkg / "__init__.py")


def test_source_checkout_prefers_the_file_over_stale_metadata(tmp_path, monkeypatch):
    """GitHub #101: an editable install's metadata is frozen at install time."""
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: "0.0.1-stale")
    f = _fake_tree(tmp_path, '[project]\nname = "wfmsynth"\nversion = "9.8.7"\n')
    assert ws._read_version(f) == "9.8.7"


def test_no_pyproject_uses_installed_metadata(tmp_path, monkeypatch):
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: "1.2.3")
    assert ws._read_version(_fake_tree(tmp_path, None)) == "1.2.3"


def test_a_host_projects_pyproject_is_not_ours(tmp_path, monkeypatch):
    """A vendored copy must not pick up the host project's version."""
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: "1.2.3")
    f = _fake_tree(tmp_path, '[project]\nname = "hostapp"\nversion = "5.5.5"\n')
    assert ws._read_version(f) == "1.2.3"

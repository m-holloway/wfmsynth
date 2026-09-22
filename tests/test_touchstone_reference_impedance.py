"""GitHub #58: `read_touchstone` ignored the option line's reference impedance (`# ... R <z0>`),
and `write_touchstone` always wrote `R 50`, so a file measured at a different system impedance
(75 Ohm video, 100 Ohm differential, ...) was silently treated as if it were 50 Ohm. `read_touchstone`
now records the file's own z0; `write_touchstone` can state one; and `touchstone_channel(...,
z0=...)` renormalizes to a stated system impedance when the file's own differs from it.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth import sparam as SP
from wfmsynth.grid import Grid


def _write_2port(tmp_path, z0, name="ch.s2p", n=201, fmax=40e9):
    f = np.linspace(1e6, fmax, n)
    s21 = 10 ** (-(0.02 * np.sqrt(f / 1e9) + 0.01 * f / 1e9) / 20) * np.exp(-1j * 2 * np.pi * f * 5e-10)
    S = np.zeros((n, 2, 2), complex)
    S[:, 1, 0] = s21
    S[:, 0, 1] = s21
    path = str(tmp_path / name)
    SP.write_touchstone(path, f, S, z0=z0)
    return path, f, S


def test_read_touchstone_reports_the_files_own_reference_impedance(tmp_path):
    path, _f, _S = _write_2port(tmp_path, z0=75.0)
    freqs, S = SP.read_touchstone(path)          # still unpacks as exactly (freqs, S)
    result = SP.read_touchstone(path)
    assert result.z0 == 75.0


def test_default_reference_impedance_is_50_when_unstated(tmp_path):
    path, f, S = _write_2port(tmp_path, z0=75.0)
    # hand-write a file with no R token at all, the plain Touchstone 1.0 minimum
    bare = str(tmp_path / "bare.s2p")
    with open(path) as fh:
        text = fh.read()
    header, rest = text.split("\n", 1)
    with open(bare, "w") as fh:
        fh.write("# GHZ S RI\n" + rest)
    result = SP.read_touchstone(bare)
    assert result.z0 == 50.0


def test_write_touchstone_round_trips_a_stated_reference_impedance(tmp_path):
    path, f, S = _write_2port(tmp_path, z0=100.0)
    freqs, S2 = SP.read_touchstone(path)
    assert np.allclose(freqs, f, rtol=1e-6)
    assert np.allclose(S2, S, atol=1e-6)
    assert SP.read_touchstone(path).z0 == 100.0


def test_default_r_50_rendering_is_unchanged(tmp_path):
    """The compatibility bar the issue states explicitly: a plain R 50 file renders identically
    whether or not z0= is passed, and passing the file's OWN z0 is a no-op."""
    path, _f, _S = _write_2port(tmp_path, z0=50.0)
    g = Grid(fs=80e9, n=2048)
    x = np.zeros(2048); x[64:96] = 1.0
    a = SP.touchstone_channel(x, path, grid=g)
    b = SP.touchstone_channel(x, path, grid=g, z0=50.0)
    assert np.array_equal(a, b)


def test_renormalizing_to_a_different_z0_actually_changes_the_response(tmp_path):
    path, _f, _S = _write_2port(tmp_path, z0=50.0)
    g = Grid(fs=80e9, n=2048)
    x = np.zeros(2048); x[64:96] = 1.0
    native = SP.touchstone_channel(x, path, grid=g)
    renorm = SP.touchstone_channel(x, path, grid=g, z0=100.0)
    assert not np.allclose(native, renorm)


def test_renormalization_round_trips_back_to_the_original_matrix(tmp_path):
    """The general correctness gate, independent of the exact formula: renormalizing 50 -> 100
    -> 50 must recover the original S matrix, for a matched (S11=S22=0) loss-only 2-port."""
    path, f, S = _write_2port(tmp_path, z0=50.0)
    freqs, S_native = SP.read_touchstone(path)
    up = SP.renormalize_s(S_native, 50.0, 100.0)
    back = SP.renormalize_s(up, 100.0, 50.0)
    assert np.allclose(back, S_native, atol=1e-9)
    assert not np.allclose(up, S_native, atol=1e-6)


def test_renormalize_s_is_the_identity_when_z0_is_unchanged():
    S = np.random.default_rng(0).standard_normal((5, 2, 2)) * 0.1
    assert np.array_equal(SP.renormalize_s(S, 50.0, 50.0), S)

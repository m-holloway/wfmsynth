"""GitHub #63: `touchstone_channel(..., z0=...)` can renormalize a Touchstone network to a
stated system impedance, but `Signal.sparam(...)` -- the composable, recipe-recording API --
dropped `z0` on the floor instead of forwarding it, the same class of bug #57 already fixed for
`ports`. Existing (z0-omitted) behavior must stay exactly as it was.
"""
from __future__ import annotations

import numpy as np

from wfmsynth import sparam as SP
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid


def _write_2port(tmp_path, z0=50.0, n=201, fmax=40e9):
    f = np.linspace(1e6, fmax, n)
    s21 = 10 ** (-(0.02 * np.sqrt(f / 1e9) + 0.01 * f / 1e9) / 20) * np.exp(-1j * 2 * np.pi * f * 5e-10)
    S = np.zeros((n, 2, 2), complex)
    S[:, 1, 0] = s21
    S[:, 0, 1] = s21
    path = str(tmp_path / "ch.s2p")
    SP.write_touchstone(path, f, S, z0=z0)
    return path


def test_signal_sparam_forwards_z0_to_touchstone_channel(tmp_path):
    path = _write_2port(tmp_path, z0=50.0)
    g = Grid(fs=80e9, n=2048)
    x = (Signal(seed=1, grid=g).carrier("nrz", n_ui=64, causal=True)
         .sparam(path=path, z0=100.0)).waveform()
    direct = SP.touchstone_channel(
        (Signal(seed=1, grid=g).carrier("nrz", n_ui=64, causal=True)).waveform(),
        path, grid=g, z0=100.0)
    assert np.array_equal(x, direct)


def test_signal_sparam_z0_actually_changes_the_rendered_waveform(tmp_path):
    path = _write_2port(tmp_path, z0=50.0)
    g = Grid(fs=80e9, n=2048)
    native = (Signal(seed=1, grid=g).carrier("nrz", n_ui=64, causal=True)
              .sparam(path=path)).waveform()
    renorm = (Signal(seed=1, grid=g).carrier("nrz", n_ui=64, causal=True)
              .sparam(path=path, z0=100.0)).waveform()
    assert not np.array_equal(native, renorm)


def test_signal_sparam_z0_round_trips_through_the_recipe(tmp_path):
    path = _write_2port(tmp_path, z0=50.0)
    g = Grid(fs=80e9, n=2048)
    sig = (Signal(seed=1, grid=g).carrier("nrz", n_ui=64, causal=True)
           .sparam(path=path, z0=75.0))
    x = sig.waveform()
    x2 = Signal.from_recipe(sig.recipe()).waveform()
    assert np.array_equal(x, x2)
    assert sig.recipe()["ops"][-1]["z0"] == 75.0


def test_signal_sparam_without_z0_is_unchanged():
    """The compatibility bar the issue states explicitly: omitting z0 behaves exactly as
    before -- no z0 key even reaches touchstone_channel's call."""
    g = Grid(fs=80e9, n=2048)
    sig = Signal(seed=1, grid=g).carrier("nrz", n_ui=64, causal=True).sparam(
        freqs=np.linspace(1e6, 40e9, 201),
        s21=10 ** (-(0.02 * np.sqrt(np.linspace(1e6, 40e9, 201) / 1e9)) / 20))
    x = sig.waveform()
    assert np.all(np.isfinite(x))
    assert "z0" not in sig.recipe()["ops"][-1]

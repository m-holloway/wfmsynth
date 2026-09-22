"""GitHub #57: `Signal.sparam`'s `_op_sparam` wrapped `ports` in `tuple(...)` before forwarding
it to `touchstone_channel`. That destroys a mixed-mode pairing given as a STRING --
`tuple("13_24")` is `('1', '3', '_', '2', '4')`, not the pairing -- so a mixed-mode chain built
through `Signal` silently fell back to some other (wrong, or erroring) interpretation of
`ports` instead of the one `touchstone_channel` documents and `SP.touchstone_channel` itself
already gets right when called directly.
"""
from __future__ import annotations

import numpy as np

from wfmsynth import sparam as SP
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid
from test_sparam_mixed_mode import modal_4port, se_from_modal, _grid_freqs


def _four_port_file(tmp_path, pairing="13_24", name="pair.s4p"):
    f = _grid_freqs(401)
    M, _td, _r = modal_4port(f, 1.5e-9, 1.542e-9, 9.0, 7.5)
    S = se_from_modal(M, pairing)
    path = str(tmp_path / name)
    SP.write_touchstone(path, f, S)
    return path


def test_signal_sparam_string_pairing_reaches_touchstone_channel(tmp_path):
    path = _four_port_file(tmp_path)
    g = Grid(fs=80e9, n=2048)
    x = (Signal(seed=1, grid=g).carrier("nrz", n_ui=64, causal=True)
         .sparam(path=path, ports="13_24")).waveform()
    direct = SP.touchstone_channel(
        (Signal(seed=1, grid=g).carrier("nrz", n_ui=64, causal=True)).waveform(),
        path, grid=g, ports="13_24")
    assert np.array_equal(x, direct)


def test_signal_sparam_tuple_of_tuples_pairing_reaches_touchstone_channel(tmp_path):
    path = _four_port_file(tmp_path)
    g = Grid(fs=80e9, n=2048)
    x = (Signal(seed=1, grid=g).carrier("nrz", n_ui=64, causal=True)
         .sparam(path=path, ports=((1, 3), (2, 4)))).waveform()
    direct = SP.touchstone_channel(
        (Signal(seed=1, grid=g).carrier("nrz", n_ui=64, causal=True)).waveform(),
        path, grid=g, ports=((1, 3), (2, 4)))
    assert np.array_equal(x, direct)


def test_signal_sparam_default_ports_still_works(tmp_path):
    """The plain single-ended default (2, 1) must round-trip through the recipe unchanged --
    the fix must not disturb the ordinary case, only the pairing forms it was breaking."""
    path = _four_port_file(tmp_path, pairing="12_34", name="se.s4p")
    g = Grid(fs=80e9, n=2048)
    sig = (Signal(seed=1, grid=g).carrier("nrz", n_ui=64, causal=True)
           .sparam(path=path, ports=(3, 1), mode="se"))
    x = sig.waveform()
    x2 = Signal.from_recipe(sig.recipe()).waveform()
    assert np.array_equal(x, x2)
    assert np.any(np.abs(x) > 1e-6)

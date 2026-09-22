"""GitHub #62: Touchstone is one network; MDIF holds the same S-matrix over one or more swept
independent variables. `read_mdif` reads exactly one well-documented dialect (see the module
comment above `read_mdif` in wfmsynth/sparam.py) and returns an `MdifSweep` -- `.freqs`, `.S`,
`.axes`, plus `.select(**kwargs)` to pick one sweep point for reuse with `sparam_channel`.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth import sparam as SP
from wfmsynth.grid import Grid


def _write_mdif(path, stub_mms, z0=50.0, n=201, fmax=40e9):
    """Two-port sweep over `stub_mms`, each block's S21 magnitude scaled by (1 - stub_mm) so the
    blocks are numerically distinguishable."""
    f = np.linspace(1e6, fmax, n)
    lines = []
    for stub_mm in stub_mms:
        s21 = (1.0 - 0.5 * stub_mm) * np.exp(-1j * 2 * np.pi * f * 5e-10)
        s11 = 0.01 * stub_mm * np.ones_like(f)
        lines.append("BEGIN ACDATA")
        lines.append(f"VAR stub_mm(real) = {stub_mm}")
        lines.append(f"VAR z0(real) = {z0}")
        lines.append("%Freq(Hz) S[1,1] S[2,1] S[1,2] S[2,2]")
        for k in range(n):
            row = [f[k], s11[k], 0.0, s21[k].real, s21[k].imag, s21[k].real, s21[k].imag, s11[k], 0.0]
            lines.append(" ".join(f"{v:.9g}" for v in row))
        lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return f, stub_mms


def test_read_mdif_returns_shared_freqs_full_s_and_axes(tmp_path):
    path = str(tmp_path / "via.mdf")
    f, stub_mms = _write_mdif(path, [0.2, 0.4, 0.6])
    sweep = SP.read_mdif(path)
    assert np.allclose(sweep.freqs, f, rtol=1e-6)
    assert sweep.S.shape == (3, len(f), 2, 2)
    assert np.allclose(sweep.axes["stub_mm"], stub_mms)
    assert np.allclose(sweep.axes["z0"], [50.0, 50.0, 50.0])


def test_select_picks_the_one_matching_block(tmp_path):
    path = str(tmp_path / "via.mdf")
    f, _stub_mms = _write_mdif(path, [0.2, 0.4, 0.6])
    sweep = SP.read_mdif(path)
    f_sel, S_sel = sweep.select(stub_mm=0.4, z0=50)
    assert np.allclose(f_sel, f, rtol=1e-6)
    want_s21 = (1.0 - 0.5 * 0.4) * np.exp(-1j * 2 * np.pi * f * 5e-10)
    assert np.allclose(S_sel[:, 1, 0], want_s21, atol=1e-6)


def test_select_with_no_match_raises(tmp_path):
    path = str(tmp_path / "via.mdf")
    _write_mdif(path, [0.2, 0.4])
    sweep = SP.read_mdif(path)
    with pytest.raises(ValueError, match="no block matches"):
        sweep.select(stub_mm=0.99)


def test_select_result_feeds_sparam_channel_directly(tmp_path):
    path = str(tmp_path / "via.mdf")
    _write_mdif(path, [0.2, 0.4])
    sweep = SP.read_mdif(path)
    f_sel, S_sel = sweep.select(stub_mm=0.2)
    g = Grid(fs=80e9, n=2048)
    x = np.zeros(2048)
    x[64:96] = 1.0
    y = SP.sparam_channel(x, f_sel, S_sel[:, 1, 0], grid=g)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_n_ports_can_be_overridden_when_not_a_perfect_square_column_count(tmp_path):
    """A 1-port file: 1 S-parameter column, which IS a perfect square (1x1) -- exercise the
    override path by asserting it is honored even though it is also inferable."""
    f = np.linspace(1e6, 10e9, 21)
    s11 = 0.2 * np.exp(-1j * 2 * np.pi * f * 1e-10)
    lines = ["BEGIN ACDATA", "VAR z0(real) = 50.0", "%Freq(Hz) S[1,1]"]
    for k in range(len(f)):
        lines.append(f"{f[k]:.9g} {s11[k].real:.9g} {s11[k].imag:.9g}")
    lines.append("END")
    path = str(f"{tmp_path}/one_port.mdf")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    sweep = SP.read_mdif(path, n_ports=1)
    assert sweep.S.shape == (1, len(f), 1, 1)
    assert np.allclose(sweep.S[0, :, 0, 0], s11, atol=1e-6)


def test_frequency_unit_suffix_on_the_header_is_honored(tmp_path):
    f_ghz = np.linspace(0.001, 40.0, 51)
    s21 = 0.9 * np.ones_like(f_ghz, dtype=complex)
    lines = ["BEGIN ACDATA", "VAR z0(real) = 50.0",
             "%Freq(GHz) S[1,1] S[2,1] S[1,2] S[2,2]"]
    for k in range(len(f_ghz)):
        row = [f_ghz[k], 0.0, 0.0, s21[k].real, s21[k].imag, s21[k].real, s21[k].imag, 0.0, 0.0]
        lines.append(" ".join(f"{v:.9g}" for v in row))
    lines.append("END")
    path = str(tmp_path / "ghz.mdf")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    sweep = SP.read_mdif(path)
    assert np.allclose(sweep.freqs, f_ghz * 1e9, rtol=1e-6)


def test_unrecognised_column_raises_rather_than_guessing(tmp_path):
    path = str(tmp_path / "bad.mdf")
    with open(path, "w") as fh:
        fh.write("BEGIN ACDATA\nVAR z0(real) = 50.0\n%Freq(Hz) MAG[S11] ANG[S11]\n"
                  "1.0 0.5 0.0\nEND\n")
    with pytest.raises(ValueError, match="unrecognised column"):
        SP.read_mdif(path)


def test_mismatched_var_names_across_blocks_raises(tmp_path):
    path = str(tmp_path / "mismatch.mdf")
    with open(path, "w") as fh:
        fh.write(
            "BEGIN ACDATA\nVAR stub_mm(real) = 0.2\n%Freq(Hz) S[1,1] S[2,1] S[1,2] S[2,2]\n"
            "1.0e9 0.0 0.0 0.9 0.0 0.9 0.0 0.0 0.0\nEND\n"
            "BEGIN ACDATA\nVAR z0(real) = 50.0\n%Freq(Hz) S[1,1] S[2,1] S[1,2] S[2,2]\n"
            "1.0e9 0.0 0.0 0.9 0.0 0.9 0.0 0.0 0.0\nEND\n")
    with pytest.raises(ValueError, match="different VAR names"):
        SP.read_mdif(path)


def test_mismatched_frequency_axes_across_blocks_raises(tmp_path):
    path = str(tmp_path / "mismatch_f.mdf")
    with open(path, "w") as fh:
        fh.write(
            "BEGIN ACDATA\nVAR stub_mm(real) = 0.2\n%Freq(Hz) S[1,1] S[2,1] S[1,2] S[2,2]\n"
            "1.0e9 0.0 0.0 0.9 0.0 0.9 0.0 0.0 0.0\nEND\n"
            "BEGIN ACDATA\nVAR stub_mm(real) = 0.4\n%Freq(Hz) S[1,1] S[2,1] S[1,2] S[2,2]\n"
            "2.0e9 0.0 0.0 0.9 0.0 0.9 0.0 0.0 0.0\nEND\n")
    with pytest.raises(ValueError, match="share one frequency axis"):
        SP.read_mdif(path)


def test_a_data_row_of_the_wrong_length_raises(tmp_path):
    path = str(tmp_path / "short_row.mdf")
    with open(path, "w") as fh:
        fh.write("BEGIN ACDATA\nVAR z0(real) = 50.0\n%Freq(Hz) S[1,1] S[2,1] S[1,2] S[2,2]\n"
                  "1.0e9 0.0 0.0 0.9 0.0\nEND\n")
    with pytest.raises(ValueError, match="data row has"):
        SP.read_mdif(path)


def test_no_begin_end_block_raises(tmp_path):
    path = str(tmp_path / "empty.mdf")
    with open(path, "w") as fh:
        fh.write("! just a comment, no data\n")
    with pytest.raises(ValueError, match="no BEGIN/END"):
        SP.read_mdif(path)

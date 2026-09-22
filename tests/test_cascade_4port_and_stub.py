"""GitHub #59 and #60: two new cascade section kinds.

#59 -- a measured 4-port block (a via, a connector) sitting between two DIFFERENTIAL `line`
sections, instead of dropping to a single-ended 2-port. `measured(..., left=, right=)` converts
the file's full N-port S-matrix to its mixed-mode SDD 2-port using the existing `se2mm`
machinery, so a chain of 4-port blocks composes exactly like a chain of 2-port ones.

#60 -- an open or short stub as its own section: a length, a characteristic impedance, and an
end condition, using the exact lossless transmission-line stub formula (not the phenomenological
band-pass shape `physics.resonant_reflection` already provides for a fitted resonance).
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth import sparam as SP
from test_sparam_mixed_mode import modal_4port, se_from_modal, _grid_freqs


# --------------------------------------------------------------------------------- #59
def test_measured_left_right_matches_se2mm_directly(tmp_path):
    f = _grid_freqs(801)
    M, _td, _r = modal_4port(f, 1.5e-9, 1.542e-9, 9.0, 7.5)
    S = se_from_modal(M, "13_24")
    path = str(tmp_path / "via.s4p")
    SP.write_touchstone(path, f, S)

    section = SP._section({"file": {"path": path, "left": (1, 3), "right": (2, 4)}}, f)
    mm = SP.se2mm(S, ((1, 3), (2, 4)))
    assert np.allclose(section.s11, mm[:, 0, 0], atol=1e-9)
    assert np.allclose(section.s12, mm[:, 0, 1], atol=1e-9)
    assert np.allclose(section.s21, mm[:, 1, 0], atol=1e-9)
    assert np.allclose(section.s22, mm[:, 1, 1], atol=1e-9)


def test_measured_left_right_needs_both(tmp_path):
    f = _grid_freqs(101)
    M, _td, _r = modal_4port(f, 1.5e-9, 1.542e-9, 9.0, 7.5)
    S = se_from_modal(M, "13_24")
    path = str(tmp_path / "via.s4p")
    SP.write_touchstone(path, f, S)
    with pytest.raises(ValueError, match="left.*right|right.*left"):
        SP.measured(f, path=path, left=(1, 3))


def test_a_4port_section_cascades_between_two_lines(tmp_path):
    """End-to-end: line -> 4-port via -> line, cross-checked against cascading the SAME
    sections built by hand from se2mm + cascade_2port."""
    f = _grid_freqs(801)
    M, _td, _r = modal_4port(f, 1.5e-9, 1.542e-9, 9.0, 7.5)
    S = se_from_modal(M, "13_24")
    path = str(tmp_path / "via.s4p")
    SP.write_touchstone(path, f, S)

    path_spec = [{"line": {"td_ps": 200.0}},
                 {"file": {"path": path, "left": (1, 3), "right": (2, 4)}},
                 {"line": {"td_ps": 150.0}}]
    got = SP.cascade(path_spec, f)

    l1 = SP.line(f, td_ps=200.0)
    mm = SP.se2mm(S, ((1, 3), (2, 4)))
    via = SP.TwoPort(f, mm[:, 0, 0], mm[:, 0, 1], mm[:, 1, 0], mm[:, 1, 1])
    l2 = SP.line(f, td_ps=150.0)
    want = SP.cascade_2port(SP.cascade_2port(l1, via), l2)

    assert np.allclose(got.s21, want.s21, atol=1e-9)
    assert np.allclose(got.s11, want.s11, atol=1e-9)


# --------------------------------------------------------------------------------- #60
def test_open_stub_does_nothing_at_dc():
    f = _grid_freqs(401, fmax=20e9)
    f[0] = 0.0
    sec = SP.stub(f, length_in=0.08, z0=50.0, end="open")
    assert abs(sec.s21[0] - 1.0) < 1e-6
    assert abs(sec.s11[0]) < 1e-6


def test_short_stub_shorts_the_line_at_dc():
    f = _grid_freqs(401, fmax=20e9)
    f[0] = 0.0
    sec = SP.stub(f, length_in=0.08, z0=50.0, end="short")
    assert abs(sec.s21[0]) < 1e-6
    assert abs(sec.s11[0] + 1.0) < 1e-6


def test_open_stub_notches_the_through_response_at_quarter_wave():
    """A quarter-wavelength open stub looks like a short to the main line: full reflection,
    zero transmission, at the frequency where the electrical length is 90 degrees."""
    length_in = 0.5
    pp = SP.ps_per_inch(4.0)
    td_ps = length_in * pp
    f_notch = 1.0 / (4.0 * td_ps * 1e-12)          # quarter-wave: beta*L = pi/2
    f = np.linspace(1e6, 2 * f_notch, 4001)
    sec = SP.stub(f, length_in=length_in, z0=50.0, end="open")
    k = int(np.argmin(np.abs(f - f_notch)))
    assert abs(sec.s21[k]) < 0.02
    assert abs(sec.s11[k]) > 0.98


def test_stub_is_lossless_lossless_reciprocal_and_symmetric():
    f = _grid_freqs(501, fmax=20e9)
    f[0] = 1e6
    sec = SP.stub(f, length_in=0.15, z0=50.0, end="open")
    passivity = np.abs(sec.s11) ** 2 + np.abs(sec.s21) ** 2
    assert np.allclose(passivity, 1.0, atol=1e-6)
    assert np.array_equal(sec.s11, sec.s22)
    assert np.array_equal(sec.s12, sec.s21)


def test_stub_end_must_be_open_or_short():
    f = _grid_freqs(11, fmax=20e9)
    with pytest.raises(ValueError, match="open.*short|short.*open"):
        SP.stub(f, length_in=0.1, z0=50.0, end="bogus")


def test_stub_section_reaches_through_the_declarative_path(tmp_path):
    """Uses a plain DC-starting rfft axis, unlike the stub-only DC/quarter-wave tests above --
    `line(length_in=...)` has real loss and its default `causal=True` needs a uniform
    DC..Nyquist axis (`_is_rfft_axis`), which an `f[0] != 0` grid (used elsewhere in this file
    to dodge the stub's own tan(0) pole) violates."""
    f = _grid_freqs(801, fmax=20e9)
    path_spec = [{"line": {"length_in": 3.0}},
                 {"stub": {"length_in": 0.08, "z0": 50.0, "end": "open"}},
                 {"line": {"length_in": 3.0}}]
    got = SP.cascade(path_spec, f)
    direct = SP.cascade_2port(
        SP.cascade_2port(SP.line(f, length_in=3.0), SP.stub(f, length_in=0.08, z0=50.0, end="open")),
        SP.line(f, length_in=3.0))
    assert np.allclose(got.s21, direct.s21, atol=1e-9)

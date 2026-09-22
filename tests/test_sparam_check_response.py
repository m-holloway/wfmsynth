"""GitHub #61: applying a Touchstone file as H(f) had no check that the response is passive,
or that its impulse response is causal. `sparam.check_response` reports both and leaves the
samples alone -- the same idea as `sparam_channel`'s `band=refuse`: name the problem.
"""
from __future__ import annotations

import numpy as np

from wfmsynth import sparam as SP


def _freqs(n=2001, fmax=40e9):
    return np.linspace(1e6, fmax, n)


def test_a_passive_lossy_term_reports_passive_true():
    f = _freqs()
    s21 = 10 ** (-(0.02 * np.sqrt(f / 1e9) + 0.01 * f / 1e9) / 20) * np.exp(-1j * 2 * np.pi * f * 5e-10)
    info = SP.check_response(f, s21)
    assert info["passive"] is True
    assert info["max_singular_value"] <= 1.0 + 1e-9


def test_an_amplifying_term_reports_passive_false():
    f = _freqs()
    s21 = 1.5 * np.ones_like(f, dtype=complex)     # |S21| = 1.5 everywhere: not passive
    info = SP.check_response(f, s21)
    assert info["passive"] is False
    assert info["max_singular_value"] > 1.4


def test_a_causal_delayed_lowpass_has_near_zero_precursor_energy():
    f = _freqs()
    fc = 8e9
    delay_s = 5e-10
    s21 = (1.0 / (1.0 + 1j * f / fc)) * np.exp(-1j * 2 * np.pi * f * delay_s)
    info = SP.check_response(f, s21)
    assert info["precursor_frac"] < 0.02


def test_a_zero_phase_magnitude_only_term_is_flagged_noncausal():
    """A magnitude-only response (no phase at all) is the textbook non-causal construction.
    `tests/audit_causality.py`'s own calibration puts a zero-phase 4th-order Bessel around
    0.35; this single-pole magnitude shape measures ~0.11 -- an order of magnitude above the
    causal case, which is the property under test, not the exact figure."""
    f = _freqs()
    fc = 8e9
    causal = (1.0 / (1.0 + 1j * f / fc)) * np.exp(-1j * 2 * np.pi * f * 5e-10)
    s21 = (1.0 / (1.0 + (f / fc) ** 2)) ** 0.5      # |H|, zero phase
    info = SP.check_response(f, s21)
    info_causal = SP.check_response(f, causal)
    assert info["precursor_frac"] > 20 * info_causal["precursor_frac"]
    assert info["precursor_frac"] > 0.05


def test_check_response_does_not_alter_the_input():
    f = _freqs()
    s21 = 10 ** (-(0.02 * np.sqrt(f / 1e9)) / 20) * np.exp(-1j * 2 * np.pi * f * 3e-10)
    before = s21.copy()
    SP.check_response(f, s21)
    assert np.array_equal(s21, before)


def test_check_response_accepts_a_full_s_matrix_for_passivity():
    f = _freqs(n=501)
    n = len(f)
    S = np.zeros((n, 2, 2), complex)
    s21 = 10 ** (-(0.02 * np.sqrt(f / 1e9)) / 20)
    S[:, 1, 0] = s21
    S[:, 0, 1] = s21
    info = SP.check_response(f, S)
    assert info["passive"] is True

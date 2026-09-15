"""Phase 3, item 1: a clamped-exponential rail event -- the generic behavioural shape behind
load-dump / ESD / supply-glitch style transients (exponential toward a target, clamped at a
stated rail). Which standard's pulse NUMBERS apply is the caller's own registry
(docs/DATASET-METHODOLOGY.md's mechanisms-in/knowledge-out boundary) -- this only tests the
mechanism: amp, tau and the clamp, each against a known closed-form answer.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS = 1e9
G = Grid(fs=FS, n=20000)


def _place(**kw):
    sig = Signal(seed=1, grid=G).carrier("dc", level=0.0)
    return sig.events("clamped_exp", on="times", **kw).waveform()


def test_reaches_the_requested_amplitude_after_several_time_constants():
    tau_s, amp = 2e-9, 1.0
    y = _place(times=[5e-6], amp=amp, tau_s=tau_s)
    i0 = int(round(5e-6 * FS))
    i_settled = i0 + int(round(8 * tau_s * FS)) - 1     # last sample of the applied window
    assert y[i_settled] == pytest.approx(amp, abs=0.01)


def test_time_constant_matches_the_request():
    tau_s, amp = 3e-9, 1.0
    y = _place(times=[5e-6], amp=amp, tau_s=tau_s)
    i0 = int(round(5e-6 * FS))
    i_tau = i0 + int(round(tau_s * FS))
    assert y[i_tau] == pytest.approx(amp * (1.0 - np.exp(-1.0)), abs=0.02)


def test_clamp_caps_the_excursion_even_when_amp_asks_for_more():
    y = _place(times=[5e-6], amp=5.0, tau_s=2e-9, v_clamp=1.2, width_s=30e-9)
    i0 = int(round(5e-6 * FS))
    assert y[i0 : i0 + int(30e-9 * FS)].max() <= 1.2 + 1e-6


def test_no_clamp_reaches_the_full_requested_amplitude():
    y = _place(times=[5e-6], amp=1.0, tau_s=2e-9, width_s=30e-9)
    i0 = int(round(5e-6 * FS))
    assert y[i0 : i0 + int(30e-9 * FS)].max() == pytest.approx(1.0, abs=0.02)


def test_negative_direction_clamps_on_the_low_side():
    y = _place(times=[5e-6], amp=5.0, tau_s=2e-9, v_clamp=-0.8, polarity=-1.0, width_s=30e-9)
    i0 = int(round(5e-6 * FS))
    seg = y[i0 : i0 + int(30e-9 * FS)]
    assert seg.min() >= -0.8 - 1e-6
    assert seg.min() < -0.5


def test_the_record_is_flat_outside_the_event():
    y = _place(times=[10e-6], amp=1.0, tau_s=2e-9, width_s=20e-9)
    assert np.allclose(y[: int(9e-6 * FS)], 0.0, atol=1e-9)
    assert np.allclose(y[int(11e-6 * FS) :], 0.0, atol=1e-9)


def test_clamped_exp_is_a_registered_mechanism():
    from wfmsynth import events as EV
    assert "clamped_exp" in EV.MECHANISMS
    assert "clamped_exp" in EV._APPLY


def test_clamped_exp_composes_through_a_recipe_and_round_trips():
    sig = (Signal(seed=1, grid=G).carrier("dc", level=0.0)
           .events("clamped_exp", on="times", times=[5e-6], amp=1.0, tau_s=2e-9, v_clamp=0.9)
           .lossy(loss_db=3.0, loss_at_ghz=1e-3))
    x = sig.waveform()
    x2 = Signal.from_recipe(sig.recipe()).waveform()
    assert np.array_equal(x, x2)


@pytest.mark.parametrize("knob,base,alt", [
    ("amp", dict(tau_s=2e-9), 2.0),
    ("tau_s", dict(amp=1.0), 5e-9),
    ("v_clamp", dict(amp=5.0, tau_s=2e-9), 0.5),
    ("width_s", dict(amp=1.0, tau_s=2e-9), 5e-9),
])
def test_every_clamped_exp_knob_moves_the_output(knob, base, alt):
    a = _place(times=[5e-6], **base)
    b = _place(times=[5e-6], **{**base, knob: alt})
    assert not np.allclose(a, b, atol=1e-9), f"clamped_exp.{knob} did not move the output"

"""Where each one-shot source rests and where it goes (GitHub #103).

`step` and `pulse` are bipolar about `offset` (rest at offset - amp, go to offset + amp), like
`sine` and `square`; `exp` is one-sided (offset toward offset + amp). Pinned so a change to
either is a decision, not an accident.
"""
from __future__ import annotations

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import physics as P

G = ws.Grid(fs=2e6, n=6000)
KW = {"step": dict(t_step_s=1.5e-3, tr_s=50e-6),
      "pulse": dict(t_start_s=1e-3, width_s=0.25e-3, tr_s=50e-6),
      "exp": dict(t_start_s=1e-3, tau_s=1e-4)}


def render(kind, amp, offset):
    return ws.Signal(seed=1, grid=G).carrier(kind, amp=amp, offset=offset, **KW[kind]).waveform()


@pytest.mark.parametrize("amp,offset", [(-1.0, 0.0), (0.5, 2.0), (2.0, -1.0)])
def test_step_rests_at_offset_minus_amp_and_goes_to_offset_plus_amp(amp, offset):
    x = render("step", amp, offset)
    assert x[0] == pytest.approx(offset - amp, abs=1e-9)
    assert x[-1] == pytest.approx(offset + amp, abs=1e-9)


@pytest.mark.parametrize("amp,offset", [(-1.0, 0.0), (0.5, 2.0), (2.0, -1.0)])
def test_pulse_rests_at_offset_minus_amp_and_goes_to_offset_plus_amp(amp, offset):
    x = render("pulse", amp, offset)
    assert x[0] == pytest.approx(offset - amp, abs=1e-9)
    assert x[-1] == pytest.approx(offset - amp, abs=1e-9)
    mid = x[2250]                       # inside the pulse, past its edge
    assert mid == pytest.approx(offset + amp, abs=1e-3 * abs(amp))     # edge ring
    assert abs((mid - x[0])) == pytest.approx(2 * abs(amp), abs=2e-3 * abs(amp))    # excursion 2*amp


@pytest.mark.parametrize("amp,offset", [(-1.0, 0.0), (0.5, 2.0), (2.0, -1.0)])
def test_exp_rests_at_offset_and_heads_for_offset_plus_amp(amp, offset):
    x = render("exp", amp, offset)
    assert x[0] == offset
    assert x[-1] == pytest.approx(offset + amp, abs=1e-6)


def test_the_issues_example_as_documented():
    x = render("pulse", -1.0, 0.0)                     # the surprise: rests at +1, dips to -1
    assert x[0] == pytest.approx(1.0, abs=1e-9) and x.min() < -0.99
    y = render("pulse", -0.65, -0.65)                  # "rest at 0, dip to -1.3"
    assert y[0] == pytest.approx(0.0, abs=1e-9) and y.min() == pytest.approx(-1.3, abs=0.02)


@pytest.mark.parametrize("kind", ["step", "pulse"])
@pytest.mark.parametrize("rest,to", [(0.0, -1.3), (3.3, 2.0), (-1.0, 4.0)])
def test_levels_says_rest_and_target_without_arithmetic(kind, rest, to):
    x = ws.Signal(seed=1, grid=G).carrier(kind, **P.levels(rest, to), **KW[kind]).waveform()
    assert x[0] == pytest.approx(rest, abs=1e-9)
    far = x[-1] if kind == "step" else x[2250]
    assert far == pytest.approx(to, abs=1e-3 * abs(to - rest))


def test_levels_values_and_refusals():
    assert P.levels(0.0, -1.3) == {"amp": -0.65, "offset": -0.65}
    for bad in ((0.0, 0.0), (float("nan"), 1.0), (0.0, float("inf")), (True, 1.0), ("0", 1.0),
                (None, 1.0)):
        with pytest.raises(ValueError, match="levels"):
            P.levels(*bad)

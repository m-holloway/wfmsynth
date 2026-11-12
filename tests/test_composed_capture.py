"""examples/composed_capture.py finds what it says it finds.

The example uses the composition ops together -- sequence, use, gain(by=), delay(by=), combine,
ws.tol with a group, trigger(on=) -- and reads one conclusion off the record. That conclusion is
asserted here, on the example's own numbers, so the example cannot drift into printing a table
that no longer supports its text.

Resolution: the rms of n edges is known to about 1/sqrt(2n): 8 % for the preamble's 87 edges
and 4 % for the payload's 330. Every bound below is several of those.
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parent.parent / "examples" / "composed_capture.py"


@pytest.fixture(scope="module")
def run():
    spec = importlib.util.spec_from_file_location("composed_capture", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            out = mod.main()
    return mod, out


def test_the_clock_carries_the_pll_and_the_clock_triggered_scope_does_not_see_it(run):
    mod, out = run
    cap, var = out["capture"], out["variants"]
    assert cap["edges"][0] > 60 and cap["edges"][1] > 250
    assert abs(cap["clock"] / mod.PLL_PS - 1) < 0.10, cap["clock"]      # the PLL, on the clock
    assert var["PLL jitter removed"]["clock"] < 0.05 * mod.PLL_PS      # and nothing else is
    # shared, the data's edges do not move when the PLL is taken out ...
    assert abs(var["PLL jitter removed"]["payload"] / cap["payload"] - 1) < 0.12
    # ... and with an oscillator each they carry both: about sqrt(14^2 + 2 * 8^2) = 18 ps
    want = np.hypot(cap["payload"], np.sqrt(2) * mod.PLL_PS)
    assert abs(var["a PLL each, not one shared"]["payload"] / want - 1) < 0.15


def test_the_payload_is_the_channel_and_the_preamble_is_the_neighbour(run):
    _mod, out = run
    cap, var = out["capture"], out["variants"]
    assert cap["payload"] > 1.8 * cap["preamble"]
    quiet = var["channel loss 6 -> 0.5 dB"]
    assert quiet["payload"] < 0.25 * cap["payload"] and quiet["preamble"] < 0.5 * cap["preamble"]
    lone = var["133 MHz neighbour removed"]
    assert lone["preamble"] < 0.75 * cap["preamble"]                   # 3 sigma of 8 %
    assert abs(lone["payload"] / cap["payload"] - 1) < 0.12
    for what in ("supply ripple removed", "scope noise removed"):      # neither is the cause
        assert abs(var[what]["payload"] / cap["payload"] - 1) < 0.12
        assert abs(var[what]["preamble"] / cap["preamble"] - 1) < 0.25


def test_the_spread_follows_the_boards_loss_from_unit_to_unit_and_ffe_has_an_optimum(run):
    _mod, out = run
    loss, spread = np.array(out["units"]).T
    assert len(set(np.round(loss, 6))) == 5 and np.all((loss > 5.1) & (loss < 6.9))
    assert np.array_equal(np.argsort(loss), np.argsort(spread))        # more loss, more spread
    assert spread.max() > 1.3 * spread.min()
    ffe = out["ffe"]
    assert ffe[-0.1]["payload"] < 0.85 * ffe[1.0]["payload"] < ffe[-0.2]["payload"]
    assert out["replayed"]

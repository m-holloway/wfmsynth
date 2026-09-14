"""An eye measurement has to be told how many levels the record has.

`eye_height` defaulted to four. On a binary record that measures a four-level eye that is not
there, and it does it silently: MEASURED on one NRZ record through a lossy channel, reading it as
four levels gives 0.0017 against the 0.0432 it actually has -- a 25x understatement that looks
like a closed eye.

Swapping the default to two only moves the trap onto PAM4. The level count is a property of the
record, not a preference, and the caller always knows it because the caller chose the carrier -- so
it is required, and a caller who has not said cannot get a plausible wrong answer.
"""
from __future__ import annotations

import numpy as np
import pytest

import wfmsynth as ws

BAUD, FS = 16e9, 256e9


def _record(kind, pattern):
    g = ws.Grid(fs=FS, baud=BAUD, n=1 << 15, v_full=0.8)
    x = (ws.Signal(seed=1, grid=g)
         .carrier(kind, pattern=pattern, n_ui=2048, tr_frac=0.35, causal=True, seed=3)
         .lossy(loss_db=12.5, loss_at_ghz=8.0, causal=True)).waveform()
    return np.asarray(x, float), g


def test_eye_height_requires_the_level_count():
    x, g = _record("nrz", "prbs13")
    with pytest.raises(TypeError, match="levels"):
        ws.eye_height(x, g)


def test_best_phase_requires_the_level_count():
    x, g = _record("nrz", "prbs13")
    with pytest.raises(TypeError, match="levels"):
        ws.best_phase(x, g)


def test_reading_a_binary_record_as_four_levels_understates_it_enormously():
    """The number that motivated making it required. Both answers are finite and plausible; only
    one is the eye that is there."""
    x, g = _record("nrz", "prbs13")
    as_two = ws.eye_height(x, g, levels=2, defn="contour")
    as_four = ws.eye_height(x, g, levels=4, defn="contour")
    assert as_two > 10 * as_four, (
        f"levels=2 gives {as_two:.6f}, levels=4 gives {as_four:.6f}")


def test_both_level_counts_are_accepted_for_the_carrier_that_wants_them():
    for kind, pattern, levels in (("nrz", "prbs13", 2), ("pam4", "prbs13q", 4)):
        x, g = _record(kind, pattern)
        h = ws.eye_height(x, g, levels=levels, defn="contour")
        assert np.isfinite(h) and h > 0.0, f"{kind}: {h}"


def test_the_contour_definition_floors_once_the_eye_closes():
    """So it is not a health check on a closed eye: past closure it sits near zero whatever you do
    to the channel. `sigma` goes negative instead -- which is the eye being shut, not an error --
    and stays well away from zero, so it still distinguishes one closed eye from another."""
    g = ws.Grid(fs=FS, baud=BAUD, n=1 << 15, v_full=0.8)

    def eye(loss_db, defn):
        x = (ws.Signal(seed=1, grid=g)
             .carrier("pam4", pattern="prbs13q", n_ui=2048, tr_frac=0.35, causal=True, seed=3)
             .lossy(loss_db=loss_db, loss_at_ghz=8.0, causal=True)).waveform()
        return ws.eye_height(np.asarray(x, float), g, levels=4, defn=defn)

    closed = [eye(db, "contour") for db in (18.0, 24.0, 30.0)]
    assert max(closed) < 0.01, f"expected a floored contour, got {closed}"
    assert max(closed) - min(closed) < 0.005, f"contour is not floored, it varies: {closed}"

    sigma = [eye(db, "sigma") for db in (18.0, 24.0, 30.0)]
    assert all(v < -0.05 for v in sigma), f"sigma should be clearly negative: {sigma}"
    # it still separates one closed eye from another, which the contour no longer does
    assert max(sigma) - min(sigma) > 2 * (max(closed) - min(closed)), (
        f"sigma spread {max(sigma)-min(sigma):.4f} vs contour {max(closed)-min(closed):.4f}")

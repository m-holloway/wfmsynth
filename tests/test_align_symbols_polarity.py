"""GitHub #52: `measure.align_symbols` picked the offset with `argmax(corr)`, so an inverted
record (a differential pair wired backwards, or any inverting stage) has every true-alignment
correlation NEGATIVE, and the max lands on noise. `argmax(|corr|)` finds the same alignment
either way and reports the sign, which is the only way to detect a swapped pair -- eye height,
peak-to-peak and RMS are all identical under negation.
"""
from __future__ import annotations

import numpy as np

import wfmsynth as ws
from wfmsynth import physics as P, measure as M

G = ws.Grid(fs=256e9, baud=28e9, n=16384)


def _chain():
    return (ws.Signal(seed=7, grid=G).carrier("nrz", n_ui=1792, causal=True)
            .lossy(loss_db=8.0, loss_at_ghz=14.0, causal=True)).waveform()


def test_inverted_record_locks_onto_the_same_alignment_with_a_negative_sign():
    x = _chain()
    tx = P.carrier_symbols("nrz", 1792, 1, "legacy")
    off_pos, corr_pos, _ = M.align_symbols(x, G, tx, levels=2)
    off_neg, corr_neg, _ = M.align_symbols(-x, G, tx, levels=2)
    assert off_neg == off_pos
    assert corr_neg == -corr_pos
    assert corr_pos > 0.9
    assert corr_neg < -0.9


def test_upright_record_is_unaffected_the_positive_lock_still_wins():
    """The fix must not change the normal (non-inverted) case: argmax(|corr|) picks the same
    offset argmax(corr) already did whenever the true alignment is the global magnitude peak."""
    x = _chain()
    tx = P.carrier_symbols("nrz", 1792, 1, "legacy")
    off, corr, corr_zero = M.align_symbols(x, G, tx, levels=2)
    assert off == 1
    assert corr > 0.9
    assert corr > corr_zero + 0.4

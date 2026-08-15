"""
wfmsynth.cdr — clock recovery as part of "what the scope records".

An instrument does not show you the raw record; it shows you the record folded by a
**recovered clock**. The CDR is a phase-locked loop that tracks the data timing, so it
tracks OUT timing jitter slower than its loop bandwidth and passes jitter faster than it.
The loop bandwidth and order therefore materially change what an eye looks like — an emitted
eye is only meaningful alongside the recovery that produced it.

`recover_clock` is the standard linearized-PLL jitter transfer. Given a per-symbol timing
phase (the transmitted timing jitter, in any unit — seconds or UI), it returns:

  * ``clock``    the recovered-clock phase — a LOW-pass copy of the input (the loop follows
                 slow wander), corner at ``loop_bw``.
  * ``residual`` the timing error the sampler (and hence the eye) actually sees — a
                 HIGH-pass copy, ``input - clock``. This is the jitter the recorded eye shows.

Order picks the loop type: order 1 (single pole) leaves a static phase error under a
frequency offset; order 2 (type-2, a double DC zero in the residual) tracks a frequency
offset out to zero. `loop_bw` is in Hz, `baud` the symbol rate.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as _sig


def _transfer(loop_bw, order, damping):
    """Analog (clock, residual) transfer functions of a linearized order-N PLL CDR."""
    wn = 2 * np.pi * loop_bw
    if order == 1:
        return ([wn], [1, wn]), ([1, 0], [1, wn])                       # LP1 clock, HP1 residual
    if order == 2:
        clk = ([2 * damping * wn, wn ** 2], [1, 2 * damping * wn, wn ** 2])
        res = ([1, 0, 0], [1, 2 * damping * wn, wn ** 2])               # type-2 HP residual
        return clk, res
    raise ValueError(f"order must be 1 or 2 (got {order!r})")


def recover_clock(phase, baud, loop_bw, order=2, damping=0.707):
    """Run a CDR over a per-symbol timing ``phase`` sequence. Returns ``(clock, residual)``:
    ``clock`` is the recovered-clock phase (low-pass, the loop follows slow jitter) and
    ``residual = phase - clock`` is the timing error the recorded eye shows (high-pass,
    corner ~ ``loop_bw``). Units of the outputs match ``phase``."""
    phase = np.asarray(phase, float)
    (cn, cd), (en, ed) = _transfer(loop_bw, order, damping)
    bc, ac = _sig.bilinear(cn, cd, fs=baud)
    be, ae = _sig.bilinear(en, ed, fs=baud)
    return _sig.lfilter(bc, ac, phase), _sig.lfilter(be, ae, phase)


def jitter_transfer(phase, baud, loop_bw, order=2, damping=0.707):
    """The residual (eye) timing jitter after clock recovery — i.e. ``recover_clock(...)[1]``.
    This is the jitter an emitted eye actually exhibits; measure/label eyes against it, not
    against the transmitted jitter."""
    return recover_clock(phase, baud, loop_bw, order=order, damping=damping)[1]


def tracked_out_fraction(phase, baud, loop_bw, order=2, damping=0.707, warmup=0.5):
    """How much of the input timing jitter the CDR tracks out: ``1 - ptp(residual)/ptp(phase)``
    over the steady-state tail (after ``warmup`` fraction of the record). ~1 for jitter well
    below the loop bandwidth, ~0 for jitter well above it."""
    res = jitter_transfer(phase, baud, loop_bw, order=order, damping=damping)
    s = int(warmup * len(phase))
    return 1.0 - np.ptp(res[s:]) / (np.ptp(np.asarray(phase, float)[s:]) + 1e-30)

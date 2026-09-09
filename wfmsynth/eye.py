"""
wfmsynth.eye — the RECORDED eye: a record folded on the clock recovered from it.

An eye diagram is not a picture of a waveform; it is a picture of a waveform *as a
receiver saw it*. The receiver decides where each symbol begins, and it decides that
with a phase-locked loop that tracks slow timing wander and cannot track fast timing
jitter. Fold at the rate a grid *states* and you have assumed away the very timing
error the eye exists to show. So this module does what a receiver does, in order:

  1. find the threshold crossings (hysteresis-armed, interpolated) — `crossings`
  2. fit a symbol period and phase to them by least squares — `period_fit`
  3. run the library's CDR over the per-symbol timing phase — `recover`
  4. fold the record at the decision instants that come out — `density`

Step 3 is `wfmsynth.cdr.recover_clock` and nothing else: the linearised type-2 PLL,
order 2, damping 0.707, loop bandwidth ``baud/1667`` by default (the PCIe Gen-3
figure). Its jitter transfer IS the measurement — 0.20 UI of sinusoidal jitter at
0.02x the loop bandwidth comes back as 0.0015 UI of residual, the same amplitude at
3.33x comes back at 0.1406 UI against 0.1414 built. Nothing in this module may change
those numbers, and `spikes/eye3m/verify.py` re-measures them on every change.

WHY IT IS WRITTEN THE WAY IT IS. At the size this was built for — 3 M UI, 48 Mpt,
384 MB — the obvious transcription of each step allocates several times the record to
say something small about it, and that allocation, not the arithmetic, is the cost.
The edge finder is the case in point: marking every sample that sits outside the
hysteresis band produces 45 M "events" of which 1.5 M survive, and walking back to
each straddling sample pair with a running maximum over an int64 index array costs
five full-length passes and 2.5 GB. Both are avoided here without changing a single
returned number — the surviving events are exactly the *first* sample of each armed
run, and the walk-back target is exactly the last threshold crossing before it, so
both are found on arrays of 750 k rather than 48 M. Measured on the 48 Mpt record:
324 ms and 2.6 GB of intermediates become 76 ms and 190 MB, and the crossing
positions are bit-identical.

Everything here is exact. The counts in the returned density are integers and every
float that feeds them is computed in the same order and the same precision as the
straightforward version; the parallel paths differ only in *when* work happens, never
in what it computes. `spikes/eye3m/` holds the before/after measurements and the
identity proofs.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import cdr as _cdr

# The loop bandwidth a receiver is assumed to run, as a fraction of the symbol rate.
# 1/1667 is the PCIe Gen-3 figure and the usual textbook starting point.
LOOP_DIVISOR = 1667.0

# A period fit whose leftover timing error is worse than this did not lock. A quarter
# of a unit interval is where an edge stops belonging to one identifiable symbol, so
# past it the fold would be arbitrary.
LOCK_RMS_UI = 0.25

# The per-thread scan block. Small enough that a chunk's four boolean masks stay in
# L2 and the record streams through once; large enough that the per-block Python
# overhead is noise. Measured flat from 1 M to 4 M samples on an M4.
_BLOCK = 1 << 21

# Why a picture is folded on the clock it is folded on. `report['reason']` is one of
# these keys, always -- a reader is told which of the two claims is being made, and a
# fallback to the grid's rate is never silent.
REASONS = {
    "recovered": "folded on the clock recovered from this record",
    "short": "record too short to recover a clock from",
    "edges": "too few edges to recover a clock from",
    "nolock": "the edges do not line up on any one symbol period",
    "toofast": "symbols shorter than two samples: no clock to recover",
    "fromgrid": "period fitted to the edges, but seeded and decided by the grid's rate",
    "still": "record shorter than three loop cycles: the loop never moved",
    "nowindow": ("the recovered clock leaves too little of the record inside the fold "
                 "window to draw an eye"),
}


def _threads(want=None):
    """How many workers to use. One on Emscripten/Pyodide, where there are none."""
    if sys.platform == "emscripten":
        return 1
    if want is None:
        want = os.environ.get("WFMSYNTH_EYE_THREADS")
        want = int(want) if want else min(8, os.cpu_count() or 1)
    return max(1, int(want))


def _map(fn, items, threads):
    """`fn` over `items`, in order, on `threads` workers. Order is preserved, so
    nothing downstream can tell how many workers ran."""
    items = list(items)
    if threads <= 1 or len(items) < 2:
        return [fn(i) for i in items]
    with ThreadPoolExecutor(min(threads, len(items))) as ex:
        return list(ex.map(fn, items))


def levels(x, threads=None):
    """``(mean, min, max)`` of the record — the threshold the edge finder uses and
    the volts axis the fold maps onto, in one place because every one of them is a
    full pass over 384 MB and the frame used to make five of them (mean and ptp for
    the edges, min and max again for the picture). Each reduction is numpy's own,
    unsplit, so the summation order and therefore the last bit of the mean is
    unchanged; they are merely computed at the same time. 14.0 ms -> 6.1 ms."""
    x = np.asarray(x, float).reshape(-1)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    fns = (np.mean, np.min, np.max)
    return tuple(float(v) for v in _map(lambda f: f(x), fns, _threads(threads)))


# ------------------------------------------------------------------ the edges

def _scan(x, s, e, level, hyst):
    """One block of the record, reduced to the four small index sets the edge finder
    needs: where an armed HIGH run starts, where an armed LOW run starts, and the
    sample just before each upward and each downward crossing of the threshold
    itself. The block is extended one sample to the left so a start or a crossing
    that straddles the boundary belongs to exactly one block."""
    off = s - 1 if s > 0 else 0
    b = x[off:e]
    up = b > level + hyst
    dn = b < level - hyst
    hi = up if hyst == 0.0 else b > level
    lw = dn if hyst == 0.0 else b < level
    if s > 0:
        us = np.flatnonzero(up[1:] & ~up[:-1]) + (off + 1)
        ds = np.flatnonzero(dn[1:] & ~dn[:-1]) + (off + 1)
    else:
        m = np.empty(up.size, bool); m[0] = up[0]; m[1:] = up[1:] & ~up[:-1]
        us = np.flatnonzero(m)
        m = np.empty(dn.size, bool); m[0] = dn[0]; m[1:] = dn[1:] & ~dn[:-1]
        ds = np.flatnonzero(m)
    be = np.flatnonzero(~hi[:-1] & hi[1:]) + off
    ae = np.flatnonzero(~lw[:-1] & lw[1:]) + off
    return us, ds, be, ae


def _last_before(ends, ev):
    """For each ``ev``, the position in ``ends`` of the last element strictly below
    it, or -1. Both inputs are sorted and share no value, so this is a MERGE — and a
    merge written as one stable argsort plus a running count is 11 ms where
    ``np.searchsorted`` over the same 750 k needles is 42 ms, because a binary search
    per needle walks the haystack in a pattern no cache can predict."""
    if ends.size == 0:
        return np.full(ev.size, -1, np.int64)
    comb = np.concatenate((ends, ev))
    order = np.argsort(comb, kind="stable")
    is_ev = order >= ends.size
    return np.cumsum(~is_ev)[is_ev] - 1


def crossings(x, level, hyst, threads=None):
    """Hysteresis-armed threshold crossings, interpolated, in fractional sample index.

    The signal must travel ``hyst`` volts past ``level`` to arm the next crossing —
    without it a record with any noise on it recrosses several times per edge and the
    recovered period collapses — but the position reported is still the interpolated
    crossing of the threshold itself, so hysteresis costs no timing accuracy.

    Returns ``(positions, directions)``: fractional sample indices and +1/-1.
    """
    x = np.asarray(x, float)
    n = int(x.size)
    empty = (np.zeros(0), np.zeros(0, np.int8))
    if n < 2:
        return empty
    level = float(level)
    hyst = float(hyst) if hyst > 0 else 0.0

    nt = _threads(threads)
    bounds = [(s, min(n, s + _BLOCK)) for s in range(0, n, _BLOCK)]
    parts = _map(lambda se: _scan(x, se[0], se[1], level, hyst), bounds, nt)
    us = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.int64)
    ds = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.int64)
    if us.size + ds.size == 0:
        return empty

    # Every event the exhaustive version would keep is the FIRST sample of an armed
    # run: if the sample before an armed HIGH sample is also armed HIGH, the previous
    # event carried the same label and the alternation filter drops it. So the filter
    # runs over 1.5 M run starts instead of 45 M armed samples, and keeps the same set.
    ev = np.concatenate((us, ds))
    lab = np.concatenate((np.ones(us.size, bool), np.zeros(ds.size, bool)))
    order = np.argsort(ev, kind="stable")
    ev = ev[order]; lab = lab[order]
    keep = np.empty(ev.size, bool)
    keep[0] = lab[0] != (x[0] >= level)
    keep[1:] = lab[1:] != lab[:-1]
    ev = ev[keep]; lab = lab[keep]
    if ev.size == 0:
        return empty

    # The straddling pair. The last sample at or below the threshold before an upward
    # event is the last upward THRESHOLD crossing before it, which is one of the small
    # index sets the scan already produced -- no running maximum over the record.
    be = np.concatenate([p[2] for p in parts])
    ae = np.concatenate([p[3] for p in parts])
    j0 = np.empty(ev.size, np.int64)
    for mask, ends in ((lab, be), (~lab, ae)):
        k = _last_before(ends, ev[mask])
        j0[mask] = np.where(k >= 0, ends[np.maximum(k, 0)], -1) if ends.size else -1
    good = (j0 >= 0) & (j0 + 1 < n)
    j0 = j0[good]; lab = lab[good]
    if j0.size == 0:
        return empty

    a = x[j0] - level
    b = x[j0 + 1] - level
    den = a - b
    frac = np.where(den == 0.0, 0.0, a / np.where(den == 0.0, 1.0, den))
    return j0 + frac, np.where(lab, 1, -1).astype(np.int8)


# ----------------------------------------------------------------- the period

def period_fit(cross, ui_seed):
    """Symbol period and phase by least squares over the crossing positions, plus the
    leftover timing error per edge in unit intervals (the TIE).

    The UI index accumulates from CONSECUTIVE spacings, not from the distance back to
    the first crossing: adjacent edges are one to a few UI apart, so an error in the
    starting period cannot mis-number an edge, while indexing from t[0] lets a 3 %
    period error mis-number everything past UI 17 and then fits a straight line to
    nonsense. Returns ``(tie, m, ui, phase)`` or None.
    """
    t = np.asarray(cross, float)
    n = int(t.size)
    if n < 4 or not (ui_seed > 0):
        return None
    ui = float(ui_seed)
    phase = float(t[0])
    m = np.zeros(n)
    for _ in range(4):
        step = np.maximum(1.0, np.round(np.diff(t) / ui))
        m = np.concatenate(([0.0], np.cumsum(step)))
        sm = m.sum(); st = t.sum(); smm = float(m @ m); smt = float(m @ t)
        den = n * smm - sm * sm
        if den == 0:
            return None
        nxt = (n * smt - sm * st) / den
        if not (nxt > 0):
            return None
        done = abs(nxt - ui) < 1e-12 * abs(ui)
        ui = nxt
        phase = (st - ui * sm) / n
        if done:
            break
    tie = (t - (phase + ui * m)) / ui
    return tie, m.astype(np.int64), ui, phase


def _seeds(cross, spb_nominal):
    """Periods to try. From the DATA first: the shortest spacing that occurs often
    between edges is one unit interval, whatever the grid believes. The grid's figure
    is tried too, last, so a correct grid costs nothing and a wrong one costs the
    picture nothing either. The four percentiles are taken in ONE call — one partition
    of the spacings instead of four, and the same four numbers out."""
    d = np.diff(cross)
    d = d[d > 0]
    out = []
    if d.size:
        qs = np.percentile(d, [5.0, 10.0, 20.0, 35.0])
        out = [("record", float(q)) for q in qs]
    if spb_nominal > 1.0:
        out.append(("grid", float(spb_nominal)))
    return out


# -------------------------------------------------------------- the recovery

def _recover_clock(ph, baud, loop_bw, order, damping, threads):
    """`wfmsynth.cdr.recover_clock`, with its two filters run at the same time.

    An IIR loop filter is sequential and stays sequential -- this does not touch the
    recurrence. It only notices that `recover_clock` runs the CLOCK transfer and the
    RESIDUAL transfer over the same input, independently, and that scipy's lfilter
    releases the interpreter lock while it works: 22.6 ms becomes 11.2 ms and both
    outputs are bit-identical to the serial call. The transfer functions, the
    bilinear transform and the filter itself are the library's, untouched."""
    if threads < 2:
        return _cdr.recover_clock(ph, baud, loop_bw, order=order, damping=damping)
    from scipy import signal as _sig
    (cn, cd), (en, ed) = _cdr._transfer(loop_bw, order, damping)
    bc, ac = _sig.bilinear(cn, cd, fs=baud)
    be, ae = _sig.bilinear(en, ed, fs=baud)
    both = _map(lambda t: _sig.lfilter(t[0], t[1], ph), ((bc, ac), (be, ae)), 2)
    return both[0], both[1]


def recover(x, fs, spb_nominal, order=2, damping=0.707, loop_bw_hz=None,
            loop_divisor=LOOP_DIVISOR, threads=None, stats=None):
    """Recover the symbol clock FROM THE RECORD, then run the library's CDR over the
    per-symbol timing phase.

    Returns ``(report, fold)`` where ``fold`` is ``(centres, ui)`` — the decision
    instant of each folded symbol, in samples, and the recovered unit interval — or
    ``None`` when nothing locked, in which case ``report['reason']`` says why -- a key
    into `REASONS`, so a caller can print the sentence or branch on the code.

    Why not the nominal clock: folding at ``fs/baud`` with one measured phase assumes
    the very clock whose imperfection the eye exists to reveal, and it keeps drawing a
    confident picture when the waveform is no longer running at that rate at all.
    """
    x = np.asarray(x, float).reshape(-1)
    n = int(x.size)
    nt = _threads(threads)
    rep = {"source": "nominal", "nominalSamplesPerUi": _num(spb_nominal)}
    if n < 256:
        rep["reason"] = "short"
        return rep, None
    mean, v_min, v_max = levels(x, nt) if stats is None else stats
    level = float(mean)
    hyst = (float(v_max) - float(v_min)) * 0.1
    rep["thresholdVolts"] = _num(level)
    cross, _dirs = crossings(x, level, hyst, threads=nt)
    rep["edges"] = int(cross.size)
    if cross.size < 32:
        rep["reason"] = "edges"
        return rep, None

    seeds = _seeds(cross, spb_nominal)
    # The candidate fits are independent of one another, so they are run at once. A
    # fit is ~11 ms on 1.5 M crossings and there are five of them; in parallel the
    # ballot costs about what one fit costs. Same fits, same ballot, same winner --
    # the scoring below is done afterwards, in seed order, on the collected results.
    memo = {}
    for _o, s in seeds:
        memo.setdefault(float(s), None)
    keys = list(memo)
    for k, fit in zip(keys, _map(lambda s: period_fit(cross, s), keys, nt)):
        memo[k] = fit

    best = None
    saw_short = False
    from_record = []
    for origin, s in seeds:
        if not (s > 1.0):
            continue
        fit = memo[float(s)]
        if fit is None:
            continue
        tie, m, ui, phase = fit
        if not (ui >= 2.0):
            saw_short = True
            continue
        if int(m[-1]) < 8:
            continue
        # SCORE IN SAMPLES, NOT IN UNIT INTERVALS. `tie` is the leftover timing error
        # divided by the very period being scored, so a candidate at twice the true
        # period halves its own score for free. On clean data every seed converges to
        # the same period and the bias is invisible; where the edges are sparse -- a
        # closed eye -- the seeds start at a MULTIPLE of the true unit interval and
        # the bias decides the outcome.
        score = float(np.std(tie)) * ui
        if origin == "record":
            from_record.append(ui)
        if best is None or score < best[0]:
            best = (score, tie, m, ui, phase)
    if best is None:
        rep["reason"] = "toofast" if saw_short else "nolock"
        return rep, None
    _score, tie, m, ui, phase = best
    rms = float(np.std(tie))
    rep["lockRmsUi"] = _num(rms)
    if not (rms < LOCK_RMS_UI):
        rep["reason"] = "nolock"
        return rep, None
    supported = any(abs(u - ui) <= 0.01 * ui for u in from_record)
    rep["periodFrom"] = "record" if supported else "grid"

    baud = (fs / ui) if (fs > 0 and ui > 0) else 0.0
    loop_bw = (baud / loop_divisor) if loop_bw_hz is None else float(loop_bw_hz)
    loop_bw = float(loop_bw) if loop_bw > 0 else 0.0

    # One phase sample per SYMBOL, held between transitions: a receiver only updates
    # its phase estimate on a transition, so holding the last measurement is the
    # honest reconstruction. Interpolating across a long run invents edges. `m` is
    # strictly increasing, so the hold is a run-length expansion -- exact copies of
    # the same numbers the fill-forward produced, with no NaN pass over the symbols.
    n_sym = int(m[-1]) + 1
    held = np.zeros(n_sym, np.int64)
    held[m[1:]] = 1
    np.cumsum(held, out=held)
    ph = tie[held]

    tracked = None
    if loop_bw > 0 and baud > 0 and n_sym >= 8:
        clock, residual = _recover_clock(ph, baud, loop_bw, order, damping, nt)
        # `tracked_out_fraction` is 1 - ptp(residual)/ptp(phase) over the steady-state
        # tail, and the residual it needs is the one just computed. Calling it would
        # run the same two IIR filters a second time for the same answer.
        s0 = int(0.5 * ph.size)
        tracked = _num(1.0 - np.ptp(residual[s0:]) / (np.ptp(ph[s0:]) + 1e-30))
    else:
        clock = np.zeros(n_sym)
        residual = ph.copy()

    periods = (loop_bw * n_sym / baud) if (loop_bw > 0 and baud > 0) else 0.0
    moved = periods >= 3.0
    # The loop starts from rest, so its first time constants are a settling transient
    # rather than jitter -- dropped from the fold. When the record is shorter than
    # three cycles of the loop bandwidth the loop never moved at all, so there is no
    # transient to drop and dropping a quarter of the traces would only cost ink.
    warm = int(min(n_sym // 4, round(baud / loop_bw))) if (loop_bw > 0 and baud > 0) else 0
    skip = warm if moved else 0
    tail = residual[skip:] if residual.size > skip else residual

    rep["source"] = "recovered"
    rep["reason"] = "recovered" if supported else "fromgrid"
    rep["samplesPerUi"] = _num(ui)
    rep["phaseSamples"] = _num(phase)
    rep["baudHz"] = _num(baud)
    rep["loopBwHz"] = _num(loop_bw)
    rep["order"] = int(order)
    rep["damping"] = float(damping)
    rep["symbols"] = int(n_sym)
    rep["warmupSymbols"] = int(skip)
    rep["rmsUi"] = _num(np.std(tie))
    rep["ppUi"] = _num(np.ptp(tie))
    rep["residualRmsUi"] = _num(np.std(tail)) if tail.size else 0.0
    rep["trackedFraction"] = tracked
    rep["loopPeriods"] = _num(periods)
    if not moved:
        rep["stillReason"] = "still"

    # The decision instant of symbol k: half a unit interval past the recovered edge,
    # displaced by the phase the loop is tracking at that symbol. THIS is what lets
    # the eye show a timing impairment -- the instants move with the data instead of
    # marching at a rate the record was never asked to have.
    centres = phase + ui * (np.arange(n_sym, dtype=float) + 0.5 + np.asarray(clock, float))
    return rep, (centres[skip:], float(ui))


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# ------------------------------------------------------------------- the fold

# Traces per inner block. The block's scratch is a handful of arrays of
# ``block * w`` doubles, so 2048 x 128 keeps it near 2 MB -- resident in L1/L2 --
# while the record samples it gathers from span 2048 UI, about 260 kB, resident
# too. Measured on the 48 Mpt record, one thread: 512 traces 2301 ms, 2048
# 2087 ms, 8192 2049 ms, 65536 2191 ms; on eight threads 2048 is the fastest of
# the four and holds a quarter of 8192's scratch, so it is the one kept.
_FOLD_BLOCK = 2048


def _fold_block(x, c, ph, w, h, v_min, span, cols):
    """One worker's share of the traces, as counts on the h x w grid.

    Written as in-place ufuncs on two scratch buffers rather than as an
    expression: the expression form allocates ten temporaries of block*w doubles
    per block and reads each of them back, and at 384 M interpolated points that
    traffic is 8 % of the fold (2227 ms -> 2049 ms on one thread, 627 -> 553 on
    eight). Every operation is in the same order and the same precision as the
    expression it replaces, so the counts are bit-identical."""
    img = np.zeros(h * w, np.int64)
    for k in range(0, c.size, _FOLD_BLOCK):
        cb = c[k:k + _FOLD_BLOCK]
        pos = (cb[:, None] + ph[None, :]).ravel()
        # A two-tap lerp on a precomputed integer index, not np.interp against a
        # 0..n-1 axis. np.interp would binary-search a 48 M-point axis for every one
        # of these positions, and building that axis is a 384 MB allocation on its
        # own -- 12.4 ms of a 12.7 ms fold at 400 traces. The values agree to the
        # last bit: with a unit axis the interpolant IS x[i] + (x[i+1]-x[i])*frac.
        i = pos.astype(np.int64)
        np.subtract(pos, i, out=pos)                     # the fractional part
        lo = x[i]
        np.add(i, 1, out=i)
        v = x[i]
        np.subtract(v, lo, out=v)
        np.multiply(v, pos, out=v)
        np.add(v, lo, out=v)                             # the interpolated volts
        np.subtract(v, v_min, out=v)
        np.divide(v, span, out=v)
        np.subtract(1.0, v, out=v)
        np.multiply(v, h - 1, out=v)                     # row 0 = highest volts
        rows = v.astype(np.int64)
        np.clip(rows, 0, h - 1, out=rows)
        np.multiply(rows, w, out=rows)
        np.add(rows, cols[:rows.size], out=rows)
        img += np.bincount(rows, minlength=h * w)
    return img


def density(x, centres, ui, w=128, h=96, ui_span=2.0, v_min=None, v_max=None,
            max_traces=400, threads=None):
    """Fold the record at GIVEN sampling instants — one per symbol, from a recovered
    clock — into a ``h x w`` count density, row 0 = highest volts.

    Returns ``(img, traces)`` with ``img`` a ``uint32`` array of shape ``(h, w)``, or
    None when the window leaves nothing to draw. The counts are integers, so the
    per-worker partial images sum to the same picture in any order.
    """
    x = np.asarray(x, float).reshape(-1)
    n = int(x.size)
    v_min = float(np.min(x)) if v_min is None else float(v_min)
    v_max = float(np.max(x)) if v_max is None else float(v_max)
    span = v_max - v_min
    c = np.asarray(centres, float)
    if span <= 0 or not (ui > 0) or c.size == 0 or w < 2 or h < 1:
        return None
    half = 0.5 * float(ui_span) * float(ui)
    c = c[(c - half >= 0.0) & (c + half <= n - 1.0)]
    if c.size < 4:
        return None
    cap = int(max_traces) if int(max_traces) > 0 else 1
    step = max(1, -(-int(c.size) // cap))
    c = c[::step]
    ph = np.linspace(-half, half, w)
    cols = np.tile(np.arange(w, dtype=np.int64), _FOLD_BLOCK)
    nt = _threads(threads)
    nt = min(nt, max(1, c.size // _FOLD_BLOCK))
    parts = np.array_split(c, nt) if nt > 1 else [c]
    imgs = _map(lambda cc: _fold_block(x, cc, ph, w, h, v_min, span, cols), parts, nt)
    img = imgs[0]
    for extra in imgs[1:]:
        img += extra
    return img.astype(np.uint32).reshape(h, w), int(c.size)


# ------------------------------------------------------------------ the frame

def eye(x, grid, w=128, h=96, ui_span=2.0, max_traces=400, order=2, damping=0.707,
        loop_bw_hz=None, threads=None):
    """The whole recorded eye: recover the clock from ``x``, fold on it, and report
    both the picture and the clock that made it.

    Returns ``(img, meta)``. ``meta['clock']`` is the recovery report — the period the
    edges actually determined, the loop the CDR ran, and the residual jitter the
    picture therefore shows. Read the picture against it; an eye without its clock is
    a claim without its units.
    """
    x = np.asarray(x, float).reshape(-1)
    n = int(x.size)
    nt = _threads(threads)
    spb = float(grid.samples_per_ui) if (grid is not None and grid.baud is not None) else 0.0
    fs = float(grid.fs) if grid is not None else 0.0
    stats = levels(x, nt) if n else (0.0, 0.0, 0.0)
    v_min, v_max = stats[1], stats[2]
    rep, fold = recover(x, fs, spb, order=order, damping=damping, loop_bw_hz=loop_bw_hz,
                        threads=nt, stats=stats)
    img = None
    traces = 0
    fold_spb = spb
    if fold is not None:
        r = density(x, fold[0], fold[1], w, h, ui_span, v_min, v_max, max_traces, threads=nt)
        if r is None:
            rep = dict(rep, source="nominal", reason="nowindow")
        else:
            img, traces = r
            fold_spb = fold[1]
    if img is None:
        img = np.zeros((h, w), np.uint32)
    meta = {"w": w, "h": h, "uiSpan": ui_span, "vMin": v_min, "vMax": v_max,
            "samplesPerUi": fold_spb, "traces": traces, "total": n, "clock": rep}
    return img, meta


# Package-level aliases. `wfmsynth.eye_density(x, grid)` reads better than
# `wfmsynth.eye.eye(x, grid)`, and exporting `eye` from the package would shadow the
# module of the same name.
eye_density = eye
eye_recover = recover
eye_crossings = crossings

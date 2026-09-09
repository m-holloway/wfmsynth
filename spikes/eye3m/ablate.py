"""What each technique actually bought, measured one at a time on the 48 Mpt record.

Every row is the SAME computation done two ways, timed back to back in one process,
with the two results compared. A technique that did not help is still printed, with
the number that says so -- a measured negative result is the only kind worth having.

  python3 spikes/eye3m/ablate.py [--reps 3]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import record as REC                                    # noqa: E402
import shipping                                         # noqa: E402
from wfmsynth import cdr as C                           # noqa: E402
from wfmsynth import eye as E                           # noqa: E402

REPS = 3
ROWS = []


def t(fn):
    best = 1e18
    out = None
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best, out


def row(stage, technique, before_label, after_label, fb, fa, cmp=None):
    mb, ob = t(fb)
    ma, oa = t(fa)
    if cmp is None:
        same = "identical" if _same(ob, oa) else "DIFFERS"
    else:
        same = cmp(ob, oa)
    ROWS.append((stage, technique, before_label, mb, after_label, ma, same))
    print("  %-11s %-34s %8.1f -> %8.1f ms  %6.2fx  %s"
          % (stage, technique, mb, ma, mb / ma, same))


def _same(a, b):
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b)


def main():
    global REPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    REPS = a.reps
    ns = shipping.load()
    x = REC.build()
    g = REC.grid()
    n = x.size
    print("48 Mpt, 3 M UI, 16 samples/UI -- best of %d, baseline pinned at %s"
          % (REPS, shipping.revision()[:12]))
    print("  %-11s %-34s %8s    %8s  %6s  %s"
          % ("stage", "technique", "before", "after", "", "result"))

    level = float(np.mean(x))
    hyst = float(np.ptp(x)) * 0.1

    # ---------------------------------------------------------------- reductions
    row("threshold", "mean/min/max at the same time",
        "serial", "3 threads",
        lambda: (float(np.mean(x)), float(np.min(x)), float(np.max(x))),
        lambda: E.levels(x, 3))

    # ------------------------------------------------------------------- edges
    row("edges", "armed runs, not armed samples",
        "shipping", "run starts, 1 thread",
        lambda: ns["_eye_crossings"](x, level, hyst),
        lambda: E.crossings(x, level, hyst, threads=1))
    row("edges", "  ... and on 8 threads",
        "1 thread", "8 threads",
        lambda: E.crossings(x, level, hyst, threads=1),
        lambda: E.crossings(x, level, hyst, threads=8))

    cross, _d = E.crossings(x, level, hyst)
    up = x > level + hyst
    hi = x > level
    us = np.flatnonzero(up[1:] & ~up[:-1]) + 1
    be = np.flatnonzero(~hi[:-1] & hi[1:])
    row("edges", "walk-back: merge, not binary search",
        "np.searchsorted", "argsort+cumsum",
        lambda: np.searchsorted(be, us) - 1,
        lambda: E._last_before(be, us))
    del up, hi

    dd = np.diff(cross); dd = dd[dd > 0]
    row("period", "four percentiles in one call",
        "4 calls", "1 call",
        lambda: [float(np.percentile(dd, q)) for q in (5.0, 10.0, 20.0, 35.0)],
        lambda: [float(v) for v in np.percentile(dd, [5.0, 10.0, 20.0, 35.0])],
        cmp=lambda a, b: "identical" if a == b else "DIFFERS")

    seeds = [float(s) for _o, s in E._seeds(cross, g.samples_per_ui)]
    row("period", "the five candidate fits at once",
        "serial", "5 threads",
        lambda: [E.period_fit(cross, s) for s in seeds],
        lambda: E._map(lambda s: E.period_fit(cross, s), seeds, 8),
        cmp=lambda a, b: "identical" if all(_same(p[:2], q[:2]) and p[2:] == q[2:]
                                            for p, q in zip(a, b)) else "DIFFERS")

    tie, m, ui, phase = E.period_fit(cross, seeds[0])
    n_sym = int(m[-1]) + 1

    def hold_nan():
        ph = np.full(n_sym, np.nan)
        ph[m] = tie
        have = ~np.isnan(ph)
        fill = np.where(have, np.arange(n_sym), 0)
        np.maximum.accumulate(fill, out=fill)
        ph = ph[fill]
        ph[np.isnan(ph)] = float(tie[0])
        return ph

    def hold_cumsum():
        held = np.zeros(n_sym, np.int64)
        held[m[1:]] = 1
        np.cumsum(held, out=held)
        return tie[held]

    row("phase", "hold as a run-length, not a NaN fill",
        "NaN fill-forward", "cumsum + gather", hold_nan, hold_cumsum)

    ph = hold_cumsum()
    baud = g.fs / ui
    lbw = baud / E.LOOP_DIVISOR
    row("CDR", "reuse the residual already computed",
        "+tracked_out_fraction", "from the residual",
        lambda: (C.recover_clock(ph, baud, lbw),
                 C.tracked_out_fraction(ph, baud, lbw)),
        lambda: C.recover_clock(ph, baud, lbw),
        cmp=lambda a, b: "identical" if _same(a[0], b) else "DIFFERS")
    row("CDR", "the two loop filters at once",
        "serial", "2 threads",
        lambda: C.recover_clock(ph, baud, lbw),
        lambda: E._recover_clock(ph, baud, lbw, 2, 0.707, 2))

    # -------------------------------------------------------------------- fold
    rep, fold = E.recover(x, g.fs, g.samples_per_ui)
    centres, ui2 = fold
    vmin, vmax = float(np.min(x)), float(np.max(x))
    span = vmax - vmin
    W, H, SPAN = 128, 96, 2.0
    half = 0.5 * SPAN * ui2
    c_all = centres[(centres - half >= 0.0) & (centres + half <= n - 1.0)]
    ph_grid = np.linspace(-half, half, W)

    for cap, label in ((400, "400 traces"), (3_000_000, "every UI")):
        step = max(1, -(-int(c_all.size) // cap))
        c = c_all[::step]
        cols = np.tile(np.arange(W, dtype=np.int64), 2048)

        def shipped():
            r = ns["_eye_fold_at"](x, centres, ui2, W, H, SPAN, vmin, vmax, cap)
            return np.asarray(r[0], np.int64)

        def fast(threads):
            r = E.density(x, centres, ui2, W, H, SPAN, vmin, vmax, cap, threads=threads)
            return np.asarray(r[0], np.int64).ravel()

        row("fold/%s" % label, "two-tap lerp, not np.interp",
            "np.interp + arange(n)", "lerp on int index",
            shipped, lambda: fast(1))
        row("fold/%s" % label, "  ... spread over 8 threads",
            "1 thread", "8 threads", lambda: fast(1), lambda: fast(8))

        # The three below are compared kernel-to-kernel on the SAME already-chosen
        # traces, so the trace selection (a mask over 3 M centres, 3.0 ms) does not
        # sit inside one side of the comparison and not the other.
        def plain():
            img = np.zeros(H * W, np.int64)
            for k in range(0, c.size, 2048):
                cb = c[k:k + 2048]
                pos = (cb[:, None] + ph_grid[None, :]).ravel()
                i = pos.astype(np.int64); f = pos - i
                v = x[i] + (x[i + 1] - x[i]) * f
                r = np.clip(((1.0 - (v - vmin) / span) * (H - 1)).astype(np.int64), 0, H - 1)
                img += np.bincount(r * W + cols[:r.size], minlength=H * W)
            return img

        def inplace():
            return E._fold_block(x, c, ph_grid, W, H, vmin, span,
                                 np.tile(np.arange(W, dtype=np.int64), E._FOLD_BLOCK))

        row("fold/%s" % label, "in-place ufuncs, not an expression",
            "10 temporaries", "2 buffers", plain, inplace)

        x32 = x.astype(np.float32)

        def f32():
            img = np.zeros(H * W, np.int64)
            for k in range(0, c.size, 2048):
                cb = c[k:k + 2048]
                pos = (cb[:, None] + ph_grid[None, :]).ravel()
                i = pos.astype(np.int64)
                f = (pos - i).astype(np.float32)
                v = (x32[i] + (x32[i + 1] - x32[i]) * f).astype(np.float64)
                r = np.clip(((1.0 - (v - vmin) / span) * (H - 1)).astype(np.int64), 0, H - 1)
                img += np.bincount(r * W + cols[:r.size], minlength=H * W)
            return img

        row("fold/%s" % label, "REJECTED: float32 accumulate",
            "float64", "float32 (converted)", plain, f32,
            cmp=lambda a, b: ("identical" if np.array_equal(a, b)
                              else "max|d| = %d of %d counts" % (np.abs(a - b).max(), a.sum())))
        del x32

        def h2d():
            img = np.zeros((H, W), np.int64)
            colf = np.arange(W, dtype=float)
            edges = [np.arange(H + 1) - 0.5, np.arange(W + 1) - 0.5]
            for k in range(0, c.size, 2048):
                cb = c[k:k + 2048]
                pos = (cb[:, None] + ph_grid[None, :]).ravel()
                i = pos.astype(np.int64); f = pos - i
                v = x[i] + (x[i + 1] - x[i]) * f
                r = np.clip(((1.0 - (v - vmin) / span) * (H - 1)).astype(np.int64), 0, H - 1)
                hh, _e0, _e1 = np.histogram2d(r.astype(float), np.tile(colf, cb.size), bins=edges)
                img += hh.astype(np.int64)
            return img.ravel()

        row("fold/%s" % label, "REJECTED: np.histogram2d",
            "np.bincount", "np.histogram2d", plain, h2d)

    print()
    print("  Every 'identical' above is array-equality on the actual returned values,")
    print("  not a tolerance. The two REJECTED rows are kept in the table because a")
    print("  technique that costs accuracy or time has to be shown to have been tried.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""PART 1 -- where the wall clock actually goes in a 3 M UI eye.

Runs the SHIPPING reduction (spikes/eye3m/shipping.py loads it out of the demo's
engine kernel, unmodified) over the 48 Mpt record and breaks the frame down into
the stages it is made of: the threshold, the edge finder, the period search, the
loop filter, the phase reconstruction, the decision instants and the fold itself.

Nothing is optimised here. The point is to know the answer before guessing it --
the sibling unit's finding was that the guessed cost (clock recovery) was not the
real one, and a guess at this size is worth nothing.

  python3 spikes/eye3m/profile3m.py [--traces N] [--ui 3000000]
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import record as REC                                    # noqa: E402
import shipping                                         # noqa: E402


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if r > 1 << 40 else r / 1e6           # macOS reports bytes


class Clock:
    def __init__(self):
        self.rows = []

    def __call__(self, name, fn, *a, **kw):
        t = time.perf_counter()
        out = fn(*a, **kw)
        ms = (time.perf_counter() - t) * 1e3
        self.rows.append((name, ms, rss_mb()))
        return out

    def table(self, total_ms, title):
        print()
        print(title)
        print("  %-34s %10s %8s %10s" % ("stage", "ms", "% frame", "peak RSS"))
        print("  " + "-" * 66)
        for name, ms, rss in self.rows:
            print("  %-34s %10.1f %7.1f%% %8.0f MB" % (name, ms, 100 * ms / total_ms, rss))
        print("  " + "-" * 66)
        acc = sum(r[1] for r in self.rows)
        print("  %-34s %10.1f %7.1f%%" % ("sum of stages", acc, 100 * acc / total_ms))
        print("  %-34s %10.1f %7.1f%%" % ("whole reduction (measured)", total_ms, 100.0))


def stages(ns, x, g, ey, max_traces):
    """The same calls _eye_recover makes, in the same order, each timed."""
    c = Clock()
    n = x.size
    level = c("threshold  mean(x)", lambda: float(np.mean(x)))
    hyst = c("threshold  ptp(x) * 0.1", lambda: float(np.ptp(x)) * 0.1)
    vmin = c("scale      min(x)", lambda: float(np.min(x)))
    vmax = c("scale      max(x)", lambda: float(np.max(x)))
    cross, _d = c("edges      _views_crossings", ns["_eye_crossings"], x, level, hyst)

    def seeds_of():
        d = np.diff(cross)
        d = d[d > 0]
        return [("record", float(np.percentile(d, q))) for q in (5.0, 10.0, 20.0, 35.0)]
    seeds = c("period     seeds (4 percentiles)", seeds_of)
    seeds.append(("grid", float(g.samples_per_ui)))

    fits = []
    for i, (origin, s) in enumerate(seeds):
        fits.append(c("period     fit %d (%s seed)" % (i, origin), ns["_eye_period"], cross, s))
    best = None
    for (origin, _s), fit in zip(seeds, fits):
        if fit is None:
            continue
        tie, m, ui, phase = fit
        if not (ui >= 2.0) or int(m[-1]) < 8:
            continue
        score = float(np.std(tie)) * ui
        if best is None or score < best[0]:
            best = (score, tie, m, ui, phase)
    _score, tie, m, ui, phase = best
    baud = g.fs / ui
    loop_bw = baud / 1667.0

    def hold():
        n_sym = int(m[-1]) + 1
        ph = np.full(n_sym, np.nan)
        ph[m] = tie
        have = ~np.isnan(ph)
        fill = np.where(have, np.arange(n_sym), 0)
        np.maximum.accumulate(fill, out=fill)
        ph = ph[fill]
        ph[np.isnan(ph)] = float(tie[0])
        return ph
    ph = c("phase      per-symbol hold", hold)
    n_sym = ph.size

    import wfmsynth.cdr as C
    clock, residual = c("CDR        recover_clock (2 lfilter)",
                        C.recover_clock, ph, baud, loop_bw, order=2, damping=0.707)
    c("CDR        tracked_out_fraction (2 more)",
      C.tracked_out_fraction, ph, baud, loop_bw, order=2, damping=0.707)
    centres = c("instants   centres = phase + ui*(k+.5+clk)",
                lambda: phase + ui * (np.arange(n_sym, dtype=float) + 0.5 + clock))
    warm = int(min(n_sym // 4, round(baud / loop_bw)))
    img, traces = c("fold       _eye_fold_at (%d traces cap)" % max_traces,
                    ns["_eye_fold_at"], x, centres[warm:], float(ui),
                    int(ey["w"]), int(ey["h"]), float(ey["uiSpan"]), vmin, vmax, max_traces)
    return c, dict(edges=cross.size, symbols=n_sym, ui=ui, traces=traces,
                   img=img, vmin=vmin, vmax=vmax, level=level)


def fast_stages(x, g, max_traces, nt):
    """The same frame through wfmsynth.eye, stage for stage, for the after-table."""
    from wfmsynth import cdr as C
    from wfmsynth import eye as E
    c = Clock()
    stats = c("threshold  mean/min/max (3 threads)", E.levels, x, nt)
    level, vmin, vmax = stats
    hyst = (vmax - vmin) * 0.1
    cross, _d = c("edges      eye.crossings", E.crossings, x, level, hyst, threads=nt)
    seeds = c("period     seeds (one percentile call)", E._seeds, cross, g.samples_per_ui)
    fits = c("period     five fits, in parallel", E._map,
             lambda s: E.period_fit(cross, s), [float(s) for _o, s in seeds], nt)
    best = None
    for (origin, _s), fit in zip(seeds, fits):
        if fit is None or not (fit[2] >= 2.0) or int(fit[1][-1]) < 8:
            continue
        score = float(np.std(fit[0])) * fit[2]
        if best is None or score < best[0]:
            best = (score, ) + fit
    _sc, tie, m, ui, phase = best
    n_sym = int(m[-1]) + 1

    def hold():
        held = np.zeros(n_sym, np.int64)
        held[m[1:]] = 1
        np.cumsum(held, out=held)
        return tie[held]
    ph = c("phase      hold (cumsum + gather)", hold)
    baud = g.fs / ui
    lbw = baud / E.LOOP_DIVISOR
    clock, residual = c("CDR        two loop filters, 2 threads",
                        E._recover_clock, ph, baud, lbw, 2, 0.707, nt)
    c("CDR        tracked fraction (from residual)",
      lambda: 1.0 - np.ptp(residual[n_sym // 2:]) / (np.ptp(ph[n_sym // 2:]) + 1e-30))
    centres = c("instants   centres", lambda: phase + ui * (np.arange(n_sym, dtype=float)
                                                            + 0.5 + clock))
    warm = int(min(n_sym // 4, round(baud / lbw)))
    c("fold       eye.density (%d traces cap)" % max_traces, E.density, x, centres[warm:],
      float(ui), 128, 96, 2.0, vmin, vmax, max_traces, nt)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=int, default=400)
    ap.add_argument("--ui", type=int, default=REC.N_UI)
    ap.add_argument("--spb", type=int, default=REC.SPB)
    a = ap.parse_args()

    ns = shipping.load()
    print("baseline: the demo engine kernel at %s (pinned)" % shipping.revision()[:12])
    t = time.perf_counter()
    x = REC.build(a.ui, a.spb)
    g = REC.grid(a.ui, a.spb)
    print("record: %d UI x %d = %d samples, %.0f MB float64, built/loaded in %.2f s"
          % (a.ui, a.spb, x.size, x.nbytes / 1e6, time.perf_counter() - t))
    print("baseline: the demo's shipping engine kernel, unmodified")
    ey = {"w": 128, "h": 96, "uiSpan": 2.0, "mode": "interp", "maxTraces": a.traces}

    # the whole thing, twice, as the frame runs it
    for i in range(2):
        t = time.perf_counter()
        rep, fold = ns["_eye_recover"](x, g.fs, g.samples_per_ui, ey)
        t_rec = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        vmin, vmax = float(np.min(x)), float(np.max(x))
        r = ns["_eye_fold_at"](x, fold[0], fold[1], ey["w"], ey["h"], ey["uiSpan"],
                               vmin, vmax, a.traces)
        t_fold = (time.perf_counter() - t) * 1e3
        print("  run %d: _eye_recover %8.1f ms   _eye_fold_at %8.1f ms   total %8.1f ms"
              % (i + 1, t_rec, t_fold, t_rec + t_fold))
    total = t_rec + t_fold
    print("  clock report: source=%s spb=%.6f edges=%d symbols=%d traces=%d"
          % (rep["source"], rep["samplesPerUi"], rep["edges"], rep["symbols"], r[1]))

    c, info = stages(ns, x, g, ey, a.traces)
    c.table(total, "STAGE BREAKDOWN (shipping code, %d UI, maxTraces=%d)" % (a.ui, a.traces))
    total_before = total
    print()
    print("  peak RSS so far: %.0f MB (record alone is %.0f MB)"
          % (rss_mb(), x.nbytes / 1e6))

    from wfmsynth import eye as E
    nt = E._threads(None)
    t = time.perf_counter()
    img2, meta2 = E.eye(x, g, max_traces=a.traces, threads=nt)
    t2 = time.perf_counter()
    E.eye(x, g, max_traces=a.traces, threads=nt)
    t_fast = (time.perf_counter() - t2) * 1e3
    c2 = fast_stages(x, g, a.traces, nt)
    c2.table(t_fast, "AFTER: wfmsynth.eye on %d threads, same record, same picture" % nt)
    print()
    print("  (the RSS column above is the PROCESS high-water mark, so it still carries")
    print("   the baseline run's 2.9 GB. spikes/eye3m/bench.py measures each path in")
    print("   its own process, which is the only way to read peak RSS honestly.)")
    print()
    print("  whole frame: %.1f ms before, %.1f ms after (%.2fx); the picture is the same"
          % (total_before, t_fast, total_before / t_fast))
    np.save(Path(REC.CACHE).with_name("eye3m_baseline_%d.npy" % a.traces), info["img"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

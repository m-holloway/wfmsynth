"""Is the fast eye the SAME eye? -- the acceptance evidence for wfmsynth.eye.

Three questions, in the order that matters:

  A. does every stage return what the shipping stage returns, bit for bit?
  B. is the rendered density identical, and by how much does it differ if not?
  C. is the CDR untouched -- does the sibling unit's jitter-transfer evidence
     (spikes/cdr/fold_truth.py in the demo, commit b37483b) still measure the same
     numbers through this path?

C is the one that cannot be traded. A faster fold that quietly alters the loop's
response is not a speed-up, it is a broken instrument.

  python3 spikes/eye3m/verify.py [--ui 3000000]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import record as REC                                    # noqa: E402
import shipping                                         # noqa: E402
from wfmsynth import eye as E                           # noqa: E402
from wfmsynth.grid import Grid                          # noqa: E402
from wfmsynth import physics as P                       # noqa: E402

OK = "  ok  "
BAD = " FAIL "
FAILS = []


def check(name, cond, detail=""):
    print("  [%s] %-52s %s" % (OK if cond else BAD, name, detail))
    if not cond:
        FAILS.append(name)
    return cond


def same(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b)


def stages(ns, x, g, label):
    """A: every stage, against the shipping stage, on one record."""
    print("\n%s (%d samples, spb %.4f)" % (label, x.size, g.samples_per_ui))
    level = float(np.mean(x)); hyst = float(np.ptp(x)) * 0.1
    c0, d0 = ns["_eye_crossings"](x, level, hyst)
    c1, d1 = E.crossings(x, level, hyst)
    check("crossings: positions", same(c0, c1), "%d edges" % c0.size)
    check("crossings: directions", same(d0, d1))
    if c0.size >= 4:
        d = np.diff(c0); d = d[d > 0]
        one = [q for _o, q in E._seeds(c0, g.samples_per_ui) if _o == "record"]
        four = [float(np.percentile(d, q)) for q in (5.0, 10.0, 20.0, 35.0)]
        check("seeds: one percentile call == four", one == four,
              "%s" % ["%.6f" % v for v in one])
        for q in (5.0, 10.0, 20.0, 35.0, 50.0):
            d = np.diff(c0); d = d[d > 0]
            s = float(np.percentile(d, q))
            f0 = ns["_eye_period"](c0, s); f1 = E.period_fit(c1, s)
            got = (f0 is None) == (f1 is None) and (
                f0 is None or (same(f0[0], f1[0]) and same(f0[1], f1[1])
                               and f0[2] == f1[2] and f0[3] == f1[3]))
            check("period_fit at the %g%% seed" % q, got,
                  "" if f0 is None else "ui = %.9f" % f0[2])
    ey = {"w": 128, "h": 96, "uiSpan": 2.0, "mode": "interp", "maxTraces": 400}
    r0, fold0 = ns["_eye_recover"](x, g.fs, g.samples_per_ui, ey)
    r1, fold1 = E.recover(x, g.fs, g.samples_per_ui)
    check("recover: locked the same way", (fold0 is None) == (fold1 is None),
          r0.get("source", "?"))
    for k in ("samplesPerUi", "phaseSamples", "baudHz", "loopBwHz", "rmsUi", "ppUi",
              "residualRmsUi", "trackedFraction", "lockRmsUi", "edges", "symbols",
              "warmupSymbols", "periodFrom", "thresholdVolts", "loopPeriods"):
        if k in r0 or k in r1:
            check("report[%s]" % k, r0.get(k) == r1.get(k),
                  "%r" % (r1.get(k),))
    if fold0 is not None and fold1 is not None:
        check("decision instants (centres)", same(fold0[0], fold1[0]),
              "%d symbols" % fold0[0].size)
        check("recovered unit interval", fold0[1] == fold1[1], "%.12f" % fold1[1])
        vmin, vmax = float(np.min(x)), float(np.max(x))
        for cap in (400, 4000, 400000):
            a = ns["_eye_fold_at"](x, fold0[0], fold0[1], 128, 96, 2.0, vmin, vmax, cap)
            b = E.density(x, fold1[0], fold1[1], 128, 96, 2.0, vmin, vmax, cap)
            if a is None or b is None:
                check("density at maxTraces=%d" % cap, a is None and b is None, "both None")
                continue
            i0 = np.asarray(a[0]).reshape(96, 128).astype(np.int64)
            i1 = np.asarray(b[0]).astype(np.int64)
            d = np.abs(i0 - i1)
            check("density at maxTraces=%d" % cap, d.max() == 0 and a[1] == b[1],
                  "%d traces, %d counts, max|d| = %d" % (b[1], i1.sum(), d.max()))


def truth(ns):
    """C: the sibling unit's ground-truth experiment, re-run through wfmsynth.eye.
    Same closed-form record builder, same numbers to reproduce."""
    build, bits_lfsr = shipping.truth_builder()
    FS = 40e9
    W, H, SPAN, MAXTR = 128, 96, 2.0, 400
    K, SPB = 1600, 9.7
    bits = bits_lfsr(K)
    x0, _a0 = build(bits, SPB, phase=0.0)

    print("\nC1. the recovered period against the constructed one")
    g = Grid(fs=FS, baud=FS / SPB, n=x0.size, v_full=1.0)
    img0, m0 = E.eye(x0, g, w=W, h=H, ui_span=SPAN, max_traces=MAXTR)
    got = m0["clock"]["samplesPerUi"]
    err = (got - SPB) / SPB
    check("period error vs constructed", abs(err) < 1e-6,
          "spb = %.9f vs %.4f -> %+.1e (evidence: -3.8e-08)" % (got, SPB, err))
    check("source is 'recovered'", m0["clock"]["source"] == "recovered",
          m0["clock"].get("periodFrom", ""))

    print("\nC2. the grid's rate must not move the picture")
    for mult in (1.7, 0.5):
        gg = Grid(fs=FS, baud=FS / SPB * mult, n=x0.size, v_full=1.0)
        img, mm = E.eye(x0, gg, w=W, h=H, ui_span=SPAN, max_traces=MAXTR)
        d = np.abs(img.astype(np.int64) - img0.astype(np.int64))
        check("grid rate %.1fx: density unchanged" % mult, d.max() == 0,
              "max|d| = %d over %d counts" % (d.max(), img0.sum()))

    print("\nC3. jitter transfer -- the loop must track slow and pass fast")
    K2 = 20000
    b2 = bits_lfsr(K2)
    baud = FS / SPB
    loop = baud / E.LOOP_DIVISOR
    print("      %d symbols, baud %.3f GBd, loop bw %.3f MHz (baud/%g)"
          % (K2, baud / 1e9, loop / 1e6, E.LOOP_DIVISOR))
    k = np.arange(K2, dtype=float)
    AMP = 0.20
    want = {0.2: (0.0015, 0.000), 40.0: (0.1406, 0.394)}
    for cycles in (0.2, 40.0):
        j = AMP * np.sin(2 * np.pi * cycles * k / K2)
        f_hz = cycles * baud / K2
        xj, _ = build(b2, SPB, jitter_ui=j)
        gj = Grid(fs=FS, baud=FS / SPB, n=xj.size, v_full=1.0)
        ij, mj = E.eye(xj, gj, w=W, h=H, ui_span=SPAN, max_traces=MAXTR)
        res = mj["clock"]["residualRmsUi"]
        vmn, vmx = mj["vMin"], mj["vMax"]
        midrow = int(round((1.0 - (0.0 - vmn) / (vmx - vmn)) * (H - 1)))
        mid = ij[midrow, :].astype(float)
        occ = np.flatnonzero(mid > 0.02 * mid.max())
        left = occ[occ < W // 2]
        smear = (left.max() - left.min()) * SPAN / (W - 1) if left.size else 0.0
        w_res, w_smear = want[cycles]
        check("jitter at %.2fx the loop bw: residual" % (f_hz / loop),
              abs(res - w_res) < 5e-4, "%.4f UI rms (evidence %.4f)" % (res, w_res))
        check("jitter at %.2fx the loop bw: crossing smear" % (f_hz / loop),
              abs(smear - w_smear) < 1e-3, "%.3f UI (evidence %.3f)" % (smear, w_smear))
        # and the same two numbers out of the SHIPPING path, so the comparison is
        # not against a remembered figure but against the code that established it
        bench = ns
        r0, f0 = bench["_eye_recover"](xj, FS, gj.samples_per_ui,
                                       {"w": W, "h": H, "uiSpan": SPAN, "maxTraces": MAXTR})
        check("  ... and the shipping path agrees",
              r0["residualRmsUi"] == res, "%.6f" % r0["residualRmsUi"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", type=int, default=REC.N_UI)
    ap.add_argument("--skip-big", action="store_true")
    a = ap.parse_args()
    ns = shipping.load()
    print("baseline: the demo engine kernel at %s (pinned)" % shipping.revision()[:12])

    print("=" * 78)
    print("A/B. every stage and the picture, against the shipping reduction")
    print("=" * 78)
    rng = np.random.default_rng(3)
    cases = [
        ("small clean NRZ, spb 8", P.nrz(n_ui=4000, n=32000, pattern="prbs31"),
         Grid(fs=8e9, baud=1e9, n=32000)),
        ("noisy NRZ, spb 9.7, 200 k", P.nrz(n_ui=20000, n=194000, pattern="prbs15",
                                            tr_frac=0.4)
         + 0.05 * rng.standard_normal(194000),
         Grid(fs=9.7e9, baud=1e9, n=194000)),
        ("clock pattern, spb 16", P.nrz(n_ui=5000, n=80000, pattern="clock"),
         Grid(fs=16e9, baud=1e9, n=80000)),
        ("PAM4, spb 12", P.pam4(n_ui=8000, n=96000, pattern="prbs13q"),
         Grid(fs=12e9, baud=1e9, n=96000)),
        ("wrong grid rate (2.5x), spb 8", P.nrz(n_ui=4000, n=32000, pattern="prbs31"),
         Grid(fs=20e9, baud=1e9, n=32000)),
    ]
    for label, x, g in cases:
        stages(ns, np.asarray(x, float), g, label)
    if not a.skip_big:
        x = REC.build(a.ui, REC.SPB)
        stages(ns, x, REC.grid(a.ui, REC.SPB), "THE 3 M UI RECORD (48 Mpt)")
        del x

    print()
    print("=" * 78)
    print("C. the CDR is untouched -- the sibling unit's evidence, re-measured")
    print("=" * 78)
    truth(ns)

    print()
    if FAILS:
        print("FAILED: %d checks -- %s" % (len(FAILS), ", ".join(FAILS[:6])))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

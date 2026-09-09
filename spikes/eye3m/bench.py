"""PART 2 -- before and after, at 3 M UI, with the peak RSS beside the wall clock.

Each path runs in its OWN process, because peak RSS is a high-water mark: measure
the fast path after the slow one in the same process and it inherits the slow one's
2.9 GB and reports it as its own.

  python3 spikes/eye3m/bench.py                 both paths, both trace caps
  python3 spikes/eye3m/bench.py --mode fast     one path, in this process
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import record as REC                                    # noqa: E402

CAPS = (400, 3_000_000)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def run_shipping(x, g, caps, reps):
    import shipping
    ns = shipping.load()
    out = {}
    for cap in caps:
        ey = {"w": 128, "h": 96, "uiSpan": 2.0, "mode": "interp", "maxTraces": cap}
        best = None
        for _ in range(reps):
            t = time.perf_counter()
            v_min = float(np.min(x)); v_max = float(np.max(x))
            rep, fold = ns["_eye_recover"](x, g.fs, g.samples_per_ui, ey)
            t_rec = time.perf_counter() - t
            t2 = time.perf_counter()
            img, traces = ns["_eye_fold_at"](x, fold[0], fold[1], 128, 96, 2.0,
                                             v_min, v_max, cap)
            t_fold = time.perf_counter() - t2
            tot = time.perf_counter() - t
            if best is None or tot < best[0]:
                best = (tot, t_rec, t_fold, traces, np.asarray(img).reshape(96, 128).copy())
        out[cap] = best
    return out


def run_fast(x, g, caps, reps, threads=None):
    from wfmsynth import eye as E
    out = {}
    for cap in caps:
        best = None
        for _ in range(reps):
            t = time.perf_counter()
            stats = E.levels(x, threads)
            v_min, v_max = stats[1], stats[2]
            rep, fold = E.recover(x, g.fs, g.samples_per_ui, threads=threads, stats=stats)
            t_rec = time.perf_counter() - t
            t2 = time.perf_counter()
            img, traces = E.density(x, fold[0], fold[1], 128, 96, 2.0, v_min, v_max,
                                    cap, threads=threads)
            t_fold = time.perf_counter() - t2
            tot = time.perf_counter() - t
            if best is None or tot < best[0]:
                best = (tot, t_rec, t_fold, traces, np.asarray(img).copy())
        out[cap] = best
    return out


def child(mode, ui, spb, reps, threads):
    x = REC.build(ui, spb)
    g = REC.grid(ui, spb)
    rss_load = rss_mb()
    caps = [c for c in CAPS]
    fn = run_shipping if mode == "shipping" else run_fast
    kw = {} if mode == "shipping" else {"threads": threads}
    out = fn(x, g, caps, reps, **kw)
    res = {"mode": mode, "threads": threads, "rssAfterLoad": rss_load, "peakRss": rss_mb(),
           "caps": {}}
    for cap, (tot, t_rec, t_fold, traces, img) in out.items():
        res["caps"][str(cap)] = {"totalMs": tot * 1e3, "recoverMs": t_rec * 1e3,
                                 "foldMs": t_fold * 1e3, "traces": int(traces),
                                 "sum": int(img.sum()),
                                 "sha": int(np.frombuffer(img.astype(np.int64).tobytes(),
                                                          np.int64).sum())}
        np.save(Path(REC.CACHE).with_name("eye3m_%s_%d.npy" % (mode, cap)), img)
    print("RESULT " + json.dumps(res))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("shipping", "fast"), default=None)
    ap.add_argument("--ui", type=int, default=REC.N_UI)
    ap.add_argument("--spb", type=int, default=REC.SPB)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.mode:
        return child(a.mode, a.ui, a.spb, a.reps, a.threads)

    import shipping
    print("baseline: the demo engine kernel at %s (pinned)" % shipping.revision()[:12])
    REC.build(a.ui, a.spb)                     # warm the cache before either child
    runs = {}
    for mode, threads in (("shipping", None), ("fast", 1), ("fast", None)):
        cmd = [sys.executable, str(HERE / "bench.py"), "--mode", mode,
               "--ui", str(a.ui), "--spb", str(a.spb), "--reps", str(a.reps)]
        if threads:
            cmd += ["--threads", str(threads)]
        env = dict(os.environ, PYTHONPATH=str(HERE.parents[1]))
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(r.stdout, r.stderr)
            raise SystemExit("child %s failed" % mode)
        runs[(mode, threads)] = json.loads(line[0][7:])

    print("=" * 84)
    print("3 M UI (%d samples, %.0f MB float64) -- best of %d, each path in its own process"
          % (a.ui * a.spb, a.ui * a.spb * 8 / 1e6, a.reps))
    print("=" * 84)
    for cap in CAPS:
        print("\nmaxTraces = %d" % cap)
        print("  %-28s %10s %10s %10s %10s %9s"
              % ("path", "recover", "fold", "TOTAL", "peak RSS", "traces"))
        base = runs[("shipping", None)]["caps"][str(cap)]["totalMs"]
        for key, label in ((("shipping", None), "shipping (demo kernel)"),
                           (("fast", 1), "wfmsynth.eye, 1 thread"),
                           (("fast", None), "wfmsynth.eye, threaded")):
            r = runs[key]; c = r["caps"][str(cap)]
            print("  %-28s %8.1fms %8.1fms %8.1fms %8.0fMB %9d   %.2fx"
                  % (label, c["recoverMs"], c["foldMs"], c["totalMs"], r["peakRss"],
                     c["traces"], base / c["totalMs"]))
    print()
    ref = None
    for key in (("shipping", None), ("fast", 1), ("fast", None)):
        for cap in CAPS:
            img = np.load(Path(REC.CACHE).with_name("eye3m_%s_%d.npy"
                                                    % (key[0], cap))).astype(np.int64)
            if key == ("shipping", None):
                np.save(Path(REC.CACHE).with_name("ref_%d.npy" % cap), img)
                continue
            r0 = np.load(Path(REC.CACHE).with_name("ref_%d.npy" % cap)).astype(np.int64)
            d = np.abs(img - r0)
            print("  density %s threads=%s at maxTraces=%-8d max|d| = %d over %d counts"
                  % (key[0], key[1], cap, d.max(), r0.sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

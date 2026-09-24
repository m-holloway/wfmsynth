#!/usr/bin/env python3
"""Time and peak-memory benchmark for long-record rendering, with a regression gate.

WHY PEAK MEMORY IS THE PRIMARY GATE AND WALL TIME IS THE LOOSE ONE
------------------------------------------------------------------
Peak memory is a deterministic function of the code: the same case on the same input allocates
the same bytes on every host, so a tight bound on it catches a real change and never a noisy
one. Wall time on a shared CI runner is not -- a cold cache, a noisy neighbour or a throttled
core moves it tens of percent with the library unchanged, and a gate tuned tight enough to
catch a 20 % slowdown fires constantly on nothing and gets switched off within a fortnight.

So the two are gated differently, on purpose:

  * MEMORY  -- tight (`MEM_TOL`). A record-length allocation that was not there before is a
    design change, and it is exactly what makes a long render impossible in a GUI or a demo
    rather than merely slow.
  * TIME    -- loose (`TIME_TOL`), sized to catch an ALGORITHMIC regression (an O(n log n)
    stage becoming O(n^2), a vectorised loop becoming a Python one) rather than a constant
    factor. A 2.5x gate on a 4x-noisy quantity is not a tight gate that has been relaxed; it
    is the only kind of time gate that survives contact with a shared runner.
  * SCALING -- the structural check that does not depend on the host's speed at all. Timing
    the same case at n and 2n and taking the ratio divides the host out: O(n) gives ~2.0,
    O(n log n) ~2.1, and O(n^2) gives ~4.0. `--check` flags a ratio above `SCALE_MAX`, which
    is the one way to catch a complexity regression without trusting the clock's absolute value.

USAGE
    python3 tools/benchmark.py                  # human table, default sizes
    python3 tools/benchmark.py --quick          # the cheap CI subset
    python3 tools/benchmark.py --json out.json  # machine readable
    python3 tools/benchmark.py --baseline       # rewrite tools/benchmark_baseline.json
    python3 tools/benchmark.py --check          # compare against the baseline, exit 1 on regression

The committed baseline is captured on the maintainer's machine; `--check` compares MEMORY and
SCALING against it (host-independent) and reports time as information rather than a gate unless
`--gate-time` is passed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wfmsynth import Grid, Signal, dataset                           # noqa: E402
from wfmsynth import physics as P, instrument as INST, resample as RS  # noqa: E402
from wfmsynth.acquire import AcquisitionProfile                      # noqa: E402

BASELINE = Path(__file__).resolve().parent / "benchmark_baseline.json"

MEM_TOL = 0.15      # peak memory may grow 15 % before it is a regression
# ...but only once there is enough of it to mean anything. A stage whose peak is a few tens of
# kilobytes (a symbol sequence, a bit array) swings 100 % on a rounding of the allocator, and a
# percentage gate on it is a gate that cries wolf. The quantity this gate exists to protect is
# RECORD-SCALE memory, so anything under a megabyte is reported and not failed.
MEM_FLOOR_MB = 1.0
TIME_TOL = 2.5      # wall time may grow 2.5x before it is a regression (runner noise is large)
SCALE_MAX = 2.8     # t(2n)/t(n) above this is a complexity regression, not a constant factor

FS = 80e9
BAUD = 10e9
SPUI = int(FS / BAUD)


# ------------------------------------------------------------------ the cases
# Each builder takes a record length and returns a zero-argument callable to measure. The
# INPUT is built outside that callable so the measurement isolates the stage under test.
def _grid(n):
    return Grid(fs=FS, baud=BAUD, n=n)


def _carrier(n):
    return np.asarray(P.nrz(n_ui=n // SPUI, tr_frac=0.4, seed=1, n=n), float)[:n]


def op_carrier_nrz(n):
    return lambda: P.nrz(n_ui=n // SPUI, tr_frac=0.4, seed=1, n=n)


def op_prbs(n):
    return lambda: P.prbs(13, n // SPUI)


def op_lossy_causal(n):
    x, g = _carrier(n), _grid(n)
    return lambda: P.lossy_channel(x, grid=g, length_in=8.0, tand=0.02, causal=True)


def op_lossy_flat(n):
    x, g = _carrier(n), _grid(n)
    return lambda: P.lossy_channel(x, grid=g, length_in=8.0, tand=0.02, causal=False)


def op_probe_loading(n):
    x, g = _carrier(n), _grid(n)
    return lambda: INST.probe_loading(x, g, c_load_f=0.5e-12, r_source=50.0)


def op_scope_bessel(n):
    x, g = _carrier(n), _grid(n)
    return lambda: INST.scope_bandwidth(x, g, bw_hz=25e9, kind="bessel", order=4)


def op_scope_brickwall(n):
    x, g = _carrier(n), _grid(n)
    return lambda: INST.scope_bandwidth(x, g, bw_hz=25e9, kind="brickwall")


def op_resample_at(n):
    """The acquisition/jitter/DCD hot path: arbitrary-position bandlimited interpolation."""
    x = _carrier(n)
    src = np.clip(np.arange(n, dtype=float) * (1.0 + 300e-6), 0, n - 1)
    return lambda: RS.resample_at(x, src)


def op_reflect(n):
    x, g = _carrier(n), _grid(n)
    return lambda: P.multi_reflection(x, grid=g, td_ps=200.0, gamma_s=0.06, gamma_l=0.06)


def op_ac_couple(n):
    x, g = _carrier(n), _grid(n)
    return lambda: P.ac_couple(x, grid=g, fc_hz=2e6)


def op_digitize(n):
    x, g = _carrier(n), _grid(n)
    return lambda: INST.digitize(x, grid=g, bits=8, clip_full_scale=1.2)


def chain_serial_link(n):
    """A serial-link render: source -> channel -> reflections -> front end -> converter."""
    def run():
        return (Signal(seed=1, grid=_grid(n))
                .carrier("nrz", n_ui=n // SPUI, pattern="prbs13", tr_frac=0.4, causal=True)
                .lossy(length_in=8.0, tand=0.02, causal=True)
                .reflect(td_ps=200.0, gamma_s=0.06, gamma_l=0.06)
                .scope(bw_hz=25e9, kind="bessel", order=4)
                .digitize(bits=8, full_scale=1.2)).waveform()
    return run


def chain_instrument(n):
    """The acquisition path -- the one that reaches `resample_at`, and the client's slow case."""
    def run():
        prof = AcquisitionProfile(sample_rate_hz=FS / 2.0, record_length=n // 2,
                                  input_bandwidth_hz=25e9, enob=7.0)
        return (Signal(seed=1, grid=_grid(n))
                .carrier("nrz", n_ui=n // SPUI, pattern="prbs13", tr_frac=0.4, causal=True)
                .lossy(length_in=8.0, tand=0.02, causal=True)
                .probe(c_load_f=0.5e-12, r_source=50.0)
                .acquire(prof)).waveform()
    return run


def chain_analog(n):
    """A non-serial source, so a regression in the analog path is visible too."""
    def run():
        return (Signal(seed=1, grid=_grid(n))
                .carrier("two_tone", f1_hz=1e9, f2_hz=1.1e9)
                .lossy(length_in=4.0, tand=0.02, causal=True)
                .scope(bw_hz=25e9, kind="bessel", order=4)).waveform()
    return run


def batch_dataset(n):
    """Many records through ONE channel. Guards two things a single-record case cannot see:
    `dataset` streaming each record into the output array instead of collecting them all first
    (which held the whole set twice, at float64 and again at float32), and the minimum-phase
    response being reused across the batch rather than rebuilt per record."""
    # Enough records that the STACKED ARRAY dominates the peak rather than any one
    # record's rendering temporaries -- otherwise this case would not actually be
    # sensitive to how the set is assembled, which is the thing it is here to guard.
    records = 32
    per = max(n // records, 1 << 11)

    def build(rng):
        s = int(rng.integers(1 << 30))
        return (Signal(seed=s, grid=_grid(per))
                .carrier("nrz", n_ui=per // SPUI, pattern="prbs13", tr_frac=0.4,
                         causal=True, seed=s)
                .lossy(length_in=8.0, tand=0.02, causal=True))

    def run():
        return dataset(build, records)
    return run


CASES = {
    "op:carrier_nrz": op_carrier_nrz,
    "op:prbs": op_prbs,
    "op:lossy_causal": op_lossy_causal,
    "op:lossy_flat": op_lossy_flat,
    "op:probe_loading": op_probe_loading,
    "op:scope_bessel": op_scope_bessel,
    "op:scope_brickwall": op_scope_brickwall,
    "op:resample_at": op_resample_at,
    "op:reflect": op_reflect,
    "op:ac_couple": op_ac_couple,
    "op:digitize": op_digitize,
    "chain:serial_link": chain_serial_link,
    "chain:instrument": chain_instrument,
    "chain:analog": chain_analog,
    "batch:dataset": batch_dataset,
}

# The CI subset: everything that is cheap at the quick size. `resample_at` and the chains that
# reach it are the slowest, and they are exactly the ones worth gating, so they stay in.
QUICK_SIZES = (1 << 16, 1 << 17)
FULL_SIZES = (1 << 20, 1 << 21)


# ------------------------------------------------------------------ measurement
def measure(fn, repeats=3):
    """Best-of-`repeats` wall time, and peak memory from a separate untimed run.

    Memory is measured on its own pass because `tracemalloc` itself costs time; timing under
    it would report the tracer, not the code."""
    fn()                                                   # warm caches, JIT-free but fair
    best = min(_once(fn) for _ in range(repeats))
    tracemalloc.start()
    fn()
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return best, peak


def _once(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def run(names, sizes, repeats=3):
    out = {}
    for name in names:
        builder = CASES[name]
        entry = {}
        for n in sizes:
            fn = builder(n)
            t, mem = measure(fn, repeats)
            entry[str(n)] = {"time_s": t, "peak_mb": mem / 1e6,
                             "peak_x_record": mem / (n * 8.0)}
        ts = [entry[str(n)]["time_s"] for n in sizes]
        entry["scale"] = ts[-1] / ts[0] if len(ts) > 1 and ts[0] > 0 else None
        out[name] = entry
    return out


# ------------------------------------------------------------------ reporting
def table(results, sizes):
    big = str(sizes[-1])
    print(f"{'case':<22} {'time @' + big:>13} {'peak MB':>9} {'xrecord':>8} {'scale':>6}")
    print("-" * 64)
    for name, e in results.items():
        r = e[big]
        sc = e.get("scale")
        print(f"{name:<22} {r['time_s']*1e3:10.1f} ms {r['peak_mb']:9.1f} "
              f"{r['peak_x_record']:8.1f} {sc if sc is None else f'{sc:6.2f}'}")


def check(results, base, sizes, gate_time=False, mem_tol=MEM_TOL):
    """Compare against the committed baseline. Returns a list of regression strings."""
    bad = []
    for name, e in results.items():
        b = base.get(name)
        if b is None:
            print(f"  (new case, not in baseline: {name})")
            continue
        for n in sizes:
            k = str(n)
            if k not in b:
                continue
            got, want = e[k]["peak_mb"], b[k]["peak_mb"]
            if max(got, want) < MEM_FLOOR_MB:
                continue
            if want > 0 and got > want * (1.0 + mem_tol):
                bad.append(f"{name} @n={n}: peak memory {got:.1f} MB vs baseline "
                           f"{want:.1f} MB (+{100*(got/want-1):.0f} %, tol {100*mem_tol:.0f} %)")
            if gate_time:
                gt, wt = e[k]["time_s"], b[k]["time_s"]
                if wt > 0 and gt > wt * TIME_TOL:
                    bad.append(f"{name} @n={n}: time {gt*1e3:.1f} ms vs baseline "
                               f"{wt*1e3:.1f} ms ({gt/wt:.1f}x, tol {TIME_TOL}x)")
        sc = e.get("scale")
        if sc is not None and sc > SCALE_MAX:
            bad.append(f"{name}: t(2n)/t(n) = {sc:.2f}, over {SCALE_MAX} -- "
                       f"this is a COMPLEXITY regression, not a constant factor")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="the cheap CI subset/sizes")
    ap.add_argument("--json", metavar="PATH", help="write results as JSON")
    ap.add_argument("--baseline", action="store_true", help="rewrite the committed baseline")
    ap.add_argument("--check", action="store_true", help="compare against the baseline")
    ap.add_argument("--gate-time", action="store_true", help="also gate on wall time")
    ap.add_argument("--only", metavar="SUBSTR", help="only cases containing SUBSTR")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mem-tol", type=float, default=MEM_TOL,
                    help="fractional peak-memory headroom before it is a regression. The "
                         "committed baseline is captured on one machine, so a CI runner on "
                         "another platform wants more slack than a local re-run does.")
    a = ap.parse_args()

    sizes = QUICK_SIZES if a.quick else FULL_SIZES
    names = [k for k in CASES if not a.only or a.only in k]
    if not names:
        print(f"no case matches {a.only!r}; have: {', '.join(CASES)}")
        return 2

    t0 = time.perf_counter()
    results = run(names, sizes, repeats=1 if a.quick else a.repeats)
    elapsed = time.perf_counter() - t0
    table(results, sizes)
    print(f"\n{len(names)} cases at n={sizes} in {elapsed:.1f} s")

    payload = {"sizes": list(sizes), "cases": results}
    if a.json:
        Path(a.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {a.json}")
    if a.baseline:
        BASELINE.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote baseline {BASELINE}")
    if a.check:
        if not BASELINE.exists():
            print(f"no baseline at {BASELINE}; run --baseline first")
            return 2
        base = json.loads(BASELINE.read_text())
        if list(base.get("sizes", [])) != list(sizes):
            print(f"baseline was captured at sizes {base.get('sizes')}, not {list(sizes)} -- "
                  f"compare like with like (use the same --quick/full mode)")
            return 2
        bad = check(results, base["cases"], sizes, gate_time=a.gate_time, mem_tol=a.mem_tol)
        if bad:
            print("\nREGRESSIONS:")
            for b in bad:
                print(f"  * {b}")
            return 1
        print("\nno regression against the baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

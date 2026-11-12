"""Three hundred relay closures, each captured on its own: what the population says.

Run from the repository root after ``pip install -e .``:

    python examples/segmented_events.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
A segmented acquisition of an event that happens rarely (`ws.Segments`). A relay switches a
12 V rail onto a capacitive load whenever a machine cycles, at irregular intervals of about a
second. A scope in segmented memory keeps a millisecond around each closure. Rendering the
whole run and triggering it would be five minutes at 5 MSa/s, 1.5 billion samples; here
each closure is rendered on its own 1.3 ms grid and the run costs n x window.

What changes from one closure to the next, each written where the number goes:

    ws.tol(0.985, 0.25, per="segment")  the contact's resistance, drawn for every closure
    ws.trend("ramp", rate_per_s=...)    the rail sagging as the battery runs down
    variants={"chatter": dict(p=0.03)}  now and then the contact chatters: five bounces, not two
    arrival=lognormal                   an operator, not a clock

THE QUESTION
------------
Given only the segments and their timestamps -- what an analysis tool receives -- how many
kinds of closure are there, which are rare, and what does the sag follow: the number of
closures (contact wear) or the time (the battery)? Then the recipe answers it, and the
acquisition with the sag removed (`without`) shows the classification did not depend on it.
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth import Signal

FS = 5e6                               # the RC edge is 2 us: 10 samples on it
GRID = ws.Grid(fs=FS, n=6500)          # one closure's render, 1.3 ms
RAIL_V, SAG_PER_S = 12.0, -4e-4        # the rail loses 0.04 % of itself per second
N = 300


def closure(bounces):
    """One closure: the rail, switched by a contact that bounces `bounces` times, into an RC
    load, seen through a probe with 20 mV of noise."""
    contact = ws.recipes.bounce(t_s=100e-6, gap_s=160e-6, n=bounces)
    return (Signal().carrier("dc", level=RAIL_V).gate(by=contact)
            .gain(by=ws.tol(0.985, 0.25, per="segment"))             # this closure's contact
            .gain(by=ws.trend("ramp", start=1.0, rate_per_s=SAG_PER_S))      # the battery
            .filter("butter", f_hz=175e3, order=1)
            .rx_noise(rms=0.02))


def acquisition():
    return ws.Segments(
        closure(2), n=N, seed=4, grid=GRID,
        arrival=dict(dist="lognormal", median_s=0.5, sigma=1.2),
        trigger=dict(pre_s=50e-6, post_s=1.0e-3, level=RAIL_V / 2, hysteresis=1.0),
        variants={"chatter": dict(p=0.03, chain=closure(5))})


def analyse(x, events):
    """What a tool sees: the windows and the trigger timestamps. Returns per closure its
    bounce count (rising crossings of half the rail), its settled level and its time."""
    rows = x.reshape(N, -1)
    fired = [e for e in events if e.kind == "trigger"]
    bounces = np.array([len(ws.trigger.find(r, FS, level=RAIL_V / 2, hysteresis=1.0)["t_s"])
                        for r in rows])
    level = rows[:, -1000:].mean(1)                     # the last 200 us: closed and settled
    stamp = np.array([e.params["timestamp_s"] for e in fired])
    return bounces, level, stamp


def fit(xs, y):
    """Least-squares line -> (slope, rms residual)."""
    a = np.vstack([xs, np.ones_like(xs)]).T
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return float(coef[0]), float(np.sqrt(np.mean((y - a @ coef) ** 2)))


def main():
    warnings.simplefilter("ignore")
    seg = acquisition()
    x, events = seg.realize()
    bounces, level, stamp = analyse(x, events)
    truth = seg.occurrences()

    print(f"THE ACQUISITION  {N} closures over {stamp[-1]:.0f} s, each a "
          f"{len(x) // N}-sample window; {GRID.n * N:,} samples rendered")
    print(f"   re-rollable: {', '.join(r for r in seg.roles() if not r.startswith('segment/'))}"
          f", and segment/<k>/rx_noise/5 for each closure ({len(seg.roles())} roles)")

    print("\n1. HOW MANY KINDS OF CLOSURE  (rising crossings of half the rail, per window)")
    kinds, counts = np.unique(bounces, return_counts=True)
    for k, c in zip(kinds, counts):
        print(f"   {k} closures of the contact: {c:4d} windows")
    rare = bounces == kinds[np.argmin(counts)]
    said = np.array([o["variant"] == "chatter" for o in truth])
    print(f"   the rare family is {rare.sum()} windows; the recipe's 'chatter' occurrences are "
          f"{said.sum()}, and they are the same windows: {bool(np.array_equal(rare, said))}")

    print("\n2. WHAT THE SAG FOLLOWS  (settled level, fitted against each)")
    k = np.arange(N, dtype=float)
    s_t, r_t = fit(stamp, level)
    s_k, r_k = fit(k, level)
    print(f"   against time      {s_t * 60e3:7.1f} mV/min, residual {r_t * 1e3:5.1f} mV rms")
    print(f"   against closures  {s_k * 1e5:7.1f} mV/100, residual {r_k * 1e3:5.1f} mV rms")
    drawn = np.array([o["values"]["vary/2/by" if o.get("variant") == "main"
                                  else "chatter/vary/2/by"] for o in truth])
    follows = "TIME" if r_t < r_k else "the COUNT"
    print(f"   it follows {follows}: against time the residual is the contact's own spread "
          f"({RAIL_V * drawn.std() * 1e3:.1f} mV rms drawn, +/-0.25 % of 0.985); against the "
          f"count the irregular arrivals add {np.sqrt(max(r_k ** 2 - r_t ** 2, 0)) * 1e3:.1f} "
          f"mV rms to it")
    print(f"   the recipe says {RAIL_V * 0.985 * SAG_PER_S * 60e3:.1f} mV/min at the load "
          f"(rate_per_s={SAG_PER_S:g} of {RAIL_V:g} V through the 0.985 contact)")
    rel = level / (RAIL_V * drawn * (1 + SAG_PER_S * np.array([o["arrival_s"] for o in truth])))
    print(f"   settled level / (rail x drawn contact x trend): {rel.mean():.5f} mean, "
          f"{rel.std():.5f} rms over the {N} closures")

    print("\n3. THE SAG REMOVED  (seg.without('trend/*'))")
    flat = seg.without("trend/*")
    xf, ef = flat.realize()
    bf, lf, sf = analyse(xf, ef)
    s_f, r_f = fit(sf, lf)
    kept = [(o["arrival_s"], o.get("variant"), o["values"].get("vary/2/by"))
            for o in flat.occurrences()] == [(o["arrival_s"], o.get("variant"),
                                              o["values"].get("vary/2/by")) for o in truth]
    print(f"   against time {s_f * 60e3:7.1f} mV/min (was {s_t * 60e3:.1f}); arrivals, "
          f"variants and contact draws {'identical' if kept else 'CHANGED'}; bounce counts "
          f"{'identical' if np.array_equal(bf, bounces) else 'CHANGED'}")
    print(f"   timestamps moved by at most {np.abs(sf - stamp).max() * 1e9:.0f} ns: from a "
          f"higher rail the RC edge reaches half of it sooner")

    print("\n4. ANOTHER DRAW OF WHICH CLOSURES CHATTER  (seg.contrast('variant'))")
    xv = seg.contrast("variant")
    bv = np.array([len(ws.trigger.find(r, FS, level=RAIL_V / 2, hysteresis=1.0)["t_s"])
                   for r in xv.reshape(N, -1)])
    both = (bv == kinds[0]) & ~rare
    same_rows = bool(np.array_equal(xv.reshape(N, -1)[both], x.reshape(N, -1)[both]))
    print(f"   {int((bv == kinds[-1]).sum())} chatter windows, at "
          f"{np.flatnonzero(bv == kinds[-1])[:5].tolist()}... (were "
          f"{np.flatnonzero(rare)[:5].tolist()}...); the {int(both.sum())} closures that are "
          f"ordinary in both are bit-identical: {same_rows}")

    again = ws.Segments.from_recipe(json.loads(seg.to_json()))
    same = bool(np.array_equal(again.waveform(), x))
    print(f"\nrecipe {seg.sha256()[:16]}: {len(seg.to_json())} bytes of JSON for "
          f"{len(x):,} samples, replayed {'bit-identically' if same else 'DIFFERENTLY'}")
    return {"kinds": dict(zip(kinds.tolist(), counts.tolist())),
            "rare_match": bool(np.array_equal(rare, said)), "follows": follows,
            "residual": (r_t, r_k), "per_min": (s_t * 60, s_f * 60), "kept": kept,
            "replayed": same}


if __name__ == "__main__":
    main()

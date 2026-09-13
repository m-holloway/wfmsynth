#!/usr/bin/env python3
"""Fill a seed archive's waveforms from its recipe, in place.

A SEED is a complete archive with every group, label and parameter present and the sample arrays
declared but unwritten. An unwritten array is valid and reads as its fill value, so a seed and a
filled archive differ in data and not in structure. That is what lets a small file travel and expand
into the full dataset here.

This writes ONLY sample data. It creates no group, no array, and resizes nothing, so the archive you
end up with is the one the seed described -- the labels, the decision instants, the parameters and
their provenance were already in it.

    python fill_zarr.py SEED.zarr recipes.json
    python fill_zarr.py SEED.zarr recipes.json --only pcie-4-clean-r0 --dry-run

Needs zarr and numpy alongside this library. Nothing else.

Each record's samples are stored as int16 counts, and the volts-per-count scale comes FROM THE SEED
rather than being recomputed from the rendered waveform. That matters: recomputing it from a
waveform's own peak would drift if the render differed by a least significant bit, and then every
count would differ while the volts stayed right. Reading the scale makes the counts reproducible.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from replay import render                                             # noqa: E402


def counts_for(volts, inc, offset=0.0):
    """Volts -> stored int16 counts, using the archive's own scale."""
    if not np.isfinite(inc) or inc == 0:
        raise ValueError("volts_per_count is zero or not finite; the seed cannot be filled")
    return np.rint((np.asarray(volts, float) - float(offset)) / float(inc)).astype("int16")


def main(argv=None):
    import zarr
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", help="the seed archive, filled in place")
    ap.add_argument("recipes", help="recipe document: id -> {ops, grid, ...}")
    ap.add_argument("--only", action="append", metavar="ID", help="fill just this record; repeatable")
    ap.add_argument("--dry-run", action="store_true",
                    help="render and compare, write nothing")
    a = ap.parse_args(argv)

    doc = json.load(open(a.recipes))
    root = zarr.open_group(a.store, mode="r" if a.dry_run else "a")
    samples = root["acquisition/samples"]

    ids = [str(v) for v in np.asarray(root["acquisition/records/record_id"][:])]
    inc = np.asarray(root["acquisition/channels/volts_per_count"][:], float)
    off = np.asarray(root["acquisition/channels/volt_offset"][:], float)
    scale = {i: (float(c), float(o)) for i, c, o in zip(ids, inc, off)}

    todo = [r for r in ids if r in doc and (not a.only or r in a.only)]
    missing = [r for r in ids if r not in doc]
    if missing:
        print(f"  {len(missing)} record(s) in the archive have no recipe, e.g. {missing[:3]}")

    print(f"{'record':30s} {'samples':>11s}  {'state':s}")
    filled = bad = 0
    for rid in todo:
        volts = render(doc[rid])
        if isinstance(volts, tuple):
            volts = volts[0]
        # `render` returns the NORMALISED waveform, full scale +/-1. An archive stores volts, so the
        # recipe's own `v_full` converts: a full-scale excursion spans v_full peak-to-peak, hence
        # half of it per side. Getting this wrong is silent -- the counts come out scaled by a
        # constant, every waveform still looks correct, and only a digest notices.
        v_full = float(doc[rid].get("grid", {}).get("v_full", 2.0))
        volts = np.asarray(volts, float).ravel() * 0.5 * v_full
        want = int(samples[rid].shape[0])
        if volts.size != want:
            print(f"{rid:30s} {volts.size:11,d}  LENGTH MISMATCH, archive declares {want:,}")
            bad += 1
            continue
        c, o = scale.get(rid, (None, 0.0))
        if c is None:
            print(f"{rid:30s} {volts.size:11,d}  no volts_per_count in the archive")
            bad += 1
            continue
        data = counts_for(volts, c, o)
        if not a.dry_run:
            samples[rid][:] = data
        filled += 1
        print(f"{rid:30s} {volts.size:11,d}  {'rendered' if a.dry_run else 'written'}")

    print(f"\n{filled} record(s) {'checked' if a.dry_run else 'filled'}, {bad} refused")
    if bad:
        print("A refusal means the recipe and the archive disagree. Nothing partial was written for it.")
        return 1
    if not filled:
        print("NOTHING WAS FILLED. Check --only, or that the recipe covers this archive.")
        return 2
    if not a.dry_run:
        print("Every sample array is written. Read the archive with your usual reader.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

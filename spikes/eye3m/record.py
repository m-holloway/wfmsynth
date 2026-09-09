"""The record every measurement in this directory is made on: 3 M unit intervals
at 16 samples/UI -- 48 Mpt, 384 MB as float64.

Built with the library's own carrier so the edge density, the run lengths and the
transition shape are what the eye actually gets handed: PRBS31 (the compliance
pattern -- it never repeats inside the record, so the long runs that starve an
edge finder are present), transmitter Rj and Pj at the symbol edge times, and
receiver noise added after shaping. Cached to disk because regenerating it is not
what is being timed.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from wfmsynth.grid import Grid
from wfmsynth import physics as P

N_UI = 3_000_000
SPB = 16
N = N_UI * SPB                      # 48 Mpt
BAUD = 16e9
FS = BAUD * SPB                     # 256 GS/s

CACHE = Path(os.environ.get("WFMSYNTH_SCRATCH", "/tmp")) / "eye3m_record.npy"


def grid(n_ui=N_UI, spb=SPB):
    return Grid(fs=BAUD * spb, baud=BAUD, n=n_ui * spb, v_full=1.0)


def build(n_ui=N_UI, spb=SPB, seed=7, noise=0.02, cache=True):
    """The 48 Mpt record, float64, as the reduction receives it."""
    n = int(n_ui) * int(spb)
    if cache and n_ui == N_UI and spb == SPB and CACHE.exists():
        return np.load(CACHE, mmap_mode=None)
    rng = np.random.default_rng(seed)
    g = grid(n_ui, spb)
    jit = P.Jitter(rj=0.02 * spb, pj=0.03 * spb, f_pj=900.0, dcd=0.01 * spb)
    x = P.nrz(n_ui=int(n_ui), n=n, pattern="prbs31", tr_frac=0.35, jitter=jit, rng=rng)
    if noise:
        x += rng.standard_normal(n) * noise
    if cache and n_ui == N_UI and spb == SPB:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE, x)
    return x


if __name__ == "__main__":
    import time
    t = time.perf_counter()
    x = build()
    print("record %d samples (%.0f MB) in %.2f s  ptp=%.3f  cache=%s"
          % (x.size, x.nbytes / 1e6, time.perf_counter() - t, np.ptp(x), CACHE))

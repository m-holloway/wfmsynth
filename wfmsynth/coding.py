"""
wfmsynth.coding — line coding & scrambling (DC balance, run-length control).

Raw PRBS is a stand-in for real transmitted bits: its running disparity random-walks (so the
DC content wanders) and it can have long runs. Real links run a line code or scrambler that
bounds both, which changes how the signal interacts with AC-coupling (#baseline wander) and
what an analyser locks to. Two primitives:

  * ``dc_balanced`` — the principle behind 8b/10b: each block is sent as-is or inverted
    (whichever pulls the running disparity toward zero), with a flag bit marking inversion.
    The accumulated disparity stays bounded → DC-balanced (a random-walking PRBS is not). It
    bounds disparity, not run length as tightly as a full 8b/10b table (logged as #45).
  * ``scramble_64b66b`` — the 64b/66b self-synchronous scrambler (x^58 + x^39 + 1) that real
    high-rate links use to break up patterns (2 sync bits per 64-bit block).
"""
from __future__ import annotations

import numpy as np


def dc_balanced(bits, block=8):
    """DC-balanced line code (8b/10b-style running-disparity block inversion). ``bits`` is a
    0/1 array; each ``block`` bits are emitted with a leading flag bit, inverted when that pulls
    the running disparity toward zero. Returns the coded 0/1 stream — bounded disparity, hence
    DC-balanced with bounded run length."""
    bits = np.asarray(bits).astype(int)
    out = []
    disp = 0
    for i in range(0, len(bits) - block + 1, block):
        blk = bits[i:i + block]
        a = np.concatenate([[0], blk])                    # as-is (flag 0)
        b = np.concatenate([[1], 1 - blk])                # inverted (flag 1)
        da = disp + int((2 * a - 1).sum())
        db = disp + int((2 * b - 1).sum())
        if abs(db) < abs(da):
            out.extend(b.tolist()); disp = db
        else:
            out.extend(a.tolist()); disp = da
    return np.array(out, dtype=int)


def scramble_64b66b(bits):
    """64b/66b self-synchronous scrambler (polynomial x^58 + x^39 + 1). Prepends a 2-bit sync
    header to each 64-bit block and scrambles the payload; whitens the spectrum without the
    lookup tables of 8b/10b. ``bits`` is a 0/1 array (truncated to whole 64-bit blocks)."""
    bits = np.asarray(bits).astype(int)
    state = np.ones(58, dtype=int)
    out = []
    nblk = len(bits) // 64
    for j in range(nblk):
        out.extend([0, 1])                                # sync header
        for b in bits[j * 64:(j + 1) * 64]:
            fb = state[57] ^ state[38]                    # x^58 + x^39
            s = b ^ fb
            out.append(int(s))
            state = np.roll(state, 1); state[0] = s
    return np.array(out, dtype=int)


def running_disparity(bits):
    """Cumulative running disparity of a 0/1 stream (sum of ±1). A bounded envelope means the
    code is DC-balanced; a random walk (raw PRBS) means it is not."""
    return np.cumsum(2 * np.asarray(bits).astype(int) - 1)


def max_run(bits):
    """Longest run of identical bits."""
    b = np.asarray(bits).astype(int)
    if len(b) == 0:
        return 0
    return int(np.max(np.diff(np.flatnonzero(np.concatenate(([1], np.diff(b) != 0, [1]))))))

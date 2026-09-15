"""wfmsynth.capture -- importing a real capture from disk as a `Signal` source.

A synthesized carrier's provenance is a polynomial, a seed and a length; a captured record's
provenance is the FILE, so this module gives that trade its own honesty gates rather than
pretending a captured record is just another array:

  * a missing file is a NAMED error, not zeros (`load_values` raises `FileNotFoundError` with
    the path in the message);
  * a recipe records a sha256 DIGEST OVER THE SAMPLE VALUES, not the file's bytes -- the same
    values re-saved through a different codec or dtype must not move the digest, only the
    numbers do (docs/ARCHITECTURE.md's archive-digest rule, applied here);
  * a file that no longer matches its recorded digest is a named error at replay time, not a
    silently different waveform under the same recipe.

v1 formats: ``.npy``, ``.npz`` (an array stored under the key ``"values"``), and a two-column
``.csv`` (time, value -- the value column is read; a header row bare CSV would confuse for
strictly numeric input is not supported, so cell 0 must be a number).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def digest(values):
    """sha256 over the sample VALUES, in one canonical dtype and byte order -- so re-saving the
    same numbers through a different codec, dtype or endianness does not move the digest."""
    v = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(v.tobytes()).hexdigest()


def load_values(path):
    """Read a capture file's samples. Raises with the path named on any failure -- a missing or
    unreadable file is an error to fix, never a record of zeros."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"capture: no such file {path!r}")
    suffix = p.suffix.lower()
    if suffix == ".npy":
        y = np.load(p)
    elif suffix == ".npz":
        z = np.load(p)
        if "values" not in z.files:
            raise ValueError(f"capture: {path!r} is an .npz with no 'values' array "
                             f"(has: {list(z.files)})")
        y = z["values"]
    elif suffix == ".csv":
        raw = np.loadtxt(p, delimiter=",")
        raw = np.atleast_2d(raw)
        y = raw[:, 1] if raw.shape[1] >= 2 else raw[:, 0]
    else:
        raise ValueError(f"capture: unsupported file type {suffix!r} for {path!r} "
                         f"(use .npy, .npz with a 'values' array, or a two-column .csv)")
    y = np.asarray(y, dtype=float).ravel()
    if y.size == 0:
        raise ValueError(f"capture: {path!r} contains no samples")
    return y

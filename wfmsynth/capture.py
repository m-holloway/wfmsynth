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

A COMPLEX capture (I/Q) is a complex ``.npy`` / ``.npz``, or a three-column ``.csv`` (time, I,
Q), and it is read only into a chain whose grid says ``domain="iq"``. Complex samples offered
to a real record are refused by name: casting them to float keeps the real part and discards
the rest, which numpy does with a warning and no error.

I/Q EXCHANGE formats -- interleaved binary, SigMF, CSV with named columns, MAT -- are read by
`wfmsynth/iqfile.py` through the same op (`capture(path=..., format=...)`), under the same
digest rule.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def digest(values):
    """sha256 over the sample VALUES, in one canonical dtype and byte order -- so re-saving the
    same numbers through a different codec, dtype or endianness does not move the digest."""
    # Complex samples digest as complex. Real ones digest exactly as they always have, so no
    # recorded real capture moves.
    v = np.asarray(values)
    v = np.ascontiguousarray(v.astype("<c16" if np.iscomplexobj(v) else "<f8"))
    return hashlib.sha256(v.tobytes()).hexdigest()


_COMPLEX_MSG = (
    "capture: {what} complex (I/Q) samples, and this chain's record is real (domain 'real'). "
    "A real record cannot hold them and nothing drops the imaginary part for you: start the "
    "chain on Signal(grid=Grid(..., domain='iq', fc_hz=...)), or pass the part you mean "
    "(values=x.real).")


def encode(values, domain="real"):
    """Samples -> the list a recipe holds: floats for a real capture (unchanged), and
    ``[[re, im], ...]`` for a complex one, since JSON has no complex number."""
    a = np.asarray(values)
    if np.iscomplexobj(a):
        if domain != "iq":
            raise ValueError(_COMPLEX_MSG.format(what="values= holds"))
        return [[float(v.real), float(v.imag)] for v in a.ravel()]
    return [float(v) for v in values]


def decode(values, domain="real"):
    """A recipe's `values` -> samples in the record's domain."""
    a = np.asarray(values)
    paired = np.iscomplexobj(a) or (a.ndim == 2 and a.shape[-1] == 2)
    if domain != "iq":
        if paired:
            raise ValueError(_COMPLEX_MSG.format(what="values= holds"))
        return np.asarray(values, float)
    from . import iq as IQ
    return IQ.as_complex(a)


_PICKLE_MSG = (
    "capture: {path!r} holds a PICKLED object array, not plain samples. Loading one executes "
    "code, so it is refused rather than read. If you produced this file, save the samples as a "
    "numeric array (np.save(path, np.asarray(values, float))); if you did not, do not load it."
)


def _load_plain(p, path):
    """`np.load` with pickling off, and a message that says what happened. numpy's own error
    ("Object arrays cannot be loaded when allow_pickle=False") is accurate and tells a caller
    of THIS library nothing about which file or what to do."""
    try:
        return np.load(p, allow_pickle=False)
    except ValueError as e:
        if "allow_pickle" in str(e):
            raise ValueError(_PICKLE_MSG.format(path=path)) from e
        raise


def load_values(path, domain="real"):
    """Read a capture file's samples. Raises with the path named on any failure -- a missing or
    unreadable file is an error to fix, never a record of zeros. `domain="iq"` reads complex
    samples; a complex file read as real is refused rather than cast."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"capture: no such file {path!r}")
    suffix = p.suffix.lower()
    # `allow_pickle=False` is numpy's own default since 1.16.3, and it is stated here anyway.
    # A .npy/.npz can carry a PICKLED object array, and unpickling runs code -- so a capture
    # file is an executable if it is loaded with pickling on. This library reads capture files
    # a user did not necessarily write (that is the point of `Signal.capture()`), so the flag
    # is written down rather than inherited: it survives a change of default, and it says to
    # the next reader that the restriction is deliberate rather than accidental.
    if suffix == ".npy":
        y = _load_plain(p, path)
    elif suffix == ".npz":
        z = _load_plain(p, path)
        if "values" not in z.files:
            raise ValueError(f"capture: {path!r} is an .npz with no 'values' array "
                             f"(has: {list(z.files)})")
        try:
            y = z["values"]
        except ValueError as e:                       # an object array inside the archive
            raise ValueError(_PICKLE_MSG.format(path=path)) from e
    elif suffix == ".csv":
        raw = np.loadtxt(p, delimiter=",")
        raw = np.atleast_2d(raw)
        if domain == "iq":
            if raw.shape[1] < 3:
                raise ValueError(f"capture: {path!r} has {raw.shape[1]} column(s); an I/Q "
                                 f".csv is three -- time, I, Q")
            y = raw[:, 1] + 1j * raw[:, 2]
        else:
            y = raw[:, 1] if raw.shape[1] >= 2 else raw[:, 0]
    else:
        raise ValueError(f"capture: unsupported file type {suffix!r} for {path!r} "
                         f"(use .npy, .npz with a 'values' array, or a two-column .csv)")
    if np.iscomplexobj(y) and domain != "iq":
        raise ValueError(_COMPLEX_MSG.format(what=f"{path!r} holds"))
    y = np.asarray(y, dtype=complex if domain == "iq" else float).ravel()
    if y.size == 0:
        raise ValueError(f"capture: {path!r} contains no samples")
    return y

"""wfmsynth.iqfile -- I/Q recordings in open formats, in and out.

    sig = ws.read_iq("rec", format="sigmf")                          # a chain that starts from it
    sig = ws.read_iq("rec.iq", format="iq_interleaved", dtype="int16", fs_hz=250e6, fc_hz=1e9)
    ws.to_sigmf("out", x, sig.stored_grid(x), events=ev)             # and back out
    ws.to_interleaved("out.iq", x, dtype="int16", scale=1e-4, events=ev)

`read_iq` is `Signal(grid=<the file's rate, length and centre>).capture(path=..., format=...)`;
the same `capture()` call works on a grid you built yourself. Four formats are read:

    "iq_interleaved"   raw I, Q, I, Q, ... ; you state `dtype` (int8 | int16 | int32 | float32 |
                       float64), `byteorder` ("little" | "big") and `scale`, because the file
                       states nothing
    "sigmf"            a `<name>.sigmf-meta` / `<name>.sigmf-data` pair; the datatype, sample
                       rate, centre frequency and annotations come from the metadata, and the
                       annotations come back as EVENTS
    "csv"              `columns=(i, q)`: two header names, or two 0-based column numbers
    "mat"              `variable="x"`: a complex array in a MATLAB file (up to v7.2)

and two are written: SigMF and interleaved binary. Vendor containers are not read; convert them
to one of these first.

WHAT A SAMPLE IS. An integer file holds counts; the record holds ``count * scale``. `scale` is
the value of ONE COUNT, and it defaults to ``1 / 2**(bits - 1)`` (full scale is +/-1). A float
file's default scale is 1. The scale that was used is always written into the recipe.

THE DIGEST RULE is `capture()`'s: the recipe records a sha256 over the sample VALUES, so the
same samples re-saved in another format still match, and `embed=True` puts the samples in the
recipe and needs no file afterwards.

WHAT IS CHECKED, AND WHAT IS NOT. Nothing in a headerless file says how to read it, so a wrong
`byteorder`, `dtype` or `scale` yields a perfectly well-formed array of wrong numbers. Two of
those are detectable from the bytes and are refused with the likely cause:

  * an INTEGER file read in the wrong byte order, by either of two signs, over files of at
    least `MIN_CHECKED` values (below that neither means anything and the check does not run):
      - ENTROPY. In a sampled signal the least significant byte is the busiest and the most
        significant the quietest; read backwards, that is reversed. Refused when the byte read
        as most significant carries more than `ENTROPY_MARGIN_BITS` more Shannon entropy than
        the byte read as least significant. This is the sign that catches a recording well
        below full scale, noise included. It is skipped for samples whose lowest bit never
        changes (left-justified or zero-padded data is legitimately quiet at the bottom).
      - ROUGHNESS, the rms sample-to-sample step over sqrt(2) times the rms about the mean (1
        for white samples, less for an oversampled signal). Refused when the other byte order
        reads at under `ROUGHNESS_RATIO` of the stated one's. This is the sign that catches a
        file normalised to full scale, where both bytes are busy.
    WHITE samples spread evenly over the whole integer range have neither sign and are not
    caught: nothing in those bytes distinguishes the two readings. (Gaussian noise normalised
    to its own peak is not that case; its top byte is still the quieter one, and it is caught.)
  * a FLOAT file read in the wrong byte order, or an integer file read as float: non-finite
    samples, or a median nonzero magnitude outside 1e-30 .. 1e30.

A file whose length is not a whole number of I/Q pairs of the stated dtype is refused too. A
wrong SCALE is not detectable (every scale gives a self-consistent record), so it is not
guessed at: `describe()` reports the peak in counts and as a fraction of full scale so you can
see it, and the export refuses a scale that would clip. `check=False` turns the byte-order
check off for a file that really is unusual.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from . import capture as CAP
from .events import Event

FORMATS = ("iq_interleaved", "sigmf", "csv", "mat")
DTYPES = ("int8", "int16", "int32", "float32", "float64")
BYTEORDERS = {"little": "<", "big": ">"}

MIN_CHECKED = 4096             # I and Q values (2048 complex samples) before the check runs
ENTROPY_MARGIN_BITS = 0.25     # measured headroom for both: tests/test_iq_files.py
ROUGHNESS_RATIO = 0.5

SIGMF_VERSION = "1.0.0"
_SIGMF_TYPE = {"int8": "i8", "int16": "i16", "int32": "i32", "float32": "f32", "float64": "f64"}
_SIGMF_DTYPE = {v: k for k, v in _SIGMF_TYPE.items()}
_DATATYPE = re.compile(r"^(c|r)(f32|f64|i32|i16|u32|u16|i8|u8)(_le|_be)?$")

# which layout parameter each format reads; anything else is refused, not ignored
_READS = {"iq_interleaved": ("dtype", "byteorder", "scale", "check"),
          "sigmf": ("scale", "check"),
          "csv": ("columns",),
          "mat": ("variable",)}
_LAYOUT = ("dtype", "byteorder", "scale", "columns", "variable", "check")


# ------------------------------------------------------------------------------- the layout
def _layout(format, layout):
    """Validate the layout parameters against the format; return only the ones given."""
    if format not in FORMATS:
        raise ValueError(f"capture: unknown format {format!r}. I/Q formats are "
                         f"{', '.join(FORMATS)}; a vendor container is converted to one of "
                         f"these first. (.npy / .npz files need no format=.)")
    out = {}
    for k, v in layout.items():
        if k not in _LAYOUT:
            raise ValueError(f"capture: unknown parameter {k!r}. A file's layout is stated "
                             f"with {', '.join(_LAYOUT)}.")
        if v is None or (k == "check" and v is True):
            continue
        if k not in _READS[format]:
            users = [f for f in FORMATS if k in _READS[f]]
            why = (" -- a SigMF recording states its own datatype and byte order in the "
                   "metadata" if format == "sigmf" and k in ("dtype", "byteorder") else "")
            raise ValueError(f"capture: format={format!r} does not read {k}={v!r}{why}. "
                             f"{k} belongs to format {' / '.join(repr(u) for u in users)}; "
                             f"drop it.")
        out[k] = v
    return out


def _np_dtype(dtype, byteorder, who="capture"):
    if dtype is None:
        raise ValueError(f"{who}: format='iq_interleaved' needs dtype= (one of "
                         f"{', '.join(DTYPES)}). The file does not say what it holds, and a "
                         f"guess would read as plausible numbers.")
    if dtype not in DTYPES:
        hint = (" Unsigned samples are offset binary: subtract the midpoint and pass the "
                "result as capture(values=...)." if str(dtype).startswith("uint") else "")
        raise ValueError(f"{who}: dtype={dtype!r} is not one of {', '.join(DTYPES)}.{hint}")
    order = "little" if byteorder is None else byteorder
    if order not in BYTEORDERS:
        raise ValueError(f"{who}: byteorder={byteorder!r} is not 'little' or 'big'.")
    return np.dtype(dtype).newbyteorder(BYTEORDERS[order]), order


def default_scale(dtype):
    """The value of one count when none is stated: full scale is +/-1 for an integer type."""
    dt = np.dtype(dtype)
    return 1.0 if dt.kind == "f" else 1.0 / float(1 << (8 * dt.itemsize - 1))


def _scale(scale, dtype, who="capture"):
    if scale is None:
        return default_scale(dtype)
    s = float(scale)
    if not (np.isfinite(s) and s > 0):
        raise ValueError(f"{who}: scale={scale!r} must be a positive number -- the value of "
                         f"one count (default {default_scale(dtype):g} for {dtype}).")
    return s


# -------------------------------------------------------------------- the byte-order check
def _entropy_bits(b):
    c = np.bincount(b, minlength=256)
    q = c[c > 0] / b.size
    return float(-(q * np.log2(q)).sum())


def _roughness(v):
    """rms step between consecutive samples of each of I and Q, over sqrt(2) * rms about the
    mean: 1 for white samples. None for a constant."""
    v = v.astype(np.float64)
    var = v[0::2].var() + v[1::2].var()
    if var == 0:
        return None
    step = np.mean(np.diff(v[0::2]) ** 2) + np.mean(np.diff(v[1::2]) ** 2)
    return float(np.sqrt(step / (2 * var)))


def byte_order_evidence(raw, dt):
    """What the bytes say about the stated byte order. Returns None when it has nothing to say,
    else a sentence naming the evidence (the caller turns it into the refusal)."""
    dt = np.dtype(dt)
    if dt.itemsize == 1 or raw.size < MIN_CHECKED:
        return None
    other = "big" if dt.byteorder in ("<", "=") else "little"
    if dt.kind == "f":
        def bad(v):
            if not np.isfinite(v).all():
                return f"{int((~np.isfinite(v)).sum())} of {v.size} values are not finite"
            mag = np.abs(v[v != 0])
            if mag.size and not 1e-30 <= float(np.median(mag)) <= 1e30:
                return f"the median nonzero magnitude is {float(np.median(mag)):.3g}"
            return None
        why = bad(raw)
        if why is None:
            return None
        if bad(raw.byteswap()) is None:
            return (f"{why}. Read as {other}-endian the same bytes are finite and of ordinary "
                    f"size, so the file is probably {other}-endian.")
        return (f"{why}. Read as {other}-endian the same bytes are no better; the dtype is "
                f"probably wrong (an integer file read as float?).")
    r, r_other = _roughness(raw), _roughness(raw.byteswap())
    if r is not None and r_other is not None and r_other < ROUGHNESS_RATIO * r:
        return (f"as stated the samples step by {r:.2f} of a white record's step, and read as "
                f"{other}-endian by {r_other:.2f} (under {ROUGHNESS_RATIO} of it): the other "
                f"reading is a smooth signal and this one is not, so the file is probably "
                f"{other}-endian.")
    varying = np.bitwise_or.reduce(raw ^ raw[0])
    if not int(varying) & 1:
        return None                       # the lowest bit never moves: padded data, not a swap
    b = raw.view(np.uint8).reshape(-1, dt.itemsize)
    lo, hi = (0, -1) if dt.byteorder in ("<", "=") else (-1, 0)
    h_lo, h_hi = _entropy_bits(b[:, lo]), _entropy_bits(b[:, hi])
    if h_hi <= h_lo + ENTROPY_MARGIN_BITS:
        return None
    return (f"the byte read as MOST significant carries {h_hi:.2f} bits of entropy and the "
            f"byte read as LEAST significant {h_lo:.2f} (a sampled signal is the other way "
            f"round; the margin is {ENTROPY_MARGIN_BITS} bit), so the file is probably "
            f"{other}-endian.")


def _counts(path, dt, check=True, who="capture"):
    """The raw interleaved values of a file, as stored (I, Q, I, Q, ...)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{who}: no such file {str(path)!r}")
    size = p.stat().st_size
    if size == 0:
        raise ValueError(f"{who}: {str(path)!r} contains no samples")
    if size % (2 * dt.itemsize):
        raise ValueError(
            f"{who}: {str(path)!r} is {size} bytes, which is not a whole number of I/Q pairs "
            f"of {dt.name} ({2 * dt.itemsize} bytes each). The dtype is probably wrong, or the "
            f"file has a header -- strip it first; nothing is skipped or truncated for you.")
    raw = np.fromfile(p, dtype=dt)
    if check:
        why = byte_order_evidence(raw, dt)
        if why is not None:
            raise ValueError(
                f"{who}: {str(path)!r} does not look like "
                f"{'big' if dt.byteorder == '>' else 'little'}-endian {dt.name}: {why} Pass "
                f"the byte order the file was written in; if the file really is like this, "
                f"check=False reads it as stated.")
    return raw


def _pairs(raw, scale):
    v = raw.astype(np.float64) * scale
    return v[0::2] + 1j * v[1::2]


# ------------------------------------------------------------------------------------ SigMF
def sigmf_paths(path):
    """(metadata file, data file) for a recording named by its base name or either file."""
    s = str(path)
    for suffix in (".sigmf-meta", ".sigmf-data"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return s + ".sigmf-meta", s + ".sigmf-data"


def _sigmf(path, who="capture"):
    """A SigMF recording's metadata, reduced to what this library reads. Everything it cannot
    honour is refused here by name."""
    meta_path, data_path = sigmf_paths(path)
    if not Path(meta_path).exists():
        raise FileNotFoundError(f"{who}: no such file {meta_path!r} (a SigMF recording is a "
                                f".sigmf-meta / .sigmf-data pair; pass its base name)")
    try:
        meta = json.loads(Path(meta_path).read_text())
        g = meta["global"]
        datatype = g["core:datatype"]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{who}: {meta_path!r} is not SigMF metadata (it needs a 'global' "
                         f"object with 'core:datatype'): {e}") from e
    m = _DATATYPE.match(str(datatype))
    if not m:
        raise ValueError(f"{who}: {meta_path!r} has core:datatype {datatype!r}, which is not "
                         f"a SigMF datatype (e.g. 'cf32_le', 'ci16_le').")
    kind, word, end = m.groups()
    if kind == "r":
        raise ValueError(
            f"{who}: {meta_path!r} is a REAL recording ({datatype}); this reader is for I/Q. "
            f"Read it with np.fromfile and start a real chain with capture(values=...).")
    if word.startswith("u"):
        raise ValueError(
            f"{who}: {meta_path!r} holds unsigned samples ({datatype}), which are offset "
            f"binary. Subtract the midpoint and pass the result as capture(values=...).")
    if end is None and word not in ("i8",):
        raise ValueError(f"{who}: core:datatype {datatype!r} has no byte order (_le / _be).")
    for key, what in (("core:dataset", "a non-conforming dataset file"),
                      ("core:metadata_only", "a metadata-only recording"),
                      ("core:trailing_bytes", "trailing bytes")):
        if g.get(key):
            raise ValueError(f"{who}: {meta_path!r} declares {what} ({key}), which this "
                             f"reader does not handle; convert it to a plain recording.")
    if int(g.get("core:num_channels", 1)) != 1:
        raise ValueError(f"{who}: {meta_path!r} has {g['core:num_channels']} channels; a "
                         f"record is one. Split the channels into separate recordings.")
    caps = meta.get("captures") or [{}]
    if len(caps) > 1 or int(caps[0].get("core:sample_start", 0)) or \
            int(caps[0].get("core:header_bytes", 0)):
        raise ValueError(
            f"{who}: {meta_path!r} has {len(caps)} capture segment(s), or one that does not "
            f"start at sample 0 or carries header bytes. A record has one rate and one centre "
            f"frequency throughout: split the recording at its segment boundaries.")
    if not Path(data_path).exists():
        raise FileNotFoundError(f"{who}: no such file {data_path!r} (the samples of "
                                f"{meta_path!r})")
    want = g.get("core:sha512")
    if want is not None:
        got = hashlib.sha512(Path(data_path).read_bytes()).hexdigest()
        if got != str(want).lower():
            raise ValueError(f"{who}: {data_path!r} does not match the core:sha512 its "
                             f"metadata records; the data file changed or is not the one "
                             f"this metadata describes.")
    dtype = _SIGMF_DTYPE[word]
    return {"data": data_path, "dtype": dtype, "byteorder": "big" if end == "_be" else "little",
            "fs_hz": g.get("core:sample_rate"), "fc_hz": caps[0].get("core:frequency"),
            "scale": g.get("wfmsynth:scale"),
            "annotations": [e.to_dict() for e in
                            annotations_to_events(meta.get("annotations") or [])]}


_CARRIED = ("severity", "t_s", "ui", "direction", "placement", "params", "at", "on_record")


def events_to_annotations(events):
    """Events -> SigMF annotations, sorted by start as the spec requires. `core:sample_start`,
    `core:sample_count` (the event's width) and `core:label` (its kind) are core fields any
    SigMF reader shows; the rest of the event travels in `wfmsynth:event` so it reads back
    exactly."""
    out = []
    for e in events or ():
        d = e.to_dict() if isinstance(e, Event) else dict(e)
        a = {"core:sample_start": int(d["sample"]), "core:label": str(d["kind"]),
             "core:generator": "wfmsynth"}
        if d.get("width") is not None:
            a["core:sample_count"] = int(d["width"])
        a["wfmsynth:event"] = {k: d[k] for k in _CARRIED if k in d}
        out.append(a)
    return sorted(out, key=lambda a: a["core:sample_start"])


def annotations_to_events(annotations):
    """SigMF annotations -> events. One written by `to_sigmf` comes back as the event it was.
    Any other becomes ``Event(sample=core:sample_start, width=core:sample_count,
    kind=core:label or "annotation", placement="annotation")`` with every remaining field in
    `params` under its SigMF name less the ``core:`` prefix (`comment`, `freq_lower_edge`...)."""
    out = []
    for a in annotations:
        if "core:sample_start" not in a:
            raise ValueError(f"capture: a SigMF annotation has no core:sample_start, which "
                             f"the spec requires: {a!r}")
        base = {"sample": int(a["core:sample_start"])}
        if a.get("core:sample_count") is not None:
            base["width"] = int(a["core:sample_count"])
        if isinstance(a.get("wfmsynth:event"), dict):
            out.append(Event(kind=str(a.get("core:label", "annotation")), **base,
                             **{k: v for k, v in a["wfmsynth:event"].items() if k in _CARRIED}))
            continue
        rest = {(k[5:] if k.startswith("core:") else k): v for k, v in a.items()
                if k not in ("core:sample_start", "core:sample_count", "core:label")}
        out.append(Event(kind=str(a.get("core:label") or "annotation"),
                         placement="annotation", params=rest, **base))
    return out


# ---------------------------------------------------------------------------- CSV and MAT
def _read_csv(path, columns):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"capture: no such file {str(path)!r}")
    cols = (1, 2) if columns is None else tuple(columns)
    if len(cols) != 2:
        raise ValueError(f"capture: columns={columns!r} must name two columns, I then Q.")
    with open(p) as f:
        first = f.readline().strip().split(",")
    try:
        [float(c) for c in first]
        header = None
    except ValueError:
        header = [c.strip() for c in first]
    idx = []
    for c in cols:
        if isinstance(c, str):
            if header is None or c not in header:
                raise ValueError(f"capture: {str(path)!r} has no column named {c!r}; its "
                                 f"header is {header if header else 'absent (first row is numbers)'}.")
            idx.append(header.index(c))
        else:
            idx.append(int(c))
    raw = np.atleast_2d(np.loadtxt(p, delimiter=",", skiprows=0 if header is None else 1))
    if max(idx) >= raw.shape[1] or min(idx) < 0:
        raise ValueError(f"capture: {str(path)!r} has {raw.shape[1]} column(s) and "
                         f"columns={columns!r} asks for one that is not there (0-based).")
    return raw[:, idx[0]] + 1j * raw[:, idx[1]]


def _read_mat(path, variable):
    from scipy.io import loadmat, whosmat
    if not Path(path).exists():
        raise FileNotFoundError(f"capture: no such file {str(path)!r}")
    try:
        names = [w[0] for w in whosmat(str(path))]
    except (NotImplementedError, ValueError) as e:
        raise ValueError(f"capture: {str(path)!r} cannot be read as a MATLAB file up to v7.2 "
                         f"({e}). A v7.3 file is HDF5: re-save it with -v7, or export the "
                         f"array as interleaved binary.") from e
    if variable is None or variable not in names:
        raise ValueError(f"capture: format='mat' needs variable=<name of the complex array>; "
                         f"{str(path)!r} holds {names}" +
                         ("." if variable is None else f", not {variable!r}."))
    y = np.asarray(loadmat(str(path), variable_names=[variable])[variable])
    if not np.iscomplexobj(y):
        raise ValueError(f"capture: variable {variable!r} in {str(path)!r} is real. An I/Q "
                         f"record is a complex array; build one (I + 1j*Q) and pass it as "
                         f"capture(values=...) if the file keeps the two apart.")
    return y.ravel()


# --------------------------------------------------------------------------------- reading
def read(path, format, **layout):
    """The complex samples of an I/Q file. `layout` is the format's own parameters (module
    docstring); one the format does not read is refused."""
    lay = _layout(format, layout)
    if format == "csv":
        y = _read_csv(path, lay.get("columns"))
    elif format == "mat":
        y = _read_mat(path, lay.get("variable"))
    else:
        if format == "sigmf":
            m = _sigmf(path)
            path, dtype, order = m["data"], m["dtype"], m["byteorder"]
            scale = lay.get("scale", m["scale"])
        else:
            dtype, order, scale = lay.get("dtype"), lay.get("byteorder"), lay.get("scale")
        dt, _ = _np_dtype(dtype, order)
        y = _pairs(_counts(path, dt, lay.get("check", True)), _scale(scale, dt.name))
    y = np.asarray(y, complex).ravel()
    if y.size == 0:
        raise ValueError(f"capture: {str(path)!r} contains no samples")
    return y


def describe(path, format, **layout):
    """What a file holds, without building a chain: ``{"format", "n", "fs_hz", "fc_hz",
    "dtype", "byteorder", "scale", "peak", "rms", "peak_counts", "full_scale_fraction",
    "annotations"}``. `fs_hz` / `fc_hz` are None where the file does not state them;
    `peak_counts` and `full_scale_fraction` are None for a format with no integer counts.
    This is where a wrong scale shows: it cannot be detected, only looked at."""
    y = read(path, format, **layout)
    lay = _layout(format, layout)
    out = {"format": format, "n": int(y.size), "fs_hz": None, "fc_hz": None, "dtype": None,
           "byteorder": None, "scale": None, "peak": float(np.abs(y).max()),
           "rms": float(np.sqrt(np.mean(np.abs(y) ** 2))), "peak_counts": None,
           "full_scale_fraction": None, "annotations": 0}
    if format == "sigmf":
        m = _sigmf(path)
        out.update(fs_hz=m["fs_hz"], fc_hz=m["fc_hz"], dtype=m["dtype"],
                   byteorder=m["byteorder"], annotations=len(m["annotations"]),
                   scale=_scale(lay.get("scale", m["scale"]), m["dtype"]))
    elif format == "iq_interleaved":
        dt, order = _np_dtype(lay.get("dtype"), lay.get("byteorder"))
        out.update(dtype=dt.name, byteorder=order, scale=_scale(lay.get("scale"), dt.name))
    if out["dtype"] is not None and np.dtype(out["dtype"]).kind == "i":
        peak = max(np.abs(y.real).max(), np.abs(y.imag).max()) / out["scale"]
        out["peak_counts"] = int(round(peak))
        out["full_scale_fraction"] = float(peak / (1 << (8 * np.dtype(out["dtype"]).itemsize - 1)))
    return out


def check_centre(fc_hz, grid, who="capture"):
    """A file's centre frequency must be the record's: nothing retunes it."""
    have = getattr(grid, "fc_hz", None) if grid is not None else None
    if fc_hz is not None and (have is None or float(have) != float(fc_hz)):
        raise ValueError(
            f"{who}: the file is centred on fc_hz={float(fc_hz):g} and the record's grid says "
            f"{'no centre frequency' if have is None else f'fc_hz={float(have):g}'}. Build "
            f"the chain on Grid(..., domain='iq', fc_hz={float(fc_hz):g}), or use "
            f"ws.read_iq(...), which takes the grid from the file.")


def capture_params(grid, path, format, fs_hz=None, fc_hz=None, embed=False, **layout):
    """The `capture` op's parameters for an I/Q file: what `Signal.capture(format=...)` records.
    Everything the metadata states (rate, centre, annotations) is copied into the recipe here,
    so a replay reads only the samples."""
    lay = _layout(format, layout)
    if getattr(grid, "domain", "real") != "iq":
        raise ValueError(CAP._COMPLEX_MSG.format(what=f"{str(path)!r} (format={format!r}) holds"))
    y = read(path, format, **lay)
    params = {"format": format, **lay, "sha256": CAP.digest(y)}
    if "columns" in params:
        params["columns"] = list(params["columns"])
    if format in ("sigmf", "iq_interleaved"):
        m = _sigmf(path) if format == "sigmf" else {}
        for name, mine, theirs in (("fs_hz", fs_hz, m.get("fs_hz")),
                                   ("fc_hz", fc_hz, m.get("fc_hz"))):
            if mine is not None and theirs is not None and float(mine) != float(theirs):
                raise ValueError(f"capture: {name}={float(mine):g} was passed and the SigMF "
                                 f"metadata says {float(theirs):g}. Drop the argument; the "
                                 f"file's own statement is used.")
        fs_hz = fs_hz if fs_hz is not None else m.get("fs_hz")
        fc_hz = fc_hz if fc_hz is not None else m.get("fc_hz")
        dtype = m.get("dtype", lay.get("dtype"))
        params["scale"] = _scale(lay.get("scale", m.get("scale")), dtype)
        if m.get("annotations"):
            params["annotations"] = m["annotations"]
    check_centre(fc_hz, grid)
    if embed:
        params["values"] = CAP.encode(y, "iq")
    else:
        params["path"] = str(path)
    if fs_hz is not None:
        params["fs_hz"] = float(fs_hz)
    if fc_hz is not None:
        params["fc_hz"] = float(fc_hz)
    return params


def emit(annotations, sink, n_file, n_record, grid):
    """Put a capture's annotations on the record's event list. A resampled capture moves them
    with the samples."""
    if sink is None:
        return
    fs = getattr(grid, "fs", None) if grid is not None else None
    k = n_record / n_file
    for d in annotations:
        e = Event(**d)
        if n_record != n_file:
            e.sample = int(round(e.sample * k))
            e.width = None if e.width is None else max(int(round(e.width * k)), 1)
        if e.t_s is None and fs:
            e.t_s = e.sample / float(fs)
        sink.append(e)


def read_iq(path, format, fs_hz=None, fc_hz=None, baud=None, seed=0, embed=False, **layout):
    """A chain that starts from an I/Q file, on a grid taken FROM the file: its length, its
    sample rate and its centre frequency.

        sig = ws.read_iq("rec", format="sigmf")
        x, events = sig.realize()                 # annotations are the events

    `fs_hz` is required for a format that does not state it; `fc_hz` is optional (without it
    the record has no centre, and `upconvert` will ask for one). No file states a symbol rate:
    `baud=` puts one on the grid for the ops and measurements that read it (`filter("rrc")`,
    `iq.demodulate`)."""
    from .compose import Signal
    from .grid import Grid
    d = describe(path, format, **layout)
    for name, mine in (("fs_hz", fs_hz), ("fc_hz", fc_hz)):
        if mine is not None and d[name] is not None and float(mine) != float(d[name]):
            raise ValueError(f"read_iq: {name}={float(mine):g} was passed and the file's "
                             f"metadata says {float(d[name]):g}. Drop the argument.")
    fs = fs_hz if fs_hz is not None else d["fs_hz"]
    fc = fc_hz if fc_hz is not None else d["fc_hz"]
    if fs is None:
        raise ValueError(f"read_iq: {str(path)!r} (format={format!r}) does not state its "
                         f"sample rate; pass fs_hz=.")
    grid = Grid(fs=float(fs), baud=baud, n=d["n"], domain="iq",
                fc_hz=None if fc is None else float(fc))
    return Signal(seed=seed, grid=grid).capture(path=path, format=format, fs_hz=fs_hz,
                                                fc_hz=fc_hz, embed=embed, **layout)


# --------------------------------------------------------------------------------- writing
def _iq_record(x, who):
    """The samples a writer was handed, as a complex array -- or the refusal that says what
    it was handed instead."""
    if hasattr(x, "realize") and hasattr(x, "recipe"):
        raise ValueError(f"{who}: got a {type(x).__name__}, and this writes SAMPLES. Render "
                         f"it first: x, events = sig.realize(), then pass x"
                         f"{', sig.stored_grid(x)' if who == 'to_sigmf' else ''} and "
                         f"events=events.")
    x = np.asarray(x)
    if not np.iscomplexobj(x):
        raise ValueError(f"{who}: the record is real, and this writes I/Q. A real record is "
                         f"exported with np.save or wfmsynth.hdf5; downconvert(...) makes an "
                         f"I/Q record of a passband one.")
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"{who}: needs a one-dimensional, non-empty record; got shape {x.shape}.")
    return x


def _encode(x, dtype, byteorder, scale, who):
    x = _iq_record(x, who)
    dt, order = _np_dtype(dtype, byteorder, who)
    s = _scale(scale, dt.name, who)
    v = np.empty(2 * x.size)
    v[0::2], v[1::2] = x.real / s, x.imag / s
    if dt.kind == "i":
        v = np.rint(v)
        top = (1 << (8 * dt.itemsize - 1)) - 1
        peak = float(np.abs(v).max())
        if peak > top:
            fit = float(max(np.abs(x.real).max(), np.abs(x.imag).max())) / top
            raise ValueError(
                f"{who}: the record peaks at {peak:.0f} counts with scale={s:g}, beyond "
                f"{dt.name}'s +/-{top}. Nothing is clipped for you: pass scale={fit:.6g} (one "
                f"count; the record's peak then sits at full scale) or a larger one for headroom.")
    return v.astype(dt), dt, order, s


def _marker_bytes(events, n):
    m = np.zeros(n, np.uint8)
    for e in events or ():
        e = e if isinstance(e, Event) else Event(**e)
        if not e.on_record:
            continue                        # e.g. an aggressor's: not on these samples
        lo, hi = e.span(n)
        m[lo:hi] = 1
    return m


def to_interleaved(path, x, dtype="float32", byteorder="little", scale=None, events=None):
    """Write an I/Q record as raw interleaved samples (I, Q, I, Q, ...).

        ws.to_interleaved("out.iq", x, dtype="int16", scale=1e-4, events=ev)

    A float32 file holds the record rounded to float32 (relative error up to 6e-8); float64
    holds it exactly; an integer file holds ``rint(x / scale)``, within half a count, and is
    refused rather than clipped if the record does not fit.

    `events=` also writes a MARKER channel, ``<path>.markers``: one uint8 per sample, 1 across
    each event's span (its `sample` for `width` samples) and 0 elsewhere. Events that are not
    on these samples (`Event.on_record` False: an aggressor's, a trigger) are left out. The file states nothing
    about itself, so the returned dict is what a reader needs:
    ``{"path", "markers", "dtype", "byteorder", "scale", "n"}``."""
    v, dt, order, s = _encode(x, dtype, byteorder, scale, "to_interleaved")
    v.tofile(str(path))
    markers = None
    if events is not None:
        markers = str(path) + ".markers"
        _marker_bytes(events, len(x)).tofile(markers)
    return {"path": str(path), "markers": markers, "dtype": dt.name, "byteorder": order,
            "scale": s, "n": int(len(x))}


def to_sigmf(path, x, grid, events=None, dtype="float32", scale=None, description=None):
    """Write an I/Q record as a SigMF recording: `<path>.sigmf-data` and `<path>.sigmf-meta`.

        ws.to_sigmf("out", x, sig.stored_grid(x), events=ev)

    `grid` is the grid that describes `x` (`sig.stored_grid(x)`): its `fs` is written as
    `core:sample_rate` and its `fc_hz`, when it has one, as the capture's `core:frequency`.
    `events` become annotations (`events_to_annotations`) and read back as the same events.
    Samples are little-endian `dtype`; an integer type also records its scale as
    `wfmsynth:scale`, which this library's reader uses and other readers ignore.

    Fields written, all from SigMF v1.0.0: global `core:datatype`, `core:version`,
    `core:sample_rate`, `core:sha512`, `core:num_channels`, `core:extensions` (declaring the
    optional `wfmsynth` namespace), `core:description` when given; one capture with
    `core:sample_start` and `core:frequency`; annotations with `core:sample_start`,
    `core:sample_count`, `core:label`, `core:generator`. Returns ``{"meta", "data",
    "datatype", "scale", "n"}``."""
    x = _iq_record(x, "to_sigmf")
    if getattr(grid, "domain", "real") != "iq":
        raise ValueError(
            "to_sigmf: the grid is a real record's (domain 'real'), so it has no centre "
            "frequency to write and the file would read back with none. Pass the grid of "
            "THIS record, sig.stored_grid(x): after a downconvert it is not sig.grid.")
    if len(x) != int(grid.n):
        raise ValueError(f"to_sigmf: the record has {len(x)} samples and the grid describes "
                         f"{int(grid.n)}. Pass the grid of THIS record: sig.stored_grid(x).")
    v, dt, _, s = _encode(x, dtype, "little", scale, "to_sigmf")
    meta_path, data_path = sigmf_paths(path)
    raw = v.tobytes()
    Path(data_path).write_bytes(raw)
    datatype = "c" + _SIGMF_TYPE[dt.name] + ("" if dt.itemsize == 1 else "_le")
    g = {"core:datatype": datatype, "core:version": SIGMF_VERSION,
         "core:sample_rate": float(grid.fs), "core:num_channels": 1,
         "core:sha512": hashlib.sha512(raw).hexdigest(),
         "core:extensions": [{"name": "wfmsynth", "version": "1.0.0", "optional": True}]}
    if description is not None:
        g["core:description"] = str(description)
    if dt.kind == "i":
        g["wfmsynth:scale"] = s
    cap = {"core:sample_start": 0}
    if getattr(grid, "fc_hz", None) is not None:
        cap["core:frequency"] = float(grid.fc_hz)
    meta = {"global": g, "captures": [cap], "annotations": events_to_annotations(events)}
    Path(meta_path).write_text(json.dumps(meta, indent=2) + "\n")
    return {"meta": meta_path, "data": data_path, "datatype": datatype, "scale": s,
            "n": int(len(x))}

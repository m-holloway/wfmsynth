"""Write a record as HDF5, in the shape a bench oscilloscope reads.

Two consumers, one file. Any HDF5 reader gets the samples and the axis metadata it needs to put
them in volts and seconds. An instrument that expects this particular layout also loads it,
provided the file carries the identifying metadata that instrument looks for -- which is not in
this library, because it is not physics and it is not ours to state. `like=` copies it from a
capture you already have; see `identity_from`.

The layout, as read off real captures:

    /FileType/<name>                      scalar, fixed-length bytes: the format marker
    /Frame/TheFrame                       scalar compound: (Model, Serial, Date)
    /Waveforms                            attrs: NumWaveforms
    /Waveforms/Channel N                  attrs: the axis, scale and display metadata
    /Waveforms/Channel N/Channel NData    int16 sample codes, chunked, extendible

Samples are stored as raw signed 16-bit codes, not volts, and the axes are reconstructed from
scalar attributes:

    volts   = code * YInc + YOrg
    seconds = index * XInc + XOrg

Storing codes rather than floats is what makes the file half the size and what an instrument
expects; it also means the vertical scale is a property of the file rather than of the reader.

`h5py` is needed to write, and is not a dependency of the rest of this library.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

# The compound record describing what produced a capture. The field widths are part of the layout.
FRAME_DTYPE = np.dtype([("Model", "S12"), ("Serial", "S12"), ("Date", "S22")])

# The chunk length observed in real captures, regardless of record length.
CHUNK = 200_000

# A neutral marker, used when no reference file is given. An instrument looking for its own marker
# will not accept this; every other HDF5 reader is indifferent to it.
DEFAULT_FILETYPE_NAME = "WaveformFileType"
DEFAULT_FILETYPE_VALUE = b"WaveformFile"


def _h5py():
    try:
        import h5py
    except ImportError as e:                       # pragma: no cover - environment dependent
        raise ImportError(
            "writing HDF5 needs h5py, which the rest of this library does not require: "
            "pip install h5py") from e
    return h5py


def identity_from(path):
    """Read the identifying metadata out of an existing capture.

    Returns a dict to hand to `write_hdf5(like=...)`. The format marker's dataset NAME and value,
    and the frame record, are what an instrument uses to recognise a file as its own. They are
    specific to whoever wrote that capture, so they come from the capture rather than from here.
    """
    h5py = _h5py()
    out = {}
    with h5py.File(str(path), "r") as f:
        ft = f.get("FileType")
        if ft is not None:
            for name in ft:                        # one scalar dataset, whatever it is called
                out["filetype_name"] = name
                out["filetype_value"] = bytes(np.asarray(ft[name]).item())
                break
        frame = f.get("Frame/TheFrame")
        if frame is not None:
            out["frame"] = np.asarray(frame).item()
            out["frame_dtype"] = frame.dtype
    if "filetype_name" not in out:
        raise ValueError(f"{path} carries no /FileType marker to copy")
    return out


def counts_for(volts, full_scale=None, bits=16):
    """Signed integer codes for a voltage array, plus the scale and offset that invert it.

    The code range is centred on the data rather than on zero, so the full width of the converter
    is spent on the signal that is actually there. Returns `(codes, y_inc, y_org)` with
    `volts == codes * y_inc + y_org` to within half a code.
    """
    v = np.asarray(volts, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("no samples to write")
    lo, hi = float(np.min(v)), float(np.max(v))
    if full_scale is not None:
        span = float(full_scale)
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * span, mid + 0.5 * span
    if hi <= lo:                                   # a constant record still has to round-trip
        hi = lo + 1.0
    limit = 2 ** (int(bits) - 1) - 1               # leave the most negative code as the fill value
    y_inc = (hi - lo) / (2 * limit)
    y_org = 0.5 * (lo + hi)
    raw = np.rint((v - y_org) / y_inc)
    over = int(np.count_nonzero(np.abs(raw) > limit))
    if over:
        # A window narrower than the record destroys the peaks, and clipping quietly is the worst
        # way to do it: the file writes, opens, and looks plausible. The excursion that matters is
        # the one the record actually has after the channel and the defects, which is not the
        # grid's v_full and not the transmitter's amplitude.
        span = float(np.max(v) - np.min(v))
        raise ValueError(
            f"full_scale={float(full_scale):.6g} V clips {over:,} of {v.size:,} samples. This "
            f"record spans {span:.6g} V peak-to-peak, from {float(np.min(v)):.6g} to "
            f"{float(np.max(v)):.6g}. Pass a window that contains it, or omit full_scale and the "
            f"window is taken from the record.")
    codes = raw.astype("int16")
    return codes, y_inc, y_org


def write_hdf5(path, channels, *, fs, t0=0.0, full_scale=None, like=None,
               model=b"", serial=b"", date=None):
    """Write one or more channels of a record to `path`.

    channels    a 1-D array of volts, or {channel_number: volts}
    fs          sample rate [Hz]; sets XInc
    t0          time of the first sample [s]; sets XOrg. Negative where the record is
                pre-trigger, which is what a captured record normally is.
    full_scale  the vertical window in volts peak-to-peak, which must contain the record. Omit and
                it is taken from the
                data, which spends the whole code range on it but makes the scale record-dependent
                -- pass it explicitly when several records have to share one scale.
    like        a path to a reference capture, or the dict `identity_from` returns. Supplies the
                format marker and frame record so an instrument recognises the file.
    """
    h5py = _h5py()
    if not isinstance(channels, dict):
        channels = {1: channels}
    if not channels:
        raise ValueError("no channels to write")

    ident = dict(like) if isinstance(like, dict) else (identity_from(like) if like else {})
    ft_name = ident.get("filetype_name", DEFAULT_FILETYPE_NAME)
    ft_value = ident.get("filetype_value", DEFAULT_FILETYPE_VALUE)
    frame_dtype = ident.get("frame_dtype", FRAME_DTYPE)
    if "frame" in ident:
        frame = np.array(ident["frame"], dtype=frame_dtype)
    else:
        stamp = date if date is not None else _dt.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        frame = np.array((_b(model), _b(serial), _b(stamp)), dtype=frame_dtype)

    x_inc = 1.0 / float(fs)
    with h5py.File(str(path), "w") as f:
        f.create_dataset(f"FileType/{ft_name}", data=np.bytes_(ft_value),
                         dtype=h5py.string_dtype("ascii", 40))
        f.create_dataset("Frame/TheFrame", data=frame, dtype=frame_dtype)
        wf = f.create_group("Waveforms")
        wf.attrs["NumWaveforms"] = np.int32(len(channels))

        for n, volts in sorted(channels.items()):
            codes, y_inc, y_org = counts_for(volts, full_scale=full_scale)
            npts = int(codes.size)
            g = wf.create_group(f"Channel {n}")
            d = g.create_dataset(f"Channel {n}Data", data=codes, dtype="int16",
                                 chunks=(min(CHUNK, npts),), maxshape=(None,), fillvalue=0)
            _channel_attrs(g, npts=npts, x_inc=x_inc, x_org=float(t0),
                           y_inc=y_inc, y_org=y_org)
            _data_attrs(d, npts=npts)
    return str(path)


def _b(v):
    return v if isinstance(v, bytes) else str(v).encode("ascii", "replace")


def _channel_attrs(g, *, npts, x_inc, x_org, y_inc, y_org):
    """The axis, scale and display metadata. The four load-bearing values are XInc/XOrg and
    YInc/YOrg; the rest is the shape a reader expects to find, written at its resting value."""
    a = g.attrs
    a["NumPoints"] = np.int32(npts)
    a["XInc"], a["XOrg"] = np.float64(x_inc), np.float64(x_org)
    a["XUnits"] = np.bytes_(b"Second")
    a["YInc"], a["YOrg"] = np.float64(y_inc), np.float64(y_org)
    a["YUnits"] = np.bytes_(b"Volt")
    a["YReference"] = np.int32(1)
    # the display window: the whole record, and the full vertical span
    a["XDispOrigin"] = np.float64(x_org)
    a["XDispRange"] = np.float32(npts * x_inc)
    a["YDispOrigin"] = np.float64(y_org)
    a["YDispRange"] = np.float32(y_inc * 2 * (2 ** 15 - 1))
    a["Count"] = np.int32(1)
    a["WaveformType"] = np.int16(1)
    a["Start"] = np.int32(0)
    a["NumSegments"] = np.int32(0)
    a["InterpSetting"] = np.int16(0)
    a["DispInterpFactor"] = np.int32(1)
    a["SavedInterpFactor"] = np.int32(1)
    a["MinBandwidth"] = np.float64(0.0)
    a["MaxBandwidth"] = np.float64(0.0)
    a["FFT_RBW"] = np.float64(0.0)
    a["ColorGradeWidth"] = np.int32(0)
    a["ColorGradeHeight"] = np.int32(0)
    a["WavAttr"] = np.int32(0)
    a["ClipHighQ"] = np.int32(0)
    a["ClipLowQ"] = np.int32(0)
    a["HoleQ"] = np.int32(0)
    a["CenterFrequency"] = np.float64(0.0)
    a["Span"] = np.float64(0.0)
    a["Rotation"] = np.int16(0)


def _data_attrs(d, *, npts):
    a = d.attrs
    a["DataType"] = np.int32(1)
    a["DecMode"] = np.int32(1)
    a["InfoValid"] = np.uint8(1)
    a["PktSize"] = np.int32(1)
    a["RawNumPts"] = np.int32(npts)
    a["ReductionAllowed"] = np.uint8(1)
    a["SegmentedTimeTag"] = np.float64(0.0)
    a["SegmentedXOrg"] = np.float64(0.0)
    a["StartIndex"] = np.uint32(0)


def read_hdf5(path, channel=None):
    """Read a file written by `write_hdf5` (or one of the captures it is modelled on).

    Returns `(volts, t)` for one channel, or `{n: (volts, t)}` for all of them. Included so the
    exporter can be checked against its own output rather than asserted to be correct.
    """
    h5py = _h5py()
    out = {}
    with h5py.File(str(path), "r") as f:
        for name, g in sorted(f["Waveforms"].items()):
            n = int(str(name).split()[-1])
            d = g[f"{name}Data"]
            a = g.attrs
            codes = np.asarray(d[:], dtype=np.int64)
            volts = codes * float(a["YInc"]) + float(a["YOrg"])
            t = np.arange(codes.size, dtype=float) * float(a["XInc"]) + float(a["XOrg"])
            out[n] = (volts, t)
    if channel is not None:
        return out[channel]
    return out

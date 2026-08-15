"""
wfmsynth.sparam — measured S-parameter (Touchstone) channels.

The analytic √f + f loss model (`physics.lossy_channel`) is smooth and monotonic: it cannot
produce resonances, fibre-weave periodicity, connector structure, or interacting multiple
bounces. Real ``.sNp`` files carry all of it. This module reads Touchstone, and applies a
port-to-port S-parameter as a frequency-domain channel, so a measured trace can drive
synthesis. The analytic model stays the dependency-free default; this is the opt-in
"use my real channel" path. numpy/scipy only.

    f, S = read_touchstone("thru.s2p")           # S: (nf, n, n) complex
    y = sparam_channel(x, grid=g, freqs=f, s21=S[:, 1, 0])   # apply S21 (port2 <- port1)
    # or in one step from a file:
    y = touchstone_channel(x, "thru.s2p", grid=g)
"""
from __future__ import annotations

import numpy as np

_FUNIT = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}


def _n_ports_from_ext(path):
    ext = str(path).rsplit(".", 1)[-1].lower()
    if ext.startswith("s") and ext.endswith("p") and ext[1:-1].isdigit():
        return int(ext[1:-1])
    return None


def read_touchstone(path, n_ports=None):
    """Read a Touchstone (.sNp) file. Returns ``(freqs_hz, S)`` where ``S`` has shape
    ``(nf, n, n)`` complex. Supports RI / MA / DB formats and HZ/KHZ/MHZ/GHZ; comments
    (``!``) and multi-line frequency rows are handled. Port count is taken from the ``.sNp``
    extension (override with ``n_ports``)."""
    with open(path) as fh:
        text = fh.read()
    return _parse_touchstone(text, n_ports if n_ports else (_n_ports_from_ext(path) or 2))


def _parse_touchstone(text, n):
    funit, fmt = 1e9, "MA"
    nums = []
    for line in text.splitlines():
        line = line.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            toks = line[1:].split()
            if toks:
                funit = _FUNIT[toks[0].upper()]
            if len(toks) >= 3:
                fmt = toks[2].upper()
            continue
        nums.extend(float(t) for t in line.split())
    per = 1 + 2 * n * n
    if len(nums) % per != 0:
        raise ValueError(f"Touchstone: {len(nums)} numbers not a multiple of {per} for {n}-port")
    rows = np.asarray(nums, float).reshape(-1, per)
    freqs = rows[:, 0] * funit
    pairs = rows[:, 1:].reshape(len(rows), n * n, 2)
    if fmt == "RI":
        flat = pairs[:, :, 0] + 1j * pairs[:, :, 1]
    elif fmt == "MA":
        flat = pairs[:, :, 0] * np.exp(1j * np.deg2rad(pairs[:, :, 1]))
    elif fmt == "DB":
        flat = 10 ** (pairs[:, :, 0] / 20.0) * np.exp(1j * np.deg2rad(pairs[:, :, 1]))
    else:
        raise ValueError(f"unknown Touchstone format {fmt!r} (use RI, MA or DB)")
    S = flat.reshape(len(rows), n, n)
    if n == 2:                          # Touchstone's 2-port quirk: order is S11 S21 S12 S22
        S = S.transpose(0, 2, 1)
    return freqs, S


def write_touchstone(path, freqs_hz, S, fmt="RI", funit="HZ"):
    """Write ``(freqs_hz, S)`` (S shape ``(nf, n, n)``) to a Touchstone file. Mainly for
    tests and round-trips; RI format by default."""
    S = np.asarray(S)
    n = S.shape[1]
    scale = _FUNIT[funit.upper()]
    lines = [f"# {funit.upper()} S {fmt.upper()} R 50"]
    for i, f in enumerate(freqs_hz):
        mat = S[i].T if n == 2 else S[i]        # invert the 2-port quirk on the way out
        vals = [f / scale]
        for e in mat.reshape(-1):
            if fmt.upper() == "RI":
                vals += [e.real, e.imag]
            elif fmt.upper() == "MA":
                vals += [abs(e), np.rad2deg(np.angle(e))]
            else:
                vals += [20 * np.log10(abs(e) + 1e-30), np.rad2deg(np.angle(e))]
        lines.append(" ".join(f"{v:.9g}" for v in vals))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def sparam_channel(x, freqs, s21, grid=None, dt=None):
    """Apply a measured transfer ``s21`` (complex, sampled at ``freqs`` in Hz) to ``x`` as a
    frequency-domain channel. Provide the sample spacing via ``grid=Grid(...)`` or ``dt``.
    The response is interpolated (real/imag) onto the signal's FFT grid and zeroed outside
    the measured band. Unlike the analytic model this reproduces resonances and structure."""
    x = np.asarray(x, float)
    if dt is None:
        if grid is None:
            raise ValueError("sparam_channel needs grid=Grid(...) or dt=")
        dt = grid.dt
    freqs = np.asarray(freqs, float)
    s21 = np.asarray(s21, complex)
    fg = np.fft.rfftfreq(len(x), d=dt)
    H = np.interp(fg, freqs, s21.real) + 1j * np.interp(fg, freqs, s21.imag)
    H[(fg < freqs.min()) | (fg > freqs.max())] = 0.0        # no extrapolation beyond the trace
    return np.fft.irfft(np.fft.rfft(x) * H, len(x))


def touchstone_channel(x, path, grid=None, dt=None, ports=(2, 1), n_ports=None):
    """Read a Touchstone file and apply ``S[ports[0]-1, ports[1]-1]`` (default S21) to ``x``."""
    freqs, S = read_touchstone(path, n_ports=n_ports)
    return sparam_channel(x, freqs, S[:, ports[0] - 1, ports[1] - 1], grid=grid, dt=dt)

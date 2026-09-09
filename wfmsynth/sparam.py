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

It also CASCADES. A channel is not one lumped block — it is a path, and where a discontinuity
sits decides how much loss its echo pays. `cascade_channel` composes sections of line (each with
its own loss) and discontinuities (each with its own Gamma) into one S-matrix, so an echo is
generated where it physically is and is attenuated by the segment it actually traverses:

    path = [{"line": {"length_in": 1.0}}, {"disc": {"gamma": 0.055}},     # connector at 1.0 in
            {"line": {"length_in": 9.0}}, {"disc": {"gamma": 0.055}},     # via at 10.0 in
            {"line": {"length_in": 2.0}}]
    first_order_echoes(path)                     # where the echoes MUST be, before simulating
    y = cascade_channel(x, path, grid=g, node="source")     # driver-plane reverse wave

See the long comment above `C_IN_PER_NS` for why the topology lives here and not as a
"sectioned lossy/reflect".
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


# =====================================================================================
# CASCADED CHANNELS — topology, not one lumped block
# =====================================================================================
# A real Gen4 path is package -> short trace -> connector -> longer trace -> via -> trace ->
# package. Every discontinuity in that list reflects a wave that has taken only the loss UP TO
# THAT POINT, and the echo it sends back pays that same loss AGAIN on the way out. One lumped
# `lossy` followed by one lumped `reflect` cannot express this: it applies the whole channel's
# loss once and then reflects, so every echo arrives having paid the full insertion loss exactly
# once no matter where it came from, and a near reflection and a far reflection of equal Gamma
# come out the SAME SIZE. They are not the same size on any real board, and the difference is
# the entire basis on which an echo's arrival time is read as a distance.
#
# WHY THIS LIVES IN sparam AND NOT AS A "SECTIONED lossy/reflect"
# --------------------------------------------------------------
# Cascading two 2-ports is
#
#     D = 1 - A22*B11
#     S21 = A21*B21/D      S11 = A11 + A12*A21*B11/D
#     S12 = A12*B12/D      S22 = B22 + B12*B21*A22/D
#
# and that 1/D is not bookkeeping — it is the closed form of the infinite bounce series between
# A's output plane and B's input plane, carrying the round-trip loss of whatever sits between
# them. Expand the two-discontinuity case (Gamma1, a line of one-way transmission L, Gamma2):
#
#     S11 = Gamma1 + t1^2 * Gamma2 * L^2 / (1 + Gamma1*Gamma2*L^2)
#                    ^^^^^^^^^^^^^^^^^^^ the far echo: its own Gamma, the segment's loss TWICE,
#                                        and the near discontinuity's transmission twice
#
# The positional attenuation falls out of the algebra. A time-domain "sectioned lossy/reflect"
# would have to hand-code that series, would be a second implementation of physics that already
# has one here, and would get the second- and higher-order bounces (the terms that make a real
# board's return loss ripple) wrong. So: **sparam carries the path**, `physics.lossy_channel`
# stays the loss law each section is built from (via `physics.insertion_loss_db`, shared code,
# not a copy), and `physics.multi_reflection` stays exactly as it is — the lumped legacy op,
# still correct for the one thing it models, a line mismatched at both ends and nowhere else.

C_IN_PER_NS = 11.8028                    # speed of light in vacuum, inches per nanosecond


def ps_per_inch(eps_r=4.0):
    """One-way propagation delay [ps/inch] in a dielectric of effective permittivity `eps_r`.

    eps_r=4.0 gives 169.45 ps/inch. `wfmplan.fold.PS_PER_INCH_ONE_WAY` is 169.5, from the
    same arithmetic with c rounded to 11.8 in/ns — 0.03 % apart, which is 0.5 mil on a 1.66-inch
    echo. Pass `ps_per_inch=` explicitly to any section that must pin a downstream convention."""
    return 1000.0 / C_IN_PER_NS * np.sqrt(float(eps_r))


class TwoPort:
    """A 2-port S-matrix sampled on a frequency axis: four complex arrays, all same length.

    Deliberately not a dataclass of one (nf,2,2) array — the cascade formula reads better
    element-wise, and a section is usually built one element at a time."""

    __slots__ = ("freqs", "s11", "s12", "s21", "s22")

    def __init__(self, freqs, s11, s12, s21, s22):
        self.freqs = np.asarray(freqs, float)
        self.s11, self.s12 = np.asarray(s11, complex), np.asarray(s12, complex)
        self.s21, self.s22 = np.asarray(s21, complex), np.asarray(s22, complex)

    def __repr__(self):
        return f"<TwoPort {len(self.freqs)} pts, {self.freqs[-1]/1e9:.1f} GHz>"

    def matrix(self):
        """(nf, 2, 2) complex, in the (row=out, col=in) convention `read_touchstone` returns."""
        S = np.empty((len(self.freqs), 2, 2), complex)
        S[:, 0, 0], S[:, 0, 1] = self.s11, self.s12
        S[:, 1, 0], S[:, 1, 1] = self.s21, self.s22
        return S


def cascade_2port(A, B):
    """Cascade two 2-ports (A's port 2 tied to B's port 1). Both must share a frequency axis.

    The 1/(1 - A22*B11) denominator is the infinite series of bounces trapped between the two
    junctions; it is what makes a reflection pay the loss of the segment it actually traverses."""
    if len(A.freqs) != len(B.freqs) or not np.allclose(A.freqs, B.freqs):
        raise ValueError("cascade_2port: sections must share one frequency axis")
    d = 1.0 - A.s22 * B.s11
    return TwoPort(A.freqs,
                   A.s11 + A.s12 * A.s21 * B.s11 / d,
                   A.s12 * B.s12 / d,
                   A.s21 * B.s21 / d,
                   B.s22 + B.s12 * B.s21 * A.s22 / d)


def _is_rfft_axis(freqs):
    """True when `freqs` is a uniform DC-to-Nyquist rfft grid (what min-phase needs)."""
    f = np.asarray(freqs, float)
    return len(f) > 2 and f[0] == 0.0 and np.allclose(np.diff(f), f[1] - f[0])


def line(freqs, td_ps=None, length_in=None, eps_r=4.0, ps_per_in=None, causal=True,
         tand=0.02, skin_k=0.0, loss_db=None, loss_at_ghz=None, trend=None,
         trend_floor_db=80.0, loss_length_in=None):
    """A matched section of transmission line: pure delay plus that section's own loss.

    S11 = S22 = 0 (matched: a uniform line reflects nothing — the reflections live in the
    `discontinuity` sections between lines), S21 = S12 = |H(f)| * exp(-j*2*pi*f*td).

    Length and delay
        `td_ps` states the one-way delay outright. `length_in` derives it as
        `length_in * ps_per_inch(eps_r)` — override the conversion with `ps_per_in=`.

    Loss
        The section's insertion loss uses `physics.insertion_loss_db`, i.e. exactly the law
        `physics.lossy_channel` applies, so a section and a lumped channel cannot drift apart.
        By default the loss is computed for `length_in` inches of the given `tand`/`eps_r`
        stackup; `loss_db`+`loss_at_ghz` pin this section's loss at a frequency, and
        `trend=(a,b,c)` hands it a measured board's whole fitted |S21| curve. Set
        `loss_length_in=` to give the loss a different length from the delay (a package or a
        connector footprint whose per-inch coefficients are not the board's).

    `causal=True` (default) gives the loss its minimum-phase (Kramers-Kronig) phase, so the
    section is causal and dispersive — the same treatment `lossy_channel(causal=True)` applies.
    It needs a uniform DC-to-Nyquist frequency axis; on any other axis pass `causal=False`."""
    from . import physics as P
    freqs = np.asarray(freqs, float)
    if td_ps is None:
        if length_in is None:
            raise ValueError("line() needs td_ps= or length_in=")
        pp = ps_per_in if ps_per_in is not None else ps_per_inch(eps_r)
        td_ps = float(length_in) * pp
    li = loss_length_in if loss_length_in is not None else (length_in if length_in is not None else 0.0)
    il_db = P.insertion_loss_db(freqs / 1e9, length_in=li, tand=tand, eps_r=eps_r,
                                skin_k=skin_k, loss_db=loss_db, loss_at_ghz=loss_at_ghz,
                                trend=trend, trend_floor_db=trend_floor_db)
    mag = 10.0 ** (-il_db / 20.0)
    if causal and np.any(il_db > 0):
        if not _is_rfft_axis(freqs):
            raise ValueError("line(causal=True) needs a uniform DC..Nyquist axis; pass causal=False")
        n = 2 * (len(freqs) - 1)
        H = P._min_phase_H(mag, n)[:len(freqs)]
    else:
        H = mag.astype(complex)
    H = H * np.exp(-1j * 2.0 * np.pi * freqs * (float(td_ps) * 1e-12))
    z = np.zeros(len(freqs), complex)
    return TwoPort(freqs, z, H, H, z.copy())


def discontinuity(freqs, gamma):
    """A lossless, reciprocal impedance step of reflection coefficient `gamma`.

        S = [[G, t], [t, -G]],   t = sqrt(1 - G^2)

    which is unitary (|G|^2 + t^2 = 1): the junction reflects and transmits, it does not absorb.
    The sign flip on S22 is the step seen from the other side — it is why the second-order
    bounce between two discontinuities has the sign it does, and dropping it is how a hand-rolled
    bounce sum ends up with the ripple of a real board's return loss inverted.

    `gamma` is a scalar (frequency-flat: a clean impedance step) or an array over `freqs`
    (a resonant discontinuity — a via stub, an open — whose |Gamma| and phase both move with
    frequency; `physics.resonant_reflection` documents that shape)."""
    freqs = np.asarray(freqs, float)
    G = np.broadcast_to(np.asarray(gamma, complex), freqs.shape).astype(complex)
    t = np.sqrt(1.0 - G ** 2)
    return TwoPort(freqs, G, t, t, -G)


def measured(freqs, path, ports=(2, 1), n_ports=None, S=None, file_freqs=None):
    """One SECTION taken from a measured Touchstone file — a real connector or via dropped into
    an otherwise synthetic path. The file's full 2-port block (both reflections, both
    transmissions) is interpolated onto `freqs`; outside the measured band every element is
    zeroed, which makes the section an open stub there rather than an extrapolation nobody
    measured. `ports=(2,1)` names which two ports of an N-port file form the section."""
    if S is None:
        file_freqs, Sfull = read_touchstone(path, n_ports=n_ports)
    else:
        Sfull = np.asarray(S, complex)
    i, j = ports[0] - 1, ports[1] - 1
    freqs = np.asarray(freqs, float)

    def _ip(z):
        out = np.interp(freqs, file_freqs, z.real) + 1j * np.interp(freqs, file_freqs, z.imag)
        out[(freqs < file_freqs.min()) | (freqs > file_freqs.max())] = 0.0
        return out

    return TwoPort(freqs, _ip(Sfull[:, j, j]), _ip(Sfull[:, j, i]),
                   _ip(Sfull[:, i, j]), _ip(Sfull[:, i, i]))


def build_sections(path, freqs):
    """Turn a declarative path spec into `TwoPort` sections on `freqs`.

    `path` is a list of one-key dicts, in physical order from the driver to the receiver:

        [{"line": {"length_in": 1.0, "tand": 0.02, "eps_r": 4.0}},
         {"disc": {"gamma": 0.055}},                       # the connector, 1.0 in out
         {"line": {"length_in": 5.0}},
         {"disc": {"gamma": 0.055}},                       # the via, 6.0 in out
         {"line": {"length_in": 2.0}},
         {"file": {"path": "connector.s2p"}}]              # a measured section

    Plain JSON, so it round-trips through a recipe. A bare `TwoPort` may also appear in the
    list and is used as-is."""
    out = []
    for k, sec in enumerate(path):
        if isinstance(sec, TwoPort):
            out.append(sec); continue
        if len(sec) != 1:
            raise ValueError(f"path[{k}]: expected one key of line|disc|file, got {sorted(sec)}")
        kind, args = next(iter(sec.items()))
        args = dict(args)
        if kind == "line":
            if args.get("trend") is not None:
                args["trend"] = tuple(float(t) for t in args["trend"])   # JSON gives a list
            out.append(line(freqs, **args))
        elif kind == "disc":
            g = args.get("gamma")
            if isinstance(g, (list, tuple)):
                g = np.asarray(g, complex)
            out.append(discontinuity(freqs, g))
        elif kind == "file":
            out.append(measured(freqs, **args))
        else:
            raise ValueError(f"path[{k}]: unknown section kind {kind!r} (line|disc|file)")
    return out


def cascade(path, freqs):
    """Cascade a path spec (or a list of `TwoPort`s) into one `TwoPort` on `freqs`."""
    secs = build_sections(path, freqs)
    if not secs:
        raise ValueError("cascade: empty path")
    out = secs[0]
    for s in secs[1:]:
        out = cascade_2port(out, s)
    return out


def first_order_echoes(path, eps_r_default=4.0):
    """The arithmetic's answer, ahead of any simulation: for each `disc` in `path`, the
    round-trip delay of the echo it sends back to the DRIVER and that echo's low-frequency
    amplitude, as a fraction of the launched wave.

    For a discontinuity of coefficient G_k sitting `d_k` of one-way delay from the driver,
    behind discontinuities G_1..G_(k-1):

        delay = 2 * d_k                       (out and back)
        amp   = G_k * prod_{i<k} (1 - G_i^2)  (transmitted through each earlier junction twice)

    The DC amplitude is stated here because it is loss-free by construction; the whole point of
    the cascade is that the echo is then attenuated by 2x the insertion loss of the `d_k` of line
    it actually traverses, which is frequency-dependent and so is not a single number. Use this
    to know WHERE the echoes must be and roughly how big, then measure the realised waveform.

    Returns a list of dicts with keys: `index` (position in `path`), `td_ps` (one way to the
    discontinuity), `delay_ps` (the round trip, = 2*td_ps), `gamma`, `amp`, `distance_inch`
    (td_ps at eps_r_default), and `loss_length_ps` -- the electrical length whose insertion loss
    this echo pays, which is the round trip, not the one-way."""
    out, td, through = [], 0.0, 1.0
    for k, sec in enumerate(path):
        if isinstance(sec, TwoPort) or len(sec) != 1:
            raise ValueError("first_order_echoes needs a declarative path spec (no bare TwoPort)")
        kind, args = next(iter(sec.items()))
        if kind == "line":
            if "td_ps" in args:
                td += float(args["td_ps"])
            else:
                pp = args.get("ps_per_in") or ps_per_inch(args.get("eps_r", eps_r_default))
                td += float(args["length_in"]) * pp
        elif kind == "disc":
            g = args["gamma"]
            if isinstance(g, (list, tuple, np.ndarray)):
                continue                     # frequency-dependent: no single first-order number
            g = float(g)
            out.append({"index": k, "td_ps": td, "delay_ps": 2.0 * td, "gamma": g,
                        "amp": g * through,
                        "distance_inch": td / ps_per_inch(eps_r_default),
                        "loss_length_ps": 2.0 * td})
            through *= (1.0 - g ** 2)
        elif kind == "file":
            raise ValueError("first_order_echoes: a measured section has no closed form; "
                             "cascade it and read the impulse response instead")
    return out


def cascade_channel(x, path, grid=None, dt=None, node="load", eps_r_default=4.0):
    """Apply a CASCADED channel to `x`: sections of line with their own loss, separated by
    discontinuities with their own reflection coefficients.

    This op REPLACES `lossy` + `reflect` for a path with structure; it is not stacked on top of
    them. It carries the insertion loss (all sections, once, on the direct path) AND every
    reflection, each generated where it physically sits.

    `node="load"`   the received waveform at the far end: S21 of the cascade. Every echo that
                    has bounced an even number of times and arrived forward again is in it,
                    each having paid the loss of the segment it bounced within.
    `node="source"` the driver-plane waveform: incident plus reverse wave, (1 + S11)*X. This is
                    the plane a near-end echo is seen at, and the one TDR-style distance
                    estimation reads.

    The sections are built directly on the signal's own rfft grid, so there is no interpolation
    of an analytic path and the delays are exact (a linear phase, not a rounded sample count —
    `physics.multi_reflection` rounds `td_ps` to the nearest sample)."""
    x = np.asarray(x, float)
    if dt is None:
        if grid is None:
            raise ValueError("cascade_channel needs grid=Grid(...) or dt=")
        dt = grid.dt
    if node not in ("load", "source"):
        raise ValueError("node must be 'load' or 'source'")
    freqs = np.fft.rfftfreq(len(x), d=dt)
    tp = cascade(path, freqs)
    H = tp.s21 if node == "load" else (1.0 + tp.s11)
    return np.fft.irfft(np.fft.rfft(x) * H, len(x))

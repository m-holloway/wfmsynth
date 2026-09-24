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

A FOUR-PORT FILE IS A DIFFERENTIAL CHANNEL, and its transfer is the mixed-mode SDD21, not S21 —
S21 is one single-ended term and on a coupled pair it reports the AVERAGE of the two modes. The
port convention is not recorded in a Touchstone 1.0 file, so it is a required argument and it is
MEASURED against the alternatives rather than trusted:

    y = touchstone_channel(x, "pair.s4p", grid=g, ports="13_24")   # pairs (1,3) and (2,4)
    check_pairing(f, S, "13_24")                 # is that pairing the through path at all?
    sdd21 = mixed_mode_term(S, "13_24")          # or the whole (nf,4,4) matrix: se2mm(S, ...)

See the comment above `PAIR_CONVENTIONS` for the 6.2 dB this costs, and the comment above
`_band_energy` for the two silent lies about the band edges (a truncation above the file's top
frequency, and a DC block below its lowest) that `band=` and `dc=` now make explicit.

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

import re
import warnings

import numpy as np

_FUNIT = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}


def _n_ports_from_ext(path):
    ext = str(path).rsplit(".", 1)[-1].lower()
    if ext.startswith("s") and ext.endswith("p") and ext[1:-1].isdigit():
        return int(ext[1:-1])
    return None


class TouchstoneResult(tuple):
    """``(freqs, S)`` -- unpacks exactly like the plain tuple `read_touchstone` always
    returned -- plus ``.z0``, the file's own reference impedance (the option line's ``R``
    value, or 50 if unstated). A ``[Reference]`` block naming a DIFFERENT impedance per port
    (Touchstone 2.0) makes ``.z0`` a per-port array instead of a scalar; `renormalize_s`
    only handles the uniform (scalar) case."""

    def __new__(cls, freqs, S, z0):
        obj = super().__new__(cls, (freqs, S))
        obj.z0 = z0
        return obj


def read_touchstone(path, n_ports=None):
    """Read a Touchstone (.sNp) file. Returns ``(freqs_hz, S)`` where ``S`` has shape
    ``(nf, n, n)`` complex, plus a ``.z0`` attribute (the file's own reference impedance --
    the option line's ``R`` value, 50 if unstated, or a per-port array from a Touchstone 2.0
    ``[Reference]`` block). Supports RI / MA / DB formats and HZ/KHZ/MHZ/GHZ; comments
    (``!``) and multi-line frequency rows are handled. Port count is taken from the ``.sNp``
    extension (override with ``n_ports``)."""
    with open(path) as fh:
        text = fh.read()
    return _parse_touchstone(text, n_ports if n_ports else (_n_ports_from_ext(path) or 2))


def _parse_touchstone(text, n):
    funit, fmt, z0 = 1e9, "MA", 50.0
    ref = None                                             # a Touchstone 2.0 [Reference] block
    nums = []
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("[reference]"):
            toks = line.split()[1:]
            j = i
            while len(toks) < n and j + 1 < len(lines):
                j += 1
                cont = lines[j].split("!", 1)[0].strip()
                if not cont or cont.startswith("[") or cont.startswith("#"):
                    break
                toks += cont.split()
            if toks:
                vals = [float(t) for t in toks[:n]]
                ref = vals[0] if len(set(vals)) == 1 else np.asarray(vals, float)
            continue
        if line.startswith("["):                           # an unhandled Touchstone 2.0 keyword
            continue
        if line.startswith("#"):
            toks = line[1:].split()
            if toks:
                funit = _FUNIT[toks[0].upper()]
            if len(toks) >= 3:
                fmt = toks[2].upper()
            if "R" in (t.upper() for t in toks):
                ri = [t.upper() for t in toks].index("R")
                if ri + 1 < len(toks):
                    z0 = float(toks[ri + 1])
            continue
        nums.extend(float(t) for t in line.split())
    if ref is not None:
        z0 = ref
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
    return TouchstoneResult(freqs, S, z0)


def renormalize_s(S, z0_old, z0_new):
    """Renormalize an S-matrix ``(nf, n, n)`` from a UNIFORM real reference impedance
    ``z0_old`` (the same on every port) to ``z0_new``, via ``S' = (S - G*I)(I - G*S)^-1``
    with ``G = (z0_new - z0_old) / (z0_new + z0_old)`` -- the standard closed form for a
    reference-impedance change applied equally to every port. ``z0_old == z0_new`` is the
    identity (no computation, no rounding). Does not handle a per-port `[Reference]` block
    whose values differ -- that needs a per-port renormalization this does not attempt."""
    if z0_old == z0_new:
        return S
    S = np.asarray(S, complex)
    g = (z0_new - z0_old) / (z0_new + z0_old)
    n = S.shape[-1]
    eye = np.eye(n)
    out = np.empty_like(S)
    for k in range(S.shape[0]):
        out[k] = (S[k] - g * eye) @ np.linalg.inv(eye - g * S[k])
    return out


def write_touchstone(path, freqs_hz, S, fmt="RI", funit="HZ", z0=50.0):
    """Write ``(freqs_hz, S)`` (S shape ``(nf, n, n)``) to a Touchstone file. Mainly for
    tests and round-trips; RI format by default. ``z0`` is the option line's ``R`` value
    (the reference impedance the S-matrix was measured/computed at), 50 by default."""
    S = np.asarray(S)
    n = S.shape[1]
    scale = _FUNIT[funit.upper()]
    lines = [f"# {funit.upper()} S {fmt.upper()} R {z0:g}"]
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


# =====================================================================================
# MDIF -- one S-matrix over one or more swept independent variables
# =====================================================================================
# A Touchstone file is one network. An MDIF (.mdf) file is a sequence of ``BEGIN ... END``
# blocks, each one full S-matrix sweep over frequency at one point in an outer parameter sweep
# (a stub length, a via stackup, a temperature) stated by that block's own ``VAR`` lines. The
# format has no single governing standard the way Touchstone's is versioned -- tool vendors
# disagree on block names, VAR placement and whether the S-parameter columns carry RI, MA or dB
# pairs. This reads exactly ONE well-documented dialect (below) and RAISES, naming the line, on
# anything outside it, rather than guessing at a layout it cannot confirm:
#
#     BEGIN ACDATA
#     VAR stub_mm(real) = 0.20
#     VAR z0(real) = 50.0
#     %Freq(GHz) S[1,1] S[2,1] S[1,2] S[2,2]
#     1.0  0.11 -30.0  0.89 10.0  0.89 10.0  0.11 -30.0
#     ...
#     END
#     BEGIN ACDATA
#     VAR stub_mm(real) = 0.40
#     ...
#     END
#
# -- one ``%`` column-header line per block naming each column ``S[i,j]`` (so the port mapping
# is read off the header rather than assumed, unlike Touchstone's 2-port transpose quirk); each
# numeric column pair is REAL, IMAGINARY (no MA/dB support -- an unrecognised column raises);
# the frequency column's own name may carry a ``(UNIT)`` suffix (``Hz``/``kHz``/``MHz``/``GHz``),
# defaulting to Hz when absent; every block must share one frequency axis and the same set of
# VAR names, since together they are one N-dimensional sweep, not N unrelated networks.

_MDIF_S_RE = re.compile(r"[Ss]\[\s*(\d+)\s*,\s*(\d+)\s*\]")
_MDIF_UNIT_RE = re.compile(r"^(.*?)\(([A-Za-z]+)\)$")
_MDIF_VAR_RE = re.compile(r"^VAR\s+(\w+)(?:\([^)]*\))?\s*=\s*(.+)$", re.IGNORECASE)


class MdifSweep:
    """The result of `read_mdif`: an S-matrix swept over frequency AND one or more outer
    variables (a stub length, a Z0, ...).

    ``.freqs`` -- the one shared frequency axis, ``(nf,)`` Hz.
    ``.S`` -- ``(n_blocks, nf, n, n)`` complex, one full S-matrix per outer sweep point.
    ``.axes`` -- ``{var_name: array of length n_blocks}``, that VAR's value in each block, in
    the file's own block order -- e.g. ``{"stub_mm": [0.2, 0.4, ...], "z0": [50.0, 50.0, ...]}``.

    `.select(**kwargs)` names one point in the outer sweep and returns ``(freqs, S)`` for it,
    ``S`` shape ``(nf, n, n)`` -- exactly `sparam_channel`'s/`touchstone_channel`'s own slicing
    (``S[:, 1, 0]`` for S21) applies to the result unchanged."""

    __slots__ = ("freqs", "S", "axes")

    def __init__(self, freqs, S, axes):
        self.freqs = freqs
        self.S = S
        self.axes = axes

    def select(self, **kwargs):
        """The one block whose VAR values match every given `axis=value` (within float
        tolerance). Raises if zero or more than one block matches -- narrow the axes rather
        than silently take the first hit."""
        if not kwargs:
            raise ValueError("MdifSweep.select needs at least one axis=value")
        mask = np.ones(self.S.shape[0], dtype=bool)
        for name, val in kwargs.items():
            if name not in self.axes:
                raise ValueError(f"MdifSweep.select: unknown axis {name!r}; have {sorted(self.axes)}")
            mask &= np.isclose(self.axes[name], float(val), rtol=1e-9, atol=1e-12)
        hits = np.flatnonzero(mask)
        if len(hits) == 0:
            raise ValueError(f"MdifSweep.select: no block matches {kwargs}")
        if len(hits) > 1:
            raise ValueError(f"MdifSweep.select: {len(hits)} blocks match {kwargs}; narrow the axes")
        return self.freqs, self.S[hits[0]]


def read_mdif(path, n_ports=None):
    """Read an MDIF (.mdf) file of one or more `BEGIN`/`END` S-parameter sweeps into one
    `MdifSweep`. See the module comment above for the exact dialect read; anything outside it
    (an unrecognised column, MA/dB data, mismatched blocks) raises rather than guesses.
    `n_ports` overrides the port count inferred from the column header's `S[i,j]` count."""
    with open(path) as fh:
        text = fh.read()
    return _parse_mdif(text, n_ports)


def _parse_mdif_columns(header, n_ports):
    toks = header.split()
    if not toks:
        raise ValueError("MDIF: empty '%' column header")
    freq_tok, s_toks = toks[0], toks[1:]
    m = _MDIF_UNIT_RE.match(freq_tok)
    if m:
        unit = m.group(2).upper()
        if unit not in _FUNIT:
            raise ValueError(f"MDIF: unknown frequency unit {unit!r} in column {freq_tok!r}")
        scale = _FUNIT[unit]
    else:
        scale = 1.0
    pairs = []
    for tok in s_toks:
        m = _MDIF_S_RE.fullmatch(tok)
        if not m:
            raise ValueError(
                f"MDIF: unrecognised column {tok!r} -- only 'S[i,j]' real/imaginary pairs are "
                "supported (MA/dB dialects are not)")
        pairs.append((int(m.group(1)), int(m.group(2))))
    n = n_ports
    if n is None:
        root = int(round(len(pairs) ** 0.5))
        if root * root != len(pairs):
            raise ValueError(
                f"MDIF: {len(pairs)} S-parameter columns is not a perfect square; pass n_ports=")
        n = root
    elif len(pairs) != n * n:
        raise ValueError(f"MDIF: {len(pairs)} S-parameter columns, expected {n * n} for a {n}-port file")
    return scale, pairs, n


def _parse_mdif(text, n_ports):
    blocks = []
    in_block, cur_vars, cols, rows = False, None, None, None
    for raw in text.splitlines():
        stripped = raw.split("!", 1)[0].strip()
        if not stripped:
            continue
        low = stripped.lower()
        if low.startswith("begin"):
            if in_block:
                raise ValueError("MDIF: nested BEGIN before a matching END")
            in_block, cur_vars, cols, rows = True, {}, None, []
            continue
        if not in_block:
            continue                                       # ignore anything outside a block
        if low.startswith("end"):
            if cols is None:
                raise ValueError("MDIF: a block ended with no '%' column header")
            if not rows:
                raise ValueError("MDIF: a block has a column header but no data rows")
            blocks.append((cur_vars, cols, rows))
            in_block = False
            continue
        if stripped.startswith("%"):
            cols = _parse_mdif_columns(stripped[1:], n_ports)
            continue
        m = _MDIF_VAR_RE.match(stripped)
        if m:
            cur_vars[m.group(1)] = float(m.group(2))
            continue
        if low.startswith("var"):
            raise ValueError(f"MDIF: cannot parse VAR line {stripped!r}")
        if cols is None:
            raise ValueError(f"MDIF: data row before a '%' column header: {stripped!r}")
        _scale, pairs, _n = cols
        toks = stripped.split()
        expected = 1 + 2 * len(pairs)
        if len(toks) != expected:
            raise ValueError(f"MDIF: data row has {len(toks)} numbers, expected {expected}: {stripped!r}")
        rows.append([float(t) for t in toks])
    if in_block:
        raise ValueError("MDIF: BEGIN with no matching END")
    if not blocks:
        raise ValueError("MDIF: no BEGIN/END data block found")

    var_sets = [set(vs) for vs, _, _ in blocks]
    if any(vs != var_sets[0] for vs in var_sets[1:]):
        raise ValueError("MDIF: blocks declare different VAR names; cannot form one sweep")
    var_names = list(blocks[0][0])
    n_ref = blocks[0][1][2]

    freqs0 = None
    S_list = []
    axes = {name: [] for name in var_names}
    for vs, (scale, pairs, n), rows_b in blocks:
        if n != n_ref:
            raise ValueError("MDIF: blocks disagree on port count")
        arr = np.asarray(rows_b, float)
        f = arr[:, 0] * scale
        if freqs0 is None:
            freqs0 = f
        elif len(f) != len(freqs0) or not np.allclose(f, freqs0):
            raise ValueError("MDIF: blocks do not share one frequency axis")
        S = np.zeros((len(f), n, n), complex)
        for col_idx, (i, j) in enumerate(pairs):
            S[:, i - 1, j - 1] = arr[:, 1 + 2 * col_idx] + 1j * arr[:, 2 + 2 * col_idx]
        S_list.append(S)
        for name in var_names:
            axes[name].append(vs[name])

    axes = {name: np.asarray(v, float) for name, v in axes.items()}
    return MdifSweep(freqs0, np.stack(S_list, axis=0), axes)


# =====================================================================================
# MIXED-MODE — a 4-port differential file's transfer is SDD21, not S21
# =====================================================================================
# A differential channel measured as four single-ended ports does NOT hand you its transfer
# function in any one S-parameter. The wave a differential driver launches is the odd mode; the
# wave the receiver's subtractor keeps is the odd mode; the term that relates them is SDD21, and
# it is a combination of FOUR single-ended terms:
#
#     SDD21 = 0.5 * (S[o+,i+] - S[o+,i-] - S[o-,i+] + S[o-,i-])
#
# The habit of reaching for `S21` is not a small error. For a coupled pair the single-ended
# through term is the AVERAGE of the two modes,
#
#     S[o+,i+] = (T_odd + T_even) / 2        S[o-,i+] = (T_even - T_odd) / 2
#
# so `S21` reports (T_odd + T_even)/2 while the channel is T_odd. Where the two modes have
# accumulated the same phase the two agree to a fraction of a dB and the mistake is invisible;
# where they have not, |T_odd + T_even| partially CANCELS and the single-ended term reads far
# too small. The gap is exactly ``-20*log10|cos(pi * f * dtau)|`` for equal-loss modes separated
# by an even/odd skew `dtau`: 8.5 ps of skew costs 0.2 dB at 8 GHz and 42 ps of skew costs
# 6.2 dB at 8 GHz. A caller who validates on a low-skew pair and ships against a high-skew one
# ships a 6 dB error in the channel, and every eye and every margin downstream of it.
#
# THE PORT CONVENTION IS NOT DISCOVERABLE FROM THE FILE. Touchstone 1.0 records four ports and
# no statement of which two are a pair. Two orderings are in circulation and they disagree about
# every index:
#
#     "12_34"   pair 1 = ports (1, 2)   pair 2 = ports (3, 4)   through term is S31
#     "13_24"   pair 1 = ports (1, 3)   pair 2 = ports (2, 4)   through term is S21
#
# Guess wrong and the four terms you combine are not a transfer at all — with the pairing
# transposed, SDD21 is built from near-end intra-pair coupling, which on a matched pair is ZERO.
# So the convention is a REQUIRED argument here, and `check_pairing` measures the choice against
# the alternatives rather than trusting it.

PAIR_CONVENTIONS = {
    "12_34": ((1, 2), (3, 4)),
    "13_24": ((1, 3), (2, 4)),
    "14_23": ((1, 4), (2, 3)),
}
PAIR_CONVENTIONS["1234"] = PAIR_CONVENTIONS["12_34"]
PAIR_CONVENTIONS["1324"] = PAIR_CONVENTIONS["13_24"]
PAIR_CONVENTIONS["1423"] = PAIR_CONVENTIONS["14_23"]

# The mixed-mode matrix `se2mm` returns is indexed by MODE, in this order.
MM_MODES = ("d1", "d2", "c1", "c2")
MM_TERMS = {                        # name -> (row, col) into that matrix
    "SDD11": (0, 0), "SDD12": (0, 1), "SDD21": (1, 0), "SDD22": (1, 1),
    "SDC11": (0, 2), "SDC12": (0, 3), "SDC21": (1, 2), "SDC22": (1, 3),
    "SCD11": (2, 0), "SCD12": (2, 1), "SCD21": (3, 0), "SCD22": (3, 1),
    "SCC11": (2, 2), "SCC12": (2, 3), "SCC21": (3, 2), "SCC22": (3, 3),
}


def diff_pairs(ports, n_ports=4):
    """Normalise a differential port-convention spec to ``((i_p, i_n), (o_p, o_n))``, 1-based.

    Accepts a name from `PAIR_CONVENTIONS` (``"12_34"``, ``"13_24"``, ``"14_23"``, or the same
    without the underscore) or the pairing written out: ``((1, 2), (3, 4))`` means ports 1 and 2
    are the P and N of the DRIVEN pair and ports 3 and 4 the P and N of the RECEIVING pair.

    Raises rather than guessing. Every port must appear exactly once and lie in ``1..n_ports``:
    a pairing that names port 5 of a 4-port file, or names port 2 twice, is a caller error that
    used to surface as an all-zero response and nothing else."""
    if isinstance(ports, str):
        key = ports.strip().lower()
        if key not in PAIR_CONVENTIONS:
            raise ValueError(
                f"unknown differential port convention {ports!r}; known: "
                f"{sorted(set(k for k in PAIR_CONVENTIONS if '_' in k))} "
                f"(or write the pairing out, e.g. ports=((1, 2), (3, 4)))")
        pairs = PAIR_CONVENTIONS[key]
    else:
        pairs = tuple(tuple(int(p) for p in pr) for pr in ports)
    if len(pairs) != 2 or any(len(pr) != 2 for pr in pairs):
        raise ValueError(f"differential ports= must be two pairs of two ports, got {ports!r}")
    flat = [p for pr in pairs for p in pr]
    if sorted(flat) != sorted(set(flat)):
        raise ValueError(f"differential ports={ports!r} names a port twice: {flat}")
    bad = [p for p in flat if not 1 <= p <= n_ports]
    if bad:
        raise ValueError(f"differential ports={ports!r} names port(s) {bad}, outside 1..{n_ports}")
    return tuple((int(a), int(b)) for a, b in pairs)


def se2mm(S, ports):
    """Single-ended to mixed-mode. ``S`` is ``(nf, n, n)``; returns ``(nf, 4, 4)`` indexed by
    `MM_MODES` = (d1, d2, c1, c2), so ``M[:, 1, 0]`` is SDD21 and ``M[:, 3, 0]`` is SCD21.

    The transform is the orthonormal one, ``M = T S T^T`` with rows
    ``d_k = (e_p - e_n)/sqrt(2)``, ``c_k = (e_p + e_n)/sqrt(2)``. T is orthogonal, so this is a
    similarity transform: it conserves power, and the mixed-mode reference impedances it implies
    are ``2*Z0`` differential and ``Z0/2`` common — the ordinary convention, and the one that
    makes SDD21 of an UNCOUPLED pair come out exactly equal to the single-ended transfer of one
    half of it (which is the closed-form check in the tests).

    ``ports`` is anything `diff_pairs` accepts."""
    S = np.asarray(S, complex)
    if S.ndim != 3 or S.shape[1] != S.shape[2]:
        raise ValueError(f"se2mm: S must be (nf, n, n), got {S.shape}")
    n = S.shape[1]
    (ip, inn), (op, onn) = diff_pairs(ports, n_ports=n)
    T = np.zeros((4, n))
    r = 1.0 / np.sqrt(2.0)
    T[0, ip - 1], T[0, inn - 1] = r, -r          # d1
    T[1, op - 1], T[1, onn - 1] = r, -r          # d2
    T[2, ip - 1], T[2, inn - 1] = r, r           # c1
    T[3, op - 1], T[3, onn - 1] = r, r           # c2
    return T @ S @ T.T


def mixed_mode_term(S, ports, term="SDD21"):
    """One named mixed-mode term of a 4-port file as a complex array over frequency.

    ``term`` is a key of `MM_TERMS`: SDD21 is the differential insertion loss (the channel a
    differential receiver sees), SDD11 the differential return loss, SCD21 the differential-to-
    common conversion an asymmetric pair radiates, SCC21 the common-mode transfer."""
    t = str(term).upper()
    if t not in MM_TERMS:
        raise ValueError(f"unknown mixed-mode term {term!r}; known: {sorted(MM_TERMS)}")
    i, j = MM_TERMS[t]
    return se2mm(S, ports)[:, i, j]


def check_pairing(freqs, S, ports, term="SDD21", tol_db=6.0, band_frac=0.05, raise_=True):
    """MEASURE a differential port convention instead of trusting it.

    A 4-port file states no pairing, so a caller's `ports=` is a claim. This checks it the only
    way the file allows: it computes ``term`` under the chosen pairing and under the two other
    ways of partitioning four ports into two pairs, and compares their magnitudes over the
    LOW-frequency band (below ``band_frac`` of the file's top frequency), where any passive
    through path is close to 0 dB whatever its loss at Nyquist. A transposed pairing builds the
    term out of intra-pair coupling, which on a matched pair is zero — so the chosen pairing
    coming out far below an alternative is the signature of a wrong convention, not of a lossy
    channel.

    Returns a dict: ``chosen``, ``candidates`` (name -> low-band dB), ``best``, ``margin_db``
    (chosen minus best; <= 0), ``ok``. With ``raise_=True`` (the default) a chosen pairing more
    than ``tol_db`` below the best, or one that is identically zero, raises `ValueError` naming
    the pairing that does work."""
    freqs = np.asarray(freqs, float)
    S = np.asarray(S, complex)
    lo = freqs <= max(freqs.min(), band_frac * freqs.max())
    if not lo.any():
        lo = np.zeros(len(freqs), bool)
        lo[0] = True

    def level(pr):
        h = mixed_mode_term(S, pr, term=term)
        return float(20.0 * np.log10(np.median(np.abs(h[lo])) + 1e-30)), h

    want = diff_pairs(ports, n_ports=S.shape[1])
    cand = {}
    for name in ("12_34", "13_24", "14_23"):
        cand[name] = level(PAIR_CONVENTIONS[name])[0]
    chosen_db, h = level(want)
    best = max(cand, key=cand.get)
    out = {"chosen": want, "chosen_db": chosen_db, "candidates": cand, "best": best,
           "best_db": cand[best], "margin_db": chosen_db - cand[best],
           "peak": float(np.max(np.abs(h))), "term": str(term).upper()}
    out["ok"] = out["peak"] > 1e-12 and out["margin_db"] > -abs(tol_db)
    if raise_ and not out["ok"]:
        tab = ", ".join(f"{k}={v:.2f} dB" for k, v in sorted(cand.items()))
        raise ValueError(
            f"port convention ports={ports!r} does not look like a differential through path: "
            f"{out['term']} is {chosen_db:.2f} dB in the low band (peak |{out['term']}| over the "
            f"whole band = {out['peak']:.3e}), while pairing {best!r} gives {cand[best]:.2f} dB. "
            f"All three pairings: {tab}. A transposed pairing builds {out['term']} out of "
            f"intra-pair coupling, which on a matched pair is zero — this used to return an "
            f"all-zero response and raise nothing. Pass ports={best!r} if that is your file's "
            f"ordering, or check_pairing(..., raise_=False) to inspect without raising.")
    return out


# =====================================================================================
# THE BAND EDGES — what a file does NOT say, and the two silent lies about it
# =====================================================================================
# A Touchstone file covers f_min..f_max. The transform grid covers 0..fs/2. Those are different
# intervals, and the old behaviour filled BOTH gaps with zero:
#
#   ABOVE f_max   a 15 GHz file driven at 256 GS/s zeroes everything from 15 to 128 GHz. For an
#                 ideal (rectangular-edge) 16 GBd NRZ record that is 9.71 % of the record's
#                 power, deleted, with no warning; for a 0.15 UI rise time at 256 GS/s it is
#                 1.93 %, and for 0.35 UI it is 0.011 %. Which of those is acceptable is the
#                 CALLER's judgement, so the amount removed is MEASURED from the record and
#                 either warned about or refused — not assumed away.
#
#   BELOW f_min   most measured files start at 10-50 MHz, so bin 0 was zeroed and every record
#                 came out DC-BLOCKED. A passive through path does not block DC; its transmission
#                 there is real and close to its magnitude at the lowest measured point. Feeding
#                 a constant in and getting exactly zero out is not a measurement of anything.
#                 The default now carries the low band: H(0) = |S(f_min)|, real, interpolated up
#                 to the first measured point.
#
# `band=` and `dc=` name those two choices, and each has a mode that restores the old behaviour
# so the two can be compared directly.

_BAND_MODES = ("refuse", "zero", "hold")
_DC_MODES = ("extend", "zero")


def _band_energy(x, dt, f_lo, f_hi):
    """Fraction of ``x``'s power below `f_lo` and above `f_hi`, measured on the record itself.

    This is the honest number for "how much does truncating at the file's edge remove": it is a
    property of THIS record, not of the modulation in the abstract. A rectangular-edge NRZ
    record and a 0.35 UI one, at the same baud and through the same file, differ by three orders
    of magnitude here."""
    X = np.abs(np.fft.rfft(np.asarray(x, float))) ** 2
    fg = np.fft.rfftfreq(len(x), d=dt)
    tot = float(X.sum())
    if tot <= 0:
        return 0.0, 0.0
    return float(X[fg < f_lo].sum()) / tot, float(X[fg > f_hi].sum()) / tot


def _interp_response(fg, freqs, z, band="zero", dc="zero"):
    """Interpolate a measured complex response onto the grid `fg`, deciding both band edges.

    `dc`   ``"extend"``  H(0) = |z(f_min)| (real), linearly interpolated up to f_min. A passive
                         path's transmission is real at DC and smooth through f_min, so this
                         adds no structure — it just stops deleting the low band.
           ``"zero"``    the old behaviour: everything below f_min is zero, so every record is
                         DC-blocked by the channel.
    `band` ``"zero"``    everything above f_max is zero (a brick wall nobody measured).
           ``"hold"``    hold |z(f_max)| with the group delay of the top of the file continued —
                         an EXTRAPOLATION, offered because a brick wall is also an extrapolation
                         and at least this one is smooth. Never a default.
           ``"refuse"``  same fill as ``"zero"``; the refusal itself is the caller's gate in
                         `sparam_channel`, which knows the record and so can measure what the
                         truncation costs."""
    fg = np.asarray(fg, float)
    freqs = np.asarray(freqs, float)
    z = np.asarray(z, complex)
    if len(freqs) != len(z):
        raise ValueError(f"response: {len(freqs)} frequencies but {len(z)} points")
    order = np.argsort(freqs)
    freqs, z = freqs[order], z[order]
    fx, zx = freqs, z
    if dc == "extend" and freqs[0] > 0.0:
        fx = np.concatenate(([0.0], freqs))
        zx = np.concatenate(([complex(abs(z[0]), 0.0)], z))
    H = np.interp(fg, fx, zx.real) + 1j * np.interp(fg, fx, zx.imag)
    below, above = fg < fx[0], fg > freqs[-1]
    if dc == "zero":
        H[below] = 0.0
    elif dc != "extend":
        raise ValueError(f"dc= must be one of {_DC_MODES}, got {dc!r}")
    if band in ("zero", "refuse"):
        H[above] = 0.0
    elif band == "hold":
        if above.any():
            k = max(2, min(8, len(freqs)))
            ph = np.unwrap(np.angle(z[-k:]))
            w = 2.0 * np.pi * freqs[-k:]
            tau = -np.polyfit(w, ph, 1)[0]
            H[above] = abs(z[-1]) * np.exp(1j * (ph[-1] - 2.0 * np.pi * tau
                                                 * (fg[above] - freqs[-1])))
    else:
        raise ValueError(f"band= must be one of {_BAND_MODES}, got {band!r}")
    return H


def sparam_channel(x, freqs, s21, grid=None, dt=None, linear=True, guard=None,
                   band="refuse", dc="extend", band_tol=1e-3, method="fft"):
    """Apply a measured transfer ``s21`` (complex, sampled at ``freqs`` in Hz) to ``x`` as a
    frequency-domain channel. Provide the sample spacing via ``grid=Grid(...)`` or ``dt``.
    Unlike the analytic model this reproduces resonances and structure.

    THE BAND EDGES ARE NOW A DECISION, NOT A DEFAULT (see the comment above `_band_energy`):

    ``band="refuse"``  measure the fraction of THIS record's power that lies above the file's
                       top frequency; warn if any of it is being removed, and RAISE if it
                       exceeds ``band_tol`` (0.1 %). A 15 GHz file cannot be asked to answer for
                       a record with 2 % of its energy above 15 GHz, and silently zeroing it is
                       how a channel loses 9 % of a spectrum without anyone noticing.
    ``band="zero"``    the previous behaviour: zero above f_max, warn with the measured fraction.
    ``band="hold"``    extrapolate above f_max (documented in `_interp_response`).

    ``dc="extend"``    carry the low band: H(0) = |s21(f_min)|, real. The default, because the
                       previous default DC-BLOCKED every record whose file did not start at 0 Hz.
    ``dc="zero"``      the previous behaviour, with a warning naming the power removed.

    The convolution is LINEAR (`physics.apply_transfer`): the record is zero-padded past the
    response's own length before transforming, so the channel's answer to the record's tail
    does not wrap onto its head. ``linear=False`` restores the pinned-length circular form."""
    from . import physics as P
    x = np.asarray(x, float)
    if dt is None:
        if grid is None:
            raise ValueError("sparam_channel needs grid=Grid(...) or dt=")
        dt = grid.dt
    freqs = np.asarray(freqs, float)
    s21 = np.asarray(s21, complex)
    if band not in _BAND_MODES:
        raise ValueError(f"band= must be one of {_BAND_MODES}, got {band!r}")
    if dc not in _DC_MODES:
        raise ValueError(f"dc= must be one of {_DC_MODES}, got {dc!r}")

    f_lo, f_hi, nyq = float(freqs.min()), float(freqs.max()), 0.5 / dt
    lost_lo, lost_hi = _band_energy(x, dt, f_lo, f_hi)
    if nyq > f_hi and lost_hi > 0.0:
        msg = (f"the response stops at {f_hi/1e9:.3f} GHz but the record reaches "
               f"{nyq/1e9:.3f} GHz: {100*lost_hi:.4g} % of the record's power lies above the "
               f"measured band")
        if band == "refuse":
            if lost_hi > band_tol:
                raise ValueError(
                    msg + f", above the {100*band_tol:.4g} % band_tol. Zeroing it would delete "
                    f"that power with no trace in the output. Choose: resample the record so "
                    f"its Nyquist is <= {f_hi/1e9:.3f} GHz, use a file that reaches "
                    f"{nyq/1e9:.3f} GHz, raise band_tol= if that loss is acceptable, or state "
                    f"the extrapolation with band='hold' / the truncation with band='zero'.")
            warnings.warn(msg + " and is being zeroed (within band_tol).", RuntimeWarning,
                          stacklevel=2)
        else:
            warnings.warn(msg + f" and band={band!r}.", RuntimeWarning, stacklevel=2)
    if dc == "zero" and f_lo > 0.0 and lost_lo > 0.0:
        warnings.warn(f"dc='zero': the response starts at {f_lo/1e6:.4g} MHz and everything "
                      f"below it is zeroed, removing {100*lost_lo:.4g} % of the record's power "
                      f"and DC-BLOCKING it. dc='extend' carries the low band.",
                      RuntimeWarning, stacklevel=2)
    elif dc == "extend" and lost_lo > band_tol:
        warnings.warn(f"dc='extend': the response starts at {f_lo/1e6:.4g} MHz and "
                      f"{100*lost_lo:.4g} % of the record's power is below that, so that much of "
                      f"the output rests on holding |S(f_min)| down to DC rather than on a "
                      f"measurement.", RuntimeWarning, stacklevel=2)

    def make_H(nfft):
        return _interp_response(np.fft.rfftfreq(nfft, d=dt), freqs, s21, band=band, dc=dc)

    return P.apply_transfer(x, make_H, linear=linear, guard=guard, method=method)


def check_response(freqs, S, tol=1e-6, n=None):
    """Report whether a measured S-parameter response is PASSIVE and whether its impulse
    response is CAUSAL, without altering the samples -- the same idea `sparam_channel`'s
    ``band=refuse`` already applies to a truncated band: name the problem, leave the data alone.

    ``S`` is either a single term ``(nf,)`` complex (e.g. ``S[:, 1, 0]``, as read off
    `read_touchstone`) or a full scattering matrix ``(nf, n, n)``.

        info = check_response(freqs, S[:, 1, 0])
        info["passive"]           True if the max singular value is <= 1 (+ tol) at every
                                   frequency. For a single term this is just |S(f)| <= 1 --
                                   NECESSARY but not sufficient for passivity of the full
                                   network; pass the full matrix for the real test.
        info["max_singular_value"]  the worst-case value the check above is against.
        info["precursor_frac"]    the fraction of impulse-response ENERGY that arrives before
                                   t=0 -- ~0 for a causal response, and the same construction
                                   `tests/audit_causality.py`'s `pre_energy_fraction` uses
                                   elsewhere in this repo (a unit impulse centred in an
                                   n-sample buffer, room on both sides so a circular wrap
                                   cannot masquerade as a real precursor).

    Measured on a single term, ``precursor_frac`` is a diagnostic on the term you handed it,
    not the network as a whole -- a mixed-mode term built from four single-ended ones
    (`mixed_mode_term`) may show energy the individual terms do not.
    """
    freqs = np.asarray(freqs, float)
    S = np.asarray(S, complex)
    is_matrix = S.ndim == 3
    if is_matrix:
        sv = np.array([np.linalg.svd(S[k], compute_uv=False)[0] for k in range(S.shape[0])])
        term = S[:, 1, 0] if S.shape[1] > 1 else S[:, 0, 0]
    else:
        sv = np.abs(S)
        term = S
    max_sv = float(sv.max()) if sv.size else 0.0
    passive = bool(np.all(sv <= 1.0 + tol))

    fmax = float(freqs.max())
    n = n or max(256, int(2 ** np.ceil(np.log2(2 * len(freqs)))))
    fs_eff = 2.0 * fmax
    fu = np.fft.rfftfreq(n, d=1.0 / fs_eff)
    mag = np.interp(fu, freqs, np.abs(term))
    ph = np.interp(fu, freqs, np.unwrap(np.angle(term)))
    m = n // 2
    shift = np.exp(-2j * np.pi * fu * m / fs_eff)   # centres the impulse at index m
    h = np.fft.irfft(mag * np.exp(1j * ph) * shift, n)
    e = h ** 2
    precursor_frac = float(e[:m].sum() / (e.sum() + 1e-300))
    return {"passive": passive, "max_singular_value": max_sv, "precursor_frac": precursor_frac}


def touchstone_channel(x, path, grid=None, dt=None, ports=(2, 1), n_ports=None,
                       linear=True, guard=None, mode=None, term="SDD21",
                       band="refuse", dc="extend", band_tol=1e-3, check=True, z0=None,
                       method="fft"):
    """Read a Touchstone file and apply one of its transfers to ``x``.

    ``ports`` says WHICH transfer, and its form says which kind:

        ports=(2, 1)              a SINGLE-ENDED term, ``S[2,1]``. Fine for a 2-port file.
        ports="13_24"             DIFFERENTIAL: pairs (1,3) and (2,4); applies SDD21.
        ports="12_34"             DIFFERENTIAL: pairs (1,2) and (3,4); applies SDD21.
        ports=((1, 2), (3, 4))    the same, written out as ((in+, in-), (out+, out-)).

    On a file of FOUR OR MORE PORTS a bare single-ended pair now RAISES. A 4-port file is a
    differential channel measured single-ended; its transfer is SDD21, which is a combination of
    four of its terms and is not equal to any one of them (see the comment above
    `PAIR_CONVENTIONS`). Pass a pairing, or say ``mode="se"`` to mean the single-ended term on
    purpose. ``term=`` picks a different mixed-mode term (SDD11, SCD21, SCC21, ...).

    ``z0=`` renormalizes the file's S-matrix to a stated system impedance FIRST, if the file's
    own reference impedance (`read_touchstone`'s ``.z0``, from the option line's ``R`` or 50 if
    unstated) differs from it -- see `renormalize_s`. Default ``None`` is the file's own
    impedance, unchanged (an existing ``R 50`` file renders exactly as before). Left unhandled:
    a Touchstone 2.0 file whose ``[Reference]`` block states a DIFFERENT impedance per port --
    ``z0=`` on one of those raises rather than renormalizing it wrongly.

    ``check=True`` runs `check_pairing`: the chosen pairing is measured against the other two
    partitions of the four ports, and a pairing that yields an all-zero or far-too-small
    response raises instead of returning zeros. ``band=``/``dc=``/``band_tol=`` are
    `sparam_channel`'s band-edge decisions."""
    result = read_touchstone(path, n_ports=n_ports)
    freqs, S = result
    if z0 is not None:
        if isinstance(result.z0, np.ndarray):
            raise ValueError(
                f"{path}: this file states a per-port reference impedance "
                f"({result.z0.tolist()}), which renormalize_s does not handle; renormalize "
                f"it yourself with a per-port method before calling touchstone_channel")
        S = renormalize_s(S, result.z0, z0)
    n = S.shape[1]
    is_pairing = isinstance(ports, str) or (
        len(ports) == 2 and all(isinstance(p, (tuple, list)) for p in ports))
    explicit = mode is not None
    if mode is None:
        mode = "diff" if is_pairing else "se"
    if mode not in ("se", "diff"):
        raise ValueError(f"mode= must be 'se' or 'diff', got {mode!r}")
    if mode == "diff":
        if not is_pairing:
            raise ValueError(f"mode='diff' needs a pairing, e.g. ports='13_24' or "
                             f"ports=((1, 2), (3, 4)); got ports={ports!r}")
        pairs = diff_pairs(ports, n_ports=n)
        if check:
            check_pairing(freqs, S, pairs, term=term)
        h = mixed_mode_term(S, pairs, term=term)
    else:
        if is_pairing:
            raise ValueError(f"mode='se' takes two port numbers, e.g. ports=(2, 1); "
                             f"got ports={ports!r}")
        if n >= 4 and not explicit:
            raise ValueError(
                f"{path}: a {n}-port file is a differential channel measured single-ended, and "
                f"its transfer is SDD21 — a combination of four of its terms, not "
                f"S[{ports[0]},{ports[1]}]. On a coupled pair the single-ended through term is "
                f"(T_odd + T_even)/2 while the channel is T_odd, and the two differ by "
                f"-20*log10|cos(pi*f*dtau)| for an even/odd skew dtau (0.2 dB at 8 GHz for "
                f"8.5 ps, 6.2 dB for 42 ps). Pass a port convention — ports='13_24' (pairs 1,3 "
                f"and 2,4) or ports='12_34' (pairs 1,2 and 3,4) — or, to take the single-ended "
                f"term on purpose, mode='se'.")
        i, j = int(ports[0]) - 1, int(ports[1]) - 1
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"{path}: ports={ports!r} outside 1..{n}")
        h = S[:, i, j]
    return sparam_channel(x, freqs, h, grid=grid, dt=dt, linear=linear, guard=guard,
                          method=method,
                          band=band, dc=dc, band_tol=band_tol)
# =====================================================================================
# CASCADED CHANNELS — topology, not one lumped block
# =====================================================================================
# A real fast path is package -> short trace -> connector -> longer trace -> via -> trace ->
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

    eps_r=4.0 gives 169.45 ps/inch. A caller that rounds c to 11.8 in/ns gets 169.5 from the same
    arithmetic — 0.03 % apart, which is 0.5 mil on a 1.66-inch echo, so the two conventions are
    interchangeable for a lag-to-distance estimate and NOT for a length pinned to a mil. Pass
    `ps_per_inch=` explicitly to any section that must pin a downstream convention."""
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
        H = P._min_phase_H(mag, n, half=True)      # n is even here, so this IS len(freqs)
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


def measured(freqs, path=None, ports=(2, 1), n_ports=None, S=None, file_freqs=None,
             band="zero", dc="zero", left=None, right=None):
    """One SECTION taken from a measured Touchstone file — a real connector or via dropped into
    an otherwise synthetic path. The file's full 2-port block (both reflections, both
    transmissions) is interpolated onto `freqs`; outside the measured band every element is
    zeroed, which makes the section an open stub there rather than an extrapolation nobody
    measured. `ports=(2,1)` names which two ports of an N-port file form the section.

    `left=`/`right=` treat the file as a DIFFERENTIAL section instead, for a 4-port (or more)
    block — a via, a connector — sitting between two differential `line` sections without
    dropping to a single-ended 2-port: each is a (P, N) port pair, 1-based, the same convention
    `diff_pairs` already uses (`left` is the driven/input pair, `right` the receiving/output
    pair). The section's own S-matrix becomes the mixed-mode SDD block (`se2mm`), so a chain of
    these composes exactly like a chain of ordinary 2-port sections. Both must be given, or
    neither.

    `band=`/`dc=` are `_interp_response`'s band-edge decisions and DEFAULT TO THE LEGACY
    ZEROING here, deliberately: a cascade section is a two-port block whose neighbours' bounce
    series is closed over it, and "an open stub outside the measured band" is a defensible
    statement about a SECTION in a way it is not about a whole channel. `dc="extend"` is
    available and is what a section that carries DC (a connector, a package) should use — see
    `sparam_channel` for why zeroing the low band DC-blocks the record."""
    if (left is None) != (right is None):
        raise ValueError("measured: left= and right= must both be given, or neither")
    if S is None:
        file_freqs, Sfull = read_touchstone(path, n_ports=n_ports)
    else:
        Sfull = np.asarray(S, complex)
    freqs = np.asarray(freqs, float)

    def _ip(z):
        return _interp_response(freqs, file_freqs, z, band=band, dc=dc)

    if left is not None:
        n = Sfull.shape[1]
        Si = np.empty((len(freqs), n, n), complex)
        for a in range(n):
            for b in range(n):
                Si[:, a, b] = _ip(Sfull[:, a, b])
        mm = se2mm(Si, (tuple(left), tuple(right)))
        return TwoPort(freqs, mm[:, 0, 0], mm[:, 0, 1], mm[:, 1, 0], mm[:, 1, 1])

    i, j = ports[0] - 1, ports[1] - 1
    return TwoPort(freqs, _ip(Sfull[:, j, j]), _ip(Sfull[:, j, i]),
                   _ip(Sfull[:, i, j]), _ip(Sfull[:, i, i]))


def stub(freqs, length_in=None, td_ps=None, z0=50.0, end="open", eps_r=4.0, ps_per_in=None,
         sys_z0=50.0):
    """An OPEN or SHORT stub, as its own inline cascade SECTION: a length, a characteristic
    impedance, and an end condition — the exact lossless transmission-line formula, not the
    phenomenological band-pass shape `physics.resonant_reflection` fits for a resonance already
    measured:

        Z_in = -j*z0/tan(beta*L)   (open)          Z_in = j*z0*tan(beta*L)   (short)

    seen as a shunt admittance `Y = 1/Z_in` across the main (`sys_z0`-ohm) line, giving the
    standard shunt two-port

        S11 = S22 = -Y*sys_z0 / (2 + Y*sys_z0)      S21 = S12 = 2 / (2 + Y*sys_z0)

    which is lossless and reciprocal by construction, like `discontinuity`. An OPEN stub is
    invisible at DC (`Y -> 0`, S21 -> 1) and looks like a dead short at its quarter-wave
    resonance; a SHORT stub is the mirror image. `length_in`/`td_ps`/`eps_r`/`ps_per_in` size the
    electrical length exactly as `line` does."""
    freqs = np.asarray(freqs, float)
    if td_ps is None:
        if length_in is None:
            raise ValueError("stub() needs td_ps= or length_in=")
        pp = ps_per_in if ps_per_in is not None else ps_per_inch(eps_r)
        td_ps = float(length_in) * pp
    if end not in ("open", "short"):
        raise ValueError(f"stub: end must be 'open' or 'short', got {end!r}")
    beta_l = 2.0 * np.pi * freqs * (float(td_ps) * 1e-12)
    ratio = float(sys_z0) / float(z0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tan_bl = np.tan(beta_l)
        # Y*sys_z0, built without inverting Z_in, so the only pole left is the physical one --
        # Y -> infinity (a dead short: the open stub at its quarter-wave, the short stub at DC).
        yz = 1j * tan_bl * ratio if end == "open" else -1j * ratio / tan_bl
        s11 = -yz / (2.0 + yz)
        s21 = 2.0 / (2.0 + yz)
    shorted = ~np.isfinite(yz)
    s11 = np.where(shorted, -1.0 + 0j, s11)
    s21 = np.where(shorted, 0j, s21)
    return TwoPort(freqs, s11, s21, s21, s11)


def build_sections(path, freqs):
    """Turn a declarative path spec into `TwoPort` sections on `freqs`.

    `path` is a list of one-key dicts, in physical order from the driver to the receiver:

        [{"line": {"length_in": 1.0, "tand": 0.02, "eps_r": 4.0}},
         {"disc": {"gamma": 0.055}},                       # the connector, 1.0 in out
         {"line": {"length_in": 5.0}},
         {"disc": {"gamma": 0.055}},                       # the via, 6.0 in out
         {"line": {"length_in": 2.0}},
         {"file": {"path": "connector.s2p"}},              # a measured 2-port section
         {"file": {"path": "via.s4p", "left": (1, 3), "right": (2, 4)}},  # a measured 4-port,
                                                             # taken differentially (see `measured`)
         {"stub": {"length_in": 0.08, "z0": 50.0, "end": "open"}}]  # an open/short stub

    Plain JSON, so it round-trips through a recipe. A bare `TwoPort` may also appear in the
    list and is used as-is."""
    return [_section(sec, freqs, k) for k, sec in enumerate(path)]


def _section(sec, freqs, k=0):
    """One `TwoPort` from one path entry. Split out of `build_sections` so `cascade` can fold
    a path one section at a time: a section carries several complex arrays as long as the
    signal's own rfft axis, and materialising every section before folding any of them was the
    largest single allocation in a deep-record render."""
    if isinstance(sec, TwoPort):
        return sec
    if len(sec) != 1:
        raise ValueError(f"path[{k}]: expected one key of line|disc|file, got {sorted(sec)}")
    kind, args = next(iter(sec.items()))
    args = dict(args)
    if kind == "line":
        if args.get("trend") is not None:
            args["trend"] = tuple(float(t) for t in args["trend"])   # JSON gives a list
        return line(freqs, **args)
    if kind == "disc":
        g = args.get("gamma")
        if isinstance(g, (list, tuple)):
            g = np.asarray(g, complex)
        return discontinuity(freqs, g)
    if kind == "file":
        return measured(freqs, **args)
    if kind == "stub":
        return stub(freqs, **args)
    raise ValueError(f"path[{k}]: unknown section kind {kind!r} (line|disc|file|stub)")


def cascade(path, freqs):
    """Cascade a path spec (or a list of `TwoPort`s) into one `TwoPort` on `freqs`."""
    if len(path) == 0:
        raise ValueError("cascade: empty path")
    out = None
    for k, sec in enumerate(path):
        s = _section(sec, freqs, k)
        out = s if out is None else cascade_2port(out, s)
        del s                      # the fold consumed it; do not hold the whole path at once
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


def cascade_channel(x, path, grid=None, dt=None, node="load", eps_r_default=4.0,
                    linear=True, guard=None, method="fft"):
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

    The sections are built directly on the transform's rfft grid, so there is no interpolation
    of an analytic path and the delays are exact (a linear phase, not a rounded sample count —
    `physics.multi_reflection` rounds `td_ps` to the nearest sample).

    The convolution is LINEAR (`physics.apply_transfer`): the record is zero-padded past the
    cascade's own impulse-response length — which for a path with structure is dominated by the
    LAST echo's round trip — so no echo of the record's tail is delivered to its head. A path
    whose echoes take longer than the record itself simply pads further. ``linear=False``
    restores the pinned-length circular form for comparison."""
    from . import physics as P
    x = np.asarray(x, float)
    if dt is None:
        if grid is None:
            raise ValueError("cascade_channel needs grid=Grid(...) or dt=")
        dt = grid.dt
    if node not in ("load", "source"):
        raise ValueError(f"cascade_channel: node must be 'load' or 'source', got {node!r}")

    def make_H(nfft):
        tp = cascade(path, np.fft.rfftfreq(nfft, d=dt))
        H = tp.s21 if node == "load" else (1.0 + tp.s11)
        del tp                     # only H is needed past here; the other three ports are not
        return H

    return P.apply_transfer(x, make_H, linear=linear, guard=guard, method=method)

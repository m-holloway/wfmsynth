"""`sparam` against four measured defects: the mixed-mode transfer, the port convention, the
band edge, and the DC block.

Every assertion here is anchored to a CONSTRUCTED answer or to a file on disk, and every gate
has been observed failing on the behaviour it replaces — the legacy path is kept in this file
(`_legacy_*`) precisely so the gate can be pointed at it and seen to fire.

WHAT WAS WRONG, with the numbers that establish it
--------------------------------------------------
1. A 4-port file was read as if `S21` were its transfer. It is not. For a coupled pair the
   single-ended through term is ``(T_odd + T_even)/2`` while the channel a differential
   receiver sees is ``T_odd``, so the two differ by ``-20*log10|cos(pi*f*dtau)|`` for an
   even/odd skew `dtau`. Measured here on a constructed pair at 8 GHz: **0.20 dB** at 8.53 ps
   of skew and **6.21 dB** at 42.16 ps, both matching the closed form to < 0.01 dB. The 0.2 dB
   case is how the 6.2 dB case ships — a caller validates on the forgiving pair and believes it.
2. The port convention was positional and unvalidated. On `stub_diff_B.s4p` the old default
   ``ports=(2, 1)`` returns a response whose peak magnitude over the whole 10 MHz..40 GHz band
   is **exactly 0.0** and raises nothing (`test_a_transposed_pairing_returned_zeros`).
3. Everything above the file's top frequency was zeroed with no warning. Measured on a 16 GBd
   NRZ record through a 15 GHz file: **9.71 %** of an ideal rectangular-edge record's power,
   **1.94 %** at a 0.15 UI rise time and 256 GS/s, **0.027 %** at 0.35 UI. The first two are
   now refused, the third warned.
4. Every record was DC-blocked. A CONSTANT input through a passive through path must come out
   constant. On `stub_diff_A.s4p`: measured **0.1155** where the file's own
   ``|SDD21(f_min)| = 0.999757`` says **0.9997** — 18.7 dB low; ``dc="extend"`` returns
   **0.999747**, an error of 1.0e-5. On a constructed 10 MHz..40 GHz file the old path returns
   **-0.1597** against a truth of **0.9957** — the wrong SIGN, not merely the wrong size, since
   with bin 0 gone all that is left is the ringing of the record's own edges.
"""
import os
import warnings

import numpy as np
import pytest

from wfmsynth import physics as P
from wfmsynth import sparam as SP
from wfmsynth.grid import Grid

# The two .s4p files these tests read. They live outside this repo (this repo ships no binary
# fixtures), so absent is a SKIP, never a failure — point `WFMSYNTH_S4P_DIR` at a directory
# holding a pair of 4-port differential files to run the on-disk half of this file.
_REAL = os.environ.get("WFMSYNTH_S4P_DIR", "")
REAL_A = os.path.join(_REAL, "stub_diff_A.s4p")      # pairs (1,3) and (2,4) -> "13_24"
REAL_B = os.path.join(_REAL, "stub_diff_B.s4p")      # pairs (1,2) and (3,4) -> "12_34"
needs_real = pytest.mark.skipif(
    not (os.path.exists(REAL_A) and os.path.exists(REAL_B)),
    reason=f"4-port fixtures absent from {_REAL} (set WFMSYNTH_S4P_DIR)")

FREF = 8.0e9                      # the frequency every dB figure in this file is quoted at


# ------------------------------------------------------------------ constructed 4-ports
def _T(freqs, pairs):
    """The orthonormal single-ended -> mixed-mode transform, written out independently of
    `se2mm` so the test is not checking the implementation against itself."""
    (ip, inn), (op, onn) = pairs
    T = np.zeros((4, 4))
    r = 1.0 / np.sqrt(2.0)
    T[0, ip - 1], T[0, inn - 1] = r, -r
    T[1, op - 1], T[1, onn - 1] = r, -r
    T[2, ip - 1], T[2, inn - 1] = r, r
    T[3, op - 1], T[3, onn - 1] = r, r
    return T


def modal_4port(freqs, td_d, td_c, loss_d_db=9.0, loss_c_db=9.0, r_d=0.0, r_c=0.0, xconv=0.0):
    """A symmetric coupled pair built FROM ITS MODES, so its mixed-mode matrix is known exactly.

    A coupled pair of transmission lines carries exactly two modes. The odd (differential) mode
    and the even (common) mode each see their own effective permittivity, so each has its own
    one-way delay and its own loss — this is not an idealisation, it is why intra-pair skew and
    common-mode conversion exist at all. Each mode is a matched 2-port:

        T_d = 10**(-loss_d_db*sqrt(f/FREF)/20) * exp(-j*2*pi*f*td_d)      (sqrt-f loss)
        T_c = same with the even mode's numbers

    Returns ``(M, T_d, T_c)`` where M is ``(nf, 4, 4)`` in `SP.MM_MODES` order (d1, d2, c1, c2),
    so ``M[:, 1, 0]`` IS SDD21 by construction and nothing has been converted yet."""
    f = np.asarray(freqs, float)
    nf = len(f)
    Td = 10 ** (-(loss_d_db * np.sqrt(f / FREF)) / 20) * np.exp(-1j * 2 * np.pi * f * td_d)
    Tc = 10 ** (-(loss_c_db * np.sqrt(f / FREF)) / 20) * np.exp(-1j * 2 * np.pi * f * td_c)
    M = np.zeros((nf, 4, 4), complex)
    M[:, 1, 0] = M[:, 0, 1] = Td
    M[:, 3, 2] = M[:, 2, 3] = Tc
    M[:, 0, 0] = M[:, 1, 1] = r_d
    M[:, 2, 2] = M[:, 3, 3] = r_c
    M[:, 3, 0] = M[:, 0, 3] = xconv          # differential in -> common out, and reciprocal
    M[:, 2, 1] = M[:, 1, 2] = xconv
    return M, Td, Tc


def se_from_modal(M, ports):
    """Push a mixed-mode matrix DOWN to single-ended: ``S = T^T M T`` (T orthogonal). This is
    the file a VNA would have written for that structure with that port assignment."""
    T = _T(None, SP.diff_pairs(ports))
    return T.T @ M @ T


def _grid_freqs(n=4001, fmax=40e9, fmin=0.0):
    f = np.linspace(fmin, fmax, n)
    return f


def _at(f, q=FREF):
    return int(np.argmin(np.abs(np.asarray(f) - q)))


def _db(z):
    return 20.0 * np.log10(np.abs(z) + 1e-300)


# ------------------------------------------------------------------ 1. the transform itself
@pytest.mark.parametrize("conv", ["12_34", "13_24", "14_23", ((2, 4), (1, 3)), ((3, 1), (4, 2))])
def test_se2mm_recovers_a_constructed_modal_matrix(conv):
    """THE constructed answer. Build the mixed-mode matrix, write it down single-ended under a
    known port assignment, and demand the whole 4x4 back — not just SDD21, all sixteen terms,
    including the mode-conversion terms an asymmetric pair lives on.

    This is the check the old code could not have passed under any pairing, because it never
    formed the combination at all."""
    f = _grid_freqs()
    M, Td, Tc = modal_4port(f, 1.500e-9, 1.542e-9, loss_d_db=9.0, loss_c_db=7.5,
                            r_d=0.04, r_c=0.09, xconv=0.012)
    S = se_from_modal(M, conv)
    Mr = SP.se2mm(S, conv)
    err = float(np.max(np.abs(Mr - M)))
    assert err < 1e-12, f"se2mm(ports={conv!r}) did not recover the modal matrix: max err {err:.3e}"
    assert float(np.max(np.abs(SP.mixed_mode_term(S, conv, "SDD21") - Td))) < 1e-12
    assert float(np.max(np.abs(SP.mixed_mode_term(S, conv, "SCC21") - Tc))) < 1e-12
    assert float(np.max(np.abs(SP.mixed_mode_term(S, conv, "SCD21") - 0.012))) < 1e-12


def test_se2mm_conserves_power_because_the_transform_is_orthogonal():
    """A similarity transform by an orthogonal matrix cannot create or destroy power. If this
    ever fails, the 1/sqrt(2) normalisation has drifted and every dB in this module is wrong by
    a constant nobody would notice."""
    f = _grid_freqs(801)
    M, _, _ = modal_4port(f, 1.5e-9, 1.55e-9, 9.0, 7.5, r_d=0.05, r_c=0.08, xconv=0.02)
    S = se_from_modal(M, "12_34")
    got = SP.se2mm(S, "12_34")
    fro_se = np.linalg.norm(S, axis=(1, 2))
    fro_mm = np.linalg.norm(got, axis=(1, 2))
    assert float(np.max(np.abs(fro_mm - fro_se))) < 1e-12


def test_sdd21_of_an_UNCOUPLED_pair_is_exactly_the_single_ended_transfer():
    """The degenerate case that makes the defect survivable, and so worth pinning.

    With no coupling the two modes are identical (T_d == T_c), so SDD21 == the single-ended
    through term EXACTLY. Every uncoupled fixture validates `S21` perfectly — which is why the
    error ships. Both .s4p files on disk are of this kind."""
    f = _grid_freqs(1601)
    M, Td, Tc = modal_4port(f, 1.5e-9, 1.5e-9, 9.0, 9.0)
    assert np.allclose(Td, Tc)
    S = se_from_modal(M, "12_34")
    sdd = SP.mixed_mode_term(S, "12_34")
    se = S[:, 2, 0]                       # pairs (1,2)/(3,4): the through term is S31
    assert float(np.max(np.abs(sdd - se))) < 1e-12
    assert float(np.max(np.abs(sdd - Td))) < 1e-12


# ------------------------------------------------------------------ 2. the 6.2 dB
SKEW_CASES = [
    (8.53, 0.20, "the forgiving pair a caller validates on"),
    (42.16, 6.21, "the pair that ships the error"),
    (20.0, 1.14, None),
    (55.0, 14.55, None),
]


@pytest.mark.parametrize("dtau_ps,want_db,_why", SKEW_CASES)
def test_the_single_ended_error_is_the_closed_form_cosine(dtau_ps, want_db, _why):
    """``20log10|SDD21| - 20log10|S_through| = -20 log10 |cos(pi f dtau)|`` for equal-loss modes.

    Derivation, so the number is not a fit: the single-ended through term of a symmetric coupled
    pair is ``(T_d + T_c)/2``. With ``|T_d| = |T_c| = A`` and ``T_c = T_d * exp(-j 2 pi f dtau)``,

        |(T_d + T_c)/2| = A * |1 + exp(-j 2 pi f dtau)| / 2 = A * |cos(pi f dtau)|

    so the ratio to ``|SDD21| = A`` is ``|cos(pi f dtau)|`` and nothing else — no loss term
    survives, which is why the error is invisible to any check that looks at insertion loss.

    8.53 ps of even/odd skew reads 0.20 dB at 8 GHz; 42.16 ps reads 6.21 dB. Both are ordinary
    numbers for an edge-coupled pair: 42 ps over 10 inches is 4.2 ps/inch of odd-vs-even delay,
    a few percent of effective permittivity between the modes."""
    f = np.linspace(0.0, 40e9, 4001)
    f[_at(f)] = FREF                                     # land a bin exactly on 8 GHz
    M, Td, Tc = modal_4port(f, 1.5e-9, 1.5e-9 + dtau_ps * 1e-12, 9.0, 9.0)
    S = se_from_modal(M, "12_34")
    k = _at(f)
    sdd = SP.mixed_mode_term(S, "12_34")
    se = S[:, 2, 0]
    gap = _db(sdd[k]) - _db(se[k])
    closed = -20.0 * np.log10(abs(np.cos(np.pi * FREF * dtau_ps * 1e-12)))
    assert abs(gap - closed) < 1e-6, (
        f"dtau={dtau_ps} ps: measured gap {gap:.4f} dB, closed form {closed:.4f} dB")
    assert abs(gap - want_db) < 0.01, (
        f"dtau={dtau_ps} ps: gap {gap:.3f} dB, expected {want_db:.2f} dB")
    # and SDD21 is the loss law it was built from, untouched by the skew
    assert abs(_db(sdd[k]) - (-9.0)) < 1e-9


def test_the_6_2_db_reaches_the_WAVEFORM_not_just_the_spectrum():
    """A dB at Nyquist is only worth arguing about if it moves a record. Drive the same 16 GBd
    NRZ record through the correct transfer (SDD21) and through the single-ended term of the
    SAME file and compare eye height. The wrong term is 6.2 dB down at Nyquist, so it closes an
    eye that is open."""
    fs, baud, spb = 128e9, 16e9, 8
    f = np.linspace(0.0, fs / 2, 4097)
    M, Td, Tc = modal_4port(f, 1.5e-9, 1.5e-9 + 42.16e-12, 9.0, 9.0)
    S = se_from_modal(M, "12_34")
    sdd, se = SP.mixed_mode_term(S, "12_34"), S[:, 2, 0]
    g = Grid(fs=fs, baud=baud, n=4096 * spb)
    x = P.nrz(n_ui=4096, n=g.n, seed=11, tr_frac=0.25)
    y_ok = SP.sparam_channel(x, f, sdd, dt=1 / fs)
    y_bad = SP.sparam_channel(x, f, se, dt=1 / fs)

    def eye(y):
        y = y[len(y) // 4:]
        m = (len(y) // spb) * spb
        fr = y[:m].reshape(-1, spb)
        best = -9e9
        for ph in range(spb):
            c = fr[:, ph]
            md = np.median(c)
            hi, lo = c[c > md], c[c <= md]
            if len(hi) < 50 or len(lo) < 50:
                continue
            best = max(best, np.percentile(hi, 1.0) - np.percentile(lo, 99.0))
        return best

    e_ok, e_bad = eye(y_ok), eye(y_bad)
    rel = float(np.max(np.abs(y_bad - y_ok))) / float(np.std(y_ok))
    assert e_bad < 0.75 * e_ok, (f"the single-ended term should visibly close the eye: "
                                 f"SDD21 {e_ok:.4f} vs S31 {e_bad:.4f}")
    assert rel > 0.5, f"peak difference only {rel:.2f} of the output rms"
    print(f"\n  eye height: SDD21 {e_ok:.4f}   single-ended S31 {e_bad:.4f}   "
          f"({100*(1-e_bad/e_ok):.1f} % closed)   peak diff {rel:.2f} x output rms")


# ------------------------------------------------------------------ 3. the port convention
def _legacy_touchstone_term(path, ports=(2, 1)):
    """WHAT THE OLD CODE DID: take one single-ended term, positionally, unvalidated. Kept so the
    gates below can be pointed at it and seen to fire."""
    _, S = SP.read_touchstone(path)
    return S[:, ports[0] - 1, ports[1] - 1]


@needs_real
def test_a_transposed_pairing_returned_zeros_and_raised_nothing():
    """GATE OBSERVED FAILING, on a file on disk.

    `stub_diff_B.s4p` pairs ports (1,2) and (3,4), so its through terms are S31 and S42. The old
    default `ports=(2, 1)` asks for near-P -> near-N, which on a matched pair is identically
    zero. The old path returned that silently; the assertion below records the exact number, and
    then the new path is required to raise on the same file."""
    f, S = SP.read_touchstone(REAL_B)
    legacy = _legacy_touchstone_term(REAL_B, (2, 1))
    peak = float(np.max(np.abs(legacy)))
    assert peak == 0.0, (f"expected the legacy default to be identically zero on this file, "
                         f"got peak |S21| = {peak:.3e}")
    # the new path refuses, twice over: single-ended on a 4-port, and the wrong pairing
    x = np.zeros(4096)
    x[100:140] = 1.0
    with pytest.raises(ValueError, match="differential channel measured single-ended"):
        SP.touchstone_channel(x, REAL_B, dt=1 / 80e9, ports=(2, 1))
    with pytest.raises(ValueError, match="does not look like a differential through path"):
        SP.touchstone_channel(x, REAL_B, dt=1 / 80e9, ports="13_24")
    r = SP.check_pairing(f, S, "13_24", raise_=False)
    assert r["peak"] == 0.0 and r["best"] == "12_34"
    assert SP.check_pairing(f, S, "12_34", raise_=False)["ok"]
    print(f"\n  {os.path.basename(REAL_B)}: legacy ports=(2,1) peak |S21| = {peak:.3e} over "
          f"{f.min()/1e6:g} MHz..{f.max()/1e9:g} GHz; check_pairing picks {r['best']!r} "
          f"({r['candidates']})")


@needs_real
@pytest.mark.parametrize("path,conv,se_term", [("A", "13_24", (2, 1)), ("B", "12_34", (3, 1))])
def test_the_real_files_are_uncoupled_so_they_forgive_the_defect(path, conv, se_term):
    """Both .s4p files on disk are UNCOUPLED pairs, so SDD21 equals their single-ended through
    term to 0.0000 dB. That is the measurement that matters here: these files cannot detect
    defect 1, and a caller who validated on them would have seen nothing wrong."""
    p = REAL_A if path == "A" else REAL_B
    f, S = SP.read_touchstone(p)
    sdd = SP.mixed_mode_term(S, conv)
    se = S[:, se_term[0] - 1, se_term[1] - 1]
    k = _at(f)
    gap = _db(sdd[k]) - _db(se[k])
    assert abs(gap) < 1e-9, f"{path}: expected an uncoupled pair, gap {gap:.6f} dB"
    print(f"\n  {os.path.basename(p)} ({conv}): SDD21 {_db(sdd[k]):.2f} dB, "
          f"S{se_term[0]}{se_term[1]} {_db(se[k]):.2f} dB at 8 GHz -> gap {gap:+.4f} dB")


def test_a_bare_single_ended_pair_on_a_4_port_file_raises(tmp_path):
    """Defect 1's fix at the front door: on a 4-port file, `ports=(2, 1)` is now an error that
    names the alternative, and `mode="se"` is how a caller says they meant it."""
    f = _grid_freqs(401)
    M, Td, _ = modal_4port(f, 1.5e-9, 1.542e-9, 9.0, 7.5)
    S = se_from_modal(M, "12_34")
    p = str(tmp_path / "pair.s4p")
    SP.write_touchstone(p, f, S)
    x = np.zeros(2048)
    x[64:96] = 1.0
    with pytest.raises(ValueError, match="SDD21"):
        SP.touchstone_channel(x, p, dt=1 / 80e9, ports=(3, 1))
    y = SP.touchstone_channel(x, p, dt=1 / 80e9, ports=(3, 1), mode="se")   # on purpose: allowed
    assert np.any(np.abs(y) > 1e-6)
    y2 = SP.touchstone_channel(x, p, dt=1 / 80e9, ports="12_34")
    assert float(np.max(np.abs(y2 - y))) > 0.05 * float(np.std(y2))


@pytest.mark.parametrize("bad,match", [
    ("11_22", "unknown differential port convention"),
    ("13-24", "unknown differential port convention"),
    (((1, 2), (2, 3)), "names a port twice"),
    (((1, 2), (3, 5)), r"outside 1\.\.4"),
    (((1, 2), (3,)), "two pairs of two ports"),
    (((1, 2),), "two pairs of two ports"),
])
def test_a_nonsense_port_convention_raises_instead_of_returning_zeros(bad, match):
    """Defect 2: the convention was positional with no validation. Each of these used to reach
    an indexing expression and either throw an opaque IndexError or, worse, index a valid but
    meaningless term."""
    with pytest.raises(ValueError, match=match):
        SP.diff_pairs(bad, n_ports=4)


def test_check_pairing_does_not_cry_wolf_on_a_genuinely_lossy_channel():
    """The gate must not fire just because a channel is lossy. 30 dB at 40 GHz, with the correct
    pairing, has to pass — the check reads the LOW band, where every passive through path is
    close to 0 dB whatever it does at Nyquist."""
    f = _grid_freqs(1201, fmin=1e7)
    M, _, _ = modal_4port(f, 1.5e-9, 1.55e-9, loss_d_db=21.2, loss_c_db=18.0, r_d=0.06)
    S = se_from_modal(M, "13_24")
    k = _at(f, 40e9)
    assert _db(SP.mixed_mode_term(S, "13_24")[k]) < -29.0
    r = SP.check_pairing(f, S, "13_24")
    # 14_23 happens to read 0.10 dB higher on a coupled pair (it mixes the modes
    # differently); the point is that the margin is nowhere near the 6 dB tolerance.
    assert r["ok"] and -1.0 < r["margin_db"] <= 0.0, r


# ------------------------------------------------------------------ 4. the band edge
def _fifteen_ghz_file(tmp_path, fmax=15e9, n=601):
    """A file that stops at 15 GHz — the ordinary case, since a 15 or 20 GHz VNA is what most
    measured channels come from."""
    f = np.linspace(1e7, fmax, n)
    il = 0.35 * np.sqrt(f / 1e9) + 0.2 * (f / 1e9)
    s21 = 10 ** (-il / 20) * np.exp(-1j * 2 * np.pi * f * 500e-12)
    return f, s21


def test_the_band_edge_refusal_names_the_power_it_would_have_deleted(tmp_path):
    """Defect 3, with the number measured from the record rather than asserted about modulation.

    A 15 GHz response, a 16 GBd record at 256 GS/s (Nyquist 128 GHz): the old path zeroed
    15..128 GHz and said nothing."""
    f, s21 = _fifteen_ghz_file(tmp_path)
    fs, spb = 256e9, 16
    x = P.nrz(n_ui=2048, n=2048 * spb, seed=3, tr_frac=0.15)
    lost_lo, lost_hi = SP._band_energy(x, 1 / fs, f.min(), f.max())
    assert 0.015 < lost_hi < 0.03, f"expected ~1.9 % above 15 GHz, measured {100*lost_hi:.3g} %"

    with pytest.raises(ValueError) as ei:
        SP.sparam_channel(x, f, s21, dt=1 / fs)                     # band='refuse' by default
    msg = str(ei.value)
    assert f"{100*lost_hi:.4g} %" in msg and "15.000 GHz" in msg and "128.000 GHz" in msg

    # the legacy behaviour is still reachable, and now says what it is doing
    with pytest.warns(RuntimeWarning, match="of the record's power lies above"):
        y_zero = SP.sparam_channel(x, f, s21, dt=1 / fs, band="zero")
    # and so is the explicit extrapolation
    y_hold = SP.sparam_channel(x, f, s21, dt=1 / fs, band="hold")
    # opting in to the loss by raising the tolerance is allowed, and warns
    with pytest.warns(RuntimeWarning):
        y_tol = SP.sparam_channel(x, f, s21, dt=1 / fs, band_tol=0.05)
    assert np.allclose(y_zero, y_tol)
    d = float(np.max(np.abs(y_hold - y_zero))) / float(np.std(y_zero))
    assert d > 0.02, f"'hold' and 'zero' differ by only {d:.3g} of the output rms"
    print(f"\n  15 GHz file, 16 GBd @256 GS/s tr=0.15 UI: {100*lost_hi:.3g} % of the record's "
          f"power above f_max (refused at band_tol=0.1 %); zero-vs-hold differ by "
          f"{100*d:.1f} % of output rms")


def test_a_record_that_fits_inside_the_file_is_not_refused(tmp_path):
    """The refusal has to be proportionate or it will be turned off. A 0.35 UI rise time puts
    0.027 % above 15 GHz: warned, not refused. And a record whose Nyquist is inside the file's
    band is neither."""
    f, s21 = _fifteen_ghz_file(tmp_path)
    x = P.nrz(n_ui=2048, n=2048 * 16, seed=3, tr_frac=0.35)
    _, lost_hi = SP._band_energy(x, 1 / 256e9, f.min(), f.max())
    assert lost_hi < 1e-3, f"expected < 0.1 % above 15 GHz, measured {100*lost_hi:.3g} %"
    with pytest.warns(RuntimeWarning, match="within band_tol"):
        SP.sparam_channel(x, f, s21, dt=1 / 256e9)
    # Nyquist inside the file: silence
    x2 = P.nrz(n_ui=1024, n=1024 * 2, seed=3, tr_frac=0.5)          # fs = 2*baud = 32 GS/s
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SP.sparam_channel(x2, f, s21, dt=1 / 30e9)                  # Nyquist 15 GHz == f_max


def test_hold_extrapolation_is_continuous_at_the_band_edge():
    """`band="hold"` is an extrapolation and is documented as one; what it must not be is a
    discontinuity, which would ring for the length of the record."""
    f = np.linspace(1e7, 15e9, 601)
    s21 = 10 ** (-(0.35 * np.sqrt(f / 1e9)) / 20) * np.exp(-1j * 2 * np.pi * f * 500e-12)
    fg = np.fft.rfftfreq(1 << 14, d=1 / 64e9)
    H = SP._interp_response(fg, f, s21, band="hold", dc="extend")
    k = int(np.searchsorted(fg, f.max()))
    step = abs(H[k] - H[k - 1])
    typ = float(np.median(np.abs(np.diff(H[:k]))))
    assert step < 20 * typ, f"discontinuity {step:.3e} vs typical bin step {typ:.3e}"
    assert np.all(np.abs(np.abs(H[k:]) - abs(s21[-1])) < 1e-12)


# ------------------------------------------------------------------ 5. the DC block
def test_a_constant_input_comes_out_constant(tmp_path):
    """Defect 4, as a constructed answer nobody can argue with.

    A passive through path does not block DC. Its transmission there is real and, for a file
    whose lowest point is at 10 MHz, within a hair of |S(f_min)|. So a CONSTANT input must come
    out as that constant times that number. The old default zeroed every bin below f_min,
    including bin 0, and returned something else entirely."""
    f = np.linspace(1e7, 40e9, 2001)
    il = 0.35 * np.sqrt(f / 1e9) + 0.2 * (f / 1e9)
    s21 = 10 ** (-il / 20) * np.exp(-1j * 2 * np.pi * f * 200e-12)
    dc_truth = float(abs(s21[0]))
    x = np.ones(8192)
    y_new = SP.sparam_channel(x, f, s21, dt=1 / 64e9, band="zero")
    y_old = SP.sparam_channel(x, f, s21, dt=1 / 64e9, band="zero", dc="zero")
    mid_new, mid_old = float(y_new[len(y_new) // 2]), float(y_old[len(y_old) // 2])
    assert abs(mid_new - dc_truth) < 5e-3 * dc_truth, (
        f"dc='extend' gave {mid_new:.6f}, the file's own |S(f_min)| is {dc_truth:.6f}")
    assert abs(mid_old) < 0.5 * dc_truth, (
        f"GATE NEVER FIRED: dc='zero' was supposed to be the DC-blocking defect but gave "
        f"{mid_old:.6f} against a truth of {dc_truth:.6f}")
    print(f"\n  constant-1 in: dc='extend' -> {mid_new:.6f}  (truth |S(f_min)| = {dc_truth:.6f}, "
          f"err {abs(mid_new-dc_truth):.2e});  dc='zero' -> {mid_old:.6f}, which is off by "
          f"{abs(mid_old-dc_truth):.4f} — not merely attenuated but the WRONG SIGN, because "
          f"zeroing bin 0 leaves only the ringing of the record's own edges")


def test_dc_zero_warns_that_it_is_blocking_dc():
    f = np.linspace(1e7, 40e9, 801)
    s21 = 10 ** (-(0.35 * np.sqrt(f / 1e9)) / 20)
    x = np.ones(4096) + 0.01 * np.random.default_rng(0).standard_normal(4096)
    with pytest.warns(RuntimeWarning, match="DC-BLOCKING"):
        SP.sparam_channel(x, f, s21.astype(complex), dt=1 / 64e9, band="zero", dc="zero")


def test_dc_extension_is_the_file_and_not_an_invention():
    """The extension adds no structure: H(0) is the file's own |S(f_min)| and every bin at or
    above f_min is untouched, bit for bit, by the choice of `dc`."""
    f = np.linspace(5e7, 20e9, 401)
    s21 = (10 ** (-(0.4 * np.sqrt(f / 1e9)) / 20)) * np.exp(-1j * 2 * np.pi * f * 300e-12)
    fg = np.fft.rfftfreq(1 << 13, d=1 / 40e9)
    He = SP._interp_response(fg, f, s21, band="zero", dc="extend")
    Hz = SP._interp_response(fg, f, s21, band="zero", dc="zero")
    inb = (fg >= f.min()) & (fg <= f.max())
    assert np.array_equal(He[inb], Hz[inb])
    assert abs(He[0].real - abs(s21[0])) < 1e-15 and He[0].imag == 0.0
    assert np.all(Hz[fg < f.min()] == 0)


@needs_real
def test_the_dc_block_on_a_file_on_disk():
    """The same thing on real bytes: a 10 MHz-to-40 GHz .s4p, its own SDD21, a constant in."""
    f, S = SP.read_touchstone(REAL_A)
    h = SP.mixed_mode_term(S, "13_24")
    truth = float(abs(h[0]))
    x = np.ones(4096)
    new = float(SP.sparam_channel(x, f, h, dt=1 / 64e9, band="zero")[2048])
    old = float(SP.sparam_channel(x, f, h, dt=1 / 64e9, band="zero", dc="zero")[2048])
    assert abs(new - truth) < 1e-3 and old < 0.3 * truth
    print(f"\n  {os.path.basename(REAL_A)} f_min={f.min()/1e6:g} MHz, |SDD21(f_min)|={truth:.6f}: "
          f"dc='extend' -> {new:.6f}, dc='zero' -> {old:.6f} "
          f"({_db(truth)-_db(old):.1f} dB low)")


# ------------------------------------------------------------------ 6. the standard this
#     project holds itself to, applied to this path (tests/test_no_fabricated_data.py's four
#     properties; `sparam` is not in that file's STAGES because it takes a file, not a scalar).
def _mm_channel(x, dt=1 / 128e9):
    f = np.linspace(0.0, 0.5 / dt, 2049)
    M, _, _ = modal_4port(f, 1.5e-9, 1.5e-9 + 30e-12, 9.0, 7.0, r_d=0.05, xconv=0.01)
    S = se_from_modal(M, "13_24")
    return SP.sparam_channel(x, f, SP.mixed_mode_term(S, "13_24"), dt=dt)


def test_every_input_sample_influences_the_mixed_mode_output():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(4096) * 0.3
    base = _mm_channel(x)
    unmoved = []
    for k in (0, 1, 7, 1000, 2048, 4095):
        y = x.copy()
        y[k] += 0.05
        if float(np.max(np.abs(_mm_channel(y) - base))) < 1e-12:
            unmoved.append(k)
    assert not unmoved, f"perturbing samples {unmoved} changed nothing"


def test_the_mixed_mode_channel_obeys_superposition():
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal(4096) * 0.3, rng.standard_normal(4096) * 0.3
    ya, yb, yab, y3 = (_mm_channel(v) for v in (a, b, a + b, 3.0 * a))
    add = float(np.max(np.abs(yab - (ya + yb)))) / (float(np.std(yab)) + 1e-30)
    homo = float(np.max(np.abs(y3 - 3.0 * ya))) / (float(np.std(y3)) + 1e-30)
    assert add < 1e-9 and homo < 1e-9, f"additivity {add:.2e}, homogeneity {homo:.2e}"


def test_the_channel_realises_the_response_it_was_asked_for():
    """A filter must realise the corner it was asked for. Here: a notch put into SDD21 has to
    come out of the record at the frequency it was put in, and the loss law has to be the loss
    law — measured through the record, not read back off the interpolator."""
    dt, n = 1 / 128e9, 1 << 15
    f = np.linspace(0.0, 0.5 / dt, 4097)
    M, Td, _ = modal_4port(f, 1.5e-9, 1.53e-9, 9.0, 7.0)
    notch = 1 - 0.93 * np.exp(-((f - 18e9) / 0.8e9) ** 2)
    M[:, 1, 0] = M[:, 0, 1] = Td * notch
    S = se_from_modal(M, "12_34")
    h = SP.mixed_mode_term(S, "12_34")
    imp = np.zeros(n)
    imp[0] = 1.0
    H = np.fft.rfft(SP.sparam_channel(imp, f, h, dt=dt, linear=False))
    fg = np.fft.rfftfreq(n, d=dt)
    got = _db(H[_at(fg, 18e9)]) - _db(H[_at(fg, 10e9)])
    want = _db(h[_at(f, 18e9)]) - _db(h[_at(f, 10e9)])
    assert abs(got - want) < 0.5, f"notch depth realised {got:.2f} dB, asked for {want:.2f} dB"
    for q in (2e9, 8e9, 30e9, 50e9):
        assert abs(_db(H[_at(fg, q)]) - _db(h[_at(f, q)])) < 0.2, f"|H| wrong at {q/1e9:g} GHz"

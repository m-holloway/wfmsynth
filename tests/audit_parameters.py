"""Assertions against a CLASS of defect: a parameter that is not honoured.

The two defects that motivated this file were both found by asking one question of a knob --
"is the number that comes OUT the number that went IN?" -- and both had gone unnoticed for a
long time:

  * `ac_couple` clamped any corner below 1e-4 of Nyquist. A 50 kHz ask became 12.8 MHz, 256x,
    with nothing said. (Fixed: the floor is now 1e-5 and it warns.)
  * `scope_bandwidth` ran a 4th-order Bessel through `sosfiltfilt`, i.e. twice, so a 32 GHz
    request realised -3 dB at 16 GHz. (Being fixed as this file is written.)

Three general properties, applied to every op and every NUMERIC parameter that has a closed
form or an invariant to check it against. None of them pins today's output; each one is a
statement about physics that survives a refactor.

  1. THE REALISED EFFECT TRACKS THE REQUEST, monotonically and at the right ABSOLUTE value,
     measured from the output. Not "the parameter was stored" -- measured.
  2. A SILENT SUBSTITUTION IS THE DEFECT, independent of the value. Every internal `np.clip`,
     `max(`, `min(` or round that can stand in a different number for the one asked for must
     either be lifted or warn. `ac_couple` now warns; nothing else in the kernel does.
  3. EVERY PARAMETER DEMONSTRABLY DOES SOMETHING. A knob that can be moved without changing
     the output is broken or undocumented. This project shipped a `noise_mv` that was stored
     and never read for weeks.

GROUND TRUTH FIRST -- see `test_corner_instrument_*`. The corner-bisector below is calibrated
in BOTH directions before any claim rests on it: it recovers a constructed single-pass
first-order corner to 0.00x and separately reports 1.553x for the SAME design run through
`sosfiltfilt`, which is the whole distinction the `ac_couple` finding turns on. And then the
clause that bites: the same measurement is repeated with a REAL capture as the probe signal
(40 GSa/s, 1,843 distinct codes, non-periodic, not white) and agrees to 0.9 %. A stop-band
floor claim in this project stood wrong by 3 dB for months because both of its gates used
constructed noise with no signal in it to leak; a sine into an LTI stage is that same kind of
too-easy fixture, so the real record is the gate that is allowed to fail.

Runnable two ways:
    pytest wfmsynth/tests/audit_parameters.py
    python  wfmsynth/tests/audit_parameters.py      # prints every measured number
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
from scipy import signal as _sig

from wfmsynth import cdr as CDR
from wfmsynth import impairments as IMP
from wfmsynth import instrument as INST
from wfmsynth import optical as OPT
from wfmsynth import physics as P
from wfmsynth.compose import _EXEC
from wfmsynth.grid import Grid
from wfmsynth.streams import Streams

FS = 256e9
BAUD = 16e9
N = 1 << 16
SPB = int(FS / BAUD)


def _grid(fs=FS, baud=BAUD, n=N):
    return Grid(fs=fs, baud=baud, n=n, v_full=0.8)


def _run(op, params, x, grid=None, seed=0):
    """One op through the composer's own dispatch table, so the test exercises the path a
    recipe takes rather than a primitive a recipe never reaches."""
    grid = grid or _grid()
    return np.asarray(_EXEC[op](x, dict(params, op=op), Streams(seed), grid, 0), float)


def _nrz(n_ui=1024, n=N, tr_frac=0.3, seed=5, scale=1.0):
    return scale * P.nrz(n_ui=n_ui, n=n, tr_frac=tr_frac, causal=True, seed=seed)


# ------------------------------------------------------------------ the instrument
def _H_at(fn, f_hz, n=N, fs=FS):
    """Complex |H| and phase at one frequency, by driving a single exact DFT bin.

    Exact for any LTI stage and immune to how the stage pads or truncates, which is why it is
    used instead of dividing spectra: the frequency-domain ops now apply a LINEAR convolution,
    so ``Y = H*X`` does not hold at the record length at all.
    """
    k = max(1, int(round(f_hz * n / fs)))
    t = np.sin(2 * np.pi * k * np.arange(n) / n)
    y = np.asarray(fn(t), float)[:n]
    H = np.fft.rfft(y)[k] / np.fft.rfft(t)[k]
    return H, k * fs / n


def _bisect_3db(fn, f0, rising, n=N, fs=FS, span=60.0, iters=40):
    """The realised -3 dB point, bisected in log frequency. `rising` for a high-pass."""
    lo, hi = f0 / span, min(0.45 * fs, f0 * span)
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        g = abs(_H_at(fn, mid, n=n, fs=fs)[0])
        if (g < 0.70710678) == rising:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def _slope_db_per_octave(fn, f_hz, n=N, fs=FS):
    """Asymptotic roll-off, measured one octave apart well inside the stop band."""
    g1 = abs(_H_at(fn, f_hz / 10.0, n=n, fs=fs)[0])
    g2 = abs(_H_at(fn, f_hz / 20.0, n=n, fs=fs)[0])
    return 20.0 * np.log10(g1 / g2)


def _n_zero_crossings(x):
    s = np.signbit(np.asarray(x, float))
    return int(np.count_nonzero(s[:-1] != s[1:]))


def _rise_fall_20_80(y, spb):
    """Median 20-80 % rise and fall time in samples, from the record's own level histogram."""
    y = np.asarray(y, float)
    lo, hi = np.percentile(y, 2), np.percentile(y, 98)
    a, b, mid = lo + 0.2 * (hi - lo), lo + 0.8 * (hi - lo), 0.5 * (lo + hi)
    s = np.signbit(y - mid)
    out = {True: [], False: []}
    for i in np.flatnonzero(s[:-1] != s[1:]):
        up = y[i + 1] > y[i]
        seg = y[max(0, i - spb // 2):i + spb // 2]
        if len(seg) < 4:
            continue
        v = seg if up else seg[::-1]
        if not (v[0] < a < v[-1] and v[0] < b < v[-1]):
            continue
        t = np.arange(len(v))
        out[up].append(abs(np.interp(b, v, t) - np.interp(a, v, t)))
    if not out[True] or not out[False]:
        return float("nan"), float("nan")
    return float(np.median(out[True])), float(np.median(out[False]))


def _corner_through_record(x, fs, fn, fc, nperseg=1 << 15):
    """The realised -3 dB corner of `fn`, measured with a REAL record as the probe signal.

    Welch magnitude ratio rather than a single DFT bin, because a real record is not periodic
    and its spectrum is neither flat nor free of nulls: `sqrt(Pyy/Pxx)` of the same record
    filtered and unfiltered IS |H|, and both estimates carry the same window kernel, which
    cancels.

    A DEFENSIVE NORMALISATION IS WHAT FIRST BROKE THIS. An earlier version divided H by its
    median over a nominal passband above the corner, to guard against a record with no energy
    there. But a first-order response is 0.9701 at 4x its corner, not 1, so the "passband"
    median sat below unity and pulled the recovered corner DOWN by a consistent amount that
    grew with fc -- measured across all three real captures: one pole 0.9860 / 0.9763 / 0.9616
    at fc = 0.5 / 1 / 2 GHz, and the squared design 1.5403 / 1.4995 / 1.4442 against its 1.5538.
    The bias was systematic, not noise, and it was entirely the estimator's. Removing the
    normalisation gives 0.9990-1.0075 and 1.5356-1.5529 over the same nine cases. The guard was
    the defect; the empty-band case is handled by returning NaN and letting the caller say so.
    """
    y = np.asarray(fn(x), float)[:len(x)]
    f, Pxx = _sig.welch(x, fs=fs, nperseg=nperseg)
    _f, Pyy = _sig.welch(y, fs=fs, nperseg=nperseg)
    H = np.sqrt(Pyy / (Pxx + 1e-300))
    win = (f > fc / 30.0) & (f < fc * 30.0)
    idx = np.flatnonzero((H[:-1] < 0.70710678) & (H[1:] >= 0.70710678) & win[:-1])
    if not len(idx):
        return float("nan")
    i = idx[0]
    return float(np.interp(0.70710678, [H[i], H[i + 1]], [f[i], f[i + 1]]))


# ------------------------------------------------------------------ real captures (optional)
# These live outside this repo, which ships no binary fixtures, so absent is a SKIP rather than a
# failure. Point WFMSYNTH_REALCAPS at a directory of .wfm captures and WFMSYNTH_WFM_TOOLS at a
# checkout that exposes `load_waveform` to run the on-disk half of this file.
_REALCAPS = os.environ.get("WFMSYNTH_REALCAPS", "")
_WFMREVERSE = os.environ.get("WFMSYNTH_WFM_TOOLS", "")


def _load_real(window=1 << 19):
    """One real capture, or None. This repo is public and standard-neutral; the captures live
    in a private repo, so every gate that uses them skips cleanly when they are absent."""
    if not os.path.isdir(_REALCAPS):
        return None
    caps = sorted(f for f in os.listdir(_REALCAPS) if f.endswith(".wfm"))
    if not caps:
        return None
    import sys
    if _WFMREVERSE and _WFMREVERSE not in sys.path:
        sys.path.insert(0, _WFMREVERSE)
    try:
        import wfm as _wfm
    except Exception:
        return None
    try:
        x, fs = _wfm.load_waveform(os.path.join(_REALCAPS, caps[0]), window=window)
    except Exception:
        return None
    return np.asarray(x, float), float(fs), caps[0]


# ==================================================================================
# 0. GROUND TRUTH -- calibrate the instrument before believing a number it produces
# ==================================================================================
def test_corner_instrument_recovers_a_constructed_corner_and_separates_a_squared_pole():
    """The bisector must recover a KNOWN corner, and must tell a single pole from that same
    pole applied twice. Everything below rests on this second half: a doubly-applied
    first-order pole has its -3 dB at ``1/sqrt(sqrt(2)-1) = 1.5538x`` the design corner, and
    an instrument that cannot resolve 1.00x from 1.55x cannot make the `ac_couple` claim.
    """
    fs, n = FS, N
    for fc in (0.5e9, 2e9, 8e9):
        w = fc / (fs / 2.0)
        sos = _sig.butter(1, w, btype="high", output="sos")
        one = _bisect_3db(lambda z: _sig.sosfilt(sos, z), fc, True)
        two = _bisect_3db(lambda z: _sig.sosfiltfilt(sos, z), fc, True)
        assert abs(one / fc - 1.0) < 0.01, f"single pass: {one/fc:.4f}x at fc={fc:g}"
        assert abs(two / fc - 1.5538) < 0.02, f"double pass: {two/fc:.4f}x at fc={fc:g}"
        assert two / one > 1.5, "the instrument cannot separate one pole from two"


def test_corner_instrument_agrees_when_the_probe_is_a_real_capture():
    """THE CLAUSE THAT BITES. A sine into an LTI stage is an easy fixture: exactly periodic,
    one bin, no quantisation, nothing to leak. A real record is none of those things. Measure
    the same two constructed corners through a real 40 GSa/s capture -- non-periodic,
    1,843 distinct stored codes, a spectrum that is anything but white -- with a Welch
    magnitude ratio, and the answers must agree with the sine probe.

    MEASURED over all three real captures at fc = 0.5 / 1 / 2 GHz: the single pole comes back
    at 0.9990x to 1.0075x of the request, and the same design through `sosfiltfilt` at 1.5356x
    to 1.5529x against its closed-form 1.5538x. The sine probe on a constructed grid gives
    1.0002-1.0039x and 1.5466-1.5537x, so the real record costs about a percent and settles
    nothing else: the two hypotheses are still 1.0 against 1.55.
    """
    real = _load_real()
    if real is None:
        pytest.skip("no real captures available (private repo)")
    x, fs, name = real
    for fc in (0.5e9, 1e9, 2e9):
        sos = _sig.butter(1, fc / (fs / 2.0), btype="high", output="sos")
        for label, fn, expect in (("one pole", lambda z: _sig.sosfilt(sos, z), 1.0),
                                  ("two poles", lambda z: _sig.sosfiltfilt(sos, z), 1.5538)):
            got = _corner_through_record(x, fs, fn, fc)
            assert np.isfinite(got), (
                f"{label}: no -3 dB crossing found in the real record's spectrum")
            assert abs(got / fc / expect - 1.0) < 0.02, (
                f"{name}: {label} at fc={fc/1e9:g} GHz measured {got/fc:.4f}x through the real "
                f"record, expected {expect:.4f}x -- the instrument does not survive a record "
                f"that is not a sine")


# ==================================================================================
# 1. FINDINGS -- each a measured silent substitution, encoded as a strict xfail
# ==================================================================================
@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `physics.ac_couple` documents 'a 1st-order high-pass' and realises its "
    "SQUARE. `signal.sosfiltfilt` runs the butter(1) design forwards AND backwards, so at "
    "fc_hz=2e9 on a 256 GSa/s grid the realised |H(fc)| is 0.5000 (-6.02 dB) where a first-order "
    "pole gives 0.7071 (-3.01), the realised -3 dB point is 3.1074 GHz (1.5537x the request, the "
    "closed form for a doubly-applied pole), the asymptotic roll-off is 11.64 dB/octave instead "
    "of 6.02 (11.98 at a 10 GHz corner, where the low-frequency bins are better resolved), and the phase at the corner is -0.008 deg where the pole's is +45.000 -- the stage "
    "is zero-phase, group delay +0.0000 samples. This is `scope_bandwidth`'s defect in a second "
    "op: same cause (`sosfiltfilt`), same magnitude class, and it is the ONLY remaining one, "
    "`probe_loading(causal=True)` having already been given the exact single pole (measured "
    "|H(pole)| = 0.70707, phase -45.00 deg). It is invisible to the existing gate in "
    "test_no_fabricated_data.py because that gate tolerates 2x deliberately, to let a squared "
    "pole through while still catching a 256x clamp. Note the AC_COUPLE_MIN_FRAC comment "
    "calibrates its floor by bisecting the realised |H| = 0.5 point -- which IS this squared "
    "response's -3 dB point -- so the squaring was measured around, not noticed. The fix is a "
    "single forward `sosfilt` initialised to steady state at x[0], exactly what the causal "
    "branch of `scope_bandwidth` now does; that also gives the stage the group delay a series "
    "capacitor has."))
def test_ac_couple_realises_the_first_order_high_pass_it_documents():
    g = _grid()
    fc = 2e9
    fn = lambda z: P.ac_couple(z, fc_hz=fc, grid=g)
    H, f_actual = _H_at(fn, fc)
    db_at_corner = 20.0 * np.log10(abs(H))
    phase_deg = float(np.degrees(np.angle(H)))
    f3 = _bisect_3db(fn, fc, True)
    slope = _slope_db_per_octave(fn, fc)
    msg = (f"ask fc={fc/1e9:g} GHz: |H(fc)| = {abs(H):.4f} ({db_at_corner:+.3f} dB, first order "
           f"says -3.010), phase {phase_deg:+.3f} deg (says +45.000), realised -3 dB at "
           f"{f3/1e9:.4f} GHz ({f3/fc:.4f}x), roll-off {slope:.2f} dB/octave (says 6.02)")
    assert abs(db_at_corner + 3.010) < 0.25, msg
    assert abs(f3 / fc - 1.0) < 0.10, msg
    assert abs(slope - 6.02) < 1.0, msg
    assert phase_deg > 30.0, msg


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `instrument.scope_bandwidth` clamps the requested bandwidth to 0.99 of "
    "Nyquist with `wn = min(bw_hz / (grid.fs / 2.0), 0.99)` and says NOTHING. On a 100 GSa/s "
    "grid (Nyquist 50 GHz) asking for 60 GHz realises 49.500 GHz, 110 GHz realises 49.500 and "
    "1000 GHz realises 49.500 -- measured from the brickwall's own zeroed bins -- with 0 warnings "
    "raised in all three cases. 110 GHz is not a hypothetical: it is the top row of the real "
    "instrument's published ENOB table that `converter_noise_rms`'s docstring is anchored on, "
    "and 100 GSa/s is a perfectly ordinary render rate, so the substitution is 2.22x on a "
    "combination a caller will actually write. This is precisely the `ac_couple` class -- a "
    "clamp that stands a different number in silently -- and `ac_couple` is now the only op in "
    "the kernel that warns when it substitutes. Either honour it (a front end wider than the "
    "grid's Nyquist is a no-op band limit, which is the physically right answer) or warn."))
def test_scope_bandwidth_does_not_silently_substitute_a_bandwidth_above_nyquist():
    g = _grid(fs=100e9, baud=16e9, n=1 << 14)
    x = np.random.default_rng(1).standard_normal(g.n)
    worst = []
    for bw in (60e9, 110e9, 1000e9):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                y = INST.scope_bandwidth(x, g, bw, kind="brickwall")
            except ValueError:
                continue                      # refusing outright is an honest answer
            X = np.fft.rfft(y)
            nz = np.flatnonzero(np.abs(X) > 1e-9)
            realised = nz[-1] * g.fs / g.n
        if not w and realised < 0.99 * min(bw, g.f_nyquist):
            worst.append(f"ask {bw/1e9:g} GHz -> realised {realised/1e9:.3f} GHz, "
                         f"{len(w)} warnings")
    assert not worst, ("scope_bandwidth substituted a bandwidth silently: " + "; ".join(worst)
                       + f" (grid Nyquist {g.f_nyquist/1e9:g} GHz)")


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `compose._op_rx_ffe` computes the tap spacing as "
    "`max(1, int(round(spb * spacing_ui)))`, so on any grid whose samples-per-UI is not an "
    "integer -- which `Grid`'s own docstring advertises as the realistic case ('2.2857..., as in "
    "reality') -- the realised spacing is silently a different fraction of a UI than the one "
    "asked for. Measured from the op's own impulse response: at fs=80 GSa/s / 25.78125 GBd "
    "(spb = 3.1030), spacing_ui=0.5 -- 'the usual receiver form' per `rx.ffe`'s docstring -- "
    "realises 2 samples = 0.6445 UI, +28.91 %; spacing_ui=0.25 realises 1 sample = 0.3223 UI, "
    "+28.91 % (the `max(1, ...)` floor); at fs=100 GSa/s / 16 GBd (spb = 6.25), spacing_ui=1.0 "
    "realises 6 samples = 0.9600 UI (-4.00 %) and 0.25 realises 2 = 0.3200 UI (+28.00 %). A "
    "T/2-spaced FFE that is actually T/0.645-spaced is not the equaliser the recipe names, and "
    "nothing is said. `rx.ffe` itself rounds again (`int(round((k - pre) * tap_spacing))`), so "
    "even an explicit fractional `tap_spacing` lands on integer samples. `physics.tx_ffe` "
    "already does this correctly, with `np.interp` at fractional delay."))
def test_rx_ffe_realises_the_tap_spacing_it_was_asked_for():
    bad = []
    for fs, baud in ((256e9, 16e9), (100e9, 16e9), (80e9, 25.78125e9)):
        g = _grid(fs=fs, baud=baud, n=1 << 12)
        spb = g.samples_per_ui
        for want in (1.0, 0.5, 0.25):
            imp = np.zeros(g.n)
            imp[g.n // 2] = 1.0
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                y = _run("rx_ffe", dict(taps=[0.3, 1.0, -0.2], spacing_ui=want, pre=1), imp, g)
            pos = np.flatnonzero(np.abs(y) > 1e-12)
            got = float(np.diff(pos)[0]) / spb
            if abs(got / want - 1.0) > 0.05 and not w:
                bad.append(f"fs={fs/1e9:g} spb={spb:.4f} ask {want:g} UI -> {got:.4f} UI "
                           f"({100*(got/want-1):+.2f} %)")
    assert not bad, "rx_ffe substituted the tap spacing silently: " + "; ".join(bad)


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `physics.multi_reflection` rounds `td_ps` to a WHOLE SAMPLE "
    "(`td_samples = round(td_ps * 1e-12 * grid.fs)`) and, when that rounds to zero, the op "
    "silently stops being a reflection at all. At 256 GSa/s one sample is 3.906 ps, so any "
    "requested one-way delay below 1.953 ps gives d = 0: measured at td_ps=1.5 with "
    "gamma_s=0.3, gamma_l=0.4, n_bounce=6, the output is x * 1.054545 -- a flat +0.461 dB GAIN, "
    "one nonzero sample in an impulse response, NO echo -- and no warning. Above the threshold "
    "the rounding is a real but disclosed-nowhere error: td_ps=2.0 realises 3.906 ps (+95.3 %), "
    "41.0 realises 39.062 (-4.73 %), 40.0 realises 39.062 (-2.34 %). A 1.5 ps via stub or "
    "package discontinuity is an ordinary thing to model and it is exactly what vanishes. "
    "`physics.resonant_reflection`, in the same module, already applies td as "
    "`exp(-1j*2*pi*f*td)` at full fractional resolution -- measured against its closed form to "
    "8.4e-04 -- so the fix already exists next door."))
def test_reflect_realises_the_delay_it_was_asked_for():
    g = _grid(n=1 << 13)
    imp = np.zeros(g.n)
    imp[100] = 1.0
    bad = []
    for td_ps in (1.0, 1.5, 2.0, 3.9, 40.0, 41.0, 100.0):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = _run("reflect", dict(td_ps=td_ps, gamma_s=0.3, gamma_l=0.4, n_bounce=6), imp, g)
        if w:
            continue
        pos = np.flatnonzero(np.abs(y) > 1e-12)
        echoes = pos[pos > 100]
        if len(echoes) == 0:
            bad.append(f"ask td_ps={td_ps:g} -> NO echo at all, output = x * {y[100]:.6f} "
                       f"({20*np.log10(y[100]):+.3f} dB of pure gain)")
            continue
        realised_ps = (echoes[0] - 100) / 2.0 / g.fs * 1e12      # first echo is at 2*td
        if abs(realised_ps - td_ps) * 1e-12 * g.fs > 0.5:
            bad.append(f"ask td_ps={td_ps:g} -> {realised_ps:.3f} ps "
                       f"({100*(realised_ps/td_ps-1):+.2f} %)")
    assert not bad, "reflect substituted the delay silently: " + "; ".join(bad)


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `impairments.drift` documents three kinds -- \"'gain' | 'amplitude' | "
    "'dc'\" -- and two of them are the same branch: `if kind in ('gain', 'amplitude')`. "
    "Measured on the same input, `drift(kind='gain', amount=0.2)` and "
    "`drift(kind='amplitude', amount=0.2)` differ by 0.000e+00 and are bit-identical "
    "(np.array_equal is True). A knob position that cannot be observed in the output is either "
    "an undocumented alias or a missing implementation; a docstring that lists it as a distinct "
    "kind makes it the second. Either implement 'amplitude' as something other than a "
    "multiplicative gain (an envelope on the signal's excursion about its mean, say, which is "
    "what the two words mean apart) or say it is an alias."))
def test_drift_kinds_are_distinguishable():
    x = _nrz(n_ui=256, n=1 << 13, seed=2)
    try:
        a = IMP.drift(x, kind="gain", amount=0.2)
        b = IMP.drift(x, kind="amplitude", amount=0.2)
    except ValueError:
        return                          # the alias was removed: an honest answer
    assert not np.array_equal(a, b), (
        f"drift(kind='gain') and drift(kind='amplitude') are bit-identical: "
        f"max|a-b| = {np.max(np.abs(a-b)):.3e}")


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `cdr.timing_source`'s `rj_ps` -- documented 'gaussian random jitter, "
    "1-sigma in ps' -- is added as INDEPENDENT PER-SAMPLE white noise, and "
    "`physics.inject_jitter`'s docstring states in as many words why that is wrong: 'Per-sample "
    "white noise would scramble the waveform, not jitter its edges -- validated.' A timing "
    "parameter must be realised as a time WARP, and a warp is monotone, so it can neither "
    "create nor destroy an edge. Measured on a 4,096-UI NRZ record at 256 GSa/s (2,063 zero "
    "crossings): `timing(rj_ps=2.0)` gives 2,191 crossings (+6.2 %, 128 fabricated) and "
    "`rj_ps=4.0` gives 2,979 (+44.4 %, 916 fabricated), because `idx - phase` stops being "
    "monotone and the record folds back on itself. `inject_jitter` at the SAME 1.0240-sample "
    "rms displacement gives 2,063 -- zero fabricated -- and the amplitude rms the two add is "
    "almost identical (0.11523 against 0.11457), so a noise-floor or an SNR check cannot see "
    "the difference and the crossing count can. The fix is `inject_jitter`'s: band-limit the "
    "Rj sequence before it becomes a phase, then renormalise to the requested rms."))
def test_a_timing_parameter_is_realised_as_a_monotone_time_warp():
    x = _nrz(n_ui=4096, n=4096 * SPB, seed=8)
    g = _grid(n=len(x))
    base = _n_zero_crossings(x)
    bad = []
    for rj_ps in (1.0, 2.0, 4.0):
        ph = CDR.timing_source(len(x), g, rj_ps=rj_ps, rng=np.random.default_rng(3))
        y = CDR.apply_phase(x, ph)
        got = _n_zero_crossings(y)
        if got != base:
            bad.append(f"rj_ps={rj_ps:g} ({rj_ps*1e-12*g.fs:.4f} samples rms): {got} crossings "
                       f"vs {base} ({got-base:+d} fabricated, {100*(got/base-1):+.1f} %)")
    assert not bad, ("a time warp cannot create edges: " + "; ".join(bad))


def test_inject_jitter_is_realised_as_a_monotone_time_warp():
    """The same invariant on `physics.inject_jitter`, which PASSES. It is the control that makes
    the failure above a defect in one implementation rather than an impossible assertion:
    measured, 2,063 crossings unchanged at every rms up to 1.0240 samples."""
    x = _nrz(n_ui=4096, n=4096 * SPB, seed=8)
    base = _n_zero_crossings(x)
    for sigma in (0.064, 0.256, 1.024):
        y = P.inject_jitter(x, sigma_rj=sigma, rng=np.random.default_rng(3))
        assert _n_zero_crossings(y) == base, (
            f"sigma_rj={sigma:g} samples: {_n_zero_crossings(y)} crossings vs {base}")


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `physics.nominal_nonlinearity`'s `rise_fall_ratio` is DISCONTINUOUS at "
    "its own nominal value, because the whole asymmetry block is behind "
    "`if rise_fall_ratio != 1.0` and the one-pole it then runs has a_base = 0.4 whether the "
    "asymmetry is 1e-7 or 1. Measured on a 64-samples/UI NRZ record with tr_frac=0.25 "
    "(ptp 2.033): ratio 1.0 -> 1.0000001 moves the waveform by max 0.12048, 5.923 % of its "
    "2.0342 span (rms 0.02887), and moves the 20-80 % rise time from 16.528 to 16.799 samples "
    "(+1.64 %) with the FALL time moving identically -- for a requested asymmetry change of one "
    "part in ten million. For scale, the ENTIRE remaining sweep from there to ratio=2.0 moves "
    "the waveform 0.08008, 3.937 % of span, and the realised fall/rise ratio only to 1.0332 -- "
    "so the step at the origin is 1.50x the knob's whole remaining range, and it is not asymmetry at all, it is an "
    "unrequested a=0.4 one-pole band limit switching on. Anyone sweeping rise/fall asymmetry "
    "from nominal reads that step as signal. Separately, `a_r = min(0.95, a_base*sqrt(r))` "
    "substitutes silently once a_base*sqrt(r) > 0.95 (r > 5.64): the realised internal a_r/a_f "
    "is 5.82 for a request of 6, 11.88 for 25 and 23.75 for 100. The realised 20-80 % ratio "
    "stays monotone through that clamp (1.116 at r=5.64, 1.294 at 16, 1.872 at 64), so the "
    "clamp costs sensitivity rather than direction -- the discontinuity is the defect. Fix by "
    "running the pole always, with a_r = a_f = a_base at ratio 1.0."))
def test_nonlinearity_rise_fall_ratio_is_continuous_at_its_nominal_value():
    spb = 64
    sym = np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0] * 40)
    x = P.from_symbols(sym, n=len(sym) * spb, tr_frac=0.25, causal=True)
    y1 = P.nominal_nonlinearity(x, compression=0.0, rise_fall_ratio=1.0)
    y2 = P.nominal_nonlinearity(x, compression=0.0, rise_fall_ratio=1.0 + 1e-7)
    jump = float(np.max(np.abs(y2 - y1))) / float(np.ptp(x))
    tr1, _tf1 = _rise_fall_20_80(y1, spb)
    tr2, _tf2 = _rise_fall_20_80(y2, spb)
    y_far = P.nominal_nonlinearity(x, compression=0.0, rise_fall_ratio=2.0)
    span = float(np.max(np.abs(y_far - y2))) / float(np.ptp(x))
    assert jump < 0.1 * span, (
        f"a 1e-7 nudge off rise_fall_ratio=1.0 moves the waveform {100*jump:.3f} % of span "
        f"(20-80 % rise time {tr1:.2f} -> {tr2:.2f} samples) while the whole remaining sweep to "
        f"ratio=2.0 moves it {100*span:.3f} % -- the knob has a step at its nominal value")


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT (a knob that does nothing, silently): `instrument.scope_bandwidth` accepts "
    "`order=` for every kind and reads it only for 'bessel'. Measured on the same input at "
    "bw_hz=32e9, max|y(order=k) - y(order=1)| for k in (2, 4, 8) is [1.051, 2.112, 2.696] for "
    "'bessel' and exactly [0.0, 0.0, 0.0] for 'gaussian' and [0.0, 0.0, 0.0] for 'brickwall'. "
    "`probe`'s docstring passes the flag straight through -- 'kind/order are scope_bandwidth's' "
    "-- so `probe(kind='gaussian', order=8)` states a front-end order it does not have. The "
    "module has ALREADY made this exact argument about a different flag, and acted on it: "
    "kind='brickwall' now RAISES on `causal=True` because 'quietly accepting causal=True there "
    "would make the flag a lie.' `order` on a Gaussian or a brickwall is the same lie. Raise, "
    "the way `causal` does."))
def test_scope_bandwidth_rejects_an_order_it_ignores():
    g = _grid(n = 1 << 14)
    x = np.random.default_rng(0).standard_normal(g.n)
    bad = []
    for kind in ("gaussian", "brickwall"):
        moved = False
        try:
            base = INST.scope_bandwidth(x, g, 32e9, kind=kind, order=1)
            for order in (2, 4, 8):
                y = INST.scope_bandwidth(x, g, 32e9, kind=kind, order=order)
                moved = moved or float(np.max(np.abs(y - base))) > 1e-12
        except (TypeError, ValueError):
            continue                    # refusing the parameter is the honest answer
        if not moved:
            bad.append(f"kind={kind!r}: order in (2, 4, 8) changes nothing (max|dy| = 0.0)")
    assert not bad, "scope_bandwidth accepted an order it ignores: " + "; ".join(bad)


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT, and the most severe one here: `physics.insertion_loss_db` applies the "
    "absolute loss anchor only under `if loss_db is not None and loss_at_ghz is not None`, so "
    "HALF A PAIR IS DISCARDED IN SILENCE. On a 256 GSa/s grid with the default geometry, "
    "`lossy(loss_db=6.0)`, `lossy(loss_db=12.0)` and `lossy(loss_db=24.0)` -- each with "
    "`loss_at_ghz` omitted -- all realise 10.5192 dB of insertion loss at 8 GHz, which is the "
    "length_in/tand default law (closed form 10.5183 dB), and the three output waveforms are "
    "BIT-IDENTICAL: max|y(6 dB) - y(24 dB)| = 0.000e+00, np.array_equal is True. `loss_at_ghz` "
    "alone is dropped the same way. Zero warnings in every case. Paired, the anchor is exact "
    "(6.0001 and 12.0011 dB measured for 6 and 12), so the arithmetic is right and only the "
    "guard is wrong. Two consequences worth naming: a caller who states a channel's loss "
    "budget and forgets the frequency receives a 1.75x different channel and is told nothing, "
    "and a SWEEP over loss_db with the frequency omitted emits a dataset whose every record is "
    "byte-identical while its label column varies -- the exact shape of the `noise_mv` that "
    "shipped stored-and-never-read. Raise when exactly one of the pair is given."))
def test_lossy_does_not_silently_discard_half_of_the_loss_anchor():
    g = _grid()
    imp = np.zeros(g.n)
    imp[g.n // 2] = 1.0
    realised, outs = [], []
    for loss_db in (6.0, 12.0, 24.0):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                y = _run("lossy", dict(causal=True, loss_db=loss_db), imp, g)
            except (ValueError, TypeError):
                return                      # refusing half a pair is the honest answer
            H, _f = _H_at(lambda z: _run("lossy", dict(causal=True, loss_db=loss_db), z, g), 8e9)
        if w:
            return                          # warning about the substitution is also honest
        realised.append(-20.0 * np.log10(abs(H)))
        outs.append(y)
    assert not np.array_equal(outs[0], outs[-1]), (
        f"loss_db=6 and loss_db=24 with loss_at_ghz omitted are bit-identical "
        f"(max|dy| = {np.max(np.abs(outs[0]-outs[-1])):.3e}); realised IL at 8 GHz was "
        f"{['%.4f dB' % v for v in realised]} for requests of 6 / 12 / 24 dB")


# ==================================================================================
# 2. CLEAN -- parameters that DO track their request, at the right absolute value.
#    A check never observed failing is not a check, so each of these is stated against
#    a closed form or an invariant, not against a stored number.
# ==================================================================================
@pytest.mark.parametrize("loss_db,at_ghz", [(3.0, 4.0), (6.0, 8.0), (10.0, 8.0), (20.0, 16.0)])
def test_lossy_realises_loss_db_at_loss_at_ghz(loss_db, at_ghz):
    """The absolute-anchor form of a lossy channel: `loss_db` dB of insertion loss AT
    `loss_at_ghz`. Measured error 0.000-0.001 dB across the sweep."""
    H, f = _H_at(lambda z: _run("lossy", dict(loss_db=loss_db, loss_at_ghz=at_ghz,
                                              causal=True), z), at_ghz * 1e9)
    got = -20.0 * np.log10(abs(H))
    assert abs(got - loss_db) < 0.05, f"ask {loss_db} dB @ {at_ghz} GHz, realised {got:.3f} dB"


@pytest.mark.parametrize("db", [1.0, 3.0, 6.0, 9.0])
def test_de_emphasis_realises_its_stated_db(db):
    """`de_emphasis_taps` claims 20*log10(V_transition/V_steady) = db. Measured exact to
    1e-4 dB on ideal rectangular symbols at integer samples/UI, which is where the closed form
    is the answer and nothing of ours is."""
    sym = np.array([-1.0] * 8 + [1.0] * 8 + [-1.0] * 8 + [1.0] * 8)
    x = np.repeat(sym, SPB)
    y = P.tx_ffe(x, P.de_emphasis_taps(db), SPB, pre=0)
    trans = y[8 * SPB + SPB // 2]
    steady = y[8 * SPB + 7 * SPB + SPB // 2]
    got = 20.0 * np.log10(abs(trans / steady))
    assert abs(got - db) < 0.01, f"ask {db} dB, realised {got:.4f} dB"


@pytest.mark.parametrize("dc_gain", [0.4, 0.65, 1.0])
def test_ctle_realises_dc_gain_and_its_analytic_transfer(dc_gain):
    """`ctle`'s docstring: 'dc_gain is the exact DC gain'. And the whole H(s) is a closed form,
    so the realised |H| is checkable at every frequency, not just at DC. Measured: DC gain to
    5e-5, and |H| within 0.02 dB of the analytic response out to 20 GHz (the residual is the
    bilinear warp, which reaches -0.69 dB only at 40 GHz against an 8 GHz second pole)."""
    fz, fp1, fp2 = 2.0, 8.0, 32.0
    fn = lambda z: _run("ctle", dict(fz_ghz=fz, fp1_ghz=fp1, fp2_ghz=fp2, dc_gain=dc_gain), z)

    def analytic(f):
        s = 1j * 2 * np.pi * f
        wz, wp1, wp2 = (2 * np.pi * v * 1e9 for v in (fz, fp1, fp2))
        return dc_gain * (1 + s / wz) / ((1 + s / wp1) * (1 + s / wp2))

    H0, _ = _H_at(fn, 20e6)
    assert abs(abs(H0) / dc_gain - 1.0) < 1e-3, f"DC gain ask {dc_gain}, realised {abs(H0):.5f}"
    for f_ghz in (1.0, 4.0, 10.0, 20.0):
        H, f_a = _H_at(fn, f_ghz * 1e9)
        err = 20.0 * np.log10(abs(H) / abs(analytic(f_a)))
        assert abs(err) < 0.15, f"|H| at {f_ghz} GHz is {err:+.3f} dB off the closed form"


@pytest.mark.parametrize("c_load_f", [0.1e-12, 0.45e-12, 1.0e-12])
def test_probe_realises_the_rc_pole_its_capacitance_names(c_load_f):
    """`probe` applies `probe_loading(causal=True)`, which claims the exact single pole
    H = 1/(1 + j f/fc) with fc = 1/(2*pi*R*C). Measured |H(fc)| = 0.70705-0.70710 against
    0.70711 and phase -45.00 deg against -45 -- magnitude AND phase, which is what separates
    an honest single pole from `sosfiltfilt`'s square (see the ac_couple finding)."""
    g = _grid()
    fc = INST.rc_pole_hz(50.0, c_load_f)
    H, _f = _H_at(lambda z: _run("probe", dict(c_load_f=c_load_f, r_source=50.0), z, g), fc)
    assert abs(abs(H) - 0.70711) < 0.002, f"|H(pole)| = {abs(H):.5f} at {fc/1e9:.3f} GHz"
    assert abs(np.degrees(np.angle(H)) + 45.0) < 1.0, (
        f"phase at the pole is {np.degrees(np.angle(H)):+.2f} deg, not -45")


@pytest.mark.parametrize("atten", [1.0, 0.5, 0.1])
def test_probe_atten_is_the_divider_ratio_it_claims(atten):
    x = _nrz()
    g = _grid()
    y = _run("probe", dict(atten=atten), x, g)
    ref = _run("probe", dict(atten=1.0), x, g)
    assert abs(np.std(y) / np.std(ref) - atten) < 1e-6, (
        f"atten={atten}: realised gain {np.std(y)/np.std(ref):.6f}")


@pytest.mark.parametrize("noise_rms", [1e-4, 1e-3, 1e-2])
def test_probe_noise_rms_lands_at_the_rms_it_names(noise_rms):
    x = _nrz()
    g = _grid()
    y = INST.probe(x, g, noise_rms=noise_rms, rng=np.random.default_rng(0))
    got = float(np.std(y - INST.probe(x, g, noise_rms=0.0)))
    assert abs(got / noise_rms - 1.0) < 0.02, f"ask {noise_rms:g}, realised {got:.6g}"


@pytest.mark.parametrize("coupling", [0.01, 0.05, 0.2])
@pytest.mark.parametrize("kind", ["fext", "next"])
def test_crosstalk_coupling_is_the_fraction_of_the_victim_span_it_claims(coupling, kind):
    """'`coupling` scales the aggressor relative to x's span'. Measured 0.01000 / 0.05000 /
    0.20000 of ptp for both kinds -- exact, and the same for FEXT (a derivative) and NEXT (a
    delayed copy), which is the point of normalising the kernel to unit peak."""
    x = _nrz(n_ui=512, n=1 << 14, seed=6)
    g = _grid(n=1 << 14)
    y = _run("crosstalk", dict(coupling=coupling, kind=kind, td_frac=0.05,
                               aggressor=dict(n_ui=64, seed=7)), x, g)
    got = float(np.max(np.abs(y - x)) / np.ptp(x))
    assert abs(got - coupling) < 1e-4, f"coupling={coupling} {kind}: realised {got:.5f}"


@pytest.mark.parametrize("q,gamma0", [(3.0, 0.2), (9.0, 0.2), (30.0, 0.2),
                                      (9.0, 0.05), (9.0, 0.6)])
def test_resonant_reflect_matches_its_closed_form_gamma_at_f0(q, gamma0):
    """`resonant_reflection`'s Gamma(f) is written down in the docstring, so the realised |H| at
    f0 is a closed form: 1 + gamma0*(sv/q)/(sv^2 + sv/q + 1)*exp(-j*2*pi*f*td). Measured error
    6.9e-05 to 8.4e-04 across the sweep -- and note td is honoured at FULL FRACTIONAL
    resolution here, which is exactly what `multi_reflection` does not do."""
    g = _grid()
    td_ps, f0_ghz = 90.0, 10.7
    H, f = _H_at(lambda z: _run("resonant_reflect", dict(td_ps=td_ps, f0_ghz=f0_ghz, q=q,
                                                         gamma0=gamma0), z, g), f0_ghz * 1e9)
    sv = 1j * (f / (f0_ghz * 1e9))
    G = gamma0 * (sv / q) / (sv ** 2 + sv / q + 1.0)
    expect = 1.0 + G * np.exp(-1j * 2 * np.pi * f * td_ps * 1e-12)
    assert abs(abs(H) - abs(expect)) < 5e-3, (
        f"q={q} gamma0={gamma0}: |H(f0)| = {abs(H):.6f}, closed form {abs(expect):.6f}")


@pytest.mark.parametrize("bits", [8, 10, 11, 12, 14])
def test_store_record_bits_lands_on_its_own_lattice_floor(bits):
    """A uniform quantiser of step q has error rms q/sqrt(12) = 0.288675 LSB, so `bits` is
    checkable without a reference output. THE FIXTURE MATTERS AND IT BIT ME: measured on a bare
    NRZ record with no noise the answer is 0.2414 / 0.3570 / 0.2414 LSB at 11 / 10 / 11 bits and
    not even monotone, because a two-level waveform's rounding error is deterministic, not
    uniform -- the same trap as gating a noise floor on noise with no signal in it. With the
    input dithered by real noise (4 mV rms here, well under an LSB of no consequence) every
    depth from 8 to 14 bits lands within 0.22 % of the closed form."""
    x = _nrz(scale=0.4) + np.random.default_rng(9).normal(0, 0.004, N)
    y = INST.store_record(x, bits=bits)
    lsb = 2.0 * 1.05 * float(np.max(np.abs(x))) / 2.0 ** bits
    got = float(np.std(y - x)) / lsb
    assert abs(got / 0.288675 - 1.0) < 0.02, f"bits={bits}: err rms {got:.5f} LSB"
    assert float(np.max(np.abs(y - x))) / lsb < 0.501, "a sample moved more than half an LSB"


@pytest.mark.parametrize("dither_lsb", [0.0, 0.2887, 0.5, 1.0])
def test_store_record_dither_lsb_adds_the_power_it_names(dither_lsb):
    """Independent dither of d LSB rms in front of the rounder gives total error rms
    sqrt(1/12 + d^2). Measured 0.28931 / 0.40678 / 0.57675 / 1.03633 against 0.28868 / 0.40827 /
    0.57735 / 1.04083 -- and 0.2887 LSB lands at +2.979 dB over q**2/12, against the +3.01 dB
    the docstring names it for."""
    x = _nrz(scale=0.4) + np.random.default_rng(9).normal(0, 0.004, N)
    y = INST.store_record(x, bits=11, dither_lsb=dither_lsb, rng=np.random.default_rng(1))
    lsb = 2.0 * 1.05 * float(np.max(np.abs(x))) / 2.0 ** 11
    got = float(np.std(y - x)) / lsb
    want = float(np.sqrt(1.0 / 12.0 + dither_lsb ** 2))
    assert abs(got / want - 1.0) < 0.02, (
        f"dither_lsb={dither_lsb}: err rms {got:.5f} LSB, closed form {want:.5f}")


@pytest.mark.parametrize("rms_ps", [0.1, 0.5, 2.0, 10.0])
def test_timebase_jitter_realises_the_rms_it_names(rms_ps):
    """Measured on a linear ramp, where the interpolation is exact and the displacement is
    readable directly: 0.0998 / 0.4988 / 1.9951 / 9.9754 ps for 0.1 / 0.5 / 2 / 10."""
    g = _grid()
    ramp = np.arange(g.n, dtype=float)
    y = INST.timebase_jitter(ramp, g, rms_ps=rms_ps, rng=np.random.default_rng(3))
    got = float(np.std(ramp - y)) / g.fs * 1e12
    assert abs(got / rms_ps - 1.0) < 0.02, f"ask {rms_ps} ps, realised {got:.4f} ps"


@pytest.mark.parametrize("am_depth", [0.005, 0.02, 0.1])
def test_supply_coupling_am_depth_is_the_modulation_index(am_depth):
    """y = x*(1 + am_depth*sin(...)), so on a DC input the realised half-span IS am_depth.
    MEASURE IT OVER WHOLE CYCLES: on the 256 ns record a 256 GSa/s grid gives, a 2.5 MHz
    ripple is 0.64 of one cycle and the sine never reaches 1, which reads as a 0.885x defect
    that is entirely the fixture's. On a 32.8 us record it is exact."""
    g = _grid(fs=2e9, baud=1e9, n=1 << 16)
    y = P.supply_coupling(np.ones(g.n), g, f_ripple_hz=2.5e6, am_depth=am_depth)
    got = 0.5 * float(np.ptp(y))
    assert abs(got / am_depth - 1.0) < 0.01, f"ask {am_depth:g}, realised {got:.6g}"


@pytest.mark.parametrize("psij_ps", [0.5, 2.0, 10.0])
def test_supply_coupling_psij_ps_is_the_peak_timing_deviation(psij_ps):
    g = _grid()
    ramp = np.arange(g.n, dtype=float)
    y = P.supply_coupling(ramp, g, f_ripple_hz=8e9, am_depth=0.0, psij_ps=psij_ps)
    got = float(np.max(np.abs((ramp - y)[10:-10]))) / g.fs * 1e12
    assert abs(got / psij_ps - 1.0) < 0.02, f"ask {psij_ps} ps, realised {got:.4f} ps"


@pytest.mark.parametrize("f_ripple_hz", [50e6, 250e6, 1e9])
def test_supply_coupling_puts_the_ripple_tone_where_it_was_asked_for(f_ripple_hz):
    g = _grid()
    y = P.supply_coupling(np.ones(g.n), g, f_ripple_hz=f_ripple_hz, am_depth=0.02)
    k = int(np.argmax(np.abs(np.fft.rfft(y - 1.0))))
    got = k * g.fs / g.n
    assert abs(got - f_ripple_hz) < 1.5 * g.fs / g.n, (
        f"ask {f_ripple_hz/1e6:g} MHz, peak at {got/1e6:g} MHz")


@pytest.mark.parametrize("spread", [0.001, 0.005, 0.02])
def test_ssc_spread_is_the_peak_fractional_frequency_deviation(spread):
    """`ssc_phase` returns cumulative phase, so the instantaneous fractional frequency
    deviation is its first difference and its peak must be `spread`. MEASURE OVER A WHOLE SSC
    PERIOD: on a 4.1 us record at 256 GSa/s a 32 kHz triangle has covered 13 % of one period
    and peaks at 0.262*spread, which is the fixture, not the op. Over 4 periods: 1.0000x."""
    fs, f_ssc = 1e9, 32e3
    ph = CDR.ssc_phase(int(4 * fs / f_ssc), fs, f_ssc, spread, "down")
    got = float(np.max(np.abs(np.diff(ph))))
    assert abs(got / spread - 1.0) < 0.01, f"ask {spread:g}, realised {got:.6f}"


@pytest.mark.parametrize("amount", [0.004, 0.05, 0.2])
@pytest.mark.parametrize("kind", ["gain", "dc"])
def test_drift_amount_reaches_exactly_the_amount_at_the_record_end(kind, amount):
    """'linear' is a 0->1 profile across the record, so at the last sample the realised effect
    is exactly `amount` (gain) or `amount * ptp` (dc). Measured exact to 1e-6 for both."""
    x = np.ones(4096) if kind == "gain" else _nrz(n_ui=128, n=4096, seed=8)
    y = IMP.drift(x, kind=kind, amount=amount, shape="linear")
    want = amount if kind == "gain" else amount * float(np.ptp(x))
    assert abs((y[-1] - x[-1]) - want) < 1e-6 * max(1.0, abs(want)), (
        f"kind={kind} amount={amount}: realised {y[-1]-x[-1]:+.6f}, expected {want:+.6f}")


@pytest.mark.parametrize("sigma_rj", [0.05, 0.2, 1.0])
def test_inject_jitter_sigma_rj_lands_at_the_edges(sigma_rj):
    """Documented 'renormalized to RMS = sigma_rj samples'. Measured as the rms displacement of
    the record's own zero crossings: 0.05068 / 0.20396 / 1.01929 for 0.05 / 0.2 / 1.0 -- 1.4 to
    2.0 % high, which is the band-limited sequence being slightly hotter at an edge than over
    the whole record."""
    x = _nrz(n_ui=2048, n=2048 * SPB, seed=8)
    y = P.inject_jitter(x, sigma_rj=sigma_rj, rng=np.random.default_rng(2))

    def crossings(v):
        s = np.signbit(v)
        i = np.flatnonzero(s[:-1] != s[1:])
        return i + (0.0 - v[i]) / (v[i + 1] - v[i])

    a, b = crossings(x), crossings(y)
    m = min(len(a), len(b))
    got = float(np.std(b[:m] - a[:m]))
    assert abs(got / sigma_rj - 1.0) < 0.05, f"ask {sigma_rj} samples, realised {got:.5f}"


@pytest.mark.parametrize("er_db", [3.0, 6.0, 10.0, 20.0])
def test_to_optical_realises_the_extinction_ratio_and_mean_power(er_db):
    """ER = 10*log10(P_high/P_low) and the mean is scaled to p_avg. Measured exact to 1e-4 dB
    and 1e-6 respectively."""
    x = _nrz(n_ui=256, n=1 << 13, seed=2)
    p = OPT.to_optical(x, er_db=er_db, p_avg=2.0)
    got = 10.0 * np.log10(float(np.max(p) / np.min(p)))
    assert abs(got - er_db) < 0.01, f"ask {er_db} dB, realised {got:.4f} dB"
    assert abs(float(np.mean(p)) - 2.0) < 1e-6, f"mean power {np.mean(p):.6f}, ask 2.0"


@pytest.mark.parametrize("value", [0.0, 0.01, 0.05])
def test_interleave_adc_realises_a_stated_per_core_gain_and_offset(value):
    """The deterministic `mismatch=(g, o, s)` path, which is the only one whose realised value
    is the requested one (the rng path draws M samples from a distribution of that std, so a
    4-core draw is not 1 %). Measured: a uniform +1 % core gain gives exactly 1.010000x, and a
    +/-o offset pattern gives exactly o of per-core offset rms."""
    x = _nrz(n_ui=512, n=1 << 14, seed=6)
    y = INST.interleave_adc(x, m_cores=4, mismatch=(np.full(4, value), np.zeros(4), np.zeros(4)))
    assert abs(np.std(y) / np.std(x) - (1.0 + value)) < 1e-6, (
        f"gain {value:g}: realised {np.std(y)/np.std(x):.6f}")
    y = INST.interleave_adc(x, m_cores=4,
                            mismatch=(np.zeros(4), np.array([value, -value, value, -value]),
                                      np.zeros(4)))
    assert abs(float(np.std(y - x)) - value) < 1e-9, (
        f"offset {value:g}: realised {np.std(y-x):.6g}")


@pytest.mark.parametrize("skew_ps", [1.0, 4.0, 10.0])
def test_intra_pair_skew_is_exactly_a_delay_on_the_n_leg(skew_ps):
    """`differential_pair`'s common mode is an identity, not an approximation:
    (p+n)/2 = 0.25*(x - x delayed by skew). Asserting the identity checks the parameter's
    realised value without needing a separate model of it. Measured cm rms 0.007221 / 0.028863 /
    0.071001 at 1 / 4 / 10 ps, monotone in the request."""
    g = _grid(n=1 << 14)
    x = _nrz(n_ui=512, n=g.n, seed=8)
    p, nn = P.differential_pair(x, grid=g, skew_ps=skew_ps)
    cm = P.common_mode(p, nn)
    d = skew_ps * 1e-12 * g.fs
    idx = np.arange(g.n)
    want = 0.25 * (x - np.interp(idx - d, idx, x, left=x[0], right=x[-1]))
    assert float(np.max(np.abs(cm - want))) < 1e-12, (
        f"skew_ps={skew_ps}: common mode is not 0.25*(x - x(t-{d:.4f} samples))")


def test_intra_pair_skew_common_mode_grows_with_the_request():
    g = _grid(n=1 << 14)
    x = _nrz(n_ui=512, n=g.n, seed=8)
    got = [float(np.std(P.common_mode(*P.differential_pair(x, grid=g, skew_ps=s))))
           for s in (0.5, 1.0, 2.0, 4.0, 10.0)]
    assert all(b > a for a, b in zip(got, got[1:])), f"not monotone in skew_ps: {got}"


# ==================================================================================
# 3. THE noise_mv GATE -- every numeric parameter must demonstrably do something
# ==================================================================================
# (op, base params, param, two values that must produce different output)
MOVABLE = [
    ("lossy", dict(causal=True, loss_at_ghz=8.0), "loss_db", (6.0, 12.0)),
    ("lossy", dict(causal=True, loss_db=6.0), "loss_at_ghz", (4.0, 16.0)),
    ("lossy", dict(causal=True, trend=(-1.2, -0.05, 0.0)), "trend_floor_db", (6.0, 20.0)),
    ("lossy", dict(causal=True), "length_in", (4.0, 12.0)),
    ("lossy", dict(causal=True), "tand", (0.01, 0.025)),
    ("lossy", dict(causal=True), "eps_r", (3.5, 4.5)),
    ("reflect", dict(td_ps=40.0, gamma_l=0.06), "gamma_s", (0.02, 0.2)),
    ("reflect", dict(td_ps=40.0, gamma_s=0.06), "gamma_l", (0.02, 0.2)),
    ("reflect", dict(td_ps=40.0, gamma_s=0.3, gamma_l=0.4), "n_bounce", (1, 6)),
    ("resonant_reflect", dict(td_ps=90.0, f0_ghz=10.7, gamma0=0.2), "q", (3.0, 30.0)),
    ("resonant_reflect", dict(td_ps=90.0, q=9.0, gamma0=0.2), "f0_ghz", (6.0, 18.0)),
    ("resonant_reflect", dict(td_ps=90.0, f0_ghz=10.7, q=9.0), "gamma0", (0.05, 0.6)),
    ("ctle", dict(fp1_ghz=8.0, fp2_ghz=16.0), "fz_ghz", (1.0, 4.0)),
    ("ctle", dict(fz_ghz=2.0, fp2_ghz=16.0), "fp1_ghz", (6.0, 12.0)),
    ("ctle", dict(fz_ghz=2.0, fp1_ghz=8.0), "fp2_ghz", (12.0, 40.0)),
    ("ctle", dict(fz_ghz=2.0, fp1_ghz=8.0, fp2_ghz=16.0), "dc_gain", (0.4, 1.0)),
    ("de_emphasis", dict(), "db", (1.0, 6.0)),
    ("ac_couple", dict(), "fc_hz", (5e8, 4e9)),
    ("scope", dict(kind="bessel"), "bw_hz", (16e9, 48e9)),
    ("scope", dict(kind="bessel", bw_hz=32e9), "order", (1, 4)),
    ("probe", dict(), "c_load_f", (0.1e-12, 1.0e-12)),
    ("probe", dict(c_load_f=0.45e-12), "r_source", (25.0, 100.0)),
    ("probe", dict(), "atten", (0.1, 1.0)),
    ("probe", dict(), "noise_rms", (0.0, 0.01)),
    ("probe", dict(), "bw_hz", (20e9, 60e9)),
    ("crosstalk", dict(kind="fext"), "coupling", (0.01, 0.2)),
    ("crosstalk", dict(kind="next", coupling=0.1), "td_frac", (0.02, 0.2)),
    ("supply_coupling", dict(f_ripple_hz=1e9), "am_depth", (0.0, 0.05)),
    ("supply_coupling", dict(f_ripple_hz=1e9, am_depth=0.02), "psij_ps", (0.0, 4.0)),
    ("supply_coupling", dict(am_depth=0.02), "f_ripple_hz", (5e8, 4e9)),
    ("nonlinearity", dict(), "compression", (0.01, 0.1)),
    ("nonlinearity", dict(compression=0.04), "level_noise", (0.0, 0.02)),
    ("nonlinearity", dict(compression=0.0), "rise_fall_ratio", (1.5, 8.0)),
    ("nonlinearity", dict(compression=0.0, rise_fall_ratio=2.0), "a_base", (0.2, 0.6)),
    ("drift", dict(kind="gain"), "amount", (0.004, 0.2)),
    ("intra_pair_skew", dict(), "skew_ps", (1.0, 8.0)),
    ("intra_pair_skew", dict(skew_ps=2.0), "gain_imbalance", (0.0, 0.05)),
    ("timebase", dict(), "rms_ps", (0.1, 4.0)),
    ("store", dict(), "bits", (8, 12)),
    ("store", dict(bits=11), "headroom", (1.05, 2.0)),
    ("store", dict(bits=11), "dither_lsb", (0.0, 0.5)),
    ("digitize", dict(), "noise_rms", (0.0, 0.01)),
    ("digitize", dict(), "snr_db", (30.0, 50.0)),
    ("digitize", dict(), "bits", (8, 12)),
    ("tx_ffe", dict(taps=[-0.1, 0.75, -0.15]), "pre", (0, 1)),
    ("rx_ffe", dict(taps=[-0.05, 1.0, -0.18], pre=1), "spacing_ui", (0.5, 1.0)),
    ("dfe", dict(levels=[-1.0, 1.0]), "taps", ([0.02, 0.01], [0.3, 0.15])),
    ("ssc", dict(f_ssc=32e3), "spread", (0.001, 0.02)),
    ("timing", dict(), "rj_ps", (0.0, 2.0)),
]


@pytest.mark.parametrize("op,base,param,values", MOVABLE,
                         ids=[f"{o}.{p}" for o, _b, p, _v in MOVABLE])
def test_every_numeric_parameter_demonstrably_does_something(op, base, param, values):
    """THE `noise_mv` GATE. This project shipped a `noise_mv` that was stored and never read for
    weeks. Move each knob between two values inside its documented range; the output must move.
    Two things this catches that a spec review does not: a parameter accepted and dropped by a
    forwarding whitelist (see BACKLOG #53, where `_op_lossy` still drops `linear` and `guard`),
    and one read only on a branch the default never takes."""
    g = _grid(n=1 << 14)
    x = _nrz(n_ui=512, n=g.n, seed=6, scale=0.4)
    lo, hi = values
    a = _run(op, dict(base, **{param: lo}), x, g, seed=11)
    b = _run(op, dict(base, **{param: hi}), x, g, seed=11)
    m = min(len(a), len(b))
    moved = float(np.max(np.abs(a[:m] - b[:m])))
    assert moved > 1e-12, (
        f"{op}.{param}: moving it from {lo!r} to {hi!r} changed the output by {moved:.3e} -- "
        f"the knob is stored and never read, or read only on a branch the default avoids")


# ==================================================================================
# 4. MONOTONICITY -- the realised effect must track the request in the right direction
# ==================================================================================
MONOTONE = [
    ("lossy loss_db -> insertion loss at 8 GHz",
     lambda v: -20 * np.log10(abs(_H_at(lambda z: _run("lossy", dict(loss_db=v, loss_at_ghz=8.0,
                                                                     causal=True), z), 8e9)[0])),
     (3.0, 6.0, 10.0, 16.0, 24.0), True),
    ("de_emphasis db -> |H| ratio 8 GHz vs 0.5 GHz",
     lambda v: (abs(_H_at(lambda z: _run("de_emphasis", dict(db=v), z), 8e9)[0])
                / abs(_H_at(lambda z: _run("de_emphasis", dict(db=v), z), 5e8)[0])),
     (0.5, 1.0, 3.0, 6.0, 9.0), True),
    ("scope bw_hz -> realised -3 dB point",
     lambda v: _bisect_3db(lambda z: _run("scope", dict(bw_hz=v, kind="bessel"), z), v,
                           False, span=8.0),
     (8e9, 16e9, 32e9, 64e9), True),
    ("ac_couple fc_hz -> realised -3 dB point",
     lambda v: _bisect_3db(lambda z: _run("ac_couple", dict(fc_hz=v), z), v, True, span=20.0),
     (2e8, 5e8, 1e9, 2e9, 4e9), True),
    ("probe c_load_f -> realised pole (falling)",
     lambda v: _bisect_3db(lambda z: _run("probe", dict(c_load_f=v, r_source=50.0), z),
                           INST.rc_pole_hz(50.0, v), False, span=20.0),
     (0.1e-12, 0.2e-12, 0.45e-12, 1.0e-12), False),
    ("store bits -> distinct stored codes",
     lambda v: len(np.unique(INST.store_record(
         _nrz(scale=0.4) + np.random.default_rng(9).normal(0, 0.004, N), bits=v))),
     (8, 9, 10, 11, 12), True),
]


@pytest.mark.parametrize("label,measure,values,rising", MONOTONE,
                         ids=[m[0].split(" ")[0] + "." + m[0].split(" ")[1] for m in MONOTONE])
def test_the_realised_effect_tracks_the_request_monotonically(label, measure, values, rising):
    got = [float(measure(v)) for v in values]
    ok = all((b > a) == rising for a, b in zip(got, got[1:]))
    assert ok, f"{label}: requests {values} -> realised {['%.4g' % v for v in got]}"


# ==================================================================================
# direct run
# ==================================================================================
def _report():
    line = "-" * 92
    print(__doc__.split("Runnable two ways")[0].strip()[:0] or "", end="")
    print(f"\n{line}\nPARAMETER AUDIT -- measured numbers\n{line}")

    g = _grid()
    print("\n[0] GROUND TRUTH -- the corner bisector, calibrated both ways")
    for fc in (0.5e9, 2e9, 8e9):
        sos = _sig.butter(1, fc / (FS / 2.0), btype="high", output="sos")
        one = _bisect_3db(lambda z: _sig.sosfilt(sos, z), fc, True)
        two = _bisect_3db(lambda z: _sig.sosfiltfilt(sos, z), fc, True)
        print(f"    constructed butter(1,'high') fc={fc/1e9:5.2f} GHz: one pass {one/fc:.4f}x, "
              f"sosfiltfilt {two/fc:.4f}x (closed form 1.0000 / 1.5538)")
    real = _load_real()
    if real is None:
        print("    real capture: unavailable (private repo) -- the harder gate is SKIPPED")
    else:
        x, fs, name = real
        print(f"    real capture {name[:44]}...: {len(x)} samples at {fs/1e9:g} GSa/s, "
              f"{len(np.unique(x))} distinct codes")
        for fc in (0.5e9, 1e9, 2e9):
            sos = _sig.butter(1, fc / (fs / 2.0), btype="high", output="sos")
            for lab, fn in (("one pass  ", lambda z: _sig.sosfilt(sos, z)),
                            ("two passes", lambda z: _sig.sosfiltfilt(sos, z))):
                got = _corner_through_record(x, fs, fn, fc)
                print(f"      through the REAL record, fc={fc/1e9:.2f} GHz {lab}: "
                      f"{got/fc:.4f}x (closed form 1.0000 / 1.5538)")

    print("\n[1] ac_couple -- documented 1st order, realised as its square")
    for fc in (5e8, 2e9, 1e10):
        fn = lambda z: P.ac_couple(z, fc_hz=fc, grid=g)
        H, _f = _H_at(fn, fc)
        print(f"    ask {fc/1e9:6.2f} GHz: |H(fc)| {abs(H):.4f} ({20*np.log10(abs(H)):+.3f} dB "
              f"vs -3.010), phase {np.degrees(np.angle(H)):+8.3f} deg (vs +45.000), "
              f"-3 dB at {_bisect_3db(fn, fc, True)/fc:.4f}x, "
              f"slope {_slope_db_per_octave(fn, fc):5.2f} dB/oct (vs 6.02)")

    print("\n[2] scope_bandwidth -- a bandwidth above Nyquist, substituted in silence")
    g2 = _grid(fs=100e9, baud=16e9, n=1 << 14)
    xr = np.random.default_rng(1).standard_normal(g2.n)
    for bw in (40e9, 49e9, 60e9, 110e9, 1000e9):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = INST.scope_bandwidth(xr, g2, bw, kind="brickwall")
        nz = np.flatnonzero(np.abs(np.fft.rfft(y)) > 1e-9)
        print(f"    ask {bw/1e9:7.1f} GHz (Nyquist {g2.f_nyquist/1e9:g}) -> realised "
              f"{nz[-1]*g2.fs/g2.n/1e9:7.3f} GHz, {len(w)} warnings")

    print("\n[3] rx_ffe -- tap spacing, measured from the impulse response")
    for fs, baud in ((256e9, 16e9), (100e9, 16e9), (80e9, 25.78125e9)):
        gg = _grid(fs=fs, baud=baud, n=1 << 12)
        for want in (1.0, 0.5, 0.25):
            imp = np.zeros(gg.n)
            imp[gg.n // 2] = 1.0
            y = _run("rx_ffe", dict(taps=[0.3, 1.0, -0.2], spacing_ui=want, pre=1), imp, gg)
            pos = np.flatnonzero(np.abs(y) > 1e-12)
            got = float(np.diff(pos)[0]) / gg.samples_per_ui
            print(f"    fs={fs/1e9:6.1f} GSa/s spb={gg.samples_per_ui:8.4f}: ask {want:4.2f} UI "
                  f"-> {got:7.4f} UI ({100*(got/want-1):+8.2f} %)")

    print("\n[4] reflect -- td_ps rounded to a whole sample (3.906 ps here)")
    gg = _grid(n=1 << 13)
    imp = np.zeros(gg.n)
    imp[100] = 1.0
    for td in (1.0, 1.5, 2.0, 3.9, 40.0, 41.0, 100.0):
        y = _run("reflect", dict(td_ps=td, gamma_s=0.3, gamma_l=0.4, n_bounce=6), imp, gg)
        pos = np.flatnonzero(np.abs(y) > 1e-12)
        ech = pos[pos > 100]
        if len(ech) == 0:
            print(f"    ask {td:6.2f} ps -> NO echo; output = x * {y[100]:.6f} "
                  f"({20*np.log10(y[100]):+.3f} dB of pure gain)")
        else:
            r = (ech[0] - 100) / 2.0 / gg.fs * 1e12
            print(f"    ask {td:6.2f} ps -> {r:8.3f} ps ({100*(r/td-1):+7.2f} %), "
                  f"{len(ech)} echoes")

    print("\n[5] lossy -- half a loss anchor, discarded in silence")
    for pp in (dict(causal=True), dict(causal=True, loss_db=6.0), dict(causal=True, loss_db=12.0),
               dict(causal=True, loss_db=24.0), dict(causal=True, loss_at_ghz=8.0),
               dict(causal=True, loss_db=6.0, loss_at_ghz=8.0),
               dict(causal=True, loss_db=12.0, loss_at_ghz=8.0)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            H, _f = _H_at(lambda z: _run("lossy", pp, z), 8e9)
        kw = ", ".join(f"{k}={v}" for k, v in pp.items() if k != "causal") or "(defaults)"
        print(f"    lossy({kw:34s}) -> IL(8 GHz) = {-20*np.log10(abs(H)):8.4f} dB, "
              f"{len(w)} warnings")
    ia = _run("lossy", dict(causal=True, loss_db=6.0), _nrz(scale=0.4))
    ib = _run("lossy", dict(causal=True, loss_db=24.0), _nrz(scale=0.4))
    print(f"    max|lossy(loss_db=6) - lossy(loss_db=24)| = {np.max(np.abs(ia-ib)):.3e}  "
          f"bit-identical = {np.array_equal(ia, ib)}")

    print("\n[6] drift -- 'gain' and 'amplitude' are one branch")
    xd = _nrz(n_ui=256, n=1 << 13, seed=2)
    a = IMP.drift(xd, kind="gain", amount=0.2)
    b = IMP.drift(xd, kind="amplitude", amount=0.2)
    print(f"    max|drift('gain') - drift('amplitude')| = {np.max(np.abs(a-b)):.3e}  "
          f"bit-identical = {np.array_equal(a, b)}")

    print("\n[7] a timing parameter must be a monotone warp (crossings cannot be created)")
    xt = _nrz(n_ui=4096, n=4096 * SPB, seed=8)
    gt = _grid(n=len(xt))
    base = _n_zero_crossings(xt)
    print(f"    reference: {base} zero crossings in 4096 UI")
    for rj in (0.5, 1.0, 2.0, 4.0):
        ph = CDR.timing_source(len(xt), gt, rj_ps=rj, rng=np.random.default_rng(3))
        y = CDR.apply_phase(xt, ph)
        z = P.inject_jitter(xt, sigma_rj=rj * 1e-12 * FS, rng=np.random.default_rng(3))
        print(f"    rj = {rj:4.2f} ps ({rj*1e-12*FS:.4f} samples rms): "
              f"cdr.timing_source {_n_zero_crossings(y):6d} ({_n_zero_crossings(y)-base:+5d}), "
              f"physics.inject_jitter {_n_zero_crossings(z):6d} "
              f"({_n_zero_crossings(z)-base:+5d}); amplitude rms added "
              f"{np.std(y-xt):.5f} vs {np.std(z-xt):.5f}")

    print("\n[8] nonlinearity rise_fall_ratio -- a step at its own nominal value")
    spb = 64
    sym = np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0] * 40)
    xn = P.from_symbols(sym, n=len(sym) * spb, tr_frac=0.25, causal=True)
    y1 = P.nominal_nonlinearity(xn, compression=0.0, rise_fall_ratio=1.0)
    for r in (1.0, 1.0000001, 1.5, 2.0, 5.64, 16.0, 64.0):
        y = P.nominal_nonlinearity(xn, compression=0.0, rise_fall_ratio=r)
        tr, tf = _rise_fall_20_80(y, spb)
        ar, af = min(0.95, 0.4 * np.sqrt(r)), min(0.95, 0.4 / np.sqrt(r))
        print(f"    ask ratio={r!r:<12} max|y-y(1.0)| {np.max(np.abs(y-y1))/np.ptp(xn):7.5f} of "
              f"span; 20-80 % tr {tr:6.3f} tf {tf:6.3f} -> tf/tr {tf/tr:7.4f}; internal "
              f"a_r/a_f {ar/af:8.4f}"
              f"{'  <- a_r CLAMPED at 0.95' if 0.4*np.sqrt(r) > 0.95 else ''}")

    print("\n[9] scope_bandwidth order -- read on one kind of three")
    xo = np.random.default_rng(0).standard_normal(1 << 14)
    go = _grid(n=1 << 14)
    for kind in ("bessel", "gaussian", "brickwall"):
        b0 = INST.scope_bandwidth(xo, go, 32e9, kind=kind, order=1)
        d = [float(np.max(np.abs(INST.scope_bandwidth(xo, go, 32e9, kind=kind, order=o) - b0)))
             for o in (2, 4, 8)]
        print(f"    kind={kind:10s} max|y(order) - y(order=1)| for order 2/4/8: "
              f"{d[0]:.4f} / {d[1]:.4f} / {d[2]:.4f}")

    print("\n[10] CLEAN -- parameters measured against a closed form and found honest")
    for db in (1.0, 3.0, 6.0, 9.0):
        s = np.repeat(np.array([-1.0] * 8 + [1.0] * 8 + [-1.0] * 8 + [1.0] * 8), SPB)
        y = P.tx_ffe(s, P.de_emphasis_taps(db), SPB, pre=0)
        got = 20 * np.log10(abs(y[8*SPB+SPB//2] / y[8*SPB+7*SPB+SPB//2]))
        print(f"    de_emphasis db      ask {db:6.2f} dB   -> {got:9.4f} dB")
    for ldb, at in ((3.0, 4.0), (6.0, 8.0), (20.0, 16.0)):
        H, _f = _H_at(lambda z: _run("lossy", dict(loss_db=ldb, loss_at_ghz=at, causal=True), z),
                      at * 1e9)
        print(f"    lossy loss_db       ask {ldb:6.2f} dB   -> {-20*np.log10(abs(H)):9.4f} dB "
              f"at {at:g} GHz")
    for c in (0.1e-12, 0.45e-12, 1.0e-12):
        fc = INST.rc_pole_hz(50.0, c)
        H, _f = _H_at(lambda z: _run("probe", dict(c_load_f=c, r_source=50.0), z), fc)
        print(f"    probe c_load_f      ask {c*1e12:5.2f} pF   -> pole {fc/1e9:7.3f} GHz, "
              f"|H| {abs(H):.5f} (0.70711), phase {np.degrees(np.angle(H)):+7.2f} deg (-45)")
    xs = _nrz(scale=0.4) + np.random.default_rng(9).normal(0, 0.004, N)
    for bits in (8, 10, 11, 12, 14):
        y = INST.store_record(xs, bits=bits)
        lsb = 2 * 1.05 * float(np.max(np.abs(xs))) / 2.0 ** bits
        print(f"    store bits          ask {bits:4d} bits -> err rms "
              f"{np.std(y-xs)/lsb:.5f} LSB (q/sqrt12 = 0.28868), "
              f"{len(np.unique(y)):5d} distinct codes")
    for d in (0.0, 0.2887, 0.5, 1.0):
        y = INST.store_record(xs, bits=11, dither_lsb=d, rng=np.random.default_rng(1))
        lsb = 2 * 1.05 * float(np.max(np.abs(xs))) / 2.0 ** 11
        r = float(np.std(y - xs)) / lsb
        print(f"    store dither_lsb    ask {d:6.4f} LSB -> {r:.5f} LSB "
              f"(closed form {np.sqrt(1/12+d*d):.5f}), {20*np.log10(r/np.sqrt(1/12)):+6.3f} dB "
              f"over q**2/12")
    for c in (0.01, 0.05, 0.2):
        xc = _nrz(n_ui=512, n=1 << 14, seed=6)
        y = _run("crosstalk", dict(coupling=c, kind="fext", aggressor=dict(n_ui=64, seed=7)),
                 xc, _grid(n=1 << 14))
        print(f"    crosstalk coupling  ask {c:6.3f}      -> "
              f"{np.max(np.abs(y-xc))/np.ptp(xc):.5f} of victim span")
    for sp in (0.001, 0.005, 0.02):
        ph = CDR.ssc_phase(int(4e9 / 32e3), 1e9, 32e3, sp, "down")
        print(f"    ssc spread          ask {sp:6.4f}      -> "
              f"{np.max(np.abs(np.diff(ph))):.6f} peak df/f")
    for er in (3.0, 10.0, 20.0):
        p = OPT.to_optical(_nrz(n_ui=256, n=1 << 13, seed=2), er_db=er, p_avg=2.0)
        print(f"    optical er_db       ask {er:6.2f} dB   -> "
              f"{10*np.log10(np.max(p)/np.min(p)):9.4f} dB, mean {np.mean(p):.6f}")
    print(f"\n{line}\n")


if __name__ == "__main__":
    _report()
    raise SystemExit(pytest.main([__file__, "-q", "--no-header", "-rxX"]))

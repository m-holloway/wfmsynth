"""wfmsynth.iq -- complex baseband: a source with known symbols, and the two mixers that move a
record between I/Q and a real passband voltage.

    g = Grid(fs=8e9, baud=25e6, n=1 << 16, domain="iq", fc_hz=1e9)
    s = (Signal(seed=1, grid=g).iq("qam16", prbs=15)      # complex symbols, RRC-shaped
         .upconvert()                                      # -> real passband at g.fc_hz
         .probe(bw_hz=3e9)                                 # every real-record op applies again
         .downconvert(fc_hz=1e9, lpf=dict(pass_hz=20e6, stop_hz=60e6)))
    x = s.waveform()                                       # complex128
    sent, t_s = s.transmitted()                            # the ground truth, and where it sits
    evm(demodulate(x, s.stored_grid(x)), sent)

THE CONVENTIONS, all of which a caller eventually needs and none of which is guessable:

  * passband = Re{ x * exp(j*(2*pi*fc*t + phase)) } = I*cos - Q*sin. The envelope's magnitude
    is the passband PEAK, so a unit-circle constellation makes a +/-1 carrier.
  * downconvert multiplies by 2*exp(-j*...) and low-passes, so it returns the SAME x: the 2 is
    the half that mixing sends to the image at 2*fc.
  * t = n / fs, sample 0 at t = 0, for the mixers and for the source alike. Neither mixer
    resamples: one grid carries the envelope and the carrier.
  * symbol k is centred at (k + 0.5) / baud -- the middle of its unit interval, where the rest
    of this library puts a symbol too.
  * a constellation has UNIT AVERAGE POWER, and the pulse has unit energy per symbol, so a
    matched filter sampled at the symbol centre returns the symbol itself.

WHAT IS STATED RATHER THAN ASSUMED. The pulse is a root-raised-cosine truncated to `span_ui`
symbols, and truncation is the only reason a matched filter does not return the symbols
exactly. The down-converter's low-pass is a Kaiser-windowed linear-phase FIR applied with its
delay removed, described by three numbers (`pass_hz`, `stop_hz`, `atten_db`), and
`lowpass_error` MEASURES what those three numbers bought on the taps actually used.

Mechanisms live here; which constellation, roll-off and symbol rate a given radio standard
names is the caller's to state.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

# name -> bits per symbol. Square Gray-mapped QAM, plus BPSK on the real axis.
CONSTELLATIONS = {"bpsk": 1, "qpsk": 2, "qam16": 4, "qam64": 6}
RRC_BETA = 0.35           # the default roll-off
RRC_SPAN_UI = 16          # the default pulse length, in symbols (+/- 8 about its centre)
LPF_ATTEN_DB = 80.0       # the default stopband of downconvert's low-pass
FOLD_DB = -40.0           # the default limit on envelope power that upconvert would fold


def bits_per_symbol(constellation):
    if constellation not in CONSTELLATIONS:
        raise ValueError(f"iq: unknown constellation {constellation!r} "
                         f"(use one of {sorted(CONSTELLATIONS)}, or pass symbols=[...])")
    return CONSTELLATIONS[constellation]


def _gray_levels(bits, m):
    """`m` bits per rail -> a level in {-(L-1), ..., -1, +1, ..., L-1}, L = 2**m, Gray-coded so
    neighbouring levels differ in one bit (00 01 11 10 -> -3 -1 +1 +3)."""
    gray = np.zeros(len(bits), dtype=np.int64)
    for j in range(m):
        gray = (gray << 1) | bits[:, j]
    idx = gray.copy()
    shift = gray >> 1
    while shift.any():                   # Gray -> binary
        idx ^= shift
        shift >>= 1
    return 2.0 * idx - (2 ** m - 1)


def map_bits(constellation, bits):
    """Bits -> complex symbols of unit average power.

    Square QAM takes the first half of each symbol's bits as the I level and the second half
    as the Q level, each Gray-coded; `qpsk` is the 4-point case, (+/-1 +/- 1j)/sqrt(2), a 1
    bit being the positive rail. `bpsk` is +/-1 on the real axis. A bit count that is not a
    whole number of symbols is refused rather than padded or truncated."""
    k = bits_per_symbol(constellation)
    b = np.asarray(bits).astype(np.int64).ravel()
    if b.size == 0 or b.size % k:
        raise ValueError(f"iq: {b.size} bits is not a whole number of {constellation} symbols "
                         f"({k} bits each). Pass a multiple of {k}; nothing is padded.")
    if ((b != 0) & (b != 1)).any():
        raise ValueError("iq: bits must be 0 or 1")
    b = b.reshape(-1, k)
    if constellation == "bpsk":
        return (2.0 * b[:, 0] - 1.0).astype(complex)
    m = k // 2
    L = 2 ** m
    return (_gray_levels(b[:, :m], m) + 1j * _gray_levels(b[:, m:], m)) \
        / np.sqrt(2.0 * (L * L - 1) / 3.0)


def as_complex(values):
    """A recipe's complex list -> a complex array. JSON has no complex number, so a recipe
    holds I/Q as ``[[re, im], ...]``; a flat list of reals is an I/Q record with Q = 0."""
    a = np.asarray(values)
    if np.iscomplexobj(a):
        return a.astype(complex).ravel()
    a = a.astype(float)
    if a.ndim == 2 and a.shape[1] == 2:
        return a[:, 0] + 1j * a[:, 1]
    if a.ndim == 1:
        return a.astype(complex)
    raise ValueError(f"iq: expected complex samples as [[re, im], ...] or a flat list, got an "
                     f"array of shape {a.shape}")


def as_pairs(values):
    """A complex array -> the ``[[re, im], ...]`` a recipe holds."""
    a = np.asarray(values, complex).ravel()
    return [[float(v.real), float(v.imag)] for v in a]


# ------------------------------------------------------------------ the pulse
def rrc(t_ui, beta=RRC_BETA):
    """The root-raised-cosine pulse at `t_ui` symbols from its centre, with unit energy per
    symbol (the integral of h**2 over t_ui is 1). Its autocorrelation is a raised cosine, which
    is zero at every other symbol centre: that is the whole reason for the shape."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"iq: the roll-off beta must be in [0, 1], got {beta!r}")
    t = np.asarray(t_ui, float)
    out = np.empty_like(t)
    at0 = np.abs(t) < 1e-12
    den = 1.0 - (4.0 * beta * t) ** 2
    sing = (np.abs(den) < 1e-9) & ~at0
    reg = ~(at0 | sing)
    tr = t[reg]
    out[reg] = ((np.sin(np.pi * tr * (1 - beta)) + 4 * beta * tr * np.cos(np.pi * tr * (1 + beta)))
                / (np.pi * tr * den[reg]))
    out[at0] = 1.0 - beta + 4.0 * beta / np.pi
    if sing.any():                      # t = +/- 1/(4*beta): the closed-form limit
        out[sing] = (beta / np.sqrt(2.0)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                             + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    return out


def _rate(grid, who):
    fs = getattr(grid, "fs", None) if grid is not None else None
    baud = getattr(grid, "baud", None) if grid is not None else None
    if fs is None or baud is None:
        raise ValueError(f"{who}: needs the sample rate and the symbol rate -- "
                         f"Grid(fs=..., baud=..., n=..., domain='iq')")
    return float(fs), float(baud)


def symbols_in_record(grid, n=None):
    """How many whole symbols fit in the record."""
    fs, baud = _rate(grid, "iq")
    return int(np.floor((grid.n if n is None else n) * baud / fs + 1e-9))


def symbol_times(n_sym, grid):
    """The centre of each symbol, in seconds from sample 0."""
    _, baud = _rate(grid, "iq")
    return (np.arange(n_sym) + 0.5) / baud


def _taps(n, fs, baud, span_ui, delay_s=0.0):
    """For each sample, the symbols whose pulse reaches it. Yields ``(k, d, near)``: a symbol
    index per sample, that sample's distance from the symbol's centre in symbols, and whether
    the truncated pulse reaches that far."""
    u = (np.arange(n) / fs - delay_s) * baud - 0.5      # position, in symbols past symbol 0
    k0 = np.floor(u).astype(np.int64)
    half = 0.5 * span_ui
    for j in range(-int(np.ceil(half)), int(np.ceil(half)) + 2):
        k = k0 + j
        d = u - k
        yield k, d, np.abs(d) <= half


def _check_pulse(pulse, beta, span_ui, fs, baud, who):
    if pulse != "rrc":
        raise ValueError(f"{who}: unknown pulse {pulse!r}. 'rrc' (root-raised-cosine) is the "
                         f"only shape; beta=0 is a sinc and beta=1 the widest.")
    if not span_ui >= 2:
        raise ValueError(f"{who}: span_ui={span_ui!r} truncates the pulse to less than one "
                         f"symbol either side of its centre; use span_ui >= 2")
    if fs <= (1.0 + beta) * baud:
        raise ValueError(
            f"{who}: {fs / baud:.4g} samples per symbol cannot hold a root-raised-cosine of "
            f"roll-off {beta:g}, which occupies +/-{0.5 * (1 + beta):.4g} of the symbol rate: "
            f"it would alias. Use Grid(fs > {(1 + beta) * baud:.6g}).")


def shape(symbols, grid, n=None, pulse="rrc", beta=RRC_BETA, span_ui=RRC_SPAN_UI, amp=1.0):
    """Complex symbols -> a complex baseband record on `grid`: ``amp * sum_k a_k h(t - t_k)``
    with `h` the truncated root-raised-cosine and ``t_k = (k + 0.5) / baud``.

    Samples-per-symbol need not be an integer; the pulse is evaluated where the samples are.
    Symbols that do not fit in the record are refused rather than dropped. Fewer symbols than
    fit leave the rest of the record at zero -- a burst, which is what was asked for."""
    fs, baud = _rate(grid, "iq")
    n = int(grid.n if n is None else n)
    _check_pulse(pulse, beta, span_ui, fs, baud, "iq")
    a = np.asarray(symbols, complex).ravel()
    fit = symbols_in_record(grid, n)
    if a.size > fit:
        raise ValueError(f"iq: {a.size} symbols do not fit in a record of {n} samples at "
                         f"{fs / baud:.4g} samples per symbol ({fit} whole symbols). Lengthen "
                         f"the record or pass fewer; nothing is truncated.")
    x = np.zeros(n, complex)
    for k, d, near in _taps(n, fs, baud, span_ui):
        ok = near & (k >= 0) & (k < a.size)
        x[ok] += a[k[ok]] * rrc(d[ok], beta)
    return amp * x


def demodulate(x, grid, n_sym=None, pulse="rrc", beta=RRC_BETA, span_ui=RRC_SPAN_UI,
               delay_s=0.0):
    """Matched-filter the record and sample at each symbol centre -> the received symbols.

    `grid` must describe `x` (use `Signal.stored_grid(x)` after anything that resamples).
    `delay_s` is how late the record is against `symbol_times` -- a channel's or an
    instrument's group delay; nothing here estimates it. No gain or phase is fitted either:
    what comes back is what the record holds, so `evm` against the sent symbols counts a
    gain error as error."""
    fs, baud = _rate(grid, "demodulate")
    x = np.asarray(x)
    if not np.iscomplexobj(x):
        raise ValueError("demodulate: the record is real, not complex baseband. "
                         "downconvert(fc_hz=..., lpf=...) a passband record first.")
    _check_pulse(pulse, beta, span_ui, fs, baud, "demodulate")
    n = len(x)
    n_sym = symbols_in_record(grid, n) if n_sym is None else int(n_sym)
    re, im = np.zeros(n_sym), np.zeros(n_sym)
    for k, d, near in _taps(n, fs, baud, span_ui, delay_s):
        ok = near & (k >= 0) & (k < n_sym)
        w = x[ok] * rrc(d[ok], beta)
        re += np.bincount(k[ok], weights=w.real, minlength=n_sym)
        im += np.bincount(k[ok], weights=w.imag, minlength=n_sym)
    return (re + 1j * im) * (baud / fs)


def complex_gain(received, sent):
    """The one complex number `g` that best maps sent onto received (least squares):
    ``received ~ g * sent``. Its magnitude is the path's gain and its angle the carrier phase
    the path added -- a front end with group delay tau rotates the constellation by
    ``-2*pi*fc*tau``, which is usually the first thing a raw EVM is measuring."""
    r, s = np.asarray(received, complex), np.asarray(sent, complex)
    if r.shape != s.shape:
        raise ValueError(f"{r.size} received symbols against {s.size} sent")
    return complex(np.vdot(s, r) / np.vdot(s, s))


def evm(received, sent, fit=False):
    """RMS error between two sets of SYMBOLS you already hold, as a FRACTION of the sent
    symbols' RMS: the arithmetic `rf.evm` ends in.

    THE EVM OF A RECORD IS `wfmsynth.rf.evm(x, grid, sent)`: it runs the receiver, reports
    percent and dB beside the fraction, and names every correction it makes. Use this one
    where there is no receiver to run -- symbols from your own demodulator, or a record that
    was already matched-filtered.

    By default nothing is fitted, so a gain or a carrier-phase rotation counts in full.
    `fit=True` divides out `complex_gain` first (what `rf.evm(correct=("gain", "phase"))`
    removes, as one complex least-squares gain); timing is never fitted."""
    r, s = np.asarray(received, complex), np.asarray(sent, complex)
    if fit:
        r = r / complex_gain(r, s)
    elif r.shape != s.shape:
        raise ValueError(f"evm: {r.size} received symbols against {s.size} sent")
    return float(np.sqrt(np.mean(np.abs(r - s) ** 2) / np.mean(np.abs(s) ** 2)))


# ------------------------------------------------------------------ the stated low-pass
def lowpass_spec(lpf):
    """Normalise `lpf` to ``{"pass_hz", "stop_hz", "atten_db"}``. A bare number is the passband
    edge; the stopband then starts at twice it, 80 dB down."""
    if lpf is None:
        raise ValueError("downconvert needs its low-pass STATED: lpf=<passband edge in Hz>, or "
                         "lpf=dict(pass_hz=..., stop_hz=..., atten_db=...). The filter decides "
                         "what the I/Q record contains, so it has no silent default.")
    if isinstance(lpf, dict):
        extra = sorted(set(lpf) - {"pass_hz", "stop_hz", "atten_db"})
        if extra or "pass_hz" not in lpf:
            raise ValueError(f"downconvert: lpf takes pass_hz (required), stop_hz and atten_db"
                             + (f"; it does not take {extra}" if extra else ""))
        spec = dict(lpf)
    elif isinstance(lpf, (int, float, np.integer, np.floating)) and not isinstance(lpf, bool):
        spec = {"pass_hz": float(lpf)}
    else:
        raise ValueError(f"downconvert: lpf is a passband edge in Hz or a dict of pass_hz, "
                         f"stop_hz, atten_db; got {lpf!r}")
    spec.setdefault("stop_hz", 2.0 * spec["pass_hz"])
    spec.setdefault("atten_db", LPF_ATTEN_DB)
    p, s, a = (float(spec[k]) for k in ("pass_hz", "stop_hz", "atten_db"))
    if not 0 < p < s:
        raise ValueError(f"downconvert: lpf needs 0 < pass_hz < stop_hz, got {p:g} and {s:g}")
    if not a > 21.0:
        raise ValueError(f"downconvert: lpf atten_db={a:g} is below what any window gives "
                         f"(21 dB); state a real stopband")
    return {"pass_hz": p, "stop_hz": s, "atten_db": a}


def lowpass(lpf, fs):
    """The taps of the stated low-pass at sample rate `fs`: a Kaiser-windowed sinc, odd length,
    linear phase, unit gain at DC, its cutoff midway between `pass_hz` and `stop_hz`. The
    length and the window come from Kaiser's design formulas; `lowpass_error` measures what
    they delivered."""
    spec = lowpass_spec(lpf)
    p, s, a = spec["pass_hz"], spec["stop_hz"], spec["atten_db"]
    if s > 0.5 * fs:
        raise ValueError(f"downconvert: lpf stop_hz={s:g} is above Nyquist ({0.5 * fs:g} Hz) "
                         f"at fs={fs:g}")
    beta = 0.1102 * (a - 8.7) if a > 50 else 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    numtaps = int(np.ceil((a - 7.95) / (2.285 * 2 * np.pi * (s - p) / fs))) + 1
    numtaps += 1 - numtaps % 2
    m = np.arange(numtaps) - (numtaps - 1) // 2
    fc = (p + s) / fs                                    # cutoff, in cycles/sample, times two
    h = fc * np.sinc(fc * m) * np.kaiser(numtaps, beta)
    return h / h.sum()


def lowpass_error(lpf, fs):
    """``(pass_err, stop_db)`` MEASURED on the taps: the largest ``|H - 1|`` for |f| <= pass_hz
    and the largest ``|H|`` in dB for |f| >= stop_hz. This is the passband error a
    downconvert(upconvert(x)) round trip is held to."""
    spec = lowpass_spec(lpf)
    h = lowpass(spec, fs)
    nfft = 1 << int(np.ceil(np.log2(len(h) * 32)))
    H = np.abs(np.fft.rfft(h, nfft))                     # zero phase about the centre tap
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    return (float(np.abs(H[f <= spec["pass_hz"]] - 1.0).max()),
            float(20 * np.log10(H[f >= spec["stop_hz"]].max())))


# ------------------------------------------------------------------ the mixers
def _carrier_phase(n, fc_hz, fs, phase_rad):
    return 2.0 * np.pi * fc_hz * (np.arange(n) / fs) + phase_rad


def outside_db(x, room_hz, fs):
    """How much of an envelope's power lies at ``|f| >= room_hz``, in dB relative to its total.

    Measured through a Hann window: the record's two ends are where the capture stops, not
    signal, and a rectangular window would report their step as wideband power (-36 dB for a
    single off-bin tone, which has no power there at all)."""
    x = np.asarray(x, complex)
    X = np.abs(np.fft.fft(x * np.hanning(len(x)))) ** 2
    tot = float(X.sum())
    if tot == 0.0:
        return -np.inf
    f = np.fft.fftfreq(len(X), 1.0 / fs)
    out = float(X[np.abs(f) >= room_hz].sum())
    return 10 * np.log10(out / tot) if out > 0 else -np.inf


def folded_db(x, fc_hz, fs):
    """How much of an envelope's power lies where up-conversion to `fc_hz` would fold it, in dB
    relative to its total: outside ``|f| < min(fc_hz, fs/2 - fc_hz)``. Content beyond +that
    passes Nyquist; content beyond -that crosses DC and lands on its own mirror image.
    `outside_db` is the measurement."""
    return outside_db(x, min(fc_hz, 0.5 * fs - fc_hz), fs)


def upconvert(x, fc_hz, fs, phase_rad=0.0, fold_db=FOLD_DB):
    """Complex baseband -> a real passband record on the SAME grid:
    ``Re{x * exp(j*(2*pi*fc_hz*t + phase_rad))}``.

    The grid has to be fast enough for the carrier AND for the envelope riding on it. A carrier
    at or above Nyquist is refused outright. An envelope wide enough to fold -- past Nyquist on
    the high side, or through DC onto its own image on the low side -- is refused when more
    than `fold_db` of its power would (`folded_db` is the measurement). Neither is resampled
    away: the fix is a faster grid, and the message says how fast."""
    x = np.asarray(x)
    if not 0.0 < fc_hz < 0.5 * fs:
        raise ValueError(
            f"upconvert: the grid is not fast enough for the carrier -- fc_hz={fc_hz:g} needs "
            f"fs > {2 * fc_hz:g} just to exist, and this grid is fs={fs:g}. Build the chain on "
            f"Grid(fs=...) at 8 or more samples per carrier cycle ({8 * fc_hz:g}); upconvert "
            f"does not resample.")
    got = folded_db(x, fc_hz, fs)
    if got > fold_db:
        room = min(fc_hz, 0.5 * fs - fc_hz)
        raise ValueError(
            f"upconvert: {got:.1f} dB of the envelope's power lies outside +/-{room:g} Hz, "
            f"where a carrier at fc_hz={fc_hz:g} on fs={fs:g} folds it (past Nyquist above, "
            f"through DC below); the limit is fold_db={fold_db:g}. Use a faster grid, move "
            f"fc_hz away from 0 and fs/2, or band-limit the envelope. Pass fold_db= to accept "
            f"a stated amount.")
    return (x * np.exp(1j * _carrier_phase(len(x), fc_hz, fs, phase_rad))).real


def downconvert(y, fc_hz, fs, lpf, fs_out=None, phase_rad=0.0):
    """A real passband record -> complex baseband about `fc_hz`:
    ``lowpass{ 2 * y * exp(-j*(2*pi*fc_hz*t + phase_rad)) }``, optionally decimated to
    `fs_out`.

    The low-pass is the stated one (`lowpass`), applied with its delay removed, so the output
    is time-aligned with the input. It has to reject the mixing image, which sits
    ``min(2*fc_hz, fs - 2*fc_hz)`` from DC; a filter whose stopband starts later than that
    image's near edge is refused. The first and last half-filter-length of the record see the
    record's edge (the line is taken as quiet outside it) and are not within the stated error.

    `fs_out` keeps every D-th sample, D = fs / fs_out a whole number, and the same filter is
    the anti-alias filter: a stopband starting above the new Nyquist is refused."""
    y = np.asarray(y)
    if np.iscomplexobj(y):
        raise ValueError("downconvert: the record is already complex")
    spec = lowpass_spec(lpf)
    if not 0.0 < fc_hz < 0.5 * fs:
        raise ValueError(f"downconvert: fc_hz={fc_hz:g} is not below Nyquist ({0.5 * fs:g} Hz) "
                         f"on this record's fs={fs:g}; the carrier is not in the record")
    image = min(2 * fc_hz, fs - 2 * fc_hz)
    if spec["stop_hz"] > image - spec["pass_hz"]:
        raise ValueError(
            f"downconvert: mixing puts an image {image:g} Hz from DC, whose near edge is at "
            f"{image - spec['pass_hz']:g} Hz; lpf stop_hz={spec['stop_hz']:g} starts later "
            f"than that, so the image would come through. Lower stop_hz, or sample faster.")
    step = 1
    if fs_out is not None:
        step = int(round(fs / fs_out))
        if step < 1 or abs(step * fs_out - fs) > 1e-9 * fs:
            raise ValueError(f"downconvert: fs_out={fs_out:g} is not fs={fs:g} divided by a "
                             f"whole number. Decimation keeps every D-th sample; nothing is "
                             f"interpolated.")
        if spec["stop_hz"] > 0.5 * fs_out:
            raise ValueError(f"downconvert: lpf stop_hz={spec['stop_hz']:g} is above the "
                             f"output Nyquist ({0.5 * fs_out:g} Hz), so decimating to "
                             f"fs_out={fs_out:g} would alias. Lower stop_hz or raise fs_out.")
    h = lowpass(spec, fs)
    if len(h) > len(y):
        raise ValueError(f"downconvert: the stated low-pass needs {len(h)} taps at fs={fs:g} "
                         f"and the record has {len(y)} samples. Widen the transition "
                         f"(stop_hz - pass_hz), lower atten_db, or lengthen the record.")
    z = 2.0 * y * np.exp(-1j * _carrier_phase(len(y), fc_hz, fs, phase_rad))
    return fftconvolve(z, h, mode="same")[::step]

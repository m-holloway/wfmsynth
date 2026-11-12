"""wfmsynth.rf -- what a real transmitter, oscillator and channel do to an I/Q record, one op
per effect, each stated so that it can be measured back off the record.

    g = Grid(fs=400e6, baud=25e6, n=1 << 16, domain="iq", fc_hz=2.4e9)
    s = (Signal(seed=1, grid=g).iq("qam16", prbs=15)
         .dac(bits=10, headroom_pct=5)                    # the converter's steps and range
         .iq_imbalance(gain_db=0.3, phase_deg=2.0)        # the modulator's two rails
         .iq_offset(i=0.01)                               # carrier leakage
         .phase_noise(profile=[(1e4, -80), (1e6, -120)])  # the oscillator
         .freq_offset(f_hz=12e3)                          # the two ends' oscillators disagree
         .awgn(es_n0_db=20))                              # the channel
    s.rf_report()      # what each of those came to: image rejection, N0, crest factor ...
    s.lo_phase()       # the oscillator's phase at every sample, in rad -- the label

Every op here acts on COMPLEX BASEBAND (domain 'iq') and is refused by name on a real record.

THE CONVENTIONS, none of which is guessable:

  iq_imbalance   The Q rail has gain `gain_db` and its carrier is `phase_deg` away from
                 quadrature, both RELATIVE TO THE I RAIL, at the modulator:
                     passband = I*cos(wt) - g*Q*sin(wt + phi)
                     y = I + j*g*exp(j*phi)*Q = mu*x + nu*conj(x)
                 with mu = (1 + g*exp(j*phi))/2, nu = (1 - g*exp(j*phi))/2. The conj(x) term IS
                 the image: a tone at +f gains one at -f, |mu/nu|**2 below it
                 (`image_rejection_db`). There is no separate image op -- an image is what
                 imbalance produces; `imbalance_for(rejection_db)` gives the gain that makes one.
  iq_offset      y = x + (i + j*q), in the record's own units: a DC term on each rail, which
                 at RF is the carrier leaking through.
  freq_offset    y = x * exp(j*2*pi*(f_hz*t + drift_hz_per_s*t**2/2)), t = n/fs from sample 0.
                 Positive f_hz moves the spectrum UP.
  phase_noise    y = x * exp(j*theta(t)), theta a zero-mean Gaussian process whose one-sided
                 phase PSD is 2*L(f): `profile` is L(f) in dBc/Hz at offsets in Hz (L is
                 DEFINED as half the phase PSD, IEEE Std 1139), joined by straight lines on
                 log-frequency / dB axes. NOTHING is drawn outside the stated offsets, so the
                 integrated phase is the profile's own (`phase_rms_rad`).
  awgn           y = x + n, n circular complex Gaussian, white across the whole sampled band.
                 Its level is one of three ratios against a REFERENCE POWER:
                     snr_db    P_ref / (N0 * bw_hz)    -- `bw_hz` is required: an SNR without
                                                          a bandwidth is not a number
                     es_n0_db  (P_ref / baud) / N0
                     ebn0_db   (P_ref / baud / bits_per_symbol) / N0
                 P_ref is `ref_power=` if stated, else the mean of |x|**2 over the record as
                 it reaches this op. The total noise power added is N0*fs, split equally
                 between the rails.
  spur           y = x + A*exp(j*(2*pi*f_hz*t + phase_rad)), A**2 = P_ref * 10**(dbc/10):
                 one tone `f_hz` from the centre frequency, `dbc` relative to the same
                 reference power.
  dac            each rail rounded onto 2**bits codes spanning -FS .. +FS - 1 LSB and saturated
                 there. FS is `full_scale=`, or is set so that the record's largest rail value
                 times (1 + headroom_pct/100) lands ON the last code. A sample more than half
                 a step beyond the codes is clipped, and a record with any is refused unless
                 clip="hard" says clipping is the point. The record keeps its units.

An oscillator impairment on the ENVELOPE is that impairment on the up-converting mixer's
oscillator, exactly. After `downconvert` it stands for the receiver's oscillator as long as
its phase moves slowly against the down-converter's low-pass.

WHAT IS REUSED. `awgn` draws each rail through `rx.input_noise` (so the realised noise power
is the stated one, not the stated one plus a chi-square wobble); `dac` rounds each rail with
the `stages` law "quantize"; `freq_offset` measures what it would wrap past Nyquist with the
measurement `upconvert` uses (`iq.outside_db`). `timing(phase_noise=...)` is NOT this op: it
moves a baseband waveform in TIME by a one-slope process scaled to an rms, where an
oscillator's phase noise ROTATES an envelope by a process with a stated spectrum.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import iq as IQ
from . import rx as RX
from . import stages as ST
from .streams import Streams

OPS = ("iq_imbalance", "iq_offset", "freq_offset", "phase_noise", "awgn", "spur", "dac")
# the ops that read the record's sample rate, which after a decimating `downconvert` is not
# the synthesis grid's
RATE_AWARE = ("freq_offset", "phase_noise", "awgn", "spur")
AWGN_KEYS = ("snr_db", "es_n0_db", "ebn0_db", "bw_hz", "bits_per_symbol", "ref_power")
DAC_KEYS = ("bits", "headroom_pct", "full_scale", "clip")


def _envelope(x, who):
    if x is None:
        raise ValueError(f"{who}: there is no record yet for it to act on -- start the chain "
                         f"with iq(...) or capture(...).")
    x = np.asarray(x)
    if not np.iscomplexobj(x):
        raise ValueError(f"{who}: the record is real, not complex baseband. {who} acts on I/Q "
                         f"-- downconvert(fc_hz=..., lpf=...) a passband record first.")
    return x


def _number(v, who, name):
    if isinstance(v, (bool, str)) or not np.isscalar(v) or not np.isfinite(v):
        raise ValueError(f"{who}: {name} is one finite number, got {v!r}")
    return float(v)


# ------------------------------------------------------------------------- iq_imbalance
def imbalance_coefficients(gain_db=0.0, phase_deg=0.0):
    """``(mu, nu)`` of ``y = mu*x + nu*conj(x)`` -- see the module docstring for the rail
    model they come from."""
    q = 10.0 ** (gain_db / 20.0) * np.exp(1j * np.deg2rad(phase_deg))
    return complex((1.0 + q) / 2.0), complex((1.0 - q) / 2.0)


def image_rejection_db(gain_db=0.0, phase_deg=0.0):
    """How far below a tone its image sits: ``|mu/nu|**2`` in dB. Infinite for balanced rails.
    1 dB of gain error alone is 24.8 dB; 1 degree of phase error alone is 41.2 dB."""
    mu, nu = imbalance_coefficients(gain_db, phase_deg)
    return float("inf") if nu == 0 else float(20.0 * np.log10(abs(mu) / abs(nu)))


def imbalance_for(rejection_db):
    """The `iq_imbalance` parameters that put the image `rejection_db` below the signal, as a
    pure GAIN error: ``.iq_imbalance(**rf.imbalance_for(30))``. Many gain/phase pairs give the
    same rejection; this is the one with the rails still in quadrature."""
    r = 10.0 ** (_number(rejection_db, "imbalance_for", "rejection_db") / 20.0)
    if not r > 1.0:
        raise ValueError(f"imbalance_for: rejection_db={rejection_db!r} must be positive -- an "
                         f"image at or above the signal is not an imbalance of two rails.")
    return {"gain_db": float(20.0 * np.log10((r + 1.0) / (r - 1.0))), "phase_deg": 0.0}


def iq_imbalance(x, gain_db=None, phase_deg=None):
    if gain_db is None and phase_deg is None:
        raise ValueError("iq_imbalance needs gain_db=, phase_deg= or both: the Q rail's gain "
                         "and quadrature error relative to the I rail. For a stated image "
                         "level use iq_imbalance(**wfmsynth.rf.imbalance_for(rejection_db)).")
    g = 0.0 if gain_db is None else _number(gain_db, "iq_imbalance", "gain_db")
    ph = 0.0 if phase_deg is None else _number(phase_deg, "iq_imbalance", "phase_deg")
    x = _envelope(x, "iq_imbalance")
    mu, nu = imbalance_coefficients(g, ph)
    return mu * x + nu * np.conj(x)


# ---------------------------------------------------------------------------- iq_offset
def iq_offset(x, i=None, q=None):
    if i is None and q is None:
        raise ValueError("iq_offset needs i=, q= or both: the DC on each rail, in the record's "
                         "units. For a leak stated in dBc use spur(f_hz=0, dbc=...).")
    dc = complex(0.0 if i is None else _number(i, "iq_offset", "i"),
                 0.0 if q is None else _number(q, "iq_offset", "q"))
    return _envelope(x, "iq_offset") + dc


# ---------------------------------------------------------------------- the oscillator
def lo_ramp(n, fs, f_hz=0.0, drift_hz_per_s=0.0):
    """The phase a frequency offset adds, in rad at each of `n` samples:
    ``2*pi*(f_hz*t + drift_hz_per_s*t**2/2)``."""
    t = np.arange(int(n)) / float(fs)
    return 2.0 * np.pi * (f_hz * t + 0.5 * drift_hz_per_s * t * t)


def freq_offset(x, fs, f_hz=None, drift_hz_per_s=None, fold_db=IQ.FOLD_DB):
    if f_hz is None and drift_hz_per_s is None:
        raise ValueError("freq_offset needs f_hz= (the offset at t = 0), drift_hz_per_s= or "
                         "both")
    f0 = 0.0 if f_hz is None else _number(f_hz, "freq_offset", "f_hz")
    d = 0.0 if drift_hz_per_s is None else _number(drift_hz_per_s, "freq_offset",
                                                   "drift_hz_per_s")
    x = _envelope(x, "freq_offset")
    f_end = f0 + d * (len(x) - 1) / fs
    reach = max(abs(f0), abs(f_end))
    if reach >= 0.5 * fs:
        raise ValueError(
            f"freq_offset: the offset reaches {reach:g} Hz (f_hz={f0:g}, {f_end:g} at the end "
            f"of the record), at or beyond Nyquist {0.5 * fs:g} of this fs={fs:g} record -- it "
            f"would wrap around the band. Use a faster grid; nothing is resampled.")
    got = IQ.outside_db(x, 0.5 * fs - reach, fs)
    if got > fold_db:
        raise ValueError(
            f"freq_offset: {got:.1f} dB of the record's power lies within {reach:g} Hz of "
            f"Nyquist, where this offset would wrap it to the other side of the band; the "
            f"limit is fold_db={fold_db:g}. Use a faster grid or band-limit the record first "
            f"(filter). A record that is white to Nyquist -- one after awgn -- always has "
            f"some of its noise there; fold_db= accepts a stated amount.")
    return x * np.exp(1j * lo_ramp(len(x), fs, f0, d))


def _profile(profile):
    try:
        a = np.asarray(profile, float)
    except (TypeError, ValueError):
        a = np.zeros(0)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.all(np.isfinite(a)):
        raise ValueError(
            f"phase_noise: profile is two or more (f_offset_hz, dbc_hz) points, e.g. "
            f"[(1e4, -80), (1e5, -100), (1e6, -120)], got {profile!r}. One point has no slope "
            f"and no span; a flat floor is two points at the same level.")
    f, db = a[:, 0], a[:, 1]
    if f[0] <= 0 or np.any(np.diff(f) <= 0):
        raise ValueError(f"phase_noise: the profile's offsets must be positive and increasing "
                         f"(they sit on a log axis), got {f.tolist()}")
    return f, db


def profile_dbc_hz(f_hz, profile):
    """L(f) in dBc/Hz at `f_hz`: the profile's points joined on log-frequency / dB axes, and
    -inf outside the stated offsets (where the op draws nothing)."""
    f, db = _profile(profile)
    fq = np.atleast_1d(np.asarray(f_hz, float))
    out = np.full(fq.shape, -np.inf)
    inside = (fq >= f[0]) & (fq <= f[-1])
    out[inside] = np.interp(np.log10(fq[inside]), np.log10(f), db)
    return out


def phase_rms_rad(profile):
    """The rms phase the profile integrates to, in rad: ``sqrt(integral of 2*L(f) df)`` over
    the stated offsets, in closed form for each straight segment."""
    f, db = _profile(profile)
    total = 0.0
    for f1, f2, d1, d2 in zip(f[:-1], f[1:], db[:-1], db[1:]):
        a = (d2 - d1) / (10.0 * np.log10(f2 / f1))          # L = L1 * (f/f1)**a
        l1 = 10.0 ** (d1 / 10.0)
        total += l1 * f1 * (np.log(f2 / f1) if abs(a + 1.0) < 1e-12
                            else ((f2 / f1) ** (a + 1.0) - 1.0) / (a + 1.0))
    return float(np.sqrt(2.0 * total))


def lo_noise(n, fs, profile, rng):
    """An oscillator's phase in rad at each of `n` samples: Gaussian, zero-mean, one-sided PSD
    ``2*L(f)`` at the profile's offsets and nothing elsewhere.

    White Gaussian samples shaped in the frequency domain, so the spectrum is the stated one
    at every bin rather than a filter's approximation of it; the process is periodic in the
    record. It is NOT rescaled to the profile's rms: one record's rms phase is an estimate of
    it (`Signal.rf_report()` gives both)."""
    f, _ = _profile(profile)
    n = int(n)
    if f[0] < fs / n or f[-1] > 0.5 * fs:
        raise ValueError(
            f"phase_noise: the profile spans {f[0]:g} .. {f[-1]:g} Hz and this record resolves "
            f"{fs / n:g} .. {0.5 * fs:g} Hz ({n} samples at fs={fs:g}). An offset below 1/T "
            f"has no bin and one above Nyquist no room: start the profile at {fs / n:g} Hz or "
            f"above, end it at or below Nyquist, or lengthen the record. Nothing is clipped "
            f"to fit.")
    fb = np.fft.rfftfreq(n, 1.0 / fs)
    gain = np.zeros(len(fb))
    # unit white noise has a one-sided PSD of 2/fs, and the phase wants 2*L(f)
    gain[1:] = np.sqrt(10.0 ** (profile_dbc_hz(fb[1:], profile) / 10.0) * fs)
    return np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * gain, n)


def phase_noise(x, fs, profile=None, rng=None):
    if profile is None:
        raise ValueError("phase_noise needs profile=[(f_offset_hz, dbc_hz), ...]: the "
                         "oscillator's L(f). There is no default oscillator.")
    x = _envelope(x, "phase_noise")
    return x * np.exp(1j * lo_noise(len(x), fs, profile, rng))


# ------------------------------------------------------------------------ additive terms
def _ref_power(x, ref_power, who):
    if ref_power is not None:
        p = _number(ref_power, who, "ref_power")
        if not p > 0:
            raise ValueError(f"{who}: ref_power is a power (a mean |x|**2) and must be "
                             f"positive, got {ref_power!r}")
        return p
    p = float(np.mean(np.abs(x) ** 2))
    if not p > 0:
        raise ValueError(f"{who}: the record is all zeros here, so there is no signal power to "
                         f"be relative to. State it: ref_power=<the signal's mean |x|**2>.")
    return p


def awgn_levels(x, fs, baud=None, snr_db=None, es_n0_db=None, ebn0_db=None, bw_hz=None,
                bits_per_symbol=None, ref_power=None):
    """``(ref_power, n0)``: the reference power an `awgn` request is relative to and the noise
    density it resolves to, in power per Hz. The noise added is ``n0 * fs`` in total."""
    asked = {"snr_db": snr_db, "es_n0_db": es_n0_db, "ebn0_db": ebn0_db}
    given = [k for k, v in asked.items() if v is not None]
    if len(given) != 1:
        raise ValueError(
            f"awgn takes exactly one of snr_db= (with bw_hz=), es_n0_db= or ebn0_db= (with "
            f"bits_per_symbol=), got {given or 'none'}. They are three statements of one "
            f"noise density, and two of them would have to agree.")
    kind = given[0]
    ratio = 10.0 ** (_number(asked[kind], "awgn", kind) / 10.0)
    if kind == "snr_db":
        if bw_hz is None:
            raise ValueError(
                f"awgn(snr_db=...) is ambiguous without bw_hz=, the bandwidth the noise is "
                f"counted in: the noise is white across this record's whole {fs:g} Hz band, "
                f"and its power in any narrower band is smaller in proportion. Pass "
                f"bw_hz={fs:g} for the SNR of the record as stored, bw_hz=<the signal's "
                f"bandwidth> for in-band SNR, or use es_n0_db=.")
        band = _number(bw_hz, "awgn", "bw_hz")
        if not 0 < band <= fs:
            raise ValueError(f"awgn: bw_hz={bw_hz!r} must be positive and no wider than the "
                             f"record's band, fs={fs:g}")
    else:
        if bw_hz is not None:
            raise ValueError(f"awgn: bw_hz= goes with snr_db=; {kind} is already a density "
                             f"(per symbol or per bit), so there is no bandwidth to state.")
        if not baud:
            raise ValueError(f"awgn({kind}=...) needs the symbol rate: Grid(baud=...). Without "
                             f"one, state the noise as snr_db= with bw_hz=.")
        band = float(baud)
    if kind == "ebn0_db":
        if bits_per_symbol is None:
            raise ValueError(
                "awgn(ebn0_db=...) needs bits_per_symbol= (1 bpsk, 2 qpsk, 4 qam16, 6 qam64; "
                "times the code rate if the bits are coded). Signal.awgn fills it in from an "
                "`iq` source's constellation; a recipe or a symbols= source has to state it.")
        k = _number(bits_per_symbol, "awgn", "bits_per_symbol")
        if not k > 0:
            raise ValueError(f"awgn: bits_per_symbol must be positive, got {bits_per_symbol!r}")
        band *= k
    elif bits_per_symbol is not None:
        raise ValueError(f"awgn: bits_per_symbol= goes with ebn0_db=; {kind} does not use it.")
    p = _ref_power(x, ref_power, "awgn")
    return p, p / (band * ratio)


def awgn(x, fs, baud=None, rng=None, **request):
    x = _envelope(x, "awgn")
    _, n0 = awgn_levels(x, fs, baud, **request)
    rail = float(np.sqrt(0.5 * n0 * fs))
    return (RX.input_noise(x.real, rms=rail, rng=rng)
            + 1j * RX.input_noise(x.imag, rms=rail, rng=rng))


def spur(x, fs, f_hz=None, dbc=None, phase_rad=0.0, ref_power=None):
    if f_hz is None or dbc is None:
        raise ValueError("spur needs f_hz= (its offset from the centre frequency; negative is "
                         "below) and dbc= (its power relative to the signal's)")
    f = _number(f_hz, "spur", "f_hz")
    x = _envelope(x, "spur")
    if abs(f) >= 0.5 * fs:
        raise ValueError(f"spur: f_hz={f:g} is at or beyond Nyquist {0.5 * fs:g} of this "
                         f"fs={fs:g} record, where it would alias to another frequency. Use a "
                         f"faster grid.")
    amp = np.sqrt(_ref_power(x, ref_power, "spur")
                  * 10.0 ** (_number(dbc, "spur", "dbc") / 10.0))
    return x + amp * np.exp(1j * (lo_ramp(len(x), fs, f)
                                  + _number(phase_rad, "spur", "phase_rad")))


# ---------------------------------------------------------------------------------- dac
def dac_scale(x, bits=None, headroom_pct=None, full_scale=None, clip=None):
    """``(full_scale, peak, n_over)``: the converter's range, the record's largest rail value
    and how many rail samples are CLIPPED -- nearer to a code that does not exist than to
    one that does (more than half a step past the first or last code)."""
    if bits is None:
        raise ValueError("dac needs bits=, the converter's resolution")
    if isinstance(bits, bool) or int(bits) != bits or not 2 <= bits <= 32:
        raise ValueError(f"dac: bits is a whole number from 2 to 32, got {bits!r}")
    top = 1.0 - 2.0 / 2 ** int(bits)              # the last code, as a fraction of full scale
    if clip not in (None, "hard"):
        raise ValueError(f"dac: clip is 'hard' (the rails saturate at full scale) or omitted "
                         f"(a record beyond full scale is refused), got {clip!r}")
    if headroom_pct is not None and full_scale is not None:
        raise ValueError("dac takes full_scale= (the range, stated) or headroom_pct= (the "
                         "range, as a margin over this record's peak), not both")
    x = _envelope(x, "dac")
    peak = float(max(np.abs(x.real).max(), np.abs(x.imag).max()))
    if full_scale is not None:
        fsc = _number(full_scale, "dac", "full_scale")
        if not fsc > 0:
            raise ValueError(f"dac: full_scale must be positive, got {full_scale!r}")
    else:
        h = 0.0 if headroom_pct is None else _number(headroom_pct, "dac", "headroom_pct")
        if not h > -100.0:
            raise ValueError(f"dac: headroom_pct={headroom_pct!r} leaves no range at all "
                             f"(-100 % is a full scale of zero)")
        if not peak > 0:
            raise ValueError("dac: the record is all zeros here, so a full scale cannot be "
                             "taken from its peak. State it: full_scale=...")
        fsc = peak * (1.0 + h / 100.0) / top       # the peak, plus its margin, ON the last code
    half = fsc / 2 ** int(bits)                    # half a step
    rails = np.concatenate([x.real, x.imag])
    over = int(np.count_nonzero((rails > fsc * top + half * (1 + 1e-9))
                                | (rails < -fsc - half * (1 + 1e-9))))
    if over and clip != "hard":
        raise ValueError(
            f"dac: {over} rail samples lie beyond the converter's codes (-{fsc:g} .. "
            f"{fsc * top:g}; the record's largest rail value is {peak:g}) and would be "
            f"clipped. Pass clip='hard' if clipping is the effect wanted, or give the record "
            f"room: a larger full_scale=, or headroom_pct >= 0.")
    return fsc, peak, over


def dac(x, **p):
    fsc, _, _ = dac_scale(x, **p)
    return (ST.apply_law(x.real, "quantize", bits=p["bits"], lo=-fsc, hi=fsc)
            + 1j * ST.apply_law(x.imag, "quantize", bits=p["bits"], lo=-fsc, hi=fsc))


# ----------------------------------------------------------------------- what it came to
def _db(ratio):
    return float(10.0 * np.log10(ratio)) if ratio > 0 else float("-inf")


def _stated(op, keys):
    return {k: op[k] for k in keys if k in op}


def _describe(op, x, grid, rng):
    """One op's derived numbers. `x` is the record as it reaches the op, `grid` what it is on."""
    name, fs = op["op"], float(grid.fs)
    mean = float(np.mean(np.abs(x) ** 2))
    if name == "iq_imbalance":
        return {"image_rejection_db": image_rejection_db(op.get("gain_db") or 0.0,
                                                         op.get("phase_deg") or 0.0)}
    if name == "iq_offset":
        dc = complex(op.get("i") or 0.0, op.get("q") or 0.0)
        return {"leak_dbc": _db(abs(dc) ** 2 / mean) if mean else float("inf")}
    if name == "freq_offset":
        f0, d = op.get("f_hz") or 0.0, op.get("drift_hz_per_s") or 0.0
        return {"f_start_hz": float(f0), "f_end_hz": float(f0 + d * (len(x) - 1) / fs)}
    if name == "phase_noise":
        theta = lo_noise(len(x), fs, op.get("profile"), rng)
        return {"rms_rad": phase_rms_rad(op["profile"]),
                "realised_rms_rad": float(np.sqrt(np.mean(theta ** 2)))}
    if name == "awgn":
        p, n0 = awgn_levels(x, fs, grid.baud, **_stated(op, AWGN_KEYS))
        out = {"ref_power": p, "n0": n0, "noise_power": n0 * fs, "snr_db": _db(p / (n0 * fs))}
        if grid.baud:
            out["es_n0_db"] = _db(p / (grid.baud * n0))
        return out
    if name == "spur":
        p = _ref_power(x, op.get("ref_power"), "spur")
        return {"ref_power": p, "amplitude": float(np.sqrt(p * 10.0 ** (op["dbc"] / 10.0)))}
    fsc, peak, over = dac_scale(x, **_stated(op, DAC_KEYS))
    lsb = 2.0 * fsc / 2 ** int(op["bits"])
    return {"full_scale": fsc, "lsb": lsb, "rail_peak": peak,
            "crest_factor_db": _db(float(np.abs(x).max()) ** 2 / mean),
            "backoff_db": _db(2.0 * fsc ** 2 / mean),
            "clipped_fraction": over / (2.0 * len(x)),
            "sqnr_db": _db(mean / (2.0 * lsb ** 2 / 12.0))}


def report(sig):
    """What `Signal.rf_report()` returns -- see there."""
    from . import compose as C
    out = []
    for i, op in enumerate(sig.ops):
        if op["op"] in OPS:
            x = dataclasses.replace(sig, ops=sig.ops[:i]).waveform()
            grid = C._grid_now(sig.ops[:i], sig.grid, x)
            rng = Streams(sig.seed).role(f"phase_noise/{i}")
            out.append({"op": i, "name": op["op"], **_describe(op, x, grid, rng)})
    return out


def lo_phase(sig, streams=None):
    """What `Signal.lo_phase()` returns -- see there."""
    from . import compose as C
    hits = [i for i, op in enumerate(sig.ops) if op["op"] in ("freq_offset", "phase_noise")]
    if not hits:
        raise ValueError("lo_phase(): this chain has no freq_offset or phase_noise op, so no "
                         "oscillator phase was added to it.")
    st = streams if streams is not None else Streams(sig.seed)
    total, at = None, None
    for i in hits:
        op = sig.ops[i]
        fs, n = C._stored_rate(sig.ops[:i], float(sig.grid.fs), int(sig.grid.n))
        if at is not None and (fs, n) != at:
            raise ValueError(
                f"lo_phase(): op {i} ({op['op']}) acts on {n} samples at fs={fs:g} and an "
                f"earlier oscillator op on {at[1]} samples at fs={at[0]:g} -- the record was "
                f"resampled between them, so their phases are not on one timebase to add.")
        at = (fs, n)
        ph = (lo_noise(n, fs, op.get("profile"), st.role(f"phase_noise/{i}"))
              if op["op"] == "phase_noise"
              else lo_ramp(n, fs, op.get("f_hz") or 0.0, op.get("drift_hz_per_s") or 0.0))
        total = ph if total is None else total + ph
    return total

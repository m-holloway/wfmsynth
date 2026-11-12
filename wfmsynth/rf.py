"""wfmsynth.rf -- radio records: what a real transmitter, oscillator and channel do to an I/Q
record, one op per effect and each stated so that it can be measured back off the record; and
the measures that do the reading -- where the power is in frequency (`spectrum`, `band_power`,
`aclr`), how the envelope's power is distributed (`ccdf`), and how far the symbols are from the
ones that were sent (`evm`, `constellation_density`).

The two halves check each other: an impairment states what it came to (`Signal.rf_report()`,
`Signal.lo_phase()`), and a measure that was told nothing reads the same number off the record
-- the stated Es/N0 as EVM, the oscillator's offset in `evm(...)["removed"]`, a spur's level in
`band_power` (tests/test_rf_measures.py).

THE IMPAIRMENTS

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
                 log-frequency / dB axes. NOTHING is drawn outside the stated offsets, and
                 each FFT bin carries the profile's power over its own fs/n of offsets, so
                 the expected mean-square phase is the profile's integral (`phase_rms_rad`)
                 whatever the record length.
  awgn           y = x + n, n circular complex Gaussian, white across the whole sampled band.
                 Its level is one of three ratios against a REFERENCE POWER:
                     snr_db    P_ref / (N0 * bw_hz)    -- `bw_hz` is required: an SNR without
                                                          a bandwidth is not a number
                     es_n0_db  (P_ref / baud) / N0
                     eb_n0_db   (P_ref / baud / bits_per_symbol) / N0
                 P_ref is `ref_power=` if stated, else the mean of |x|**2 over the record as
                 it reaches this op. The total noise power added is N0*fs, split equally
                 between the rails. As the FIRST op of a chain it is noise with no signal
                 under it -- a complex Gaussian source, `ref_power` then required.
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

THE MEASURES

    x = sig.waveform()                              # complex baseband, from iq(...) or downconvert
    g = sig.stored_grid(x)                          # the grid that describes x
    sent, _ = sig.transmitted()                     # the ground truth
    rf.evm(x, g, sent)["evm_pct"]                   # nothing fitted: the record as it stands
    rf.evm(x, g, sent, correct=("gain", "phase"))   # ...and what was removed is in ["removed"]
    rf.ccdf(x)["papr_db"]
    rf.aclr(x, g, channel_bw_hz=33.75e6, offsets_hz=[40e6])["aclr_db"]
    s = rf.spectrum(x, g, rbw_hz=1e6)               # s["f_hz"], s["psd_per_hz"], s["rbw_hz"]

WHY THIS IS NOT IN `measure`. `measure` and `eye` fold a REAL record at the symbol period and
cluster what they find, deliberately without the transmitted data. These work in the frequency
domain or on a complex envelope, and `evm` is defined against the symbols the source sent. The
two families share no code and no argument beyond ``(x, grid)``; one module holding both would
need the reader to know which half applies to the record in hand.

THE CONVENTIONS

  * POWER is the mean of ``|x|**2`` in the record's own units squared. For a complex envelope
    that is the envelope's power, which is TWICE the power of the passband voltage it stands
    for (``Re{x*exp(j*w*t)}`` has mean square ``|x|**2 / 2``). Ratios -- ACLR, PAPR, EVM -- do
    not care; an absolute level does, so a spectrum of the upconverted record reads 3.01 dB
    below a spectrum of its envelope.
  * A complex record's spectrum is two-sided and its frequencies are OFFSETS from the centre
    frequency (`fc_hz` is returned beside them). A real record's is one-sided, 0 to fs/2, with
    both halves' power in it, so either way ``sum(psd_per_hz) * bin_hz`` is the record's power.
  * The RESOLUTION BANDWIDTH is the window's equivalent noise bandwidth: the width of the
    ideal filter that would pass the same noise power. With that definition a tone's power is
    its peak density times `rbw_hz` (to the window's scalloping loss), and a noise floor reads
    the same density at any resolution.
  * EVM is RMS error over RMS REFERENCE, as a fraction. `reference="truth"` compares with the
    symbols that were sent; `reference="decided"` with the nearest constellation point, which
    is all a receiver without the data can do and which reads LOW as soon as a decision is
    wrong.

WHAT EVM DOES NOT DO UNLESS ASKED. Nothing is fitted by default: a level error, a rotation, an
offset, a frequency error or a timing error counts as error in full. Each correction is named
in `correct=`, and the amount it took out is returned in ``["removed"]`` with its unit, so the
figure can never be read without the impairment that was hidden to get it.

The receiver is always the source's own: a root-raised-cosine matched filter sampled at the
symbol centres (`iq.demodulate`). A record that was already matched-filtered, or whose pulse is
not a root-raised-cosine, needs its own receiver and `iq.evm` on the symbols.

On a REAL passband record `spectrum`, `band_power` and `aclr` apply as they are (pass
`center_hz`); `ccdf` and `evm` are about the envelope and say to `downconvert` first.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import get_window, welch

from . import iq as IQ
from . import rx as RX
from . import stages as ST
from .streams import Streams

OPS = ("iq_imbalance", "iq_offset", "freq_offset", "phase_noise", "awgn", "spur", "dac")
# the ops that read the record's sample rate, which after a decimating `downconvert` is not
# the synthesis grid's
RATE_AWARE = ("freq_offset", "phase_noise", "awgn", "spur")
AWGN_KEYS = ("snr_db", "es_n0_db", "eb_n0_db", "bw_hz", "bits_per_symbol", "ref_power")
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


def _phase_power(f_hz, profile):
    """``integral of 2*L(f) df`` from the profile's first offset up to each of `f_hz`, in
    rad**2: closed form on every straight segment, and flat outside the stated offsets."""
    f, db = _profile(profile)
    fq = np.clip(np.atleast_1d(np.asarray(f_hz, float)), f[0], f[-1])
    total = np.zeros(fq.shape)
    for f1, f2, d1, d2 in zip(f[:-1], f[1:], db[:-1], db[1:]):
        a = (d2 - d1) / (10.0 * np.log10(f2 / f1))          # L = L1 * (f/f1)**a
        r = np.clip(fq, f1, f2) / f1
        total += 10.0 ** (d1 / 10.0) * f1 * (np.log(r) if abs(a + 1.0) < 1e-12
                                             else (r ** (a + 1.0) - 1.0) / (a + 1.0))
    return 2.0 * total


def phase_rms_rad(profile):
    """The rms phase the profile integrates to, in rad: ``sqrt(integral of 2*L(f) df)`` over
    the stated offsets, in closed form for each straight segment."""
    return float(np.sqrt(_phase_power(_profile(profile)[0][-1], profile)[0]))


def lo_noise(n, fs, profile, rng):
    """An oscillator's phase in rad at each of `n` samples: Gaussian, zero-mean, one-sided PSD
    ``2*L(f)`` at the profile's offsets and nothing elsewhere.

    White Gaussian samples shaped in the frequency domain. EACH BIN CARRIES ITS OWN BAND'S
    POWER: the integral of ``2*L(f)`` over the ``fs/n`` of offsets nearest that bin, not L at
    the bin's centre times its width. So the expected mean-square phase is the profile's
    integral (`phase_rms_rad`) at ANY record length -- including a profile whose close-in
    part, where most of the power is, is narrower than a bin -- and a bin many of which span
    a segment still reads the stated density. The process is periodic in the record. One
    record is NOT rescaled to the profile's rms: its rms phase is one draw's estimate of it
    (`Signal.rf_report()` gives both)."""
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
    edges = np.minimum(np.append(fb[0], fb) + 0.5 * fs / n, 0.5 * fs)     # between the bins
    power = np.diff(_phase_power(edges, profile))            # rad**2 in each bin's band
    # unit white noise puts 2/n of its power in each bin (1/n at DC and at Nyquist)
    share = np.full(len(fb), 2.0 / n)
    share[0] = 1.0 / n
    if n % 2 == 0:
        share[-1] = 1.0 / n
    gain = np.sqrt(power / share)
    gain[0] = 0.0                                            # a constant phase is not noise
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


def awgn_levels(x, fs, baud=None, snr_db=None, es_n0_db=None, eb_n0_db=None, bw_hz=None,
                bits_per_symbol=None, ref_power=None):
    """``(ref_power, n0)``: the reference power an `awgn` request is relative to and the noise
    density it resolves to, in power per Hz. The noise added is ``n0 * fs`` in total."""
    asked = {"snr_db": snr_db, "es_n0_db": es_n0_db, "eb_n0_db": eb_n0_db}
    given = [k for k, v in asked.items() if v is not None]
    if len(given) != 1:
        raise ValueError(
            f"awgn takes exactly one of snr_db= (with bw_hz=), es_n0_db= or eb_n0_db= (with "
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
    if kind == "eb_n0_db":
        if bits_per_symbol is None:
            raise ValueError(
                "awgn(eb_n0_db=...) needs bits_per_symbol= (1 bpsk, 2 qpsk, 4 qam16, 6 qam64; "
                "times the code rate if the bits are coded). Signal.awgn fills it in from an "
                "`iq` source's constellation; a recipe or a symbols= source has to state it.")
        k = _number(bits_per_symbol, "awgn", "bits_per_symbol")
        if not k > 0:
            raise ValueError(f"awgn: bits_per_symbol must be positive, got {bits_per_symbol!r}")
        band *= k
    elif bits_per_symbol is not None:
        raise ValueError(f"awgn: bits_per_symbol= goes with eb_n0_db=; {kind} does not use it.")
    p = _ref_power(x, ref_power, "awgn")
    return p, p / (band * ratio)


def silence(n, ref_power):
    """The record an `awgn` that STARTS a chain adds its noise to: `n` zeros. Noise with no
    signal under it is a complex Gaussian source -- the scatter a fading process is made of
    (`wfmsynth.recipes.fading`) -- and has nothing to be relative to, so `ref_power` is
    required."""
    if ref_power is None:
        raise ValueError(
            "awgn: this op starts the chain, so there is no signal for its level to be "
            "relative to. State ref_power=<a power> with the ratio: "
            "awgn(snr_db=0, bw_hz=<band>, ref_power=1.0) is noise with unit power in that "
            "band. (To add noise TO a signal, start the chain with iq(...) or capture(...).)")
    if n is None:
        raise ValueError("awgn: a chain that starts with noise needs a record length -- "
                         "Signal(grid=Grid(fs=..., n=...)).")
    return np.zeros(int(n), complex)


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


def _acting(sig):
    """Every op that acts on `sig`'s record, in order: ``(chain, i, op, upstream, before, st)``.

    For a lone Signal that is its own ops. For a channel of a several-channel record
    (`Channels.signal`) it also enters the chains its `use` ops stand for -- a shared
    oscillator is an impairment of every channel that uses it -- and `chain` is then that
    chain's name (None for the channel's own ops), `i` the op's index in it. `upstream` is
    every op before it, `before()` the record as it reaches the op, and `st` the streams the
    op draws from: its own chain's, so the numbers are the ones in the record."""
    flat = sig.ops if sig.within is None else sig.within[0].expand(sig.within[1])[1]
    k = [0]

    def walk(s, x_in, chain):
        for i, op in enumerate(s.ops):
            def before(i=i):
                return dataclasses.replace(s, ops=s.ops[:i])._run(_x=x_in)
            if op["op"] == "use" and s.within is not None:
                rec, name = s.within[0], op["name"]
                if rec.is_signal(name):
                    yield from walk(rec.signal(name), None, name)
                else:
                    x = before()
                    stage = rec._signal(name, like=x)
                    stage.within = (rec, name)
                    yield from walk(stage, x, name)
                continue
            yield chain, i, op, flat[:k[0]], before, s._streams()
            k[0] += 1

    yield from walk(sig, None, None)


def report(sig):
    """What `Signal.rf_report()` returns -- see there."""
    from . import compose as C
    out = []
    for chain, i, op, upstream, before, st in _acting(sig):
        if op["op"] in OPS:
            if op["op"] == "awgn" and not upstream:     # noise that starts the chain
                x, grid = silence(sig.grid.n, op.get("ref_power")), sig.grid
            else:
                x = before()
                grid = C._grid_now(upstream, sig.grid, x,
                                   *C.DOMAIN.trace(upstream, sig.grid)[-1])
            out.append({"op": i, "name": op["op"], **({} if chain is None else {"chain": chain}),
                        **_describe(op, x, grid, st.role(f"phase_noise/{i}"))})
    return out


def lo_phase(sig, streams=None):
    """What `Signal.lo_phase()` returns -- see there."""
    from . import compose as C
    hits = [h for h in _acting(sig) if h[2]["op"] in ("freq_offset", "phase_noise")]
    if not hits:
        raise ValueError("lo_phase(): this chain has no freq_offset or phase_noise op, so no "
                         "oscillator phase was added to it.")
    total, at = None, None
    for chain, i, op, upstream, _before, st in hits:
        if streams is not None:
            st = streams
        fs, n = C._stored_rate(upstream, float(sig.grid.fs), int(sig.grid.n))
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

# =============================================================================================
# THE MEASURES. Functions of (x, grid): no op, no recipe key.
# =============================================================================================
WINDOWS = ("hann", "hamming", "blackman", "blackmanharris", "flattop", "boxcar")
CORRECTIONS = ("timing", "frequency", "offset", "gain", "phase", "equalizer")
MIN_SEGMENT = 16          # samples: the shortest Welch segment `spectrum` will use
DECIDED_PASSES = 3        # decide, fit, correct -- this many times -- for reference="decided"
# A decided-reference frequency correction is ACQUIRED blind before any decision is made, from
# the carrier line in the M-th power of the symbols (M the constellation's rotational order:
# 2 for BPSK, 4 for QPSK and square QAM). Two numbers say whether it locked, and outside them
# `evm` refuses rather than return a figure:
LOCK_LINE = 6.0           # the line's height over the incoherent level sqrt(sum |s^M|^2)
LOCK_TURN = 0.25          # turns of s^M left across the counted symbols after correction
# With no line to acquire the decision-directed fit from zero offset is kept only if it is
# self-consistent:
FALLBACK_INSIDE = 0.5     # rms error, as a fraction of the distance to a decision boundary
FALLBACK_SETTLED = 0.01   # turns of s^M by which one more pass may move the frequency
FALLBACK_PULL_TURN = 0.15 # turns of s^M across the record it is stated to pull in from
#                           (measured, 112 clean QAM-16 symbols: locks at 0.18, refused at 0.21)


def _record(x, who):
    a = np.asarray(x)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{who}: expected one record (a 1-D array), got shape {a.shape}")
    return a


def _fs(grid, who):
    fs = getattr(grid, "fs", None)
    if fs is None:
        raise ValueError(f"{who}: needs the sample rate of the record -- pass the grid that "
                         f"describes it (sig.stored_grid(x), or Grid(fs=...))")
    return float(fs)


def _measured_envelope(x, who):
    a = _record(x, who)
    if not np.iscomplexobj(a):
        raise ValueError(
            f"{who}: the record is real, and this is a measure of the complex envelope. "
            f"downconvert(fc_hz=..., lpf=...) a passband record first; the envelope of a "
            f"record is not its samples.")
    return a


# ------------------------------------------------------------------ spectrum
def spectrum(x, grid, rbw_hz=None, window="hann", overlap=0.5):
    """Power spectral density by Welch's method: the record cut into overlapping segments,
    each windowed and transformed, the squared magnitudes averaged.

    Returns a dict:

      ``f_hz``         frequencies, ascending. Complex record: offsets from the centre
                       frequency, -fs/2 .. fs/2. Real record: 0 .. fs/2.
      ``psd_per_hz``   power per hertz at each (record units squared per Hz); a real record's
                       is one-sided, so both add up to the record's power
      ``rbw_hz``       the resolution bandwidth actually used (the window's noise bandwidth)
      ``bin_hz``       spacing of `f_hz`
      ``window``, ``segment``, ``segments``, ``overlap``   what produced it
      ``fc_hz``        the record's centre frequency if its grid has one, else None
      ``two_sided``    True for a complex record

    `rbw_hz` picks the segment length; the realised value is returned, since a segment is a
    whole number of samples. Unstated, an eighth of the record is used. A resolution the record
    is too short for is refused with the finest it can give. Nothing is detrended: a DC term or
    a carrier leak is part of the record."""
    a = _record(x, "spectrum")
    fs = _fs(grid, "spectrum")
    if window not in WINDOWS:
        raise ValueError(f"spectrum: unknown window {window!r} (use one of {list(WINDOWS)})")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"spectrum: overlap is a fraction of a segment in [0, 1), got "
                         f"{overlap!r}")

    def enbw_bins(m):
        w = get_window(window, m)
        return m * float(np.sum(w * w)) / float(np.sum(w)) ** 2

    n = a.size
    if rbw_hz is None:
        seg = max(min(n, MIN_SEGMENT), n // 8)
    else:
        if not rbw_hz > 0:
            raise ValueError(f"spectrum: rbw_hz must be positive, got {rbw_hz!r}")
        seg = int(round(enbw_bins(4096) * fs / rbw_hz))
    if seg > n:
        raise ValueError(
            f"spectrum: rbw_hz={rbw_hz:g} needs a {seg}-sample segment and the record has {n}. "
            f"The finest a {window!r} window resolves here is {enbw_bins(n) * fs / n:.6g} Hz; "
            f"lengthen the record or ask for a wider resolution.")
    if seg < MIN_SEGMENT:
        raise ValueError(
            f"spectrum: that resolution leaves a {seg}-sample segment (fs={fs:g}), too short "
            f"to be a spectrum. Ask for rbw_hz below {enbw_bins(MIN_SEGMENT) * fs / MIN_SEGMENT:.6g}"
            f" Hz, or use a longer record.")
    step = seg - int(overlap * seg)
    two = bool(np.iscomplexobj(a))
    f, p = welch(a, fs=fs, window=window, nperseg=seg, noverlap=seg - step, detrend=False,
                 return_onesided=not two, scaling="density")
    if two:
        f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    return {"f_hz": f, "psd_per_hz": p, "rbw_hz": enbw_bins(seg) * fs / seg, "bin_hz": fs / seg,
            "window": window, "segment": seg, "segments": 1 + (n - seg) // step,
            "overlap": float(overlap), "fc_hz": getattr(grid, "fc_hz", None), "two_sided": two}


def _weights(spec, lo_hz, hi_hz, who):
    """The fraction of each spectrum bin that lies inside ``[lo_hz, hi_hz]`` -- a bin is the
    interval of width `bin_hz` about its frequency, so a band edge between two bins takes the
    right share of each."""
    f, df = spec["f_hz"], spec["bin_hz"]
    if not lo_hz < hi_hz:
        raise ValueError(f"{who}: the band needs lo < hi, got {lo_hz:g} and {hi_hz:g} Hz")
    lo_f, hi_f = f[0] - 0.5 * df, f[-1] + 0.5 * df
    if lo_hz < lo_f - 1e-9 * df or hi_hz > hi_f + 1e-9 * df:
        raise ValueError(
            f"{who}: the band {lo_hz:g} .. {hi_hz:g} Hz reaches outside the spectrum, which "
            f"covers {lo_f:g} .. {hi_f:g} Hz"
            + (" as offsets from the centre frequency" if spec["two_sided"] else "")
            + ". Nothing outside it was measured.")
    return np.clip((np.minimum(f + 0.5 * df, hi_hz) - np.maximum(f - 0.5 * df, lo_hz)) / df,
                   0.0, 1.0)


def band_power(spec, lo_hz, hi_hz):
    """The power between two frequencies of a `spectrum` result: its density integrated over
    the band, in record units squared. Frequencies are in the spectrum's own terms (offsets
    from the centre for a complex record). A band reaching past the spectrum is refused."""
    w = _weights(spec, lo_hz, hi_hz, "band_power")
    return float(np.sum(spec["psd_per_hz"] * w) * spec["bin_hz"])


def _rrc_power_response(f_hz, bw_hz, beta):
    """``|H(f)|**2`` of a root-raised-cosine receive filter of symbol rate `bw_hz`: a raised
    cosine, 1 in the middle, whose noise bandwidth is exactly `bw_hz`."""
    af = np.abs(f_hz) / bw_hz
    out = np.zeros_like(af)
    out[af <= 0.5 * (1 - beta)] = 1.0
    roll = (af > 0.5 * (1 - beta)) & (af < 0.5 * (1 + beta))
    if beta > 0:
        out[roll] = 0.5 * (1.0 + np.cos(np.pi / beta * (af[roll] - 0.5 * (1 - beta))))
    return out


def aclr(x, grid, channel_bw_hz, offsets_hz, center_hz=None, filter="rect", beta=None,
         rbw_hz=None, window="hann"):
    """Adjacent-channel leakage ratio: the power in the wanted channel over the power in a
    channel of the same width centred `offsets_hz` away, in dB. POSITIVE is good (the
    neighbour is that far below the channel).

    `filter` is the measuring filter, applied alike to the channel and to each neighbour:
    ``"rect"`` integrates the density across `channel_bw_hz`; ``"rrc"`` weights it with a
    root-raised-cosine of symbol rate `channel_bw_hz` and roll-off `beta` (the receive filter
    some standards specify; it reaches ``(1 + beta) / 2`` of the width either side). Both pass
    the same power of a flat spectrum.

    `center_hz` is where the wanted channel sits: an offset from the centre frequency for a
    complex record (default 0), an absolute frequency for a real passband record (required).

    Returns ``{"aclr_db": [...], "offsets_hz": [...], "channel_power", "adjacent_power": [...],
    "channel_bw_hz", "center_hz", "filter", "beta", "rbw_hz", "window"}``. The spectrum behind
    it is `spectrum(x, grid, rbw_hz, window)`, `rbw_hz` defaulting to a fiftieth of the channel;
    a resolution coarser than a tenth of the channel is refused, because the window would then
    smear the channel into its neighbour and the ratio would be the window's."""
    a = _record(x, "aclr")
    if not channel_bw_hz > 0:
        raise ValueError(f"aclr: channel_bw_hz must be positive, got {channel_bw_hz!r}")
    if filter not in ("rect", "rrc"):
        raise ValueError(f"aclr: unknown filter {filter!r} (use 'rect' or 'rrc')")
    if filter == "rrc" and (beta is None or not 0.0 <= beta <= 1.0):
        raise ValueError("aclr: filter='rrc' needs its roll-off stated, beta= in [0, 1]")
    if filter == "rect" and beta is not None:
        raise ValueError("aclr: beta is the roll-off of filter='rrc'; filter='rect' has none")
    if center_hz is None:
        if not np.iscomplexobj(a):
            raise ValueError("aclr: a real passband record needs center_hz=, the carrier "
                             "frequency the wanted channel sits on")
        center_hz = 0.0
    offs = np.atleast_1d(np.asarray(offsets_hz, float))
    if offs.size == 0 or np.any(offs == 0):
        raise ValueError("aclr: offsets_hz are the neighbours' centres relative to the "
                         "channel's, at least one and none of them 0")
    if rbw_hz is None:
        rbw_hz = channel_bw_hz / 50.0
    elif rbw_hz > channel_bw_hz / 10.0:
        raise ValueError(
            f"aclr: rbw_hz={rbw_hz:g} is more than a tenth of the {channel_bw_hz:g} Hz channel, "
            f"so the window's own skirt would set the ratio. Use rbw_hz <= "
            f"{channel_bw_hz / 10.0:g}.")
    spec = spectrum(a, grid, rbw_hz=rbw_hz, window=window)
    half = 0.5 * channel_bw_hz * (1.0 + beta if filter == "rrc" else 1.0)

    def power(c):
        w = _weights(spec, c - half, c + half, "aclr")
        if filter == "rrc":
            w = w * _rrc_power_response(spec["f_hz"] - c, channel_bw_hz, beta)
        return float(np.sum(spec["psd_per_hz"] * w) * spec["bin_hz"])

    main = power(float(center_hz))
    if not main > 0:
        raise ValueError("aclr: there is no power in the wanted channel; check center_hz and "
                         "channel_bw_hz against the record")
    adj = [power(float(center_hz) + o) for o in offs]
    return {"aclr_db": [float(10 * np.log10(main / p)) if p > 0 else float("inf") for p in adj],
            "offsets_hz": [float(o) for o in offs], "channel_power": main,
            "adjacent_power": adj, "channel_bw_hz": float(channel_bw_hz),
            "center_hz": float(center_hz), "filter": filter, "beta": beta,
            "rbw_hz": spec["rbw_hz"], "window": window}


# ------------------------------------------------------------------ envelope statistics
def ccdf(x, levels_db=None):
    """How often the envelope's instantaneous power exceeds a level above its average.

    Returns ``{"papr_db", "avg_power", "peak_power", "levels_db", "probability", "n"}``:
    ``probability[i]`` is the fraction of samples whose ``|x|**2`` is above the average by more
    than ``levels_db[i]``. `levels_db` defaults to 0 .. the peak in 0.25 dB steps. For complex
    Gaussian noise the law is ``exp(-10**(level_db / 10))``.

    A probability is a count over `n` samples, so nothing below ``1/n`` is resolved and the
    peak-to-average ratio is the largest of THIS record, not a bound. The whole record counts:
    slice off a burst's turn-on before calling if it is not what you are measuring."""
    p = np.abs(_measured_envelope(x, "ccdf")) ** 2
    avg = float(p.mean())
    if not avg > 0:
        raise ValueError("ccdf: the record is all zero; it has no average power to refer to")
    peak = float(p.max())
    papr = 10 * np.log10(peak / avg)
    lv = (np.arange(0.0, papr + 0.25, 0.25) if levels_db is None
          else np.atleast_1d(np.asarray(levels_db, float)))
    ratio = np.sort(p) / avg
    prob = 1.0 - np.searchsorted(ratio, 10 ** (lv / 10), side="right") / p.size
    return {"papr_db": float(papr), "avg_power": avg, "peak_power": peak, "levels_db": lv,
            "probability": prob, "n": int(p.size)}


def constellation_density(symbols, bins=64, extent=None):
    """A 2-D histogram of complex points, for plotting a constellation (or a whole envelope's
    trajectory) as a density rather than a scatter.

    Returns ``{"density", "i_edges", "q_edges"}``: ``density[q, i]`` is the COUNT of points in
    that cell -- rows are Q, columns are I, so ``imshow(density, origin="lower")`` is the right
    way up. `extent` is the half-width of the square about the origin; unstated it is 1.1 times
    the largest |I| or |Q|. A point outside a stated extent is refused, not dropped."""
    s = np.asarray(symbols)
    if s.ndim != 1 or s.size == 0 or not np.iscomplexobj(s):
        raise ValueError("constellation_density: expected a 1-D complex array (symbols from "
                         "evm(...)['received'] or iq.demodulate, or an envelope)")
    reach = float(max(np.abs(s.real).max(), np.abs(s.imag).max()))
    if extent is None:
        extent = 1.1 * reach if reach > 0 else 1.0
    elif reach > extent:
        raise ValueError(f"constellation_density: a point reaches {reach:g} and extent={extent:g}"
                         f" would leave it out of the picture; widen extent")
    edges = np.linspace(-extent, extent, int(bins) + 1)
    h, _, _ = np.histogram2d(s.imag, s.real, bins=(edges, edges))
    return {"density": h, "i_edges": edges, "q_edges": edges}


# ------------------------------------------------------------------ EVM
def constellation_points(constellation):
    """The points of a named constellation (`iq.CONSTELLATIONS`), or a given alphabet as a
    complex array."""
    if isinstance(constellation, str):
        k = IQ.bits_per_symbol(constellation)
        bits = (np.arange(2 ** k)[:, None] >> np.arange(k - 1, -1, -1)) & 1
        return IQ.map_bits(constellation, bits.ravel())
    pts = IQ.as_complex(constellation)
    if pts.size < 2:
        raise ValueError("evm: a constellation needs at least two points")
    return pts


def decide(received, constellation):
    """The nearest constellation point to each received symbol."""
    pts = constellation_points(constellation)
    r = np.asarray(received, complex)
    return pts[np.argmin(np.abs(r[:, None] - pts[None, :]), axis=1)]


def _frequency(r, s, t):
    """The frequency that best explains a steady rotation of `r` against `s`: the peak of
    ``|sum r * conj(s) * exp(-j*2*pi*f*t)|``, found on a fine FFT grid and then refined."""
    z = r * np.conj(s)
    dt = float(t[1] - t[0])
    nfft = 1 << int(np.ceil(np.log2(8 * len(z))))
    f = np.fft.fftfreq(nfft, dt)
    f0 = float(f[int(np.argmax(np.abs(np.fft.fft(z, nfft))))])
    step = 1.0 / (nfft * dt)
    tt = t - t[0]
    best = minimize_scalar(lambda v: -abs(np.sum(z * np.exp(-2j * np.pi * v * tt))),
                           bounds=(f0 - step, f0 + step), method="bounded",
                           options={"xatol": 1e-9 * step})
    return float(best.x)


def symmetry_order(constellation):
    """The rotational order M of a constellation: the smallest power whose mean over the
    points is not zero, so that ``s**M`` of any symbol stream carries a line at M times the
    carrier offset. 1 for an alphabet with a DC term, 2 for BPSK, 4 for QPSK and square QAM."""
    pts = constellation_points(constellation)
    for m in range(1, 9):
        if abs(np.mean(pts ** m)) > 1e-6 * np.mean(np.abs(pts) ** m):
            return m
    raise ValueError("evm: this constellation has no rotational order up to 8, so its "
                     "carrier cannot be found blind; give reference='truth' with sent=")


def carrier_line(r, t, constellation):
    """The carrier of a symbol stream found WITHOUT decisions, from the line in its M-th power.

    Returns ``(frequency_hz, phase_rad, line, order)``: the offset (unambiguous within
    ``+/- baud / (2 * M)``; a larger one comes back modulo ``baud / M``), the phase at `t[0]`
    that puts the constellation upright (modulo ``2*pi / M``), the line's height over the
    incoherent level -- about ``c * sqrt(n)`` for n symbols, with c = 0.36 for clean QAM-16
    and 1 for QPSK, and about ``sqrt(ln n)`` when there is no line at all -- and M."""
    pts = constellation_points(constellation)
    m = symmetry_order(pts)
    z = np.asarray(r, complex) ** m
    f = _frequency(z, np.ones(z.size, complex), t)
    tt = t - t[0]
    peak = np.sum(z * np.exp(-2j * np.pi * f * tt))
    level = float(np.sqrt(np.sum(np.abs(z) ** 2)))
    phase = (np.angle(peak) - np.angle(np.mean(pts ** m))) / m
    phase = (phase + np.pi / m) % (2 * np.pi / m) - np.pi / m
    return f / m, float(phase), (float(abs(peak)) / level if level > 0 else 0.0), m


def _upright(r, constellation):
    """The rotation (modulo ``2*pi / M``) that puts a symbol stream with no frequency offset
    upright on its constellation, from the phase of the mean of its M-th power."""
    pts = constellation_points(constellation)
    m = symmetry_order(pts)
    phase = (np.angle(np.mean(np.asarray(r, complex) ** m)) - np.angle(np.mean(pts ** m))) / m
    return float((phase + np.pi / m) % (2 * np.pi / m) - np.pi / m)


def _refuse_unlocked(why, baud, order, n):
    return ValueError(
        f"evm: reference='decided' with correct='frequency' did not lock: {why}. The carrier "
        f"is acquired blind from the line in the {order}th power of the symbols, which finds "
        f"an offset within +/- baud/(2*{order}) = +/-{baud / (2 * order):.6g} Hz given enough "
        f"symbols for the noise ({n} counted here). No figure is returned, because it would "
        f"describe the failure to lock and not the signal. Give reference='truth' with "
        f"sent=, or take the known offset out first (sig.lo_phase()).")


def _equalizer(y, s, keep, taps):
    """Least-squares symbol-spaced FIR: the `taps` weights that best turn `y` into `s` over the
    kept symbols. Returns ``(weights, equalised y)``; the centre weight multiplies the symbol
    itself, and neighbours beyond the record's ends are taken as zero."""
    h = taps // 2
    pad = np.concatenate([np.zeros(h, complex), y, np.zeros(h, complex)])
    A = np.stack([pad[j:j + len(y)] for j in range(taps)], axis=1)   # column j: y[k + j - h]
    w = np.linalg.lstsq(A[keep], s[keep], rcond=None)[0]
    return w, A @ w


def evm(x, grid, sent=None, reference="truth", constellation=None, correct=(), skip=0,
        delay_s=0.0, equalizer_taps=None, beta=IQ.RRC_BETA, span_ui=IQ.RRC_SPAN_UI):
    """Error-vector magnitude of a complex baseband record: RMS distance between the received
    symbols and the reference, over the RMS of the reference.

        rf.evm(x, grid, sent)                                   # against the truth, as it stands
        rf.evm(x, grid, sent, correct=("gain", "phase"))        # after one complex gain
        rf.evm(x, grid, reference="decided", constellation="qam16")

    The receiver is a root-raised-cosine matched filter (`beta`, `span_ui`) sampled at the
    symbol centres, `delay_s` later than the transmitter's (a known group delay; whole symbols
    of delay go here). `skip` leaves that many symbols out at EACH end -- a burst's first and
    last symbols lack their neighbours' tails -- and is 0 unless stated.

    `reference`:
      ``"truth"``    `sent`, the transmitted symbols (`Signal.transmitted()[0]`)
      ``"decided"``  the nearest point of `constellation` (a name or an alphabet). When `sent`
                     is given as well it is used ONLY to count ``decision_errors``, the number
                     of symbols where the decided reference is wrong -- every one of which
                     makes the decided figure read lower than the truth.

    `correct` names what to take out before measuring, and nothing is taken out otherwise:
      ``"timing"``     the sampling instant, searched within +/- half a symbol of `delay_s`
      ``"frequency"``  a steady rotation: a residual carrier-frequency offset, turned back
                       about the first counted symbol (so the phase there is left for
                       ``"phase"``)
      ``"offset"``     a constant added to every symbol: carrier leak, a DC offset. Fitted
                       AFTER the frequency correction, so it is a constant in the
                       transmitter's frame; an offset added after a frequency error rotates
                       against the symbols and is not removed by this
      ``"gain"``       one real scale factor
      ``"phase"``      one rotation
      ``"equalizer"``  a symbol-spaced linear equaliser of `equalizer_taps` taps (odd, stated)
    They are estimated in that order, each by least squares against the reference. With
    `reference="decided"` the reference depends on the correction, so decide / fit / correct
    is run three times -- which only REFINES. A carrier is therefore acquired first, blind,
    from the line in the M-th power of the symbols (`carrier_line`; M = 2 for BPSK, 4 for
    QPSK and square QAM), when ``"frequency"`` or ``"phase"`` is named:

      * the frequency is found within ``+/- baud / (2 * M)`` -- 3.125 MHz for QAM-16 at
        25 MBd -- whatever the phase and without one right decision; a larger offset comes
        back modulo ``baud / M``, a quarter-turn per symbol being invisible at the symbol
        instants (and by then the matched filter has lost much of the signal);
      * the phase is found modulo ``2*pi / M``, the constellation's own symmetry;
      * with ``"frequency"`` named the lock is CHECKED on the corrected symbols: a carrier
        line at least `LOCK_LINE` high, turning less than `LOCK_TURN` across the record;
      * where there is NO line to acquire (too few symbols: 128 of clean QAM-16 stand about
        4 to 6 high) the decision-directed fit is run from zero offset instead, and kept only if it
        is self-consistent: rms error under `FALLBACK_INSIDE` of the distance to a decision
        boundary, and one more pass moving the frequency by under `FALLBACK_SETTLED` of a
        turn. Its pull-in is a small fraction of ``1 / (M * record length)``. A record that
        passes neither route is REFUSED, saying which numbers failed;
      * ``["acquisition"]`` always states the route taken, ``pull_in_hz`` and
        ``alias_spacing_hz`` (``baud / M``): WITHOUT `sent`, an offset outside the pull-in
        range is returned as its alias and nothing in the symbols can show it, so read the
        frequency against those two numbers. WITH `sent` the alias is checked -- decisions
        that agree with the sent symbols far better with the constellation turned a symmetry
        step per symbol mean the wrong alias -- and such a record is refused, naming the
        offset it actually has.

    A ``reference="decided"`` figure still reads LOW once decisions fail in noise; give
    `sent` to count them.

    Returns a dict:
      ``evm``, ``evm_pct``, ``evm_db``   RMS, as a fraction, a percentage and 20*log10
      ``evm_peak``                       the worst symbol, same normalisation
      ``n_symbols``, ``skip``, ``reference``, ``receiver``, ``delay_s``
      ``removed``                        what each named correction took out: ``timing_s``,
                                         ``frequency_hz``, ``offset`` (complex, symbol units),
                                         ``gain_db``, ``phase_rad``, ``equalizer_taps``
      ``received``, ``reference_symbols``  the corrected symbols and what they were compared
                                           with (after `skip`), for a plot or a bit count
      ``decision_errors``                only for "decided" with `sent` given. Where a carrier
                                         was acquired blind the count is taken with the
                                         decided points ``symmetry_steps`` steps of
                                         ``2*pi / M`` ahead of the sent ones, whichever
                                         explains them best: a constellation locked a
                                         quarter-turn round has every decision right
      ``acquisition``                    only for "decided" with ``"frequency"``: ``method``,
                                         ``order``, ``pull_in_hz``, ``alias_spacing_hz``,
                                         ``coarse_frequency_hz``, ``line``, and
                                         ``residual_turn`` (blind) or ``inside`` and
                                         ``settled_turn`` (decision-directed from zero)
    """
    a = _measured_envelope(x, "evm")
    fs, baud = IQ._rate(grid, "evm")
    if reference not in ("truth", "decided"):
        raise ValueError(f"evm: reference is 'truth' (against sent=) or 'decided' (against the "
                         f"nearest point of constellation=), got {reference!r}")
    if reference == "truth":
        if sent is None:
            raise ValueError("evm: reference='truth' needs the transmitted symbols, "
                             "sent=sig.transmitted()[0]. Without them use reference='decided' "
                             "with constellation=.")
        if constellation is not None:
            raise ValueError("evm: constellation= is what reference='decided' decides onto; "
                             "reference='truth' compares with sent= and does not use it")
    elif constellation is None:
        raise ValueError("evm: reference='decided' needs constellation= (a name such as "
                         "'qam16', or the alphabet as a complex array)")
    correct = (correct,) if isinstance(correct, str) else tuple(correct)
    bad = [c for c in correct if c not in CORRECTIONS]
    if bad or len(set(correct)) != len(correct):
        raise ValueError(f"evm: correct= takes each of {list(CORRECTIONS)} at most once, got "
                         f"{list(correct)}")
    if ("equalizer" in correct) != (equalizer_taps is not None):
        raise ValueError("evm: an equaliser is asked for with BOTH correct=('equalizer', ...) "
                         "and equalizer_taps=<odd count>; one without the other does nothing")
    if equalizer_taps is not None and (int(equalizer_taps) != equalizer_taps
                                       or equalizer_taps < 1 or equalizer_taps % 2 == 0):
        raise ValueError(f"evm: equalizer_taps must be a positive odd count (the centre tap "
                         f"is the symbol itself), got {equalizer_taps!r}")
    fit = IQ.symbols_in_record(grid, a.size)
    truth = None if sent is None else np.asarray(sent, complex).ravel()
    n_sym = fit if truth is None else truth.size
    if n_sym > fit:
        raise ValueError(f"evm: {n_sym} sent symbols against a record that holds {fit} at "
                         f"{fs / baud:.4g} samples per symbol. Is `grid` the one that describes "
                         f"this record (sig.stored_grid(x))?")
    skip = int(skip)
    if skip < 0 or n_sym - 2 * skip < 2:
        raise ValueError(f"evm: skip={skip} at each end leaves {n_sym - 2 * skip} of {n_sym} "
                         f"symbols; nothing to measure")
    keep = slice(skip, n_sym - skip)
    pts = None if reference == "truth" else constellation_points(constellation)

    def receive(d):
        return IQ.demodulate(a, grid, n_sym=n_sym, beta=beta, span_ui=span_ui, delay_s=d)

    removed = {}
    if "timing" in correct:
        # with the truth: the instant at which the symbols correlate best with what was sent.
        # without it: the instant of greatest symbol power, which a root-raised-cosine pair
        # (zero ISI at the right instant, ISI either side) puts at the same place.
        def score(d):
            r = receive(delay_s + d)[keep]
            return -(abs(np.vdot(truth[keep], r)) if reference == "truth"
                     else float(np.mean(np.abs(r) ** 2)))
        ui = 1.0 / baud
        coarse = np.linspace(-0.5, 0.5, 33) * ui
        d0 = float(coarse[int(np.argmin([score(d) for d in coarse]))])
        best = minimize_scalar(score, bounds=(d0 - ui / 32, d0 + ui / 32), method="bounded",
                               options={"xatol": 1e-6 * ui})
        removed["timing_s"] = float(best.x)
        delay_s = delay_s + removed["timing_s"]
    raw = receive(delay_s)
    t = IQ.symbol_times(n_sym, grid)

    def corrected(ref, src=None):
        """`raw` (or `src`) with every named correction estimated against `ref` and taken out."""
        y, out = (raw if src is None else src).copy(), {}
        if "frequency" in correct:
            out["frequency_hz"] = _frequency(y[keep], ref[keep], t[keep])
            y = y * np.exp(-2j * np.pi * out["frequency_hz"] * (t - t[keep][0]))
        if {"offset", "gain", "phase"} & set(correct):
            cols = [ref[keep]] + ([np.ones(n_sym - 2 * skip, complex)] if "offset" in correct
                                  else [])
            sol = np.linalg.lstsq(np.stack(cols, axis=1), y[keep], rcond=None)[0]
            if "offset" in correct:
                out["offset"] = complex(sol[1])
                y = y - sol[1]
            if "gain" in correct:
                out["gain_db"] = float(20 * np.log10(abs(sol[0])))
                y = y / abs(sol[0])
            if "phase" in correct:
                out["phase_rad"] = float(np.angle(sol[0]))
                y = y * np.exp(-1j * out["phase_rad"])
        if "equalizer" in correct:
            out["equalizer_taps"], y = _equalizer(y, ref, keep, int(equalizer_taps))
        return y, out

    if reference == "truth":
        ref = truth
        y, fitted = corrected(ref)
    else:
        # ACQUIRE before deciding. A decision-directed fit only refines: started on a record
        # that turns a fraction of a symmetry step across its length, its decisions are wrong,
        # the fit follows them, and the figure that comes out is neither the signal's nor a
        # sign that anything failed. So the carrier is found blind first.
        base, pre, order, blind = raw, {}, None, False
        span = float(t[keep][-1] - t[keep][0])
        if "frequency" in correct:
            f0, _, line0, order = carrier_line(raw[keep], t[keep], pts)
            blind = line0 >= LOCK_LINE
            if blind:
                pre["frequency_hz"] = f0
                base = base * np.exp(-2j * np.pi * f0 * (t - t[keep][0]))
        if "phase" in correct and (blind or "frequency" not in correct):
            pre["phase_rad"] = (carrier_line(base[keep], t[keep], pts)[1] if blind
                                else _upright(base[keep], pts))
            base = base * np.exp(-1j * pre["phase_rad"])
        y = base
        if "gain" in correct:            # a level to decide at; the fit below states the gain
            y = base * np.sqrt(np.mean(np.abs(pts) ** 2) / np.mean(np.abs(base[keep]) ** 2))
        fitted = {}
        for _ in range(DECIDED_PASSES if correct else 0):
            y, fitted = corrected(decide(y, pts), base)
        ref = decide(y, pts)
        if "frequency" in correct and blind:     # CHECK it held, on the symbols as corrected
            f1, _, line, _ = carrier_line(y[keep], t[keep], pts)
            turn = abs(f1) * order * span
            if line < LOCK_LINE or turn > LOCK_TURN:
                raise _refuse_unlocked(
                    f"after correction the symbols' {order}th power still turns {turn:.2f} of "
                    f"a turn across the record (limit {LOCK_TURN:g}), line height {line:.1f} "
                    f"(needs {LOCK_LINE:g})", baud, order, n_sym - 2 * skip)
            acquisition = {"method": f"line in the {order}th power of the symbols",
                           "order": order, "pull_in_hz": baud / (2 * order),
                           "alias_spacing_hz": baud / order,
                           "coarse_frequency_hz": pre["frequency_hz"], "line": line,
                           "residual_turn": turn}
        elif "frequency" in correct:
            # NO LINE TO ACQUIRE: too few symbols for the constellation (128 of clean QAM-16
            # stand about 4 to 6 high). What is left is the decision-directed fit started from
            # zero offset, which is right when the record's decisions were mostly right to begin
            # with and garbage otherwise -- so it is accepted only if it is self-consistent:
            # the error vector sits well inside the decision regions the fit rested on, and
            # one more pass leaves the frequency where it was.
            apart = np.abs(pts[:, None] - pts[None, :])[~np.eye(pts.size, dtype=bool)]
            half = 0.5 * float(apart.min())
            level = float(np.sqrt(np.mean(np.abs(ref[keep]) ** 2) / np.mean(np.abs(pts) ** 2)))
            inside = float(np.sqrt(np.mean(np.abs(y[keep] - ref[keep]) ** 2))) / (half * level)
            again = corrected(ref, base)[1]["frequency_hz"]
            moved = abs(again - fitted["frequency_hz"]) * order * span
            if inside > FALLBACK_INSIDE or moved > FALLBACK_SETTLED:
                raise _refuse_unlocked(
                    f"the symbols' {order}th power has no carrier line to acquire (height "
                    f"{line0:.1f} over the incoherent level, {LOCK_LINE:g} needed), and the "
                    f"decision-directed fit from zero offset is not self-consistent: its rms "
                    f"error is {inside:.2f} of the way to a decision boundary (limit "
                    f"{FALLBACK_INSIDE:g}) and one more pass moves the frequency by "
                    f"{moved:.3f} of a turn across the record (limit {FALLBACK_SETTLED:g})",
                    baud, order, n_sym - 2 * skip)
            acquisition = {"method": "decision-directed from zero offset (no carrier line: "
                                     f"height {line0:.1f}, {LOCK_LINE:g} needed)",
                           "order": order, "pull_in_hz": FALLBACK_PULL_TURN / (order * span),
                           "alias_spacing_hz": baud / order, "coarse_frequency_hz": 0.0,
                           "line": line0, "inside": inside, "settled_turn": moved}
        for k, v in pre.items():
            fitted[k] = fitted.get(k, 0.0) + v
        if "phase_rad" in fitted:
            fitted["phase_rad"] = float((fitted["phase_rad"] + np.pi) % (2 * np.pi) - np.pi)
    removed.update(fitted)
    err = y[keep] - ref[keep]
    scale = float(np.sqrt(np.mean(np.abs(ref[keep]) ** 2)))
    if not scale > 0:
        raise ValueError("evm: the reference symbols are all zero; there is nothing to "
                         "normalise the error by")
    val = float(np.sqrt(np.mean(np.abs(err) ** 2))) / scale
    out = {"evm": val, "evm_pct": 100.0 * val,
           "evm_db": float(20 * np.log10(val)) if val > 0 else float("-inf"),
           "evm_peak": float(np.abs(err).max()) / scale,
           "n_symbols": n_sym - 2 * skip, "skip": skip, "reference": reference,
           "receiver": f"rrc matched filter, beta={beta:g}, span_ui={span_ui:g}",
           "delay_s": float(delay_s), "removed": removed,
           "received": y[keep], "reference_symbols": ref[keep]}
    if reference == "decided" and truth is not None:
        # a blind carrier is found modulo the constellation's own symmetry, so the decided
        # points may sit a whole number of symmetry steps from the sent ones with every
        # decision right: count the errors at the step that explains them best, and say it
        steps = symmetry_order(pts) if {"frequency", "phase"} & set(correct) else 1
        errs = [int(np.sum(~np.isclose(ref[keep] * np.exp(-2j * np.pi * k / steps), truth[keep])))
                for k in range(steps)]
        out["decision_errors"] = min(errs)
        out["symmetry_steps"] = int(np.argmin(errs))
        if "frequency" in correct and steps > 1:
            # THE AMBIGUITY, checked where it can be: a carrier one alias off turns the
            # constellation a whole symmetry step per symbol, which the symbols alone cannot
            # show and the sent ones can. If the decisions agree with what was sent far
            # better under such a turn, the lock is on the wrong alias and the frequency
            # reported would be wrong by a multiple of baud / M.
            k_sym = np.arange(n_sym)[keep]
            alias = {a: min(int(np.sum(~np.isclose(
                ref[keep] * np.exp(-2j * np.pi * (a * k_sym + c) / steps), truth[keep])))
                for c in range(steps)) for a in range(1, steps)}
            a_best = min(alias, key=alias.get)
            counted = n_sym - 2 * skip
            # "far better": by more than chance moves a count of this many decisions
            if min(errs) - alias[a_best] > max(0.05 * counted, 4.0 * np.sqrt(counted)):
                step_hz = baud / steps
                turns = a_best if a_best <= steps // 2 else a_best - steps
                raise ValueError(
                    f"evm: reference='decided' locked onto the WRONG ALIAS of the carrier. "
                    f"Against sent=, {min(errs)} of {n_sym - 2 * skip} decisions are wrong as "
                    f"corrected and {alias[a_best]} if the constellation turns {turns:+d} "
                    f"symmetry step(s) per symbol: the offset is "
                    f"{removed['frequency_hz'] + turns * step_hz:.6g}"
                    f" Hz (or that plus a multiple of {steps * step_hz:.6g}), not the "
                    f"{removed['frequency_hz']:.6g} Hz "
                    f"found. A blind estimate is unambiguous only within "
                    f"+/-{baud / (2 * steps):.6g} Hz and repeats every {step_hz:.6g} Hz. Use "
                    f"reference='truth', or take a coarse estimate of the offset out of the "
                    f"record first (sig.lo_phase(), or x * exp(-2j*pi*f*t)).")
    if reference == "decided" and "frequency" in correct:
        out["acquisition"] = acquisition
    return out

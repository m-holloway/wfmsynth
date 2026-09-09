"""
wfmsynth.instrument — front-end / digitizer models. A real high-speed scope is a
TIME-INTERLEAVED ADC: M sub-ADC cores sample in round-robin, and the mismatch between
them is much of what makes a measured capture look measured. Perfectly white noise on a
perfectly uniform grid is exactly the "too clean" signature a model trained on synthetic
learns to rely on and then breaks on real data.

`interleave_adc` injects per-core **gain**, **offset** and **timing-skew** mismatch,
which produce the characteristic interleave spurs: offset mismatch -> tones at k*fs/M
(input-independent); gain/skew mismatch -> image tones mirrored about fs/(2M), i.e. at
k*fs/M +/- f_in. Set all mismatch to zero for an ideal ADC (no spurs). numpy/scipy only.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as _sig


def interleave_adc(x, m_cores=4, gain_mm=0.0, offset_mm=0.0, skew_mm=0.0,
                   rng=None, mismatch=None, offset_v=None):
    """Pass x through an M-way time-interleaved ADC with per-core mismatch.

      gain_mm    per-core gain error, std (fractional, e.g. 0.01 = 1%)
      offset_mm  per-core offset, std as a fraction of the signal span
      offset_v   per-core offset, std in ABSOLUTE amplitude units (volts). A real
                 converter's offset error is a property of the ADC, not the signal: it
                 stays put when the signal shrinks (which is when it matters most).
                 Takes precedence over offset_mm when given.
      skew_mm    per-core sampling-time skew, std in SAMPLES (sub-sample)

    Cores are assigned round-robin: sample i is taken by core i % m_cores. Draw the M
    per-core errors once (seed via rng), or pass `mismatch=(g, o, s)` explicitly for a
    reproducible fixed pattern. Returns the digitized array (same length)."""
    x = np.asarray(x, float)
    n = len(x)
    span = np.ptp(x) + 1e-9
    core = np.arange(n) % m_cores
    if mismatch is not None:
        g, o, s = (np.asarray(v, float) for v in mismatch)
    else:
        rng = rng or np.random.default_rng()
        g = rng.normal(0.0, gain_mm, m_cores)
        o_std = offset_v if offset_v is not None else offset_mm * span   # absolute vs fraction
        o = rng.normal(0.0, o_std, m_cores)
        s = rng.normal(0.0, skew_mm, m_cores)
    y = x.copy()
    if skew_mm > 0 or (mismatch is not None and np.any(s)):
        xi = np.arange(n)                          # sub-sample timing skew per core
        for c in range(m_cores):
            idx = np.where(core == c)[0]
            y[idx] = np.interp(idx + s[c], xi, x)
    y = y * (1.0 + g[core]) + o[core]              # per-core gain + offset
    return y


def scope_bandwidth(x, grid, bw_hz, kind="bessel", order=4):
    """A bandwidth limit. ``kind='bessel'`` (flat group delay) or ``'gaussian'`` model the
    ANALOG front end, which is band-limited by physics and rolls off gently. ``'brickwall'``
    models the instrument's DIGITAL selected-bandwidth filter, which sits after the converter
    and is not gentle at all: MEASURED on the three real Keysight exports, the record's noise
    floor drops **38 dB across 2 GHz** at the corner and is flat on both sides of it. Use the
    analog kinds before the converter and ``'brickwall'`` after it."""
    x = np.asarray(x, float)
    wn = min(bw_hz / (grid.fs / 2.0), 0.99)
    if kind == "brickwall":
        X = np.fft.rfft(x)
        X[np.fft.rfftfreq(len(x)) > wn / 2.0] = 0.0   # rfftfreq's Nyquist is 0.5
        return np.fft.irfft(X, len(x))
    if kind == "gaussian":
        f = np.fft.rfftfreq(len(x))
        H = np.exp(-0.5 * (f / (wn / 2.0 + 1e-12)) ** 2)
        return np.fft.irfft(np.fft.rfft(x) * H, len(x))
    return _sig.sosfiltfilt(_sig.bessel(order, wn, output="sos"), x)


def probe_loading(x, grid, c_load_f=0.5e-12, r_source=50.0):
    """A passive probe's input capacitance LOADS the node it measures — an RC low-pass with a
    pole at 1/(2·pi·R·C) that attenuates high frequency. Real probes perturb the DUT."""
    fc = 1.0 / (2 * np.pi * r_source * c_load_f)
    return scope_bandwidth(x, grid, fc, kind="bessel", order=1)


def timebase_jitter(x, grid, rms_ps=0.5, rng=None):
    """Sample-clock / timebase jitter — each sample is taken at a slightly wrong time, which
    smears the eye HORIZONTALLY. Adds independent per-sample timing error (RMS in ps)."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, float)
    n = len(x)
    dev = (rms_ps * 1e-12 * grid.fs) * rng.standard_normal(n)
    idx = np.arange(n)
    return np.interp(idx - dev, idx, x, left=x[0], right=x[-1])


def sinad_noise_rms(enob, full_scale):
    """The in-band noise+distortion rms that an ENOB figure stands for, at `+/- full_scale`.

    ENOB IS A SINAD FIGURE, NOT A BIT DEPTH. It is defined from a full-scale sine:
    `SINAD_dB = 6.02*ENOB + 1.76`, so the noise it names is `(2*full_scale/2**enob)/sqrt(12)`
    -- the rms a uniform quantiser of that many bits WOULD have had. That equivalence is the
    whole reason ENOB is quoted in bits, and it is also the trap: the equivalent rms is a
    CONTINUOUS noise level, and rounding a record to `2**enob` codes reproduces the level while
    inventing a lattice the converter does not have. See `converter_noise_rms`."""
    return (2.0 * float(full_scale) / 2.0 ** float(enob)) / np.sqrt(12.0)


def converter_noise_rms(enob, full_scale, bandwidth_hz, nyquist_hz, bits=None):
    """The CONVERTER's own wideband noise rms behind a published `enob` measured at
    `bandwidth_hz`, given the converter runs to `nyquist_hz`.

    A converter's noise is (to first order) white across its own Nyquist band; the instrument's
    selected-bandwidth filter sits AFTER it and passes only `bandwidth_hz/nyquist_hz` of that
    power. So one fixed converter floor produces a whole ENOB-vs-bandwidth table:

        enob(B) = enob(B_ref) + 0.5*log2(B_ref/B)

    MEASURED against the Keysight UXR1104A's published table (13 bandwidth/ENOB pairs, 10 to
    110 GHz): anchored at 110 GHz -> 5.0, the other twelve settings come back at **-0.05 to
    +0.31 bits**, rms 0.16 -- one number reproducing twelve. Adding a second, frequency-rising
    noise term only improves the rms to 0.08, which is not enough structure to justify it. The
    filter's SHAPE cancels out entirely (any fixed shape has noise bandwidth proportional to
    its -3 dB point, and the proportionality is absorbed by the anchor), so this needs no
    filter model to be useful.

    `bits` (the converter's real depth) splits the answer honestly: the returned rms is the
    converter's ADDED noise with its own lattice's `q**2/12` taken out, so that
    `quantize_adc(x + noise, bits=bits)` lands on the published ENOB rather than overshooting
    it. It raises if the depth alone cannot reach the requested ENOB.

    For the UXR1104A -- 10 bits, +/-423 mV, ENOB 5.0 at 110 GHz, Nyquist 128 GHz -- this is
    8.23 mV rms, **9.96 LSB of the converter's own lattice**: the lattice carries 0.084 % of
    the noise power that sets ENOB, and would on its own be worth 10.00 bits. That ratio is
    the entire argument for modelling the two mechanisms apart."""
    wide = sinad_noise_rms(enob, full_scale) * np.sqrt(float(nyquist_hz) / float(bandwidth_hz))
    if bits is None:
        return float(wide)
    q = 2.0 * float(full_scale) / 2.0 ** int(bits)
    added = wide ** 2 - q ** 2 / 12.0
    if added <= 0.0:
        raise ValueError(
            f"a {bits}-bit converter cannot reach ENOB {enob} at {bandwidth_hz / 1e9:g} GHz: its own "
            f"lattice already contributes {q / np.sqrt(12.0):.3e} V rms against the {wide:.3e} V rms "
            f"the figure allows")
    return float(np.sqrt(added))


def quantize_adc(x, enob=None, full_scale=None, bits=None):
    """Quantise to an ADC lattice — the one thing a converter does that nothing else does.

      bits        the converter's REAL depth -> exactly 2**bits codes across +/- full_scale.
                  This is the physical lattice: a UXR1104A is 10 bits, full stop.
      enob        legacy: treat an effective-bits figure as if it were a lattice depth.
                  KEPT FOR COMPATIBILITY AND IT IS THE CONFLATION THIS MODULE NOW SEPARATES --
                  ENOB is a SINAD figure (noise AND distortion, continuous); bit depth is a
                  lattice (discrete). Rounding to 2**enob codes gets the noise POWER roughly
                  right and the record's structure entirely wrong: a UXR's 10-bit lattice is
                  9.96 LSB below its own noise (so fully dithered, and invisible in a
                  histogram), while an ENOB-5.9 lattice is 17x coarser than the real one and
                  produces a comb no real capture has. Use `bits` + `converter_noise_rms`.
      full_scale  +/- range of the lattice; None takes it from the signal's peak

    Exactly one of `bits`/`enob`. Returns the quantised array; each sample moves by at most
    half an LSB. Pairs with clip_adc (clip first so out-of-range samples land on the top code,
    not beyond it)."""
    if (bits is None) == (enob is None):
        raise ValueError("quantize_adc needs exactly one of bits= (a converter depth) or "
                         "enob= (the legacy effective-bits lattice)")
    x = np.asarray(x, float)
    fs = float(np.max(np.abs(x))) + 1e-12 if full_scale is None else float(full_scale)
    lsb = 2.0 * fs / 2 ** (int(bits) if bits is not None else enob)
    return np.round(x / lsb) * lsb


def digitize(x, grid=None, interleave=None, clip_full_scale=None, enob=None,
             noise_floor=None, rng=None, bits=None):
    """Compose the ADC stages in the physically correct order and return ``(y, info)``.

    Order — all of it AFTER the channel and the additive impairment: additive noise floor
    -> interleave mismatch (at the sampling instant) -> hard clip (at the ADC input) ->
    quantise (last). Getting this order wrong is silent: quantising before the noise, or
    clipping after quantising, yields a plausible waveform with the wrong noise floor. It
    lives here once so a caller cannot get it wrong.

      noise_floor       kwargs for shaped_noise_floor, e.g. {"rms": 1e-3, "shape": "pink"}
      interleave        kwargs for interleave_adc, e.g. {"m_cores": 4, "offset_v": 1e-3}
      clip_full_scale   hard-clip level (absolute); None to skip. Also sets the quantiser range.
      bits              the converter's REAL depth for the final lattice; None to skip. Pair it
                        with a `noise_floor` sized by `converter_noise_rms` -- that is the honest
                        two-mechanism converter: a discrete lattice at the depth the part has,
                        and a continuous noise that sets the SINAD the data sheet publishes.
      enob              legacy: quantise to a 2**enob lattice instead, which models the two as
                        one thing. See `quantize_adc`.

    ``info`` records the applied settings and the clipped-sample mask fraction — feeding
    provenance (#5) and measured ground truth (#8). ``grid`` is accepted for API symmetry;
    amplitude stages need no rate conversion."""
    x = np.asarray(x, float)
    rng = rng or np.random.default_rng()
    info = {}
    if noise_floor:
        x = x + shaped_noise_floor(len(x), rng=rng, **noise_floor)
        info["noise_floor"] = dict(noise_floor)
    if interleave:
        x = interleave_adc(x, rng=rng, **interleave)
        info["interleave"] = dict(interleave)
    if clip_full_scale is not None:
        x, mask = clip_adc(x, clip_full_scale)
        info["clip_full_scale"] = float(clip_full_scale)
        info["clipped_fraction"] = float(mask.mean())
    if bits is not None and enob is not None:
        raise ValueError("digitize takes bits= (a converter depth) or enob= (the legacy "
                         "effective-bits lattice), not both")
    if bits is not None:
        x = quantize_adc(x, bits=bits, full_scale=clip_full_scale)
        info["bits"] = int(bits)
    elif enob is not None:
        x = quantize_adc(x, enob=enob, full_scale=clip_full_scale)
        info["enob"] = float(enob)
    return x, info


def store_record(x, bits=11, full_scale=None, headroom=1.05, clip=True,
                 dither_lsb=0.0, rng=None):
    """The LAST thing a real-time DSO does before it hands you a file: write integer codes.

    THIS IS AN EXPORT STEP, NOT A NOISE MODEL, AND IT IS WHY REAL RECORDS HAVE A FLOOR
    ---------------------------------------------------------------------------------
    `quantize_adc` models the CONVERTER, which sits in the middle of the chain: a real
    instrument filters, corrects and decimates AFTER its converter, and every one of those
    stages is a weighted sum of many codes, so by the time the record reaches memory the
    converter's lattice is gone. What comes back out of the instrument is on a lattice
    again -- because the record is STORED as int16 codes -- and that second, terminal
    lattice is the record's noise floor at every frequency the signal does not occupy.

    Measured on three real Keysight compliance-suite exports of 8 M samples each: 1,851 /
    1,880 / 2,035 distinct values, 1.0000 of them on the lattice, occupancy 0.997-1.000
    (a full lattice, not a sparse one), and a stop-band PSD floor equal to the lattice's
    own `q**2/12/(fs/2)` to under 0.1 dB in 3 of 3. A rendered record that stops at the
    converter and then filters holds ~7 M distinct values and a stop band at -245 dB/Hz,
    which is float64 numerical zero: 60-90 dB of missing floor, and an amplitude histogram
    5-11x smoother than any real capture's.

      bits        stored code width. `lsb = 2*full_scale/2**bits`.
      full_scale  +/- range the codes span, in the same amplitude units as `x`. None means the
                  vertical was RANGED TO THIS ACQUISITION: `headroom * max|x|`, chosen once for
                  the whole record. That is what an operator does before pressing Run, and it
                  is not the same thing as data-dependent rounding -- the pitch is constant
                  across the record either way, so the floor does not move with the signal
                  within it. Give a number instead to model a fixed vertical setting.
      headroom    how far above the record's peak the vertical sits when `full_scale` is None.
                  The three real captures hold 1,851 / 1,880 / 2,035 codes, so at 11 bits their
                  own ranging left 0.6-10.7 % of headroom; 1.05 is the middle of that.
      clip        clip to the representable code range (a real export cannot store a code
                  it has no room for). Set False to keep an out-of-range sample on-lattice
                  but out-of-range, which is a rendering, not an acquisition.
      dither_lsb  rms, in LSB, of independent noise added immediately BEFORE the rounding.
                  MEASURED, not assumed: the three real captures' stop bands sit at
                  **+2.68 / +3.00 / +3.08 dB above** their own lattice's `q**2/12/(fs/2)`, flat
                  from the instrument's DSP corner to Nyquist, on a PSD whose normalisation is
                  Parseval-exact (integral = variance to 1.0000). A bare rounder cannot produce
                  that: its error power IS `q**2/12`. A rounder whose input carries an extra
                  `q**2/12` of independent noise -- one LSB of RPDF dither, or a fixed-point DSP
                  stage rounding at the same word width just upstream -- gives `q**2/6`, i.e.
                  **+3.01 dB**, in the same place, and matches all three within 0.33 dB.
                  `1/sqrt(12) = 0.2887` is that value. Default 0.0 keeps a bare rounder.

    NOTE ON THE +3 dB. An earlier unit reported these same three captures matching `q**2/12`
    to "under 0.1 dB, 3 of 3". That measurement is off by a factor of two (a one-sided /
    two-sided PSD convention); re-measured with an instrument checked by Parseval AND by
    recovering the closed form on constructed uniform error of a known step to 0.02 dB, the
    real floor is 2.0x the bare-rounding prediction. A model that reproduces `q**2/12` exactly
    is therefore 3 dB SHORT of a real record, not on it.

    A NOTE ON THE "COMB RATIO", because it is the metric most likely to be aimed at here.
    `mean|diff(hist)|/mean(hist)` at 2000 bins is NOT a physics measurement: it is the beat
    between the histogram's bin grid and the record's lattice, and it is a non-monotonic
    function of the distinct-value count almost alone. MEASURED on constructed Gaussian noise
    on a lattice, with no signal physics at all: 1,851 codes -> 0.184, 1,880 -> 0.153,
    2,035 -> 0.057, 2,099 -> 0.115, 2,399 -> 0.346. The three real captures' 0.197 / 0.144 /
    0.091 are recovered by their code counts and nothing else, the metric has a MINIMUM right
    where distinct ~ bins, and dither moves it by under 0.001. So a record reading just under
    the real band is telling you its code count is slightly high, not that its noise is wrong,
    and tuning noise to hit a comb number tunes against an aliasing artefact. Aim at the
    distinct count / occupancy / effective bits instead; the comb follows.

    Returns floats -- the voltage each stored code stands for -- so the record stays in the
    pipeline's units. Every value is an exact multiple of one LSB and, for `bits <= 16`,
    an exactly representable int16 code times that LSB."""
    x = np.asarray(x, float)
    bits = int(bits)
    if bits < 1:
        raise ValueError("store_record needs at least 1 bit of stored code width")
    if full_scale is None:
        full_scale = float(headroom) * float(np.max(np.abs(x)))
        if full_scale <= 0.0:
            return x * 0.0                        # an all-zero record has no range to set
    lsb = 2.0 * float(full_scale) / 2.0 ** bits
    if dither_lsb:
        rng = rng or np.random.default_rng()
        x = x + rng.normal(0.0, float(dither_lsb) * lsb, len(x))
    codes = np.round(x / lsb)
    if clip:
        codes = np.clip(codes, -(2.0 ** (bits - 1)), 2.0 ** (bits - 1) - 1.0)
    return codes * lsb


def quantisation_floor_db_per_hz(lsb, fs_hz):
    """The one-sided PSD of a quantiser's own error, `q**2/12/(fs/2)`, in dB/Hz.

    A uniform quantiser of step `q` contributes `q**2/12` of total noise power spread flat
    over the Nyquist band, so a stored record's stop band cannot sit below this. It is the
    prediction that matched three real captures to under 0.1 dB, and it is how a caller
    checks a rendered record has the floor it claims rather than asserting that it does."""
    return float(10.0 * np.log10((float(lsb) ** 2 / 12.0) / (float(fs_hz) / 2.0)))


def shaped_noise_floor(n, rms=0.01, shape="pink", rng=None):
    """A frequency-shaped noise floor (real front ends are not flat). shape in
    {'white','pink','blue'} — pink ~ 1/sqrt(f), blue ~ sqrt(f). Returns length-n noise
    with the requested RMS. Add to a signal to model a coloured floor."""
    rng = rng or np.random.default_rng()
    w = rng.standard_normal(n)
    if shape != "white":
        W = np.fft.rfft(w)
        f = np.arange(W.shape[0], dtype=float); f[0] = 1.0
        W = W / np.sqrt(f) if shape == "pink" else W * np.sqrt(f)
        w = np.fft.irfft(W, n)
    return (rms / (w.std() + 1e-12)) * w


def clip_adc(x, full_scale=1.0):
    """Hard-clip to +/- full_scale (a real ADC saturates at full scale). Returns
    (clipped_signal, clipped_mask) so downstream tools can exclude saturated samples."""
    x = np.asarray(x, float)
    mask = np.abs(x) >= full_scale
    return np.clip(x, -full_scale, full_scale), mask


# General-purpose alias (issue #51): the analog input-stage bandwidth limit.
input_bandwidth = scope_bandwidth

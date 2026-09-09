"""
wfmsynth.cdr — clock recovery as part of "what the scope records".

An instrument does not show you the raw record; it shows you the record folded by a
**recovered clock**. The CDR is a phase-locked loop that tracks the data timing, so it
tracks OUT timing jitter slower than its loop bandwidth and passes jitter faster than it.
The loop bandwidth and order therefore materially change what an eye looks like — an emitted
eye is only meaningful alongside the recovery that produced it.

`recover_clock` is the standard linearized-PLL jitter transfer. Given a per-symbol timing
phase (the transmitted timing jitter, in any unit — seconds or UI), it returns:

  * ``clock``    the recovered-clock phase — a LOW-pass copy of the input (the loop follows
                 slow wander), corner at ``loop_bw``.
  * ``residual`` the timing error the sampler (and hence the eye) actually sees — a
                 HIGH-pass copy, ``input - clock``. This is the jitter the recorded eye shows.

Order picks the loop type: order 1 (single pole) leaves a static phase error under a
frequency offset; order 2 (type-2, a double DC zero in the residual) tracks a frequency
offset out to zero. `loop_bw` is in Hz, `baud` the symbol rate.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as _sig


def _transfer(loop_bw, order, damping):
    """Analog (clock, residual) transfer functions of a linearized order-N PLL CDR."""
    wn = 2 * np.pi * loop_bw
    if order == 1:
        return ([wn], [1, wn]), ([1, 0], [1, wn])                       # LP1 clock, HP1 residual
    if order == 2:
        clk = ([2 * damping * wn, wn ** 2], [1, 2 * damping * wn, wn ** 2])
        res = ([1, 0, 0], [1, 2 * damping * wn, wn ** 2])               # type-2 HP residual
        return clk, res
    raise ValueError(f"order must be 1 or 2 (got {order!r})")


def ssc_phase(n, fs, f_ssc=32e3, spread=0.005, profile="down"):
    """Spread-spectrum-clocking timing phase: the cumulative clock-timing deviation (in
    SAMPLES) from a triangular ~``f_ssc`` modulation of the clock frequency. SSC is
    near-universal across high-speed serial standards for EMI, and it is a large low-frequency
    wander that a CDR must track. ``spread`` is the fractional frequency deviation (e.g.
    0.005 = 0.5%); ``profile`` is 'down' (0..−spread, the common case), 'up' (0..+spread) or
    'center' (−spread..+spread). Feed to a carrier as jitter, or use `apply_ssc` to warp a
    waveform."""
    k = np.arange(int(n))
    tri = 1.0 - np.abs(2.0 * ((k * f_ssc / fs) % 1.0) - 1.0)     # 0..1..0 triangle, period 1/f_ssc
    if profile == "down":
        dfrac = -spread * tri
    elif profile == "up":
        dfrac = spread * tri
    elif profile == "center":
        dfrac = spread * (2.0 * tri - 1.0)
    else:
        raise ValueError(f"unknown SSC profile {profile!r} (use 'down', 'up' or 'center')")
    return np.cumsum(dfrac)


def apply_ssc(x, fs, f_ssc=32e3, spread=0.005, profile="down"):
    """Embed spread-spectrum clocking in a waveform by warping its time base onto the
    SSC-modulated clock. Spreads the spectrum (the point of SSC) and adds the low-frequency
    wander a downstream CDR has to track."""
    x = np.asarray(x, float)
    n = len(x)
    cum = ssc_phase(n, fs, f_ssc, spread, profile)
    return np.interp(np.arange(n) - cum, np.arange(n), x, left=x[0], right=x[-1])


def recover_and_fold(x, grid, n_blocks=16, levels=4, defn="contour"):
    """Fold a waveform onto the clock a CDR recovers from it, and return the recorded eye
    height — the literal "record folded by the recovered clock". Models the CDR as tracking the
    local optimal sampling phase across ``n_blocks`` (loop bandwidth ~ n_blocks / record
    duration): low-frequency timing jitter is tracked out, so the recorded eye is MORE OPEN than
    a single fixed sampling phase, while high-frequency jitter is not tracked and the two agree.
    An emitted eye is only meaningful alongside the recovery that produced it; this is that
    recovery at the waveform level (the phase-domain jitter transfer is `recover_clock`)."""
    from .measure import eye_height
    blocks = np.array_split(np.asarray(x, float), int(n_blocks))
    eyes = [eye_height(b, grid, levels=levels, defn=defn) for b in blocks if len(b) > levels * 10]
    return float(np.median(eyes)) if eyes else 0.0


def phase_noise(n, grid, rms_ps=1.0, slope=2.0, rng=None):
    """Colored oscillator PHASE NOISE as a timing sequence (in samples) whose power spectral
    density falls as 1/f**``slope``. slope≈2 is a free-running oscillator's close-in
    (−20 dB/dec) profile; real clocks have a spectrum, not the single Pj tone. Scaled to
    ``rms_ps``. This is the phase-noise source consumed by `timing_source`."""
    rng = rng or np.random.default_rng()
    w = rng.standard_normal(int(n))
    W = np.fft.rfft(w)
    f = np.arange(len(W), dtype=float); f[0] = 1.0
    col = np.fft.irfft(W / f ** (slope / 2.0), int(n))
    return (rms_ps * 1e-12 * grid.fs) * col / (col.std() + 1e-12)


def apply_phase(x, phase_samples):
    """Warp a waveform by a per-sample timing deviation (in samples) — the general timing-
    modulation primitive. Feed it a composed phase from `timing_source` (SSC + phase noise +
    wander + Pj/Rj all at once), or any custom timing sequence."""
    x = np.asarray(x, float)
    idx = np.arange(len(x))
    return np.interp(idx - np.asarray(phase_samples, float), idx, x, left=x[0], right=x[-1])


def timing_source(n, grid, ssc=None, pj=None, wander=None, rj_ps=0.0, phase_noise=None, rng=None):
    """Compose a per-sample timing-phase deviation (in SAMPLES) by summing named sources into
    one sequence — the enabler for realistic COMBINED jitter feeding a carrier (`apply_phase`)
    or the CDR (`recover_clock`). Today jitter is parametric Rj/Pj/DCD only; this lets any mix
    of clock effects compose:

      ssc          dict(f_ssc, spread, profile)  -> spread-spectrum wander (`ssc_phase`)
      pj           dict(amp_ps, f_hz)            -> sinusoidal (periodic) jitter
      wander       dict(amp_ps, f_hz)            -> low-frequency wander
      rj_ps        gaussian random jitter, 1-sigma in ps
      phase_noise  dict(rms_ps, slope)           -> colored random jitter (1/f**(slope/2))
    """
    t = np.arange(int(n)) / grid.fs
    ph = np.zeros(int(n))
    if ssc:
        ph += ssc_phase(n, grid.fs, ssc.get("f_ssc", 32e3), ssc.get("spread", 0.005),
                        ssc.get("profile", "down"))
    if pj:
        ph += (pj["amp_ps"] * 1e-12 * grid.fs) * np.sin(2 * np.pi * pj["f_hz"] * t)
    if wander:
        ph += (wander["amp_ps"] * 1e-12 * grid.fs) * np.sin(2 * np.pi * wander["f_hz"] * t)
    if rj_ps:
        rng = rng or np.random.default_rng()
        ph += (rj_ps * 1e-12 * grid.fs) * rng.standard_normal(int(n))
    if phase_noise:
        ph += globals()["phase_noise"](n, grid, phase_noise["rms_ps"],
                                       phase_noise.get("slope", 2.0), rng=rng)
    return ph


def recover_clock(phase, baud, loop_bw, order=2, damping=0.707):
    """Run a CDR over a per-symbol timing ``phase`` sequence. Returns ``(clock, residual)``:
    ``clock`` is the recovered-clock phase (low-pass, the loop follows slow jitter) and
    ``residual = phase - clock`` is the timing error the recorded eye shows (high-pass,
    corner ~ ``loop_bw``). Units of the outputs match ``phase``."""
    phase = np.asarray(phase, float)
    (cn, cd), (en, ed) = _transfer(loop_bw, order, damping)
    bc, ac = _sig.bilinear(cn, cd, fs=baud)
    be, ae = _sig.bilinear(en, ed, fs=baud)
    return _sig.lfilter(bc, ac, phase), _sig.lfilter(be, ae, phase)


def jitter_transfer(phase, baud, loop_bw, order=2, damping=0.707):
    """The residual (eye) timing jitter after clock recovery — i.e. ``recover_clock(...)[1]``.
    This is the jitter an emitted eye actually exhibits; measure/label eyes against it, not
    against the transmitted jitter."""
    return recover_clock(phase, baud, loop_bw, order=order, damping=damping)[1]


def tracked_out_fraction(phase, baud, loop_bw, order=2, damping=0.707, warmup=0.5):
    """How much of the input timing jitter the CDR tracks out: ``1 - ptp(residual)/ptp(phase)``
    over the steady-state tail (after ``warmup`` fraction of the record). ~1 for jitter well
    below the loop bandwidth, ~0 for jitter well above it."""
    res = jitter_transfer(phase, baud, loop_bw, order=order, damping=damping)
    s = int(warmup * len(phase))
    return 1.0 - np.ptp(res[s:]) / (np.ptp(np.asarray(phase, float)[s:]) + 1e-30)


GARDNER_SLOPE = 5.0        # nominal Gardner S-curve slope (error per UI) at unit mean square;
                           # only the loop's SEED gain -- `refine` measures the real one


def _interp1(x, t):
    """Linear interpolation of ``x`` at one fractional sample index (hot inner-loop form)."""
    n = len(x)
    if t <= 0.0:
        return float(x[0])
    if t >= n - 1:
        return float(x[n - 1])
    i = int(t)
    f = t - i
    return float(x[i] + (x[i + 1] - x[i]) * f)


def _gardner_gain(x, instants, spb, step=0.05):
    """The Gardner detector's S-curve SLOPE (error per UI) measured on this record, around
    ``instants``. The loop gains are scaled by it so ``loop_bw`` means the same thing whatever
    the record's amplitude, transition density or rise time — none of which the detector's raw
    output is independent of."""
    xi = np.arange(len(x), dtype=float)
    out = []
    for d in (-step, step):
        t = np.asarray(instants, float) + d * spb
        y = np.interp(t, xi, x)
        ym = np.interp(t - 0.5 * spb, xi, x)
        e = ym * (y - np.roll(y, 1))
        out.append(float(e[1:].mean()))
    return (out[1] - out[0]) / (2.0 * step)


def recover_symbol_instants(x, grid=None, loop_bw_hz=None, spb=None, loop_bw_ui=None,
                            order=2, damping=0.707, phase0=None, n_sym=None, refine=True):
    """The decision instants a receiver's clock recovery actually produces, as FRACTIONAL
    sample indices — one per symbol.

    This is `recover_clock`'s loop closed around the waveform instead of around a phase
    sequence someone already knew. A Gardner timing-error detector reads the sampling error
    off the record (the sample halfway between two decisions is at the transition midpoint
    only when the decisions are centred), and the same second-order loop drives an NCO that
    produces the next instant. Because the loop INTEGRATES, the instants follow a symbol rate
    that moves — which a fixed stride cannot do: a stride quantised to whole samples, or
    matched to a rate that then changes, walks off the symbol centres and never comes back.

    ``order`` carries the same meaning as in `recover_clock`: order 2 (a type-2 loop, the
    default) drives a frequency offset to zero static error, order 1 leaves one. Loop
    bandwidth is given either as ``loop_bw_hz`` with a ``grid`` that has ``baud``, or
    directly as ``loop_bw_ui`` = loop bandwidth / symbol rate. ``phase0`` is the starting
    instant in samples (default: half a symbol, the nominal first eye centre); the loop pulls
    in from any starting phase within about half a symbol.

    ``refine`` re-measures the detector gain around the instants the first pass found and runs
    the loop again with it, so the realized loop bandwidth does not depend on the initial
    amplitude guess.
    """
    x = np.asarray(x, float)
    if spb is None:
        if grid is None or grid.samples_per_ui is None:
            raise ValueError("recover_symbol_instants needs spb= or a Grid with baud set")
        spb = float(grid.samples_per_ui)
    spb = float(spb)
    if loop_bw_ui is None:
        if loop_bw_hz is None:
            raise ValueError("recover_symbol_instants needs loop_bw_hz= (with a Grid) or loop_bw_ui=")
        if grid is None or grid.baud is None:
            raise ValueError("loop_bw_hz= needs a Grid with baud set (or pass loop_bw_ui=)")
        loop_bw_ui = float(loop_bw_hz) / float(grid.baud)
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2 (got {order!r})")
    t0 = 0.5 * spb if phase0 is None else float(phase0)
    if n_sym is None:
        n_sym = int((len(x) - 1 - t0) / spb)
    n_sym = max(0, int(n_sym))
    if n_sym == 0:
        return np.zeros(0)

    # SEED gain. The detector's slope cannot be measured before the loop has locked: on a
    # record whose rate moves, a uniform stride slides through every sampling phase and the
    # measured slope averages to zero (it is periodic in the symbol). So seed from the
    # record's mean square -- GARDNER_SLOPE is a nominal slope for a unit-mean-square signal,
    # not a claim about this record -- and let `refine` replace it with the measured value
    # once there are locked instants to measure around.
    power = float(np.mean(x * x))
    kd = GARDNER_SLOPE * power
    if not np.isfinite(kd) or kd <= 0.0:                 # a flat record: nothing to lock to
        return t0 + np.arange(n_sym, dtype=float) * spb

    wn = 2.0 * np.pi * float(loop_bw_ui)                 # rad per symbol
    tau = None
    for _pass in range(3 if refine else 1):
        kp = 2.0 * damping * wn / kd
        ki = (wn * wn / kd) if order == 2 else 0.0
        tau = np.empty(n_sym)
        t = t0
        integ = 0.0
        yprev = _interp1(x, t - spb)
        for k in range(n_sym):
            tau[k] = t
            y = _interp1(x, t)
            ym = _interp1(x, t - 0.5 * spb)
            e = ym * (y - yprev)
            yprev = y
            integ += ki * e
            t += spb - spb * (kp * e + integ)
        if not refine:
            break
        kd2 = _gardner_gain(x, tau, spb)                 # now measurable: tau is locked
        if not np.isfinite(kd2) or kd2 <= 0.0 or abs(kd2 - kd) <= 1e-3 * kd:
            break
        kd = kd2
    return tau


def sample_at_instants(x, instants):
    """The record's value at fractional decision ``instants`` — the samples a receiver slices."""
    x = np.asarray(x, float)
    return np.interp(np.asarray(instants, float), np.arange(len(x), dtype=float), x)

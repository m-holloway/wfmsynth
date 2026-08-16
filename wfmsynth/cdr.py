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
    near-universal in PCIe/USB/SATA/DisplayPort for EMI, and it is a large low-frequency
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

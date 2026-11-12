"""wfmsynth.recipes -- useful ARRANGEMENTS of ops, as plain functions that return a chain.

THE RULE (docs/ARCHITECTURE.md, "What becomes an op"). An op is one mechanism nothing else
expresses. Anything that can be written as a chain of existing ops is not an op: it lives here,
as a function that returns a `Signal()` sub-chain. Nothing is hidden by that -- `summary()`,
the recipe, `roles()` and `contrast()` show the ops it is made of, and a caller can take the
chain and change one of them. A function here adds no mechanism and no parameter to any op.
Each states what it claims and over what range, and `wfmsynth.validate` measures it.

A CHANNEL, for an I/Q record

    fading        a moving, scattering path's complex gain (Rayleigh, or Rician with a line
                  of sight): complex noise through `filter("doppler")`. For `gain(by=)`.
    multipath     delayed, scaled (and optionally fading) copies of one transmitted envelope,
                  joined by `combine`.

WHAT A CAPTURE CARRIES THAT IS NOT THE SIGNAL, each for `combine`

    flicker       noise with a 1/f spectrum between two stated frequencies: two-state
                  `carrier("states")` processes superposed, their corners spaced evenly in
                  log frequency. `flicker_psd` is the spectrum the arrangement has.
    hum           mains pickup: a sine and its harmonics, and a slow wander of the line
                  frequency that moves every harmonic together (`delay`).
    ripple        a converter's output ripple: a triangle at the switching frequency, a ring
                  at each switching edge (a square through a resonant `filter`), and a
                  spread-spectrum clock (`ssc`).
    pickup        a carrier that is not yours, coupled in at a stated level, plain or
                  modulated.

A CONTACT, each an enable for `gate(by=)` or `gain(by=)`

    bounce        a contact closing: it touches, flies open and touches again, each flight
                  shorter than the last (a `sequence` of levels). `bounce_times` is when.
    intermittent  a contact that opens briefly at random (`carrier("states")`).

A SOURCE

    oscillator_startup   an oscillator coming up: amplitude growing exponentially into its
                  limit, frequency settling from an initial error (`modulate` and `gain`).
    multitone     evenly spaced equal tones with a stated phase policy and an optional notch:
                  a noise power ratio stimulus, on a voltage or on I/Q. `multitone_plan` is
                  its frequencies and phases.
    buck          a synchronous buck converter as a CIRCUIT: a PWM switch node into an
                  inductor, a capacitor and a load, solved by `net`. In volts.
                  `buck_ripple` is the textbook figures it is checked against.

ON A SWEPT LASER'S FIELD (`wfmsynth.swept`), devices and what a bench adds

    mzi           a Mach-Zehnder interferometer: the field and a delayed copy of it,
                  `delay` + `gain` + `combine`. Fringes every 1/delay_s of optical
                  frequency; also the wavelength reference of a sweep.
    stray_reflection   a weak delayed copy added to the field: the fringe a pair of
                  facets or connectors puts across a spectrum.
    ring          a `resonator` stated the way a ring is specified: Q, extinction, free
                  spectral range in nm.
    gas_cell      absorption lines at stated wavelengths, one `resonator` each: an
                  absolute wavelength reference.

    import wfmsynth as ws
    tx = ws.Signal(seed=1, grid=g).iq("qpsk", prbs=11)
    rx = ws.recipes.multipath(tx, delays_s=[0.0, 2e-6, 5e-6], gains=[1.0, 0.5, 0.25],
                              doppler_hz=200.0).awgn(es_n0_db=20)
    print(rx.summary())                                    # every op of every path

    tx.gain(by=ws.recipes.fading(doppler_hz=200.0))        # a unit-power complex gain

    noise = ws.recipes.flicker(f_lo_hz=1e3, f_hi_hz=1e6, rms=0.02)
    ws.Signal(seed=1, grid=g).carrier("sine", f_hz=1e5).combine(noise)

    clutter = [ws.recipes.hum(50.0, amp=0.02, harmonics={3: 0.3}),
               ws.recipes.ripple(250e3, amp=0.01, spike=0.03, f_ring_hz=20e6)]
    ws.Signal(seed=1, grid=g).carrier("sine", f_hz=1e3).combine(clutter)

    ws.recipes.multitone(1e6, 9e6, 33, notch=(4.6e6, 5.4e6), sig=ws.Signal(seed=1, grid=g))

    laser = ws.Signal(seed=1, grid=g).laser(start_nm=1549.0, stop_nm=1551.0)
    dut = ws.recipes.ring(laser, wavelength_nm=1550.0, q=1e5, extinction_db=15.0)
    dut = ws.recipes.stray_reflection(dut, delay_s=40e-12, reflectivity=0.02).photodetect()

A SOURCE takes `sig=`, the Signal to build on. Left out it is `Signal()`, a sub-chain for
another op; given `ws.Signal(seed=1, grid=g)`, the result is the record.

WHAT IS NOT HERE, because it is not a chain of what exists (GitHub #91):

    a contact that opens fast and recovers slowly    needs one draw used twice: the enable
                  and a filtered copy of it, joined by `combine(how="min")`. A sub-chain
                  written twice is drawn twice, so this is a `Channels` record with the
                  enable as a shared chain, not a function that returns one chain.
    an oscillator as a loop                          a resonator, a gain and a limiter in
                  FEEDBACK. A chain has no feedback; `oscillator_startup` is the closed-form
                  solution of that loop's amplitude equation, and says so.
    a capacitor's series resistance in `buck`        `net` solves node voltages and every
                  node needs capacitance, so the node between a capacitor and its series
                  resistance is refused. `ripple` is the ESR-dominated shape, as a waveform.
"""
from __future__ import annotations

import numpy as np

from . import swept as SW
from .compose import Signal


def fading(doppler_hz, k_db=None, los_hz=0.0, shape="jakes"):
    """A FADING GAIN: the complex factor a moving, scattering path puts on an envelope, as a
    chain to hand to `gain(by=)`. Mean power 1.

        sig.gain(by=ws.recipes.fading(doppler_hz=100.0))                # Rayleigh
        sig.gain(by=ws.recipes.fading(doppler_hz=100.0, k_db=6.0))      # Rician, K = 6 dB

    It is three ops, each doing what it says:

        awgn(snr_db=0, bw_hz=2*doppler_hz, ref_power=P)    complex Gaussian scatter, with
                                                           power P in the Doppler band
        filter("doppler", f_hz=doppler_hz, shape=shape)    shaped to the Doppler spectrum;
                                                           the power in that band is kept
        spur(f_hz=los_hz, dbc=0, ref_power=1 - P)          Rician only: the line-of-sight
                                                           path, a steady tone at its own
                                                           Doppler shift `los_hz`

    `doppler_hz` is the largest Doppler shift, speed / wavelength. `k_db` is the Rician K
    factor, line-of-sight power over scattered power: P = 1 / (1 + K). Left out, there is no
    line of sight, P = 1 and |gain| is Rayleigh. The one random role is the scatter's
    (`.../awgn/0`), so `contrast()` re-rolls the fade and nothing else. The record has to hold
    at least 4 Doppler periods (`filter("doppler")` says why)."""
    if not doppler_hz > 0:
        raise ValueError(f"fading: doppler_hz is the largest Doppler shift (speed / "
                         f"wavelength) and must be positive, got {doppler_hz!r}")
    scatter = 1.0 if k_db is None else 1.0 / (1.0 + 10.0 ** (float(k_db) / 10.0))
    if k_db is None and los_hz:
        raise ValueError(f"fading: los_hz={los_hz!r} is the Doppler shift of the "
                         f"line-of-sight path, and without k_db= there is none. State the "
                         f"Rician K factor, k_db=.")
    s = (Signal().awgn(snr_db=0.0, bw_hz=2.0 * float(doppler_hz), ref_power=scatter)
         .filter("doppler", f_hz=float(doppler_hz), shape=shape))
    if k_db is not None:
        s = s.spur(f_hz=float(los_hz), dbc=0.0, ref_power=1.0 - scatter)
    return s


def multipath(tx, delays_s, gains, doppler_hz=None, shape="jakes"):
    """A MULTIPATH channel: the sum of delayed, scaled copies of one transmitted envelope.

        tx = ws.Signal().iq("qpsk", prbs=11)
        ws.recipes.multipath(tx, delays_s=[0.0, 2e-6], gains=[1.0, 0.5])

    Path k is `tx.delay(by=delays_s[k]).gain(by=gains[k])`, and the paths are joined by
    `combine`. On an I/Q record `delay` turns the carrier by -2*pi*fc*delay as well as
    shifting the envelope, so each path's PHASE is its delay's, as on air; a `gains` entry may
    be complex to state a phase on top (a reflection's sign). With `doppler_hz=` every path
    also gets its own `fading(doppler_hz, shape=shape)`, independent of the others, so the
    power delay profile is `abs(gains)**2` on average and each tap is Rayleigh.

    `tx` is the transmitter's chain. Written with the record's seed and grid, the result is
    the record; written as `Signal()`, a sub-chain for another op. It is rendered once per
    path, so it must be DETERMINISTIC -- a copy that drew its own noise would be a different
    signal, not a delayed one -- and a `tx` with random roles is refused. For a transmitter
    with randomness, name it in a `Channels` record and pass `ws.Signal().use(name)`: a named
    chain is the same samples wherever it is used.

    Returns the chain: path 0, then one `combine` of the others. Add the receiver after it."""
    delays_s, gains = list(delays_s), list(gains)
    if not delays_s or len(delays_s) != len(gains):
        raise ValueError(f"multipath: one delay and one gain per path; got {len(delays_s)} "
                         f"delays and {len(gains)} gains.")
    if not isinstance(tx, Signal) or not tx.ops:
        raise ValueError("multipath: tx is the transmitter's chain, a Signal that already "
                         "has its source: ws.Signal(seed=1, grid=g).iq('qpsk', prbs=11).")
    _same_on_every_path(tx, "multipath")

    def path(d, g):
        s = tx.copy()
        if d:
            s = s.delay(by=float(d))
        if doppler_hz is not None:
            s = s.gain(by=fading(doppler_hz, shape=shape))
        if g != 1:
            s = s.gain(by=complex(g) if np.iscomplexobj(g) else float(g))
        return s

    paths = [path(d, g) for d, g in zip(delays_s, gains)]
    return paths[0].combine(paths[1:]) if len(paths) > 1 else paths[0]


def _same_on_every_path(tx, who):
    """Refuse a chain that would not be the same samples each time it is rendered."""
    drawn = tx.roles()
    if drawn and tx.ops[0]["op"] != "use":
        raise ValueError(
            f"{who}: tx draws random roles {drawn}, and it is rendered once per path "
            f"under that path's own names, so each path would carry different noise rather "
            f"than a delayed copy. Put the transmitter in a Channels record as a named chain "
            f"and pass ws.Signal().use(<name>), which is the same samples wherever it is "
            f"used; or move the randomness after the {who}.")


def _corners(f_lo_hz, f_hi_hz, per_decade):
    if not (np.isfinite(f_lo_hz) and np.isfinite(f_hi_hz) and 0 < f_lo_hz < f_hi_hz):
        raise ValueError(f"flicker: the band is 0 < f_lo_hz < f_hi_hz, got {f_lo_hz!r} and "
                         f"{f_hi_hz!r}.")
    if int(per_decade) != per_decade or per_decade < 1:
        raise ValueError(f"flicker: per_decade is a whole number of processes per decade of "
                         f"frequency, at least 1; got {per_decade!r}.")
    m = int(round(per_decade * np.log10(f_hi_hz / f_lo_hz))) + 1
    return np.geomspace(float(f_lo_hz), float(f_hi_hz), max(m, 2))


def flicker(f_lo_hz, f_hi_hz, rms=1.0, per_decade=2, tr_s=None):
    """Noise whose spectrum falls as 1/f between `f_lo_hz` and `f_hi_hz`, as a sub-chain.

    It is what 1/f noise is in a small device: many traps, each a two-state telegraph process
    with its own time constant. One symmetric process with mean dwell ``1 / (pi * fc)`` has a
    Lorentzian spectrum with its corner at `fc`; `per_decade` of them per decade of corner
    frequency, equal in amplitude, sum to 1/f between the lowest and highest corner, flat
    below and 1/f^2 above (`flicker_psd`). The samples are a sum of `M` two-level signals, so
    they are NOT Gaussian: `M + 1` discrete levels, which is the point of building it this
    way. Each process has its own role (`states/0`, `combine/1/others/<k>/states/0`).

    `rms` is the standard deviation of the whole sum. The record has to be long enough to
    hold the slowest process -- several times ``1 / f_lo_hz`` -- and the grid fast enough to
    sample the fastest: its mean dwell ``1 / (pi * f_hi_hz)`` must be at least one sample,
    which `carrier("states")` enforces. `tr_s` is each jump's 10-90 % time."""
    fc = _corners(f_lo_hz, f_hi_hz, per_decade)
    a = float(rms) / np.sqrt(fc.size)
    edge = {} if tr_s is None else {"tr_s": tr_s}
    traps = [Signal().carrier("states", levels=[-a, a], dwell=float(1.0 / (np.pi * f)), **edge)
             for f in fc]
    return traps[0].combine(traps[1:])


def flicker_psd(f_hz, f_lo_hz, f_hi_hz, rms=1.0, per_decade=2):
    """The one-sided power spectral density of `flicker(...)` at `f_hz`, per hertz: the sum of
    its processes' Lorentzians, ``(2 a^2 / (pi fc)) / (1 + (f / fc)^2)`` each. Away from the
    ends of the band it is ``a^2 * per_decade / (ln(10) * f)``. Edge shaping (`tr_s`) and the
    grid take it down further near Nyquist; this is the process itself."""
    fc = _corners(f_lo_hz, f_hi_hz, per_decade)
    a2 = float(rms) ** 2 / fc.size
    f = np.asarray(f_hz, float)[..., None]
    return np.sum((2.0 * a2 / (np.pi * fc)) / (1.0 + (f / fc) ** 2), axis=-1)


def _positive(who, **values):
    for name, v in values.items():
        if isinstance(v, (bool, str)) or not np.isscalar(v) or not np.isfinite(v) or not v > 0:
            raise ValueError(f"{who}: {name} must be a positive number, got {v!r}.")


def _both(who, what, **pair):
    """Two arguments that only mean something together: both given, or neither."""
    given = [v is not None for v in pair.values()]
    if any(given) and not all(given):
        a, b = pair
        raise ValueError(f"{who}: {what} needs both {a}= and {b}=; give both or neither.")
    return all(given)


def _onto(who, sig):
    """The Signal a source builds on: `Signal()` unless one is given, which must be empty."""
    if sig is None:
        return Signal()
    if not isinstance(sig, Signal) or sig.ops:
        raise ValueError(f"{who}: sig is the EMPTY Signal to build on -- "
                         f"ws.Signal(seed=1, grid=g) for a record -- because this is a "
                         f"source, a chain's first op. To add it to a chain that already has "
                         f"one, leave sig out and hand the result to combine().")
    return sig


# ------------------------------------------------------------------ clutter, for `combine`
def hum(f_hz, amp, harmonics=None, phase_rad=0.0, f_drift_hz=None, f_wander_hz=None):
    """MAINS HUM: a sine at the line frequency and its harmonics, as a sub-chain to `combine`.

        sig.combine(ws.recipes.hum(50.0, amp=0.02))
        sig.combine(ws.recipes.hum(60.0, amp=0.02, harmonics={2: 0.5, 3: 0.3},
                                   f_drift_hz=0.05, f_wander_hz=0.2))

    `amp` is the fundamental's amplitude. `harmonics` maps a harmonic NUMBER to its amplitude
    relative to the fundamental: `{3: 0.3, 5: 0.1}` is the odd series of a saturating
    transformer. `phase_rad` is the fundamental's phase at t = 0 and harmonic k gets k times
    it, so changing it slides the whole waveform in time without changing its shape.

    The line frequency is not constant. `f_drift_hz` is the furthest the FUNDAMENTAL moves
    from `f_hz`, and `f_wander_hz` how often it goes there and back. It is one `delay` on the
    summed hum, by a slow sine that starts at zero,

        by = A * (1 - cos(2 pi f_wander_hz t)),     A = f_drift_hz / (2 pi f_wander_hz f_hz)

    which puts harmonic k at ``k * (f_hz - f_drift_hz * sin(2 pi f_wander_hz t))``: every
    harmonic moves by the same FRACTION, as on a real line. The wander is stated, not drawn
    (no role); a random one is a noise sub-chain in that `delay`.

    Ops: `carrier("sine")` [`combine` of one sine per harmonic] [`delay`]."""
    _positive("hum", f_hz=f_hz, amp=amp)
    harmonics = dict(harmonics or {})
    for k, r in harmonics.items():
        if isinstance(k, (bool, str)) or int(k) != k or k < 2:
            raise ValueError(f"hum: harmonics maps a harmonic number (2, 3, ...) to its "
                             f"amplitude relative to the fundamental; got the key {k!r}. The "
                             f"fundamental's own amplitude is amp=.")
        _positive("hum", **{f"harmonics[{k}]": r})
    s = Signal().carrier("sine", f_hz=float(f_hz), amp=float(amp), phase_rad=float(phase_rad))
    if harmonics:
        s = s.combine([Signal().carrier("sine", f_hz=float(k * f_hz), amp=float(amp * r),
                                        phase_rad=float(k * phase_rad))
                       for k, r in sorted(harmonics.items())])
    if _both("hum", "a wandering line frequency", f_drift_hz=f_drift_hz,
             f_wander_hz=f_wander_hz):
        _positive("hum", f_drift_hz=f_drift_hz, f_wander_hz=f_wander_hz)
        a = float(f_drift_hz) / (2.0 * np.pi * float(f_wander_hz) * float(f_hz))
        s = s.delay(by=Signal().carrier("sine", f_hz=float(f_wander_hz), amp=a, offset=a,
                                        phase_rad=-0.5 * np.pi))
    return s


def ripple(f_sw_hz, amp, duty=0.5, spike=None, f_ring_hz=None, q=4.0, spread_pct=None,
           f_spread_hz=None):
    """A switching converter's OUTPUT RIPPLE as a waveform, a sub-chain to `combine`.

        sig.combine(ws.recipes.ripple(500e3, amp=0.01))
        sig.combine(ws.recipes.ripple(500e3, amp=0.01, duty=0.3, spike=0.04, f_ring_hz=60e6,
                                      spread_pct=6.0, f_spread_hz=9e3))

    The ripple is a triangle at `f_sw_hz` of amplitude `amp` (2 * amp peak to peak), rising
    for `duty` of the period: the inductor's ripple current in the output capacitor's series
    resistance, which dominates with an electrolytic or a small ceramic. (An ideal
    capacitor's ripple is the integral of that, and comes from the circuit: `buck`.)

    `spike` and `f_ring_hz` add the ring at each switching edge: the switch node's step
    exciting the loop's parasitic inductance and capacitance. It is a square on the same clock
    through a second-order resonance at `f_ring_hz` of quality factor `q`, so each edge starts

        +/- spike * exp(-pi f_ring_hz t / q) * sin(2 pi f_d t),   f_d = f_ring_hz * sqrt(1 - 1/(4 q**2))

    up at the trough, down at the peak. `spike` is what an ideal step would start; the
    square's own edge is an eighth of a ring period (two samples at least), which takes 3 to
    7 % off it (measured for q from 2 to 8, on grids with 16 to 64 samples per ring period,
    the fewest it should be given).

    `spread_pct` and `f_spread_hz` are a spread-spectrum clock: the waveform's time base is
    swept by a triangle (`ssc`, profile "center"), so the switching frequency goes between
    ``f_sw_hz * (1 - spread_pct / 100)`` and ``f_sw_hz * (1 + spread_pct / 100)`` and back
    `f_spread_hz` times a second, and every line of the ripple becomes a band.

    Nothing is drawn. Ops: `carrier("triangle")` [`combine` of `carrier("square")` ->
    `filter("butter", btype="bandpass", order=1)`] [`ssc`]."""
    _positive("ripple", f_sw_hz=f_sw_hz, amp=amp)
    if not 0.0 < duty < 1.0:
        raise ValueError(f"ripple: duty is the fraction of the period the switch is on, "
                         f"strictly between 0 and 1; got {duty!r}.")
    s = Signal().carrier("triangle", f_hz=float(f_sw_hz), amp=float(amp), symmetry=float(duty))
    if _both("ripple", "the ring at each switching edge", spike=spike, f_ring_hz=f_ring_hz):
        _positive("ripple", spike=spike, f_ring_hz=f_ring_hz)
        if not q > 0.5:
            raise ValueError(f"ripple: q={q!r} does not ring; a resonance oscillates only "
                             f"above q = 0.5. A switching loop is typically 2 to 10.")
        if f_ring_hz < 8.0 * f_sw_hz:
            raise ValueError(f"ripple: f_ring_hz={f_ring_hz!r} is under 8 times f_sw_hz="
                             f"{f_sw_hz!r}, so one edge's ring runs into the next and `spike` "
                             f"is not the amplitude of anything. A loop rings tens to "
                             f"hundreds of times faster than it switches.")
        half = 0.5 / float(q)
        root, damped = float(np.sqrt(1.0 + half * half)), float(np.sqrt(1.0 - half * half))
        ring = (Signal().carrier("square", f_hz=float(f_sw_hz), duty=float(duty),
                                 amp=0.5 * float(spike) * float(q) * damped,
                                 tr_frac=0.125 * float(f_sw_hz) / float(f_ring_hz))
                .filter("butter", btype="bandpass", order=1,
                        f_hz=[float(f_ring_hz) * (root - half), float(f_ring_hz) * (root + half)]))
        s = s.combine(ring)
    if _both("ripple", "a spread-spectrum clock", spread_pct=spread_pct,
             f_spread_hz=f_spread_hz):
        _positive("ripple", spread_pct=spread_pct, f_spread_hz=f_spread_hz)
        s = s.ssc(f_ssc=float(f_spread_hz), spread=0.01 * float(spread_pct), profile="center")
    return s


def pickup(fc_hz, amp, message=None, depth=0.5, dev_hz=None):
    """RADIO PICKUP: a carrier that is not yours, coupled in at a stated level; a sub-chain to
    `combine`.

        sig.combine(ws.recipes.pickup(98.5e6, amp=0.005))                   # a bare carrier
        sig.combine(ws.recipes.pickup(1.0e6, amp=0.01,
                                      message=ws.Signal().carrier("sine", f_hz=1e3)))   # AM
        sig.combine(ws.recipes.pickup(98.5e6, amp=0.005, dev_hz=75e3,
                                      message=ws.Signal().carrier("noise", rms=0.3)
                                                         .filter("butter", f_hz=15e3))) # FM

    `amp` is the carrier's amplitude as it arrives on the record: the coupling is in that
    number, not modelled. `message` is what it is modulated with, a sub-chain or a bare
    carrier spec, at the level it renders. With `dev_hz` it is FM, the instantaneous frequency
    ``fc_hz + dev_hz * message``; without, AM, ``amp * (1 + depth * message) * sin(2 pi fc_hz t)``,
    so a unit sine message puts a line ``depth / 2`` of the carrier either side of it.
    Without a message it is a plain sine. Any randomness is the message's own, under
    `.../modulate/<i>/message/...`.

    Ops: `carrier("sine")` [`modulate(kind="am")`], or `modulate(kind="fm")` alone."""
    _positive("pickup", fc_hz=fc_hz, amp=amp)
    if message is None:
        if dev_hz is not None:
            raise ValueError("pickup: dev_hz= is the frequency deviation per unit of the "
                             "message, and there is no message=.")
        return Signal().carrier("sine", f_hz=float(fc_hz), amp=float(amp))
    if dev_hz is not None:
        return Signal().modulate(kind="fm", fc_hz=float(fc_hz), dev_hz=float(dev_hz),
                                 amp=float(amp), message=message)
    return (Signal().carrier("sine", f_hz=float(fc_hz), amp=float(amp))
            .modulate(kind="am", depth=float(depth), message=message))


# ------------------------------------------------------------------ a contact, as an enable
def bounce_times(t_s, gap_s, n=3, ratio=0.5, touch_s=None):
    """When `bounce(...)` closes and opens: ``(closes_s, opens_s)``, `n + 1` and `n` instants
    in seconds. The ground truth of the arrangement, from its arguments alone."""
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError(f"bounce: n is how many times the contact flies open again, a whole "
                         f"number of at least 1; got {n!r}. A contact that closes once is a "
                         f"carrier('step').")
    if isinstance(t_s, (bool, str)) or not np.isfinite(t_s) or t_s < 0:
        raise ValueError(f"bounce: t_s is when the contact first touches, in seconds from "
                         f"the record's start; got {t_s!r}.")
    touch_s = 0.25 * gap_s if touch_s is None else touch_s
    _positive("bounce", gap_s=gap_s, ratio=ratio, touch_s=touch_s)
    closes, opens, t = [], [], float(t_s)
    for gap in _flights(gap_s, n, ratio):
        closes.append(t)
        opens.append(t + float(touch_s))
        t = opens[-1] + gap
    return np.array(closes + [t]), np.array(opens)


def _flights(gap_s, n, ratio):
    return [float(gap_s) * float(ratio) ** k for k in range(int(n))]


def bounce(t_s, gap_s, n=3, ratio=0.5, touch_s=None):
    """CONTACT BOUNCE on closing, as an enable: 0 while open, 1 while closed.

        sig.gate(by=ws.recipes.bounce(t_s=2e-3, gap_s=400e-6))

    The contact first touches at `t_s`. It then flies open `n` times: the first flight lasts
    `gap_s` and each one after it `ratio` times the one before (a bouncing mass keeps the
    same fraction of its speed at every impact), and before each flight it touches for
    `touch_s`, by default a quarter of `gap_s`. After the last flight it stays closed: `n + 1`
    closures and `n` gaps at stated instants (`bounce_times`), the last closure at

        t_s + n * touch_s + gap_s * (1 - ratio**n) / (1 - ratio).

    It is a `sequence` of constant levels with those durations, so every boundary is a
    `segment` event and the `gate` it drives emits each switching as a `transition`. Each
    duration has to be a whole number of samples: `sequence` never rounds one, and refuses
    with the nearest that fit. Nothing is drawn: for a population, vary the arguments.

    Ops: one `sequence` of 2*n + 2 `carrier("dc")` segments (one fewer when t_s is 0)."""
    bounce_times(t_s, gap_s, n, ratio, touch_s)                      # the refusals
    touch_s = 0.25 * gap_s if touch_s is None else touch_s

    def level(v, duration_s=None):
        seg = {"chain": Signal().carrier("dc", level=v)}
        return seg if duration_s is None else dict(seg, duration_s=float(duration_s))

    segments = [level(0.0, t_s)] if t_s > 0 else []
    for gap in _flights(gap_s, n, ratio):
        segments += [level(1.0, touch_s), level(0.0, gap)]
    return Signal().sequence(segments + [level(1.0)])


def intermittent(closed_s, open_s, tr_s=None):
    """AN INTERMITTENT CONTACT, as an enable: 1 while it holds, 0 while it is open.

        sig.gate(by=ws.recipes.intermittent(closed_s=5e-3, open_s=50e-6))       # hard opens
        sig.gain(by=ws.recipes.intermittent(5e-3, 50e-6, tr_s=10e-6))           # soft ones

    `closed_s` and `open_s` are the two dwells. A number is the MEAN of an exponential (opens
    then arrive as a Poisson process); any dwell spec of `carrier("states")` serves, such as a
    lognormal for opens that are mostly short and now and then long. The record starts
    closed, with a whole dwell. `tr_s` is the 10-90 % time of each change: it matters under
    `gain(by=)`, where the enable is the contact's conductance, while `gate` only reads which
    side of 0.5 it is on.

    One role, `.../states/0`. Each interval is a `state` event (state 0 is closed, 1 open);
    read them with ``ws.states.intervals(sig.event_list(), at="gate/1/by")``. An open shorter
    than the edge does not reach 0 (`wfmsynth.states`).

    Ops: `carrier("states", levels=[1.0, 0.0], dwell=[closed_s, open_s], start=0)`."""
    edge = {} if tr_s is None else {"tr_s": tr_s}
    return Signal().carrier("states", levels=[1.0, 0.0], dwell=[closed_s, open_s], start=0,
                            **edge)


# ------------------------------------------------------------------ sources
def oscillator_startup(f_hz, startup_s, rise_s=None, ppm=0.0, settle_s=None, amp=1.0,
                       sig=None):
    """AN OSCILLATOR COMING UP: its amplitude grows exponentially and bends over into its
    limit, and its frequency starts off by `ppm` and settles.

        ws.recipes.oscillator_startup(10e6, startup_s=200e-6, sig=ws.Signal(seed=1, grid=g))
        ws.recipes.oscillator_startup(10e6, startup_s=200e-6, ppm=300.0, settle_s=100e-6)

    AMPLITUDE. An oscillator is a resonator with gain round it, so a small amplitude grows as
    ``exp(t / tau)`` until the gain compresses: ``da/dt = (a / tau) * (1 - a / amp)``, whose
    solution is the logistic

        a(t) = amp / (1 + exp(-(t - t_half) / tau)).

    `startup_s` is when it reaches 90 % of `amp` and `rise_s` its 10-90 % time (default half
    of `startup_s`): ``tau = rise_s / (2 ln 9)``, ``t_half = startup_s - rise_s / 2``. With
    the default the record starts 57 dB down. A chain has no feedback, so this is that
    equation's SOLUTION written as a chain -- a ramp (`dc`, integrated) through `law("tanh")`
    -- and not the loop.

    FREQUENCY. ``f(t) = f_hz * (1 + ppm * 1e-6 * exp(-t / settle_s))``: off by `ppm` at t = 0
    (a crystal pulled by its drive level, a supply still rising) and settling with the time
    constant `settle_s`. It is FM by a decaying exponential. With ppm = 0 the source is a
    plain `carrier("sine")`.

    Nothing is drawn. Ops: `modulate(kind="fm", message=carrier("exp", decay=True))`, then
    `gain(by=<carrier("dc") -> filter("integrate") -> offset -> law("tanh") -> offset>)`."""
    _positive("oscillator_startup", f_hz=f_hz, startup_s=startup_s, amp=amp)
    rise_s = 0.5 * startup_s if rise_s is None else rise_s
    _positive("oscillator_startup", rise_s=rise_s)
    if rise_s >= 2.0 * startup_s:
        raise ValueError(f"oscillator_startup: rise_s={rise_s!r} is the 10-90 % time and "
                         f"startup_s={startup_s!r} the time to 90 %, so the amplitude would "
                         f"be past half its limit before the record starts. Keep rise_s under "
                         f"2 * startup_s.")
    _both("oscillator_startup", "a frequency that settles", ppm=ppm or None, settle_s=settle_s)
    tau = float(rise_s) / (2.0 * np.log(9.0))
    t_half = float(startup_s) - 0.5 * float(rise_s)
    grow = (Signal().carrier("dc", level=1.0).filter("integrate", tau_s=tau)   # t / tau
            .offset(by=-t_half / tau)                                # u = (t - t_half) / tau
            .law("tanh", gain=0.25, sat=0.5)                         # tanh(u / 2) / 2
            .offset(by=0.5))                                         # 1 / (1 + exp(-u))
    s = _onto("oscillator_startup", sig)
    if ppm:
        _positive("oscillator_startup", settle_s=settle_s)
        s = s.modulate(kind="fm", fc_hz=float(f_hz), dev_hz=float(f_hz) * float(ppm) * 1e-6,
                       amp=float(amp),
                       message=Signal().carrier("exp", decay=True, tau_s=float(settle_s),
                                                t_start_s=0.0))
    else:
        s = s.carrier("sine", f_hz=float(f_hz), amp=float(amp))
    return s.gain(by=grow)


PHASES = ("newman", "schroeder", "random", "zero")


def multitone_plan(f_lo_hz, f_hi_hz, n_tones, phase="newman", notch=None, seed=None):
    """What `multitone(...)` is made of: ``(f_hz, phase_rad, kept)``, one entry per tone of the
    FULL comb; `kept` is False inside the notch. The ground truth of the arrangement."""
    if isinstance(n_tones, bool) or int(n_tones) != n_tones or n_tones < 2:
        raise ValueError(f"multitone: n_tones is a whole number of at least 2, got "
                         f"{n_tones!r}.")
    if not (np.isfinite(f_lo_hz) and np.isfinite(f_hi_hz) and f_lo_hz < f_hi_hz):
        raise ValueError(f"multitone: the tones run from f_lo_hz to f_hi_hz, both included, "
                         f"with f_lo_hz < f_hi_hz; got {f_lo_hz!r} and {f_hi_hz!r}.")
    n = int(n_tones)
    k = np.arange(n, dtype=float)
    if phase != "random" and seed is not None:
        raise ValueError(f"multitone: seed= picks the phases of phase='random'; "
                         f"phase={phase!r} has none to pick.")
    if phase == "newman":
        ph = np.pi * k * k / n
    elif phase == "schroeder":
        ph = -np.pi * k * (k + 1.0) / n
    elif phase == "zero":
        ph = np.zeros(n)
    elif phase == "random":
        if seed is None:
            raise ValueError("multitone: phase='random' needs seed=, a whole number. The "
                             "phases are STATED in the recipe, one per tone, so they are "
                             "picked here and not drawn by the record: another seed is "
                             "another stimulus, and the record's own seed does not move it.")
        ph = np.random.default_rng(int(seed)).uniform(0.0, 2.0 * np.pi, n)
    else:
        raise ValueError(f"multitone: phase is one of {PHASES}, got {phase!r}.")
    f = np.linspace(float(f_lo_hz), float(f_hi_hz), n)
    kept = np.ones(n, bool)
    if notch is not None:
        try:
            lo, hi = (float(v) for v in notch)
        except (TypeError, ValueError):
            raise ValueError(f"multitone: notch is (lo_hz, hi_hz), the band whose tones are "
                             f"left out; got {notch!r}.") from None
        kept = ~((f >= lo) & (f <= hi))
        if kept.all() or not kept.any():
            raise ValueError(f"multitone: notch=({lo!r}, {hi!r}) must hold at least one of "
                             f"the {n} tones and leave at least one; they are "
                             f"{f[1] - f[0]:.6g} Hz apart from {f[0]:.6g} Hz.")
    return f, np.mod(ph, 2.0 * np.pi), kept


def multitone(f_lo_hz, f_hi_hz, n_tones, phase="newman", notch=None, rms=1.0, seed=None,
              domain="real", sig=None):
    """A MULTITONE: `n_tones` equal tones evenly spaced from `f_lo_hz` to `f_hi_hz`.

        ws.recipes.multitone(1e6, 9e6, 33, sig=ws.Signal(seed=1, grid=g))
        ws.recipes.multitone(-8e6, 8e6, 65, notch=(-0.6e6, 0.6e6), domain="iq", sig=...)

    Tone k is ``cos(2 pi f_k t + phase_k)`` on a voltage and ``exp(j (2 pi f_k t + phase_k))``
    on an envelope, and `phase` is what makes the sum usable. Tones in phase (`"zero"`) pile
    up into one spike per period: a crest factor (peak over rms) of sqrt(2 n) on a voltage,
    sqrt(n) on an envelope. A quadratic phase spreads them out:

        "newman"      phase_k = pi * k**2 / n               (k = 0 .. n-1)
        "schroeder"   phase_k = -pi * k * (k + 1) / n
        "random"      uniform in [0, 2 pi), picked by `seed=` and STATED in the recipe

    Newman and Schroeder hold the crest factor under 2.0 on a voltage (1.4 on an envelope)
    from 9 tones to 129, where in-phase tones reach 16. Random phases sit between (2.5 to
    4.5 over twenty seeds at 33 tones) and are what Gaussian noise looks like to an
    amplifier. A random phase is not a role: the record's seed does not move it, `seed=`
    does.

    `notch=(lo_hz, hi_hz)` leaves out the tones in that band and changes nothing else, so
    whatever a device under test puts there is its distortion: the power of a kept tone over
    the power found at a left-out one is the NOISE POWER RATIO. `rms` is that of the FULL
    comb, so each tone keeps its level with or without the notch.

    `domain="real"` is a voltage, one `carrier("sine")` per tone joined by one `combine`;
    `"iq"` an envelope, one `spur` per tone in a row, the frequencies then offsets from the
    centre (negative is below). For the tones to be lines and the notch to be empty, the
    spacing and `f_lo_hz` must be whole multiples of ``fs / n``, the record holding whole
    periods; a sub-chain has no grid, so nothing here can check that.

    COST. One op per tone, and each renders the whole record: measured 0.26 ms a tone on a
    voltage and 0.73 ms on an envelope at 65536 samples, so 1024 tones are 0.3 s and 0.75 s
    and a 130 kB recipe. That is practical; it stops being READABLE long before. Past a few
    hundred tones band-limited noise (`carrier("noise", band_lo_hz=, band_hi_hz=)`, or `awgn`
    and a `filter`) is the plainer stimulus.

    `multitone_plan` returns the frequencies, the phases and which tones are kept."""
    _positive("multitone", rms=rms)
    f, ph, kept = multitone_plan(f_lo_hz, f_hi_hz, n_tones, phase, notch, seed)
    s = _onto("multitone", sig)
    if domain == "iq":
        power = float(rms) ** 2 / f.size
        for fk, pk in zip(f[kept], ph[kept]):
            s = s.spur(f_hz=float(fk), dbc=0.0, ref_power=power, phase_rad=float(pk))
        return s
    if domain != "real":
        raise ValueError(f"multitone: domain is 'real' (a voltage) or 'iq' (an envelope, "
                         f"frequencies about the centre), got {domain!r}.")
    if f[0] <= 0:
        raise ValueError(f"multitone: on a voltage the tones are at positive frequencies, "
                         f"and f_lo_hz={f_lo_hz!r}. Tones either side of a centre are "
                         f"domain='iq'.")
    a = float(rms) * float(np.sqrt(2.0 / f.size))
    tones = [{"f_hz": float(fk), "amp": a, "phase_rad": float(pk + 0.5 * np.pi)}   # a cosine
             for fk, pk in zip(f[kept], ph[kept])]
    s = s.carrier("sine", **tones[0])
    return s.combine([Signal().carrier("sine", **t) for t in tones[1:]]) if tones[1:] else s


def buck_ripple(v_in, duty, f_sw_hz, l_h, c_f, r_load_ohm, r_on_ohm=0.05):
    """The textbook figures of the converter `buck(...)` solves, in continuous conduction:

        v_out_v        duty * v_in * r_load_ohm / (r_load_ohm + r_on_ohm)
        i_ripple_pp_a  v_in * duty * (1 - duty) / (l_h * f_sw_hz)        the inductor's
        v_ripple_pp_v  i_ripple_pp_a / (8 * c_f * f_sw_hz)               the capacitor's
        f0_hz          1 / (2 pi sqrt(l_h * c_f))                        the filter's corner
        settle_s       the time constant of the slowest start-up mode

    The ripple figures take the inductor current for a triangle and the output for constant
    across a period, which holds while `f_sw_hz` is well above `f0_hz`: the circuit agrees
    with them to 0.2 % at 22 times f0 and that grows as (f0 / f_sw)**2."""
    _positive("buck", v_in=v_in, f_sw_hz=f_sw_hz, l_h=l_h, c_f=c_f, r_load_ohm=r_load_ohm,
              r_on_ohm=r_on_ohm)
    if not 0.0 < duty < 1.0:
        raise ValueError(f"buck: duty is the fraction of the period the high side is on, "
                         f"strictly between 0 and 1; got {duty!r}.")
    i_pp = float(v_in) * duty * (1.0 - duty) / (float(l_h) * float(f_sw_hz))
    # the start-up modes: s**2 + s * (1 / (R C) + r / L) + (1 + r / R) / (L C) = 0
    poles = np.roots([1.0, 1.0 / (r_load_ohm * c_f) + r_on_ohm / l_h,
                      (1.0 + r_on_ohm / r_load_ohm) / (l_h * c_f)])
    return {"v_out_v": float(duty * v_in * r_load_ohm / (r_load_ohm + r_on_ohm)),
            "i_ripple_pp_a": float(i_pp),
            "v_ripple_pp_v": float(i_pp / (8.0 * c_f * f_sw_hz)),
            "f0_hz": float(1.0 / (2.0 * np.pi * np.sqrt(l_h * c_f))),
            "settle_s": float(1.0 / np.abs(poles.real).min())}


def buck(v_in, duty, f_sw_hz, l_h, c_f, r_load_ohm, r_on_ohm=0.05, tr_s=20e-9, sig=None):
    """A SYNCHRONOUS BUCK CONVERTER as a circuit: its output voltage, in VOLTS.

        g = ws.Grid(fs=400e6, n=1 << 19)                 # 8 samples on the 20 ns edge
        ws.recipes.buck(v_in=12.0, duty=0.275, f_sw_hz=500e3, l_h=10e-6, c_f=22e-6,
                        r_load_ohm=0.5, sig=ws.Signal(seed=1, grid=g))

    With both switches driven, the switch node is at `v_in` for `duty` of each period and at
    0 for the rest, through the on-resistance `r_on_ohm` either way: one driver whose level is
    a PWM waveform. From there it is an inductor `l_h` to the output, a capacitor `c_f` and
    the load `r_load_ohm` to ground, and `net` solves it exactly -- no behavioural ripple
    model, so the ripple, its shape, the start-up overshoot and the response to anything the
    PWM chain does (change `duty` in a `sequence`, put `timing` on it) are the circuit's.

    Measured against `buck_ripple`: the mean is duty * v_in * r_load / (r_load + r_on) to
    1e-6 and the peak-to-peak ripple v_in * duty * (1 - duty) / (8 l_h c_f f_sw_hz**2) to
    0.2 % (three converters, `wfmsynth.validate`).

    WHAT TO KNOW. `net` starts a record settled at its first sample, and the PWM starts
    half-way through its off time, so the circuit starts at rest -- 0 V, 0 A -- and the
    record opens with the L-C start-up transient, overshoot included; the figures above hold
    once that has died (`buck_ripple(...)["settle_s"]` is its time constant). `tr_s` is the 10-90 % time of the
    driver's edge, and the grid needs 8 samples across it like any other edge. `net` wants a
    capacitance on every node, so the switch node has one, sized to follow that edge:
    ``r_on_ohm * C = tr_s / 2.2``. It does not change the output (the filter's corner is
    four decades below it); it is why the switch node resolves on any grid the edge does. A
    capacitor's series resistance cannot be put in -- the node between the two would have no
    capacitance -- so for ESR-dominated ripple use `ripple`. A diode in place of the low
    switch (discontinuous conduction at light load) is not this circuit either.

    Nothing is drawn. Ops: one `net`, its driver's source `carrier("cmos")`."""
    buck_ripple(v_in, duty, f_sw_hz, l_h, c_f, r_load_ohm, r_on_ohm)      # the refusals
    _positive("buck", tr_s=tr_s)
    pwm = Signal().carrier("cmos", v_lo=0.0, v_hi=float(v_in), duty=float(duty),
                           f_hz=float(f_sw_hz), tr_s=float(tr_s),
                           phase_rad=float(np.pi * (1.0 + duty)))    # mid-way through "off"
    return _onto("buck", sig).net(
        nodes=["sw", "out"],
        elements=[{"id": "c_sw", "kind": "c", "a": "sw", "b": "gnd", "farad": float(tr_s) / (2.2 * float(r_on_ohm))},
                  {"id": "l", "kind": "l", "a": "sw", "b": "out", "henry": float(l_h)},
                  {"id": "c", "kind": "c", "a": "out", "b": "gnd", "farad": float(c_f)},
                  {"id": "load", "kind": "r", "a": "out", "b": "gnd", "ohm": float(r_load_ohm)}],
        drivers=[{"id": "switch", "node": "sw", "ohm": float(r_on_ohm), "state": "driving",
                  "source": {"ops": pwm, "volts": True}}],
        probe="out")


# ------------------------------------------------------------------ on a laser's field

def _field(tx, who):
    if not isinstance(tx, Signal) or not tx.ops:
        raise ValueError(
            f"{who}: its first argument is the chain the light arrives on, a Signal that "
            f"already has its source: ws.Signal(seed=1, grid=g).laser(start_nm=1549.0, "
            f"stop_nm=1551.0), or ws.Signal().use(<name>) of a laser in a Channels record.")
    return tx


def _two_paths(tx, delay_s, direct, delayed, who):
    """``direct * E(t) + delayed * E(t - delay_s)``: the chain, and a copy of it through
    `delay` and `gain`, joined by `combine`."""
    _same_on_every_path(_field(tx, who), who)
    if not float(delay_s) > 0:
        raise ValueError(f"{who}: delay_s is how much longer the second path takes, in "
                         f"seconds, and is positive; got {delay_s!r}. A path difference "
                         f"dL in a medium of group index ng is ng * dL / c.")
    out = tx.copy() if direct == 1.0 else tx.copy().gain(by=float(direct))
    return out.combine([tx.copy().delay(by=float(delay_s)).gain(by=float(delayed))])


def mzi(tx, delay_s, extinction_db=None):
    """A MACH-ZEHNDER INTERFEROMETER with arms `delay_s` apart, one output port.

        laser = ws.Signal(seed=1, grid=g).laser(start_nm=1549.0, stop_nm=1551.0)
        ws.recipes.mzi(laser, delay_s=50e-12).photodetect(shot=False)

    The chain is `tx.gain(by=a)` combined with `tx.delay(by=delay_s).gain(by=b)`. On a
    laser's field `delay` turns the carrier by its integral over the delay, so the output
    power is ``a^2 + b^2 + 2ab*cos(2*pi*nu*delay_s)``: fringes ``1/delay_s`` apart in optical
    frequency (50 ps is 20 GHz, 0.16 nm at 1550 nm), beating in time at (sweep rate) x
    delay_s. Ideal couplers give a = b = 1/2: power from 1 down to 0. `extinction_db` is the
    ratio of the brightest to the darkest fringe, 20*log10((a + b)/(a - b)) with a + b = 1,
    for arms that are not balanced. Used as a sweep's wavelength reference, the fringe count
    is the optical frequency: `ws.measure.swept_map`.

    A passive interferometer has a delay, not a V-pi; a driven one is `eo(kind="mzm")`."""
    rho = 0.0 if extinction_db is None else 10.0 ** (-float(extinction_db) / 20.0)
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"mzi: extinction_db={extinction_db!r} is the brightest fringe over "
                         f"the darkest, in dB, and is positive (leave it out for none).")
    return _two_paths(tx, delay_s, 0.5 * (1.0 + rho), 0.5 * (1.0 - rho), "mzi")


def stray_reflection(tx, delay_s, reflectivity):
    """A STRAY REFLECTION: a weak copy of the field, `delay_s` late, added to it.

        ws.recipes.stray_reflection(dut, delay_s=40e-12, reflectivity=0.02)

    Two facets, two connectors or a facet and a detector window send a little light round
    twice; it arrives `delay_s` later (twice the gap's transit time) and interferes. The
    chain is `tx` combined with `tx.delay(by=delay_s).gain(by=reflectivity)`, so
    `reflectivity` is the FIELD ratio of the twice-reflected light (the product of the two
    amplitude reflection coefficients; the same number `optical_mpi` takes), and the power
    ripples by ``1 + r^2 + 2r*cos(2*pi*nu*delay_s)``: a fringe of period ``1/delay_s`` in
    optical frequency and 4r peak to peak. This is the first round trip only: right while r
    is small, and a cavity with strong mirrors is a `resonator`."""
    if not 0.0 < float(reflectivity) < 1.0:
        raise ValueError(f"stray_reflection: reflectivity={reflectivity!r} is the field "
                         f"ratio of the reflected copy, between 0 and 1.")
    return _two_paths(tx, delay_s, 1.0, reflectivity, "stray_reflection")


def _plain(who, **values):
    for k, v in values.items():
        if isinstance(v, dict):
            raise ValueError(
                f"{who}: {k}= is converted into the resonator's own numbers here, so it "
                f"takes a number. For a part that varies from unit to unit put the "
                f"tolerance where the number ends up: tx.resonator(wavelength_nm=, "
                f"intrinsic_hz=ws.tol(...), coupling_hz=ws.tol(...)).")


def ring(tx, wavelength_nm, q, extinction_db=None, fsr_nm=None, coupling="under",
         split_hz=None, by=None):
    """A RING RESONATOR on a bus waveguide, stated as a ring is specified.

        ws.recipes.ring(laser, wavelength_nm=1550.0, q=1e5, extinction_db=15.0, fsr_nm=0.8)

    It is one `resonator` op, with these numbers turned into its own:

        linewidth   nu0 / q                            intrinsic_hz + coupling_hz
        extinction  ((ki - ke)/(ki + ke))**2 = 10**(-extinction_db/10); left out, the ring
                    is critically coupled (ki = ke) and the dip reaches zero
        coupling    "under" (ke < ki, the usual case) or "over" (ke > ki): the same dip,
                    told apart only by phase
        fsr_nm      the next resonance sits at wavelength_nm + fsr_nm; the others follow
                    evenly in optical FREQUENCY (no dispersion)

    `q` is the LOADED Q, the one a sweep measures. `split_hz` and `by` are the
    resonator's: a backscatter doublet, and a resonance that moves during the sweep."""
    _plain("ring", wavelength_nm=wavelength_nm, q=q, extinction_db=extinction_db,
           fsr_nm=fsr_nm)
    if coupling not in ("under", "over"):
        raise ValueError(f"ring: coupling is 'under' (the bus takes less than the ring "
                         f"loses) or 'over'; got {coupling!r}.")
    if not (float(q) > 0 and float(wavelength_nm) > 0):
        raise ValueError(f"ring: wavelength_nm={wavelength_nm!r} and q={q!r} are positive.")
    width = float(SW.hz(wavelength_nm)) / float(q)
    s = 0.0 if extinction_db is None else 10.0 ** (-float(extinction_db) / 20.0)
    if not 0.0 <= s < 1.0:
        raise ValueError(f"ring: extinction_db={extinction_db!r} is how far the dip goes "
                         f"down, in dB, and is positive (leave it out for a dip to zero).")
    ke = 0.5 * width * (1.0 - s if coupling == "under" else 1.0 + s)
    extra = {}
    if fsr_nm is not None:
        extra["fsr_hz"] = float(SW.hz(wavelength_nm) - SW.hz(wavelength_nm + float(fsr_nm)))
    if split_hz is not None:
        extra["split_hz"] = float(split_hz)
    return _field(tx, "ring").copy().resonator(
        wavelength_nm=float(wavelength_nm), intrinsic_hz=width - ke, coupling_hz=ke, by=by,
        **extra)


def gas_cell(tx, lines_nm, depth=0.5, width_hz=1e9):
    """A GAS CELL: absorption lines at stated wavelengths, the absolute reference of a sweep.

        ws.recipes.gas_cell(laser, lines_nm=[1549.73, 1550.52], depth=0.4, width_hz=2e9)

    One `resonator` per line, undercoupled so that its dip is a Lorentzian `width_hz` wide
    (full width at half depth) taking the fraction `depth` of the power at its centre --
    what a pressure-broadened line does to the light, by the same equation. A line is at
    its wavelength whatever the laser does, so the time a sweep crosses one anchors the map
    a fringe count gives (`ws.measure.swept_map(at_s=, wavelength_nm=)`)."""
    _plain("gas_cell", depth=depth, width_hz=width_hz)
    if not (0.0 < float(depth) < 1.0 and float(width_hz) > 0 and len(lines_nm)):
        raise ValueError(f"gas_cell: depth={depth!r} is the fraction of the power a line "
                         f"takes at its centre (between 0 and 1), width_hz={width_hz!r} its "
                         f"full width, and lines_nm at least one wavelength.")
    ke = 0.5 * float(width_hz) * (1.0 - np.sqrt(1.0 - float(depth)))
    out = _field(tx, "gas_cell").copy()
    for lam in lines_nm:
        out = out.resonator(wavelength_nm=float(lam), intrinsic_hz=float(width_hz) - ke,
                            coupling_hz=float(ke))
    return out

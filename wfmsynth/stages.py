"""
wfmsynth.stages -- two general stages, for a shape that has no named op.

    sig.law("comparator", threshold=1.65, hysteresis=0.2, v_lo=0.0, v_hi=3.3)
    sig.filter("butter", f_hz=20e6, order=3)

`law` is a TRANSFER LAW: what comes out as a function of what goes in, sample by sample.
`filter` is a STATED LINEAR FILTER: one design, named, with its corner in hertz. Both are named
kinds with parameters and never callables, so the recipe stays data and replays with no extra
files.

WHICH IS WHICH. The named ops stay canonical, and these two are not a second way to spell them:

    a transmitter's compression and level noise      nonlinearity()
    the converter (noise, ENOB, interleave, lattice)  digitize(), store()
    the instrument's band limit                       scope(), probe()
    an equaliser                                      ctle(), tx_ffe(), rx_ffe(), dfe()
    a coupling capacitor                              ac_couple()
    a gate-controlled switch and its body diode       pass_fet()

Reach for `law` / `filter` when the shape is a comparator, a rectifier, a dead zone, a
Chebyshev anti-alias filter, an integrator -- one more shape, with no op of its own.

WHAT EACH LAW IS (x in, y out, in the units the chain carries; `diode` needs volts)

    tanh        y = sat * tanh(gain * x / sat)            small-signal slope `gain`, rails +/-sat
    poly        y = sum(coeffs[k] * x**k)                 ascending powers
    limiter     y = x held inside [lo, hi]                either side may be omitted
    deadzone    y = 0 within width/2 of `center`, and x moved in by width/2 outside it
    rectify     y = max(x, 0) for mode="half", |x| for mode="full"
    quantize    2**bits codes from `lo` in steps of (hi - lo) / 2**bits, rounding to the nearest
                and SATURATING at the first and last code (the top code is hi - 1 step)
    comparator  two output levels, switching up above threshold + hysteresis/2 and down below
                threshold - hysteresis/2. The ONE law with state: the output it is holding.
    diode       a Shockley diode in series with `r_load_ohm`; y is the load voltage. Exact
                (Lambert W), so the knee is the exponential's and not a fixed drop.

Every law but `comparator` is memoryless. The comparator's state before the record is decided
by the first sample; only when that sample lies inside the hysteresis band does `initial` (or,
by default, which side of `threshold` it is on) decide instead. Its transitions are GROUND
TRUTH and are emitted as events with the sub-sample time of the crossing -- see `apply_law`.

WHAT EACH FILTER IS

    butter, bessel, cheby1, elliptic   the classic designs, `f_hz` and `order`, `btype` one of
                low / high / bandpass / bandstop (two corners for the band types). `f_hz` is the
                -3 dB point for butter and bessel and the passband EDGE (-ripple_db) for cheby1
                and elliptic. Designed on the grid by the bilinear transform, prewarped so the
                stated corner is the realised corner.
    rrc         root-raised-cosine FIR: `beta`, `span_ui` symbols long (16), at `baud`. Unit DC
                gain. The same pulse, names and default length as the `iq` source's, so on an
                I/Q record `filter("rrc", beta=<the source's>)` is its matched filter.
    fir         your taps, as given.
    sos         your DIGITAL second-order sections, for this grid's sample rate.
    zpk         your ANALOG zeros, poles (rad/s, s-plane) and gain, moved onto the grid by the
                bilinear transform: H_grid(f) = H_analog(j * 2 fs * tan(pi f / fs)).
    integrate   the running integral from the record's first sample, trapezoid rule, / tau_s.
    differentiate   the backward difference, * tau_s.

CAUSAL BY DEFAULT. A causal filter delays, and `describe_filter` / `Signal.filter_delays()`
REPORT that group delay so an event time upstream of the filter can be corrected instead of
guessed. `zero_phase=True` is the opt-in, it is recorded in the recipe, and it applies the SAME
magnitude with the phase removed -- not a forward-backward pass, which squares the magnitude
and halves the bandwidth that was asked for (the defect `instrument.scope_bandwidth` documents).

A causal filter starts SETTLED at the record's first sample, as the analog front end does: the
line is taken to have been at x[0] for ever, so a record does not begin with a turn-on
transient the signal never had. `integrate` starts at zero, which is its definition.

REFUSALS. A corner at or above Nyquist, a pole outside the unit circle, a design that double
precision cannot realise at the stated corner, an RRC whose band does not fit under Nyquist, a
zero-phase request for a filter that has no zero-phase form, a law handed a complex field, a
parameter a kind does not read: each is refused by name with what to do instead.
"""
from __future__ import annotations

import difflib

import numpy as np
from scipy import fft as _fft
from scipy import signal as _sig
from scipy import special as _special

from . import iq as _iq
from .opkeys import SAID_AS

_REQ = object()            # a parameter with no honest default

_LAW = {
    "tanh": {"gain": 1.0, "sat": 1.0},
    "poly": {"coeffs": _REQ},
    "limiter": {"lo": None, "hi": None},
    "deadzone": {"width": _REQ, "center": 0.0},
    "rectify": {"mode": _REQ},
    "quantize": {"bits": _REQ, "lo": _REQ, "hi": _REQ},
    "comparator": {"threshold": 0.0, "hysteresis": 0.0, "v_lo": 0.0, "v_hi": 1.0,
                   "initial": None},
    "diode": {"r_load_ohm": _REQ, "i_sat": 1e-12, "ideality": 1.0, "v_thermal": 0.025852},
}
_IIR = {"f_hz": _REQ, "order": 4, "btype": "low", "zero_phase": False}
_FILTER = {
    "butter": dict(_IIR),
    "bessel": dict(_IIR),
    "cheby1": {**_IIR, "ripple_db": _REQ},
    "elliptic": {**_IIR, "ripple_db": _REQ, "stop_db": _REQ},
    "rrc": {"beta": _REQ, "baud": None, "span_ui": _iq.RRC_SPAN_UI, "zero_phase": False},
    "fir": {"taps": _REQ, "zero_phase": False},
    "sos": {"sos": _REQ, "zero_phase": False},
    "zpk": {"poles": _REQ, "zeros": (), "gain": 1.0, "zero_phase": False},
    "integrate": {"tau_s": 1.0},
    "differentiate": {"tau_s": 1.0},
}
LAW_KINDS = tuple(_LAW)
FILTER_KINDS = tuple(_FILTER)
# The parameter names each op accepts, for `compose` and the generated `opkeys` table.
LAW_KEYS = tuple(sorted({"kind"} | {k for v in _LAW.values() for k in v}))
FILTER_KEYS = tuple(sorted({"kind"} | {k for v in _FILTER.values() for k in v}))

# A shape that HAS a named op: say so rather than offering a second spelling of it.
_NAMED_ELSEWHERE = {
    "gaussian": "the instrument's front end: scope(kind='gaussian', bw_hz=...)",
    "brickwall": "the instrument's digital band limit: scope(kind='brickwall', bw_hz=...)",
    "ctle": "the receiver's equaliser: ctle(fz_ghz=..., fp1_ghz=..., fp2_ghz=...)",
    "ffe": "a feed-forward equaliser: tx_ffe(taps) or rx_ffe(taps)",
    "ac_couple": "a coupling capacitor: ac_couple(fc_hz=...)",
    "compression": "a transmitter's compression: nonlinearity(compression=...)",
    "adc": "the converter: digitize(bits=...); law('quantize') is the bare staircase",
}
_NOT_READ = {
    "zero_phase": "an integrator or a differentiator is its 90-degree phase, so there is no "
                  "zero-phase form of it. `integrate` has no group delay to remove; "
                  "`differentiate` delays by half a sample, which filter_delays() reports.",
}
ZERO_PHASE_MAX = 1 << 22     # samples of impulse response a zero-phase render will pad for
DESIGN_TOL = 1e-3            # a designed corner gain must be realised to this, relative


def plain(v):
    """A parameter as recipe DATA: arrays become lists, a complex number becomes [re, im]."""
    if isinstance(v, dict):
        return {k: plain(u) for k, u in v.items()}
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [plain(u) for u in v]
    if isinstance(v, (complex, np.complexfloating)):
        return [float(v.real), float(v.imag)] if v.imag else float(v.real)
    if isinstance(v, np.generic):
        return v.item()
    return v


def _resolve(table, what, kind, params):
    """The parameters of one kind, defaults filled in -- refusing an unknown kind, a parameter
    this kind does not read, and a missing parameter that has no honest default."""
    if kind not in table:
        near = difflib.get_close_matches(str(kind), list(table), n=2, cutoff=0.6)
        raise ValueError(
            f"{what}: unknown kind {kind!r}"
            + (f" -- that is {_NAMED_ELSEWHERE[kind]}" if kind in _NAMED_ELSEWHERE else "")
            + (f" -- did you mean {' or '.join(repr(k) for k in near)}?" if near else "")
            + f". {what} kinds: {', '.join(table)}.")
    spec = table[kind]
    for k in params:
        if k not in spec:
            owners = [o for o, s in table.items() if k in s]
            near = difflib.get_close_matches(k, list(spec), n=1, cutoff=0.6)
            raise TypeError(
                f"{what}({kind!r}) does not read {k!r}"
                + (f" -- {what} says {SAID_AS[what, k]}" if (what, k) in SAID_AS
                   else f" ({_NOT_READ[k]})" if k in _NOT_READ
                   else f" (that is a parameter of {', '.join(repr(o) for o in owners)})"
                   if owners else f" -- did you mean {near[0]!r}?" if near else "")
                + f". {what}({kind!r}) takes: {', '.join(spec)}.")
    missing = [k for k, d in spec.items() if d is _REQ and params.get(k) is None]
    if missing:
        raise TypeError(f"{what}({kind!r}) needs {', '.join(missing)}: there is no default "
                        f"that would be right for a signal it has not seen.")
    return {k: (params[k] if k in params else d) for k, d in spec.items()}


def check_law(kind, **params):
    """Validate a law's kind and parameters without applying it (what `Signal.law` calls)."""
    return _law_params(kind, params)


def check_filter(kind, **params):
    """Validate a filter's kind and parameters, as far as they can be without a sample rate."""
    return _resolve(_FILTER, "filter", kind, params)


# ------------------------------------------------------------------------------------ law
def _law_params(kind, params):
    q = _resolve(_LAW, "law", kind, params)
    if kind == "tanh" and not q["sat"] > 0:
        raise ValueError(f"law('tanh'): sat is the level the output saturates at and must be "
                         f"positive, got {q['sat']!r}")
    if kind == "poly":
        c = np.atleast_1d(np.asarray(q["coeffs"], float))
        if c.ndim != 1 or c.size == 0 or not np.all(np.isfinite(c)):
            raise ValueError("law('poly'): coeffs is a non-empty list of finite numbers in "
                             "ASCENDING powers, [c0, c1, c2, ...] for c0 + c1*x + c2*x**2 ...")
        q["coeffs"] = c
    if kind == "limiter":
        if q["lo"] is None and q["hi"] is None:
            raise TypeError("law('limiter') needs lo=, hi= or both: a limiter with no limit "
                            "would be recorded in the recipe and change nothing.")
        if q["lo"] is not None and q["hi"] is not None and not q["lo"] < q["hi"]:
            raise ValueError(f"law('limiter'): lo={q['lo']!r} must be below hi={q['hi']!r}")
    if kind == "deadzone" and not q["width"] > 0:
        raise ValueError(f"law('deadzone'): width is the full width of the zone and must be "
                         f"positive, got {q['width']!r}")
    if kind == "rectify" and q["mode"] not in ("half", "full"):
        raise ValueError(f"law('rectify'): mode is 'half' (max(x, 0)) or 'full' (|x|), got "
                         f"{q['mode']!r}")
    if kind == "quantize":
        if int(q["bits"]) != q["bits"] or not 1 <= q["bits"] <= 32:
            raise ValueError(f"law('quantize'): bits is a whole number of bits from 1 to 32, "
                             f"got {q['bits']!r}")
        if not q["lo"] < q["hi"]:
            raise ValueError(f"law('quantize'): lo={q['lo']!r} must be below hi={q['hi']!r}")
    if kind == "comparator":
        if not q["hysteresis"] >= 0:
            raise ValueError(f"law('comparator'): hysteresis is the full width between the two "
                             f"trip points and cannot be negative, got {q['hysteresis']!r}")
        if q["initial"] not in (None, "lo", "hi"):
            raise ValueError(f"law('comparator'): initial is 'lo' or 'hi' -- the output held "
                             f"before the record when its first sample lies inside the "
                             f"hysteresis band -- got {q['initial']!r}")
    if kind == "diode":
        for k in ("r_load_ohm", "i_sat", "ideality", "v_thermal"):
            if not q[k] > 0:
                raise ValueError(f"law('diode'): {k} must be positive, got {q[k]!r}")
    return q


def _diode(x, r, i_sat, a):
    # v_in = v_d + v_out, v_out = r * i_sat * (exp(v_d / a) - 1). With w = (v_out + r*i_sat)/a
    # that is w + ln(w) = L, whose solution is w = W(exp(L)). exp(L) overflows near 18 V of
    # forward drive at room temperature, so above L = 30 the same equation is solved by Newton
    # from its own asymptote L - ln(L) (quadratic; converged to the last bit in 4 steps).
    rs = r * i_sat
    L = np.log(rs / a) + (x + rs) / a
    w = np.empty_like(L)
    small = L < 30.0
    w[small] = _special.lambertw(np.exp(L[small])).real
    Lb = L[~small]
    wb = Lb - np.log(Lb)
    for _ in range(6):
        wb = wb - (wb + np.log(wb) - Lb) / (1.0 + 1.0 / wb)
    w[~small] = wb
    return a * w - rs


def schmitt(x, up, dn, initial=None):
    """THE hysteresis state machine, for everything that commits to a crossing: the output
    goes high when ``x > up``, low when ``x < dn``, and HOLDS in between (a sample exactly on
    a trip point has not crossed it). Returns ``(state, i)``: the state at every sample, and
    the index of the first sample in each new state.

    `initial` is the state held before the input first commits: True / False, or None for
    "whatever it commits to first", so the record does not open with a transition.

    `law("comparator")` and `trigger` both run this and differ only in the INSTANT they
    report for a transition, each stating its own: the comparator the time the input reached
    the trip point it crossed (`up` or `dn`: when its output switches); the trigger the last
    time the input crossed the level between them (where a scope puts t = 0)."""
    n = len(x)
    cmd = np.where(x > up, 1, np.where(x < dn, 0, -1))        # -1: inside the band, hold
    last = np.maximum.accumulate(np.where(cmd >= 0, np.arange(n), -1))
    if initial is None:
        first = int(np.argmax(last >= 0))                     # 0 when it never commits: no edge
        state = (cmd[np.maximum(last, first)] == 1)
    else:
        state = np.where(last >= 0, cmd[np.maximum(last, 0)], int(bool(initial))).astype(bool)
    return state, np.flatnonzero(state[1:] != state[:-1]) + 1


def _comparator(x, q, fs, events, placement):
    v_lo, v_hi = float(q["v_lo"]), float(q["v_hi"])
    up_at = q["threshold"] + 0.5 * q["hysteresis"]
    dn_at = q["threshold"] - 0.5 * q["hysteresis"]
    first = (q["initial"] == "hi") if q["initial"] is not None else bool(x[0] > q["threshold"])
    state, changes = schmitt(x, up_at, dn_at, initial=first)
    if events is not None:
        from .events import Event
        for i in changes:
            rising = bool(state[i])
            trip = up_at if rising else dn_at
            frac = (trip - x[i - 1]) / (x[i] - x[i - 1])     # where between the two samples
            events.append(Event(
                sample=int(i), kind="transition", severity=0.0,
                t_s=None if fs is None else float((i - 1 + frac) / fs), width=1,
                direction=1.0 if rising else -1.0, placement=placement,
                params={"trip": float(trip), "level": v_hi if rising else v_lo}))
    return np.where(state, v_hi, v_lo)


def apply_law(x, kind, fs=None, events=None, placement="law", **params):
    """Apply one named transfer law to a real waveform. See the module docstring for each.

    `comparator` is the bridge from an analog signal to a digital one, and its transitions are
    ground truth: pass a list as `events` and each transition is appended as an
    `events.Event(kind="transition")` whose `sample` is the first sample at the new level,
    `direction` is +1 rising / -1 falling, `params` carry the trip level crossed and the level
    switched to, and `t_s` (given `fs`) is the time the INPUT crossed the trip level, linearly
    interpolated between the two samples either side. With hysteresis that is the edge of the
    band, not `threshold`: a `trigger` on the same input reports the crossing of its level
    instead, which is earlier (`schmitt` states both). The output itself is an ideal step on the
    sample grid, so its edge is quantised to a sample and the event's `t_s` is not: read the
    time from the event. In a chain these arrive through `Signal.realize()` / `event_list()`.
    """
    q = _law_params(kind, params)
    x = np.asarray(x)
    if np.iscomplexobj(x):
        raise ValueError(
            f"law({kind!r}): the waveform here is complex (an optical field or a baseband "
            f"envelope) and a transfer law is defined on a real voltage. Detect it first "
            f"(photodetect()), or apply the law to the real and imaginary parts as two chains; "
            f"taking one part silently would drop the other.")
    x = x.astype(float)
    if x.size == 0:
        raise ValueError(f"law({kind!r}): the waveform is empty")
    if kind == "tanh":
        return q["sat"] * np.tanh(q["gain"] * x / q["sat"])
    if kind == "poly":
        return np.polynomial.polynomial.polyval(x, q["coeffs"])
    if kind == "limiter":
        return np.clip(x, q["lo"], q["hi"])
    if kind == "deadzone":
        d = x - q["center"]
        return q["center"] + np.sign(d) * np.maximum(np.abs(d) - 0.5 * q["width"], 0.0)
    if kind == "rectify":
        return np.maximum(x, 0.0) if q["mode"] == "half" else np.abs(x)
    if kind == "quantize":
        codes = 2 ** int(q["bits"])
        step = (q["hi"] - q["lo"]) / codes
        return q["lo"] + step * np.clip(np.round((x - q["lo"]) / step), 0, codes - 1)
    if kind == "comparator":
        return _comparator(x, q, fs, events, placement)
    return _diode(x, q["r_load_ohm"], q["i_sat"], q["ideality"] * q["v_thermal"])


def window_events(events, window, fs):
    """Events realised on a lead-in render, re-expressed on the DELIVERED window: those inside
    it, with `sample` and `t_s` measured from the window's first sample."""
    out = []
    for e in events:
        if window.start <= e.sample < window.stop:
            e.sample -= window.start
            if e.t_s is not None:
                e.t_s -= window.start / fs
            out.append(e)
    return out


# --------------------------------------------------------------------------------- filter
def _corners(kind, q, fs):
    band = q["btype"] in ("bandpass", "bandstop")
    if q["btype"] not in ("low", "high", "bandpass", "bandstop"):
        raise ValueError(f"filter({kind!r}): btype is low, high, bandpass or bandstop, got "
                         f"{q['btype']!r}")
    f = [float(v) for v in np.atleast_1d(q["f_hz"])]
    if len(f) != (2 if band else 1) or (band and not f[0] < f[1]):
        raise ValueError(
            f"filter({kind!r}, btype={q['btype']!r}): f_hz is "
            + ("the two band edges [f_lo, f_hi] in hertz, lowest first" if band
               else "one corner in hertz") + f", got {q['f_hz']!r}")
    for v in f:
        if not 0.0 < v < 0.5 * fs:
            raise ValueError(
                f"filter({kind!r}): the corner {v:.6g} Hz is not resolved on this grid, whose "
                f"Nyquist frequency is {0.5 * fs:.6g} Hz. Raise Grid(fs=) above "
                f"{2 * v:.6g} Hz, or lower the corner; it is not moved for you.")
    if int(q["order"]) != q["order"] or q["order"] < 1:
        raise ValueError(f"filter({kind!r}): order is a whole number >= 1, got {q['order']!r}")
    return f, band


def _sos_poles(sos):
    return np.concatenate([np.roots(s[3:]) for s in sos]) if len(sos) else np.zeros(0)


def _check_sos(sos, what, fix):
    sos = np.asarray(sos, float)
    if sos.ndim != 2 or sos.shape[1] != 6 or sos.shape[0] == 0 or not np.all(np.isfinite(sos)):
        raise ValueError(f"{what}: second-order sections are rows of six finite numbers "
                         f"[b0, b1, b2, a0, a1, a2], got shape {sos.shape}")
    if not np.all(sos[:, 3] == 1.0):
        raise ValueError(f"{what}: every section's a0 must be 1 (divide the row through by it)")
    worst = float(np.max(np.abs(_sos_poles(sos)), initial=0.0))
    if worst >= 1.0:
        raise ValueError(f"{what}: unstable -- a pole at |z| = {worst:.9g}, on or outside the "
                         f"unit circle, so the output would grow without bound. {fix}")
    return sos


def _gain_at(sos, f_hz, fs):
    with np.errstate(all="ignore"):           # a gain that is not finite is reported, not warned
        return np.abs(_sig.sosfreqz(sos, worN=2 * np.pi * np.atleast_1d(f_hz) / fs)[1])


def _design_iir(kind, q, fs):
    f, band = _corners(kind, q, fs)
    wn = [v / (fs / 2.0) for v in f]          # the expression the front end uses, to the bit
    wn = wn if band else wn[0]
    n, bt = int(q["order"]), q["btype"]
    if kind == "butter":
        sos, edge = _sig.butter(n, wn, bt, output="sos"), np.sqrt(0.5)
    elif kind == "bessel":
        sos, edge = _sig.bessel(n, wn, bt, output="sos", norm="mag"), np.sqrt(0.5)
    else:
        rp = float(q["ripple_db"])
        if not rp > 0:
            raise ValueError(f"filter({kind!r}): ripple_db is the passband ripple in dB and "
                             f"must be positive, got {rp!r}")
        edge = 10.0 ** (-rp / 20.0)
        if kind == "cheby1":
            sos = _sig.cheby1(n, rp, wn, bt, output="sos")
        else:
            rs = float(q["stop_db"])
            if not rs > rp:
                raise ValueError(f"filter('elliptic'): stop_db ({rs!r}) is the stopband "
                                 f"attenuation in dB and must exceed ripple_db ({rp!r})")
            sos = _sig.ellip(n, rp, rs, wn, bt, output="sos")
    fix = (f"filter({kind!r}, order={n}) at {f} Hz is not realisable in double precision on a "
           f"{fs:.6g} Hz grid. Lower the order, raise the corner, or lower Grid(fs=).")
    sos = _check_sos(sos, f"filter({kind!r})", fix)
    # What the coefficients REALISE, against what the design states: the gain at each corner,
    # and the gain of the flat end of the passband (the level a settled start is settled to).
    # A corner far enough below the sample rate puts the poles within rounding of z = 1, and
    # then the sections no longer hold the design -- the passband gain goes first.
    at = {"low": 0.0, "bandstop": 0.0, "high": 0.5 * fs}.get(bt)
    flat = 1.0 if kind in ("butter", "bessel") or n % 2 else edge
    want = np.array([edge] * len(f) + ([] if at is None else [flat]))
    got = _gain_at(sos, f + ([] if at is None else [at]), fs)
    if not np.all(np.abs(got / want - 1.0) < DESIGN_TOL):
        raise ValueError(f"unresolved design: the gain came out {got.tolist()} at "
                         f"{f + ([] if at is None else [at])} Hz against the {want.tolist()} "
                         f"this design states. {fix}")
    return sos


def _rrc_taps(q, fs, baud):
    beta, span = float(q["beta"]), q["span_ui"]
    baud = q["baud"] if q["baud"] is not None else baud
    if baud is None:
        raise TypeError("filter('rrc') needs the symbol rate: pass baud=, or put one on the "
                        "grid (Grid(fs=..., baud=...)).")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"filter('rrc'): beta is the roll-off, 0 to 1, got {beta!r}")
    if not span >= 1:
        raise ValueError(f"filter('rrc'): span_ui is the filter's length in symbols, >= 1, got "
                         f"{span!r}")
    sps = fs / float(baud)
    if sps < 1.0 + beta:
        raise ValueError(
            f"filter('rrc'): the band reaches (1 + beta) * baud / 2 = "
            f"{0.5 * (1 + beta) * baud:.6g} Hz, above this grid's Nyquist frequency "
            f"{0.5 * fs:.6g} Hz ({sps:.3g} samples per symbol; it needs {1 + beta:.3g}). "
            f"Raise Grid(fs=) or lower beta.")
    half = int(np.ceil(0.5 * span * sps))
    h = _iq.rrc(np.arange(-half, half + 1) / sps, beta)      # the pulse the `iq` source shapes with
    return h / h.sum()


def _points(v, what):
    """s-plane points from recipe data: each a number or an [re, im] pair."""
    out = []
    for p in v:
        if isinstance(p, (list, tuple)):
            if len(p) != 2:
                raise ValueError(f"filter('zpk'): each of {what} is a number or an [re, im] "
                                 f"pair in rad/s, got {p!r}")
            p = complex(p[0], p[1])
        out.append(complex(p))
    return np.array(out, complex)


def _design_zpk(q, fs):
    z, p, k = _points(q["zeros"], "zeros"), _points(q["poles"], "poles"), float(q["gain"])
    if len(p) == 0 or len(z) > len(p):
        raise ValueError(f"filter('zpk'): {len(z)} zeros over {len(p)} poles is not a proper "
                         f"transfer function (its gain would rise without limit with "
                         f"frequency). Give at least as many poles as zeros.")
    if np.any(p.real >= 0):
        raise ValueError(f"filter('zpk'): unstable -- the pole(s) {p[p.real >= 0].tolist()} "
                         f"are not in the left half of the s-plane. A decaying pole has a "
                         f"NEGATIVE real part: a 1 MHz real pole is -2*pi*1e6.")
    for name, pts in (("zeros", z), ("poles", p)):
        poly = np.atleast_1d(np.poly(pts))
        if np.abs(poly.imag).max() > 1e-9 * np.abs(poly).max():
            raise ValueError(f"filter('zpk'): the complex {name} do not come in conjugate "
                             f"pairs, so the filter's output would be complex. Give each "
                             f"complex one with its conjugate.")
        far = pts[np.abs(pts) >= np.pi * fs]
        if far.size:
            raise ValueError(
                f"filter('zpk'): the {name} at {(np.abs(far) / (2 * np.pi)).tolist()} Hz are "
                f"not resolved on this grid (Nyquist {0.5 * fs:.6g} Hz); the bilinear transform "
                f"would fold them to the band edge. Raise Grid(fs=), or drop a pole that far "
                f"out -- it does nothing to the band the record holds.")
    sos = _check_sos(_sig.zpk2sos(*_sig.bilinear_zpk(z, p, k, fs)), "filter('zpk')",
                     "The analog poles are stable; it is this grid that rounds them onto the "
                     "unit circle. Lower Grid(fs=) or raise the pole.")
    dc = float(abs(k * np.prod(-z) / np.prod(-p)))        # H(0) of what was stated
    got = float(_gain_at(sos, 0.0, fs)[0])
    if dc > 0 and not abs(got / dc - 1.0) < DESIGN_TOL:
        raise ValueError(
            f"filter('zpk'): unresolved -- the DC gain came out {got!r} "
            f"against the {dc!r} these zeros, poles and gain state. A pole at "
            f"{np.abs(p).min() / (2 * np.pi):.6g} Hz is within rounding of z = 1 on a "
            f"{fs:.6g} Hz grid; lower Grid(fs=) or raise the pole.")
    return sos


def design_filter(fs, kind, grid_baud=None, **params):
    """Design one filter on a grid (`grid_baud` is the symbol rate an `rrc` falls back to when
    it is not given its own `baud=`). Returns `{"form": "sos" | "fir" | "integrate", ...}` with
    `sos` or `taps` holding the coefficients that will be applied, and `zero_phase`.
    Raises, by name, for a design that is unstable or unresolved on the grid."""
    q = _resolve(_FILTER, "filter", kind, params)
    fs = float(fs)
    out = {"kind": kind, "zero_phase": bool(q.get("zero_phase", False))}
    if kind in ("butter", "bessel", "cheby1", "elliptic"):
        return {**out, "form": "sos", "sos": _design_iir(kind, q, fs)}
    if kind == "sos":
        return {**out, "form": "sos", "sos": _check_sos(
            q["sos"], "filter('sos')", "For a pole AT z = 1 use filter('integrate').")}
    if kind == "zpk":
        return {**out, "form": "sos", "sos": _design_zpk(q, fs)}
    if kind in ("integrate", "differentiate"):
        tau = float(q["tau_s"])
        if not tau > 0:
            raise ValueError(f"filter({kind!r}): tau_s is a time constant in seconds and must "
                             f"be positive, got {tau!r}")
        if kind == "integrate":
            return {**out, "form": "integrate", "scale": 1.0 / (fs * tau)}
        return {**out, "form": "fir", "taps": np.array([fs * tau, -fs * tau])}
    if kind == "rrc":
        taps = _rrc_taps(q, fs, grid_baud)
    else:
        taps = np.atleast_1d(np.asarray(q["taps"], float))
        if taps.ndim != 1 or taps.size == 0 or not np.all(np.isfinite(taps)):
            raise ValueError("filter('fir'): taps is a non-empty list of finite real numbers")
    if out["zero_phase"] and (len(taps) % 2 == 0 or not np.array_equal(taps, taps[::-1])):
        raise ValueError(
            f"filter({kind!r}, zero_phase=True): only a SYMMETRIC, odd-length FIR has a "
            f"zero-phase form (its delay is a whole number of samples that can be taken back "
            f"out); these {len(taps)} taps are not. Apply it causally and correct times by the "
            f"delay filter_delays() reports.")
    return {**out, "form": "fir", "taps": taps}


def _sections(d):
    if d["form"] == "fir":
        return [(d["taps"], np.ones(1))]
    return [(s[:3], s[3:]) for s in d["sos"]]


def _response(d, w):
    """Complex response of a designed sos/fir filter at `w` rad/sample."""
    z = np.exp(-1j * np.atleast_1d(w))[:, None]
    h = np.ones(len(z), complex)
    for b, a in _sections(d):
        h *= (b * z ** np.arange(len(b))).sum(1) / (a * z ** np.arange(len(a))).sum(1)
    return h


def _group_delay(d, w):
    """Group delay in SAMPLES at `w` rad/sample: -d(arg H)/dw, in closed form per section."""
    z = np.exp(-1j * w)
    tau = 0.0
    for b, a in _sections(d):
        kb, ka = np.arange(len(b)), np.arange(len(a))
        tau += ((kb * b * z ** kb).sum() / (b * z ** kb).sum()).real
        tau -= ((ka * a * z ** ka).sum() / (a * z ** ka).sum()).real
    return float(tau)


def _step_delay(d, gd):
    """How much later the designed filter delivers the MIDPOINT of an instantaneous edge, in
    samples: where its own step response first reaches half its final value, measured (not a
    formula -- it depends on the whole response). `gd` is its group delay at DC, which sets
    how long a step has to be run. None when the response never gets there."""
    n = int(16 * max(gd, 4.0)) + 64
    step = np.ones(n)
    y = (_sig.sosfilt(d["sos"], step) if d["form"] == "sos" else _sig.lfilter(d["taps"], 1.0, step))
    half = 0.5 * float(np.abs(_response(d, 0.0))[0]) * np.sign(_response(d, 0.0)[0].real)
    over = np.flatnonzero(y / half >= 1.0) if half else np.zeros(0, int)
    if not len(over):
        return None
    j = int(over[0])
    below = y[j - 1] if j else 0.0
    # the step sits half a sample before its first 1: between the last 0 and the first 1
    return float(j - 1 + (half - below) / (y[j] - below) + 0.5)


def describe_filter(fs, kind, at_hz=None, grid_baud=None, **params):
    """What a filter does to TIME:
    `{"kind", "zero_phase", "group_delay_s", "at_hz", "step_delay_s"}`.

    `group_delay_s` is how much later the filter delivers a narrow-band feature at `at_hz`,
    and -- at 0 Hz, for a filter that passes DC -- anything SLOW against the filter: a ramp,
    an envelope, an edge several time constants long. A group delay is a function of
    frequency, so `at_hz` is always stated: by default 0 Hz where the filter passes DC,
    otherwise the frequency where its gain peaks (the centre of a band-pass). Pass `at_hz=`
    to ask elsewhere.

    `step_delay_s` is how much later it delivers the midpoint of an edge FAST against the
    filter: the 50 % point of its own step response, measured. The two differ (a first-order
    low-pass: 1/(2 pi fc) against ln(2) of that), so an upstream event's time is corrected by
    the one that matches the edge that carried it; between the regimes the truth is between
    them. None where a step's midpoint is not defined: a filter that blocks DC, `integrate`,
    `differentiate`.

    Both are zero for a zero-phase filter, by construction."""
    d = design_filter(fs, kind, grid_baud=grid_baud, **params)
    fs = float(fs)
    out = {"kind": kind, "zero_phase": d["zero_phase"]}
    if d["form"] == "integrate" or d["zero_phase"]:
        return {**out, "group_delay_s": 0.0, "at_hz": 0.0 if at_hz is None else float(at_hz),
                "step_delay_s": 0.0 if d["zero_phase"] else None}
    w = np.linspace(0.0, np.pi, 4097)
    mag = np.abs(_response(d, w))
    passes_dc = mag[0] >= 0.5 * mag.max()
    if at_hz is None:
        at_hz = 0.0 if passes_dc else float(w[np.argmax(mag)] * fs / (2 * np.pi))
    step = _step_delay(d, _group_delay(d, 0.0)) if passes_dc else None
    return {**out, "group_delay_s": _group_delay(d, 2 * np.pi * float(at_hz) / fs) / fs,
            "at_hz": float(at_hz), "step_delay_s": None if step is None else step / fs}


def _impulse_extent(sos):
    """Samples until a stable filter's impulse response has fallen 200 dB below its peak."""
    m = 1024
    while True:
        h = np.abs(_sig.sosfilt(sos, np.r_[1.0, np.zeros(m - 1)]))
        live = np.flatnonzero(h > 1e-10 * h.max())
        if live[-1] < m // 2:
            return int(live[-1]) + 1
        if m >= ZERO_PHASE_MAX:
            raise ValueError(
                f"filter(zero_phase=True): this filter's impulse response is still ringing "
                f"after {m} samples, and a zero-phase render needs it to end. Its corner is "
                f"too low for this grid to hold both sides of the response; apply it causally.")
        m *= 2


def _magnitude_only(x, d):
    # The design's magnitude with no phase, as a LINEAR convolution: the record is extended by
    # the response's own extent on each side, holding its end samples (the line was already
    # there, and stays), so nothing wraps from one end of the record onto the other.
    sos = d["sos"]
    pad = _impulse_extent(sos)
    n = len(x)
    nfft = _fft.next_fast_len(n + 2 * pad, real=True)
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(nfft - n - pad, x[-1])])
    mag = np.abs(_sig.sosfreqz(sos, worN=2 * np.pi * _fft.rfftfreq(nfft))[1])

    def one(v):
        return _fft.irfft(_fft.rfft(v) * mag, nfft)[pad:pad + n]
    return one(xp.real) + 1j * one(xp.imag) if np.iscomplexobj(xp) else one(xp)


def apply_filter(x, fs, kind, grid_baud=None, **params):
    """Apply one stated linear filter to a waveform (real, or complex: a real-coefficient
    filter acts on both parts alike). See the module docstring for each kind, for what
    `zero_phase` means, and for how the record's first sample is treated."""
    d = design_filter(fs, kind, grid_baud=grid_baud, **params)
    x = np.asarray(x)
    x = x if np.iscomplexobj(x) else x.astype(float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"filter({kind!r}): the waveform must be one non-empty record")
    if d["form"] == "integrate":
        return d["scale"] * np.concatenate([x[:1] * 0.0, np.cumsum(0.5 * (x[1:] + x[:-1]))])
    if d["form"] == "sos":
        if d["zero_phase"]:
            return _magnitude_only(x, d)
        try:
            with np.errstate(all="ignore"):
                zi = _sig.sosfilt_zi(d["sos"])
        except np.linalg.LinAlgError:
            zi = np.nan
        if not np.all(np.isfinite(zi)):
            raise ValueError(
                f"filter({kind!r}): these sections have no settled state to start from -- a "
                f"pole is within rounding of z = 1, so their DC gain is not a finite number. "
                f"Raise the corner, lower Grid(fs=), or use filter('integrate') for a true "
                f"integrator.")
        y, _ = _sig.sosfilt(d["sos"], x, zi=zi * x[0])
        return y
    taps = d["taps"]
    if d["zero_phase"]:
        half = len(taps) // 2
        return np.convolve(np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])]),
                           taps, mode="valid")
    if len(taps) == 1:
        return taps[0] * x
    y, _ = _sig.lfilter(taps, 1.0, x, zi=_sig.lfilter_zi(taps, np.ones(1)) * x[0])
    return y

"""
wfmsynth.joins -- where a second signal meets the chain.

Two families, both sample by sample on the record's own grid.

COMBINE puts PEER signals together:

    combine(others=[...], how="sum")       x + g1*o1 + g2*o2 + ...     superposition
    combine(others=[...], how="product")   x * g1*o1 * g2*o2 * ...     an ideal mixer or gate
    combine(others=[...], how="min"|"max") the lowest / highest of x, g1*o1, ...   an ideal
                                           wired-AND / diode-OR, with no loading

A SIGNAL-DRIVEN PARAMETER is one stage whose single parameter `by` is a number or another signal:

    gain(by=)     x * by                   a gain riding on a supply ripple
    offset(by=)   x + by                   a baseline following a thermal drift
    gate(by=)     x where by > threshold   an enable from somewhere else
    delay(by=)    x(t - by(t))             a path whose length moves: jitter, Doppler

ON AN I/Q RECORD (`wfmsynth.domain`) three of these keep their meaning: `gain` and `gate`, whose
`by` stays a REAL signal acting on both rails alike, and `combine(how="sum")`, whose `others`
are then I/Q signals about the same centre frequency. The rest are refused there by name.

WHERE A SUM STOPS BEING TRUE. Adding two waveforms is superposition, and superposition holds
exactly when neither source changes what the other one does: a coupled ripple, an interferer, a
tone on a supply, two currents into one node. It does not hold for two DRIVERS on one node, each
with its own output impedance, because each loads the other and the node voltage is a divider
between them, not their sum. That case is `net`, which solves it. `combine` cannot tell the two
apart from samples, so it does not try: the line is the caller's, and it is this one.

WHICH NAME IS CANONICAL. `gain(by=chain)` and `combine([chain], how="product")` are the same
samples, as are `offset(by=chain)` and `combine([chain], how="sum")`: say `gain`/`offset` when
the other signal is a property of THIS one (its supply, its temperature) and `combine` when it
is a second signal in its own right. The older named ops stay, unchanged, as the shorthand for
the physical effect they name:

    crosstalk    a COUPLED sum: the aggressor goes through a coupling response first. Use it
                 for coupling; `combine(how="sum")` adds the waveform as it stands.
    modulate     AM/ASK/OOK is `combine(how="product")` with the message scaled by `depth`
                 (and 1 added unless `suppressed`); FM/PM have no combine form. Use `modulate`
                 for a modulation, `combine` for any other product.
    burst        `gate` with a periodic cadence and shaped on/off edges.
    switch       the SOURCE that chooses among several chains by a signal, with a shaped
                 change (`wfmsynth.sequence`). `gate` is the stage for the two-way case where
                 the other way is nothing: the same samples and instants as
                 `switch(select=enable, inputs=[zero, chain])`, and the same default threshold.
    drift        `gain` or `offset` by a ramp or half-sine across the record.
    timing       `delay` by a named jitter source (SSC, Pj, wander, Rj, phase noise). `timing`
                 may advance as well as delay and interpolates linearly; `delay` is causal and
                 band-limited and states its error.

THE DELAY is a variable fractional delay. Each output sample is the band-limited reconstruction
of the input at `t - by(t)`, through a 32-tap Kaiser-windowed sinc normalised to unit DC gain.
What it promises, all measured on the kernel itself by `delay_error`:

  - the error for a tone at `f` is at most `delay_error(f, fs)` times its amplitude, whatever
    the delay is doing: 2.3e-5 up to 0.4 of the sample rate (`DELAY_BAND`), rising to 1 at
    Nyquist. `Signal.delay_errors()` states it for the grid in hand.
  - it is CAUSAL: `by` is never negative. A line cannot return a sample before it went in.
  - `by` changes by less than `max_rate` seconds per second from one sample to the next
    (default 0.5, always below 1). At 1 the output stands still and beyond it the input plays
    backwards; below that a tone at `f` comes out anywhere up to `f * (1 + max_rate)`, which
    has to stay under Nyquist.
  - before the record starts the line holds the record's first sample, so the first `max(by)`
    seconds are that value; within 16 samples of either end the kernel sees the held edge.
"""
from __future__ import annotations

import functools

import numpy as np

HOWS = ("sum", "product", "min", "max")

DELAY_TAPS = 32                 # kernel length, samples
DELAY_BETA = 10.0               # Kaiser shape: 2.3e-5 to 0.4 fs, the least of 8.5..10.5
DELAY_BAND = 0.4                # fraction of the sample rate the stated error covers
DELAY_MAX_RATE = 0.5            # default bound on |d(by)/dt|, seconds per second
_CHUNK = 1 << 15                # output samples interpolated at a time (bounds the memory)


def _real(x, who):
    if np.iscomplexobj(x):
        raise ValueError(f"{who} acts on a real waveform and this one is complex; taking it "
                         f"would drop the imaginary part. Convert first (`upconvert` for I/Q, "
                         f"`photodetect` for an optical field).")
    return np.asarray(x, float)


def _envelope_ok(x):
    """A waveform that may be a complex envelope: `gain`, `gate` and a `combine` sum mean on
    I/Q what they mean on a voltage (`wfmsynth.domain`)."""
    return np.asarray(x) if np.iscomplexobj(x) else np.asarray(x, float)


def check_domain(how, domain):
    """`combine` on an I/Q record is a sum only."""
    if domain == "iq" and how != "sum":
        raise ValueError(
            f"combine: how={how!r} on an I/Q record. Envelopes about the same centre ADD (an "
            f"interferer, a blocker): how='sum'. A product of two envelopes is not a mixer "
            f"(`upconvert`/`downconvert` are, and `gain(by=)` scales an envelope by a real "
            f"signal), and complex samples have no order for min/max.")


def _same(x, y, who, peer=False):
    """`y` on `x`'s grid. A real waveform, unless it is a `peer` of a complex `x`: then an
    envelope like it."""
    if peer and np.iscomplexobj(x):
        if not np.iscomplexobj(y):
            raise ValueError(f"{who} is a real waveform and the record it joins is I/Q. A "
                             f"voltage cannot be added to an envelope: give an I/Q signal "
                             f"about the same centre frequency.")
        y = np.asarray(y)
    else:
        y = _real(y, who)
    if y.shape != x.shape:
        raise ValueError(f"{who} has {y.shape[0] if y.ndim == 1 else y.shape} samples and the "
                         f"waveform it joins has {x.shape[0]}. They must be on the same grid; "
                         f"nothing is resampled or truncated to fit.")
    return y


def check_combine(how="sum", gains=None, n_others=None):
    """Refuse a `combine` that cannot mean anything, by name. Returns the gains as floats."""
    if how not in HOWS:
        raise ValueError(f"combine: how={how!r} is not one of {HOWS}. A coupled sum is "
                         f"`crosstalk`, a modulation is `modulate`, and two drivers on one "
                         f"node is `net`.")
    if n_others is not None and n_others < 1:
        raise ValueError("combine: `others` is empty, so there is nothing to combine with. "
                         "Give at least one sub-chain.")
    if gains is None:
        return None
    if isinstance(gains, (str, bytes)) or not hasattr(gains, "__len__"):
        raise ValueError(f"combine: gains is a list with one number per entry of `others`, "
                         f"got {gains!r}. To scale the chain itself put `gain(by=)` before "
                         f"the combine.")
    g = [float(v) for v in gains]
    if not all(np.isfinite(g)):
        raise ValueError(f"combine: gains must be finite numbers, got {list(gains)!r}.")
    if n_others is not None and len(g) != n_others:
        raise ValueError(f"combine: {len(g)} gains for {n_others} entries of `others`. Give "
                         f"one gain per entry (the chain itself is taken as it stands; scale "
                         f"it with `gain(by=)` before the combine).")
    return g


def combine(x, others, how="sum", gains=None):
    """`x` and every waveform in `others`, joined sample by sample.

    `gains` scales each entry of `others` before the join (default 1 each); `x` is taken as it
    stands. See the module docstring for where a sum is physics and where it needs `net`."""
    x = _envelope_ok(x)
    check_domain(how, "iq" if np.iscomplexobj(x) else "real")
    gains = check_combine(how, gains, len(others))
    y = x
    for k, o in enumerate(others):
        o = _same(x, o, f"combine: others[{k}]", peer=True)
        if gains is not None and gains[k] != 1.0:
            o = gains[k] * o
        y = (y + o if how == "sum" else y * o if how == "product"
             else np.minimum(y, o) if how == "min" else np.maximum(y, o))
    return y


def _by(x, by, who):
    """`by` as a float or an array on `x`'s grid."""
    if np.ndim(by) == 0:
        by = float(by)
        if not np.isfinite(by):
            raise ValueError(f"{who}: by={by!r} is not a finite number.")
        return by
    return _same(x, by, f"{who}: `by`")


def gain(x, by):
    """`x * by`: `by` is the factor itself (1 leaves `x` alone), a number or a waveform."""
    x = _envelope_ok(x)
    return x * _by(x, by, "gain")


def offset(x, by):
    """`x + by`, in the waveform's own units: a number or a waveform."""
    x = _real(x, "offset")
    return x + _by(x, by, "offset")


def check_gate(threshold=0.5, off="zero"):
    if off not in ("zero", "hold"):
        raise ValueError(f"gate: off is 'zero' (the output is 0 while the gate is closed) or "
                         f"'hold' (it keeps the last value let through), got {off!r}.")
    if isinstance(threshold, bool) or not np.isfinite(float(threshold)):
        raise ValueError(f"gate: threshold={threshold!r} is not a finite number.")


def gate(x, by, threshold=0.5, off="zero", fs=None, events=None, placement="gate"):
    """`x` where `by > threshold`, and `off` elsewhere: an ideal switch, open or closed.

    `off="zero"` gives 0 while closed; `"hold"` keeps the last sample let through (the first
    sample of the record until the gate has opened once). The switch has no edge shaping and no
    hysteresis of its own: shape the enable in its sub-chain (`law("comparator", hysteresis=)`
    for a noisy one), or use `burst` for a periodic cadence with ramped edges.

    The switching instants are ground truth. Pass a list as `events` and each is appended as an
    `events.Event(kind="transition")`: `sample` is the first sample in the new state,
    `direction` +1 opening / -1 closing, and `t_s` (given `fs`) the time `by` crossed the
    threshold, linearly interpolated between the two samples either side."""
    x = _envelope_ok(x)
    check_gate(threshold, off)
    by = _by(x, by, "gate")
    thr = float(threshold)
    on = np.broadcast_to(np.asarray(by) > thr, x.shape)
    if events is not None and np.ndim(by):
        from .events import Event
        for i in np.flatnonzero(on[1:] != on[:-1]) + 1:
            frac = (thr - by[i - 1]) / (by[i] - by[i - 1])
            events.append(Event(
                sample=int(i), kind="transition", severity=0.0, width=1,
                t_s=None if fs is None else float((i - 1 + frac) / fs),
                direction=1.0 if on[i] else -1.0, placement=placement,
                params={"threshold": thr, "state": "on" if on[i] else "off"}))
    if off == "zero":
        return np.where(on, x, 0.0)
    last = np.maximum.accumulate(np.where(on, np.arange(len(x)), -1))
    return x[np.maximum(last, 0)]


# ------------------------------------------------------------------------------------ delay
def _kernel(mu):
    """Interpolation weights for the fractional positions `mu` (in [0, 1)): one row of
    `DELAY_TAPS` each, for the input samples at offsets -15..16 from the one below the target."""
    half = DELAY_TAPS // 2
    u = np.arange(-half + 1, half + 1)[None, :] - np.asarray(mu, float)[:, None]
    w = np.i0(DELAY_BETA * np.sqrt(np.clip(1.0 - (u / half) ** 2, 0.0, None))) / np.i0(DELAY_BETA)
    h = np.sinc(u) * w
    return h / h.sum(axis=1, keepdims=True)                 # a constant stays that constant


def delay_error(f_hz, fs):
    """The worst error of the delay's interpolation for a unit tone at `f_hz` on a grid at
    `fs`, over every fractional position: `max |H(f) - exp(j*2*pi*f*mu)|`, MEASURED on the
    kernel the op uses. A tone of amplitude A is off by at most A times this at any sample more
    than 16 samples from the record's ends; a waveform, by the sum over its components."""
    f = np.atleast_1d(np.asarray(f_hz, float)) / float(fs)
    if np.any(f < 0) or np.any(f > 0.5):
        raise ValueError(f"delay_error: f_hz must lie between 0 and the Nyquist frequency "
                         f"{0.5 * fs:.6g} Hz.")
    mu = np.linspace(0.0, 1.0, 257)
    h = _kernel(mu)
    k = np.arange(-DELAY_TAPS // 2 + 1, DELAY_TAPS // 2 + 1)
    out = np.array([np.abs((h * np.exp(2j * np.pi * v * k)).sum(axis=1)
                           - np.exp(2j * np.pi * v * mu)).max() for v in f])
    return float(out[0]) if np.ndim(f_hz) == 0 else out


@functools.lru_cache(maxsize=1)
def _band_error():
    """The largest `delay_error` below `DELAY_BAND`, scanned at 4001 frequencies (its lobes
    are fs/32 wide, so about 300 points each)."""
    return float(delay_error(np.linspace(0.0, DELAY_BAND, 4001), 1.0).max())


def describe_delay(fs, max_rate=DELAY_MAX_RATE):
    """What a `delay` on a grid at `fs` promises: `{"taps", "error", "below_hz", "max_rate"}`.
    `error` is the largest `delay_error` for any tone below `below_hz`."""
    check_delay(max_rate)
    return {"taps": DELAY_TAPS, "error": _band_error(), "below_hz": DELAY_BAND * float(fs),
            "max_rate": float(max_rate)}


def check_delay(max_rate=DELAY_MAX_RATE):
    if isinstance(max_rate, bool) or not 0.0 < float(max_rate) < 1.0:
        raise ValueError(f"delay: max_rate={max_rate!r} must lie strictly between 0 and 1 "
                         f"second per second. At 1 the output stands still, and beyond it the "
                         f"input would play backwards.")


def delay(x, by, fs, max_rate=DELAY_MAX_RATE):
    """`x(t - by(t))`: `by` in SECONDS, a number or a waveform on `x`'s grid.

    Causal (`by >= 0`), band-limited (see `delay_error`), and bounded in rate: `by` may not
    change by `max_rate` seconds per second or more. All three are refused by name when broken.
    A delay of a whole number of samples is an exact shift."""
    x = _real(x, "delay")
    check_delay(max_rate)
    if fs is None:
        raise ValueError("delay: `by` is in seconds, which needs the sample rate -- give the "
                         "Signal a grid (Grid(fs=...)).")
    by = _by(x, by, "delay")
    d = np.broadcast_to(np.asarray(by, float), x.shape)
    if not np.all(np.isfinite(d)):
        raise ValueError("delay: `by` has samples that are not finite.")
    lo = float(d.min())
    if lo < 0.0:
        raise ValueError(
            f"delay: `by` reaches {lo:.6g} s, and a negative delay is an advance -- the output "
            f"would need samples that have not arrived. Add a constant of at least "
            f"{-lo:.6g} s (an `offset(by=)` in the sub-chain); for jitter about zero that may "
            f"advance, the named op is `timing`.")
    rate = float(np.abs(np.diff(d)).max() * fs) if len(d) > 1 else 0.0
    if rate >= float(max_rate):
        raise ValueError(
            f"delay: `by` changes at up to {rate:.4g} seconds per second, and max_rate is "
            f"{float(max_rate):g}. At that rate a tone moves by up to {100 * rate:.3g} % in "
            f"frequency. Smooth `by` (a `filter` in its sub-chain), or raise max_rate (below "
            f"1) if the waveform's band leaves the room.")
    n = len(x)
    pos = np.arange(n) - d * fs
    base = np.floor(pos)
    mu = pos - base
    base = base.astype(np.int64)
    if not mu.any():
        return x[np.clip(base, 0, n - 1)]
    k = np.arange(-DELAY_TAPS // 2 + 1, DELAY_TAPS // 2 + 1)
    y = np.empty(n)
    for a in range(0, n, _CHUNK):
        b = min(a + _CHUNK, n)
        taps = x[np.clip(base[a:b, None] + k[None, :], 0, n - 1)]
        y[a:b] = np.einsum("ij,ij->i", taps, _kernel(mu[a:b]))
    return y

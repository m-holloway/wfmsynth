"""
wfmsynth.trigger — a triggered acquisition: render long, find a condition, keep the window.

A real capture is placed by a trigger. The scope runs continuously, a condition fires, and the
record is the stretch from `pre_s` before that instant to `post_s` after it. A synthetic record
that starts at time zero has no such placement: nothing sits at a known position, and the
record's head is the transmitter turning on.

    long = ws.Grid(fs=80e9, baud=10e9, n=1 << 16)
    sig = (ws.Signal(seed=1, grid=long).carrier("nrz", jitter=dict(rj=0.3))
             .lossy(loss_db=6).trigger(pre_s=1e-9, post_s=4e-9))
    x, events = sig.realize()               # 400 samples; the crossing sits 1 ns in

THE RECORD IS RENDERED LONG AND CUT, not placed around a chosen event, because the trigger
acts on the waveform as it arrives: noise, jitter and the channel all move the crossing, so
where it fires is known only after the chain has run. The grid is the long render; `trigger`
shortens the record to `round((pre_s + post_s) * fs)` samples per segment.

WHAT IT DOES TO THE SAMPLES: nothing. Each window is a slice of the long render, bit for bit,
starting at the sample nearest `crossing + jitter - pre_s`. A scope's sample clock is not
synchronous with its trigger, so the crossing lands BETWEEN two samples: it sits at `pre_s`
to within half a sample (plus the trigger jitter), and exactly where is the event's `t_s`.
Only `average` produces new sample values.

CONDITIONS (`condition=`), each found on a Schmitt comparator at `level` with `hysteresis`:
  edge    the waveform crosses `level` in the direction of `slope` ("rising", "falling",
          "either").
  width   a pulse whose width at `level` is inside `width_s=[min, max]` (either end may be
          None). `slope` is the pulse's LEADING edge, so "rising" is a positive pulse. It
          fires at the trailing edge, which is when the width is known.
  runt    a pulse that crosses `level` and returns without reaching `runt_level`. Fires at
          the trailing edge; `width_s` may bound it as well.
`level=None` is a scope's "set to 50 %": midway between the long render's extremes.

THE SCOPE'S ARMING, which decides WHICH occurrence fires:
  - not before `pre_s`: the pre-trigger buffer has to be full;
  - after a trigger, not again until `holdoff_s` has passed AND the window in progress has
    been stored (`pre_s + post_s`): segments never overlap. A holdoff of exactly k periods
    re-arms on the k-th edge;
  - a window that would run past the end of the render is not taken.
The selection does not depend on the jitter draw (it allows 6 sigma of room), so re-rolling
the trigger jitter moves the windows and never changes which edges fired.

TRIGGER JITTER (`jitter_s`, rms, role `trigger/<op index>`) is the scope's error in timing
its own trigger, one Gaussian draw per trigger. It moves the WINDOW: every sample in a window
shifts together and no edge is smeared. That is what separates it from timebase jitter
(`timebase`, `acquire(sample_clock_jitter_rms_s=)`), which moves each sample on its own.
Averaging jittered triggers does smear an edge, as on a real scope.

`n_segments` is segmented memory: that many windows laid end to end in one record
(`x.reshape(n_segments, -1)`), each with its timestamp. `average=n` averages n consecutive
triggers into each window.

GROUND TRUTH. `Signal.realize()` returns one `Event(kind="trigger")` per trigger:
  sample, t_s          where the crossing sits in the delivered record (t_s is sub-sample)
  direction            +1 / -1, the edge that fired
  params.segment       which window; params.shot which of its `average` triggers
  params.timestamp_s   the crossing's time in the long render
  params.jitter_s      this trigger's jitter draw (the crossing sits at pre_s - jitter_s)
  params.start         the window's first sample in the long render
  params.level, .condition, and .width_s / .peak for a width or runt trigger
  params.dropped       how many events of the long render fell outside every window
Its `at` is `trigger/<op index>`: it is an observation, not a change to the samples, so
`EventList.on_record()` and `mask()` leave it out. Events already realised on the long render
(an `events` op, a comparator's transitions, a sub-chain's) are re-timed into the window that
holds them and the rest are dropped, counted in `dropped`.

REFUSED: no trigger, or fewer than `n_segments * average` of them, and the message states the
render length that would have been enough -- never a silently untriggered record.
"""
from __future__ import annotations

import math

import numpy as np

CONDITIONS = ("edge", "width", "runt")
SLOPES = ("rising", "falling", "either")
JITTER_ROOM = 6.0        # sigmas of trigger jitter the arming leaves clear at both ends
# A holdoff of exactly k periods is the natural thing to ask for, and in exact arithmetic the
# k-th edge then satisfies it. Crossing times carry rounding error, so the comparison gives
# way by this fraction of the re-arm time rather than letting the last bit pick the edge.
TIE = 1e-9

_DEFAULTS = {"condition": "edge", "level": None, "slope": "rising", "hysteresis": 0.0,
             "width_s": None, "runt_level": None, "holdoff_s": 0.0, "jitter_s": 0.0,
             "n_segments": 1, "average": 1}


def _real(name, v, lo=0.0, strict=False):
    ok = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
    if not ok or not math.isfinite(v) or v < lo or (strict and v <= lo):
        raise ValueError(f"trigger: {name} is a number {'>' if strict else '>='} {lo:g}, "
                         f"got {v!r}")
    return float(v)


def _count(name, v):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
        raise ValueError(f"trigger: {name} is a whole number >= 1, got {v!r}")
    return int(v)


def check(params):
    """Validate a trigger's parameters and return them with every default filled in. Refuses
    a parameter the chosen condition does not read rather than ignoring it."""
    q = {**_DEFAULTS, **params}
    for k in ("pre_s", "post_s"):
        if k not in params:
            raise ValueError(f"trigger: {k} is required -- the window is pre_s before the "
                             f"trigger and post_s after it, in seconds. Neither has a default "
                             f"because the window IS the record.")
    if q["condition"] not in CONDITIONS:
        raise ValueError(f"trigger: condition is one of {CONDITIONS}, got {q['condition']!r}")
    if q["slope"] not in SLOPES:
        raise ValueError(f"trigger: slope is one of {SLOPES}, got {q['slope']!r}")
    q["pre_s"] = _real("pre_s", q["pre_s"])
    q["post_s"] = _real("post_s", q["post_s"])
    for k in ("hysteresis", "holdoff_s", "jitter_s"):
        q[k] = _real(k, q[k])
    q["n_segments"] = _count("n_segments", q["n_segments"])
    q["average"] = _count("average", q["average"])
    if q["level"] is not None:
        q["level"] = _real("level", q["level"], lo=-math.inf)
    cond = q["condition"]
    if cond == "edge" and q["width_s"] is not None:
        raise ValueError("trigger: width_s is read by condition='width' and 'runt', not by an "
                         "edge trigger. Say condition='width'.")
    if cond != "runt" and q["runt_level"] is not None:
        raise ValueError("trigger: runt_level is read only by condition='runt'.")
    if cond == "width" and q["width_s"] is None:
        raise ValueError("trigger: condition='width' needs width_s=[min, max] in seconds; "
                         "either end may be None.")
    if q["width_s"] is not None:
        w = q["width_s"]
        if not isinstance(w, (list, tuple)) or len(w) != 2 or (w[0] is None and w[1] is None):
            raise ValueError(f"trigger: width_s is [min, max] in seconds with at least one "
                             f"end given (None leaves an end open), got {w!r}")
        lo = 0.0 if w[0] is None else _real("width_s[0]", w[0])
        hi = math.inf if w[1] is None else _real("width_s[1]", w[1])
        if not lo < hi:
            raise ValueError(f"trigger: width_s is [min, max] with min < max, got {w!r}")
        q["_width"] = (lo, hi)
    if cond == "runt":
        if q["level"] is None or q["runt_level"] is None:
            raise ValueError("trigger: condition='runt' needs both thresholds: level (the one "
                             "the pulse crosses) and runt_level (the one it fails to reach).")
        if q["slope"] == "either":
            raise ValueError("trigger: condition='runt' with slope='either' has no single "
                             "runt_level. Say slope='rising' (a positive runt, runt_level above "
                             "level) or 'falling' (a negative one, runt_level below).")
        rl = _real("runt_level", q["runt_level"], lo=-math.inf)
        sign = 1.0 if q["slope"] == "rising" else -1.0
        if sign * (rl - q["level"]) <= 0.5 * q["hysteresis"]:
            raise ValueError(
                f"trigger: a {'positive' if sign > 0 else 'negative'} runt (slope="
                f"{q['slope']!r}) needs runt_level {'above' if sign > 0 else 'below'} level by "
                f"more than hysteresis/2; got level={q['level']:g}, runt_level={rl:g}, "
                f"hysteresis={q['hysteresis']:g}.")
        q["runt_level"] = rl
    return q


def window_samples(params, fs):
    """Samples in one window: `round((pre_s + post_s) * fs)`."""
    return int(round((float(params["pre_s"]) + float(params["post_s"])) * float(fs)))


def record_length(params, fs):
    """Samples in the record a `trigger` op delivers at sample rate `fs`."""
    return int(params.get("n_segments", 1)) * window_samples(params, fs)


def check_chain(ops):
    """The two things a chain cannot say about a trigger, refused where it is written."""
    at = [i for i, o in enumerate(ops) if o["op"] == "trigger"]
    if not at:
        return
    if len(at) > 1:
        raise ValueError(f"trigger (op {at[1]}): this chain already has a trigger at op "
                         f"{at[0]}. One record has one trigger; a second would cut a window "
                         f"out of a window. Use n_segments= for several windows.")
    i = at[0]
    check({k: v for k, v in ops[i].items() if k != "op" and not k.startswith("_")})
    if int(ops[i].get("n_segments", 1)) > 1 and i != len(ops) - 1:
        raise ValueError(
            f"{ops[i + 1]['op']} (op {i + 1}): the record is {ops[i]['n_segments']} separate "
            f"windows laid end to end, and an op after the trigger would run across the seams "
            f"as if they were continuous time. Put {ops[i + 1]['op']!r} before `trigger` (a "
            f"scope triggers on what it has already acquired), or use n_segments=1.")


# ------------------------------------------------------------------ finding the condition
def _crossings(x, level, hysteresis):
    """Every committed crossing of `level`: ``(j, t, rising)`` where the waveform crosses
    between samples `j` and `j + 1` at fractional sample `t`.

    A crossing is committed once the waveform has travelled from one side of the band
    `level -/+ hysteresis/2` to the other; its time is the LAST time it crossed `level`
    itself on the way, linearly interpolated."""
    n = len(x)
    k = np.arange(n)
    up, dn = level + 0.5 * hysteresis, level - 0.5 * hysteresis
    cmd = np.where(x >= up, 1, np.where(x < dn, 0, -1))          # -1: inside the band, hold
    last = np.maximum.accumulate(np.where(cmd >= 0, k, -1))
    if last[-1] < 0:
        return np.zeros(0, int), np.zeros(0), np.zeros(0, bool)
    first = int(np.argmax(last >= 0))
    state = cmd[np.maximum(last, first)]
    i = np.flatnonzero(state[1:] != state[:-1]) + 1              # first sample in the new state
    rising = state[i] == 1
    below = x < level
    last_below = np.maximum.accumulate(np.where(below, k, -1))
    last_above = np.maximum.accumulate(np.where(~below, k, -1))
    j = np.where(rising, last_below[i - 1], last_above[i - 1])
    t = j + (level - x[j]) / (x[j + 1] - x[j])
    return j, t, rising


def find(x, fs, condition="edge", level=None, slope="rising", hysteresis=0.0, width_s=None,
         runt_level=None):
    """Every occurrence of a trigger condition on `x`, before any arming.

    Returns a dict of arrays, one entry per occurrence in time order: ``t_s`` (the sub-sample
    time it fires), ``direction`` (+1 / -1, the edge it fires on), ``width_s`` and ``peak``
    (pulse conditions only, else None), plus the ``level`` used."""
    x = _waveform(x, "the waveform")
    q = check(dict(pre_s=0.0, post_s=0.0, condition=condition, level=level, slope=slope,
                   hysteresis=hysteresis, width_s=width_s, runt_level=runt_level))
    fs = float(fs)
    lvl = 0.5 * (float(x.max()) + float(x.min())) if q["level"] is None else q["level"]
    j, t, rising = _crossings(x, lvl, q["hysteresis"])
    out = {"level": lvl, "width_s": None, "peak": None}
    if condition == "edge":
        keep = np.ones(len(t), bool) if slope == "either" else rising == (slope == "rising")
        out.update(t_s=t[keep] / fs, direction=np.where(rising[keep], 1.0, -1.0))
        return out
    # a pulse is two consecutive crossings; they alternate by construction
    lead = np.arange(len(t) - 1)
    if slope != "either":
        lead = lead[rising[lead] == (slope == "rising")]
    width = (t[lead + 1] - t[lead]) / fs
    keep = np.ones(len(lead), bool)
    if "_width" in q:
        keep &= (width >= q["_width"][0]) & (width <= q["_width"][1])
    peak = None
    if len(lead):
        # the extreme of the samples strictly between the two crossings
        cuts = np.stack([j[lead] + 1, j[lead + 1] + 1], 1).ravel()
        pos = rising[lead]
        peak = np.where(pos, np.maximum.reduceat(x, cuts)[::2], np.minimum.reduceat(x, cuts)[::2])
        if condition == "runt":
            keep &= np.where(pos, peak < q["runt_level"], peak > q["runt_level"])
        peak = peak[keep]
    lead = lead[keep]
    out.update(t_s=t[lead + 1] / fs, direction=np.where(rising[lead + 1], 1.0, -1.0),
               width_s=width[keep], peak=np.zeros(0) if peak is None else peak)
    return out


def _waveform(x, what):
    if np.iscomplexobj(x):
        raise ValueError(f"trigger: {what} is complex. A trigger compares a real voltage with "
                         f"a level; trigger before `downconvert`, or after `upconvert` / "
                         f"`photodetect`.")
    x = np.asarray(x, float)
    if x.ndim != 1:
        raise ValueError(f"trigger: {what} has shape {x.shape}, not one waveform on a "
                         f"timebase (acquire(decimation=peak_hold) returns a min/max pair). "
                         f"Trigger before it.")
    if not np.isfinite(x).all():
        raise ValueError(f"trigger: {what} holds a NaN or an infinity, so a level crossing "
                         f"is undefined.")
    return x


# ------------------------------------------------------------------ the scope's arming
def select(t_s, pre_s, post_s, holdoff_s=0.0, room_s=0.0):
    """Which occurrences a scope would fire on, ignoring where the render ends: indices into
    `t_s` (sorted). Not before the pre-trigger buffer is full; then not until both the
    holdoff and the window in progress are over. `room_s` is clearance kept for jitter."""
    t_s = np.asarray(t_s, float)
    rearm = max(float(holdoff_s), pre_s + post_s + 2.0 * room_s) * (1.0 - TIE)
    out = []
    i = int(np.searchsorted(t_s, (pre_s + room_s) * (1.0 - TIE), "left"))
    while i < len(t_s):
        out.append(i)
        i = max(i + 1, int(np.searchsorted(t_s, t_s[i] + rearm, "left")))
    return np.asarray(out, int)


def _describe(q, lvl):
    edge = {"rising": "rising", "falling": "falling", "either": "rising or falling"}[q["slope"]]
    if q["condition"] == "edge":
        return f"{edge} edge through {lvl:.6g}"
    pol = {"rising": "positive", "falling": "negative", "either": "positive or negative"}
    what = f"{pol[q['slope']]} pulse at {lvl:.6g}"
    if "_width" in q:
        lo, hi = q["_width"]
        what += (f" at least {lo:.6g} s wide" if math.isinf(hi) else
                 f" at most {hi:.6g} s wide" if lo == 0.0 else
                 f" between {lo:.6g} and {hi:.6g} s wide")
    if q["condition"] == "runt":
        what += f" that does not reach {q['runt_level']:.6g}"
    return what


def _refuse(q, found, picks, n_fit, need, n, fs, room_s, src):
    t = found["t_s"]
    T = n / fs
    what = _describe(q, found["level"])
    have = f"this render is {n} samples ({T:.6g} s) where the trigger sees it"
    tail = q["post_s"] + room_s
    if len(t) == 0:
        why = ""
        if not src.min() < found["level"] < src.max():
            why = " The level is outside that range, so no length of this record would trigger."
        raise ValueError(
            f"trigger: no {what} anywhere in the long render -- {have}, spanning "
            f"{src.min():.6g} to {src.max():.6g}.{why} Nothing was recorded rather than an "
            f"untriggered record: change the condition, or render a stretch in which it occurs.")
    if len(picks) >= need:                       # enough occur; the render stops short of them
        need_s = t[picks[need - 1]] + tail
        how = f"needs at least {int(math.ceil(need_s * fs)) + 1} samples ({need_s:.6g} s)"
    else:                                        # not enough occur: extrapolate their spacing
        period = (t[-1] - t[0]) / (len(t) - 1) if len(t) > 1 else T
        rearm = max(q["holdoff_s"], q["pre_s"] + q["post_s"] + 2.0 * room_s)
        step = period * max(1, math.ceil(rearm / period - 1e-6))
        last = t[picks[-1]] if len(picks) else max(t[-1], q["pre_s"] + room_s)
        need_s = last + (need - len(picks)) * step + tail
        how = (f"needs about {int(math.ceil(need_s * fs)) + 1} samples ({need_s:.6g} s) if the "
               f"condition keeps occurring every {period:.6g} s as it does here")
    raise ValueError(
        f"trigger: {need} trigger{'s' if need > 1 else ''} wanted (n_segments x average) and "
        f"{n_fit} fit. The long render holds {len(t)} occurrence{'s' if len(t) != 1 else ''} "
        f"of a {what}; a scope armed after pre_s={q['pre_s']:.6g} s, re-arming after "
        f"max(holdoff_s, pre_s + post_s), fires on {len(picks)} of them and {n_fit} leave "
        f"post_s={q['post_s']:.6g} s before the end. It {how}; {have}. Render longer (Grid(n=), "
        f"and the record_length of an `acquire` before it), or shorten the window or the "
        f"holdoff.")


# ------------------------------------------------------------------ the acquisition itself
def apply(x, fs, *, pre_s, post_s, condition="edge", level=None, slope="rising",
          hysteresis=0.0, width_s=None, runt_level=None, holdoff_s=0.0, jitter_s=0.0,
          n_segments=1, average=1, rng=None, on=None, events=None, placement="trigger",
          event_scale=1.0):
    """Cut the triggered window(s) out of the long render `x` (sampled at `fs`). See the module
    docstring for the parameters.

    `on` is the waveform the condition is looked for on, when it is not `x` itself: a trigger
    on another channel. It must be on the same timebase, so the same length.

    `events`, when given, is the list of events already realised on the long render. It is
    rewritten in place onto the delivered record -- re-timed into the window that holds each
    one, the rest dropped -- and one `Event(kind="trigger")` per trigger is appended.
    `event_scale` converts those events' `sample` to this record's samples when an earlier op
    changed the rate (`fs` now / `fs` they were placed at)."""
    q = check(dict(pre_s=pre_s, post_s=post_s, condition=condition, level=level, slope=slope,
                   hysteresis=hysteresis, width_s=width_s, runt_level=runt_level,
                   holdoff_s=holdoff_s, jitter_s=jitter_s, n_segments=n_segments,
                   average=average))
    x = _waveform(x, "the record")
    fs = float(fs)
    n = len(x)
    src = x
    if on is not None:
        src = _waveform(on, "`on`")
        if len(src) != n:
            raise ValueError(f"trigger: `on` has {len(src)} samples and the record {n}. The "
                             f"trigger source shares the record's timebase; render both on "
                             f"the same grid.")
    n_win = window_samples(q, fs)
    if n_win < 2:
        raise ValueError(f"trigger: pre_s + post_s = {q['pre_s'] + q['post_s']:.6g} s is "
                         f"{n_win} samples at {fs:.6g} Sa/s. A window needs at least 2.")
    found = find(src, fs, condition=condition, level=q["level"], slope=slope,
                 hysteresis=q["hysteresis"], width_s=width_s, runt_level=q["runt_level"])
    t = found["t_s"]
    need = q["n_segments"] * q["average"]
    room_s = JITTER_ROOM * q["jitter_s"] + (1.0 / fs if q["jitter_s"] else 0.0)
    picks = select(t, q["pre_s"], q["post_s"], q["holdoff_s"], room_s)
    # the window is [t - pre_s, t - pre_s + n_win / fs), half a sample of rounding either way
    fits = picks[t[picks] - q["pre_s"] + (n_win + 0.5) / fs + room_s <= n / fs]
    if len(fits) < need:
        _refuse(q, found, picks, len(fits), need, n, fs, room_s, src)
    picks = fits[:need]
    if q["jitter_s"]:
        jit = (rng if rng is not None else np.random.default_rng()).normal(
            0.0, q["jitter_s"], need)
    else:
        jit = np.zeros(need)
    starts = np.rint((t[picks] + jit - q["pre_s"]) * fs).astype(int)
    if starts.min() < 0 or starts.max() + n_win > n:
        raise ValueError(f"trigger: a jitter draw beyond {JITTER_ROOM:g} sigma put a window "
                         f"outside the long render. Render it with another seed.")
    win = x[starts[:, None] + np.arange(n_win)]
    if q["average"] > 1:
        win = win.reshape(q["n_segments"], q["average"], n_win).mean(1)
    if events is not None:
        from .events import Event
        dropped = _retime(events, starts, n_win, fs, event_scale, q["average"])
        for k, p in enumerate(picks):
            seg = k // q["average"]
            pos = t[p] * fs - starts[k]                    # the crossing, in window samples
            par = {"segment": seg, "timestamp_s": float(t[p]), "jitter_s": float(jit[k]),
                   "start": int(starts[k]), "level": float(found["level"]),
                   "condition": condition, "dropped": dropped}
            if q["average"] > 1:
                par["shot"] = k % q["average"]
            if found["width_s"] is not None:
                par.update(width_s=float(found["width_s"][p]), peak=float(found["peak"][p]))
            events.append(Event(
                sample=seg * n_win + int(math.ceil(pos)), kind="trigger", severity=0.0,
                t_s=float((seg * n_win + pos) / fs), width=1,
                direction=float(found["direction"][p]), placement=placement, at=placement,
                params=par))
    return win.ravel()


def _retime(events, starts, n_win, fs, scale, average):
    """Move the long render's events onto the delivered record, in place; return how many
    fell outside every window. Windows never overlap, so an event is in at most one."""
    kept = []
    for e in events:
        pos = e.sample if scale == 1.0 else int(math.floor(e.sample * scale))
        k = int(np.searchsorted(starts, pos, "right")) - 1
        if k < 0 or pos >= starts[k] + n_win:
            continue
        seg = k // average
        move = seg * n_win - int(starts[k])
        e.sample = pos + move
        if e.t_s is not None:
            e.t_s = float(e.t_s + move / fs)
        if e.width is not None and scale != 1.0:
            e.width = int(math.ceil(e.width * scale))
        if average > 1:
            e.params["shot"] = k % average
        kept.append(e)
    dropped = len(events) - len(kept)
    events[:] = kept
    return dropped

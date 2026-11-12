"""
wfmsynth.sequence -- chains joined in TIME (`sequence`) or selected by a SIGNAL (`switch`).

Many real records are made of parts: a preamble then a payload, power-up then steady state, an
idle level then a burst, two streams multiplexed onto one line. Each part is a sub-chain
(`wfmsynth.subchain`), so it can be anything a chain can be, and these two SOURCE ops say how
the parts share the record:

    pre  = Signal(grid=g).carrier("sine", f_hz=1e6)
    data = Signal(grid=g).carrier("nrz", n_ui=400).lossy(loss_db=3.0, loss_at_ghz=4.0)

    Signal(seed=1, grid=g).sequence([{"chain": pre, "duration_s": 2e-6}, {"chain": data}])
    Signal(seed=1, grid=g).switch(select=gate, inputs=[pre, data], threshold=0.5)

Ops after either one act on the joined record: one channel, one scope, for all the parts.

SEQUENCE. Segment k occupies `duration_s` seconds, end to end; the last may leave it off and
take the rest of the record. What a segment brings across its boundary is stated per segment:

    "state": "restart"    (default) the segment is its chain rendered alone, from the chain's
                          own t = 0, placed at the boundary. NOTHING is carried: phase, pattern
                          position, filter memory and the chain's turn-on all start again.
    "state": "continue"   the chain is rendered across the whole record and the segment is the
                          window of it at the segment's own place in time. EVERYTHING the chain
                          has is carried: a tone's phase, a pattern's position and a filter's
                          memory are what they would be had the chain been running all along.

Either way the chain is rendered on the record's own grid and cut -- never on a shorter one --
so a segment is exactly `chain.waveform()[a:b]`, and nothing about the chain depends on how
long its segment is. State is never carried from one segment's chain INTO another's: two chains
do not share a filter. To pass every part through one filter, put the filter after `sequence`.

The join, in the `t_join_s` after each boundary (`ramp` is the shape):

    "cut"         none. The next segment starts on the sample after the last one ended.
    "crossfade"   the outgoing chain runs on past its boundary and the record moves from it to
                  the incoming segment along a raised cosine: a modelled transient of a stated
                  length (10 % to 90 % in 0.5903 * t_join_s), not a discontinuity.
    "hold"        the line holds the outgoing segment's last value, then the incoming segment
                  appears (already t_join_s into its own time): a dead time between modes.

Outside those windows the record IS the segments spliced, sample for sample. Each boundary is
emitted as a `segment` event; `layout()` gives every segment's samples.

SWITCH. An analog multiplexer: `inputs[k]` is on the line while `select` is above `k` of the
thresholds (one `threshold` for two inputs, 0.5 unless stated -- `gate`'s convention, and
`gate` is this op for the case where the other input is nothing; an ascending list of N-1 for
N). A change of selection takes `t_switch_s`: over that time the line moves from what it was
carrying to the newly selected input along the same raised cosine. Each change is emitted as a
`switch` event whose `t_s` is the time `select` crossed the threshold, between two samples. The
mux has no hysteresis; a noisy selector chatters, as a real one does -- put
`law("comparator", hysteresis=...)` at the end of the `select` chain to clean it.

ON AN I/Q RECORD (`Grid(domain="iq")`) the parts are I/Q chains and the joined record is their
envelope: a preamble then a payload about one centre frequency. `select` stays a real signal.
A segment boundary is still a `segment` event, so `iqfile.to_interleaved(events=)` writes it
into the marker channel and `to_sigmf` into an annotation.

ON AN OPTICAL FIELD the parts are field chains (each ends on `eo` or a field op) and the joined
record is a field: a burst after an idle, an optical switch between two sources. A grid that
states no domain follows them, as it follows I/Q parts.

One mechanism or two? Two. A sequence is a statement about TIME (durations, written in the
recipe, on the grid); a switch is a statement about a SIGNAL (the instants are an outcome, and
are emitted). A sequence written as a switch would need a staircase selector invented to carry
its durations, and the per-segment restart has no meaning for a mux, whose inputs all run.
"""
from __future__ import annotations

import numbers

import numpy as np

from . import subchain as SUB
from .events import Event

JOINS = ("cut", "crossfade", "hold")
STATES = ("restart", "continue")
_SEGMENT_KEYS = ("chain", "duration_s", "state")
GRID_TOL = 1e-6           # samples: how far t * fs may be from a whole number and still be on it


def refuse_after(before, op, parts):
    """`sequence` and `switch` are sources. `before` is what precedes one: the ops already in
    the chain (at build time) or the waveform arriving at it (at render)."""
    if before is not None and len(before):
        raise ValueError(
            f"{op} is a SOURCE: it starts a chain, and the waveform the earlier ops made "
            f"would be discarded. Make that chain one of its {parts}, and put the ops that "
            f"act on the joined record after {op}.")


def ramp(m):
    """The weight of the INCOMING signal on each of the `m` samples of a transition:
    ``0.5 * (1 - cos(pi * (j + 0.5) / m))``, a raised cosine. It is 1 from sample `m` on."""
    return 0.5 * (1.0 - np.cos(np.pi * (np.arange(int(m)) + 0.5) / max(int(m), 1)))


def _samples(t_s, fs, who, least=1):
    """A duration in seconds -> a whole number of samples, or a refusal. Never rounded."""
    if isinstance(t_s, dict):
        raise ValueError(
            f"{who} is a toleranced value, and a duration has to land on the sample grid, "
            f"which a drawn number does not. State it as a number; vary a duration across a "
            f"dataset by writing a different whole number of samples per record.")
    if isinstance(t_s, bool) or not isinstance(t_s, numbers.Real) or not np.isfinite(t_s):
        raise ValueError(f"{who} is a duration in seconds, got {t_s!r}.")
    x = float(t_s) * fs
    k = int(round(x))
    if abs(x - k) > GRID_TOL or k < least:
        lo = max(int(np.floor(x)), least)
        raise ValueError(
            f"{who}={t_s!r} s is {x:.6g} samples at fs={fs:g} Hz"
            + (f", and needs at least {least}" if k < least else ", which is not on the grid")
            + f". It is never rounded into a different record: the nearest durations that are "
              f"on it are {lo / fs!r} s ({lo} samples) and {(lo + 1) / fs!r} s ({lo + 1}).")
    return k


def _grid(grid, op):
    if grid is None or grid.n is None:
        raise ValueError(f"{op} joins chains on the record's grid and there is none -- pass "
                         f"Signal(grid=Grid(fs=..., n=...)).")
    return float(grid.fs), int(grid.n)


def layout(segments, grid):
    """Where each segment of a `sequence` sits: one dict per segment, in order, with `segment`
    (its index), `start` and `stop` (samples, stop exclusive), `t_start_s`, `t_stop_s` and
    `state`. Raises the op's own refusals, so it is also the check that the durations fit.

        for s in ws.sequence.layout(ws.op_params(recipe, "sequence")["segments"], grid): ...
    """
    fs, n = _grid(grid, "sequence")
    if not isinstance(segments, (list, tuple)) or not segments:
        raise ValueError("sequence: `segments` is a list of {'chain': <sub-chain>, 'duration_s': "
                         "<seconds>}, one per segment, in order; got "
                         + ("an empty list." if isinstance(segments, (list, tuple))
                            else f"{type(segments).__name__}."))
    out, at = [], 0
    for k, seg in enumerate(segments):
        who = f"sequence: segments[{k}]"
        if not isinstance(seg, dict) or seg.get("chain") is None:
            raise ValueError(f"{who} must be a dict with 'chain' (a Signal, {{'ops': [...]}} "
                             f"or {{'from': <recipe>}}) and 'duration_s' (how long it lasts, seconds).")
        extra = sorted(set(seg) - set(_SEGMENT_KEYS))
        if extra:
            raise ValueError(
                f"{who} has {extra}, which nothing reads; a segment takes "
                f"{list(_SEGMENT_KEYS)}. "
                + ("How long it lasts is 'duration_s'; `t_s` is an instant, on an event and "
                   "in a `net` schedule." if "t_s" in extra
                   else "Parameters of the chain go on its ops."))
        state = seg.get("state", "restart")
        if state not in STATES:
            raise ValueError(f"{who}: state is 'restart' (the chain from its own t=0) or "
                             f"'continue' (the chain as if it had been running since the "
                             f"record began), got {state!r}.")
        if seg.get("duration_s") is None:
            if k != len(segments) - 1:
                raise ValueError(f"{who} has no `duration_s`. Only the LAST segment may leave it off, "
                                 f"to take the rest of the record.")
            count = n - at
            if count < 1:
                raise ValueError(
                    f"{who} takes the rest of the record, but the segments before it already "
                    f"fill it ({at} of {n} samples, {at / fs!r} of {n / fs!r} s).")
        else:
            count = _samples(seg["duration_s"], fs, f"{who}: duration_s")
        out.append({"segment": k, "start": at, "stop": at + count, "t_start_s": at / fs,
                    "t_stop_s": (at + count) / fs, "state": state})
        at += count
    if at != n:
        raise ValueError(
            f"sequence: the segments last {at / fs!r} s ({at} samples) and the record is "
            f"{n / fs!r} s ({n} samples). They must fill it exactly -- a record is never "
            f"padded or cut to fit. Leave `duration_s` off the last segment to give it the rest"
            + (f" ({(n - at) / fs!r} s more)." if at < n else
               f", or shorten the segments by {(at - n) / fs!r} s."))
    return out


def join_samples(join, t_join_s, grid, plan=None):
    """The join window in samples for a stated `join`, refusing a combination that says
    nothing: a cut with a length, a crossfade or hold without one, or a window longer than a
    segment it opens."""
    if join not in JOINS:
        raise ValueError(f"sequence: join is one of {list(JOINS)}, got {join!r}.")
    fs, _ = _grid(grid, "sequence")
    if join == "cut":
        if t_join_s:
            raise ValueError(
                f"sequence: join='cut' has no transition, so t_join_s={t_join_s!r} would be "
                f"ignored. Use join='crossfade' or 'hold' for a join that takes time.")
        return 0
    if t_join_s is None:
        raise ValueError(f"sequence: join={join!r} takes t_join_s=<seconds>, the length of "
                         f"the transition after each boundary.")
    m = _samples(t_join_s, fs, "sequence: t_join_s")
    for s in (plan or ())[1:]:
        if s["stop"] - s["start"] < m:
            raise ValueError(
                f"sequence: t_join_s={t_join_s!r} s ({m} samples) is longer than segment "
                f"{s['segment']} ({s['stop'] - s['start']} samples), so the join into it would "
                f"run into the next one. Shorten the join or lengthen the segment.")
    return m


def check_sequence(segments, join, t_join_s, grid):
    """Everything about a `sequence` that can be refused before its chains are rendered."""
    return join_samples(join, t_join_s, grid, layout(segments, grid))


def splice(waves, plan, join="cut", m=0):
    """Full-record waveforms, one per segment, joined by `plan` (from `layout`).

    Segment k is ``waves[k][0:count]`` ("restart") or ``waves[k][start:stop]`` ("continue").
    `m` is the join window in samples; see the module docstring for what happens inside it."""
    n = plan[-1]["stop"]
    y = np.empty(n, dtype=np.result_type(float, *waves))    # complex when the chains are I/Q
    lo = [s["start"] if s["state"] == "continue" else 0 for s in plan]
    for s, w, a in zip(plan, waves, lo):
        y[s["start"]:s["stop"]] = w[a:a + s["stop"] - s["start"]]
    if m:
        r = ramp(m)
        for k in range(1, len(plan)):
            b, p = plan[k]["start"], plan[k - 1]
            if join == "hold":
                y[b:b + m] = y[b - 1]
            else:                                           # the outgoing chain runs on
                a = lo[k - 1] + p["stop"] - p["start"]
                y[b:b + m] = (1.0 - r) * waves[k - 1][a:a + m] + r * y[b:b + m]
    return y


def _render(spec, streams, grid, at, who, keep=None, shift=0, domain="real", is_record=False):
    """One sub-chain on the record's grid. Its events reach the record's list only if they
    fall in `keep` (a sample range of the CHAIN's own time), moved by `shift` samples onto the
    record's -- a segment that restarts has its own clock. `is_record`: its samples become
    the record's, so its events are on the record (`Event.on_record`)."""
    sink = getattr(streams, "sink", None)
    if sink is None or keep is None:
        return SUB.render(spec, streams, grid, at, grid.n, who, domain=domain,
                          is_record=is_record)
    own = []
    y = SUB.render(spec, SUB.Namespaced(streams, "", own), grid, at, grid.n, who, domain=domain,
                   is_record=is_record)
    for e in own:
        if keep[0] <= e.sample < keep[1]:
            e.sample += shift
            if e.t_s is not None:
                e.t_s += shift / grid.fs
            sink.append(e)
    return y


def run_sequence(streams, grid, idx, segments=None, join="cut", t_join_s=None):
    """The `sequence` op: render each segment's chain, splice, emit the boundaries."""
    plan = layout(segments, grid)
    m = join_samples(join, t_join_s, grid, plan)
    waves = []
    for s, seg in zip(plan, segments):
        k, count = s["segment"], s["stop"] - s["start"]
        a = s["start"] if s["state"] == "continue" else 0
        waves.append(_render(seg["chain"], streams, grid, f"sequence/{idx}/segments/{k}/chain",
                             f"sequence: segments[{k}]: `chain`",
                             keep=(a, a + count), shift=s["start"] - a,
                             domain=getattr(grid, "domain", "real"), is_record=True))
    sink = getattr(streams, "sink", None)
    if sink is not None:
        for s in plan[1:]:
            sink.append(Event(
                sample=s["start"], kind="segment", severity=0.0, t_s=s["t_start_s"],
                width=max(m, 1), placement=f"sequence/{idx}",
                params={"segment": s["segment"], "state": s["state"], "join": join,
                        "t_join_s": m / grid.fs}))
    return splice(waves, plan, join, m)


# ------------------------------------------------------------------------------------ switch
def thresholds(threshold, n_inputs):
    """`threshold` as an ascending array of `n_inputs - 1` levels, or a refusal."""
    if n_inputs < 2:
        raise ValueError(f"switch: `inputs` is a list of at least two sub-chains to choose "
                         f"between, got {n_inputs}. One signal needs no switch.")
    try:
        thr = np.atleast_1d(np.asarray(threshold, float))
    except (TypeError, ValueError):
        thr = np.empty((0, 0))
    if thr.ndim != 1 or len(thr) != n_inputs - 1 or not np.all(np.isfinite(thr)) \
            or np.any(np.diff(thr) <= 0):
        raise ValueError(
            f"switch: {n_inputs} inputs are separated by {n_inputs - 1} threshold"
            f"{'s' if n_inputs > 2 else ''} on `select`, strictly ascending (inputs[k] is on "
            f"while select is above k of them); got threshold={threshold!r}.")
    return thr


def mux(select, inputs, threshold=0.5, m=0):
    """An analog multiplexer on arrays -> ``(y, changes)``.

    ``inputs[k]`` is selected while `select` is above `k` of the thresholds (strictly: a sample
    exactly on a threshold has not crossed it). A change takes `m` samples, along `ramp(m)`;
    a second change before the first has finished starts from wherever the line then is.
    `changes` is one ``(sample, crossing, was, now, level)`` per change: the first sample of
    the new selection, the fractional sample at which `select` crossed `level`, and the input
    indices either side."""
    thr = thresholds(threshold, len(inputs))
    sel = np.asarray(select, float)
    idx = np.searchsorted(thr, sel, side="left")
    inputs = np.asarray(inputs)                  # real voltages, or I/Q envelopes alike
    if not np.iscomplexobj(inputs):
        inputs = inputs.astype(float)
    y = inputs[idx, np.arange(len(sel))]
    at = np.flatnonzero(np.diff(idx)) + 1
    changes = []
    for c in at:
        was, now = int(idx[c - 1]), int(idx[c])
        level = float(thr[now - 1] if now > was else thr[now])
        changes.append((int(c), c - 1 + (level - sel[c - 1]) / (sel[c] - sel[c - 1]),
                        was, now, level))
    if m and len(at):
        # each input's share of the line: its 0/1 selection through the ramp's own impulse
        # response, so the shares always sum to 1 and overlapping changes compose
        h = np.diff(np.concatenate([[0.0], ramp(m), [1.0]]))
        busy = np.zeros(len(sel), bool)
        for c in at:
            busy[c:c + m] = True
        mix = np.zeros(len(sel), dtype=inputs.dtype)
        for k, x in enumerate(inputs):
            on = np.concatenate([np.full(m, float(idx[0] == k)), (idx == k).astype(float)])
            mix += np.convolve(on, h)[m:m + len(sel)] * x
        y = np.where(busy, mix, y)                 # untouched, and exact, everywhere else
    return y, changes


def check_switch(n_inputs, threshold, t_switch_s, grid):
    """Everything about a `switch` that can be refused before its chains are rendered: returns
    ``(thresholds, switching time in samples)``."""
    fs, _ = _grid(grid, "switch")
    thr = thresholds(threshold, n_inputs)
    m = _samples(t_switch_s, fs, "switch: t_switch_s", least=0) if t_switch_s else 0
    return thr, m


def run_switch(streams, grid, idx, select=None, inputs=None, threshold=0.5, t_switch_s=0.0):
    """The `switch` op: render the selector and every input, mux, emit the changes."""
    if not isinstance(inputs, (list, tuple)):
        raise ValueError("switch: `inputs` is a list of sub-chains to choose between "
                         "(Signals, {'ops': [...]} or {'from': <recipe>}).")
    thr, m = check_switch(len(inputs), threshold, t_switch_s, grid)
    if select is None:
        raise ValueError("switch needs select=<a sub-chain>: the signal whose level chooses "
                         "the input (a Signal, {'ops': [...]} or {'from': <recipe>}).")
    sel = _render(select, streams, grid, f"switch/{idx}/select", "switch: `select`")
    sink = getattr(streams, "sink", None)
    ins, placed = [], []
    for k, s in enumerate(inputs):
        own = [] if sink is not None else None
        ins.append(_render(s, streams if own is None else SUB.Namespaced(streams, "", own),
                           grid, f"switch/{idx}/inputs/{k}", f"switch: `inputs`[{k}]",
                           domain=getattr(grid, "domain", "real"), is_record=True))
        placed.append(own)
    y, changes = mux(sel, ins, thr, m)
    if sink is not None:
        # an input's event is on the record where that input is on the line
        on_line = np.searchsorted(thr, np.asarray(sel, float), side="left")
        for k, own in enumerate(placed):
            for e in own:
                lo, hi = e.span(len(y))
                e.on_record = e.on_record and bool(np.any(on_line[lo:hi] == k))
            sink.extend(own)
        for c, cross, was, now, level in changes:
            sink.append(Event(
                sample=c, kind="switch", severity=0.0, t_s=float(cross / grid.fs),
                width=max(m, 1), direction=1.0 if now > was else -1.0,
                placement=f"switch/{idx}",
                params={"input": now, "was": was, "threshold": level,
                        "t_switch_s": m / grid.fs}))
    return y

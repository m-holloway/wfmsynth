"""
wfmsynth.events — localized, placeable mechanisms and per-window labels.

Systemic LTI stages (loss, reflections, ADC) already apply to every sample. This module
covers the complementary case: a defect with *finite time support* — one weak symbol, one
edge that rings, one additive transient, a sag during a long run. Placement (when) is
independent of mechanism (what), and both are independent of where the op sits in a
``Signal`` chain (source vs after a channel vs at the instrument).

Targeting (``on``):

  symbols     unit-interval indices (data-locked)
  edges       rising / falling / both (transition-locked)
  pattern     long runs or a motif in the symbol stream (sequence-locked)
  aggressor   another waveform's edges (coupled, not necessarily victim-locked)
  intervals   absolute sample ranges
  poisson     asynchronous rate process (not locked to any bus)
  times       explicit sample indices or seconds

Mechanisms (``kind``):

  runt          incomplete symbol excursion (and optional early fall)
  glitch        additive localized transient
  ring          causal damped sinusoid after an instant (edge-excited or not)
  overshoot     first-lobe underdamped excess in the edge direction
  undershoot    first-lobe excess against the edge direction
  nonmonotonic  opposing bump on an edge, guaranteeing a slope reversal
  droop         multiplicative sag over an interval (PDN / IR / long-run)
  slow_edge     extra local band-limit around an instant

Eye-mask failures and "height at the sampling instant" are **measured labels**, not
mechanisms — many causes produce them. Use ``ui_heights`` / ``eye_mask`` / ``label_windows``.

This module does not recover a clock or cut segments. It emits event times (sample, UI,
seconds) so an external folder — recovered clock, ±1 UI, anything else — can join labels
to windows via ``label_windows`` / ``windows_from_centers``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


PLACEMENTS = ("symbols", "edges", "pattern", "aggressor", "intervals", "poisson", "times")
MECHANISMS = ("runt", "glitch", "ring", "overshoot", "undershoot",
              "nonmonotonic", "droop", "slow_edge")

def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


@dataclass
class Event:
    """One localized occurrence. ``sample`` is the origin; ``width`` is the support
    in samples once applied (unknown until ``apply_events``)."""
    sample: int
    kind: str
    severity: float = 0.5
    t_s: Optional[float] = None
    ui: Optional[int] = None
    width: Optional[int] = None
    direction: float = 0.0
    placement: str = ""
    params: dict = field(default_factory=dict)

    def span(self, n=None):
        w = 1 if self.width is None else max(int(self.width), 1)
        lo, hi = int(self.sample), int(self.sample) + w
        if n is not None:
            lo, hi = max(0, lo), min(int(n), hi)
        return lo, hi

    def to_dict(self):
        d = {"sample": int(self.sample), "kind": str(self.kind),
             "severity": float(self.severity), "placement": str(self.placement)}
        if self.t_s is not None:
            d["t_s"] = float(self.t_s)
        if self.ui is not None:
            d["ui"] = int(self.ui)
        if self.width is not None:
            d["width"] = int(self.width)
        if self.direction:
            d["direction"] = float(self.direction)
        if self.params:
            d["params"] = {k: _jsonable(v) for k, v in self.params.items()
                           if k not in ("_n_ui",)}
        return d


@dataclass
class EventList:
    events: list = field(default_factory=list)
    n: int = 0
    grid: object = None

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def to_dicts(self):
        return [e.to_dict() if isinstance(e, Event) else dict(e) for e in self.events]

    def overlapping(self, start, stop):
        out = []
        for e in self.events:
            lo, hi = e.span(self.n)
            if lo < stop and hi > start:
                out.append(e)
        return out

    def mask(self):
        """Per-sample support (1 where any event was applied). Unknown widths count
        as a single sample — call after ``apply_events`` for the true support."""
        g = np.zeros(int(self.n))
        for e in self.events:
            lo, hi = e.span(self.n)
            if hi > lo:
                g[lo:hi] = 1.0
        return g


def step_overshoot_fraction(zeta):
    """Closed-form peak overshoot of an underdamped 2nd-order step: exp(-πζ/√(1-ζ²))."""
    z = float(zeta)
    if z <= 0.0 or z >= 1.0:
        return 0.0
    return float(np.exp(-np.pi * z / np.sqrt(1.0 - z * z)))


def second_order_step(t, wn, zeta):
    """Causal 2nd-order step response (unit DC gain). ``t`` in seconds, ``wn`` rad/s."""
    t = np.asarray(t, float)
    y = np.zeros_like(t, dtype=float)
    pos = t >= 0.0
    tt = t[pos]
    z = float(zeta)
    wn = float(wn)
    if z < 1.0:
        wd = wn * np.sqrt(1.0 - z * z)
        phi = np.arccos(np.clip(z, -1.0, 1.0))
        den = np.sqrt(1.0 - z * z)
        y[pos] = 1.0 - np.exp(-z * wn * tt) / den * np.sin(wd * tt + phi)
    elif abs(z - 1.0) < 1e-12:
        y[pos] = 1.0 - np.exp(-wn * tt) * (1.0 + wn * tt)
    else:
        rad = np.sqrt(z * z - 1.0)
        a = wn * (z + rad)
        b = wn * (z - rad)
        y[pos] = 1.0 - (a * np.exp(-b * tt) - b * np.exp(-a * tt)) / (a - b)
    return y


def damped_sinusoid(n, dt, f0, tau, amp=1.0, phase=0.0):
    """Causal ``amp * exp(-t/τ) * sin(2π f0 t + φ)`` of length ``n`` (t = k·dt)."""
    k = np.arange(max(int(n), 0), dtype=float)
    t = k * float(dt)
    tau = max(float(tau), 1e-30)
    return float(amp) * np.exp(-t / tau) * np.sin(2.0 * np.pi * float(f0) * t + float(phase))


def slope_reversals(seg, deadband=None):
    """Count sign changes of the first difference after a deadband (0 = monotonic)."""
    seg = np.asarray(seg, float)
    if seg.size < 3:
        return 0
    g = np.diff(seg)
    thr = float(deadband) if deadband is not None else 0.02 * (np.ptp(seg) + 1e-12)
    g[np.abs(g) < thr] = 0.0
    s = np.sign(g)
    s = s[s != 0]
    if s.size < 2:
        return 0
    return int(np.sum(np.diff(s) != 0))


def measure_window(x, start, stop, center=None):
    """Per-window attributes measured from ``x`` (not from event knobs)."""
    x = np.asarray(x, float)
    lo = max(0, int(start))
    hi = min(len(x), int(stop))
    seg = x[lo:hi]
    if center is None:
        c = (lo + hi) // 2
    else:
        c = int(round(center))
    c = min(max(c, 0), max(len(x) - 1, 0))
    if seg.size == 0:
        return {"height": float(x[c]) if len(x) else 0.0, "ptp": 0.0,
                "slope_reversals": 0, "peak": 0.0, "trough": 0.0}
    return {
        "height": float(x[c]),
        "ptp": float(np.ptp(seg)),
        "slope_reversals": slope_reversals(seg),
        "peak": float(seg.max()),
        "trough": float(seg.min()),
    }


def ui_heights(x, grid, phase=None, levels=2):
    """Voltage at one sample per UI. ``phase`` defaults to the measured best phase.
    This is 'height at the sampling instant' on the *nominal* grid clock unless you
    pass a recovered ``phase`` (or use ``sample_at_phase`` on your own instants)."""
    from .measure import sample_at_phase, best_phase
    if phase is None:
        phase = best_phase(x, grid, levels=levels)
    return sample_at_phase(x, grid.samples_per_ui, phase), float(phase)


def eye_mask(heights, low, high=None):
    """Boolean per-UI (or per-window) eye-mask failure. ``low``/``high`` are voltage
    thresholds on the sampled height; ``high=None`` flags ``|h| < low``."""
    h = np.asarray(heights, float)
    if high is None:
        return np.abs(h) < float(low)
    return (h < float(low)) | (h > float(high))


def windows_from_centers(centers, n, half):
    """Build ``[{i, start, stop, center}]`` of half-width ``half`` samples around
    each center. Centers may be fractional. This is the hook for an *external*
    recovered clock: pass its sampling instants."""
    out = []
    n = int(n)
    half = float(half)
    for i, c in enumerate(np.asarray(centers, float).ravel()):
        lo = max(0, int(np.floor(c - half)))
        hi = min(n, int(np.ceil(c + half)))
        if hi > lo:
            out.append({"i": int(i), "start": lo, "stop": hi, "center": float(c)})
    return out


def nominal_ui_windows(n, grid, half_ui=1.0, phase=None):
    """±``half_ui`` windows around each nominal UI sampling instant (``grid.baud``).
    Does **not** recover a clock. For a recovered clock, use ``windows_from_centers``."""
    if grid is None or grid.baud is None:
        raise ValueError("nominal_ui_windows needs grid.baud")
    spb = float(grid.samples_per_ui)
    if phase is None:
        phase = 0.5 * spb
    n_ui = int(max(0, (int(n) - 1 - phase) / spb))
    centers = np.arange(n_ui) * spb + float(phase)
    return windows_from_centers(centers, n, half_ui * spb)


def label_windows(windows, events, x=None, eye_low=None, eye_high=None,
                  nonmonotonic_rev=None):
    """Join an EventList to externally cut windows.

    Each record contains the overlapping requested events **and** optional measured
    attributes from ``x``. ``labels`` is the union of event kinds plus measured flags
    (``nonmonotonic``, ``eye_violation``) so a trainer can use one field per segment.

    Systemic effects that were never placed as events still show up in ``measured``
    (and in ``labels`` when they trip a flag) — that is how pattern-dependent
    artefacts from a resonant stub are labelled without planting a needle.

    ``nonmonotonic_rev`` is off by default: a ±1 UI *data* window already contains a
    full pulse (one slope reversal at the peak). Pass ``nonmonotonic_rev=1`` only
    for *edge-centered* windows.
    """
    evs = events if isinstance(events, EventList) else EventList(list(events), n=0)
    rows = []
    for w in windows:
        start, stop = int(w["start"]), int(w["stop"])
        hit = evs.overlapping(start, stop)
        kinds = []
        for e in hit:
            k = e.kind if isinstance(e, Event) else e.get("kind")
            if k not in kinds:
                kinds.append(k)
        row = {"i": w.get("i"), "start": start, "stop": stop,
               "center": w.get("center"), "events": [e.to_dict() if isinstance(e, Event) else dict(e)
                                                     for e in hit],
               "kinds": kinds, "labels": list(kinds)}
        if x is not None:
            m = measure_window(x, start, stop, center=w.get("center"))
            row["measured"] = m
            if (nonmonotonic_rev is not None
                    and m["slope_reversals"] >= int(nonmonotonic_rev)
                    and "nonmonotonic" not in row["labels"]):
                row["labels"].append("nonmonotonic")
            if eye_low is not None and bool(eye_mask([m["height"]], eye_low, eye_high)[0]):
                if "eye_violation" not in row["labels"]:
                    row["labels"].append("eye_violation")
        rows.append(row)
    return rows


# --------------------------------------------------------------- placement
def _n_ui(n, grid, n_ui, symbols):
    if symbols is not None:
        return int(len(symbols))
    if n_ui is not None:
        return int(n_ui)
    if grid is not None and grid.baud is not None:
        return max(1, int(int(n) / grid.samples_per_ui))
    raise ValueError("symbol/edge/pattern placement needs n_ui=, symbols=, or grid.baud")


def _spb(n, n_ui):
    return float(n) / max(int(n_ui), 1)


def _ui_bounds(n, ui, n_ui):
    """Sample half-open interval for UI ``ui``, matching ``_place_symbols`` (floor)."""
    spb = _spb(n, n_ui)
    lo = int(ui * spb)
    hi = int((int(ui) + 1) * spb)
    return max(0, lo), min(int(n), max(hi, lo + 1))


def _dt(grid):
    return float(grid.dt) if grid is not None else 1.0


def _stamp_time(ev, grid):
    if grid is not None:
        ev.t_s = ev.sample / grid.fs
    return ev


def _draw_subset(n_cand, rng, indices=None, count=None, fraction=None, every=None):
    idx = np.arange(int(n_cand))
    if every is not None:
        idx = idx[::max(int(every), 1)]
    if indices is not None:
        want = np.asarray(indices, int)
        return want[(want >= 0) & (want < n_cand)]
    if count is None and fraction is not None:
        count = int(round(float(fraction) * len(idx)))
    if count is None:
        count = 1 if len(idx) else 0
    count = min(max(int(count), 0), len(idx))
    if count == 0 or len(idx) == 0:
        return np.zeros(0, dtype=int)
    return np.sort(rng.choice(idx, size=count, replace=False))


def _mid_crossings(x, min_step=None):
    x = np.asarray(x, float)
    if x.size < 2:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    span = np.ptp(x) + 1e-12
    min_step = span * 0.25 if min_step is None else float(min_step)
    mid = 0.5 * (np.max(x) + np.min(x))
    s = np.sign(x - mid)
    s[s == 0] = 1
    raw = np.where(np.diff(s) != 0)[0] + 1
    keep, direc = [], []
    for i in raw:
        d = x[min(i, len(x) - 1)] - x[max(i - 1, 0)]
        if abs(d) >= min_step * 0.25:
            keep.append(int(i))
            direc.append(float(np.sign(d) or 1.0))
    return np.asarray(keep, int), np.asarray(direc, float)


def _symbol_edges(symbols, n):
    lv = np.asarray(symbols, float)
    d = np.diff(lv)
    uis = np.where(d != 0)[0] + 1
    spb = _spb(n, len(lv))
    samples = np.round(uis * spb).astype(int)
    return uis.astype(int), samples, np.sign(d[d != 0]).astype(float)


def _estimate_levels(x, n_ui):
    x = np.asarray(x, float)
    spb = _spb(len(x), n_ui)
    idx = np.clip(np.round((np.arange(n_ui) + 0.5) * spb), 0, len(x) - 1).astype(int)
    return x[idx]


def _runs(levels, min_run=6, tol=None):
    lv = np.asarray(levels, float)
    if lv.size == 0:
        return []
    tol = (0.15 * (np.ptp(lv) + 1e-12)) if tol is None else float(tol)
    runs = []
    start, ref = 0, lv[0]
    for i in range(1, len(lv) + 1):
        if i < len(lv) and abs(lv[i] - ref) <= tol:
            continue
        length = i - start
        if length >= int(min_run):
            runs.append((start, length, float(np.mean(lv[start:i]))))
        if i < len(lv):
            start, ref = i, lv[i]
    return runs


def _motif_starts(levels, motif, tol=None):
    lv = np.asarray(levels, float)
    m = np.asarray(motif, float)
    if m.size == 0 or lv.size < m.size:
        return np.zeros(0, dtype=int)
    tol = (0.15 * (np.ptp(lv) + 1e-12)) if tol is None else float(tol)
    hits = []
    for i in range(len(lv) - len(m) + 1):
        if np.max(np.abs(lv[i:i + len(m)] - m)) <= tol:
            hits.append(i)
    return np.asarray(hits, int)


def place_events(n, kind, on="symbols", grid=None, rng=None, x=None,
                 indices=None, count=None, fraction=None, every=None,
                 which="both", min_run=6, motif=None, intervals=None,
                 rate_hz=None, times=None, samples=None, aggressor=None,
                 n_ui=None, symbols=None, severity=0.5, transitions_only=None,
                 **params):
    """Draw an ``EventList`` for ``kind`` using targeting ``on``. Mechanism knobs
    (amp, f0_hz, tau_s, …) are stored on each event and consumed by ``apply_events``."""
    if kind not in MECHANISMS:
        raise ValueError(f"unknown kind {kind!r}; use one of {MECHANISMS}")
    if on not in PLACEMENTS:
        raise ValueError(f"unknown placement {on!r}; use one of {PLACEMENTS}")
    rng = rng or np.random.default_rng()
    n = int(n)
    sev = float(severity)
    stored = dict(params)
    events = []

    def emit(sample, ui=None, direction=0.0, width=None):
        sample = int(np.clip(sample, 0, max(n - 1, 0)))
        ev = Event(sample=sample, kind=kind, severity=sev, ui=None if ui is None else int(ui),
                   width=width, direction=float(direction), placement=on, params=dict(stored))
        if n_ui is not None or (grid is not None and grid.baud is not None) or symbols is not None:
            try:
                ev.params["_n_ui"] = _n_ui(n, grid, n_ui, symbols)
            except ValueError:
                pass
        events.append(_stamp_time(ev, grid))

    if on == "times":
        if samples is not None:
            smp = np.asarray(samples, int)
        elif times is not None:
            if grid is None:
                raise ValueError("times= in seconds requires grid")
            smp = np.round(np.asarray(times, float) * grid.fs).astype(int)
        else:
            raise ValueError("times placement needs samples= or times=")
        nui = None
        try:
            nui = _n_ui(n, grid, n_ui, symbols)
        except ValueError:
            nui = None
        spb = _spb(n, nui) if nui else None
        for s in smp:
            ui = int(s / spb) if spb else None
            emit(s, ui=ui)

    elif on == "intervals":
        if not intervals:
            raise ValueError("intervals placement needs intervals=[(start, width), ...]")
        for item in intervals:
            start, width = int(item[0]), int(item[1])
            emit(start, width=max(width, 1))

    elif on == "poisson":
        if rate_hz is not None and grid is not None:
            t, duration = 0.0, n * grid.dt
            mean = 1.0 / max(float(rate_hz), 1e-30)
            while True:
                t += float(rng.exponential(mean))
                if t >= duration:
                    break
                emit(int(t * grid.fs))
        else:
            k = int(count) if count is not None else 1
            if n > 0 and k > 0:
                for s in rng.integers(0, n, size=k):
                    emit(int(s))

    elif on == "aggressor":
        if aggressor is None:
            raise ValueError("aggressor placement needs aggressor=array")
        smp, direc = _mid_crossings(np.asarray(aggressor, float))
        if which == "rising":
            m = direc > 0
            smp, direc = smp[m], direc[m]
        elif which == "falling":
            m = direc < 0
            smp, direc = smp[m], direc[m]
        pick = _draw_subset(len(smp), rng, indices, count, fraction, every)
        for i in pick:
            emit(int(smp[i]), direction=float(direc[i]))

    elif on in ("symbols", "edges", "pattern"):
        nui = _n_ui(n, grid, n_ui, symbols)
        spb = _spb(n, nui)
        lv = None if symbols is None else np.asarray(symbols, float)
        if lv is None and x is not None:
            lv = _estimate_levels(x, nui)

        if on == "pattern":
            if lv is None:
                raise ValueError("pattern placement needs symbols= or x=")
            if motif is not None and not isinstance(motif, str):
                starts = _motif_starts(lv, motif)
                pick = _draw_subset(len(starts), rng, indices, count, fraction, every)
                for i in pick:
                    ui = int(starts[i])
                    lo, hi = _ui_bounds(n, ui, nui)
                    nmot = len(np.atleast_1d(motif))
                    hi = _ui_bounds(n, ui + nmot - 1, nui)[1]
                    emit(lo, ui=ui, width=max(1, hi - lo))
            else:
                runs = _runs(lv, min_run=min_run)
                if isinstance(motif, str) and motif in ("high_run", "low_run"):
                    med = float(np.median(lv))
                    runs = [r for r in runs if (r[2] >= med if motif == "high_run" else r[2] < med)]
                pick = _draw_subset(len(runs), rng, indices, count, fraction, every)
                for i in pick:
                    ui, length, _lvl = runs[int(i)]
                    lo, _ = _ui_bounds(n, int(ui), nui)
                    _, hi = _ui_bounds(n, int(ui) + int(length) - 1, nui)
                    emit(lo, ui=int(ui), width=max(1, hi - lo))

        elif on == "edges":
            if lv is not None:
                uis, smp, direc = _symbol_edges(lv, n)
            elif x is not None:
                smp, direc = _mid_crossings(x)
                uis = np.round(smp / spb).astype(int)
            else:
                raise ValueError("edges placement needs x= or symbols=")
            if which == "rising":
                m = direc > 0
                uis, smp, direc = uis[m], smp[m], direc[m]
            elif which == "falling":
                m = direc < 0
                uis, smp, direc = uis[m], smp[m], direc[m]
            pick = _draw_subset(len(smp), rng, indices, count, fraction, every)
            for i in pick:
                ui_i = int(uis[i]) if i < len(uis) else None
                samp = int(_ui_bounds(n, ui_i, nui)[0]) if ui_i is not None else int(smp[i])
                emit(samp, ui=ui_i, direction=float(direc[i]))

        else:  # symbols — indices are absolute UI numbers
            use_tr = transitions_only if transitions_only is not None else (kind == "runt")
            if indices is not None:
                chosen = np.asarray(indices, int)
                chosen = chosen[(chosen >= 0) & (chosen < nui)]
            else:
                if use_tr and lv is not None:
                    pool = (np.where(np.diff(lv) != 0)[0] + 1).astype(int)
                else:
                    pool = np.arange(nui, dtype=int)
                chosen = pool[_draw_subset(len(pool), rng, None, count, fraction, every)]
            direc = np.zeros(len(chosen))
            if lv is not None:
                for j, ui in enumerate(chosen):
                    if 0 < ui < len(lv):
                        direc[j] = float(np.sign(lv[ui] - lv[ui - 1]))
            for ui, d in zip(chosen, direc):
                lo, hi = _ui_bounds(n, int(ui), nui)
                emit(lo, ui=int(ui), direction=float(d), width=max(1, hi - lo))

    else:
        raise ValueError(f"unhandled placement {on!r}")

    return EventList(events, n=n, grid=grid)


def defect_symbols(symbols, events, floor=0.25, depth=None):
    """Source-level mutation of a per-UI level stream (apply *before* edge shaping).

    runt  — scale the excursion from the previous level toward ``floor`` of itself.
    droop — progressive scale-down across a run starting at ``event.ui``.
    """
    y = np.asarray(symbols, float).copy()
    evs = events.events if isinstance(events, EventList) else list(events)
    for ev in evs:
        if ev.ui is None or not (0 <= ev.ui < len(y)):
            continue
        ui = int(ev.ui)
        p = ev.params
        if ev.kind == "runt":
            fl = float(p.get("floor", floor))
            prev = y[ui - 1] if ui > 0 else 0.0
            scale = 1.0 - ev.severity * (1.0 - fl)
            y[ui] = prev + scale * (y[ui] - prev)
        elif ev.kind == "droop":
            dep = float(p.get("depth", depth if depth is not None else ev.severity * 0.35))
            ref = y[ui]
            end = ui + 1
            tol = 0.15 * (np.ptp(y) + 1e-12)
            while end < len(y) and abs(y[end] - ref) <= tol:
                end += 1
            length = max(end - ui, 1)
            for k in range(length):
                y[ui + k] = y[ui + k] * (1.0 - dep * (k + 1) / length)
    return y


# --------------------------------------------------------------- apply
def _ui_slice(n, ev):
    nui = ev.params.get("_n_ui")
    if ev.ui is not None and nui:
        return _ui_bounds(n, int(ev.ui), int(nui))
    if ev.width:
        return ev.span(n)
    return ev.span(n)


def _edge_amp(x, ev, default_frac):
    if "amp" in ev.params:
        return float(ev.params["amp"])
    return ev.severity * default_frac * (np.ptp(x) + 1e-12)


def _f0_tau(ev, grid, x):
    dt = _dt(grid)
    if "f0_hz" in ev.params:
        f0 = float(ev.params["f0_hz"])
    elif grid is not None and grid.baud:
        f0 = 0.25 * grid.baud
    else:
        f0 = 0.08 / dt
    if "tau_s" in ev.params:
        tau = float(ev.params["tau_s"])
    elif "q" in ev.params:
        tau = float(ev.params["q"]) / (np.pi * max(f0, 1e-12))
    elif "zeta" in ev.params:
        z = max(float(ev.params["zeta"]), 1e-6)
        tau = 1.0 / (z * 2.0 * np.pi * max(f0, 1e-12))
    else:
        cycles = float(ev.params.get("cycles", 3.0))
        tau = cycles / max(f0, 1e-12)
    return f0, tau, dt


def _add_kernel(y, ev, kern):
    n = len(y)
    t0 = int(ev.sample)
    if t0 >= n or kern.size == 0:
        ev.width = 0
        return
    hi = min(n, t0 + len(kern))
    y[t0:hi] = y[t0:hi] + kern[:hi - t0]
    ev.width = hi - t0


def _apply_ring_family(y, ev, grid, x0):
    f0, tau, dt = _f0_tau(ev, grid, x0)
    kind = ev.kind
    amp = _edge_amp(x0, ev, 0.25 if kind != "nonmonotonic" else 0.35)
    direc = ev.direction if ev.direction else 1.0
    if kind == "undershoot":
        direc = -direc
    if kind in ("overshoot", "undershoot") and "cycles" not in ev.params and "tau_s" not in ev.params:
        tau = 0.7 / max(f0, 1e-12)
    n_kern = int(max(8, min(len(y), np.ceil(8.0 * tau / dt))))
    if kind == "nonmonotonic":
        # opposing bump during the edge: one half-cycle against the edge direction
        width = ev.params.get("width")
        if width is None and ev.params.get("_n_ui"):
            width = max(4, int(round(0.35 * _spb(len(y), int(ev.params["_n_ui"])))))
        width = int(width or max(6, n_kern // 8))
        k = np.arange(width, dtype=float)
        bump = -direc * amp * np.sin(np.pi * k / max(width - 1, 1))
        _add_kernel(y, ev, bump)
        return
    kern = damped_sinusoid(n_kern, dt, f0, tau, amp=amp * direc, phase=0.0)
    _add_kernel(y, ev, kern)


def _apply_glitch(y, ev, grid, x0):
    dt = _dt(grid)
    amp = _edge_amp(x0, ev, 0.45)
    pol = ev.params.get("polarity", ev.direction if ev.direction else 1.0)
    pol = float(pol) if pol else 1.0
    if "width" in ev.params:
        w = int(ev.params["width"])
    elif ev.params.get("_n_ui"):
        w = max(4, int(round(0.25 * _spb(len(y), int(ev.params["_n_ui"])))))
    else:
        w = 16
    t = np.arange(w, dtype=float) * dt
    # difference of exponentials (same family as impairments.spike), peak-normalized
    tau_r = max(0.15 * w * dt, dt)
    tau_f = max(0.55 * w * dt, dt)
    g = np.exp(-t / tau_f) - np.exp(-t / tau_r)
    g = g / (np.max(np.abs(g)) + 1e-12)
    _add_kernel(y, ev, pol * amp * g)


def _apply_runt(y, ev, grid, x0):
    n = len(y)
    lo, hi = _ui_slice(n, ev)
    if hi <= lo:
        ev.width = 0
        return
    prev = y[lo - 1] if lo > 0 else y[lo]
    fl = float(ev.params.get("floor", 0.25))
    scale = 1.0 - ev.severity * (1.0 - fl)
    y[lo:hi] = prev + scale * (y[lo:hi] - prev)
    hold = ev.params.get("hold_frac")
    if hold is not None and float(hold) < 1.0:
        cut = lo + max(1, int(round(float(hold) * (hi - lo))))
        if cut < hi:
            tau = max((hi - cut) / 6.0, 1.0)
            k = np.arange(hi - cut)
            env = np.exp(-k / tau)
            y[cut:hi] = prev + (y[cut:hi] - prev) * env
    ev.width = hi - lo


def _apply_droop(y, ev, grid, x0):
    n = len(y)
    lo, hi = ev.span(n)
    if ev.ui is not None and ev.width:
        lo, hi = ev.span(n)
    if hi <= lo:
        # fall back to one UI
        lo, hi = _ui_slice(n, ev)
    if hi <= lo:
        ev.width = 0
        return
    depth = float(ev.params.get("depth", ev.severity * 0.35))
    dt = _dt(grid)
    if "tau_s" in ev.params:
        tau = max(float(ev.params["tau_s"]), dt)
        k = np.arange(hi - lo) * dt
        env = 1.0 - depth * (1.0 - np.exp(-k / tau))
    else:
        k = np.arange(hi - lo) / max(hi - lo - 1, 1)
        env = 1.0 - depth * k
    if "floor" in ev.params:
        y[lo:hi] = y[lo:hi] * env + float(ev.params["floor"]) * (1.0 - env)
    else:
        y[lo:hi] = y[lo:hi] * env
    ev.width = hi - lo


def _apply_slow_edge(y, ev, grid, x0):
    n = len(y)
    factor = float(ev.params.get("tr_factor", 1.0 + 3.0 * ev.severity))
    if ev.params.get("_n_ui"):
        half = int(round(0.55 * _spb(n, int(ev.params["_n_ui"]))))
        search_w = max(half * 2, int(round(_spb(n, int(ev.params["_n_ui"])))))
    else:
        half = int(ev.params.get("width", 16))
        search_w = max(half * 2, 16)
    half = max(half, 4)
    t0 = int(ev.sample)
    # find the visible transition near the symbol boundary (causal shaping delays it)
    lo_s, hi_s = max(0, t0 - search_w), min(n, t0 + search_w)
    seg = y[lo_s:hi_s]
    if seg.size > 3:
        g = np.diff(seg)
        if ev.direction:
            g = np.where(np.sign(g) == np.sign(ev.direction), g, 0.0)
        if np.any(g):
            t0 = lo_s + int(np.argmax(np.abs(g)))
            ev.sample = t0
    lo, hi = max(0, t0 - half), min(n, t0 + half)
    idx = np.arange(lo, hi, dtype=float)
    stretched = np.interp(t0 + (idx - t0) / factor, np.arange(n), y, left=y[0], right=y[-1])
    w = np.hanning(hi - lo)
    y[lo:hi] = y[lo:hi] * (1.0 - w) + stretched * w
    ev.width = hi - lo


_APPLY = {
    "runt": _apply_runt,
    "glitch": _apply_glitch,
    "ring": _apply_ring_family,
    "overshoot": _apply_ring_family,
    "undershoot": _apply_ring_family,
    "nonmonotonic": _apply_ring_family,
    "droop": _apply_droop,
    "slow_edge": _apply_slow_edge,
}


def apply_events(x, events, grid=None):
    """Apply each event's mechanism. Returns ``(y, mask, events)`` where ``y`` is
    bit-identical to ``x`` outside the mask, and each event's ``width`` is filled in."""
    x = np.asarray(x, float)
    y = x.copy()
    evs = events if isinstance(events, EventList) else EventList(list(events), n=len(x), grid=grid)
    evs.n = len(x)
    if evs.grid is None:
        evs.grid = grid
    for ev in evs.events:
        fn = _APPLY.get(ev.kind)
        if fn is None:
            raise ValueError(f"unknown kind {ev.kind!r}")
        fn(y, ev, grid, x)
    mask = evs.mask()
    # exact identity outside support (mask uses post-apply widths)
    out = x.copy()
    m = mask > 0
    out[m] = y[m]
    return out, mask, evs

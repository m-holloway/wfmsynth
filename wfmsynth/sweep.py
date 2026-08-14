"""
wfmsynth.sweep — confounder-controlled sweeps and realized-vs-requested labels.

When you sweep an attribute to build a labelled set, the obvious shortcut has to be held
constant or a model learns the shortcut and scores well for the wrong reason. The classic
trap: adding a reflection closes the eye, so a naive reflection sweep is *also* an
eye-height sweep, and anything trained on it can read eye height instead of ISI structure.

``hold_constant`` sweeps one parameter while pinning a measured metric — solving a second
parameter to keep it fixed. This exposes a real physical constraint: you cannot
independently vary reflection, loss **and** eye height — only two of the three. The caller
must therefore say which one is swept and which one is solved; the third (the metric) is
pinned.

``realized_table`` emits attributes measured from the output alongside the requested knobs,
plus the realized correlation matrix across a generated set — so a leak between nominally
orthogonal knobs is visible in the labels instead of latent in the model.
"""
from __future__ import annotations

import numpy as np


def solve_monotonic(f, target, lo, hi, tol, max_iter=60):
    """Bisection solve of ``f(p) == target`` for a monotonic ``f`` on ``[lo, hi]``. Returns
    ``(p, f(p))``. If ``target`` is outside the reachable range, returns the nearest bound
    (so the caller can see the realized value miss the target rather than get a false hit)."""
    flo, fhi = f(lo), f(hi)
    if (target - flo) * (target - fhi) > 0:                      # target not bracketed
        return (lo, flo) if abs(flo - target) < abs(fhi - target) else (hi, fhi)
    increasing = fhi >= flo
    p, fp = 0.5 * (lo + hi), None
    for _ in range(max_iter):
        p = 0.5 * (lo + hi)
        fp = f(p)
        if abs(fp - target) <= tol:
            break
        if (fp < target) == increasing:
            lo = p
        else:
            hi = p
    return p, fp


def hold_constant(build, vary, values, pin, target, solve, bounds, grid, measure_fn,
                  tol=None):
    """Sweep ``vary`` across ``values`` while holding the measured metric ``pin`` at
    ``target``, by solving ``solve`` within ``bounds`` for each point.

    ``build(**params) -> Signal``; ``measure_fn(waveform, grid) -> float`` measures the
    pinned metric. Returns a list of records, each carrying the requested ``vary`` value,
    the solved ``solve`` value, and the REALIZED metric (``realized_<pin>``) — so you can
    confirm the pin held and see the compensation the constraint forced."""
    tol = tol if tol is not None else 1e-3 * max(abs(target), 1.0)
    out = []
    for v in values:
        def f(s, _v=v):
            return measure_fn(build(**{vary: _v, solve: s}).waveform(), grid)
        s_star, realized = solve_monotonic(f, target, bounds[0], bounds[1], tol)
        out.append({vary: v, solve: s_star,
                    f"realized_{pin}": realized, f"target_{pin}": target})
    return out


def realized_table(build, param_sets, grid, metrics):
    """Generate a signal per entry in ``param_sets`` (each a kwargs dict for ``build``),
    measure realized attributes, and return ``(records, corr, names)``: ``records`` pairs
    requested knobs with ``realized_<metric>`` values; ``corr`` is the realized correlation
    matrix across the metrics — a leak between nominally orthogonal knobs shows up here
    instead of staying latent.

    ``metrics`` is either a dict ``{name: fn(waveform, grid)}`` or a single attributes-style
    callable ``fn(waveform, grid) -> {name: value}`` (e.g. ``measure.attributes``)."""
    is_dict = hasattr(metrics, "items")
    recs, per = [], []
    for kw in param_sets:
        x = build(**kw).waveform()
        vals = ({n: fn(x, grid) for n, fn in metrics.items()} if is_dict
                else dict(metrics(x, grid)))
        per.append(vals)
        recs.append({**kw, **{f"realized_{n}": v for n, v in vals.items()}})
    names = list(per[0]) if per else (list(metrics) if is_dict else [])
    M = np.array([[p[n] for n in names] for p in per], dtype=float)
    corr = np.corrcoef(M.T) if len(recs) > 1 else np.eye(len(names))
    return recs, corr, names

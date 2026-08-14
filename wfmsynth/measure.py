"""
wfmsynth.measure — attributes measured FROM the generated waveform, not computed from the
knobs. The two differ, and that difference is exactly where silent label noise comes from,
so ground-truth labels for training should be measured here rather than read off the recipe.

Eye height is offered under **two named definitions** because they disagree when it matters:

  - ``contour``  the measured vertical opening — the gap between the worst-case samples of
                 adjacent levels. Honest about deterministic ISI (a discrete echo makes
                 distinct branches, not a smear).
  - ``sigma``    a 3-sigma construction per level. Agrees with ``contour`` for roughly
                 Gaussian vertical distributions and **overstates** the opening when ISI is
                 deterministic. Report which one you mean.

All measurements fold the record at the symbol period implied by ``grid`` (fs/baud), search
the sampling phase, and cluster samples into levels — no knowledge of the transmitted bits.
"""
from __future__ import annotations

import numpy as np


def sample_at_phase(x, spb, phase):
    """Sample one point per symbol at ``round(k*spb + phase)``. ``spb`` may be non-integer."""
    n_sym = int((len(x) - 1 - phase) / spb)
    idx = np.round(np.arange(n_sym) * spb + phase).astype(int)
    idx = idx[(idx >= 0) & (idx < len(x))]
    return x[idx]


def _cluster_levels(samps, k):
    """Assign samples to ``k`` amplitude levels (quantile init + a few Lloyd iterations).
    Returns sorted centers and per-sample labels."""
    centers = np.quantile(samps, [(i + 0.5) / k for i in range(k)])
    for _ in range(8):
        lab = np.abs(samps[:, None] - centers[None, :]).argmin(1)
        new = np.array([samps[lab == j].mean() if np.any(lab == j) else centers[j]
                        for j in range(k)])
        new.sort()
        if np.allclose(new, centers):
            break
        centers = new
    lab = np.abs(samps[:, None] - centers[None, :]).argmin(1)
    return centers, lab


def _eye_at_phase(x, spb, phase, levels, defn):
    s = sample_at_phase(x, spb, phase)
    if len(s) < levels * 4:
        return -np.inf
    _, lab = _cluster_levels(s, levels)
    gaps = []
    for j in range(levels - 1):
        lo, hi = s[lab == j], s[lab == j + 1]
        if len(lo) == 0 or len(hi) == 0:
            return -np.inf
        if defn == "contour":
            gaps.append(hi.min() - lo.max())
        elif defn == "sigma":
            gaps.append((hi.mean() - 3 * hi.std()) - (lo.mean() + 3 * lo.std()))
        else:
            raise ValueError(f"unknown eye definition {defn!r} (use 'contour' or 'sigma')")
    return min(gaps)


def eye_height(x, grid, levels=4, defn="contour", n_phases=32):
    """Measured eye height at the best sampling phase. ``defn`` is 'contour' (measured
    opening) or 'sigma' (3-sigma construction) — see the module docstring; they diverge
    under deterministic ISI. ``levels`` is 2 for NRZ, 4 for PAM4."""
    spb = grid.samples_per_ui
    return max(_eye_at_phase(x, spb, ph, levels, defn)
               for ph in np.linspace(0, spb, n_phases, endpoint=False))


def best_phase(x, grid, levels=4, defn="contour", n_phases=32):
    """The sampling phase (in samples, within one UI) that maximizes the eye opening."""
    spb = grid.samples_per_ui
    phases = np.linspace(0, spb, n_phases, endpoint=False)
    return float(phases[int(np.argmax([_eye_at_phase(x, spb, ph, levels, defn)
                                       for ph in phases]))])


def attributes(x, grid, levels=4):
    """A dict of realized attributes measured from ``x`` — both eye definitions plus basic
    amplitude stats. Use as the ``metrics`` source for realized-vs-requested labelling."""
    return {
        "eye_contour": eye_height(x, grid, levels=levels, defn="contour"),
        "eye_sigma": eye_height(x, grid, levels=levels, defn="sigma"),
        "ptp": float(np.ptp(x)),
        "rms": float(np.sqrt(np.mean(x ** 2))),
    }

"""
wfmsynth.compose — provenance-first composable synthesis.

Build a signal as an ordered graph of components, each recording its type and the exact
knob values used, so every waveform carries a complete, serializable **recipe**. The
recipe round-trips: `Signal.from_recipe(r).waveform()` reproduces the samples bit-for-bit
(a property asserted in `wfmsynth.validate`). That turns synthetic data into
ground-truth-to-arbitrary-depth training data — you know precisely what produced each
waveform — and makes datasets reproducible, diffable and auditable.

    g = Grid(fs=256e9, baud=112e9, n=1 << 14)
    sig = (Signal(seed=42, grid=g)
           .carrier("pam4", n_ui=g.n // 8, pattern="prbs13q", jitter=dict(rj=0.4))
           .lossy(loss_db=15.0, loss_at_ghz=26.0, causal=True)
           .reflect(td_ps=55.0, gamma_s=0.4, gamma_l=0.4)
           .digitize(snr_db=32.0, enob=5.5, interleave=dict(m_cores=4, offset_mm=0.01)))
    x, recipe = sig.waveform(), sig.recipe()        # samples + full provenance (JSON-able)
    assert (Signal.from_recipe(recipe).waveform() == x).all()   # exact round-trip

Ops compose over the validated primitives (`physics`, `instrument`), so every knob is a
real, documented parameter — nothing is hidden or randomized-but-unrecorded. Randomness
(jitter, ADC noise) is driven by the Signal's single seed, threaded through the ops in
order, which is what makes the round-trip exact.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

from . import physics as P
from . import instrument as INST
from .grid import Grid


# --------------------------------------------------------------- op executors
def _grid_n(grid, p):
    return p.get("n", grid.n if grid is not None else None)


def _op_carrier(x, p, rng, grid):
    j = p.get("jitter")
    jitter = P.Jitter(**j) if j else None
    common = dict(n_ui=p.get("n_ui", 32), tr_frac=p.get("tr_frac", 0.15),
                  seed=p.get("seed", 1), n=_grid_n(grid, p), causal=p.get("causal", False),
                  jitter=jitter, rng=rng)
    if p["kind"] == "pam4":
        return P.pam4(pattern=p.get("pattern", "legacy"), **common)
    if p["kind"] == "nrz":
        return P.nrz(**common)
    raise ValueError(f"unknown carrier kind {p['kind']!r} (use 'nrz' or 'pam4')")


def _op_lossy(x, p, rng, grid):
    kw = {k: p[k] for k in ("length_in", "tand", "eps_r", "skin_k", "causal",
                            "loss_db", "loss_at_ghz") if k in p}
    return P.lossy_channel(x, grid=grid, **kw)


def _op_reflect(x, p, rng, grid):
    kw = {k: p[k] for k in ("td_frac", "td_samples", "td_ps", "gamma_s", "gamma_l",
                            "n_bounce") if k in p}
    return P.multi_reflection(x, grid=grid, **kw)


def _op_crosstalk(x, p, rng, grid):
    aggr = _op_carrier(None, {"kind": "nrz", **p.get("aggressor", {"n_ui": 32, "seed": 7})},
                       rng, grid)
    return P.crosstalk(x, aggr, coupling=p.get("coupling", 0.1),
                       kind=p.get("kind", "fext"), td_frac=p.get("td_frac", 0.05))


def _op_ac_couple(x, p, rng, grid):
    kw = {k: p[k] for k in ("fc_frac", "fc_hz") if k in p}
    return P.ac_couple(x, grid=grid, **kw)


def _op_digitize(x, p, rng, grid):
    n_out = p.get("n_out")
    if n_out and n_out != len(x):
        x = resample_poly(x, n_out, len(x))
    span = float(np.ptp(x)) + 1e-9
    if "snr_db" in p:
        x = x + rng.normal(0.0, span / 10 ** (p["snr_db"] / 20), len(x))
    if p.get("interleave"):
        x = INST.interleave_adc(x, rng=rng, **p["interleave"])
    if "enob" in p:
        lsb = span / 2 ** p["enob"]
        x = lsb * np.round(x / lsb)
    return x


_EXEC = {"carrier": _op_carrier, "lossy": _op_lossy, "reflect": _op_reflect,
         "crosstalk": _op_crosstalk, "ac_couple": _op_ac_couple, "digitize": _op_digitize}


# --------------------------------------------------------------- the Signal builder
@dataclass
class Signal:
    seed: int = 0
    grid: Optional[Grid] = None
    ops: list = field(default_factory=list)

    def _add(self, op, **params):
        self.ops.append({"op": op, **params}); return self

    def carrier(self, kind, **params):
        """First op: a carrier ('nrz'|'pam4'). params: n_ui, n, seed, tr_frac, causal,
        pattern (pam4), jitter=dict(rj,pj,f_pj,dcd) for source jitter."""
        return self._add("carrier", kind=kind, **params)

    def lossy(self, **params):
        """Lossy channel. params: length_in, tand, causal, or loss_db+loss_at_ghz (real units)."""
        return self._add("lossy", **params)

    def reflect(self, **params):
        """Multi-reflection. params: td_frac | td_samples | td_ps, gamma_s, gamma_l, n_bounce."""
        return self._add("reflect", **params)

    def crosstalk(self, **params):
        """Crosstalk. params: coupling, kind('fext'|'next'), td_frac, aggressor=dict(carrier spec)."""
        return self._add("crosstalk", **params)

    def ac_couple(self, **params):
        """AC-coupling. params: fc_frac | fc_hz."""
        return self._add("ac_couple", **params)

    def digitize(self, **params):
        """Scope digitization. params: n_out, snr_db, enob, interleave=dict(m_cores, gain_mm, ...)."""
        return self._add("digitize", **params)

    def waveform(self):
        """Execute the recipe -> samples. Deterministic given (seed, ops, grid)."""
        rng = np.random.default_rng(self.seed)
        x = None
        for op in self.ops:
            x = _EXEC[op["op"]](x, op, rng, self.grid)
        if x is None:
            raise ValueError("empty Signal: add a carrier first")
        return x

    def recipe(self):
        """A JSON-serializable dict: engine version, seed, grid, and the ordered ops
        with their exact knob values. This IS the ground truth for the waveform."""
        from wfmsynth import __version__
        r = {"wfmsynth_version": __version__, "seed": int(self.seed),
             "ops": [dict(o) for o in self.ops]}
        if self.grid is not None:
            g = self.grid
            r["grid"] = {"fs": g.fs, "baud": g.baud, "n": g.n, "v_full": g.v_full}
        return r

    @classmethod
    def from_recipe(cls, r):
        """Reconstruct a Signal from a recipe; `.waveform()` reproduces bit-for-bit."""
        grid = Grid(**r["grid"]) if r.get("grid") else None
        s = cls(seed=r["seed"], grid=grid)
        s.ops = [dict(o) for o in r["ops"]]
        return s


def dataset(build, n, seed=0):
    """Ground-truth dataset builder. `build(rng)` returns a Signal (sample its knobs from
    `rng` however you like — the SAMPLED values are baked into the Signal's ops, hence
    recorded). Returns (X, recipes): X is (n, L) stacked waveforms, recipes is a list of n
    per-sample recipes. Each waveform is exactly reproducible from its recipe."""
    rng = np.random.default_rng(seed)
    sigs = [build(rng) for _ in range(n)]
    waves = [s.waveform() for s in sigs]
    L = len(waves[0])
    X = np.empty((n, L), np.float32)
    for i, w in enumerate(waves):
        X[i] = w[:L] if len(w) >= L else np.pad(w, (0, L - len(w)))
    return X, [s.recipe() for s in sigs]

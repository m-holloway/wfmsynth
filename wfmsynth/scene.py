"""
wfmsynth.scene — multi-signal composition for realistic multi-lane scenarios.

The single-`Signal` chain models one waveform. Real captures are multi-lane and correlated:
a supply rail couples into several lanes at once, a differential pair is two lanes sharing a
clock, and aggressors are other lanes on the board. `Scene` composes several waveforms with
those SHARED sources, which is what "construct a realistic scenario" needs and a single chain
cannot express. It stays pure synthesis — analysis/root-cause tooling composes scenes, it does
not live here.

    sc = (Scene(grid).add("lane0", w0).add("lane1", w1)
          .shared_supply(f_ripple_hz=1e6, am_depth=0.03)   # one rail -> correlated across lanes
          .couple(into="lane1", frm="lane0", coupling=0.08))
    y0, y1 = sc.lane("lane0"), sc.lane("lane1")
"""
from __future__ import annotations

import numpy as np

from . import physics as P


class Scene:
    """A set of named lanes (waveforms) coupled through shared sources. Methods mutate the
    lanes in place and return ``self`` so they chain."""

    def __init__(self, grid):
        self.grid = grid
        self._lanes: dict = {}

    def add(self, name, waveform):
        self._lanes[name] = np.asarray(waveform, float)
        return self

    def lane(self, name):
        return self._lanes[name]

    def lanes(self):
        return dict(self._lanes)

    def shared_supply(self, f_ripple_hz=1e6, am_depth=0.0, psij_ps=0.0, names=None, supply=None):
        """Couple ONE supply rail into several lanes (default all) — the supply artifact is
        then CORRELATED across those lanes, as on a real board that shares its rails."""
        names = names or list(self._lanes)
        n = len(self._lanes[names[0]])
        s = supply if supply is not None else np.sin(
            2 * np.pi * f_ripple_hz * (np.arange(n) / self.grid.fs))
        for nm in names:
            self._lanes[nm] = P.supply_coupling(self._lanes[nm], self.grid,
                                                am_depth=am_depth, psij_ps=psij_ps, supply=s)
        return self

    def couple(self, into, frm, coupling, kind="fext"):
        """Couple lane ``frm`` (an aggressor) into lane ``into`` (the victim) with the given
        coupling — arbitrary lane-to-lane crosstalk (vs `crosstalk_matrix`, which generates
        its own aggressors)."""
        v = self._lanes[into]
        self._lanes[into] = v + (P.crosstalk(v, self._lanes[frm], coupling=coupling, kind=kind) - v)
        return self

    def differential(self, name, skew_ps=0.0, gain_imbalance=0.0, cm=0.0):
        """Replace lane ``name`` with its differential pair, stored as ``name+'_p'`` and
        ``name+'_n'`` — two lanes that share timing (with controlled intra-pair skew)."""
        p, n = P.differential_pair(self._lanes.pop(name), self.grid,
                                   skew_ps=skew_ps, gain_imbalance=gain_imbalance, cm=cm)
        self._lanes[f"{name}_p"] = p
        self._lanes[f"{name}_n"] = n
        return self

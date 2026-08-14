"""
wfmsynth.streams — role-tagged RNG streams.

A single `rng` threaded through generation is a correctness problem for dataset
construction: any change to one factor re-rolls everything downstream of it, so a pair
that looks controlled ("same data, same noise, different channel") actually differs in
every factor at once, and any conclusion drawn from it is confounded.

`Streams` gives each **role** — symbols, jitter, thermal noise, per-impairment, capture
nuisances — its own independent stream. Each role's stream is a pure function of
`(seed, role-name)`: independent of every other role and of the order roles are
requested. So a sibling waveform can re-roll exactly one factor (via `reroll`) and leave
all other factors bit-identical — which is what makes valid contrastive pairs and clean
ablations possible.

    s = Streams(1234)
    jit = s.role("jitter").standard_normal(64)
    noi = s.role("noise").standard_normal(64)
    s2 = s.reroll("jitter")                     # a sibling with ONLY jitter re-rolled
    assert (s2.role("noise").standard_normal(64) == noi).all()      # noise unchanged
    assert (s2.role("jitter").standard_normal(64) != jit).any()     # jitter re-rolled
"""
from __future__ import annotations

import hashlib

import numpy as np


def _role_seed(name: str) -> int:
    """A stable 64-bit seed derived from a role name. Deterministic and order-free, so
    adding a new role never disturbs the streams of existing roles."""
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "big")


class Streams:
    """Independent RNG streams keyed by role name, all derived from one base seed.

    `overrides` re-seeds individual roles (used by `reroll`); every role not overridden
    stays bit-identical to the base. Streams are cached, so repeated `role(name)` calls
    return the same generator (drawing continues where it left off)."""

    def __init__(self, seed: int = 0, overrides: dict | None = None):
        self.seed = int(seed)
        self._overrides = dict(overrides or {})
        self._cache: dict = {}

    def role(self, name: str) -> np.random.Generator:
        r = self._cache.get(name)
        if r is None:
            sub = self._overrides.get(name, self.seed)
            ss = np.random.SeedSequence([int(sub), _role_seed(name)])
            r = np.random.default_rng(ss)
            self._cache[name] = r
        return r

    def reroll(self, *roles: str, seed: int | None = None) -> "Streams":
        """A sibling `Streams` with the named roles re-seeded and everything else
        identical. With `seed=None` each named role advances to a fresh, deterministic
        stream; pass `seed` to choose the re-roll explicitly (reproducible siblings)."""
        ov = dict(self._overrides)
        base = self.seed + 1 if seed is None else int(seed)
        for i, name in enumerate(roles):
            ov[name] = base + i
        return Streams(self.seed, ov)

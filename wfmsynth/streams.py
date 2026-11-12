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
from typing import Optional

import numpy as np


def _role_seed(name: str) -> int:
    """A stable 64-bit seed derived from a role name. Deterministic and order-free, so
    adding a new role never disturbs the streams of existing roles."""
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "big")


class Made:
    """Every generator a render handed out, with the state it started in. Shared by all the
    `Streams` of one render (a several-channel record makes one per chain, a pinned
    sub-record its own), each noting under the role's full name."""

    def __init__(self, to=None, pre=""):
        self._items = [] if to is None else to._items
        self._pre = pre if to is None else to._pre + pre

    def note(self, name, gen):
        self._items.append((self._pre + name, gen, gen.bit_generator.state))

    def under(self, pre):
        """A view that notes under `pre`: for a sub-record with its own `Streams`."""
        return Made(self, pre)

    def drawn(self):
        return {name for name, gen, start in self._items if gen.bit_generator.state != start}


class Streams:
    """Independent RNG streams keyed by role name, all derived from one base seed.

    `overrides` re-seeds individual roles (used by `reroll`); every role not overridden
    stays bit-identical to the base. Streams are cached, so repeated `role(name)` calls
    return the same generator (drawing continues where it left off).

    `asked` collects every role name a render requested, and `made` every generator it was
    handed, with the state it started in. An op is handed its stream whether or not its
    parameters make it draw (`store` without dither, `acquire` without a noise floor), so
    "asked" is not "drawn": `drawn()` is the roles whose generator has MOVED, which is how
    `contrast()` tells a re-rolled role that changed the record from one that could not."""

    def __init__(self, seed: int = 0, overrides: Optional[dict] = None):
        self.seed = int(seed)
        self._overrides = dict(overrides or {})
        self._cache: dict = {}
        self.asked: set = set()
        self.made = Made()

    def drawn(self):
        """The role names something actually drew from during the renders so far."""
        return self.made.drawn()

    def idle(self, known=()):
        """The re-rolled roles nothing asked for and `known` does not list: names that match
        no randomness in the record, so re-rolling them changed nothing. (A `unit/` role is a
        toleranced part, re-drawn into the recipe rather than asked for here.)"""
        return sorted(k for k in self._overrides
                      if k not in self.asked and k not in known and "unit/" not in k)

    def undrawn(self):
        """The re-rolled roles an op was handed and did not draw from: a real name, on an op
        whose parameters make it deterministic, so the sibling is the record again."""
        drawn = self.drawn()
        return sorted(k for k in self._overrides
                      if k in self.asked and k not in drawn and "unit/" not in k)

    def role(self, name: str) -> np.random.Generator:
        self.asked.add(name)
        r = self._cache.get(name)
        if r is None:
            sub = self._overrides.get(name, self.seed)
            ss = np.random.SeedSequence([int(sub), _role_seed(name)])
            r = np.random.default_rng(ss)
            self._cache[name] = r
            self.made.note(name, r)
        return r

    def reroll(self, *roles: str, seed: Optional[int] = None) -> "Streams":
        """A sibling `Streams` with the named roles re-seeded and everything else
        identical. With `seed=None` each named role advances to a fresh, deterministic
        stream; pass `seed` to choose the re-roll explicitly (reproducible siblings)."""
        ov = dict(self._overrides)
        base = self.seed + 1 if seed is None else int(seed)
        for i, name in enumerate(roles):
            ov[name] = base + i
        return Streams(self.seed, ov)

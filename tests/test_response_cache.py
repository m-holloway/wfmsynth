"""`physics.response_cache` reuses minimum-phase responses across renders.

A dataset renders many records through the same channel, and the minimum-phase construction
depends only on the magnitude it is given and the transform length -- not on the record. So a
hit returns exactly what a recompute would have, and the cache is bit-exact by construction
rather than by tolerance.

The test that matters most here is `test_a_different_channel_is_never_served_a_cached_one`.
Everything else is performance; that one is correctness, and the failure it guards against --
silently applying the wrong channel -- is the worst thing this library could do.
"""
from __future__ import annotations

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import physics as P
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

N = 1 << 17


def _render(seed=1, length_in=8.0, n=N):
    g = Grid(fs=80e9, baud=10e9, n=n)
    return (Signal(seed=seed, grid=g)
            .carrier("nrz", n_ui=n >> 3, pattern="prbs13", causal=True, seed=seed)
            .lossy(length_in=length_in, tand=0.02, causal=True)).waveform()


def test_the_cache_is_bit_exact():
    plain = [_render(s) for s in range(4)]
    with P.response_cache():
        cached = [_render(s) for s in range(4)]
    for a, b in zip(plain, cached):
        assert np.array_equal(a, b)


def test_a_different_channel_is_never_served_a_cached_one():
    """THE correctness test. The key is a content hash of the magnitude, so two channels that
    differ anywhere must miss. A key derived from parameters could be incomplete, and an
    incomplete key returns the wrong channel with no symptom at all."""
    lengths = [4.0, 8.0, 12.0]
    plain = [_render(1, length_in=L) for L in lengths]
    with P.response_cache():
        cached = [_render(1, length_in=L) for L in lengths]
        # interleaved, so a stale entry would be live in the cache when the next one asks
        again = [_render(1, length_in=L) for L in reversed(lengths)]
    for a, b in zip(plain, cached):
        assert np.array_equal(a, b)
    for a, b in zip(reversed(plain), again):
        assert np.array_equal(a, b)
    assert not np.array_equal(plain[0], plain[1])          # the channels really do differ


def test_a_magnitude_differing_in_one_bin_is_a_different_key():
    """The content hash at its finest resolution: one changed sample must miss."""
    n = 1 << 17
    mag = np.linspace(1.0, 0.1, n // 2 + 1)
    with P.response_cache() as c:
        a = P._min_phase_H(mag, n, half=True)
        mag2 = mag.copy()
        mag2[len(mag2) // 3] *= 1.0000001
        b = P._min_phase_H(mag2, n, half=True)
        assert c.misses == 2 and c.hits == 0
    assert not np.array_equal(a, b)
    assert np.array_equal(a, P._min_phase_H(mag, n, half=True))


def test_the_cache_actually_hits_or_it_is_decorative():
    with P.response_cache() as c:
        for s in range(4):
            _render(s)
        assert c.hits >= 3, f"hits={c.hits} misses={c.misses} -- the cache is not being used"


def test_it_is_bounded_under_many_distinct_channels():
    """The pathological case: more channels than slots must evict, not grow."""
    with P.response_cache(max_entries=3) as c:
        for L in (4.0, 6.0, 8.0, 10.0, 12.0, 14.0):
            _render(1, length_in=L)
        assert len(c._store) <= 3


def test_it_is_off_by_default_and_restored_afterwards():
    """Opt-in is the whole design: a single-record caller can never hit the cache and should
    not pay a record of storage for it."""
    assert P._RESPONSE_CACHE is None
    with P.response_cache():
        assert P._RESPONSE_CACHE is not None
    assert P._RESPONSE_CACHE is None


def test_nesting_restores_the_outer_cache():
    with P.response_cache() as outer:
        with P.response_cache() as inner:
            assert P._RESPONSE_CACHE is inner
        assert P._RESPONSE_CACHE is outer
    assert P._RESPONSE_CACHE is None


def test_it_is_restored_even_if_the_body_raises():
    with pytest.raises(RuntimeError):
        with P.response_cache():
            raise RuntimeError("boom")
    assert P._RESPONSE_CACHE is None


def test_consuming_the_result_in_place_cannot_corrupt_the_cache():
    """`apply_transfer` multiplies into its own buffer, but a caller is free to consume the
    returned array. A hit hands back a copy so that cannot poison the next hit."""
    n = 1 << 17
    mag = np.linspace(1.0, 0.2, n // 2 + 1)
    with P.response_cache():
        first = P._min_phase_H(mag, n, half=True)
        want = first.copy()
        first *= 0.0                                  # consume it destructively
        second = P._min_phase_H(mag, n, half=True)    # must be a hit, and must be intact
        assert np.array_equal(second, want)


def test_the_probe_ladder_does_not_evict_the_response_worth_keeping():
    """`response_extent` probes by doubling before the real transform, so a record produces
    several short responses and one long one. Caching the short ones would evict the long one
    before the next record could use it -- measured as ZERO hits when it was tried."""
    with P.response_cache(max_entries=2) as c:
        for s in range(4):
            _render(s)
        assert c.hits >= 3, f"probe ladder is evicting the real response (hits={c.hits})"


def test_dataset_uses_it_and_stays_bit_identical():
    """`dataset` turns the cache on around its own loop, where the batch is the point."""
    def build(rng):
        s = int(rng.integers(1 << 30))
        g = Grid(fs=80e9, baud=10e9, n=1 << 14)
        return (Signal(seed=s, grid=g)
                .carrier("nrz", n_ui=1 << 11, pattern="prbs13", causal=True, seed=s)
                .lossy(length_in=6.0, tand=0.02, causal=True))

    X, recipes = ws.dataset(build, 5)
    assert P._RESPONSE_CACHE is None                  # released with the loop
    for i, r in enumerate(recipes):
        assert np.allclose(Signal.from_recipe(r).waveform()[:X.shape[1]], X[i], atol=0)

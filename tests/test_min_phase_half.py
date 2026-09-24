"""`_min_phase_H` runs at the RECORD's own length, so its temporaries decided whether a deep
record rendered at all -- it was the largest single allocation in a render.

Three record-length arrays were removed without touching the arithmetic: the symmetric mirror of
the magnitude (real and even, so `irfft` computes its inverse DFT from the half already in hand),
the complex cepstrum whose real part was copied out (`irfft` returns the real sequence directly),
and the causal-folding weight vector (entries exactly 1.0/2.0/0.0, all exact in binary floating
point, so a slice applies them in place). `half=True` additionally returns only the rfft half,
which is all a filtering caller wants.

These tests pin the PROPERTIES the construction exists for -- the magnitude it reproduces and the
causality it buys -- plus the half/full agreement, so the identity cannot silently rot.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfmsynth.physics import _min_phase_H


def _mags(m):
    f = np.arange(m, dtype=float)
    return {
        "lossy": 10.0 ** (-(0.4 * np.sqrt(f) + 0.02 * f) / 20.0),
        "flat": np.ones(m),
        "rippled": np.abs(np.cos(f / m * 9.0)) + 1e-4,
        "floored": np.full(m, 1e-13),
    }


def _half_len(n):
    return n // 2 + 1 if n % 2 == 0 else (n + 1) // 2


@pytest.mark.parametrize("n", [16, 17, 4096, 4097, 65536, 65537])
def test_half_is_exactly_the_first_half_of_the_full_result(n):
    """`half=True` must be the same numbers, not merely a similar response -- it is the same
    transform restricted to the bins a real-input FFT already produces. Both parities, because
    an odd length has no distinct Nyquist bin and the folding differs there."""
    m = _half_len(n)
    for name, mag in _mags(m).items():
        full = _min_phase_H(mag, n)
        half = _min_phase_H(mag, n, half=True)
        assert half.shape == (m,), f"{name}: n={n}"
        assert np.abs(half - full[:m]).max() < 1e-12 * max(1.0, np.abs(full).max()), name


@pytest.mark.parametrize("n", [16, 17, 4096, 4097])
def test_the_magnitude_is_preserved_by_the_cepstral_round_trip(n):
    """A minimum-phase response has the SAME magnitude as the one it was built from -- only the
    phase is added. That is the whole point of the construction, and it is the property that
    would break first if the folding or the transform parity were wrong."""
    m = _half_len(n)
    for name, mag in _mags(m).items():
        if name == "floored":
            continue                       # at 1e-13 the +1e-12 guard dominates, by design
        H = _min_phase_H(mag, n, half=True)
        assert np.abs(np.abs(H) - mag).max() < 1e-9 * mag.max(), f"{name}: n={n}"


@pytest.mark.parametrize("n", [256, 257, 4096])
def test_the_result_is_causal_energy_after_t0(n):
    """The reason to do any of this: the response must concentrate AFTER t=0. A zero-phase
    magnitude is symmetric about it, which is the non-causal construction this replaces."""
    m = _half_len(n)
    mag = _mags(m)["lossy"]
    h = np.fft.ifft(_min_phase_H(mag, n)).real
    assert np.sum(h[:n // 2] ** 2) > np.sum(h[n // 2:] ** 2)


def test_the_full_result_is_hermitian_so_its_impulse_response_is_real():
    """The two-sided form must still be the spectrum of a REAL sequence. If the folding lost
    the symmetry, the impulse response would pick up an imaginary part and every downstream
    convolution would be quietly wrong."""
    n = 4096
    mag = _mags(_half_len(n))["lossy"]
    h = np.fft.ifft(_min_phase_H(mag, n))
    assert np.abs(h.imag).max() < 1e-12 * np.abs(h.real).max()


def test_a_flat_magnitude_gives_a_flat_zero_phase_response():
    """|H| = 1 everywhere is already minimum phase: the cepstrum is zero, so the response is
    exactly 1 and the folding must not invent a phase."""
    n = 1024
    H = _min_phase_H(np.ones(_half_len(n)), n, half=True)
    assert np.abs(H - 1.0).max() < 1e-9

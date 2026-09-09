"""Layers 1-2 for U-16 -- a frequency-domain stage is a LINEAR convolution, and its transform
length is not forced to be the record length.

Two claims, and each is checked against an answer CONSTRUCTED here rather than against a
previous run:

  1. THE ARITHMETIC. `apply_transfer` must return `np.convolve(x, h)[:len(x)]` for a response
     whose taps this file wrote down. `numpy.convolve` is an independent implementation in the
     time domain, so agreement is cross-instrument agreement, not a tautology.
  2. THE PHYSICS. A path of two matched lines around one discontinuity has an exact closed
     form at the driver plane -- `x[k] + Gamma * x[k - 2D]` for a whole-sample one-way delay
     `D` -- with no loss, no dispersion and no approximation anywhere in it. The realised
     waveform must be that, sample for sample.

Both are stated twice: once for the linear default (which must recover the constructed answer)
and once for `linear=False` (which must NOT, and whose error must be exactly the wrapped tail,
in exactly the first `len(h) - 1` samples). A check that has only ever been seen passing is not
a check, so several assertions below are re-run against a deliberately broken fixture -- a
guard that is one sample too short, a smooth-length search that skips a factor -- and the test
fails if the broken fixture is accepted.
"""
import numpy as np
import pytest

import wfmsynth.physics as P
import wfmsynth.sparam as SP
from wfmsynth.grid import Grid

FS = 256e9
GRID = Grid(fs=FS, baud=16e9, n=8192)


# ============================================================ 1. the transform length itself
def test_next_smooth_length_is_the_smallest_smooth_number_at_or_above_n():
    """Checked against brute force, which is a different algorithm for the same answer."""
    def brute(n, radix=(2, 3, 5)):
        m = n
        while True:
            k = m
            for r in radix:
                while k % r == 0:
                    k //= r
            if k == 1:
                return m
            m += 1
    for n in list(range(1, 400)) + [1023, 1024, 1025, 8191, 8192, 65537, 131056]:
        assert P.next_smooth_length(n) == brute(n), n


def test_a_smooth_length_is_actually_smooth_and_never_shrinks_the_record():
    for n in (8191, 131056, 3_000_017, 33_550_336):
        m = P.next_smooth_length(n)
        assert m >= n
        k = m
        for r in (2, 3, 5):
            while k % r == 0:
                k //= r
        assert k == 1, f"{m} is not 5-smooth"
        assert m < n * 1.07, f"{m}/{n} = {m/n:.3f}: too much padding for a smooth length"


def test_the_smooth_search_rejects_a_radix_it_was_not_given():
    """The BROKEN FIXTURE for `next_smooth_length`: 7-smooth numbers must not be accepted when
    only 2, 3 and 5 were asked for. 8191*7 = 57337 is 7-smooth-ish and 5-rough."""
    assert P.next_smooth_length(105) == 108           # 105 = 3*5*7 -- 7 is not in the radix
    assert P.next_smooth_length(105, radix=(2, 3, 5, 7)) == 105
    assert P.next_smooth_length(49, radix=(7,)) == 49
    assert P.next_smooth_length(50, radix=(7,)) == 343


def test_linear_fft_length_is_at_least_the_linear_convolution_length():
    for n, g in ((8192, 37), (131056, 5000), (1 << 20, 1 << 20)):
        assert P.linear_fft_length(n, g) >= n + g - 1


# ============================================================ 2. the arithmetic, constructed
def _fir(seed=0, n=8191, taps=37):
    """A record of PRIME length (8191 -- so the old path took numpy's Bluestein branch) and a
    tap list. The last tap is forced large so the measured extent cannot miss it."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(taps)
    h[-1] = 1.5
    return rng.standard_normal(n), h


def test_apply_transfer_returns_the_linear_convolution_numpy_computes_in_the_time_domain():
    x, h = _fir()
    got = P.apply_transfer(x, lambda nf: np.fft.rfft(h, nf))
    exact = np.convolve(x, h)[:len(x)]
    assert np.abs(got - exact).max() < 1e-12, f"max|err| {np.abs(got - exact).max():.3e}"


def test_the_circular_path_does_not_and_its_error_is_exactly_the_wrapped_tail():
    """The old path, stated as the thing it gets wrong: the response to the record's last
    `len(h)-1` samples is delivered to its first ones."""
    x, h = _fir()
    circ = P.apply_transfer(x, lambda nf: np.fft.rfft(h, nf), linear=False)
    exact = np.convolve(x, h)[:len(x)]
    err = circ - exact
    assert np.abs(err).max() > 1.0, "the circular path is supposed to be WRONG here"
    # the wrapped part is the tail of the full convolution, folded back onto the head
    full = np.convolve(x, h)
    assert np.abs(err[:len(h) - 1] - full[len(x):]).max() < 1e-12
    assert np.abs(err[len(h) - 1:]).max() < 1e-12, "the wrap must not reach past the head"


def test_the_measured_guard_is_the_tap_count_and_too_short_a_guard_is_observed_failing():
    """The BROKEN FIXTURE for the padding. A guard the caller sets too small must corrupt the
    head, and corrupt EXACTLY the samples the shortfall predicts -- otherwise the padding is
    not what is preventing the wrap and this whole file proves nothing.

    The record is 8191 samples (prime) and the response 37 taps, so the linear convolution
    needs 8227. With `guard=1` the transform lands on 8192, which is 35 short, and the last 35
    samples of the full convolution wrap onto the first 35."""
    x, h = _fir()
    mk = lambda nf: np.fft.rfft(h, nf)
    assert P.response_extent(mk) == len(h)
    exact = np.convolve(x, h)[:len(x)]
    assert np.abs(P.apply_transfer(x, mk, guard=len(h)) - exact).max() < 1e-12

    nfft = P.linear_fft_length(len(x), 1)
    assert nfft == 8192                                    # the shortfall this test relies on
    short_by = len(x) + len(h) - 1 - nfft
    assert short_by == 35
    bad = np.abs(P.apply_transfer(x, mk, guard=1) - exact)
    assert bad[:short_by].min() > 1e-3, (
        f"a guard {short_by} samples short was accepted (head error {bad[0]:.2e})")
    assert bad[short_by:].max() < 1e-12, "and it should corrupt only the head it is short by"


def test_response_extent_finds_a_two_sided_response_not_just_a_late_tail():
    """A zero-phase response is symmetric about t=0, so half of it sits at the END of the
    array. Measuring 'the last index above threshold' would call that a full-length tail."""
    taps = 1.0 / (1.0 + np.arange(65))                 # 1 .. 1/65, both sides of t=0
    sym = lambda nf: np.fft.rfft(np.concatenate([taps, np.zeros(nf - 129), taps[:0:-1]]), nf)
    ext = P.response_extent(sym, n0=1024, rel=1e-3)
    assert 128 <= ext <= 130, f"support of a +/-64 tap response measured as {ext}"
    # and the linear convolution is right for it, which the "last loud index" rule would break
    x = np.random.default_rng(2).standard_normal(1000)
    full = np.concatenate([taps[::-1], taps[1:]])          # h[m] for m = -64 .. +64
    exact = np.convolve(x, full)[64:64 + len(x)]           # shift the -64 origin back to 0
    got = P.apply_transfer(x, sym, rel=1e-3)
    assert np.abs(got - exact).max() < 1e-12


# ============================================================ 3. the physics, constructed
def _exact_path(gamma, d_samples):
    """A lossless path: matched line, one discontinuity, matched line. At the driver plane the
    cascade's S11 is exactly `gamma * exp(-j*2*pi*f*2*td)`, so (1 + S11) is the two-tap FIR
    `delta[k] + gamma*delta[k - 2*d_samples]` and the answer is arithmetic."""
    td_ps = d_samples / FS * 1e12
    return [{"line": {"td_ps": td_ps}}, {"disc": {"gamma": gamma}}, {"line": {"td_ps": td_ps}}]


@pytest.mark.parametrize("gamma,d", [(0.055, 72), (0.30, 16), (0.12, 501)])
def test_a_driver_plane_echo_is_the_two_tap_answer_the_topology_predicts(gamma, d):
    x = P.nrz(n_ui=256, n=GRID.n, seed=3, tr_frac=0.15)
    exact = x + gamma * np.concatenate([np.zeros(2 * d), x[:GRID.n - 2 * d]])
    got = SP.cascade_channel(x, _exact_path(gamma, d), grid=GRID, node="source")
    assert np.abs(got - exact).max() < 1e-12, (
        f"gamma={gamma} d={d}: max|err| {np.abs(got - exact).max():.3e}")


@pytest.mark.parametrize("gamma,d", [(0.055, 72), (0.30, 16)])
def test_the_circular_cascade_delivers_that_echo_to_the_wrong_end_of_the_record(gamma, d):
    """The defect, as a number: the echo of the record's last 2*d samples arrives at its
    first 2*d samples, at full amplitude, and nowhere else."""
    x = P.nrz(n_ui=256, n=GRID.n, seed=3, tr_frac=0.15)
    exact = x + gamma * np.concatenate([np.zeros(2 * d), x[:GRID.n - 2 * d]])
    circ = SP.cascade_channel(x, _exact_path(gamma, d), grid=GRID, node="source", linear=False)
    err = circ - exact
    assert np.abs(err[:2 * d] - gamma * x[GRID.n - 2 * d:]).max() < 1e-12
    assert np.abs(err[2 * d:]).max() < 1e-12
    assert np.abs(err).max() > 0.9 * gamma * np.abs(x).max()


def test_a_matched_path_is_the_identity_through_both_forms():
    """Negative control: no discontinuity and no loss is a pure delay of zero, so both the
    linear and the circular form must hand the record straight back."""
    x = P.nrz(n_ui=256, n=GRID.n, seed=3)
    path = [{"line": {"td_ps": 0.0}}]
    for kw in ({}, {"linear": False}):
        y = SP.cascade_channel(x, path, grid=GRID, node="source", **kw)
        assert np.abs(y - x).max() < 1e-12, kw


# ============================================================ 4. the two paths carry one H
def test_the_linear_and_circular_paths_apply_the_SAME_response_in_steady_state():
    """Padding changes the convolution's GEOMETRY, not the channel. For a periodic input whose
    period divides the record, the two forms agree exactly once the response's own length has
    passed -- which is the statement that no loss or phase was added or lost by padding."""
    n = GRID.n
    d, gamma = 96, 0.2
    path = _exact_path(gamma, d)
    k = 512                                     # 512 cycles in the record: exactly periodic
    x = np.sin(2.0 * np.pi * k * np.arange(n) / n)
    lin = SP.cascade_channel(x, path, grid=GRID, node="source")
    circ = SP.cascade_channel(x, path, grid=GRID, node="source", linear=False)
    assert np.abs(lin[2 * d:] - circ[2 * d:]).max() < 1e-12
    assert np.abs(lin[:2 * d] - circ[:2 * d]).max() > 1e-3      # and differ only in the head


def test_padding_does_not_change_the_loss_a_channel_applies():
    """The same claim for the analytic channel, read as dB at a tone: measured on the pinned
    length (where a tone is an eigenvector) against the padded record's steady state."""
    g = Grid(fs=FS, baud=16e9, n=1 << 14)
    kw = dict(grid=g, loss_db=12.0, loss_at_ghz=8.0, causal=True)
    k = int(round(8e9 * g.n / g.fs))
    x = np.sin(2.0 * np.pi * k * np.arange(g.n) / g.n)
    circ = P.lossy_channel(x, linear=False, **kw)
    lin = P.lossy_channel(x, **kw)
    ref = -20.0 * np.log10(abs(np.fft.rfft(circ)[k]) / abs(np.fft.rfft(x)[k]))
    assert abs(ref - 12.0) < 1e-9, f"the pinned-length measurement itself is off: {ref:.6f} dB"
    # the padded record's amplitude, fitted over its second half (past the turn-on)
    m = np.arange(g.n // 2, g.n)
    basis = np.vstack([np.cos(2 * np.pi * k * m / g.n), np.sin(2 * np.pi * k * m / g.n)]).T
    amp = lambda y: np.hypot(*np.linalg.lstsq(basis, y[g.n // 2:], rcond=None)[0])
    got = -20.0 * np.log10(amp(lin) / amp(x))
    assert abs(got - 12.0) < 0.02, f"padded record measures {got:.4f} dB, asked for 12.00"


# ============================================================ 5. the length is not forced
def test_a_record_of_prime_length_is_transformed_at_a_smooth_length():
    """The second half of U-16: the transform length is chosen, not inherited. 8191 is prime,
    so the pinned path handed numpy a length with no direct algorithm."""
    x, h = _fir(n=8191, taps=9)
    nfft = P.linear_fft_length(len(x), P.response_extent(lambda nf: np.fft.rfft(h, nf)))
    assert nfft != len(x)
    k = nfft
    for r in (2, 3, 5):
        while k % r == 0:
            k //= r
    assert k == 1, f"chose {nfft}, which is not 5-smooth"


def test_the_op_is_indifferent_to_the_records_own_factorisation():
    """The same channel and the same signal content at a prime length and at a smooth one give
    the same samples where they overlap -- so a caller no longer has to choose between a record
    length the pattern wants and one the FFT likes."""
    rng = np.random.default_rng(4)
    h = rng.standard_normal(21)
    base = rng.standard_normal(8191)
    mk = lambda nf: np.fft.rfft(h, nf)
    a = P.apply_transfer(base, mk)
    b = P.apply_transfer(np.concatenate([base, np.zeros(8192 - 8191 + 1)]), mk)[:8191]
    assert np.abs(a - b).max() < 1e-12

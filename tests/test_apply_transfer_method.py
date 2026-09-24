"""`apply_transfer(method=)` chooses HOW the same linear convolution is computed.

The whole-record transform pays log(n) for a filter that may only reach a few thousand samples,
and it allocates several record-length arrays to do it. Overlap-save computes the same linear
convolution a block at a time, so the working set is the block. What it is NOT is bit-compatible:
the two paths sample the response on different frequency grids, so they sit 7.7e-5 of peak-to-peak
apart for this stage alone -- and a whole 8-bit code on 1.16 % of samples once a converter sees
it -- which `test_byte_identity.py` classifies as a behaviour change, not as round-off. Hence
`method="fft"` stays the default and the choice is recorded in the recipe.

These tests pin the equivalence (to a MEASURED tolerance), the cases the circular-to-linear tap
extraction can get wrong (a two-sided response, a pure delay), the auto rule, and the refusals.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from wfmsynth import physics as P

FS = 80e9
N = 1 << 17

# MEASURED. The two paths differ by 7.7e-5 of peak-to-peak for this stage alone on a long
# record, 2.5e-4 through a full chain, and a whole 8-bit code on 1.16 % of samples once a
# converter sees it -- see `apply_transfer`'s docstring. Tests here run at a shorter record
# where the fraction is larger, so the bar is loose; `test_the_disagreement_stays_the_size_it_
# is_documented_as` below is what actually pins the number.
EQUIV_TOL = 2e-3


def _signal(n=N, seed=1):
    return np.asarray(P.nrz(n_ui=n // 8, tr_frac=0.4, seed=seed, n=n), float)[:n]


def _causal_channel(length_in=8.0):
    """A minimum-phase (causal) channel: the arc starts at lag 0."""
    def make_H(nfft):
        il = P.insertion_loss_db(np.fft.rfftfreq(nfft) * FS / 1e9,
                                 length_in=length_in, tand=0.02)
        return P._min_phase_H(10.0 ** (-il / 20.0), nfft, half=True)
    return make_H


def _zero_phase_channel(length_in=8.0):
    """A magnitude-only response: symmetric about t=0, so half its arc is at NEGATIVE lag."""
    def make_H(nfft):
        il = P.insertion_loss_db(np.fft.rfftfreq(nfft) * FS / 1e9,
                                 length_in=length_in, tand=0.02)
        return (10.0 ** (-il / 20.0)).astype(complex)
    return make_H


def _pure_delay(samples):
    """An all-pass pure delay: the arc is a single tap at a POSITIVE lag."""
    def make_H(nfft):
        f = np.fft.rfftfreq(nfft)
        return np.exp(-2j * np.pi * f * samples)
    return make_H


def _rel(a, b):
    return np.abs(a - b).max() / np.ptp(b)


def test_overlap_matches_the_transform_for_a_causal_channel():
    x = _signal()
    mk = _causal_channel()
    assert _rel(P.apply_transfer(x, mk, method="overlap"),
                P.apply_transfer(x, mk, method="fft")) < EQUIV_TOL


def test_overlap_matches_the_transform_for_a_two_sided_response():
    """The case circular-to-linear tap extraction gets wrong if it assumes causality: a
    zero-phase response reaches BEFORE t=0, and those taps live at the end of the array."""
    x = _signal()
    mk = _zero_phase_channel()
    assert _rel(P.apply_transfer(x, mk, method="overlap"),
                P.apply_transfer(x, mk, method="fft")) < EQUIV_TOL


@pytest.mark.parametrize("d", [1, 37, 4096])
def test_a_pure_delay_lands_on_the_same_sample_on_both_paths(d):
    """A delay is the sharpest test of the lag bookkeeping: the answer is the input shifted by
    exactly `d`, so an off-by-one in the arc's start would be unmissable."""
    x = _signal(1 << 16)
    y = P.apply_transfer(x, _pure_delay(d), method="overlap")
    want = np.zeros_like(x)
    want[d:] = x[:len(x) - d]
    assert np.abs(y - want).max() < 1e-6 * np.ptp(x)


def test_the_two_paths_agree_on_where_the_energy_is():
    """Equivalence in the aggregate as well as sample by sample -- a path that silently
    dropped or shifted the response's tail could still pass a loose max-difference bar."""
    x = _signal()
    mk = _causal_channel()
    a = P.apply_transfer(x, mk, method="overlap")
    b = P.apply_transfer(x, mk, method="fft")
    assert abs(np.sqrt((a * a).mean()) / np.sqrt((b * b).mean()) - 1.0) < 1e-3


def test_fft_remains_the_default():
    """The compatibility bar: not passing `method` must render exactly what it always did."""
    x = _signal()
    mk = _causal_channel()
    assert np.array_equal(P.apply_transfer(x, mk), P.apply_transfer(x, mk, method="fft"))


def test_auto_takes_the_block_path_only_when_the_response_is_short_against_the_record():
    """`auto` is a rule about the FILTER's memory against the record, never about the signal."""
    x = _signal()
    short = _causal_channel(length_in=8.0)          # a few thousand samples of memory
    assert np.array_equal(P.apply_transfer(x, short, method="auto"),
                          P.apply_transfer(x, short, method="overlap"))

    # a record under the blocking threshold keeps the transform whatever the response is
    xs = _signal(1 << 12)
    assert np.array_equal(P.apply_transfer(xs, short, method="auto"),
                          P.apply_transfer(xs, short, method="fft"))


def test_an_unknown_method_raises_naming_the_choices():
    with pytest.raises(ValueError, match="unknown method"):
        P.apply_transfer(_signal(1 << 12), _causal_channel(), method="bogus")


def test_overlap_refuses_the_circular_form_rather_than_ignoring_it():
    """`linear=False` is the pinned CIRCULAR convolution. Overlap-save is linear by
    construction, so the combination is a contradiction and must be refused rather than
    silently resolved one way."""
    with pytest.raises(ValueError, match="no circular form"):
        P.apply_transfer(_signal(1 << 12), _causal_channel(), linear=False, method="overlap")


def test_the_block_path_holds_far_less_than_the_record():
    """The reason the option exists. Peak memory is a deterministic function of the code, so
    this is a real bound rather than a noisy one."""
    import tracemalloc
    n = 1 << 20
    x = _signal(n)
    mk = _causal_channel()

    def peak(method):
        tracemalloc.start()
        P.apply_transfer(x, mk, method=method)
        pk = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return pk / (n * 8.0)

    ov, fft = peak("overlap"), peak("fft")
    # MEASURED at 1 M samples: 1.2x for the block path against 4.6x for the transform.
    assert ov < 1.6, f"block path peaked at {ov:.1f}x the record"
    assert ov < fft / 2.0, f"block path {ov:.1f}x vs transform {fft:.1f}x -- no memory win"


def test_circular_arc_finds_a_wrapped_response():
    """The primitive under the tap extraction: an arc that straddles the wrap must report the
    start on the far side, not index 0."""
    loud = np.zeros(64, bool)
    loud[60:] = True            # lags -4..-1
    loud[:3] = True             # lags 0..2
    start, width = P._circular_arc(loud)
    assert (start, width) == (60, 7)


def test_circular_arc_agrees_with_the_support_it_replaced():
    """`_circular_support` is now `_circular_arc`'s width; the audited number must not move."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        loud = rng.random(64) < rng.uniform(0.02, 0.5)
        assert P._circular_support(loud) == P._circular_arc(loud)[1]


def test_a_short_response_at_a_long_delay_does_not_wrap_on_the_block_path():
    """A response that is a short arc sitting at a LONG delay -- what a cascade section with a
    real line length builds -- is the case the two paths part company on, and the block path is
    the one that is right.

    `response_extent` measures the arc's WIDTH, and `linear_fft_length` pads by that width, so a
    transform sized from it is too short by the arc's LAG: the record's tail, delayed, lands
    past the padded length and wraps onto the head at full amplitude (MEASURED: 0.95 of peak at
    index 16415 of a 65536-sample record, with a 20000-sample delay). The block path carries
    `lag_min` explicitly and puts nothing there. This is pinned as the block path's CORRECTNESS
    property; the transform path's wrap is a pre-existing defect logged in BACKLOG.md, and
    fixing it changes rendered output for every delayed response, so it is not done here."""
    n = 1 << 16
    d = 20000

    def mk(nfft):
        f = np.fft.rfftfreq(nfft)
        il = P.insertion_loss_db(f * 80.0, length_in=4.0, tand=0.02)
        return (10.0 ** (-il / 20.0)) * np.exp(-2j * np.pi * f * d)

    x = np.zeros(n)
    x[-200:] = 1.0                       # content only at the record's END
    y = P.apply_transfer(x, mk, method="overlap")
    # the response to that content arrives 20000 samples later, i.e. past the record: nothing
    # it produces may appear before the tail content itself could have reached the output
    assert np.abs(y[:n - 200]).max() < 1e-6, "the delayed tail wrapped onto the head"


def test_the_disagreement_stays_the_size_it_is_documented_as():
    """The block path's docstring quotes a number to justify not being the default, so that
    number needs a gate: it was found once to be understated by 3x, which is how a reasonable
    trade turns into a surprise.

    Pins the disagreement on THIS stage from both sides. A lower bound matters as much as an
    upper one -- if it silently fell to round-off, the honest move would be to make the block
    path the default, and nobody would notice it had become free."""
    n = 1 << 19
    fs = 80e9
    x = np.asarray(P.nrz(n_ui=n >> 3, seed=1, n=n, causal=True), float)

    def mk(nfft):
        il = P.insertion_loss_db(np.fft.rfftfreq(nfft) * fs / 1e9, length_in=8.0, tand=0.02)
        return P._min_phase_H(10.0 ** (-il / 20.0), nfft, half=True)

    a = P.apply_transfer(x, mk, method="fft")
    b = P.apply_transfer(x, mk, method="overlap")
    rel = np.abs(a - b).max() / np.ptp(a)
    assert 2e-5 < rel < 3e-4, f"documented as ~7.7e-5 of peak-to-peak, measured {rel:.2e}"


def test_tightening_the_truncation_threshold_reduces_the_disagreement():
    """The docstring used to claim this floor does NOT fall as `rel` tightens. It does -- by
    about 2.3x before it plateaus -- and the claim is load-bearing, because it is the reason a
    caller is told whether buying accuracy back is possible at all."""
    n = 1 << 19
    fs = 80e9
    x = np.asarray(P.nrz(n_ui=n >> 3, seed=1, n=n, causal=True), float)

    def mk(nfft):
        il = P.insertion_loss_db(np.fft.rfftfreq(nfft) * fs / 1e9, length_in=8.0, tand=0.02)
        return P._min_phase_H(10.0 ** (-il / 20.0), nfft, half=True)

    a = P.apply_transfer(x, mk, method="fft")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loose = np.abs(a - P.apply_transfer(x, mk, method="overlap", rel=1e-5)).max()
        tight = np.abs(a - P.apply_transfer(x, mk, method="overlap", rel=1e-7)).max()
    # Both must genuinely take the block path. Past about rel=1e-9 the taps stop being
    # resolvable within the probe cap and `apply_transfer` falls back to the transform with a
    # guard of its own -- which would make this comparison fft-against-fft and pass for the
    # wrong reason.
    assert not any("falling back" in str(c.message) for c in caught), \
        "this test must compare two BLOCK-path results, not a fallback"
    assert tight < loose, f"tightening rel did not help: {loose:.2e} -> {tight:.2e}"
    assert tight > 0.3 * loose, "it should plateau, not vanish -- the residue is not truncation"

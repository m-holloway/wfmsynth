"""PAM-N for any N, PRBS starting phase, and the analog/arbitrary carriers.

Three capabilities, and the thing each one is easy to get wrong:

  PAM-N      the LEVEL COUNT and its spacing. PAM4 standing in for PAM3 or PAM5 gives a
             signal with the wrong number of eyes, which is a data error and not a
             documentation one.
  phase      every record starting at the same place in the pattern lets a model learn the
             position instead of the signal. A real capture starts wherever the link was.
  analog     a hard step synthesised straight onto a sample grid carries content above
             Nyquist. These have to be band-limited or they are spurs pretending to be signal.
"""
import numpy as np
import pytest

import wfmsynth.physics as P
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid


# --------------------------------------------------------------------- PAM-N

@pytest.mark.parametrize("n_levels", [2, 3, 4, 5, 8])
def test_pam_n_has_n_levels_evenly_spaced_over_plus_minus_one(n_levels):
    lv = P.pam_levels(n_levels)
    assert lv.size == n_levels
    assert lv[0] == pytest.approx(-1.0) and lv[-1] == pytest.approx(1.0)
    if n_levels > 2:                            # uniform spacing: an M-level eye per gap
        gaps = np.diff(lv)
        assert gaps.max() - gaps.min() < 1e-12, f"PAM{n_levels} spacing is not uniform"
    assert lv.size - 1 == n_levels - 1          # M-1 eyes, stated so it cannot drift


def test_pam4_levels_are_the_historical_array_to_the_bit():
    """np.linspace(-1, 1, 4) differs from [-1, -1/3, 1/3, 1] by 5.6e-17, and every PAM4
    waveform this kernel has produced came from the explicit array. Computing them would
    move output that is pinned."""
    assert np.array_equal(P.pam_levels(4), np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]))
    assert not np.array_equal(P.pam_levels(4), np.linspace(-1.0, 1.0, 4))


@pytest.mark.parametrize("n_levels", [3, 5, 8])
def test_pam_n_symbols_use_every_level_roughly_evenly(n_levels):
    """Rejection sampling keeps the distribution uniform for an N that is not a power of
    two. If it did not, the outer levels would be over-represented and the eye asymmetric."""
    syms = P.pam_symbols(n_levels, 6000, seed=3)
    u, c = np.unique(syms, return_counts=True)
    assert u.size == n_levels, f"PAM{n_levels} produced {u.size} distinct levels"
    assert np.array_equal(u, P.pam_levels(n_levels))
    assert c.min() / c.max() > 0.85, f"levels unbalanced: {c.tolist()}"


def test_pam3_is_ternary_and_pam5_quinary_through_a_recipe():
    g = Grid(fs=10e9, baud=1e9, n=8192)
    for kind, want in (("pam3", 3), ("pam5", 5)):
        sig = Signal(seed=7, grid=g).carrier(kind, n_ui=512)
        assert sig.waveform().size == 8192
        got = np.unique(P.carrier_symbols(kind, 512, 7))
        assert got.size == want, f"{kind} rendered {got.size} levels"


def test_an_unknown_carrier_kind_names_what_is_available():
    with pytest.raises(ValueError, match="pam<N>"):
        P.carrier_symbols("pam", 64, 1)


# --------------------------------------------------------------------- phase

def test_phase_advances_the_pattern_by_exactly_that_many_symbols():
    base = P.carrier_symbols("nrz", 200, seed=3)
    for k in (1, 7, 63):
        shifted = P.carrier_symbols("nrz", 200, seed=3, phase=k)
        assert np.array_equal(base[k:k + 50], shifted[:50]), f"phase={k} did not shift by {k}"


def test_phase_defaults_to_zero_so_pinned_output_cannot_move():
    for kind in ("nrz", "pam4"):
        assert np.array_equal(P.carrier_symbols(kind, 128, 5),
                              P.carrier_symbols(kind, 128, 5, phase=0))


def test_random_phase_draws_distinct_starting_positions():
    """The LFSR's non-zero state space IS its phase space, so a uniform state is a uniform
    starting position -- and it costs nothing, unlike stepping the generator."""
    rng = np.random.default_rng(0)
    seeds = [P.random_phase(7, rng) for _ in range(200)]
    assert all(1 <= s <= 127 for s in seeds), "state 0 is the dead state and must not appear"
    base = (P.prbs(7, 127, seed=1) > 0).astype(int)
    starts = set()
    for sd in seeds[:40]:
        b = (P.prbs(7, 127, seed=sd) > 0).astype(int)
        k = [i for i in range(127) if np.array_equal(np.roll(base, -i), b)]
        starts.update(k[:1])
    assert len(starts) > 20, f"only {len(starts)} distinct starting positions in 40 draws"


def test_seed_zero_and_one_are_the_same_sequence_which_is_a_trap():
    """`seed & mask or 1` sends state 0 to 1, so two records seeded 0 and 1 are ONE draw,
    not two. Pinned downstream output means this cannot be changed, so it is pinned here
    instead -- a caller reaching for two seeds should not reach for these two."""
    assert np.array_equal(P.prbs(7, 64, seed=0), P.prbs(7, 64, seed=1))


# --------------------------------------------------------------------- analog

G = Grid(fs=10e9, baud=1e9, n=8192)


def test_sine_hits_the_frequency_it_was_asked_for():
    x = P.sine(f_hz=250e6, grid=G)
    f = np.fft.rfftfreq(x.size, 1.0 / 10e9)
    assert f[np.argmax(np.abs(np.fft.rfft(x))[1:]) + 1] == pytest.approx(250e6, rel=0.02)
    assert x.max() == pytest.approx(1.0, abs=1e-3)


def test_cycles_is_an_alternative_to_hertz():
    x = P.sine(cycles=8, n=1024)
    zero_crossings = np.sum(np.diff(np.sign(x)) != 0)
    assert zero_crossings in (15, 16, 17), f"{zero_crossings} crossings for 8 cycles"


@pytest.mark.parametrize("maker", [
    lambda: P.square(f_hz=200e6, grid=G),
    lambda: P.sawtooth(f_hz=200e6, grid=G),
])
def test_the_stepped_shapes_are_band_limited(maker):
    """A hard step on a sample grid puts energy at Nyquist that is not in the signal. The
    edge shaping has to hold it down; 5 % of total is the bar."""
    x = maker()
    X = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(x.size, 1.0 / 10e9)
    near_nyquist = X[f > 0.9 * 5e9].sum() / X[1:].sum()
    assert near_nyquist < 0.05, f"{100*near_nyquist:.2f} % of energy above 0.9x Nyquist"


def test_square_duty_moves_the_mean():
    lo = P.square(f_hz=100e6, duty=0.25, grid=G).mean()
    hi = P.square(f_hz=100e6, duty=0.75, grid=G).mean()
    assert lo < -0.3 < 0.3 < hi, f"duty had no effect: {lo:.3f} vs {hi:.3f}"


def test_triangle_symmetry_turns_it_into_a_ramp():
    sym = P.triangle(f_hz=100e6, symmetry=0.5, grid=G)
    ramp = P.triangle(f_hz=100e6, symmetry=1.0, grid=G)
    # a symmetric triangle rises and falls equally; a ramp mostly rises
    assert np.mean(np.diff(sym) > 0) == pytest.approx(0.5, abs=0.05)
    assert np.mean(np.diff(ramp) > 0) > 0.9


def test_dc_is_constant_and_survives_the_pipeline():
    assert np.ptp(P.dc(0.4, grid=G)) == 0.0
    y = Signal(seed=1, grid=G).carrier("dc", level=0.4).lossy(loss_db=6.0).waveform()
    assert y.size == G.n and np.all(np.isfinite(y))


def test_arbitrary_takes_a_user_function_of_seconds():
    x = P.arbitrary(lambda t: np.exp(-t / 2e-7) * np.sin(2 * np.pi * 5e7 * t), grid=G)
    assert x.size == G.n
    # a decaying envelope: the back half must be quieter than the front
    assert x[G.n // 2:].std() < 0.5 * x[:G.n // 2].std()


def test_arbitrary_refuses_a_function_that_returns_the_wrong_length():
    with pytest.raises(ValueError, match="expected"):
        P.arbitrary(lambda t: np.zeros(7), grid=G)


def test_arbitrary_needs_a_callable_through_a_recipe():
    with pytest.raises(ValueError, match="callable"):
        Signal(seed=1, grid=G).carrier("arbitrary", fn=None).waveform()


def test_every_analog_kind_runs_through_a_recipe_with_an_impairment():
    for kind in P.ANALOG_KINDS:
        sig = Signal(seed=3, grid=G).carrier(kind, f_hz=200e6).lossy(loss_db=6.0,
                                                                     loss_at_ghz=1.0)
        x = sig.waveform()
        assert x.size == G.n and np.all(np.isfinite(x)), f"{kind} did not render"
        assert [o["op"] for o in sig.recipe()["ops"]] == ["carrier", "lossy"]

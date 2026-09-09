"""The cascaded channel: topology, not one lumped block.

The defect these pin: a shipped recipe read `lossy -> reflect`, one lumped loss then one lumped
reflection, so every echo arrived having paid the WHOLE channel's loss exactly once no matter
where it came from -- and a near and a far reflection of equal Gamma came out the same size.
`test_lumped_model_cannot_tell_near_from_far` is that defect, kept as a test.
"""
import numpy as np
import pytest

import wfmsynth.physics as P
import wfmsynth.sparam as SP
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

GRID = Grid(fs=256e9, baud=16e9, n=1 << 13)
FREQ = np.fft.rfftfreq(GRID.n, d=GRID.dt)
EPS, GAM, TOTAL = 4.0, 0.055, 12.0
PPI = SP.ps_per_inch(EPS)


def _pulse():
    sym = np.zeros(64); sym[8] = 1.0
    p = P.from_symbols(sym, n=int(round(64 * GRID.samples_per_ui)), tr_frac=0.15)
    x = np.zeros(GRID.n); x[:len(p)] = p
    return x, p


def _path(d, eps=EPS, gamma=GAM, causal=True):
    return [{"line": {"length_in": d, "eps_r": eps, "causal": causal}},
            {"disc": {"gamma": gamma}},
            {"line": {"length_in": TOTAL - d, "eps_r": eps, "causal": causal}}]


# ------------------------------------------------------------------ the algebra
@pytest.mark.parametrize("g", [0.0, 0.02, 0.055, 0.3, 0.7])
def test_a_lossless_step_is_unitary(g):
    """A clean impedance step reflects and transmits; it must not absorb."""
    d = SP.discontinuity(FREQ, g)
    assert np.allclose(np.abs(d.s11) ** 2 + np.abs(d.s21) ** 2, 1.0, atol=1e-14)
    assert np.allclose(d.s22, -d.s11)          # the step seen from the other side


def test_one_discontinuity_pays_its_segments_loss_exactly_twice():
    """S11 of line(d) -> disc(G) -> line = G * H_segment(f)^2, with NO tolerance to spare.
    This is the whole physical claim: out and back through the segment, not through the channel."""
    tp = SP.cascade(_path(1.66), FREQ)
    H = SP.line(FREQ, length_in=1.66, eps_r=EPS).s21
    assert np.abs(tp.s11 - GAM * H ** 2).max() < 1e-15


def test_two_discontinuities_match_the_hand_derived_bounce_series():
    G1, G2, L1, L2 = 0.055, 0.055, 1.0, 9.0
    path = [{"line": {"length_in": L1, "eps_r": EPS}}, {"disc": {"gamma": G1}},
            {"line": {"length_in": L2, "eps_r": EPS}}, {"disc": {"gamma": G2}},
            {"line": {"length_in": TOTAL - L1 - L2, "eps_r": EPS}}]
    tp = SP.cascade(path, FREQ)
    H1 = SP.line(FREQ, length_in=L1, eps_r=EPS).s21
    Lw = SP.line(FREQ, length_in=L2, eps_r=EPS).s21
    hand = H1 ** 2 * (G1 + (1 - G1 ** 2) * G2 * Lw ** 2 / (1 + G1 * G2 * Lw ** 2))
    assert np.abs(tp.s11 - hand).max() < 1e-14


@pytest.mark.parametrize("path", [_path(1.0), _path(6.0),
                                  [{"line": {"length_in": 1.0}}, {"disc": {"gamma": 0.3}},
                                   {"line": {"length_in": 5.0}}, {"disc": {"gamma": 0.4}},
                                   {"line": {"length_in": 2.0}}]])
def test_the_cascade_is_passive(path):
    # The bound is 1e-9 and not machine epsilon for a stated reason: the causal loss goes through
    # `physics._min_phase_H`, whose cepstral round trip preserves |H| to ~6e-12 relative (measured),
    # so |S11|^2+|S21|^2 inherits ~1.2e-11 of slack. Anything materially above that is a real
    # passivity violation -- an amplifying channel -- not numerics.
    tp = SP.cascade(path, FREQ)
    assert (np.abs(tp.s11) ** 2 + np.abs(tp.s21) ** 2).max() <= 1.0 + 1e-9


def test_a_matched_path_is_the_lumped_lossy_channel_plus_a_delay():
    """One line section, no discontinuities: the cascade must reproduce `lossy_channel` exactly.
    Two implementations of the same loss law that agree is what keeps `insertion_loss_db` shared
    rather than copied."""
    x, _ = _pulse()
    lump = P.lossy_channel(x, length_in=6.0, eps_r=EPS, causal=True, grid=GRID)
    casc = SP.cascade_channel(x, [{"line": {"length_in": 6.0, "eps_r": EPS, "td_ps": 0.0}}],
                              grid=GRID, node="load")
    assert np.abs(casc - lump).max() < 1e-12


# ------------------------------------------------------------------ the physics
def test_first_order_echoes_states_the_answer_before_the_simulation():
    path = [{"line": {"length_in": 1.0}}, {"disc": {"gamma": 0.055}},
            {"line": {"length_in": 9.0}}, {"disc": {"gamma": 0.055}},
            {"line": {"length_in": 2.0}}]
    e = SP.first_order_echoes(path)
    assert len(e) == 2
    assert e[0]["delay_ps"] == pytest.approx(2 * 1.0 * PPI, rel=1e-12)
    assert e[1]["delay_ps"] == pytest.approx(2 * 10.0 * PPI, rel=1e-12)
    assert e[0]["amp"] == pytest.approx(0.055)
    # the far echo transmits through the near junction twice
    assert e[1]["amp"] == pytest.approx(0.055 * (1 - 0.055 ** 2))


@pytest.mark.parametrize("d", [1.0, 1.66, 4.0, 10.0])
def test_the_echo_arrives_at_the_arithmetics_lag(d):
    """A discontinuity at a KNOWN distance lands at 2*d*ps_per_inch. Zero-phase loss isolates
    the transport arithmetic from the causal loss's own dispersion (which is the next test)."""
    x, tmpl = _pulse()
    y = SP.cascade_channel(x, _path(d, causal=False), grid=GRID, node="source")
    c = np.abs(np.correlate(y - x, tmpl, mode="full")[len(tmpl) - 1:])
    c[:int(0.6 * 2 * d * PPI * 1e-12 * GRID.fs)] = 0.0
    k = int(np.argmax(c)); a, b, e = c[k - 1], c[k], c[k + 1]
    lag_ps = (k + 0.5 * (a - e) / (a - 2 * b + e)) * GRID.dt * 1e12
    # within one sample of the arithmetic (the residual is a fixed matched-filter bias)
    assert lag_ps == pytest.approx(2 * d * PPI, abs=GRID.dt * 1e12)


@pytest.mark.parametrize("d", [1.0, 1.66, 4.0, 10.0])
def test_the_echo_carries_the_segments_attenuation_not_the_channels(d):
    """The realised echo equals the closed form Gamma*H_seg^2 applied to the launched pulse."""
    x, _ = _pulse()
    k0 = int(np.argmax(np.abs(x)))
    # The realisation and the closed form are applied the same way -- a LINEAR convolution
    # (`physics.apply_transfer`) at the SAME transform length. Both parts matter: comparing a
    # linear realisation against a circular closed form differs by the wrapped tail (5e-8 here,
    # and far more on a shorter record), and `line(causal=True)` folds its minimum phase at the
    # transform length, so evaluating the two at different lengths differs by 4e-7. Neither is
    # the claim under test, which is the echo's ATTENUATION.
    pad = GRID.n
    y = SP.cascade_channel(x, _path(d), grid=GRID, node="source", guard=pad)
    sk = k0 + int(0.6 * 2 * d * PPI * 1e-12 * GRID.fs)
    got = np.abs((y - x)[sk:]).max()
    seg = lambda nf: GAM * SP.line(np.fft.rfftfreq(nf, d=GRID.dt), length_in=d, eps_r=EPS).s21 ** 2
    exact = np.abs(P.apply_transfer(x, seg, guard=pad)).max()
    assert got == pytest.approx(exact, abs=1e-12)


def test_near_and_far_at_equal_gamma_come_out_different_sizes():
    """The physics the lumped model cannot express, stated as an inequality."""
    x, _ = _pulse()
    k0 = int(np.argmax(np.abs(x)))
    amp = {}
    for d in (0.5, 10.0):
        y = SP.cascade_channel(x, _path(d), grid=GRID, node="source")
        amp[d] = np.abs((y - x)[k0 + int(0.6 * 2 * d * PPI * 1e-12 * GRID.fs):]).max()
    # 0.5 in pays 2 x 0.86 dB, 10 in pays 2 x 17.3 dB -> nearly 7x
    assert amp[0.5] / amp[10.0] > 5.0
    il_n = P.insertion_loss_db(np.array([8.0]), length_in=0.5, eps_r=EPS)[0]
    il_f = P.insertion_loss_db(np.array([8.0]), length_in=10.0, eps_r=EPS)[0]
    assert il_f > il_n                       # and it is the loss that separates them


def test_lumped_model_cannot_tell_near_from_far():
    """THE DEFECT, kept. `lossy` then `reflect` gives a 0.5-inch and a 10-inch echo the SAME
    amplitude, because the loss was already spent before the reflection was made. If this ever
    starts failing, `physics.multi_reflection` has grown positional structure and the note on
    `Signal.reflect` telling callers to use `cascade` instead needs revisiting."""
    x, _ = _pulse()
    k0 = int(np.argmax(np.abs(x)))
    xl = P.lossy_channel(x, length_in=TOTAL, eps_r=EPS, causal=True, grid=GRID)
    amp = {}
    for d in (0.5, 10.0):
        y = P.multi_reflection(xl, td_ps=d * PPI, gamma_s=GAM, gamma_l=GAM, n_bounce=6,
                               grid=GRID, node="source")
        amp[d] = np.abs((y - xl)[k0 + int(0.6 * 2 * d * PPI * 1e-12 * GRID.fs):]).max()
    # Flat to 1e-6: the residual is `multi_reflection` rounding td_ps to a whole sample, which
    # moves the search window, not the amplitude. The cascade's ratio for these same two
    # distances is >5x (test above); the lumped model's is 1.000000.
    assert amp[0.5] == pytest.approx(amp[10.0], rel=1e-6)     # flat: no positional structure


def test_cascade_delay_is_exact_where_multi_reflection_rounds_to_a_sample():
    """`multi_reflection` rounds td_ps to the nearest sample; the cascade is a linear phase and
    has no such grid. At 256 GSa/s a sample is 3.9 ps = 11.5 mil of one-way distance."""
    td = 1.25 * GRID.dt * 1e12                      # round trip 2.5 samples: off the grid
    x, tmpl = _pulse()
    y = SP.cascade_channel(x, [{"line": {"length_in": 0.0, "td_ps": td}},
                               {"disc": {"gamma": 0.2}},
                               {"line": {"length_in": 0.0, "td_ps": 0.0}}],
                           grid=GRID, node="source")
    c = np.abs(np.correlate(y - x, tmpl, mode="full")[len(tmpl) - 1:])
    k = int(np.argmax(c)); a, b, e = c[k - 1], c[k], c[k + 1]
    lag = (k + 0.5 * (a - e) / (a - 2 * b + e)) * GRID.dt * 1e12
    assert lag == pytest.approx(2 * td, abs=0.4 * GRID.dt * 1e12)
    assert abs(lag / (GRID.dt * 1e12) - round(lag / (GRID.dt * 1e12))) > 0.1   # not on the grid


# ------------------------------------------------------------------ the plumbing
def test_the_cascade_op_round_trips_through_a_recipe():
    import json
    path = [{"line": {"length_in": 1.0, "eps_r": EPS}}, {"disc": {"gamma": GAM}},
            {"line": {"length_in": 11.0, "eps_r": EPS}}]
    s = Signal(seed=1, grid=GRID).carrier("nrz", n_ui=512).cascade(path, node="source")
    a = s.waveform()
    ops = json.loads(json.dumps(s.ops))                     # plain JSON, as a recipe stores it
    b = Signal(seed=1, grid=GRID, ops=ops).waveform()
    assert np.array_equal(a, b)


def test_a_measured_section_can_sit_inside_a_synthetic_path(tmp_path):
    """The 'both' half of the design: a real connector's .s2p as ONE section of a cascade."""
    f = np.linspace(0, 40e9, 401)
    S = np.zeros((len(f), 2, 2), complex)
    S[:, 0, 0] = S[:, 1, 1] = 0.1
    S[:, 1, 0] = S[:, 0, 1] = np.sqrt(1 - 0.01) * np.exp(-1j * 2 * np.pi * f * 20e-12)
    p = tmp_path / "conn.s2p"
    SP.write_touchstone(str(p), f, S)
    tp = SP.cascade([{"line": {"length_in": 1.0, "eps_r": EPS}},
                     {"file": {"path": str(p)}},
                     {"line": {"length_in": 5.0, "eps_r": EPS}}], FREQ)
    inb = (FREQ > 1e9) & (FREQ < 15e9)
    assert np.abs(tp.s11[inb]).max() > 0.01          # the measured section's Gamma is present
    assert (np.abs(tp.s11) ** 2 + np.abs(tp.s21) ** 2).max() <= 1.0 + 1e-9


def test_cascade_rejects_a_malformed_path():
    with pytest.raises(ValueError):
        SP.cascade([], FREQ)
    with pytest.raises(ValueError):
        SP.cascade([{"wat": {}}], FREQ)
    with pytest.raises(ValueError):
        SP.cascade([{"line": {"length_in": 1}, "disc": {"gamma": 0.1}}], FREQ)
    with pytest.raises(ValueError):
        SP.cascade_channel(np.zeros(64), [{"line": {"length_in": 1}}])       # no grid/dt

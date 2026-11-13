"""A record that idles at a level must not open on a false edge (#100).

Every linear stage that starts the record "as if the input before it were zero" puts an event at
the head that is not in the recipe: the probe's ground-lead ring, `lossy`'s sag at BOTH ends,
the echo that has not yet arrived, a CTLE charging up from zero. The check here is the one the
issue states: a flat input, at several levels, comes out flat -- the first and last samples equal
the interior, and the interior is the stage's DC gain times the level -- for every stage that
can be driven.
"""
from __future__ import annotations

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import baseband as BB
from wfmsynth import physics as P
from wfmsynth import rx as RX
from wfmsynth.compose import Signal

G = ws.Grid(fs=1e9, baud=125e6, n=4000)
LEVELS = (3.3, -1.7, 0.4)
F = np.logspace(5, 9.5, 60)
S21 = 10 ** (-(F / 1e9) * 3 / 20) * np.exp(-1j * 2 * np.pi * F * 2e-9)

NOT_LINEAR = {"agc"}

# name -> (how to build the stage on a Signal, edge tolerance relative to the level).
# The tolerance is how far the first and last samples may sit from the interior. It is round-off
# for every stage whose response ends inside the guard, and a stated 2e-3 for `lossy`, whose
# sqrt(f) skin term has an impulse response that decays algebraically: the record holds the line
# out as far as the response's measured extent (-100 dB), and the tail past that is the residual.
STAGES = {
    "scope": (lambda s: s.scope(bw_hz=50e6), 1e-9),
    "scope_gaussian": (lambda s: s.scope(bw_hz=50e6, kind="gaussian"), 1e-6),
    "scope_zero_phase": (lambda s: s.scope(bw_hz=50e6, causal=False), 1e-9),
    "input_bandwidth": (lambda s: s.input_bandwidth(bw_hz=50e6), 1e-9),
    "probe": (lambda s: s.probe(), 1e-9),
    "probe_ground_lead": (lambda s: s.probe(l_gnd_h=3e-9, c_load_f=20e-12, r_source=90), 5e-5),
    "probe_compensation": (lambda s: s.probe(compensate=0.7, c_load_f=10e-12, r_source=1000), 5e-5),
    "probe_bandwidth": (lambda s: s.probe(bw_hz=100e6), 1e-9),
    "lossy": (lambda s: s.lossy(loss_db=3, loss_at_ghz=0.1), 2e-3),
    "lossy_causal": (lambda s: s.lossy(loss_db=3, loss_at_ghz=0.1, causal=True), 2e-3),
    "lossy_overlap": (lambda s: s.lossy(loss_db=3, loss_at_ghz=0.1, method="overlap"), 1e-9),
    "lossy_causal_overlap": (lambda s: s.lossy(loss_db=3, loss_at_ghz=0.1, causal=True,
                                               method="overlap"), 1e-9),
    "reflect": (lambda s: s.reflect(td_ps=100e3), 1e-9),
    "reflect_source": (lambda s: s.reflect(td_ps=100e3, node="source"), 1e-9),
    "reflect_fractional": (lambda s: s.reflect(td_ps=100.3e3), 1e-9),
    "resonant_reflect": (lambda s: s.resonant_reflect(td_ps=1e3, f0_ghz=0.2), 5e-5),
    "sparam": (lambda s: s.sparam(freqs=F, s21=S21), 2e-3),
    "cascade": (lambda s: s.cascade([{"line": {"td_ps": 200e3, "loss_db": 1, "loss_at_ghz": .1}},
                                     {"disc": {"gamma": 0.2}}]), 2e-3),
    "ctle": (lambda s: s.ctle(fz_ghz=0.05, fp1_ghz=0.2, fp2_ghz=0.4), 1e-9),
    "de_emphasis": (lambda s: s.de_emphasis(db=-3), 1e-9),
    "tx_ffe": (lambda s: s.tx_ffe([0.1, 0.8, 0.1]), 1e-9),
    "tx_ffe_fractional": (lambda s: s.tx_ffe([0.1, 0.8, 0.1], pre=0), 1e-9),
    "rx_ffe": (lambda s: s.rx_ffe([0.1, 0.8, 0.1], spacing_ui=1.0), 1e-9),
    "rx_ffe_half_ui": (lambda s: s.rx_ffe([0.2, 0.6, 0.2], spacing_ui=0.5, pre=1), 1e-9),
    "crosstalk_fext": (lambda s: s.crosstalk(), 1e-9),
    "hybrid_echo": (lambda s: s.hybrid_echo(td_ps=40e3), 5e-5),
    "intra_pair_skew": (lambda s: s.intra_pair_skew(skew_ps=300e3), 1e-9),
    "delay": (lambda s: s.delay(by=3.4e-8), 1e-9),
    "sample_clock": (lambda s: s.sample_clock(), 1e-9),
    "digitize_resample": (lambda s: s.digitize(n_out=1000), 1e-9),
    "timebase": (lambda s: s.timebase(), 1e-9),
    "supply_coupling": (lambda s: s.supply_coupling(), 1e-9),
    "ac_couple": (lambda s: s.ac_couple(fc_hz=1e6), 1e-9),
    "agc": (lambda s: s.agc(target=1.0), 1e-9),      # normalises: no fixed gain
    "filter_bessel": (lambda s: s.filter("bessel", f_hz=50e6), 1e-9),
    "filter_bessel_zero_phase": (lambda s: s.filter("bessel", f_hz=50e6, zero_phase=True), 1e-9),
    "filter_fir": (lambda s: s.filter("fir", taps=[0.2, 0.6, 0.2]), 1e-9),
    "filter_fir_zero_phase": (lambda s: s.filter("fir", taps=[0.2, 0.6, 0.2],
                                                 zero_phase=True), 1e-9),
    "filter_rrc": (lambda s: s.filter("rrc", beta=0.35), 1e-9),
    "filter_doppler": (lambda s: s.filter("doppler", f_hz=1e6), 1e-9),
}


def _render(build, level):
    return np.asarray(build(Signal(seed=1, grid=G).carrier("dc", level=level)).waveform(), float)


@pytest.mark.parametrize("name", sorted(STAGES))
def test_a_flat_input_stays_flat_at_both_ends(name):
    """Edge samples equal the interior to the stage's tolerance, at every level; and the
    stage is linear in the level, so its DC gain is the same number at each."""
    build, tol = STAGES[name]
    gains = []
    for level in LEVELS:
        y = _render(build, level)
        mid = y[len(y) // 2]
        edges = np.r_[y[:3], y[-3:]]
        assert np.abs(edges - mid).max() <= tol * abs(level) + 1e-12, (name, level, edges, mid)
        gains.append(mid / level)
    if name not in NOT_LINEAR:
        assert np.allclose(gains, gains[0], rtol=1e-6, atol=1e-9), (name, gains)


def test_the_issue_snippet():
    """The numbers the issue quotes, now flat: 3.3 V in, 3.3 V out at the ends."""
    g = ws.Grid(fs=400e6, n=4000)
    from wfmsynth import instrument as I
    y = I.probe(np.full(4000, 3.3), g, l_gnd_h=3e-6, c_load_f=20e-12, r_source=90)
    assert np.abs(y - 3.3).max() < 1e-4
    g = ws.Grid(fs=1e9, n=4000)
    def idle():
        return Signal(seed=1, grid=g).carrier("dc", level=3.3)
    y = idle().lossy(loss_db=3, loss_at_ghz=0.1).waveform()
    assert y[0] > 3.29 and y[-1] > 3.29 and abs(y[0] - y[2000]) < 2e-3 * 3.3
    y = idle().lossy(loss_db=3, loss_at_ghz=0.1, causal=True).waveform()
    assert np.abs(y - y[2000]).max() < 2e-3 * 3.3


def test_the_envelope_channel_holds_an_unmodulated_carrier():
    """The same on an I/Q record: a constant complex envelope is a carrier already running."""
    x = np.full(4000, 0.7 - 0.4j)
    for causal in (False, True):
        y = BB.lossy(x, 100e6, 2.4e9, causal=causal, loss_db=6, loss_at_ghz=2.4)
        assert np.abs(y[[0, 1, -2, -1]] - y[2000]).max() < 2e-3 * abs(x[0]), causal


def test_a_record_at_zero_at_both_ends_is_bit_identical():
    """Holding zero is padding with zeros, and those records take the old path unchanged."""
    x = np.zeros(4000)
    x[1500:2500] = 1.0
    chans = (lambda v: P.lossy_channel(v, loss_db=3, loss_at_ghz=0.1, f_nyq_ghz=0.5),
             lambda v: P.lossy_channel(v, loss_db=3, loss_at_ghz=0.1, f_nyq_ghz=0.5, causal=True),
             lambda v: P.multi_reflection(v, td_samples=100),
             lambda v: RX.ctle(v, G, 0.05, 0.2, 0.4),
             lambda v: RX.ffe(v, [0.2, 0.6, 0.2], 3.0, pre=1))
    for ch in chans:
        with P.zero_edges():
            before = ch(x)
        assert np.array_equal(ch(x), before)


def test_a_step_mid_record_moves_only_near_the_ends():
    """A level change in the middle of a record that idles at a level. Against the zero-padded
    form the head and tail change by the false edge they lose, and the interior only by the part
    of the line's history the zero pad was missing -- for `lossy`, whose sqrt(f) tail decays
    algebraically, that is 0.2 % (zero-phase) and 0.6 % (causal) of the step 2000 samples in
    on this deliberately slow channel, and falls from there."""
    n = 8000
    x = np.full(n, 3.3)
    x[4000:] = 1.2
    step = 3.3 - 1.2
    for causal in (False, True):
        def ch(v):
            return P.lossy_channel(v, loss_db=3, loss_at_ghz=0.1, f_nyq_ghz=0.5, causal=causal)
        with P.zero_edges():
            old = ch(x)
        d = np.abs(ch(x) - old)
        assert d[:50].max() > 0.5                       # the false edge it removes
        assert d[1000:n - 1000].max() < (0.012 if causal else 0.004) * step
        assert d[2000:n - 2000].max() < (0.008 if causal else 0.003) * step
    # a response with a short memory is untouched away from the ends, bit for bit: the echoes
    # of a held head land in the first 2*k*td samples and nowhere else
    with P.zero_edges():
        old = P.multi_reflection(x, td_samples=100)
    d = np.abs(P.multi_reflection(x, td_samples=100) - old)
    assert d[1300:].max() == 0.0
    assert d[:100].max() > 0.1


def test_the_block_path_holds_the_ends_as_the_transform_does():
    """`method="overlap"` takes the ends from the taps, `fft` from a held pad. They agree at the
    level the two paths always agreed to (`apply_transfer`'s docstring), no worse than zero-padded."""
    n = 1 << 14
    x = np.full(n, 3.3)
    x[n // 2:] = 1.2
    for kw in (dict(td_frac=0.02, f0_frac=0.25, q=10.0), dict(td_frac=0.02, f0_frac=0.05, q=30.0)):
        a = P.resonant_reflection(x, **kw)
        b = P.resonant_reflection(x, method="overlap", **kw)
        with P.zero_edges():
            gap = np.abs(P.resonant_reflection(x, **kw)
                         - P.resonant_reflection(x, method="overlap", **kw)).max()
        assert np.abs(a - b).max() < 2.0 * gap + 1e-6


def test_a_near_end_coupling_of_an_idle_aggressor_is_there_from_the_start():
    """NEXT is a delayed copy of the aggressor's level; the delay is filled from its first sample,
    not left at zero, so an aggressor idling at a level couples from sample 0."""
    a = np.full(1000, 2.0)
    assert np.allclose(P._coupled_kernel(a, "next", 100, 1000), 1.0)
    with P.zero_edges():
        assert P._coupled_kernel(a, "next", 100, 1000)[:100].max() == 0.0


def test_the_two_paths_agree_on_a_slow_channel_to_the_size_the_tail_allows():
    """Held ends, an 8-inch causal channel and a record that starts at +1: the whole-record and
    block paths each miss a different part of the line's slow history, and agree to 3e-3 of
    peak-to-peak (3.2e-3 at 2**15 samples, 1.9e-3 at 2**17) where the zero-padded forms agree to
    7.7e-5. Both are within 0.6 % of the answer with an unbounded hold."""
    n, fs = 1 << 15, 80e9
    x = np.asarray(P.nrz(n_ui=n >> 3, seed=1, n=n, causal=True), float)

    def mk(nfft):
        il = P.insertion_loss_db(np.fft.rfftfreq(nfft) * fs / 1e9, length_in=8.0, tand=0.02)
        return P._min_phase_H(10.0 ** (-il / 20.0), nfft, half=True)

    a = P.apply_transfer(x, mk, method="fft")
    b = P.apply_transfer(x, mk, method="overlap")
    assert np.abs(a - b).max() < 5e-3 * np.ptp(a)
    with P.zero_edges():
        assert np.abs(P.apply_transfer(x, mk) - P.apply_transfer(x, mk, method="overlap")
                      ).max() < 2e-4 * np.ptp(a)

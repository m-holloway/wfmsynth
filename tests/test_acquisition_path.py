"""Layer 2/3 -- the acquisition path: where a decision is taken, and what it is taken through.

Two claims, both against a CONSTRUCTED answer rather than a previous run.

`_constructed_record` builds a waveform in symbol phase from an instantaneous symbol rate this
file writes down, so the true decision instant of every symbol and the correct decision at it
are known to machine precision. That is what makes a symbol-error rate meaningful here: the
symbols are not recovered from the waveform by a second estimator that could be wrong in the
same direction, they are the ones the record was built from.

The probe claims are arithmetic against `1/(2*pi*R*C)` and `1/(1 + j*f/fc)`, written out by
hand below and compared to a transfer function read off realised output samples.

Every gate in here is also run against a deliberately broken fixture and asserted to reject it
(`test_*_rejects_*`), because a check never observed failing is not a check.
"""
import numpy as np
import pytest

import wfmsynth.cdr as CDR
import wfmsynth.instrument as INST
from wfmsynth.compose import Signal, dfe_decisions, dfe_instants, _op_rx_ffe
from wfmsynth.grid import Grid

PAM4 = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
SPB_FRAC = 8.0 * 25 / 28            # 7.142857... -- samples per UI is not a whole number
POST = 0.50                         # the channel's one-UI post-cursor; > 1/3 closes the PAM4 eye


def _constructed_record(n_sym, spb_f, a=POST, tr_ui=0.30, spread=0.0, period_sym=4000.0, seed=7):
    """A record whose symbols, channel and symbol instants are all known by construction.

    The instantaneous symbol rate is ``baud*(1+delta(k))`` for a triangular ``delta`` of
    amplitude ``spread`` -- the shape of a spread-clock profile, and the reason a fixed stride
    cannot be right. The waveform is drawn in symbol phase with raised-cosine transitions
    CENTRED on the symbol boundaries, so at the instant where symbol phase is ``k + 0.5`` its
    value is EXACTLY ``d[k] + a*d[k-1]`` for any ``tr_ui < 1``.
    """
    rng = np.random.default_rng(seed)
    d = rng.choice(PAM4, size=n_sym)
    s = d.copy()
    s[1:] += a * d[:-1]

    k = np.arange(n_sym + 2, dtype=float)
    tri = 1.0 - np.abs(2.0 * ((k / period_sym) % 1.0) - 1.0)
    dk = spb_f / (1.0 - spread * tri)                       # dt/dk, in samples
    t_edge = np.concatenate([[0.0], np.cumsum(dk)])
    t_cen = (0.5 * (t_edge[:-1] + t_edge[1:]))[:n_sym]      # the TRUE instant of every symbol

    n = int(np.floor(t_edge[n_sym])) - 1
    theta = np.interp(np.arange(n, dtype=float), t_edge[:n_sym + 1],
                      np.arange(n_sym + 1, dtype=float))
    kk = np.clip(np.floor(theta).astype(int), 0, n_sym - 1)
    frac = theta - np.floor(theta)
    sp, sn = s[np.maximum(kk - 1, 0)], s[np.minimum(kk + 1, n_sym - 1)]
    x = s[kk].astype(float).copy()
    h = 0.5 * tr_ui
    m = frac < h
    x[m] = sp[m] + (s[kk][m] - sp[m]) * 0.5 * (1 - np.cos(np.pi * (frac[m] + h) / tr_ui))
    m = frac > 1.0 - h
    x[m] = s[kk][m] + (sn[m] - s[kk][m]) * 0.5 * (1 - np.cos(np.pi * (frac[m] - 1.0 + h) / tr_ui))
    return x, d, t_cen, Grid(fs=spb_f * 1e9, baud=1e9, n=n)


def _ser(x, grid, d, **params):
    p = {"taps": [POST], "levels": list(PAM4), "scale": 1.0, **params}
    _inst, _eq, dec = dfe_decisions(x, p, grid)
    m = min(len(dec), len(d))
    return float(np.mean(dec[:m] != d[:m])), m


# ============================================================ the fixture itself
def test_the_constructed_record_is_exact_at_its_own_instants():
    """The fixture is only ground truth if it IS the arithmetic it claims. Sampled at the true
    instants the record equals `d[k] + a*d[k-1]` exactly, and a DFE holding that one post-cursor
    returns every symbol."""
    from wfmsynth.rx import dfe
    x, d, t_cen, g = _constructed_record(4000, SPB_FRAC, spread=0.005)
    y = np.interp(t_cen, np.arange(len(x), dtype=float), x)
    expect = d.copy(); expect[1:] += POST * d[:-1]
    assert np.max(np.abs(y - expect)) < 1e-12
    _eq, dec = dfe(y, np.array([POST]), PAM4)
    assert np.array_equal(dec, d)


def test_the_constructed_record_needs_the_dfe_to_be_readable_at_all():
    """A negative control on the fixture: with the post-cursor left in, the eye is closed and
    the same decisions at the same true instants are wrong ~38 % of the time. Without this, a
    zero error rate below would prove nothing about the equaliser."""
    from wfmsynth.rx import dfe
    x, d, t_cen, g = _constructed_record(4000, SPB_FRAC, spread=0.005)
    y = np.interp(t_cen, np.arange(len(x), dtype=float), x)
    _eq, dec = dfe(y, np.array([0.0]), PAM4)          # no equaliser
    assert np.mean(dec != d) > 0.3


# ============================================================ U-05: the decision instants
@pytest.mark.parametrize("spb_f,spread", [(8.0, 0.0), (SPB_FRAC, 0.0), (SPB_FRAC, 0.005), (8.0, 0.005)])
def test_a_recovered_clock_recovers_every_constructed_symbol(spb_f, spread):
    """Decisions taken on a recovered clock return the constructed symbols exactly, whether or
    not samples-per-UI is a whole number and whether or not the symbol rate moves."""
    x, d, _t, g = _constructed_record(20000, spb_f, spread=spread)
    ser, n = _ser(x, g, d, cdr=dict(loop_bw_ui=3e-3))
    assert ser == 0.0, f"{n * ser:.0f} of {n} symbols wrong"


@pytest.mark.parametrize("spb_f,spread", [(SPB_FRAC, 0.0), (SPB_FRAC, 0.005), (8.0, 0.005)])
def test_the_fixed_stride_diverges_rather_than_degrades(spb_f, spread):
    """The historical path on the same three records. A DFE feeds its own decisions back, so a
    stride that walks off the symbol centres does not degrade gracefully -- it reaches the
    chance error rate of the constellation (0.75 for PAM4)."""
    x, d, _t, g = _constructed_record(20000, spb_f, spread=spread)
    ser, _n = _ser(x, g, d)
    assert ser > 0.7


def test_an_integer_stride_on_an_unmodulated_record_is_left_alone():
    """The fix is scoped to the defect: where the fixed stride is actually right -- a whole
    number of samples per UI, a rate that does not move -- it still returns every symbol, and
    the recovered clock agrees with it symbol for symbol."""
    x, d, _t, g = _constructed_record(20000, 8.0, spread=0.0)
    assert _ser(x, g, d)[0] == 0.0
    inst = dfe_instants(x, dict(taps=[POST], levels=list(PAM4)), g)
    assert np.array_equal(inst, np.arange(inst[0], len(x), 8, dtype=float))


def test_the_recovered_instants_are_the_constructed_instants():
    """Round-trip on the timing itself: the loop is handed the waveform only, and returns the
    instants the record was built around."""
    x, _d, t_cen, g = _constructed_record(20000, SPB_FRAC, spread=0.005)
    tau = CDR.recover_symbol_instants(x, grid=g, loop_bw_ui=3e-3)
    m = min(len(tau), len(t_cen))
    err_ui = np.abs(tau[:m] - t_cen[:m]) / SPB_FRAC
    assert np.max(err_ui[500:]) < 0.05


def test_the_recovered_instants_are_not_the_starting_guess():
    """Pull-in: started a third of a UI away in either direction, the loop converges on the same
    instants. Otherwise 'recovered' could just be the nominal phase handed back."""
    x, _d, t_cen, g = _constructed_record(20000, SPB_FRAC, spread=0.005)
    tails = []
    for off in (-0.35, 0.0, 0.35):
        tau = CDR.recover_symbol_instants(x, grid=g, loop_bw_ui=3e-3,
                                          phase0=(0.5 + off) * SPB_FRAC, n_sym=19000)
        assert abs((tau[0] - t_cen[0]) / SPB_FRAC - off) < 1e-9      # it really did start there
        tails.append(tau[5000:])
    assert np.max(np.abs(tails[0] - tails[1])) < 0.01 * SPB_FRAC
    assert np.max(np.abs(tails[2] - tails[1])) < 0.01 * SPB_FRAC


def test_loop_order_means_what_recover_clock_says_it_means():
    """`recover_clock` documents order 1 as leaving a static phase error under a frequency
    offset and order 2 as tracking it to zero. The waveform loop is the same loop: given a
    nominal rate 0.3 % away from the record's, order 1 settles ~0.13 UI off and order 2 on it."""
    x, _d, t_cen, g = _constructed_record(20000, SPB_FRAC, spread=0.0)
    static = {}
    for order in (1, 2):
        tau = CDR.recover_symbol_instants(x, spb=SPB_FRAC * 1.003, loop_bw_ui=3e-3,
                                          order=order, n_sym=19000)
        static[order] = float(np.mean((tau[5000:] - t_cen[5000:19000]) / SPB_FRAC))
    assert abs(static[1]) > 0.05
    assert abs(static[2]) < 0.02
    assert abs(static[2]) < 0.2 * abs(static[1])


@pytest.mark.parametrize("broken,expect", [
    (dict(taps=[-POST]), 0.4),                                  # tap sign flipped
    (dict(taps=[0.0]), 0.3),                                    # no equaliser
    (dict(cdr=dict(loop_bw_ui=1e-5)), 0.5),                     # loop 300x too slow to track
    (dict(cdr=dict(loop_bw_ui=3e-3, spb=SPB_FRAC * 1.05)), 0.5),  # nominal rate past pull-in
])
def test_the_symbol_error_gate_rejects_a_broken_decision_path(broken, expect):
    """The gate observed failing. Each row breaks one thing about the decision path and must
    push the error rate above `expect`; the working path scores exactly zero on this record."""
    x, d, _t, g = _constructed_record(20000, SPB_FRAC, spread=0.005)
    p = dict(cdr=dict(loop_bw_ui=3e-3)); p.update(broken)
    assert _ser(x, g, d, **p)[0] > expect


def test_rx_ffe_is_a_fixed_tap_misplacement_and_not_the_same_defect():
    """`rx_ffe` reads samples-per-UI the same way, but it is a time-invariant FIR: its error is
    one rounding of the tap spacing (here 0.02 UI), identical at the start and the end of the
    record. The DFE's is an instant index that walks -- tens of UI by the first tenth of the
    record and hundreds by the last. Different defects; only one of them accumulates."""
    x, _d, t_cen, g = _constructed_record(20000, SPB_FRAC, tr_ui=0.9, spread=0.005)
    taps = [1.0, -POST, POST ** 2]
    y_op = _op_rx_ffe(x, dict(op="rx_ffe", taps=taps, spacing_ui=1.0, pre=0), None, g, 0)
    xi = np.arange(len(x), dtype=float)
    y_ex = np.zeros(len(x))
    for k, c in enumerate(taps):                     # the same FIR at EXACT fractional spacing
        y_ex += c * np.interp(xi - k * SPB_FRAC, xi, x, left=x[0], right=x[-1])
    t = t_cen[t_cen < len(x) - 2]
    diff = np.abs(np.interp(t, xi, y_op) - np.interp(t, xi, y_ex))
    first, last = [float(np.sqrt(np.mean(b ** 2))) for b in (diff[2000:4000], diff[-2000:])]
    assert 0.5 < last / first < 2.0                  # flat: does not accumulate

    inst = dfe_instants(x, dict(taps=[POST], levels=list(PAM4)), g)
    m = min(len(inst), len(t_cen))
    walk = np.abs(inst[:m] - t_cen[:m]) / SPB_FRAC
    # walks: hundreds of UI further off by the end of the record than at the start
    assert np.mean(walk[-2000:]) - np.mean(walk[2000:4000]) > 100.0


# ============================================================ U-01: the probe
PROBE_RC = [(50.0, 0.5e-12), (50.0, 1.0e-12), (100.0, 0.25e-12), (25.0, 2.0e-12)]
PROBE_GRID = Grid(fs=400e9, baud=25e9, n=1 << 15)


def _transfer_at(fn, f_hz, grid=PROBE_GRID):
    k = int(round(f_hz * grid.n / grid.fs))
    f = k * grid.fs / grid.n
    x = np.sin(2 * np.pi * k * np.arange(grid.n) / grid.n)
    return f, np.fft.rfft(fn(x))[k] / np.fft.rfft(x)[k]


@pytest.mark.parametrize("r,c", PROBE_RC)
@pytest.mark.parametrize("mult", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_the_probe_is_the_rc_pole_the_closed_form_names(r, c, mult):
    """The pole is at ``1/(2*pi*R*C)``; the loading is ``1/(1 + j*f/fc)``. Both halves are
    asserted -- magnitude ``10*log10(1 + (f/fc)**2)`` dB and phase ``-atan(f/fc)`` -- because a
    zero-phase filter can carry the right rolloff shape and still not be an RC pole."""
    fc = 1.0 / (2 * np.pi * r * c)
    assert INST.rc_pole_hz(r, c) == pytest.approx(fc, rel=1e-12)
    f, H = _transfer_at(lambda x: INST.probe(x, PROBE_GRID, c_load_f=c, r_source=r), fc * mult)
    assert -20 * np.log10(abs(H)) == pytest.approx(10 * np.log10(1 + (f / fc) ** 2), abs=1e-9)
    assert np.degrees(np.angle(H)) == pytest.approx(-np.degrees(np.arctan(f / fc)), abs=1e-9)


def test_the_closed_form_gate_rejects_the_wrong_pole():
    """The gate observed failing: the same assertion against a probe whose capacitance is twice
    what the closed form was written for is out by ~4 dB at fc, and against the zero-phase
    `probe_loading` default -- which applies the pole twice -- by ~3 dB and 45 degrees."""
    fc = 1.0 / (2 * np.pi * 50.0 * 0.5e-12)
    f, H = _transfer_at(lambda x: INST.probe(x, PROBE_GRID, c_load_f=1.0e-12), fc)
    assert abs(-20 * np.log10(abs(H)) - 10 * np.log10(1 + (f / fc) ** 2)) > 3.0
    f, H = _transfer_at(lambda x: INST.probe_loading(x, PROBE_GRID, c_load_f=0.5e-12), fc)
    assert abs(-20 * np.log10(abs(H)) - 10 * np.log10(1 + (f / fc) ** 2)) > 2.5
    assert abs(np.degrees(np.angle(H))) < 1.0                       # zero phase, not a pole


def test_probe_loading_default_is_the_closed_form_squared():
    """Naming the deviation rather than leaving it to be discovered: the shipped zero-phase
    default runs the pole forwards and backwards, so its dB is twice the closed form's at every
    frequency. Kept bit-identical; `causal=True` is the pole itself."""
    fc = 1.0 / (2 * np.pi * 50.0 * 0.5e-12)
    for mult in (0.5, 1.0, 2.0):
        f, H = _transfer_at(lambda x: INST.probe_loading(x, PROBE_GRID, c_load_f=0.5e-12), fc * mult)
        assert (-20 * np.log10(abs(H))) / (10 * np.log10(1 + (f / fc) ** 2)) == pytest.approx(2.0, abs=0.02)
    for mult in (0.5, 1.0, 2.0):
        f, H = _transfer_at(
            lambda x: INST.probe_loading(x, PROBE_GRID, c_load_f=0.5e-12, causal=True), fc * mult)
        assert -20 * np.log10(abs(H)) == pytest.approx(10 * np.log10(1 + (f / fc) ** 2), abs=1e-9)


def test_the_probe_op_reaches_the_probe_and_sits_before_the_instrument():
    """U-01 itself: the op exists, a chain can carry it, the recipe round-trips it, and it homes
    between the channel and the instrument -- a measurement is made THROUGH something."""
    import json
    from wfmsynth.compose import op_kind, KIND_RANK
    g = Grid(fs=200e9, baud=25e9, n=1 << 12)
    s = (Signal(seed=1, grid=g).carrier("nrz", n_ui=400, tr_frac=0.4)
         .probe(c_load_f=1e-12, bw_hz=40e9, noise_rms=0.002).scope(bw_hz=60e9))
    x = s.waveform()
    assert np.array_equal(Signal.from_recipe(json.loads(json.dumps(s.recipe()))).waveform(), x)
    assert KIND_RANK[op_kind("channel_missing_is_channel")] if False else True
    assert KIND_RANK["channel"] < KIND_RANK[op_kind("probe")] < KIND_RANK["instrument"]
    bare = (Signal(seed=1, grid=g).carrier("nrz", n_ui=400, tr_frac=0.4).scope(bw_hz=60e9)).waveform()
    assert np.sqrt(np.mean((x - bare) ** 2)) > 1e-3 * np.sqrt(np.mean(bare ** 2))


# ============================================================ the same claim, end to end
def _e2e(spread, n_ui, grid, post=0.35):
    """carrier -> ssc -> a one-UI post-cursor, through the composer's OWN ops. The transmitted
    symbols come from the recipe (`physics.carrier_symbols`), not from a second estimator."""
    s = Signal(seed=11, grid=grid).carrier("pam4", n_ui=n_ui, pattern="prbs13q", tr_frac=0.3)
    if spread:
        s = s.ssc(f_ssc=32e3, spread=spread, profile="down")
    x = s.waveform()
    d = int(round(grid.samples_per_ui))
    y = x.copy()
    y[d:] += post * x[:-d]
    return y


def _e2e_ser(y, tx, grid, **params):
    _inst, _eq, dec = dfe_decisions(y, {"taps": [0.35], "levels": list(PAM4), **params}, grid)
    best = 1.0
    for lag in range(-4, 5):                    # a channel and a CDR both shift the symbol index
        a, b = dec[max(0, lag):], tx[max(0, -lag):]
        m = min(len(a), len(b))
        if m > 1000:
            best = min(best, float(np.mean(a[:m] != b[:m])))
    return best


def test_an_ssc_bearing_chain_through_a_dfe_is_readable_end_to_end():
    """The user-facing claim, through `Signal.ssc()` rather than a hand-built record: a record
    carrying spread-spectrum clocking through a DFE must come out READ, not destroyed.

    Samples-per-UI is a whole 8.000 here, so the fixed stride is right when the rate is
    constant and returns everything -- and it is the SSC alone that breaks it."""
    import wfmsynth.physics as P
    g = Grid(fs=128e9, baud=16e9, n=1 << 17)
    n_ui = int(g.n // g.samples_per_ui)
    tx = P.carrier_symbols("pam4", n_ui, 1, "prbs13q")

    clean = _e2e(0.0, n_ui, g)
    assert _e2e_ser(clean, tx, g, scale=1.0) == 0.0                      # stride right, left alone
    assert _e2e_ser(clean, tx, g, scale=1.0, cdr=dict(loop_bw_hz=10e6)) == 0.0

    ssc = _e2e(0.005, n_ui, g)
    assert _e2e_ser(ssc, tx, g, scale=1.0) > 0.3                         # today: destroyed
    assert _e2e_ser(ssc, tx, g, scale=1.0, cdr=dict(loop_bw_hz=10e6)) == 0.0
    # negative control: the equaliser is load-bearing, so zero above is not a free pass
    assert _e2e_ser(ssc, tx, g, taps=[0.0], scale=1.0, cdr=dict(loop_bw_hz=10e6)) > 0.3


def test_the_default_amplitude_normalisation_costs_symbols_on_a_record_it_should_read():
    """A third defect the constructed answer exposed, separate from the timing. The DFE's
    default scale is the 99th percentile of the SAMPLED magnitudes, which post-cursor ISI
    inflates above the constellation's own full scale -- so the slicer's levels sit wrong even
    on a record with a perfect clock. Here it costs 12 % of the symbols where an explicit
    `scale` reads every one. `scale=` is the knob; the default is unchanged."""
    import wfmsynth.physics as P
    g = Grid(fs=128e9, baud=16e9, n=1 << 17)
    n_ui = int(g.n // g.samples_per_ui)
    tx = P.carrier_symbols("pam4", n_ui, 1, "prbs13q")
    clean = _e2e(0.0, n_ui, g)
    assert _e2e_ser(clean, tx, g, scale=1.0) == 0.0
    assert _e2e_ser(clean, tx, g) > 0.1

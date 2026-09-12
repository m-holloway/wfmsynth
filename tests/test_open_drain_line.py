"""
The open-drain line: a resistive rise, a driven fall, and what happens when a bit is too short to
finish charging.

`bus.open_drain` combines drivers at LOGIC level -- a wired-AND over ones and zeros. That is the
protocol, not the waveform. On a real open-drain bus the two edges are produced by different
mechanisms and are not each other's mirror:

    fall    the sink pulls the line down through its on-resistance, in parallel with the pull-up.
            Fast, and its floor is a divider, not ground.
    rise    nobody drives it. The pull-up charges the bus capacitance. RC, and slow.

Modelling that edge as a symmetric slow edge -- which is what a generic slow-edge impairment does --
deletes the asymmetry, and the asymmetry is the single most characteristic thing about the class. It
also deletes the bus's real failure mode: raise the capacitance far enough and the line never
reaches the input-high threshold within a bit, which a symmetric edge model cannot produce at all
because it always arrives.
"""
import warnings

import numpy as np
import pytest

from wfmsynth.physics import open_drain_line, rc_tau_for_rise_time


# ===================================================================================================
# the time constant, against the numbers the specification publishes
# ===================================================================================================
def test_the_tau_for_a_rise_time_inverts_the_exponential_charge():
    """On v(t) = V(1 - exp(-t/tau)), the 30 % and 70 % crossings are at 0.3567 tau and 1.2040 tau,
    so a 30-70 % rise time is 0.8473 tau. The I2C-bus specification measures tr between 30 % and
    70 % of VDD, not the 10-90 % a habit from driven logic would assume, and confusing the two is
    worth a factor of 2.6."""
    tau = rc_tau_for_rise_time(847.3e-9, lo_frac=0.3, hi_frac=0.7)
    assert abs(tau - 1e-6) < 1e-9
    # the 10-90 % convention on the same tau is 2.1972 tau -- 2.59x the 30-70 % figure
    assert abs(rc_tau_for_rise_time(1e-6, 0.1, 0.9) / rc_tau_for_rise_time(1e-6, 0.3, 0.7)
               - 0.8473 / 2.1972) < 1e-3


@pytest.mark.parametrize("tr_s,c_bus_f,expect_r_ohm", [
    # Fast-mode: tr max 300 ns at the 400 pF bus-capacitance limit
    (300e-9, 400e-12, 885.0),
    # Fast-mode Plus: tr max 120 ns at the 550 pF limit -- which is why Fm+ needs a much
    # stronger pull-up, and why it specifies a 20 mA sink to still pull that down
    (120e-9, 550e-12, 257.0),
])
def test_the_published_rise_limits_imply_a_physical_pull_up(tr_s, c_bus_f, expect_r_ohm):
    """A check that the model's units close against the specification rather than only against
    itself: the pull-up implied by (tr, Cb) has to be a resistor someone would actually fit."""
    r = rc_tau_for_rise_time(tr_s) / c_bus_f
    assert abs(r - expect_r_ohm) / expect_r_ohm < 0.02, r
    assert 100.0 < r < 10_000.0


def test_a_rise_time_round_trips_through_the_line_model():
    """Ask for a 300 ns 30-70 % rise, build the line from the implied R and C, and measure it back
    off the waveform. If this does not close, the model and the formula disagree."""
    fs = 1e9
    c_bus = 400e-12
    r = rc_tau_for_rise_time(300e-9) / c_bus
    n = 4096
    sink = np.zeros(n, dtype=bool)
    sink[:512] = True                                   # held low, then released
    v = open_drain_line(sink, fs, r_pullup_ohm=r, c_bus_f=c_bus, v_dd=3.3)
    seg = v[512:]
    t30 = np.interp(0.3 * 3.3, seg, np.arange(len(seg))) / fs
    t70 = np.interp(0.7 * 3.3, seg, np.arange(len(seg))) / fs
    assert abs((t70 - t30) - 300e-9) / 300e-9 < 0.02, (t70 - t30)


# ===================================================================================================
# the asymmetry, which is the point
# ===================================================================================================
def test_the_fall_is_much_faster_than_the_rise():
    fs = 1e9
    n = 8192
    sink = np.zeros(n, dtype=bool)
    sink[2048:6144] = True
    v = open_drain_line(sink, fs, r_pullup_ohm=885.0, c_bus_f=400e-12, v_dd=3.3,
                        r_sink_ohm=20.0)

    def edge_time(seg, lo, hi):
        a = np.interp(lo, np.sort(seg), np.argsort(seg)) if False else None
        return None
    fall = v[2048:2560]
    rise = v[6144:8192]
    # 70 % -> 30 % on the fall, 30 % -> 70 % on the rise, both in samples
    f70 = np.argmax(fall <= 0.7 * 3.3)
    f30 = np.argmax(fall <= 0.3 * 3.3)
    r30 = np.argmax(rise >= 0.3 * 3.3)
    r70 = np.argmax(rise >= 0.7 * 3.3)
    tf, tr = (f30 - f70) / fs, (r70 - r30) / fs
    assert tf > 0 and tr > 0
    assert tr / tf > 10.0, (tr, tf)


def test_the_low_level_is_a_divider_not_ground():
    """The sink's on-resistance and the pull-up form a divider, so V_OL is above ground and RISES
    with a stronger pull-up. A model that clamps the low level to 0 V loses the one DC measurement
    that reveals a pull-up too strong for the sink."""
    # 10 GSa/s: at 100 pF and a 20 ohm sink the fall time constant is 2 ns, and this test is about
    # the settled DC level, so resolve the fall rather than take the warning.
    fs = 10e9
    sink = np.ones(40960, dtype=bool)
    weak = open_drain_line(sink, fs, r_pullup_ohm=4700.0, c_bus_f=100e-12, v_dd=3.3, r_sink_ohm=20.0)
    strong = open_drain_line(sink, fs, r_pullup_ohm=250.0, c_bus_f=100e-12, v_dd=3.3, r_sink_ohm=20.0)
    assert 0.0 < weak[-1] < strong[-1]
    # 3.3 * 20/(20+250) = 0.244 V
    assert abs(strong[-1] - 3.3 * 20.0 / 270.0) < 1e-3
    assert abs(weak[-1] - 3.3 * 20.0 / 4720.0) < 1e-3


def test_the_rise_never_overshoots_and_the_fall_never_undershoots():
    """A first-order charge is monotone toward its target. Any overshoot here would be a numerical
    artefact of the integration, and on this bus an overshoot is a real diagnostic -- so the model
    must not manufacture one."""
    fs = 1e9
    sink = np.zeros(4096, dtype=bool)
    sink[1000:2000] = True
    v = open_drain_line(sink, fs, r_pullup_ohm=885.0, c_bus_f=400e-12, v_dd=3.3)
    assert v.max() <= 3.3 + 1e-12
    assert v.min() >= -1e-12
    assert np.all(np.diff(v[2000:]) >= -1e-12)          # released: monotone up
    assert np.all(np.diff(v[1000:2000]) <= 1e-12)       # sinking: monotone down


# ===================================================================================================
# the failure mode a symmetric edge model cannot represent
# ===================================================================================================
def test_too_much_bus_capacitance_stops_the_line_reaching_the_input_high_threshold():
    """The real way an over-loaded open-drain bus fails: the pull-up cannot charge Cb to VIH within
    the bit, so a released line is still read as low and the transfer corrupts. VIH is 0.7 VDD.

    This is the test that justifies the whole model. A symmetric slow edge always arrives at the
    rail, just later; an RC charge that runs out of time does not arrive at all."""
    fs = 1e9
    v_dd = 3.3
    bit = 2500                                           # 2.5 us -- a 400 kbit/s bit
    sink = np.zeros(4 * bit, dtype=bool)
    sink[:bit] = True
    reached = {}
    for c_bus in (100e-12, 400e-12, 4700e-12):
        # 1 GSa/s is a realistic capture rate for a 400 kbit/s bus and does not resolve the 1.8 ns
        # fall at the smallest Cb here. That is a warning by design and is irrelevant to this
        # test, which is about whether the RISE arrives.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = open_drain_line(sink, fs, r_pullup_ohm=885.0, c_bus_f=c_bus, v_dd=v_dd)
        reached[c_bus] = bool(v[bit:2 * bit].max() >= 0.7 * v_dd)
    assert reached[100e-12] is True
    assert reached[400e-12] is True
    assert reached[4700e-12] is False, "an 11x over-limit bus must fail to reach VIH"


def test_a_run_of_low_bits_does_not_let_the_line_creep_up():
    """State has to carry across bit boundaries. If the integrator restarts per bit, a long low run
    would show sawtooth ripple instead of a settled level."""
    fs = 1e9
    sink = np.ones(8192, dtype=bool)
    v = open_drain_line(sink, fs, r_pullup_ohm=885.0, c_bus_f=400e-12, v_dd=3.3)
    settled = v[4096:]
    assert np.ptp(settled) < 1e-6


def test_the_initial_level_can_be_given_and_defaults_to_the_settled_value():
    fs = 1e9
    sink = np.zeros(1024, dtype=bool)
    v = open_drain_line(sink, fs, r_pullup_ohm=885.0, c_bus_f=400e-12, v_dd=3.3)
    assert abs(v[0] - 3.3) < 1e-9                        # released and already settled high
    # `v0` is the level BEFORE the first sample, so v[0] is already one dt into the charge --
    # small but not zero. Asserting it IS zero would be asserting the integrator does nothing on
    # its first step.
    # 1024 samples at 1 GSa/s is 1.024 us against a 354 ns time constant -- 2.89 time constants,
    # so the charge is 94.4 % complete and NOT at the rail. Asserting it reaches 3.2 V here would be
    # asserting the exponential finishes early.
    w = open_drain_line(sink, fs, r_pullup_ohm=885.0, c_bus_f=400e-12, v_dd=3.3, v0=0.0)
    assert 0.0 < w[0] < 0.02
    assert np.all(np.diff(w) > 0)
    tau = 885.0 * 400e-12
    assert abs(w[-1] - 3.3 * (1.0 - np.exp(-len(w) / (1e9 * tau)))) < 1e-6


def test_it_refuses_a_grid_that_cannot_resolve_the_rise():
    """The rise is the edge this bus specifies. A grid too coarse to show it produces a record that
    does not represent what it claims, so this is an error and not a warning."""
    with pytest.raises(ValueError, match="cannot resolve the rise"):
        open_drain_line(np.zeros(16, dtype=bool), 1e3, r_pullup_ohm=885.0,
                        c_bus_f=400e-12, v_dd=3.3)


def test_a_grid_that_resolves_the_rise_but_not_the_fall_warns_and_still_returns():
    """The asymmetric guard, both halves. Rp >> Rs on every real open-drain bus, so the fall is one
    to two orders of magnitude faster than the rise and a capture that resolves the rise routinely
    does not resolve the fall. Refusing that would be refusing reality; saying nothing would hide
    it."""
    # Rp = 885, Cb = 100 pF, Rs = 20: the rise constant is 88.5 ns (resolved at 1 GSa/s) and the
    # fall constant is 1.96 ns (not resolved). That is the asymmetry in one call.
    sink = np.zeros(4096, dtype=bool)
    sink[:512] = True
    with pytest.warns(RuntimeWarning, match="does not resolve the fall"):
        v = open_drain_line(sink, 1e9, r_pullup_ohm=885.0, c_bus_f=100e-12, v_dd=3.3,
                            r_sink_ohm=20.0)
    assert v[-1] > 3.2                       # the rise still arrives
    assert v[400] < 0.3                      # the level is still right


def test_two_lines_combine_as_a_wired_and_at_the_analogue_level():
    """SCL and SDA are separate lines, but several devices share each one. Two sinks on one line
    means the line is low if EITHER sinks it -- and the resulting analogue level is the parallel
    combination, not a re-run of one driver."""
    fs = 1e9
    a = np.zeros(4096, dtype=bool); a[500:1500] = True
    b = np.zeros(4096, dtype=bool); b[1000:2000] = True
    both = open_drain_line(a | b, fs, r_pullup_ohm=885.0, c_bus_f=400e-12, v_dd=3.3)
    # low for the whole union of the two sink intervals
    assert both[600] < 0.3 * 3.3 and both[1900] < 0.3 * 3.3
    assert both[3000] > 0.7 * 3.3


# ===================================================================================================
# the composer op
# ===================================================================================================
def _od_signal(g, rp, cb, **kw):
    from wfmsynth import Signal
    s = Signal(seed=3, grid=g)
    s.carrier("nrz", pattern="prbs9", n_ui=g.n // int(g.samples_per_ui), tr_frac=0.2)
    s.open_drain(r_pullup_ohm=rp, c_bus_f=cb, v_dd=3.3, **kw)
    return s


def _slow_grid(n=1 << 14):
    from wfmsynth import Grid
    return Grid(fs=100e6, baud=400e3, n=n, v_full=3.3)


def test_the_op_is_a_shape_stage_so_it_sorts_after_the_carrier_and_before_the_channel():
    from wfmsynth.compose import op_kind, canonicalize
    assert op_kind("open_drain") == "shape"
    ops = [{"op": "lossy"}, {"op": "open_drain"}, {"op": "carrier"}, {"op": "scope"}]
    assert [o["op"] for o in canonicalize(ops)] == ["carrier", "open_drain", "lossy", "scope"]


def test_the_op_produces_a_unipolar_line_with_a_divider_low_level():
    """The level domain changes on purpose: this bus is single-ended between ground and its supply,
    and its settled low is a resistive divider. A centred +/-1 waveform is the wrong shape for it."""
    g = _slow_grid()
    rp, cb = rc_tau_for_rise_time(300e-9) / 400e-12, 400e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y = _od_signal(g, rp, cb).waveform()
    assert y.min() > 0.0                                    # never reaches ground
    assert abs(y.max() - 3.3) < 1e-3                        # does reach the supply
    assert abs(y.min() - 3.3 * 20.0 / (20.0 + rp)) < 1e-3   # and the low level IS the divider


def test_the_op_keeps_the_fall_steeper_than_the_rise_through_the_whole_record():
    g = _slow_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y = _od_signal(g, rc_tau_for_rise_time(300e-9) / 400e-12, 400e-12).waveform()
    d = np.diff(y)
    up, down = d[d > 0], d[d < 0]
    assert up.size and down.size
    assert abs(down.mean()) / up.mean() > 5.0, (up.mean(), down.mean())


def test_center_rescales_to_the_incoming_excursion_and_discards_the_dc_offset():
    """The escape hatch, and what it costs: `center=True` gives a downstream stage the centred
    waveform it may need, at the price of V_OL -- which is the measurement that reveals a pull-up
    too strong for the sink."""
    g = _slow_grid()
    rp, cb = rc_tau_for_rise_time(300e-9) / 400e-12, 400e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = _od_signal(g, rp, cb).waveform()
        cen = _od_signal(g, rp, cb, center=True).waveform()
    assert raw.min() > 0.0 and cen.min() < 0.0
    # centred means the MIDPOINT of the excursion sits at zero. The median does not: this signal is
    # bimodal and its two levels are not equally occupied, so the median sits on whichever rail the
    # data spends more time at, in both versions.
    mid = 0.5 * (np.percentile(cen, 1) + np.percentile(cen, 99))
    assert abs(mid) < 0.02, mid
    # and the excursion is rescaled to the incoming carrier's, which is +/-1
    assert abs(np.ptp(np.percentile(cen, [1, 99])) - 2.0) < 0.05
    # the asymmetry survives centring -- it is a time-domain property, not a level one
    d = np.diff(cen)
    assert abs(d[d < 0].mean()) / d[d > 0].mean() > 5.0


def test_the_op_threshold_follows_the_carrier_rather_than_assuming_zero():
    """Thresholding at the midpoint of the carrier's OWN excursion, not at 0 V, so an upstream stage
    that has already offset or rescaled the carrier does not silently invert the sink decision --
    which at a +5 V offset is exactly what a hard-coded zero threshold would do.

    Driven through the op directly, because the offset is the whole point and the builder has no
    DC-offset stage to apply one with."""
    from wfmsynth.compose import _op_open_drain
    from wfmsynth import Grid
    g = Grid(fs=100e6, baud=400e3, n=4096, v_full=3.3)
    carrier = np.where(np.arange(4096) % 512 < 256, 1.0, -1.0)
    kw = dict(r_pullup_ohm=rc_tau_for_rise_time(300e-9) / 400e-12, c_bus_f=400e-12, v_dd=3.3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = _op_open_drain(carrier, kw, None, g, 0)
        b = _op_open_drain(carrier + 5.0, kw, None, g, 0)        # same intent, shifted up 5 V
        c = _op_open_drain(0.25 * carrier, kw, None, g, 0)       # same intent, quarter amplitude
    assert np.allclose(a, b), "a DC offset on the carrier changed the line"
    assert np.allclose(a, c), "rescaling the carrier changed the line"

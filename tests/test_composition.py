"""Layer 3 -- composition: ops that are each right, together.

Every op here already passes on its own. These are the invariants that only exist once ops are
put in a chain, and they are where the defects live:

  * BYPASS EQUALS ABSENCE. A stage set to its identity must give the same samples as a chain
    that never had it. Anything else means the stage is doing something it does not admit to.
  * GROUP DELAY ACCUMULATES. The chain's delay is the sum of its parts, each measured alone.
  * ORDER SENSITIVITY IS ASSERTED WHERE THE PHYSICS HAS IT -- and commutation is asserted where
    it does not, because that is the half that says which model is carrying the positional
    information (see `test_lumped_loss_and_reflection_commute...`).
  * EVERY PARAMETER DEMONSTRABLY DOES SOMETHING. A `noise_mv` that was stored and never read
    survived weeks here. `test_every_knob_moves_the_output` is the sweep that ends that.

Nothing below compares against a stored output.
"""
import numpy as np
import pytest

import wfmsynth.instrument as INST
import wfmsynth.physics as P
import wfmsynth.rx as RX
import wfmsynth.sparam as SP
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

GRID = Grid(fs=256e9, baud=16e9, n=1 << 13)
WIDE = Grid(fs=200e9, n=1 << 15)


def _base():
    return Signal(seed=5, grid=GRID).carrier("nrz", n_ui=512, tr_frac=0.15, pattern="prbs7")


def _with(op, **kw):
    s = _base()
    getattr(s, op)(**kw)
    return s.waveform()


def _rel_rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a ** 2)) + 1e-30))


def _group_delay_s(fn, grid=WIDE, band=(1e9, 8e9), seed=0):
    """Group delay in seconds, measured as the slope of the unwrapped phase of the transfer
    function READ OFF the realised output (rfft(y)/rfft(x)), fitted over `band`."""
    x = np.random.default_rng(seed).standard_normal(grid.n)
    H = np.fft.rfft(fn(x)) / np.fft.rfft(x)
    f = np.fft.rfftfreq(grid.n, d=grid.dt)
    m = (f >= band[0]) & (f <= band[1])
    A = np.vstack([f[m], np.ones(int(m.sum()))]).T
    slope = np.linalg.lstsq(A, np.unwrap(np.angle(H))[m], rcond=None)[0][0]
    return -slope / (2 * np.pi)


# ======================================================== bypass equals absence
# The ops whose identity setting is EXACT arithmetic (x + 0*something, a unit tap, a zero
# warp): these must be bit-identical, and there is no tolerance to argue about.
EXACT_BYPASS = [
    ("reflect", dict(td_ps=281.25, gamma_s=0.0, gamma_l=0.0, n_bounce=6)),
    ("reflect", dict(td_ps=281.25, gamma_s=0.4, gamma_l=0.0, n_bounce=6, node="load")),
    # a matched SOURCE with a mismatched load: every bounce weight is (gs*gl)^k = 0 at the
    # load tap, so the single reflection is absorbed and the tap is the incident wave again.
    # gamma_l is non-zero here on purpose -- it is the identity that a "close enough" echo
    # term would break, and the one above would not.
    ("reflect", dict(td_ps=281.25, gamma_s=0.0, gamma_l=0.4, n_bounce=6, node="load")),
    ("crosstalk", dict(coupling=0.0)),
    # asking for no impairment must not spend the interpolator's error budget as one: both of
    # these short-circuit rather than resampling at integer positions, which is a 0.45-cutoff
    # lowpass and NOT the identity (it costs 1.9e-07 of full scale).
    ("dcd", dict(ps=0.0)),
    ("sample_clock", dict(ppm=0.0)),
    ("rx_noise", dict(rms=0.0)),
    ("supply_coupling", dict(f_ripple_hz=1e9, am_depth=0.0, psij_ps=0.0)),
    ("intra_pair_skew", dict(skew_ps=0.0, gain_imbalance=0.0)),
    ("ssc", dict(spread=0.0)),
    ("tx_ffe", dict(taps=[1.0], pre=0)),
    ("rx_ffe", dict(taps=[1.0], spacing_ui=1.0, pre=0)),
    ("de_emphasis", dict(db=0.0)),
    ("timing", dict(rj_ps=0.0)),
    ("timebase", dict(rms_ps=0.0)),
    ("nonlinearity", dict(compression=0.0, level_noise=0.0, rise_fall_ratio=1.0)),
]

# The ops that go through an FFT even at identity: a forward-inverse transform is not the
# identity in floating point, so "absence" here is a rounding error, and the bound is stated
# in units of the signal's own span rather than left vague.
TRANSFORM_BYPASS = [
    ("lossy", dict(loss_db=0.0, loss_at_ghz=8.0)),
    ("lossy", dict(trend=(0.0, 0.0, 0.0))),
    ("lossy", dict(length_in=0.0)),
    ("resonant_reflect", dict(td_ps=20.0, f0_ghz=25.0, q=10.0, gamma0=0.0)),
    ("cascade", dict(path=[{"line": {"td_ps": 0.0, "causal": False, "loss_length_in": 0.0}}])),
]


@pytest.mark.parametrize("op,kw", EXACT_BYPASS, ids=lambda v: v if isinstance(v, str) else "")
def test_a_bypassed_stage_is_bit_identical_to_its_absence(op, kw):
    absent, bypassed = _base().waveform(), _with(op, **kw)
    assert np.array_equal(absent, bypassed), (
        f"{op}{kw} at identity moved {int(np.count_nonzero(absent != bypassed))} samples, "
        f"max {np.abs(absent - bypassed).max():.3e}")


@pytest.mark.parametrize("op,kw", TRANSFORM_BYPASS, ids=lambda v: v if isinstance(v, str) else "")
def test_a_bypassed_transform_stage_is_absence_to_the_last_bit_of_the_transform(op, kw):
    absent, bypassed = _base().waveform(), _with(op, **kw)
    rel = np.abs(absent - bypassed).max() / np.ptp(absent)
    assert rel < 1e-13, f"{op}{kw} at identity moved the record by {rel:.3e} of its span"


def test_bypass_holds_in_the_middle_of_a_chain_not_just_alone():
    """The interesting case: an identity stage between stages that do something. If a bypassed
    stage resets state or resamples, it shows up here and not above. Every stage in this chain
    is deterministic; the seeded case is the test immediately below."""
    def chain(insert):
        s = _base().tx_ffe(taps=[1.0, -0.2], pre=0)
        if insert:
            s = s.reflect(td_ps=281.25, gamma_s=0.0, gamma_l=0.0, n_bounce=6)
        return s.de_emphasis(db=3.0).intra_pair_skew(skew_ps=1.5).timebase(rms_ps=0.0).waveform()
    assert np.array_equal(chain(False), chain(True))


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN DEFECT, asserted as physics rather than pinned as behaviour: role streams are keyed "
    "by OP INDEX (`noise/{idx}`), so inserting a stage -- even one set to its exact identity -- "
    "renames every later stage's random role and re-rolls it. Here the bypassed `reflect` moves "
    "the digitiser's role from 'noise/1' to 'noise/2' and the noise realisation changes "
    "completely, so 'bypass equals absence' holds for deterministic stages and silently fails "
    "for every seeded one. A key that is stable under insertion (op identity rather than "
    "position) fixes it. Remove this marker when it is."))
def test_a_bypassed_stage_does_not_reseed_the_stages_after_it():
    a = _base().digitize(snr_db=35.0).waveform()
    b = _base().reflect(td_ps=100.0, gamma_s=0.0, gamma_l=0.0).digitize(snr_db=35.0).waveform()
    assert np.array_equal(a, b), (
        f"the bypassed stage re-rolled the digitiser: rel rms {_rel_rms(a, b):.3e}")


# ======================================================== group delay accumulates
def _line(td_ps, grid=WIDE):
    path = [{"line": {"td_ps": td_ps, "causal": False, "loss_length_in": 0.0}}]
    return lambda x: SP.cascade_channel(x, path, grid=grid)


def test_group_delay_adds_across_cascaded_line_sections():
    """Two sections in series delay by the sum of their delays. Measured off the waveform."""
    a, b = _line(125.0), _line(375.0)
    ga, gb = _group_delay_s(a), _group_delay_s(b)
    assert abs(ga * 1e12 - 125.0) < 1e-6 and abs(gb * 1e12 - 375.0) < 1e-6
    both = _group_delay_s(lambda x: b(a(x)))
    assert abs(both - (ga + gb)) < 1e-15, f"{both*1e12:.6f} ps != {(ga+gb)*1e12:.6f} ps"


def test_group_delay_adds_across_ops_of_different_kinds():
    """A channel section and a receiver filter are different code paths with different delay
    mechanisms (a linear phase and a pole pair). The chain still has to be their sum."""
    line = _line(125.0)
    ctle = lambda x: RX.ctle(x, WIDE, 2.0, 8.0, 16.0, dc_gain=0.6)
    g_line, g_ctle = _group_delay_s(line), _group_delay_s(ctle)
    assert g_ctle > 1e-12, f"CTLE has no group delay ({g_ctle*1e12:.4f} ps)"
    chain = _group_delay_s(lambda x: ctle(line(x)))
    assert abs(chain - (g_line + g_ctle)) < 0.1e-12, (
        f"chain {chain*1e12:.4f} ps, parts sum to {(g_line+g_ctle)*1e12:.4f} ps")
    assert abs(_group_delay_s(lambda x: line(ctle(x))) - chain) < 0.1e-12   # LTI: order free


def test_group_delay_adds_through_the_composer_not_just_the_primitives():
    """The same claim through `Signal`, so it is the composed chain being asserted."""
    g = Grid(fs=200e9, baud=16e9, n=1 << 15)
    def sig(tds):
        s = Signal(seed=3, grid=g).carrier("nrz", n_ui=1024, tr_frac=0.15, pattern="prbs7")
        for td in tds:
            s = s.cascade(path=[{"line": {"td_ps": td, "causal": False, "loss_length_in": 0.0}}])
        return s.waveform()
    ref = sig([])
    one, two = sig([200.0]), sig([200.0, 300.0])
    R = np.conj(np.fft.rfft(ref))
    def lag(y):                                   # circular cross-correlation peak, in samples
        c = np.fft.irfft(np.fft.rfft(y) * R, len(ref))
        return int(np.argmax(c))
    assert lag(one) == round(200e-12 * g.fs), f"one section lagged {lag(one)} samples"
    assert lag(two) == round(500e-12 * g.fs), f"two sections lagged {lag(two)} samples"


@pytest.mark.parametrize("kind,gd_ps", [("bessel", 7.776), ("gaussian", 3.881)])
def test_the_analog_front_end_is_causal_and_contributes_group_delay(kind, gd_ps):
    """WAS A STRICT XFAIL. The analog kinds ran `sosfiltfilt`, which is zero-phase, so the front
    end delayed 0.00 ps and put 35 % (bessel) / 22 % (gaussian) of its impulse response BEFORE
    the impulse. They are now single-pass causal: bessel by one forward `sosfilt`, gaussian by
    the minimum-phase response of the same |H| applied as a linear convolution.

    The delay is asserted against the CLOSED FORM, not against itself: an analog Bessel of
    order 4 whose -3 dB point is at ``fc`` has group delay ``2.1139/(2*pi*fc)`` at DC, which is
    8.411 ps at 40 GHz. Measured 7.776 ps -- 7.5 % low, and that is the bilinear warp at
    fc/Nyquist = 0.3125, not a missing delay. The gaussian's 3.881 ps is the minimum-phase
    delay of a Gaussian with the same corner, for which there is no closed form here, so it is
    pinned rather than derived."""
    g = Grid(fs=256e9, n=1 << 14)
    gd = _group_delay_s(lambda x: INST.scope_bandwidth(x, g, 40e9, kind=kind),
                        grid=g, band=(1e9, 10e9))
    imp = np.zeros(g.n); imp[g.n // 2] = 1.0
    h = INST.scope_bandwidth(imp, g, 40e9, kind=kind)
    pre = float(np.sum(h[:g.n // 2] ** 2) / np.sum(h ** 2))
    assert gd * 1e12 == pytest.approx(gd_ps, abs=0.02), f"group delay {gd*1e12:.4f} ps"
    assert pre < 1e-6, f"{pre:.3f} of the front end's impulse response is before t=0"
    if kind == "bessel":                      # against the analog prototype, not against itself
        assert abs(gd - 2.1139 / (2 * np.pi * 40e9)) / (2.1139 / (2 * np.pi * 40e9)) < 0.10

    # THE GATE OBSERVED FAILING: the opt-out still has neither property.
    gd0 = _group_delay_s(lambda x: INST.scope_bandwidth(x, g, 40e9, kind=kind, causal=False),
                         grid=g, band=(1e9, 10e9))
    h0 = INST.scope_bandwidth(imp, g, 40e9, kind=kind, causal=False)
    pre0 = float(np.sum(h0[:g.n // 2] ** 2) / np.sum(h0 ** 2))
    assert abs(gd0) * 1e12 < 0.02 and pre0 > 0.2, f"{gd0*1e12:.4f} ps, pre {pre0:.3f}"


# ======================================================== order sensitivity
def _chain(*ops):
    s = _base()
    for name, kw in ops:
        getattr(s, name)(**kw)
    return s.waveform()


LOSSY = ("lossy", dict(loss_db=12.0, loss_at_ghz=8.0, causal=True))
REFLECT = ("reflect", dict(td_ps=281.25, gamma_s=0.055, gamma_l=0.055, n_bounce=6, node="source"))


def test_lumped_loss_and_reflection_commute_which_is_why_they_carry_no_position():
    """`lossy` and `reflect` are both LTI, so composing them either way is the same operator.
    That is not a bug in either op -- it is the statement that the LUMPED pair carries no
    information about WHERE the discontinuity is, which is what
    `test_a_cascade_knows_where_the_discontinuity_is...` then measures.

    Asserting the commutation is the half that keeps the next test honest: if this ever stops
    holding, the two ops have acquired positional physics and the reason for the cascade has
    changed.

    IT NOW HOLDS OVER THE WHOLE RECORD, which it did not. `reflect` has always been a
    zero-padded time-domain shift; `lossy` used to be a CIRCULAR convolution, so the two
    disagreed at the record edges by the wrapped tail -- max 4.26e-2, whole-record relative rms
    3.74e-3, against an interior of 2.48e-4. With `lossy` applying its response as a linear
    convolution (U-16) both operators zero-pad and the disagreement collapses to 1.50e-7 / rel
    rms 1.17e-7 across the whole record, ~30,000x smaller and no longer edge-shaped. The
    residual is that each op truncates to the record: composing two truncations is not quite
    truncating the composition, which is why the head is still slightly worse than the tail.
    """
    a, b = _chain(LOSSY, REFLECT), _chain(REFLECT, LOSSY)
    guard = 2 * round(281.25e-12 * GRID.fs) * 6          # the whole reflection train's span
    interior = slice(guard, -guard)
    assert _rel_rms(a[interior], b[interior]) < 5e-4, (
        f"interior rel rms {_rel_rms(a[interior], b[interior]):.3e}")
    assert _rel_rms(a, b) < 1e-6, (
        f"whole-record rel rms {_rel_rms(a, b):.3e}: the edges are wrapping again")
    assert np.abs(a - b).max() < 1e-5, f"max |a-b| {np.abs(a - b).max():.3e}"


def test_a_cascade_knows_where_the_discontinuity_is_and_the_lumped_pair_does_not():
    """The positional physics, as a number. One discontinuity of the same Gamma on the same
    12-inch channel, at 1 / 6 / 11 inches from the driver: the cascade's echo pays 2x the loss
    of the line it actually traverses, so its amplitude falls by ~25x from near to far. The
    lumped `lossy -> reflect` pair gives the same echo whatever the distance -- that was a
    shipped recipe, and it is why a near and a far reflection were indistinguishable.
    """
    g = Grid(fs=256e9, baud=16e9, n=1 << 14)
    eps_r, gamma, total = 4.0, 0.3, 12.0
    ppi = SP.ps_per_inch(eps_r)
    imp = np.zeros(g.n); imp[256] = 1.0
    casc, lump = {}, {}
    for d in (1.0, 6.0, 11.0):
        lag = int(round(2 * d * ppi * 1e-12 * g.fs))
        win = slice(256 + lag - 60, 256 + lag + 60)
        # the ECHO is the difference the discontinuity makes, so the dispersed tail of the
        # direct pulse cannot be mistaken for it (it can, if you just take a peak near the lag)
        def _sect(gam):
            return [{"line": {"length_in": d, "eps_r": eps_r}}, {"disc": {"gamma": gam}},
                    {"line": {"length_in": total - d, "eps_r": eps_r}}]
        direct = SP.cascade_channel(imp, _sect(0.0), grid=g, node="source")
        casc[d] = float(np.abs((SP.cascade_channel(imp, _sect(gamma), grid=g, node="source")
                                - direct)[win]).max())
        base = P.lossy_channel(imp, grid=g, length_in=total, eps_r=eps_r, causal=True)
        y = P.multi_reflection(base, grid=g, td_ps=d * ppi, gamma_s=0.0, gamma_l=gamma,
                               n_bounce=1, node="source")
        lump[d] = float(np.abs((y - base)[win]).max())
    assert casc[1.0] / casc[11.0] > 10.0, (
        f"cascade near/far echo ratio {casc[1.0]/casc[11.0]:.2f} (near {casc[1.0]:.6f}, "
        f"far {casc[11.0]:.6f}) -- the cascade has lost the positional attenuation")
    assert casc[1.0] > casc[6.0] > casc[11.0]
    assert max(lump.values()) / min(lump.values()) - 1.0 < 1e-3, (
        f"the lumped pair suddenly distinguishes near from far ({lump}); if that is intended, "
        "this test and the cascade's reason for existing both change")
    # and the one place the lumped model is right: at half the channel, where the echo pays
    # half the loss twice, which is the whole loss once
    assert abs(casc[6.0] / lump[6.0] - 1.0) < 1e-3, f"{casc[6.0]:.8f} vs {lump[6.0]:.8f}"


@pytest.mark.parametrize("name,op", [
    ("memoryless nonlinearity", ("nonlinearity", dict(compression=0.2))),
    ("quantiser", ("digitize", dict(enob=6))),
    ("timing warp", ("timing", dict(pj=dict(amp_ps=2.0, f_hz=2e9)))),
])
def test_a_non_lti_stage_does_not_commute_with_the_channel(name, op):
    """Where the physics HAS order: compressing before a dispersive channel is not the same as
    compressing what came out of it, and neither is quantising, or warping the time axis. If a
    test cannot tell these apart, the model has lost the ordering, and 'transmitter defect' vs
    'receiver defect' stop being different labels."""
    before, after = _chain(op, LOSSY), _chain(LOSSY, op)
    assert _rel_rms(before, after) > 0.01, (
        f"{name} commutes with the channel (rel rms {_rel_rms(before, after):.3e})")


def test_op_order_within_a_kind_is_preserved_by_canonicalisation():
    """`canonical()` stable-sorts across stage kinds. Same-kind ops do not generally commute
    (the test above), so their relative order must survive the sort untouched."""
    s = _base().lossy(loss_db=8.0, loss_at_ghz=8.0).reflect(td_ps=100.0, gamma_s=0.2, gamma_l=0.2)
    s = s.digitize(enob=8).crosstalk(coupling=0.05)
    channel_ops = [o["op"] for o in s.canonical().ops if o["op"] in ("lossy", "reflect", "crosstalk")]
    assert channel_ops == ["lossy", "reflect", "crosstalk"]
    assert s.canonical().ops[-1]["op"] == "digitize"          # instrument stays last


# ======================================================== every knob does something
# (op, base kwargs, knob, alternative value). The claim for each row: moving that one knob,
# with the seed and every other knob fixed, changes the samples. A knob that is stored in the
# recipe and never read comes out of this sweep as a dead row.
KNOBS = [
    ("lossy", dict(loss_db=12.0, loss_at_ghz=8.0, causal=True), "loss_db", 6.0),
    ("lossy", dict(loss_db=12.0, loss_at_ghz=8.0, causal=True), "loss_at_ghz", 4.0),
    ("lossy", dict(loss_db=12.0, loss_at_ghz=8.0, causal=True), "causal", False),
    ("lossy", dict(length_in=6.0, tand=0.02), "tand", 0.005),
    ("lossy", dict(length_in=6.0, tand=0.02), "length_in", 12.0),
    ("lossy", dict(length_in=6.0, tand=0.02, eps_r=4.3), "eps_r", 2.5),
    ("lossy", dict(length_in=6.0, tand=0.02, skin_k=0.35), "skin_k", 0.7),
    ("lossy", dict(trend=(-1.0, -0.3, 0.0)), "trend", (-2.0, -0.3, 0.0)),
    ("lossy", dict(trend=(-40.0, -8.0, 0.0)), "trend_floor_db", 20.0),
    ("reflect", dict(td_ps=281.25, gamma_s=0.3, gamma_l=0.3, n_bounce=6, node="source"),
     "gamma_s", 0.6),
    ("reflect", dict(td_ps=281.25, gamma_s=0.3, gamma_l=0.3, n_bounce=6, node="source"),
     "gamma_l", 0.6),
    ("reflect", dict(td_ps=281.25, gamma_s=0.3, gamma_l=0.3, n_bounce=6, node="source"),
     "td_ps", 140.0),
    ("reflect", dict(td_ps=281.25, gamma_s=0.3, gamma_l=0.3, n_bounce=6, node="source"),
     "n_bounce", 1),
    ("reflect", dict(td_ps=281.25, gamma_s=0.3, gamma_l=0.3, n_bounce=6, node="source"),
     "node", "load"),
    ("resonant_reflect", dict(td_ps=40.0, f0_ghz=25.0, q=10.0, gamma0=0.5), "f0_ghz", 40.0),
    ("resonant_reflect", dict(td_ps=40.0, f0_ghz=25.0, q=10.0, gamma0=0.5), "q", 3.0),
    ("resonant_reflect", dict(td_ps=40.0, f0_ghz=25.0, q=10.0, gamma0=0.5), "gamma0", 0.2),
    ("resonant_reflect", dict(td_ps=40.0, f0_ghz=25.0, q=10.0, gamma0=0.5), "td_ps", 120.0),
    ("ac_couple", dict(fc_hz=1e8), "fc_hz", 1e9),
    ("crosstalk", dict(coupling=0.1, kind="fext"), "coupling", 0.3),
    ("crosstalk", dict(coupling=0.1, kind="fext"), "kind", "next"),
    ("crosstalk", dict(coupling=0.1, kind="next", td_frac=0.05), "td_frac", 0.2),
    ("tx_ffe", dict(taps=[1.0, -0.2], pre=0), "taps", [1.0, -0.4]),
    ("tx_ffe", dict(taps=[-0.1, 1.0, -0.2], pre=1), "pre", 0),
    ("de_emphasis", dict(db=3.5), "db", 6.0),
    ("nonlinearity", dict(compression=0.05), "compression", 0.2),
    ("nonlinearity", dict(compression=0.05, level_noise=0.01), "level_noise", 0.05),
    ("nonlinearity", dict(compression=0.05, rise_fall_ratio=1.5), "rise_fall_ratio", 2.5),
    ("nonlinearity", dict(compression=0.05, rise_fall_ratio=1.5, a_base=0.4), "a_base", 0.2),
    ("supply_coupling", dict(f_ripple_hz=1e9, am_depth=0.05), "am_depth", 0.2),
    ("supply_coupling", dict(f_ripple_hz=1e9, am_depth=0.05), "f_ripple_hz", 4e9),
    ("supply_coupling", dict(f_ripple_hz=1e9, psij_ps=1.0), "psij_ps", 4.0),
    ("intra_pair_skew", dict(skew_ps=2.0), "skew_ps", 8.0),
    ("intra_pair_skew", dict(skew_ps=2.0, gain_imbalance=0.02), "gain_imbalance", 0.1),
    ("ssc", dict(spread=0.005, f_ssc=32e3), "spread", 0.002),
    ("ssc", dict(spread=0.005, f_ssc=32e3), "f_ssc", 100e6),
    ("ssc", dict(spread=0.005, f_ssc=32e3), "profile", "center"),
    ("timing", dict(rj_ps=0.5), "rj_ps", 2.0),
    ("timing", dict(pj=dict(amp_ps=2.0, f_hz=1e9)), "pj", dict(amp_ps=6.0, f_hz=1e9)),
    ("drift", dict(kind="gain", amount=0.05), "amount", 0.2),
    ("ctle", dict(fz_ghz=2.0, fp1_ghz=8.0, fp2_ghz=16.0, dc_gain=0.7), "fz_ghz", 1.0),
    ("ctle", dict(fz_ghz=2.0, fp1_ghz=8.0, fp2_ghz=16.0, dc_gain=0.7), "fp1_ghz", 4.0),
    ("ctle", dict(fz_ghz=2.0, fp1_ghz=8.0, fp2_ghz=16.0, dc_gain=0.7), "fp2_ghz", 30.0),
    ("ctle", dict(fz_ghz=2.0, fp1_ghz=8.0, fp2_ghz=16.0, dc_gain=0.7), "dc_gain", 0.4),
    ("rx_ffe", dict(taps=[1.0, -0.2], spacing_ui=0.5, pre=0), "taps", [1.0, -0.5]),
    ("probe", dict(c_load_f=1e-12), "c_load_f", 4e-12),
    ("probe", dict(c_load_f=1e-12, r_source=50.0), "r_source", 200.0),
    ("probe", dict(c_load_f=1e-12, bw_hz=40e9), "bw_hz", 10e9),
    ("probe", dict(c_load_f=1e-12, noise_rms=0.0), "noise_rms", 0.02),
    ("probe", dict(c_load_f=1e-12), "atten", 0.1),
    ("scope", dict(bw_hz=60e9), "bw_hz", 20e9),
    ("scope", dict(bw_hz=60e9), "kind", "gaussian"),
    ("scope", dict(bw_hz=60e9, order=4), "order", 2),
    ("timebase", dict(rms_ps=0.5), "rms_ps", 2.0),
    ("digitize", dict(snr_db=40.0), "snr_db", 20.0),
    ("digitize", dict(enob=8), "enob", 4),
    ("digitize", dict(bits=8), "bits", 4),
    ("digitize", dict(noise_rms=0.01), "noise_rms", 0.05),
    ("digitize", dict(bits=8, full_scale=1.0), "full_scale", 2.0),
    ("digitize", dict(snr_db=40.0, interleave=dict(m_cores=4, offset_mm=0.01)),
     "interleave", dict(m_cores=4, offset_mm=0.05)),
    # U-13 / U-12 / U-09
    ("dcd", dict(ps=2.0), "ps", 5.0),
    ("dcd", dict(frac_ui=0.02), "frac_ui", 0.06),
    ("agc", dict(target=0.5), "target", 1.0),
    ("agc", dict(target=0.5, metric="rms"), "metric", "peak"),
    ("rx_noise", dict(rms=0.005), "rms", 0.02),
    ("rx_noise", dict(density=2e-6, bw_hz=20e9), "bw_hz", 40e9),
    ("sample_clock", dict(ppm=100.0), "ppm", 300.0),
    ("sample_clock", dict(ppm=100.0), "phase0_s", 1.4e-12),
    ("sample_clock", dict(ppm=100.0), "drift_ppm_per_s", 5.0e7),
]


@pytest.mark.parametrize("op,kw,knob,alt", KNOBS,
                         ids=[f"{o}.{k}" for o, _, k, _ in KNOBS])
def test_every_knob_moves_the_output(op, kw, knob, alt):
    """Move one knob; hold the seed and everything else. The samples must change.

    A row that fails here is a parameter the recipe records and the physics ignores -- the
    exact shape of the `noise_mv` defect. The threshold (1e-4 of the signal's rms) is more
    than an order of magnitude below the smallest real effect in this table, so a row failing
    means dead, not subtle.
    """
    a, b = _with(op, **kw), _with(op, **{**kw, knob: alt})
    rel = _rel_rms(a, b)
    assert rel > 1e-4, f"{op}.{knob}: {kw[knob]!r} -> {alt!r} changed the output by {rel:.3e}"


def test_the_knob_sweep_covers_every_op_the_composer_can_execute():
    """The sweep is only a guarantee if it is complete. Any op reachable from `Signal` that is
    not represented above is an unswept surface, and this fails until it is listed or
    explicitly excused."""
    from wfmsynth.compose import _EXEC
    covered = {op for op, _, _, _ in KNOBS} | {op for op, _ in EXACT_BYPASS} \
        | {op for op, _ in TRANSFORM_BYPASS}
    # ops with no scalar knob of their own, or exercised by their own dedicated suites
    excused = {"carrier", "symbols", "events", "store", "acquire", "cascade", "sparam",
               "crosstalk_matrix", "dfe", "optical", "dispersion", "eo", "fiber",
               "optical_mpi", "edfa", "photodetect", "tia"}
    missing = set(_EXEC) - covered - excused
    assert not missing, f"ops with no knob-moves-the-output row: {sorted(missing)}"


def test_moving_a_knob_and_rerolling_a_seed_are_different_kinds_of_change():
    """A knob change must survive a re-roll: if the difference a knob makes is smaller than
    the difference the seed makes, the sweep above would pass on noise alone."""
    base = dict(td_ps=281.25, gamma_s=0.3, gamma_l=0.3, n_bounce=6, node="source")
    a = _with("reflect", **base)
    knob = _with("reflect", **{**base, "gamma_l": 0.6})
    s = Signal(seed=99, grid=GRID).carrier("nrz", n_ui=512, tr_frac=0.15, pattern="prbs7")
    seeded = s.reflect(**base).waveform()
    assert _rel_rms(a, knob) > 0.05
    assert _rel_rms(a, knob) > 0.5 * _rel_rms(a, seeded) or _rel_rms(a, seeded) < 1e-12

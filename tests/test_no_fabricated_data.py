"""Assertions against a CLASS of defect: an op that fabricates or destroys information.

The defect that motivated this file: `dfe` evaluated one equalised value per symbol and rebuilt
a waveform from those values alone. It passed every test it had — the decisions were right, the
symbol-error rate was right — while DELETING 30.8 dB of its input's noise and every trace of the
15 samples in 16 that fall between decision instants. Nothing caught it because every existing
assertion looked at the decisions, and the defect was in the waveform.

Four general properties, each applied to every op that claims to be a waveform-to-waveform stage.
They are cheap, they need no reference output, and each one would have caught that defect.
"""
import numpy as np
import pytest

from wfmsynth.compose import Signal, _EXEC, _LEAD_LTI
from wfmsynth.grid import Grid
from wfmsynth.streams import Streams

FS, BAUD, N_UI = 256e9, 16e9, 4000
SPB = int(FS / BAUD)


def _grid():
    return Grid(fs=FS, baud=BAUD, n=N_UI * SPB, v_full=0.8)


def _input(seed=3):
    g = _grid()
    sym = np.random.default_rng(seed).choice([-1.0, 1.0], N_UI)
    return (Signal(seed=1, grid=g).symbols(sym.tolist(), tr_frac=0.35, causal=True)
            .lossy(causal=True, loss_db=10.0, loss_at_ghz=8.0)
            .digitize(noise_rms=0.01)).waveform(), g


# Ops under test: waveform -> waveform stages a chain can place after a channel. Sources
# (`carrier`, `symbols`) and multi-signal/optical stages are excluded because they do not take a
# meaningful electrical waveform in.
# A synthetic measured response for the `sparam` stage: the same sqrt(f)+f loss law the
# analytic channel uses, with a delay, so the op exercises its real interpolation path.
#
# It reaches the record's own NYQUIST, deliberately. The superposition probe below is white
# noise, so a response that stopped short would leave most of the record's power out of band --
# and `sparam` refuses that rather than silently zeroing it (measured: 69 % above a 40 GHz top).
# The refusal is correct; a fixture that triggers it is testing the guard, not linearity.
_SP_FREQS = np.linspace(0.0, FS / 2.0, 1025)
_SP_S21 = (10 ** (-(0.02 * np.sqrt(_SP_FREQS / 1e9) + 0.01 * _SP_FREQS / 1e9) / 20)
           * np.exp(-1j * 2 * np.pi * _SP_FREQS * 3e-10))

STAGES = {
    "lossy": dict(loss_db=6.0, loss_at_ghz=8.0, causal=True),
    "reflect": dict(td_ps=40.0, gamma_s=0.06, gamma_l=0.06, n_bounce=3),
    "resonant_reflect": dict(td_ps=90.0, f0_ghz=10.7, q=9.0, gamma0=0.2),
    "ctle": dict(fz_ghz=2.0, fp1_ghz=8.0, fp2_ghz=16.0, dc_gain=0.65),
    "rx_ffe": dict(taps=[-0.05, 1.0, -0.18], spacing_ui=1.0, pre=1),
    "dfe": dict(taps=[0.12, 0.05], levels=[-1.0, 1.0]),
    "tx_ffe": dict(taps=[-0.1, 0.75, -0.15], pre=1),
    "de_emphasis": dict(db=3.0),
    "scope": dict(bw_hz=32e9, kind="bessel", order=4),   # analog: single-pass causal
    "ac_couple": dict(fc_hz=2e9),
    "supply_coupling": dict(f_ripple_hz=2.5e6, am_depth=0.02),
    "nonlinearity": dict(compression=0.04),
    "drift": dict(kind="gain", amount=0.004, shape="linear"),
    "probe": dict(c_load_f=0.45e-12, r_source=50.0),
    # Classified LTI in `compose._LEAD_LTI` and, until the gate below, never checked for it.
    "cascade": dict(path=[{"line": {"length_in": 3.0}}, {"disc": {"gamma": 0.06}},
                          {"line": {"length_in": 3.0}}]),
    "sparam": dict(freqs=_SP_FREQS, s21=_SP_S21),
    "intra_pair_skew": dict(skew_ps=3.0, gain_imbalance=0.02),
    # U-09/U-12/U-13. `sample_clock` is LINEAR (the sample positions are a function of k, not of
    # x) but not time-invariant, so it belongs in LTI below and not in CORNER_CASES. `agc` is
    # deliberately NOT in LTI: its gain is a functional of the input's own level, so
    # `agc(3x) == agc(x)` -- deleting the absolute level is the whole point of the stage.
    # `rx_noise` adds a signal-independent term and `dcd` warps time by a field derived from the
    # input's own crossings; neither is linear either.
    # band_tol=None because `test_an_op_that_claims_to_be_linear_obeys_superposition` probes
    # with WHITE noise, 22.768 % of whose energy is above the interpolator's 0.386*fs passband,
    # and `sample_clock` correctly refuses that record. The realistic input `_input()` builds is
    # 0.008769 % out of band and does not trip it; the guard's own gate is
    # `test_the_interpolator_is_wrong_above_its_passband_and_says_so` in test_link_physics.py.
    "sample_clock": dict(ppm=300.0, band_tol=None),
    "agc": dict(target=0.5, metric="rms"),
    "rx_noise": dict(rms=0.004),
    "dcd": dict(ps=4.0),
}


def _run(op, params, x, g, seed=0):
    return np.asarray(_EXEC[op](x, dict(params, op=op), Streams(seed), g, 0), float)


@pytest.mark.parametrize("op", sorted(STAGES))
def test_every_input_sample_influences_the_output(op):
    """THE assertion that catches a decimator masquerading as a waveform stage.

    Perturb ONE interior sample and the output must move. An op that reads one sample per symbol
    and rebuilds the rest ignores 15 of every 16 samples, so perturbing one of those 15 changes
    nothing at all — which is what `dfe` did before it returned its summing node.
    """
    x, g = _input()
    base = _run(op, STAGES[op], x, g)
    # probe several offsets within one UI so a decimator cannot pass by luck of alignment
    unmoved = []
    for off in range(SPB):
        k = (N_UI // 2) * SPB + off
        y = x.copy()
        y[k] += 0.05                                 # 5 % of full scale: far above any tolerance
        moved = float(np.max(np.abs(_run(op, STAGES[op], y, g) - base)))
        if moved < 1e-9:
            unmoved.append(off)
    assert not unmoved, (f"{op}: perturbing the sample at these offsets within a UI changed "
                         f"NOTHING in the output: {unmoved} of {SPB}. The op is ignoring part "
                         f"of its input.")


# DERIVED from the classification, not restated beside it. `compose._LEAD_LTI` is what the
# lead-in machinery believes is linear, and it SIZES THE GUARD on that belief -- it builds a
# combined impulse response by superposing the ops. So the list that gets superposition-tested
# has to be that same list, or the belief goes unchecked exactly where it is load-bearing.
#
# It was restated, and it had drifted: `cascade`, `sparam` and `intra_pair_skew` were classified
# LTI and never tested for it. All three measure ~4e-16, so this closed a coverage gap rather
# than a defect -- but a regression in any of them would have corrupted guard sizing silently.
LTI_EXTRA = {
    # Linear, but NOT time-invariant (the sample positions are a function of k, not of x), so it
    # is deliberately not in `_LEAD_LTI` -- and superposition still has to hold.
    "sample_clock",
}
LTI_EXCUSED: dict = {}          # op -> why superposition cannot be asserted. Empty, and should
                                # stay that way: an op in `_LEAD_LTI` that is not linear is a
                                # misclassification, not an exception.
LTI = sorted((set(_LEAD_LTI) | LTI_EXTRA) - set(LTI_EXCUSED))


def test_every_op_classified_linear_can_be_exercised():
    """Completeness is enforced by the DERIVATION above, not by this test: because `LTI` is
    built from `_LEAD_LTI`, an op added there is automatically parametrized into
    `test_an_op_that_claims_to_be_linear_obeys_superposition` and fails it if it is not linear.
    Verified by adding a non-linear op to `_LEAD_LTI` and watching that test fail, rather than
    assumed.

    So asserting "every `_LEAD_LTI` op is in `LTI`" would be tautological -- it cannot fail, and
    a test that cannot fail looks like protection while providing none.

    What this DOES check is the one way the derivation can still be defeated: an op with no
    `STAGES` entry cannot be exercised, so it would be silently skipped."""
    cannot_run = sorted(op for op in LTI if op not in STAGES)
    assert not cannot_run, (
        f"classified LTI (or listed in LTI_EXTRA) with no STAGES entry to exercise it, so "
        f"superposition is never actually asserted for it: {cannot_run}")
    for op in LTI_EXCUSED:
        assert op in _LEAD_LTI, f"{op!r} is excused from a list it is not on"


@pytest.mark.parametrize("op", LTI)
def test_an_op_that_claims_to_be_linear_obeys_superposition(op):
    """``op(a + b) == op(a) + op(b)`` and ``op(k*a) == k*op(a)``.

    Superposition is the definition of linearity, it needs no spectral division, and it is
    immune to how an op pads or truncates. Anything that decides, clamps, quantises or
    reconstructs breaks it immediately.

    TWO EARLIER VERSIONS OF THIS TEST DID NOT WORK, and both failures are worth recording.
    Watching the noise floor cannot separate honest filtering from fabrication: measured, the
    `dfe` re-render deleted 11.8 dB of the top of the band while `lossy` legitimately took
    33.8 dB, `scope` 32.1 and `probe` 23.8 — the defect sat BELOW three honest stages. And
    measuring |H| on one record to predict another fails for a different reason: the channel
    now applies a LINEAR convolution, so the output is a truncated linear convolution and the
    DFT relation ``Y = H*X`` does not hold at all. Superposition survives both problems.
    """
    g = _grid()
    rng = np.random.default_rng(5)
    a = rng.standard_normal(g.n) * 0.3
    b = rng.standard_normal(g.n) * 0.3
    ya, yb, yab = (_run(op, STAGES[op], v, g) for v in (a, b, a + b))
    m = min(len(ya), len(yb), len(yab))
    add = float(np.max(np.abs(yab[:m] - (ya[:m] + yb[:m])))) / (float(np.std(yab[:m])) + 1e-30)
    y3 = _run(op, STAGES[op], 3.0 * a, g)
    m2 = min(len(y3), len(ya))
    homo = float(np.max(np.abs(y3[:m2] - 3.0 * ya[:m2]))) / (float(np.std(y3[:m2])) + 1e-30)
    assert add < 1e-9 and homo < 1e-9, (
        f"{op}: additivity {add:.2e}, homogeneity {homo:.2e} (relative to output rms). It is "
        f"listed as linear but is not — it decides, clamps, quantises or reconstructs.")


# (op, corner keyword, corner, the rest of the op's params, allowed ratio band). The scope
# rows are the analog front end at four settings plus the DIGITAL selected-bandwidth filter,
# and they are held to 2 % rather than the 2x this test was written with, because a single-pass
# design designed with `norm='mag'` realises the corner it was asked for and there is no reason
# to allow it any slack.
CORNER_CASES = [
    ("ac_couple", "fc_hz", 2e9, {}, 2.0),
    ("scope", "bw_hz", 32e9, dict(kind="bessel", order=4), 1.02),
    ("scope", "bw_hz", 32e9, dict(kind="bessel", order=2), 1.02),
    ("scope", "bw_hz", 32e9, dict(kind="bessel", order=6), 1.02),
    ("scope", "bw_hz", 32e9, dict(kind="gaussian"), 1.02),
    ("scope", "bw_hz", 32e9, dict(kind="brickwall"), 1.02),
]


def _realised_corner(op, fc_key, fc, extra, g, rising):
    """The -3 dB point of an op, bisected on the realised output. Returns Hz."""
    n = g.n
    def gain(f_hz):
        k = max(1, round(f_hz * n / FS))
        t = np.sin(2 * np.pi * k * np.arange(n) / n)
        y = _run(op, dict(extra, **{fc_key: fc}), t, g)
        return abs(np.fft.rfft(y[:n])[k] / np.fft.rfft(t)[k])
    lo, hi = fc / 60.0, min(0.45 * FS, fc * 60.0)
    for _ in range(34):                                  # bisect the |H| = 1/sqrt(2) point
        mid = np.sqrt(lo * hi)
        if (gain(mid) < 0.70710678) == rising:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


@pytest.mark.parametrize("op,fc_key,fc,extra,tol", CORNER_CASES,
                         ids=lambda v: str(v) if not isinstance(v, dict) else
                         "-".join(f"{k}{x}" for k, x in v.items()) or "plain")
def test_a_filter_realises_the_corner_it_was_asked_for(op, fc_key, fc, extra, tol):
    """Catches a silent clamp, a doubly-applied pole, and a filter normalised to the wrong
    thing. `ac_couple` clamped any corner below 1e-4 of Nyquist — a 50 kHz request became
    12.8 MHz, 256x off, with nothing said. `scope`'s analog kinds were worse in a quieter way:
    they ran a Bessel forwards AND backwards, squaring |H|, on top of designing it with
    scipy's default ``norm='phase'`` whose own -3 dB point is at 0.68 of ``Wn``. Asked for
    32 GHz, a bessel-4 front end realised 15.90 GHz — HALF — so every chain in this library
    that stated an instrument bandwidth was getting half of it.

    The realised corner is measured on the op's own output, by bisection, and compared with
    the request. `test_the_zero_phase_opt_out_still_fails_this` is the same measurement
    observed FAILING."""
    g = _grid()
    realised = _realised_corner(op, fc_key, fc, extra, g, rising=(op == "ac_couple"))
    assert 1.0 / tol < realised / fc < tol, (f"{op} {extra}: asked for {fc:.4g} Hz, realised "
                                             f"{realised:.4g} Hz ({realised / fc:.4f}x)")


# What the pre-fix zero-phase analog path realises, as a fraction of the request. MEASURED at
# bw_hz=32e9 on fs=256e9. Two independent errors compound: the double pass squares |H|, and
# `norm='phase'` puts the single-pass -3 dB at 0.68*Wn to begin with.
ZERO_PHASE_RATIO = {("bessel", 4): 0.497, ("bessel", 2): 0.582, ("bessel", 6): 0.432,
                    ("gaussian", 4): 0.833}


@pytest.mark.parametrize("kind,order", sorted(ZERO_PHASE_RATIO))
def test_the_zero_phase_opt_out_still_fails_this(kind, order):
    """THE GATE OBSERVED FAILING. A check never observed failing is not a check, so the defect
    is kept reachable — `causal=False` — and measured. Each ratio below is what the front end
    realised before the fix, and the bessel-4 row is the 0.497 that made a 32 GHz front end a
    15.9 GHz one."""
    g = _grid()
    extra = dict(kind=kind, order=order, causal=False)
    realised = _realised_corner("scope", "bw_hz", 32e9, extra, g, rising=False)
    assert realised / 32e9 == pytest.approx(ZERO_PHASE_RATIO[(kind, order)], abs=0.005)
    with pytest.raises(AssertionError):
        test_a_filter_realises_the_corner_it_was_asked_for("scope", "bw_hz", 32e9, extra, 1.02)


def test_the_analog_and_digital_band_limits_are_not_the_same_operator():
    """The distinction is the whole point of the fix, so it is asserted rather than commented.
    An ANALOG kind delays and is strictly causal; the DIGITAL selected-bandwidth filter that
    sits after the converter is zero-phase, and asking it for a causal form raises instead of
    quietly obliging."""
    import wfmsynth.instrument as INST
    g = _grid()
    imp = np.zeros(g.n); imp[g.n // 2] = 1.0
    for kind in INST.ANALOG_KINDS:
        h = INST.scope_bandwidth(imp, g, 32e9, kind=kind)
        pre = float(np.sum(h[:g.n // 2] ** 2) / np.sum(h ** 2))
        assert pre < 1e-6, f"{kind}: {pre:.4f} of the impulse response is before t=0"
    for kind in INST.DIGITAL_KINDS:
        h = INST.scope_bandwidth(imp, g, 32e9, kind=kind)
        pre = float(np.sum(h[:g.n // 2] ** 2) / np.sum(h ** 2))
        # 0.375, not 0.5: the sinc's centre tap alone carries bw/nyquist = 1/4 of the energy,
        # and it sits in the post-impulse half. (1 - 0.25)/2 = 0.375.
        assert pre > 0.3, f"{kind} should be symmetric about t=0, pre {pre:.4f}"
        with pytest.raises(ValueError):
            INST.scope_bandwidth(imp, g, 32e9, kind=kind, causal=True)
    with pytest.raises(ValueError):
        INST.scope_bandwidth(imp, g, 32e9, kind="lowpass")

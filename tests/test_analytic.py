"""Layer 2 -- analytic ground truth: closed form, never a previous run.

Every assertion here compares a MEASURED quantity, read back off a realised waveform or a
realised transfer function, against an arithmetic answer written out by hand in the test.
Nothing is compared against a stored output, so a deliberate physics change moves these tests
only when it moves the physics.

Why this layer and not a golden file: a golden file tells you the output did not change. It
cannot tell you the output was ever right. A channel anchored to the wrong Nyquist was 9x off
here and reproduced itself byte for byte every run; `test_stated_loss_is_the_loss_measured_at_
that_frequency` is the shape of test that catches it, because it names the frequency in Hz and
goes and looks.

Measurement conventions used throughout:
  * a "tone" is a sinusoid at an EXACT rfft bin, so its amplitude ratio has no leakage;
  * a transfer function is measured as rfft(y)/rfft(x), i.e. read off the realised output;
  * dB of loss is positive (`-20*log10|H|`), matching `insertion_loss_db`.
"""
import numpy as np
import pytest
from scipy import signal as _sig

import wfmsynth.cdr as CDR
import wfmsynth.instrument as INST
import wfmsynth.physics as P
import wfmsynth.rx as RX
import wfmsynth.sparam as SP
from wfmsynth.grid import Grid

C_IN_PER_NS = 11.8028          # the same constant sparam states; restated so the test is
                               # arithmetic against a number, not against the module


# ------------------------------------------------------------------ measurement helpers
def _bin(grid, f_hz):
    """The rfft bin nearest f_hz, and the exact frequency that bin stands for."""
    k = int(round(f_hz * grid.n / grid.fs))
    return k, k * grid.fs / grid.n


def _tone(grid, k):
    return np.sin(2.0 * np.pi * k * np.arange(grid.n) / grid.n)


def _tone_loss_db(fn, grid, f_hz):
    """Insertion loss in dB that `fn` applies at f_hz, measured from the output samples."""
    k, f = _bin(grid, f_hz)
    x = _tone(grid, k)
    y = fn(x)
    return -20.0 * np.log10(abs(np.fft.rfft(y)[k]) / abs(np.fft.rfft(x)[k])), f


def _transfer(fn, grid, seed=0):
    """|H| and H measured from a broadband record: rfft(y)/rfft(x) on the realised output.

    MEASURED ON THE PINNED-LENGTH PATH. `rfft(y)/rfft(x)` recovers H exactly only when y is the
    CIRCULAR convolution of x with h -- that is the eigen-relation the ratio relies on. The ops'
    default is now a linear convolution (`physics.apply_transfer`), whose output is the same H
    applied with a zero lead-in and truncated, and a whole-record DFT of that is H plus an edge
    term (0.002-0.02 dB on the channels below, and up to 0.6 in |H| where |H| is small). So the
    transfer-function assertions here pass `linear=False` and read H off the eigen-path.

    What that leaves uncovered -- that the padded path applies this same H, and applies it as a
    linear convolution -- is `tests/test_linear_convolution.py`, against `np.convolve` and
    against a two-tap echo whose closed form is written out by hand."""
    x = np.random.default_rng(seed).standard_normal(grid.n)
    y = fn(x)
    return np.fft.rfft(y) / np.fft.rfft(x), np.fft.rfftfreq(grid.n, d=grid.dt)


# ===================================================================== channel loss
@pytest.mark.parametrize("fs,n", [(256e9, 1 << 14), (100e9, 1 << 13), (64e9, 1 << 15)])
@pytest.mark.parametrize("loss_db,at_ghz", [(12.0, 8.0), (3.0, 4.0), (28.0, 16.0)])
@pytest.mark.parametrize("causal", [False, True])
def test_stated_loss_is_the_loss_measured_at_that_frequency(fs, n, loss_db, at_ghz, causal):
    """A channel asked for X dB at f GHz must MEASURE X dB at f GHz -- on the grid's own
    frequency axis, whatever the sample rate and record length.

    This is the anchoring test. A channel that resolves `loss_at_ghz` against a hard-wired
    Nyquist instead of the grid's passes at one fs and fails at the others, which is exactly
    how a 9x loss error hid here.
    """
    g = Grid(fs=fs, baud=16e9, n=n)
    meas, f = _tone_loss_db(
        lambda x: P.lossy_channel(x, grid=g, loss_db=loss_db, loss_at_ghz=at_ghz, causal=causal),
        g, at_ghz * 1e9)
    assert abs(f - at_ghz * 1e9) < g.fs / g.n            # the bin really is that frequency
    assert abs(meas - loss_db) < 0.01, f"asked {loss_db} dB at {at_ghz} GHz, measured {meas:.4f}"


@pytest.mark.parametrize("f_ghz", [1.0, 2.0, 4.0, 8.0, 12.0, 16.0])
def test_loss_across_the_band_is_the_skin_plus_dielectric_law(f_ghz):
    """IL(f) = (0.35*sqrt(f) + 2.3*sqrt(eps_r)*tand*f) * length, scaled to hit the anchor.

    The whole curve, written out by hand, not just the anchor point. One anchor pins one
    frequency and leaves the shape free; this is the shape.
    """
    g = Grid(fs=256e9, baud=16e9, n=1 << 14)
    length_in, tand, eps_r, loss_db, at_ghz = 6.0, 0.02, 4.3, 12.0, 8.0
    meas, f = _tone_loss_db(
        lambda x: P.lossy_channel(x, grid=g, length_in=length_in, tand=tand, eps_r=eps_r,
                                  loss_db=loss_db, loss_at_ghz=at_ghz, causal=True,
                                  linear=False),
        g, f_ghz * 1e9)
    a_skin, b_diel = 0.35, 2.3 * np.sqrt(eps_r) * tand
    shape = lambda fg: (a_skin * np.sqrt(fg) + b_diel * fg) * length_in
    want = shape(f / 1e9) * (loss_db / shape(at_ghz))
    assert abs(meas - want) < 1e-6, f"{f/1e9:.3f} GHz: measured {meas:.6f}, closed form {want:.6f}"


@pytest.mark.parametrize("trend", [(-1.2, -0.35, -0.4), (-4.0, 0.0, 0.0), (0.0, -0.9, 0.0)])
@pytest.mark.parametrize("f_ghz", [2.0, 8.0, 20.0])
def test_a_fitted_trend_measures_its_own_polynomial(trend, f_ghz):
    """`trend=(a,b,c)` claims |S21|(f)[dB] = a*sqrt(f) + b*f + c. Measure it and check."""
    g = Grid(fs=256e9, baud=16e9, n=1 << 14)
    meas, f = _tone_loss_db(lambda x: P.lossy_channel(x, grid=g, trend=trend, causal=True,
                                                      linear=False), g, f_ghz * 1e9)
    a, b, c = trend
    want = -(a * np.sqrt(f / 1e9) + b * (f / 1e9) + c)
    assert abs(meas - want) < 1e-6, f"measured {meas:.6f} dB, trend says {want:.6f} dB"


def test_the_trend_stop_band_floor_is_exactly_the_stated_cap():
    """`trend_floor_db` is a hard cap on the stop band, not a soft knee: past the frequency
    where the fit reaches it, the measured loss is that number and stays that number."""
    g = Grid(fs=256e9, baud=16e9, n=1 << 14)
    trend, floor = (-8.0, -2.0, 0.0), 40.0
    unbounded = lambda fg: -(trend[0] * np.sqrt(fg) + trend[1] * fg)
    f_hit = next(fg for fg in np.arange(1.0, 128.0, 0.1) if unbounded(fg) > floor)
    below, _ = _tone_loss_db(lambda x: P.lossy_channel(x, grid=g, trend=trend, linear=False,
                                                       trend_floor_db=floor), g, 2e9)
    assert abs(below - unbounded(2.0)) < 1e-6                 # under the cap: the fit itself
    for f_ghz in (f_hit + 5.0, f_hit + 20.0, 120.0):
        deep, _ = _tone_loss_db(lambda x: P.lossy_channel(x, grid=g, trend=trend, linear=False,
                                                          trend_floor_db=floor), g, f_ghz * 1e9)
        assert abs(deep - floor) < 1e-6, f"{f_ghz:.1f} GHz measured {deep:.4f} dB, cap {floor}"


def test_the_causal_channel_keeps_the_magnitude_and_moves_the_energy_after_t0():
    """Minimum phase is a PHASE claim: |H| must be untouched (Kramers-Kronig links them, it
    does not trade them), and the impulse response must sit after t=0."""
    g = Grid(fs=256e9, baud=16e9, n=1 << 14)
    kw = dict(grid=g, loss_db=14.0, loss_at_ghz=8.0, linear=False)   # see `_transfer`
    for f_ghz in (1.0, 4.0, 8.0, 20.0):
        zp, _ = _tone_loss_db(lambda x: P.lossy_channel(x, causal=False, **kw), g, f_ghz * 1e9)
        cz, _ = _tone_loss_db(lambda x: P.lossy_channel(x, causal=True, **kw), g, f_ghz * 1e9)
        assert abs(zp - cz) < 1e-9, f"{f_ghz} GHz: |H| moved by {abs(zp-cz):.2e} dB"
    imp = np.zeros(g.n); imp[g.n // 2] = 1.0
    h = P.lossy_channel(imp, causal=True, **kw)
    pre = float(np.sum(h[:g.n // 2] ** 2)); post = float(np.sum(h[g.n // 2:] ** 2))
    assert pre / post < 1e-3, f"pre-cursor energy fraction {pre/post:.3e} (should be ~0)"
    h0 = P.lossy_channel(imp, causal=False, **kw)
    assert np.sum(h0[:g.n // 2] ** 2) / np.sum(h0[g.n // 2:] ** 2) > 0.5   # zero-phase: symmetric


# ===================================================================== reflections in time
@pytest.mark.parametrize("node", ["load", "source"])
def test_a_whole_sample_echo_arrives_at_the_arithmetic_tap(node):
    """`multi_reflection(td_ps=...)` puts bounce k at 2*k*td samples with weight (gs*gl)^k*gl at
    the load and (gs*gl)^(k-1)*gl at the source. Both are written out here and checked tap by tap
    off an impulse response -- position AND amplitude, not "an echo exists".

    A delay that is a WHOLE number of samples must stay bit-exact and sparse: it needs no
    interpolation, so it must not pick up any.
    """
    g = Grid(fs=256e9, baud=16e9, n=1 << 13)
    td_ps = 281.25                                   # exactly 72 samples at 256 GSa/s
    assert (td_ps * 1e-12 * g.fs).is_integer()
    gs, gl, n_bounce = 0.30, 0.40, 5
    imp = np.zeros(g.n); imp[64] = 1.0
    y = P.multi_reflection(imp, grid=g, td_ps=td_ps, gamma_s=gs, gamma_l=gl,
                           n_bounce=n_bounce, node=node)
    d = round(td_ps * 1e-12 * g.fs)
    want = {64: 1.0}
    for k in range(1, n_bounce + 1):
        w = ((gs * gl) ** k if node == "load" else (gs * gl) ** (k - 1)) * gl
        if 64 + 2 * d * k < g.n:
            want[64 + 2 * d * k] = want.get(64 + 2 * d * k, 0.0) + w
    got = {int(i): float(y[i]) for i in np.flatnonzero(np.abs(y) > 1e-15)}
    assert sorted(got) == sorted(want), f"taps at {sorted(got)}, arithmetic says {sorted(want)}"
    for i, v in want.items():
        assert abs(got[i] - v) < 1e-14, f"lag {i-64}: got {got[i]:.12g}, arithmetic {v:.12g}"


@pytest.mark.parametrize("td_ps", [100.0, 55.0, 3.9, 100.5])
@pytest.mark.parametrize("node", ["load", "source"])
def test_a_fractional_echo_lands_at_the_delay_asked_for(td_ps, node):
    """A delay that is not a whole number of samples has to be realised, not rounded away.

    This op used to round to the nearest sample, which made the echo position a staircase: at
    256 GSa/s, 100.0 ps is 25.6 samples, and rounding put it at 26. A fine sweep of the delay
    then did not move the echo at all, and the echo's phase within the eye -- which decides
    whether it lands on a crossing or in the middle -- was quantised. 3.9 ps is 0.998 samples and
    used to round to exactly 1.

    Asserted three ways: the energy centroid sits at the delay asked for, the echo carries the
    amplitude the arithmetic says, and no energy arrives before the echo does.
    """
    g = Grid(fs=256e9, baud=16e9, n=1 << 13)
    gs, gl = 0.30, 0.40
    imp = np.zeros(g.n); imp[1024] = 1.0
    y = P.multi_reflection(imp, grid=g, td_ps=td_ps, gamma_s=gs, gamma_l=gl,
                           n_bounce=1, node=node)
    echo = np.asarray(y, float) - imp
    d = td_ps * 1e-12 * g.fs                           # the fractional delay, in samples
    lag = 2.0 * d

    w = echo ** 2
    centroid = float((np.arange(g.n) * w).sum() / w.sum()) - 1024
    assert centroid == pytest.approx(lag, abs=0.05), (
        f"asked {lag:.4f} samples of round trip, echo centroid at {centroid:.4f}")

    want = ((gs * gl) if node == "load" else 1.0) * gl
    assert float(echo.sum()) == pytest.approx(want, rel=1e-6), "the echo lost amplitude"

    # causality: an echo must not appear before it arrives. The interpolation kernel is symmetric,
    # so it spreads over its own half width and no further.
    from wfmsynth import resample as RS
    before = int(1024 + lag) - RS.HALF_WIDTH - 1
    assert float((echo[:before] ** 2).sum() / w.sum()) < 1e-9, "energy before the echo arrives"


def test_a_cascaded_echo_lands_where_first_order_echoes_says_it_will():
    """`sparam.first_order_echoes` is the arithmetic's answer stated BEFORE the simulation.
    Run the cascade on an impulse and check the realised taps against it, including the
    second-order term whose SIGN comes from the S22 flip on a step seen from the other side.
    """
    g = Grid(fs=256e9, n=1 << 13)
    G1, G2 = 0.30, 0.20
    lossless = dict(causal=False, loss_length_in=0.0)
    path = [{"line": {"td_ps": 281.25, **lossless}}, {"disc": {"gamma": G1}},
            {"line": {"td_ps": 500.00, **lossless}}, {"disc": {"gamma": G2}},
            {"line": {"td_ps": 100.00, **lossless}}]
    imp = np.zeros(g.n); imp[0] = 1.0
    y = SP.cascade_channel(imp, path, grid=g, node="source")
    for e in SP.first_order_echoes(path):
        lag = int(round(e["delay_ps"] * 1e-12 * g.fs))
        assert abs(y[lag] - e["amp"]) < 1e-12, (
            f"echo at {e['delay_ps']} ps: realised {y[lag]:.10f}, predicted {e['amp']:.10f}")
    assert abs(SP.first_order_echoes(path)[1]["amp"] - G2 * (1 - G1 ** 2)) < 1e-15
    # the double bounce inside the 500 ps cavity: out to disc1, twice more around the cavity.
    # Its SIGN is -G1: disc1 seen from the far side is S22 = -S11. Drop that flip and this
    # term comes back positive and a real board's return-loss ripple inverts.
    lag2 = int(round((2 * 281.25 + 4 * 500.0) * 1e-12 * g.fs))
    want2 = -G1 * G2 * (G2 * (1 - G1 ** 2))
    assert abs(y[lag2] - want2) < 1e-12, (
        f"second-order bounce {y[lag2]:.10g} != -G1*G2*amp2 {want2:.10g}")


# ===================================================================== resonances
@pytest.mark.parametrize("length_in,eps_r", [(0.110, 4.0), (0.300, 4.0), (0.110, 3.5), (0.055, 4.3)])
def test_a_quarter_wave_resonator_notches_at_c_over_4L_sqrt_eps(length_in, eps_r):
    """A section of line of length L bounded by discontinuities of OPPOSITE sign is a
    quarter-wave resonator: its through path is nulled at

        f = c / (4 * L * sqrt(eps_r)) * (2m+1),     c = 11.8028 in/ns

    and the same section bounded by SAME-sign discontinuities is a half-wave one, nulled at
    c / (2*L*sqrt(eps_r)) * m. Asserting both is what proves the sign flip on S22 is really
    there -- drop it and the two cases collapse onto one frequency.

    The notch depth is closed form too: |S21|_min = t1*t2 / (1 + |G1*G2|).
    """
    g1, g2 = 0.5, 0.5
    quarter = C_IN_PER_NS / (4.0 * length_in * np.sqrt(eps_r)) * 1e9
    f = np.linspace(0.0, 9.0 * quarter, 300001)
    lossless = dict(eps_r=eps_r, causal=False, loss_length_in=0.0)
    for sign, first in ((-1.0, quarter), (+1.0, 2.0 * quarter)):
        path = [{"disc": {"gamma": g1}},
                {"line": {"length_in": length_in, **lossless}},
                {"disc": {"gamma": sign * g2}}]
        m = np.abs(SP.cascade(path, f).s21)
        mins = f[np.flatnonzero((m[1:-1] < m[:-2]) & (m[1:-1] < m[2:])) + 1]
        want = first + 2.0 * quarter * np.arange(4)
        assert abs(mins[0] - first) < 2 * (f[1] - f[0]), (
            f"gamma sign {sign:+.0f}: first null {mins[0]/1e9:.5f} GHz, "
            f"closed form {first/1e9:.5f} GHz")
        assert np.allclose(mins[:4], want, rtol=0, atol=2 * (f[1] - f[0])), (
            f"nulls at {np.round(mins[:4]/1e9, 4)} GHz, closed form {np.round(want/1e9, 4)}")
        depth = np.sqrt(1 - g1 ** 2) * np.sqrt(1 - g2 ** 2) / (1.0 + g1 * g2)
        assert abs(m[np.argmin(np.abs(f - first))] - depth) < 1e-6


def test_a_quarter_wave_notch_survives_into_the_waveform():
    """The same null, read off a realised record rather than the S-matrix -- the transfer
    function a downstream measurement would actually see."""
    g = Grid(fs=400e9, n=1 << 15)
    L, eps_r = 0.110, 4.0
    quarter = C_IN_PER_NS / (4.0 * L * np.sqrt(eps_r)) * 1e9
    lossless = dict(eps_r=eps_r, causal=False, loss_length_in=0.0)
    path = [{"disc": {"gamma": 0.5}}, {"line": {"length_in": L, **lossless}},
            {"disc": {"gamma": -0.5}}]
    H, f = _transfer(lambda x: SP.cascade_channel(x, path, grid=g, linear=False), g)
    band = (f > 0.5 * quarter) & (f < 1.5 * quarter)
    f_notch = f[band][int(np.argmin(np.abs(H)[band]))]
    assert abs(f_notch - quarter) / quarter < 0.01, (
        f"realised null at {f_notch/1e9:.4f} GHz, closed form {quarter/1e9:.4f} GHz")
    assert abs(np.abs(H)[band].min() - 0.6) < 1e-3           # sqrt(.75)^2/(1+.25) = 0.6


def test_ps_per_inch_is_the_speed_of_light_in_the_dielectric():
    """The length-to-delay conversion the notch frequency above rests on."""
    for eps_r in (1.0, 3.5, 4.0, 4.3):
        assert abs(SP.ps_per_inch(eps_r) - 1000.0 / C_IN_PER_NS * np.sqrt(eps_r)) < 1e-9
    assert abs(SP.ps_per_inch(4.0) - 169.4513) < 1e-3        # the stated 169.45 ps/in


@pytest.mark.parametrize("f0_ghz,q", [(25.0, 12.0), (10.0, 5.0), (40.0, 20.0)])
def test_resonant_gamma_is_the_second_order_bandpass_it_claims(f0_ghz, q):
    """resonant_reflection's Gamma(f) = gamma0*(s/q)/(s^2 + s/q + 1), s = j*f/f0, delayed by
    td. Recover Gamma from the realised waveform and compare to that expression -- magnitude
    AND phase -- then check the two properties that make it a resonance: it peaks at exactly
    f0 with magnitude gamma0, and its -3 dB width is f0/q.
    """
    g = Grid(fs=200e9, n=1 << 14)
    td_ps, gamma0 = 40.0, 0.5
    H, f = _transfer(lambda x: P.resonant_reflection(x, grid=g, td_ps=td_ps, f0_ghz=f0_ghz,
                                                     q=q, gamma0=gamma0, linear=False), g)
    meas = (H - 1.0) * np.exp(1j * 2 * np.pi * f * td_ps * 1e-12)
    s = 1j * (f / (f0_ghz * 1e9))
    want = gamma0 * (s / q) / (s ** 2 + s / q + 1.0)
    assert np.abs(meas - want)[1:-1].max() < 1e-9   # rfft Nyquist bin carries no phase
    k0 = int(np.argmin(np.abs(f - f0_ghz * 1e9)))
    assert int(np.argmax(np.abs(meas))) == k0                      # peaks at f0
    assert abs(np.abs(meas)[k0] - gamma0) < 1e-3                   # peak magnitude is gamma0
    half = np.flatnonzero(np.abs(meas) >= gamma0 / np.sqrt(2.0))
    bw = f[half[-1]] - f[half[0]]
    assert abs(bw / (f0_ghz * 1e9 / q) - 1.0) < 0.02, f"-3dB BW {bw/1e9:.3f} GHz, f0/q {f0_ghz/q:.3f}"


# ===================================================================== clocking
@pytest.mark.parametrize("fs,f_ssc,spread", [(256e9, 32e3, 0.005), (100e9, 33e3, 0.003),
                                             (64e9, 30e3, 0.005)])
def test_ssc_peak_to_peak_tie_equals_its_closed_form(fs, f_ssc, spread):
    """A triangular down-spread of fractional depth `spread` at `f_ssc` integrates to a
    peak-to-peak timing deviation of

        TIE_pp = spread / (8 * f_ssc)                (19.53 ns at 0.5 % / 32 kHz)

    about a mean frequency offset of -spread/2. Both numbers are derived here (the integral of
    a zero-mean triangle over a quarter period is P/16 either side) and neither is read off a
    previous run. 'center' spread has zero mean offset and twice the pp.
    """
    n = int(round(6 * fs / f_ssc))
    k = np.arange(n)
    for profile, mean_offset, pp_scale in (("down", -spread / 2, 1.0), ("up", spread / 2, 1.0),
                                           ("center", 0.0, 2.0)):
        ph = CDR.ssc_phase(n, fs, f_ssc=f_ssc, spread=spread, profile=profile)
        tie = ph - mean_offset * k                          # remove the mean frequency offset
        one = slice(int(2 * fs / f_ssc), int(3 * fs / f_ssc))
        pp_s = float(np.ptp(tie[one])) / fs
        want = pp_scale * spread / (8.0 * f_ssc)
        assert abs(pp_s / want - 1.0) < 1e-6, (
            f"{profile}: pp TIE {pp_s*1e9:.6f} ns, closed form {want*1e9:.6f} ns")
        drift = (ph[-1] - ph[0]) / (n - 1)                  # realised mean fractional offset
        assert abs(drift - mean_offset) < 1e-5 * max(spread, 1e-9) + 1e-9


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("mult", [0.02, 0.1, 0.5, 1.0, 3.33, 10.0])
def test_cdr_jitter_transfer_matches_the_linearized_loop_on_both_sides(order, mult):
    """The residual (what the eye sees) is the loop's high-pass:

        order 1:  |H_e| = |jw / (jw + wn)|
        order 2:  |H_e| = |(jw)^2 / ((jw)^2 + 2*z*wn*jw + wn^2)|

    Measured by driving a single tone through `recover_clock` and reading the residual's
    amplitude at that tone. Both sides are asserted -- rejection below the loop bandwidth AND
    transparency above it -- because a loop that is only nominally present passes one and not
    the other, and a broken loop that simply zeroes its output passes the low side alone.
    """
    baud, bw, z = 16e9, 10e6, 0.707
    n = 1 << 20; half = n // 2
    cyc = max(1, round(mult * bw * half / baud))            # integer cycles in the TAIL
    f = cyc * baud / half
    ph = np.sin(2 * np.pi * f * np.arange(n) / baud)
    clock, res = CDR.recover_clock(ph, baud, bw, order=order, damping=z)
    amp = lambda a: abs(np.fft.rfft(a[half:])[cyc])
    meas = amp(res) / amp(ph)
    w, wn = 2 * np.pi * f, 2 * np.pi * bw
    want = abs(1j * w / (1j * w + wn)) if order == 1 else \
        abs((1j * w) ** 2 / ((1j * w) ** 2 + 2 * z * wn * (1j * w) + wn ** 2))
    assert abs(meas / want - 1.0) < 2e-4, f"f/bw={mult}: measured {meas:.7f}, closed {want:.7f}"
    if mult <= 0.02:
        assert meas < (0.03 if order == 1 else 1e-3)        # rejected well below the loop BW
    if mult >= 3.33:
        assert meas > 0.95                                  # passed well above it


def test_the_recovered_clock_and_the_residual_add_back_up_to_the_input():
    """residual == phase - clock, to the bit. If they do not sum, one of them is not the
    complement of the other and 'what the eye sees' is not what the loop failed to track."""
    baud, bw = 16e9, 5e6
    ph = CDR.ssc_phase(1 << 16, baud, f_ssc=32e3, spread=0.005)
    for order in (1, 2):
        clock, res = CDR.recover_clock(ph, baud, bw, order=order)
        assert np.abs((clock + res) - ph).max() < 1e-9 * np.ptp(ph)


def test_the_order2_clock_transfer_peaks_where_the_loop_says_it_does():
    """The clock path is a low-pass with jitter PEAKING at damping 0.707 -- |H| > 1 near the
    loop bandwidth. A loop faked as a plain low-pass has no peak, so this is the assertion
    that distinguishes a real second-order loop from a first-order one wearing its name."""
    baud, bw, z = 16e9, 10e6, 0.707
    n = 1 << 20; half = n // 2
    peak_meas, peak_want = 0.0, 0.0
    for mult in (0.3, 0.6, 1.0, 1.5, 2.5):
        cyc = max(1, round(mult * bw * half / baud)); f = cyc * baud / half
        ph = np.sin(2 * np.pi * f * np.arange(n) / baud)
        clock, _ = CDR.recover_clock(ph, baud, bw, order=2, damping=z)
        m = abs(np.fft.rfft(clock[half:])[cyc]) / abs(np.fft.rfft(ph[half:])[cyc])
        w, wn = 2 * np.pi * f, 2 * np.pi * bw
        c = abs((2 * z * wn * 1j * w + wn ** 2) / ((1j * w) ** 2 + 2 * z * wn * 1j * w + wn ** 2))
        assert abs(m / c - 1.0) < 2e-4
        peak_meas, peak_want = max(peak_meas, m), max(peak_want, c)
    assert peak_meas > 1.05, f"no jitter peaking (max |H_clk| = {peak_meas:.4f})"
    assert abs(peak_meas - peak_want) < 1e-3


# ===================================================================== filters
@pytest.mark.parametrize("fc_hz", [1e7, 3e7])
@pytest.mark.parametrize("mult", [0.1, 0.5, 1.0, 2.0, 10.0])
def test_the_ac_coupling_corner_in_hz_is_the_corner_measured(fc_hz, mult):
    """`ac_couple(fc_hz=..., grid=...)` is a 1st-order Butterworth high-pass applied
    ZERO-PHASE (forward and back), so its realised magnitude is |H|^2:

        gain(f) = f^2 / (f^2 + fc^2)      -> exactly 0.5 at the corner, not 1/sqrt(2)

    Both halves matter: the exponent proves the double pass, and the corner's position in Hz
    proves the grid conversion. A corner resolved against the wrong Nyquist moves this by the
    ratio of the sample rates.
    """
    g = Grid(fs=20e9, n=1 << 16)
    k, f = _bin(g, mult * fc_hz)
    x = _tone(g, k)
    y = P.ac_couple(x, fc_hz=fc_hz, grid=g)
    s = slice(g.n // 10, -g.n // 10)                        # away from the filtfilt edges
    w = np.hanning(len(x[s]))
    meas = abs(np.fft.rfft(y[s] * w)).max() / abs(np.fft.rfft(x[s] * w)).max()
    r = f / fc_hz
    want = r ** 2 / (1.0 + r ** 2)
    assert abs(meas - want) < 1e-4, f"f/fc={r:.3f}: measured {meas:.6f}, closed form {want:.6f}"


@pytest.mark.parametrize("f_ghz", [0.01, 1.0, 2.0, 4.0, 8.0])
def test_ctle_response_is_its_pole_zero_expression(f_ghz):
    """H(s) = dc_gain*(1 + s/wz) / ((1 + s/wp1)(1 + s/wp2)), measured off the output.

    Also the two properties that make it an equalizer rather than a filter: the DC gain is
    exactly what was asked for, and there is a frequency where the gain EXCEEDS it.
    """
    g = Grid(fs=200e9, n=1 << 16)
    fz, fp1, fp2, dc = 1.0, 4.0, 8.0, 0.5
    k, f = _bin(g, f_ghz * 1e9)
    x = _tone(g, k)
    y = RX.ctle(x, g, fz, fp1, fp2, dc_gain=dc)
    s = slice(g.n // 10, -g.n // 10); w = np.hanning(len(x[s]))
    meas = abs(np.fft.rfft(y[s] * w)).max() / abs(np.fft.rfft(x[s] * w)).max()
    jw = 1j * 2 * np.pi * f
    want = abs(dc * (1 + jw / (2 * np.pi * fz * 1e9)) /
               ((1 + jw / (2 * np.pi * fp1 * 1e9)) * (1 + jw / (2 * np.pi * fp2 * 1e9))))
    assert abs(meas / want - 1.0) < 3e-3, f"{f/1e9:.3f} GHz: measured {meas:.6f}, closed {want:.6f}"
    if f_ghz <= 0.01:
        assert abs(meas - dc) < 2e-4                        # DC gain is the stated one
    if f_ghz >= 4.0:
        assert meas > dc * 1.5                              # it peaks: it is a booster


@pytest.mark.parametrize("db", [2.0, 3.5, 6.0, 9.0])
def test_de_emphasis_db_is_the_measured_transition_to_steady_ratio(db):
    """`de_emphasis_taps(db)` claims 20*log10(V_transition / V_steady) = db. Build the taps,
    run a real pulse through `tx_ffe`, and measure that ratio at the symbol centres."""
    spb = 64
    sym = np.array([-1.0] * 8 + [1.0] * 8 + [-1.0] * 8)
    x = P.from_symbols(sym, n=len(sym) * spb, tr_frac=0.15)
    y = P.tx_ffe(x, P.de_emphasis_taps(db), spb, pre=0)
    centre = lambda ui: y[int((ui + 0.5) * spb)]
    meas = 20.0 * np.log10(centre(8) / centre(15))
    assert abs(meas - db) < 0.01, f"asked {db} dB, measured {meas:.4f} dB"


def test_tx_ffe_puts_its_taps_exactly_one_ui_apart():
    """The FFE is T-spaced by construction: an impulse in must come out as the tap list, one
    UI apart, with a pre-cursor `pre` UI AHEAD of the main tap."""
    spb, taps = 32, [-0.12, 1.0, -0.25]
    imp = np.zeros(16 * spb); imp[8 * spb] = 1.0
    y = P.tx_ffe(imp, taps, spb, pre=1)
    for k, c in enumerate(taps):
        assert abs(y[8 * spb + (k - 1) * spb] - c) < 1e-12, f"tap {k} not at UI {k-1}"
    assert abs(y[8 * spb - spb] - taps[0]) < 1e-12          # the pre-cursor leads the main tap

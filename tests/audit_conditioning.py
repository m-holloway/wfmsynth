"""Numerical conditioning and degenerate inputs — corners, not centres.

WHY THIS FILE ASSERTS CLOSED FORMS AND NOT OUTPUTS. It was written while `instrument.py`,
`compose.py` and `sparam.py` were being edited in another workflow, and one of the defects it
was handed as calibration (`scope_bandwidth`'s doubly-applied Bessel) was FIXED underneath it
mid-audit: measured before, |H| = 0.7039 at 16 GHz for a 32 GHz request; measured after,
0.7071 at 32.000 GHz. A test that had pinned the first number would now be red for the wrong
reason. So every assertion here is a closed form, an invariant, or a self-consistency
property, and the only literal numbers are in `reason=` strings, where they are evidence
rather than an expectation.

Named `audit_*` rather than `test_*` deliberately, so it does not join the `test_*` suites
while those modules are still moving. Run it either way:

    pytest wfmsynth/tests/audit_conditioning.py
    python wfmsynth/tests/audit_conditioning.py        # prints every measured number

THE INSTRUMENT IS GATED BEFORE IT IS BELIEVED. `_floor_db_per_hz` below is this file's own
measuring stick for a stored record's noise floor, and this project has already been burned
by a floor estimator whose two gates both used constructed noise with no signal in them to
leak. The first fixture written here was a full-scale SINE quantised at 11 bits: its floor
read -254 dB/Hz against a closed form of -182, because a single tone's quantisation error is
deterministic harmonics, not white noise, and a median over the band lands between them. The
second was uniform white noise: +65 dB, because a white signal has no stop band to measure.
The fixture that works -- and the one `test_a_stored_records_floor_is_the_closed_form_...`
uses -- is a PRBS13 NRZ carrier through a 10 dB causal channel: wideband enough to dither its
own quantiser, band-limited enough to leave a stop band, and carrying a signal MEASURED 75.2 dB
above the floor it is used to measure, which is the only condition under which recovering
`q**2/12` proves anything. Under it the estimator recovers the closed form to within 0.13 dB
across bits 8..12 and across four record lengths, and recovers the +3.01 dB dither prediction
as +3.06.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

from wfmsynth import acquire as AQ
from wfmsynth import instrument as I
from wfmsynth import measure as MS
from wfmsynth import physics as P
from wfmsynth import rx as RX
from wfmsynth.grid import Grid

FS = 256e9
BAUD = 16e9
SPB = int(FS / BAUD)

# The project's own record shape: repeats x 8191 x 16, and 8191 is a Mersenne prime.
PATTERN_PERIOD = 8191
MERSENNE_RECORD = PATTERN_PERIOD * SPB          # 131,056
PRIME_RECORD = 65521                            # prime
SMOOTH_RECORD = 1 << 16                         # 65,536


def _grid(n, **kw):
    return Grid(fs=FS, baud=BAUD, n=n, v_full=0.8, **kw)


def _wideband(n, amp=0.9):
    """A record with a real stop band and a signal 75 dB above it: PRBS13 NRZ through a
    causal 10 dB channel, peak-normalised. See the module docstring for why the two
    obvious fixtures (a tone; white noise) cannot gate a floor estimator."""
    g = _grid(n)
    x = P.nrz(n_ui=n // SPB, n=n, causal=True, pattern="prbs13")
    x = P.lossy_channel(x, grid=g, loss_db=10.0, loss_at_ghz=8.0, causal=True)
    return amp * x / np.max(np.abs(x))


def _band_psd(y, lo, hi, window=True):
    """Mean one-sided PSD [units^2/Hz] over `lo`..`hi` as a fraction of the sample rate."""
    n = len(y)
    w = np.hanning(n) if window else np.ones(n)
    Y = np.fft.rfft(y * w)
    f = np.fft.rfftfreq(n)
    psd = (np.abs(Y) ** 2) * (2.0 / (FS * np.sum(w ** 2)))
    m = (f >= lo) & (f <= hi)
    return float(np.mean(psd[m]))


def _floor_db_per_hz(y, window=True):
    return 10.0 * np.log10(_band_psd(y, 0.30, 0.45, window))


def _signal_db_per_hz(y):
    return 10.0 * np.log10(_band_psd(y, 0.0, 0.05))


def _is_smooth(v, radix=(2, 3, 5)):
    for p in radix:
        while v % p == 0:
            v //= p
    return v == 1


def _tone_gain(op, f_hz, n):
    """|H| at `f_hz` measured on a bin-aligned tone. Only valid for an op applied
    CIRCULARLY or one with no memory past the record: a tone is an eigenvector of the
    circular product and of nothing else, which is why the loss test asks for
    ``linear=False`` explicitly."""
    k = max(1, round(f_hz * n / FS))
    t = np.sin(2 * np.pi * k * np.arange(n) / n)
    y = op(t)
    return float(abs(np.fft.rfft(y)[k] / np.fft.rfft(t)[k])), k * FS / n


# Diagnostics a test measured on the way to its assertion. Returning them from the test
# function itself is what pytest warns about (and will later refuse), so they are stashed
# here and printed by the direct runner at the bottom of this file.
MEASURED: dict = {}


def _note(value):
    import sys as _sys
    MEASURED.setdefault(_sys._getframe(1).f_code.co_name, []).append(value)


# --------------------------------------------------------------------------------------
#  record-length pathologies
# --------------------------------------------------------------------------------------
def test_the_padded_transform_length_is_minimal_smooth_and_monotone():
    """`next_smooth_length` is the whole defence against the Bluestein cliff, so it is
    checked as a closed statement of its own definition rather than against a table:
    5-smooth, at or above the ask, and minimal (nothing smaller in between is smooth).
    Brute-forced over 1..3000 and spot-checked on the lengths this project actually uses."""
    for k in range(1, 3001):
        m = P.next_smooth_length(k)
        assert m >= k and _is_smooth(m)
        assert not any(_is_smooth(j) for j in range(k, m)), f"{m} is not minimal for {k}"
    for k in (0, 1, PATTERN_PERIOD, MERSENNE_RECORD, PRIME_RECORD, 3 * MERSENNE_RECORD):
        m = P.next_smooth_length(max(k, 1))
        assert _is_smooth(m) and m >= max(k, 1)
        assert m / max(k, 1) < 1.07, f"padding {k} -> {m} costs more than 7 %"
    # monotone, so a longer record never asks for a shorter transform
    prev = 0
    for k in range(1, 2000):
        m = P.next_smooth_length(k)
        assert m >= prev
        prev = m


def test_a_linear_convolution_never_lands_on_a_large_prime_factor():
    """The invariant that keeps a `repeats x 8191 x 16` record off Bluestein: whatever the
    record length and whatever the measured response extent, the transform length handed to
    numpy is 5-smooth."""
    for n in (MERSENNE_RECORD, PRIME_RECORD, SMOOTH_RECORD, 8191, 4096, 1, 2, 3):
        for guard in (1, 2, 17, 1024, 8191, 65521):
            m = P.linear_fft_length(n, guard)
            assert m >= n + guard - 1
            assert _is_smooth(m), f"linear_fft_length({n}, {guard}) = {m} is not 5-smooth"


def test_the_bluestein_cliff_is_real_and_padding_is_what_avoids_it():
    """Not an assertion about wall-clock so much as a check that the cliff the padding
    exists for is still there. MEASURED cold, one rfft each: 131,056 samples (8191 x 16)
    takes 31.7 ms against 5.3 ms padded to 131,072, and a prime 65,521 takes 31.5 ms
    against 1.7 ms padded to 65,536. Warm, inside this test's own loop: 6.6 ms against
    0.60 ms (11.0x) and 3.58 ms against 0.26 ms (13.8x). Either way it is an order of
    magnitude, on a transform 0.01 % larger."""
    timings = {}
    for n in (MERSENNE_RECORD, PRIME_RECORD):
        x = np.random.default_rng(0).standard_normal(n)
        m = P.next_smooth_length(n)
        xp = np.zeros(m)
        xp[:n] = x
        np.fft.rfft(xp[:8])                                    # warm the plan cache
        t0 = time.perf_counter(); np.fft.rfft(x); raw = time.perf_counter() - t0
        t0 = time.perf_counter(); np.fft.rfft(xp); pad = time.perf_counter() - t0
        timings[n] = (f"{raw * 1e3:.2f} ms raw -> {pad * 1e3:.2f} ms padded "
                      f"to {m} ({raw / pad:.1f}x)")
        assert pad < raw, (f"n={n}: padded transform to {m} took {pad * 1e3:.1f} ms against "
                           f"{raw * 1e3:.1f} ms unpadded -- the cliff this padding exists "
                           f"for is gone, so `next_smooth_length` may no longer be earning "
                           f"its keep")
    _note(timings)


def test_the_whole_chain_survives_a_prime_and_a_mersenne_record_length():
    """A record whose length is prime, and one that is `8191 x 16`, through source ->
    channel -> resonant reflection -> front end -> store. Nothing may come back short,
    non-finite, or collapsed onto a single code."""
    out = {}
    for n in (PRIME_RECORD, MERSENNE_RECORD):
        g = _grid(n)
        y = P.nrz(n_ui=n // SPB, n=n, causal=True)
        y = P.lossy_channel(y, grid=g, loss_db=10.0, loss_at_ghz=8.0, causal=True)
        y = P.resonant_reflection(y, grid=g, td_ps=40.0, f0_ghz=10.0, q=9.0)
        y = I.scope_bandwidth(y, g, 32e9, kind="bessel")
        y = I.store_record(y, bits=11)
        assert len(y) == n and np.all(np.isfinite(y))
        codes = len(np.unique(y))
        assert codes > 64, f"n={n}: {codes} distinct codes is a collapsed record"
        out[n] = f"{codes} codes"
    _note(out)


# --------------------------------------------------------------------------------------
#  degenerate lengths
# --------------------------------------------------------------------------------------
_DEGENERATE_LENGTHS = (1, 2, 3, 4, 8, 15, 16, 17, 31, 127)


def _stages(g):
    """Waveform -> waveform stages whose length behaviour is a property of this file's
    remit. Sources and multi-signal stages are out of scope, as are the two zero-phase
    IIR legacy paths (`scope_bandwidth(causal=False)`, `inject_jitter(sigma_rj>0)`), which
    refuse a record shorter than their own `sosfiltfilt` padding -- reported, not asserted,
    because a refusal is not a fabrication."""
    return {
        "lossy causal": lambda x: P.lossy_channel(x, grid=g, loss_db=6.0, loss_at_ghz=8.0,
                                                  causal=True),
        "lossy zero-phase": lambda x: P.lossy_channel(x, grid=g, loss_db=6.0,
                                                      loss_at_ghz=8.0),
        "resonant_reflection": lambda x: P.resonant_reflection(x, grid=g, td_ps=40.0,
                                                               f0_ghz=10.0, q=9.0),
        "multi_reflection": lambda x: P.multi_reflection(x, td_samples=3, gamma_s=0.3,
                                                         gamma_l=0.4),
        "tx_ffe": lambda x: P.tx_ffe(x, [-0.1, 0.75, -0.15], SPB, pre=1),
        "nominal_nonlinearity": lambda x: P.nominal_nonlinearity(x, compression=0.04),
        "supply_coupling": lambda x: P.supply_coupling(x, g, f_ripple_hz=2.5e6,
                                                       am_depth=0.02, psij_ps=1.0),
        "scope bessel": lambda x: I.scope_bandwidth(x, g, 32e9, kind="bessel"),
        "scope brickwall": lambda x: I.scope_bandwidth(x, g, 32e9, kind="brickwall"),
        "scope gaussian": lambda x: I.scope_bandwidth(x, g, 32e9, kind="gaussian"),
        "probe_loading causal": lambda x: I.probe_loading(x, g, causal=True),
        "probe": lambda x: I.probe(x, g, noise_rms=1e-3, bw_hz=20e9,
                                   rng=np.random.default_rng(0)),
        "timebase_jitter": lambda x: I.timebase_jitter(x, g, rms_ps=0.5,
                                                       rng=np.random.default_rng(0)),
        "interleave_adc": lambda x: I.interleave_adc(x, m_cores=4, gain_mm=0.01,
                                                     offset_v=1e-3, skew_mm=0.02,
                                                     rng=np.random.default_rng(0)),
        "quantize_adc": lambda x: I.quantize_adc(x, bits=11, full_scale=1.0),
        "clip_adc": lambda x: I.clip_adc(x, 0.9)[0],
        "store_record": lambda x: I.store_record(x, bits=11),
        "ctle": lambda x: RX.ctle(x, g, 2.0, 8.0, 16.0),
        "ac_couple": lambda x: P.ac_couple(x, fc_hz=2e9, grid=g),
    }


@pytest.mark.parametrize("n", _DEGENERATE_LENGTHS)
def test_a_stage_that_returns_at_all_returns_the_length_it_was_given(n):
    """One sample, one UI, a single symbol, an odd length, a length shorter than the
    filter's own impulse response. A stage may refuse such a record -- that is a defensible
    answer and `ac_couple` (padlen 6) and the zero-phase legacy paths do it -- but a stage
    that ANSWERS must answer with `len(x)` finite samples. Silently returning 3 samples, or
    a NaN, is how a short record becomes fabricated data downstream."""
    g = _grid(n)
    x = np.cos(2 * np.pi * 0.05 * np.arange(n))
    refused = []
    for name, op in _stages(g).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                y = np.asarray(op(x), float)
            except Exception as exc:                       # a refusal, not a fabrication
                refused.append(f"{name}:{type(exc).__name__}")
                continue
        assert y.shape == (n,), f"n={n} {name}: returned {y.shape}, asked for ({n},)"
        assert np.all(np.isfinite(y)), f"n={n} {name}: non-finite output"
    _note(refused)


def test_a_zero_amplitude_carrier_stays_finite_and_keeps_its_length():
    """The all-zero record: a quiescent line, a muted transmitter, the head of a record
    before a lead-in is filled. Every amplitude stage has a divide-by-span or a
    range-to-peak in it, and an auto-ranging store has no range at all to set."""
    n = 4096
    g = _grid(n)
    z = np.zeros(n)
    for name, op in _stages(g).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = np.asarray(op(z), float)
        assert y.shape == (n,) and np.all(np.isfinite(y)), f"{name} on an all-zero record"
    y, info = I.digitize(z, bits=11, clip_full_scale=1.0, rng=np.random.default_rng(0))
    assert len(y) == n and np.all(np.isfinite(y))
    assert all(np.isfinite(v) for v in info.values() if isinstance(v, float))


def test_a_tap_list_of_length_one_is_the_identity():
    """The degenerate FFE. A single unit tap at zero delay must be exactly the identity --
    no normalisation, no half-sample interpolation, no edge invention -- in both the
    transmit-side and receive-side implementations, at every record length."""
    for n in (1, 2, 3, 16, 4096):
        x = np.cos(2 * np.pi * 0.05 * np.arange(n))
        assert np.max(np.abs(RX.ffe(x, [1.0], SPB, pre=0) - x)) == 0.0, f"rx.ffe n={n}"
        assert np.max(np.abs(P.tx_ffe(x, [1.0], SPB, pre=0) - x)) < 1e-12, f"tx_ffe n={n}"


# --------------------------------------------------------------------------------------
#  extreme but legal parameters
# --------------------------------------------------------------------------------------
def test_extreme_but_legal_loss_is_realised_exactly_at_its_anchor():
    """`loss_db`+`loss_at_ghz` claims IL(loss_at_ghz) == loss_db EXACTLY, so it is checked
    at the end of its range (40 dB, and 60 dB past it) rather than in the middle. Measured
    on a bin-aligned tone through the circular form, which is the only form for which a
    tone is an eigenvector: 0.000 / 6.000 / 20.000 / 40.000 / 60.000 dB.

    Also the two passivity clamps, which are the guards at the ends of `trend`'s range: a
    fit with a positive constant would deliver GAIN at DC, and an unbounded roll-off pushes
    the minimum-phase cepstrum through its own guard term."""
    n = SMOOTH_RECORD
    got = {}
    for asked in (0.0, 6.0, 20.0, 40.0, 60.0):
        for causal in (True, False):
            gain, f_at = _tone_gain(
                lambda t, a=asked, c=causal: P.lossy_channel(
                    t, grid=_grid(n), loss_db=a, loss_at_ghz=8.0, causal=c, linear=False),
                8e9, n)
            realised = -20.0 * np.log10(gain)
            got[(asked, causal)] = realised
            assert abs(realised - asked) < 0.05, (
                f"asked {asked} dB at {f_at / 1e9:.3f} GHz, realised {realised:.3f} dB "
                f"(causal={causal})")
    assert float(P.insertion_loss_db(np.array([0.0]), trend=(-1.0, -0.1, 3.0))[0]) == 0.0, \
        "a trend with a positive constant must be clamped passive at DC, not amplify"
    assert float(P.insertion_loss_db(np.array([128.0]), trend=(-3.0, -0.5, 0.0),
                                     trend_floor_db=80.0)[0]) == 80.0
    _note(got)


def test_a_reflection_coefficient_near_one_gives_the_lattice_closed_form():
    """|Gamma| -> 1 at both ends is the top of the range and the one place a bounce sum can
    run away. The lattice weights are a closed form -- the k-th echo at the load carries
    (gamma_s*gamma_l)^k * gamma_l -- so they are checked against it on an impulse rather
    than against a stored waveform. Exact to 0.0 at gamma = 0.9, 0.99 and 0.999."""
    n = 4096
    x = np.zeros(n)
    x[0] = 1.0
    for gam in (0.9, 0.99, 0.999):
        y = P.multi_reflection(x, td_samples=100, gamma_s=gam, gamma_l=gam, n_bounce=6)
        for k in range(1, 7):
            expect = (gam * gam) ** k * gam
            assert abs(y[200 * k] - expect) < 1e-12, f"gamma={gam} echo {k}"
        assert np.all(np.isfinite(y))


def test_a_reflection_at_zero_delay_is_the_lattice_sum_as_a_pure_gain():
    """The degenerate reflection: every echo lands on the incident wave. The answer is not
    NaN and not a truncated record, it is the geometric sum of the lattice weights as a
    scalar gain, and it is checked as one. MEASURED at gamma_s=0.3, gamma_l=0.4, 6 bounces:
    1.054545291674 at the load and 1.454544097280 at the source, both recovered to 4.4e-16.

    That closed form is also what makes the sub-sample-delay defect below legible: a
    reflection whose delay rounds to zero is not an echo at all, it is a gain error."""
    n = 4096
    x = P.nrz(n_ui=256, n=n, causal=True)
    for node, offset in (("load", 0), ("source", 1)):
        y = P.multi_reflection(x, td_samples=0, gamma_s=0.3, gamma_l=0.4, n_bounce=6,
                               node=node)
        k = np.arange(1, 7)
        expect = 1.0 + 0.4 * np.sum((0.3 * 0.4) ** (k - offset))
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(y - expect * x)) < 1e-12, f"node={node}"


def test_a_resonance_peaks_at_gamma0_for_q_from_a_half_to_a_hundred():
    """Q at both ends of its range (0.5, and 100 -- the top of what a via stub does) and f0
    from 1 GHz to Nyquist. The invariant: the second-order band-pass Gamma(f) has
    |Gamma| = gamma0 at f0 for EVERY Q -- a high Q makes the peak narrow, not taller -- so
    the realised transfer at f0 is the closed form |1 + gamma0*exp(-j*2*pi*f0*td)|,
    whatever Q is. Measured on bin-aligned tones through the circular form (a tone is an
    eigenvector of no other form), gamma0=0.9, td=40 ps: 1.885059599 at f0=1 GHz and
    0.594785180 at f0=10 GHz, recovered to 4.0e-14 in 6 of 6 (Q, f0) combinations.

    The time-domain peak is bounded by the L1 norm of the impulse response, not by
    max|H| -- measured L1 = 2.146 at Q=100 against max|1+Gamma| = 1.9. An earlier version
    of this test asserted the max|H| bound and failed for that reason; the bound below is
    the correct one."""
    n = 4096
    g = _grid(n)
    x = P.nrz(n_ui=256, n=n, causal=True)
    for q in (0.5, 10.0, 100.0):
        for f0_ghz in (1.0, 10.0):
            k = round(f0_ghz * 1e9 * n / FS)           # exactly on a bin
            t = np.sin(2 * np.pi * k * np.arange(n) / n)
            y = P.resonant_reflection(t, grid=g, td_ps=40.0, f0_ghz=f0_ghz, q=q,
                                      gamma0=0.9, linear=False)
            meas = abs(np.fft.rfft(y)[k] / np.fft.rfft(t)[k])
            closed = abs(1.0 + 0.9 * np.exp(-1j * 2 * np.pi * f0_ghz * 1e9 * 40e-12))
            assert abs(meas - closed) < 1e-9, (f"q={q} f0={f0_ghz}: |H(f0)| {meas:.9f} "
                                               f"against closed form {closed:.9f}")
    imp = np.zeros(n)
    imp[0] = 1.0
    for q in (0.5, 10.0, 100.0):
        for f0_ghz in (1.0, 10.0, 128.0):              # 128 GHz is this grid's Nyquist
            h = P.resonant_reflection(imp, grid=g, td_ps=40.0, f0_ghz=f0_ghz, q=q,
                                      gamma0=0.9, linear=False)
            y = P.resonant_reflection(x, grid=g, td_ps=40.0, f0_ghz=f0_ghz, q=q, gamma0=0.9)
            assert np.all(np.isfinite(y)) and len(y) == n
            assert np.max(np.abs(y)) <= np.sum(np.abs(h)) * np.max(np.abs(x)) + 1e-9, \
                f"q={q} f0={f0_ghz}: output exceeds the L1 bound of its own response"


def test_an_enob_above_the_bit_depth_is_refused_at_the_closed_form_boundary():
    """The refusal this unit was told to verify, verified as a closed form rather than as a
    message. A converter's own lattice contributes q**2/12 of the wideband noise power, and
    the instrument's filter passes bandwidth/nyquist of it, so the highest ENOB a `bits`-bit
    part can be asked for at bandwidth B is

        enob_max = bits + 0.5*log2(nyquist/B)

    -- oversampling gain, which is why a 10-bit part legitimately reaches ENOB 12 at 1 GHz
    of a 128 GHz Nyquist and cannot reach 10.000001 at 128 GHz. Bisected against the actual
    refusal: closed form 10.109320143 vs measured 10.109320143 (err 1.8e-15) for
    bits=10 at 110 GHz; exact at both other settings tried."""
    for bits, bw, nyq in [(10, 110e9, 128e9), (10, 128e9, 128e9), (8, 1e9, 128e9)]:
        emax = bits + 0.5 * np.log2(nyq / bw)

        def ok(e):
            try:
                I.converter_noise_rms(e, 0.423, bw, nyq, bits=bits)
                return True
            except ValueError:
                return False

        assert ok(emax - 0.01) and not ok(emax + 0.01)
        lo, hi = emax - 1.0, emax + 1.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        assert abs(lo - emax) < 1e-6, f"bits={bits} bw={bw:g}: boundary {lo} vs {emax}"


def test_the_converter_lattice_and_its_added_noise_recompose_to_the_published_enob():
    """The two-mechanism converter has to add back up, or the split is bookkeeping. The
    added noise and the lattice, taken through the same in-band filter, must equal the rms
    the ENOB figure names -- measured 7.631849e-03 V against 7.631849e-03 V."""
    for enob, bits, bw, nyq, fs_v in [(5.0, 10, 110e9, 128e9, 0.423),
                                      (6.0, 11, 40e9, 128e9, 0.4),
                                      (7.5, 12, 10e9, 128e9, 1.0)]:
        added = I.converter_noise_rms(enob, fs_v, bw, nyq, bits=bits)
        q = 2.0 * fs_v / 2 ** bits
        in_band = np.sqrt((added ** 2 + q ** 2 / 12.0) * bw / nyq)
        assert abs(in_band / I.sinad_noise_rms(enob, fs_v) - 1.0) < 1e-12


def test_a_bandwidth_and_a_corner_at_nyquist_stay_bounded():
    """A filter at Nyquist: the top of every bandwidth knob's range. The response must stay
    bounded and must not annihilate the record -- a band limit at Nyquist is nearly a
    no-op, not a mute."""
    n = 8192
    g = Grid(fs=FS, n=n)
    x = _wideband(n)
    for kind in ("bessel", "gaussian", "brickwall"):
        y = I.scope_bandwidth(x, g, g.f_nyquist, kind=kind)
        assert np.all(np.isfinite(y))
        assert 0.5 < np.std(y) / np.std(x) < 1.5, f"kind={kind} at Nyquist"
    for fc_frac in (0.5, 0.9, 1.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = P.ac_couple(x, fc_frac=fc_frac)
        assert np.all(np.isfinite(y)) and len(y) == n


def test_scope_bandwidth_realises_the_corner_it_was_asked_for_in_every_kind():
    """Every kind, not just the default one, and by bisecting the realised -3 dB point
    rather than by reading the design back. Measured for a 32 GHz ask: bessel 31.984 GHz,
    gaussian 31.984, brickwall 32.016 -- 1.000x in all three."""
    n = SMOOTH_RECORD
    g = Grid(fs=FS, n=n)
    realised = {}
    for kind in ("bessel", "gaussian", "brickwall"):
        lo, hi = 32e9 / 60.0, min(0.45 * FS, 32e9 * 60.0)
        for _ in range(40):
            mid = np.sqrt(lo * hi)
            gain, _f = _tone_gain(
                lambda t, k=kind: I.scope_bandwidth(t, g, 32e9, kind=k), mid, n)
            lo, hi = (lo, mid) if gain < 0.70710678 else (mid, hi)
        realised[kind] = np.sqrt(lo * hi)
        assert 0.5 < realised[kind] / 32e9 < 2.0, (
            f"kind={kind}: asked 32 GHz, realised {realised[kind] / 1e9:.3f} GHz")
    _note(realised)


def test_probe_loading_is_the_closed_form_pole_in_magnitude_and_group_delay():
    """The causal probe claims to BE the single pole 1/(1 + jf/fc), magnitude and phase.
    Both are closed forms, so both are checked: 10*log10(1 + (f/fc)**2) dB, and a DC group
    delay of 1/(2*pi*fc). Measured at fc = 6.366 GHz: -3.011 dB against -3.011 closed, and
    24.999 ps against 25.000 ps closed.

    The measurement itself needed gating: probed with a delta at sample 0 instead of a tone,
    the zero-phase legacy path reads +13.8 dB of GAIN, which is `sosfiltfilt`'s odd padding
    reflecting about x[0], not the filter. Tones, therefore, not impulses at the edge."""
    n = SMOOTH_RECORD
    g = Grid(fs=FS, n=n)
    fc = I.rc_pole_hz(50.0, 0.5e-12)
    for mult in (1.0, 2.0):
        gain, f_at = _tone_gain(lambda t: I.probe_loading(t, g, causal=True), mult * fc, n)
        closed = 1.0 / np.sqrt(1.0 + (f_at / fc) ** 2)
        assert abs(20 * np.log10(gain / closed)) < 0.02, f"|H| at {mult}*fc"
    imp = np.zeros(n)
    imp[n // 2] = 1.0                                  # centred, so no edge artefact
    h = I.probe_loading(imp, g, causal=True)
    H = np.fft.rfft(np.roll(h, -n // 2))
    f = np.fft.rfftfreq(n, d=g.dt)
    ph = np.unwrap(np.angle(H))
    gd = -np.gradient(ph, 2 * np.pi * (f[1] - f[0]))
    assert abs(gd[1] - 1.0 / (2 * np.pi * fc)) < 1e-13, (
        f"DC group delay {gd[1] * 1e12:.3f} ps against the pole's "
        f"{1e12 / (2 * np.pi * fc):.3f} ps")


# --------------------------------------------------------------------------------------
#  scale and units honesty
# --------------------------------------------------------------------------------------
def _linear_ops(g, x):
    return {
        "lossy causal": lambda v: P.lossy_channel(v, grid=g, loss_db=6.0, loss_at_ghz=8.0,
                                                  causal=True),
        "resonant_reflection": lambda v: P.resonant_reflection(v, grid=g, td_ps=40.0,
                                                               f0_ghz=10.0, q=9.0),
        "multi_reflection": lambda v: P.multi_reflection(v, td_samples=13, gamma_s=0.3,
                                                         gamma_l=0.4),
        "ac_couple": lambda v: P.ac_couple(v, fc_hz=2e9, grid=g),
        "tx_ffe": lambda v: P.tx_ffe(v, [-0.1, 0.75, -0.15], SPB, pre=1),
        "de_emphasis": lambda v: P.tx_ffe(v, P.de_emphasis_taps(3.0), SPB, pre=0),
        "rx_ffe": lambda v: RX.ffe(v, [-0.05, 1.0, -0.18], SPB, pre=1),
        "ctle": lambda v: RX.ctle(v, g, 2.0, 8.0, 16.0),
        "scope bessel": lambda v: I.scope_bandwidth(v, g, 32e9, kind="bessel"),
        "scope brickwall": lambda v: I.scope_bandwidth(v, g, 32e9, kind="brickwall"),
        "scope gaussian": lambda v: I.scope_bandwidth(v, g, 32e9, kind="gaussian"),
        "probe_loading causal": lambda v: I.probe_loading(v, g, causal=True),
        "supply_coupling am": lambda v: P.supply_coupling(v, g, f_ripple_hz=2.5e6,
                                                          am_depth=0.02),
    }


def test_a_linear_stage_on_k_times_the_input_gives_k_times_the_output():
    """Scale equivariance, over four decades of amplitude, for every stage that claims to be
    linear. It is the half of the units question that must hold: the kernel works in
    normalized +/-1 and a caller may hand it volts (`Grid.volts(x) = x*0.5*v_full`), and a
    linear stage cannot care which. MEASURED: 0.0e+00 at k=2 and at most 2.3e-15 of output
    rms at k=1e-3, in 13 of 13 stages."""
    n = 4096
    g = _grid(n)
    x = P.nrz(n_ui=256, n=n, causal=True)
    worst = {}
    for name, op in _linear_ops(g, x).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = np.asarray(op(x), float)
            for k in (2.0, 1e-3, 1e3):
                yk = np.asarray(op(k * x), float)
                rel = np.max(np.abs(yk - k * y)) / (k * np.std(y))
                worst[name] = max(worst.get(name, 0.0), rel)
                assert rel < 1e-12, f"{name} at k={k}: {rel:.3e} of output rms"
    _note(worst)


def test_the_stages_that_must_not_be_scale_equivariant_are_not():
    """The other half, and the one a units bug hides in: a compressor, a clipper, a
    FIXED-lattice quantiser, a converter offset quoted in volts and a decision-feedback
    equaliser all have an absolute amplitude built into them, so scaling the input MUST
    change the answer by more than the scale factor. A stage that passes the linearity test
    above and appears here is a stage whose absolute reference has gone missing.

    MEASURED deviations from k*y at k=2: nominal_nonlinearity 4.8e-2, clip_adc 5.2e-1,
    quantize_adc(full_scale=1.0) 1.6e-2, store_record(full_scale=1.0) 5.3e-1,
    interleave(offset_v) 4.7e-4, dfe 1.1e-1 of output rms.

    NOTE the two that are DELIBERATELY equivariant and are therefore not asserted here:
    `store_record(full_scale=None)` ranges to the acquisition (measured 0.0e+00 -- an
    operator's choice, documented) and `quantize_adc(full_scale=None)` does the same
    (5.2e-13), which means the converter's own lattice pitch follows its input. A 60 dB
    attenuated signal quantised that way keeps exactly its quantisation SNR."""
    n = 4096
    g = _grid(n)
    x = P.nrz(n_ui=256, n=n, causal=True)
    nonlinear = {
        "nominal_nonlinearity": lambda v: P.nominal_nonlinearity(v, compression=0.04),
        "clip_adc": lambda v: I.clip_adc(v, 0.9)[0],
        "quantize_adc fixed": lambda v: I.quantize_adc(v, bits=6, full_scale=1.0),
        "store_record fixed": lambda v: I.store_record(v, bits=6, full_scale=1.0),
        "interleave offset_v": lambda v: I.interleave_adc(v, m_cores=4, offset_v=1e-3,
                                                          rng=np.random.default_rng(1)),
        "dfe": lambda v: RX.dfe(v, [0.12, 0.05], levels=[-1.0, 1.0])[0],
    }
    got = {}
    for name, op in nonlinear.items():
        y = np.asarray(op(x), float)
        yk = np.asarray(op(2.0 * x), float)
        rel = np.max(np.abs(yk - 2.0 * y)) / (2.0 * np.std(y))
        got[name] = rel
        assert rel > 1e-4, (f"{name}: scaling the input by 2 scaled the output by exactly 2 "
                            f"({rel:.3e}). Its absolute amplitude reference is gone.")
    _note(got)


def test_a_converter_offset_in_volts_stays_put_when_the_signal_shrinks():
    """`interleave_adc` documents the distinction that matters most when it matters at all:
    `offset_v` is the converter's own offset (absolute -- it does not shrink with the
    signal, which is exactly when it dominates) and `offset_mm` is a fraction of the
    signal's span. Both are checked against that claim rather than against a stored value.
    MEASURED, signal scaled 1.0 -> 0.1: offset_v rms ratio 1.0000, offset_mm 0.1000."""
    n = 4096
    x = P.nrz(n_ui=256, n=n, causal=True)
    def added(kw, amp):
        y = I.interleave_adc(amp * x, m_cores=4, rng=np.random.default_rng(3), **kw)
        return float(np.std(y - amp * x))
    rv = added(dict(offset_v=1e-3), 0.1) / added(dict(offset_v=1e-3), 1.0)
    rm = added(dict(offset_mm=1e-3), 0.1) / added(dict(offset_mm=1e-3), 1.0)
    assert abs(rv - 1.0) < 1e-6, f"offset_v scaled with the signal (ratio {rv:.4f})"
    assert abs(rm - 0.1) < 1e-6, f"offset_mm did not scale with the span (ratio {rm:.4f})"
    _note({"offset_v ratio": rv, "offset_mm ratio": rm})


def test_grid_volts_is_the_only_place_the_two_scales_meet():
    """`Grid.volts(x) = x*0.5*v_full` is the whole conversion, and a chain run in volts must
    be the chain run normalized, scaled -- for the linear part. Asserted as a round trip so
    that a change to either the conversion or a stage's internal reference breaks it."""
    n = 4096
    g = _grid(n)
    assert g.volts(1.0) == 0.5 * g.v_full
    assert g.volts(np.array([-1.0, 0.0, 1.0])).tolist() == [-0.4, 0.0, 0.4]
    x = P.nrz(n_ui=256, n=n, causal=True)
    y_norm = P.lossy_channel(x, grid=g, loss_db=6.0, loss_at_ghz=8.0, causal=True)
    y_volt = P.lossy_channel(g.volts(x), grid=g, loss_db=6.0, loss_at_ghz=8.0, causal=True)
    assert np.max(np.abs(y_volt - g.volts(y_norm))) < 1e-15


def test_a_stored_records_floor_is_its_lattices_closed_form_at_every_record_length():
    """THE ground-truth gate for this file's own instrument, and a conditioning claim in its
    own right: a stored record's stop-band floor is `q**2/12/(fs/2)`, and that must not
    depend on the record's length or its factorisation.

    MEASURED against the closed form, windowed, on the same physics at four lengths:
    65,536 -0.03 dB, 65,521 (prime) -0.03, 65,528 (8191x8) -0.04, 131,056 (8191x16) +0.01.
    Across bit depths 8..12 at n=65,536: +0.13, +0.06, -0.03, -0.03, -0.02 dB.

    The UNWINDOWED estimator, on those same four records, reads +11.83, +0.79, +0.77 and
    +13.50 dB -- so an unwindowed floor claim is not merely biased, its bias is a function
    of the record length, by 12.7 dB between two lengths 0.008 % apart in this set. That is
    the same class of error as the +3 dB this project already recorded, measured here on
    records that HAVE a signal to leak (75.2 dB above the floor)."""
    out = {}
    for n in (SMOOTH_RECORD, PRIME_RECORD, PATTERN_PERIOD * 8, MERSENNE_RECORD):
        y = I.store_record(_wideband(n), bits=11, full_scale=1.0)
        closed = I.quantisation_floor_db_per_hz(2.0 / 2 ** 11, FS)
        win, raw = _floor_db_per_hz(y), _floor_db_per_hz(y, window=False)
        head = _signal_db_per_hz(y) - win
        out[n] = (f"windowed {win - closed:+.2f} dB, unwindowed "
                  f"{raw - closed:+.2f} dB, signal {head:.1f} dB up")
        assert head > 60.0, (f"n={n}: the fixture's signal is only {head:.1f} dB above the "
                             f"floor being measured -- it cannot gate a leakage error")
        assert abs(win - closed) < 0.3, (f"n={n}: floor {win:.2f} dB/Hz against closed form "
                                         f"{closed:.2f}")
    n = SMOOTH_RECORD
    x = _wideband(n)
    for bits in (8, 9, 10, 11, 12):
        y = I.store_record(x, bits=bits, full_scale=1.0)
        closed = I.quantisation_floor_db_per_hz(2.0 / 2 ** bits, FS)
        assert abs(_floor_db_per_hz(y) - closed) < 0.3, f"bits={bits}"
    _note(out)


def test_the_stored_floor_follows_the_vertical_and_the_dither_closed_forms():
    """Two closed forms on top of the gated instrument. Halving the vertical halves the LSB,
    so the floor drops 6.02 dB (measured -0.03 dB of the closed form at k = 1, 1/2, 1/4,
    1/8 -- the SAME error at every scale, which is what makes it a lattice and not a fit).
    And `dither_lsb = 1/sqrt(12)` adds one more `q**2/12`, i.e. +3.01 dB: measured +3.06."""
    n = SMOOTH_RECORD
    x = _wideband(n)
    for k in (1.0, 0.5, 0.25, 0.125):
        y = I.store_record(k * x, bits=11, full_scale=k)
        closed = I.quantisation_floor_db_per_hz(2.0 * k / 2 ** 11, FS)
        assert abs(_floor_db_per_hz(y) - closed) < 0.3, f"k={k}"
    bare = I.store_record(x, bits=11, full_scale=1.0)
    dith = I.store_record(x, bits=11, full_scale=1.0, dither_lsb=1 / np.sqrt(12),
                          rng=np.random.default_rng(7))
    delta = _floor_db_per_hz(dith) - _floor_db_per_hz(bare)
    assert abs(delta - 3.0103) < 0.25, f"dither raised the floor by {delta:+.2f} dB"
    _note(delta)


# --------------------------------------------------------------------------------------
#  MEASURED DEFECTS — strict xfail. Each fails today and turns red when it is fixed.
# --------------------------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `physics.resolve_rise_time(tr_frac=0.0, spb)` raises "
    "ZeroDivisionError -- from inside the warning message that exists to make the clamp "
    "visible ('{floor/requested:.1f}x' with requested == 0.0). tr_frac=0 is the DEGENERATE "
    "END of the parameter's range, an infinitely fast edge, and it is the one request the "
    "clamp is guaranteed to have to handle: measured, tr_frac=1e-9 returns (2.0, True) and "
    "warns, tr_frac=0.0 raises. It propagates straight to the sources -- "
    "`physics.nrz(n_ui=64, n=1024, tr_frac=0.0, causal=True)` raises the same "
    "ZeroDivisionError, as do `pam4` and `from_symbols`. Fix: compute the ratio only when "
    "requested > 0, or report the clamp as '0.000 -> 2.000 samples'."))
def test_a_zero_rise_time_request_is_clamped_not_a_crash():
    tr, clamped = P.resolve_rise_time(0.0, 16.0)
    assert clamped and tr == pytest.approx(P.TR_DEFAULT_FLOOR_SAMPLES)
    y = P.nrz(n_ui=64, n=1024, tr_frac=0.0, causal=True)
    assert np.all(np.isfinite(y))


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `physics.resonant_reflection` returns an ALL-NaN record for two "
    "parameter values at the ends of its own range, with no refusal and no warning of its "
    "own. Measured on a 4096-sample record, grid fs=256 GHz, td_ps=40: q=0.0 -> 4096 of "
    "4096 samples NaN; f0_ghz=0.0 -> 4096 of 4096 NaN (and f0_frac=0.0 on the "
    "record-fraction path likewise). Both come from `sv = 1j*(f/f0)` and `(sv/q)` -- "
    "numpy emits 'divide by zero' and 'invalid value' RuntimeWarnings and the op returns "
    "them as data. Q in 0.5..100 and f0 in 1..128 GHz are all clean (max|Gamma| = gamma0 "
    "in 9 of 9), so this is purely the degenerate end. A NaN record propagates silently: "
    "every downstream stage here is elementwise or an FFT, so one NaN becomes the whole "
    "record. Fix: refuse q <= 0 and f0 <= 0 with a ValueError naming them."))
def test_a_degenerate_resonance_refuses_rather_than_returning_nan():
    n = 4096
    g = _grid(n)
    x = P.nrz(n_ui=256, n=n, causal=True)
    for kw in (dict(q=0.0, f0_ghz=10.0), dict(q=9.0, f0_ghz=0.0)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                y = P.resonant_reflection(x, grid=g, td_ps=40.0, **kw)
            except ValueError:
                continue                              # a refusal is the right answer
        assert np.all(np.isfinite(y)), f"{kw}: {int(np.sum(~np.isfinite(y)))} non-finite"


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `rx.ffe` raises ValueError (a numpy broadcast mismatch) for a whole "
    "WINDOW of record lengths -- not the short ones, the middling ones. In `y[d:] += "
    "c*x[:n-d]`, a tap delay d > n makes `n-d` NEGATIVE, so `x[:n-d]` is a slice from the "
    "FRONT holding max(0, 2n-d) samples while `y[d:]` holds none: the two shapes disagree "
    "exactly when d/2 < n < d. Measured with taps=[-0.05,1.0,-0.18], spacing=16, pre=1 "
    "(d_max=16): raises for len(x) in 9..15, works at 1..8 and at 16+. With taps=[0.1]*6, "
    "spacing=16, pre=0 (d_max=80): raises for len(x) in 9..79. So a T-spaced receive FFE "
    "with a realistic post-cursor span cannot equalise any record shorter than its own "
    "span, and fails LOUDLY in the middle of a range where the two ends both work -- the "
    "signature of index arithmetic, not of a physical limit. Fix: `x[:max(0, n-d)]`, or "
    "skip taps with d >= n (their contribution is entirely past the record)."))
def test_a_receive_ffe_equalises_a_record_shorter_than_its_own_tap_span():
    x = np.cos(2 * np.pi * 0.05 * np.arange(128))
    for m in range(1, 97):
        y = RX.ffe(x[:m], [0.1] * 6, SPB, pre=0)
        assert len(y) == m and np.all(np.isfinite(y)), f"len {m}"


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `measure.sample_at_phase` returns ONE FEWER symbol than the record "
    "holds, always. `n_sym = int((len(x)-1-phase)/spb)` is a count of INTERVALS where the "
    "count of sampling INSTANTS is one more; the trailing `idx < len(x)` filter already "
    "makes the extra index safe. Measured, spb=16, phase=0: n=4096 returns 255 of 256 "
    "available instants; n=48 returns 2 of 3; n=32 returns 1 of 2; and n=16 -- a ONE-UI "
    "record -- returns 0 of 1, i.e. an empty array with no error. Non-integer spb is the "
    "same: spb=4.82 on 4096 samples returns 849 of 850. At deep-record lengths it is a "
    "0.4 % loss, at the corner it is the whole record, and it is the front of every eye "
    "measurement, every alignment and every DFE symbol stream in `measure`/`rx`."))
def test_sampling_a_record_returns_every_instant_the_record_holds():
    for n, spb, phase in [(16, 16.0, 0.0), (32, 16.0, 0.0), (48, 16.0, 0.0),
                          (4096, 16.0, 0.0), (4096, 16.0, 8.0), (4096, 4.82, 0.0)]:
        x = np.cos(2 * np.pi * 0.05 * np.arange(n))
        avail = 0
        while int(round(avail * spb + phase)) < n:
            avail += 1
        got = len(MS.sample_at_phase(x, spb, phase))
        assert got == avail, (f"n={n} spb={spb} phase={phase}: returned {got} of {avail} "
                              f"available sampling instants")


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `measure.eye_height` returns -inf as a MEASUREMENT, and "
    "`measure.attributes` / `ground_truth` copy it into the label dict. Measured on a "
    "one-UI record (16 samples, 16 samples/UI): eye_height = -inf, and attributes returns "
    "{'eye_contour': -inf, 'eye_sigma': -inf, 'ptp': 4.4e-16, 'rms': 1.0}. An empty record "
    "gives -inf too. The module's own docstring says these are the values training should "
    "use as ground truth INSTEAD of the recipe's knobs, so an -inf here is a poisoned "
    "label rather than a diagnostic -- it survives a min/max, an average and a CSV, and it "
    "is indistinguishable from a real closed eye. Two internal paths produce it: fewer "
    "than levels*4 samples at a phase, and an empty level cluster. Fix: raise, or return "
    "NaN with a stated 'insufficient symbols' reason, and never let it into `attributes`."))
def test_an_unmeasurable_eye_is_refused_rather_than_reported_as_minus_infinity():
    g = _grid(4096)
    x = P.nrz(n_ui=256, n=4096, causal=True)
    for short in (x[:16], x[:64]):
        try:
            h = MS.eye_height(short, g, levels=2)
        except (ValueError, ZeroDivisionError):
            continue
        assert np.isfinite(h), f"eye_height on {len(short)} samples returned {h}"
    attrs = MS.attributes(x[:16], g, levels=2)
    assert all(np.isfinite(v) for v in attrs.values()), attrs


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `acquire.record_decimation` silently DISCARDS the tail of the record "
    "whenever `depth > n/2`, because `b = n//depth` floors to 1 and only `b*depth` samples "
    "are ever read. Measured on n=4096: depth=2049 uses 2049 samples = 50.0 % of the "
    "record; depth=3000 uses 73.2 %; depth=1000 uses 97.7 %; depth=64 uses 100 %. The "
    "consequence is worst for the mode whose entire purpose is the opposite -- "
    "'narrow transients and the noise envelope survive time compression' -- a 1.0 spike at "
    "0.9*n comes back as 0.000 at depth=3000 and as 1.000 at depth=64. `mode='sample'` at "
    "such a depth is `x[::1][:depth]`, i.e. the record's HEAD, not a decimation of it. "
    "(Separately, depth=0 raises ZeroDivisionError rather than this module's own "
    "ValueError.) Fix: interval edges from `np.linspace(0, n, depth+1)` so the intervals "
    "tile the record whatever the ratio."))
def test_decimating_a_record_still_covers_the_whole_record():
    n = 4096
    spike = np.zeros(n)
    spike[int(0.9 * n)] = 1.0
    ramp = np.arange(n, dtype=float)
    for depth in (64, 1000, 2049, 3000):
        ph = AQ.record_decimation(spike, mode="peak_hold", depth=depth)
        assert np.max(ph) == pytest.approx(1.0), (
            f"peak_hold depth={depth}: the record's 1.0 peak came back as {np.max(ph):.3f}")
        av = AQ.record_decimation(ramp, mode="average", depth=depth)
        assert av[-1] > 0.9 * (n - 1), (
            f"average depth={depth}: last bin {av[-1]:.0f} of a ramp ending at {n - 1}")


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `instrument.digitize` on an empty record returns NaN in its "
    "PROVENANCE. Measured: digitize(np.zeros(0), bits=11, clip_full_scale=1.0) returns "
    "info = {'clip_full_scale': 1.0, 'clipped_fraction': nan, 'bits': 11}, from "
    "`mask.mean()` on an empty mask (numpy warns 'Mean of empty slice'). Every other stage "
    "in the chain refuses a zero-length record outright -- store_record, interleave_adc, "
    "shaped_noise_floor all raise ValueError -- so `digitize` is the one door through which "
    "an empty record becomes a RECORDED, plausible-looking number. `clipped_fraction` feeds "
    "provenance and measured ground truth, where a NaN is exactly the fabricated datum this "
    "project's gates exist to catch. Fix: refuse len(x) == 0, or report 0.0 clipped."))
def test_provenance_from_a_degenerate_record_carries_no_nan():
    y, info = I.digitize(np.zeros(0), bits=11, clip_full_scale=1.0,
                         rng=np.random.default_rng(0))
    assert len(y) == 0
    bad = {k: v for k, v in info.items()
           if isinstance(v, float) and not np.isfinite(v)}
    assert not bad, f"non-finite provenance: {bad}"


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT: `physics.multi_reflection` rounds `td_ps` to the nearest SAMPLE and "
    "says nothing when that rounding reaches ZERO, at which point the op is no longer a "
    "reflection at all -- it is a gain. Measured at fs=256 GSa/s (dt=3.906 ps): td_ps of "
    "0.5, 1.0 and 1.9 ps all round to 0 samples, and the output is bit-identical "
    "(np.array_equal) to td_samples=0, i.e. x * 1.054545291674 for gamma_s=0.3, "
    "gamma_l=0.4, 6 bounces. A 1.9 ps stub -- a package pin, a via, a connector footprint "
    "-- therefore arrives as a 0.46 dB flat gain error with no echo and no diagnostic, and "
    "the caller's recipe still says 1.9 ps. This is the same class as the three silent "
    "clamps already recorded here (ac_couple's corner, resolve_rise_time's floor, "
    "trend_floor_db), and the two of those that were fixed were fixed by WARNING. Fix: warn "
    "when the realised delay differs from the request by more than half a sample, or "
    "interpolate as `cascade_channel` does (an exact linear phase, no rounding at all)."))
def test_a_reflection_delay_below_one_sample_is_not_silently_zero():
    n = 4096
    g = _grid(n)
    x = P.nrz(n_ui=256, n=n, causal=True)
    zero = P.multi_reflection(x, td_samples=0, gamma_s=0.3, gamma_l=0.4)
    for td_ps in (0.5, 1.0, 1.9):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            y = P.multi_reflection(x, td_ps=td_ps, grid=g, gamma_s=0.3, gamma_l=0.4)
        if caught:
            continue                                  # saying so is an acceptable answer
        assert not np.array_equal(y, zero), (
            f"td_ps={td_ps} (= {td_ps * 1e-12 * g.fs:.3f} samples) delivered the "
            f"zero-delay answer with no warning")


# --------------------------------------------------------------------------------------
#  direct run
# --------------------------------------------------------------------------------------
def _xfail_reason(fn):
    for mark in getattr(fn, "pytestmark", []):
        if mark.name == "xfail":
            return mark.kwargs.get("reason", "")
    return None


def main():
    import inspect

    warnings.simplefilter("ignore")
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    n_pass = n_xfail = n_fail = n_xpass = 0
    print(f"audit_conditioning — numerical conditioning and degenerate inputs\n"
          f"{'=' * 88}")
    for name, fn in tests:
        reason = _xfail_reason(fn)
        params = [p for p in inspect.signature(fn).parameters]
        calls = [{"n": v} for v in _DEGENERATE_LENGTHS] if params == ["n"] else [{}]
        for kw in calls:
            label = name + (f"[n={kw['n']}]" if kw else "")
            try:
                fn(**kw)
                out = (MEASURED.get(name) or [None])[-1]
                if reason is not None:
                    n_xpass += 1
                    print(f"XPASS  {label}\n       *** a known defect now passes; the "
                          f"strict xfail marker must be removed ***")
                else:
                    n_pass += 1
                    extra = ""
                    if isinstance(out, dict):
                        extra = "  " + ", ".join(
                            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in list(out.items())[:6])
                    elif isinstance(out, (list, float)):
                        extra = f"  {out}"
                    print(f"pass   {label}{extra}")
            except Exception as exc:
                msg = str(exc).strip().splitlines()[0][:150]
                if reason is not None:
                    n_xfail += 1
                    print(f"XFAIL  {label}\n       {type(exc).__name__}: {msg}")
                else:
                    n_fail += 1
                    print(f"FAIL   {label}\n       {type(exc).__name__}: {msg}")
    print(f"{'=' * 88}\n{n_pass} passed, {n_xfail} xfailed (known defects, still present), "
          f"{n_fail} failed, {n_xpass} xpassed")
    return 1 if (n_fail or n_xpass) else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""GitHub #84: RF measures with ground truth -- `wfmsynth.rf`.

Every figure here is checked against something that is not the code under test: a closed form
(a tone's power, the exponential law of Gaussian envelope power, EVM = 10**(-EsN0/20)), a level
that was injected by hand, or an impairment whose size the test chose. The EVM corrections are
tested from both sides: with the correction named, the amount RETURNED is the amount injected;
without it, the impairment is still in the figure.
"""
from __future__ import annotations

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Grid, Signal
from wfmsynth import iq as IQ
from wfmsynth import rf

FS, BAUD, BETA = 400e6, 25e6, 0.35
N = 1 << 16
G = Grid(fs=FS, baud=BAUD, n=N, domain="iq", fc_hz=1e9)
T = np.arange(N) / FS
SKIP = 8
FLOOR = 0.005          # EVM of an unimpaired record: the 16-symbol truncation of the pulse


def source(phase=100):
    return Signal(seed=1, grid=G).iq("qam16", prbs=15, prbs_phase=phase)


@pytest.fixture(scope="module")
def clean():
    s = source()
    return s.waveform(), s.transmitted()[0]


def noise(n, var, seed=0):
    rng = np.random.default_rng(seed)
    return np.sqrt(var / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


# ------------------------------------------------------------------ spectrum
@pytest.mark.parametrize("window, peak_tol", [("flattop", 0.002), ("hann", 0.16)])
def test_tone_peak_and_power(window, peak_tol):
    """An off-bin complex tone of amplitude A has power A**2. Integrated over a band it is
    right to 1e-6; read off the peak it is right to the window's scalloping loss (0.2 % for a
    flat-top, up to 15 % for a Hann)."""
    a, f0 = 0.7, 12.3456e6
    s = rf.spectrum(a * np.exp(2j * np.pi * f0 * T), G, rbw_hz=100e3, window=window)
    k = int(np.argmax(s["psd_per_hz"]))
    assert abs(s["f_hz"][k] - f0) <= s["bin_hz"]
    assert abs(rf.band_power(s, f0 - 1e6, f0 + 1e6) / a ** 2 - 1) < 1e-6
    peak = s["psd_per_hz"][k] * s["rbw_hz"] / a ** 2
    assert 1 - peak_tol < peak <= 1 + 1e-9
    assert abs(s["rbw_hz"] / 100e3 - 1) < 1e-3 and s["window"] == window
    assert s["two_sided"] and s["fc_hz"] == 1e9


def test_real_record_is_one_sided_and_holds_all_the_power():
    a, f0 = 0.7, 12.3456e6
    x = a * np.cos(2 * np.pi * f0 * T)
    s = rf.spectrum(x, Grid(fs=FS, n=N), rbw_hz=100e3)
    assert s["f_hz"][0] == 0 and s["f_hz"][-1] == FS / 2 and not s["two_sided"]
    assert abs(rf.band_power(s, f0 - 1e6, f0 + 1e6) / (a ** 2 / 2) - 1) < 1e-6
    assert abs(np.sum(s["psd_per_hz"]) * s["bin_hz"] / np.mean(x ** 2) - 1) < 1e-3


def test_white_noise_density_does_not_depend_on_resolution():
    """Complex noise of variance 1 at fs has density 1/fs at every resolution: the resolution
    bandwidth is a noise bandwidth, so it scales a tone's reading and not a floor's."""
    x = noise(N, 1.0)
    for rbw in (None, 200e3, 2e6):
        s = rf.spectrum(x, G, rbw_hz=rbw)
        assert abs(np.mean(s["psd_per_hz"]) * FS - 1) < 0.02
    assert rf.spectrum(x, G)["segment"] == N // 8


def test_envelope_spectrum_is_3_db_above_its_passband():
    """The stated convention: an envelope's power is twice its passband voltage's."""
    g = Grid(fs=8e9, baud=BAUD, n=1 << 17, domain="iq", fc_hz=1e9)
    s = Signal(seed=1, grid=g).iq("qam16", prbs=15, prbs_phase=100)
    env = rf.spectrum(s.waveform(), g, rbw_hz=1e6)
    y = s.upconvert().waveform()
    pas = rf.spectrum(y, Grid(fs=8e9, n=len(y)), rbw_hz=1e6)
    ratio = rf.band_power(env, -20e6, 20e6) / rf.band_power(pas, 1e9 - 20e6, 1e9 + 20e6)
    assert abs(10 * np.log10(ratio) - 10 * np.log10(2)) < 0.01


def test_spectrum_refusals():
    x = noise(4096, 1.0)
    g = Grid(fs=FS, n=4096, domain="iq")
    with pytest.raises(ValueError, match="finest"):
        rf.spectrum(x, g, rbw_hz=1e3)
    with pytest.raises(ValueError, match="too short"):
        rf.spectrum(x, g, rbw_hz=100e6)
    with pytest.raises(ValueError, match="unknown window"):
        rf.spectrum(x, g, window="kaiser")
    with pytest.raises(ValueError, match="sample rate"):
        rf.spectrum(x, None)
    with pytest.raises(ValueError, match="1-D"):
        rf.spectrum(np.zeros((2, 64)), g)
    with pytest.raises(ValueError, match="outside the spectrum"):
        rf.band_power(rf.spectrum(x, g), 0, 300e6)
    with pytest.raises(ValueError, match="lo < hi"):
        rf.band_power(rf.spectrum(x, g), 5e6, 1e6)


# ------------------------------------------------------------------ ACLR
@pytest.mark.parametrize("filt, beta, bw", [("rect", None, (1 + BETA) * BAUD),
                                            ("rrc", BETA, BAUD)])
@pytest.mark.parametrize("level_db", [-20.0, -40.0])
def test_aclr_reads_an_injected_adjacent_channel(clean, filt, beta, bw, level_db):
    """A second, independent QAM signal placed 40 MHz up at a known level: the ratio measured
    is the ratio injected, to 0.1 dB. The other side has nothing in it and reads the source's
    own skirt, above 55 dB."""
    x, _ = clean
    other = source(phase=5000).waveform()
    injected = 10 ** (level_db / 20) * other * np.exp(2j * np.pi * 40e6 * T)
    want = 10 * np.log10(np.mean(np.abs(x) ** 2) / np.mean(np.abs(injected) ** 2))
    r = rf.aclr(x + injected, G, channel_bw_hz=bw, offsets_hz=[-40e6, 40e6], filter=filt,
                beta=beta)
    assert abs(r["aclr_db"][1] - want) < 0.1
    assert r["aclr_db"][0] > 55
    assert r["filter"] == filt and r["offsets_hz"] == [-40e6, 40e6]


def test_aclr_filters_pass_the_same_noise_power():
    """Both measuring filters have a noise bandwidth of the channel width."""
    x = noise(N, 1.0)
    rect = rf.aclr(x, G, channel_bw_hz=20e6, offsets_hz=[50e6])
    rrc = rf.aclr(x, G, channel_bw_hz=20e6, offsets_hz=[50e6], filter="rrc", beta=0.22)
    assert abs(rect["channel_power"] / (20e6 / FS) - 1) < 0.03
    assert abs(rrc["channel_power"] / rect["channel_power"] - 1) < 0.03
    assert abs(rect["aclr_db"][0]) < 0.3


def test_aclr_on_a_real_passband_record_agrees_with_its_envelope():
    """The same measure on the real record: the channel is named by its carrier frequency."""
    g = Grid(fs=8e9, baud=BAUD, n=1 << 18, domain="iq", fc_hz=1e9)
    t = np.arange(g.n) / g.fs
    a = Signal(seed=1, grid=g).iq("qam16", prbs=15, prbs_phase=100).waveform()
    b = Signal(seed=1, grid=g).iq("qam16", prbs=15, prbs_phase=5000).waveform()
    x = a + 0.03 * b * np.exp(2j * np.pi * 40e6 * t)
    y = Signal(seed=0, grid=g).capture(values=x).upconvert().waveform()
    kw = dict(channel_bw_hz=(1 + BETA) * BAUD, offsets_hz=[40e6])
    env = rf.aclr(x, g, **kw)
    pas = rf.aclr(y, Grid(fs=8e9, n=len(y)), center_hz=1e9, **kw)
    assert abs(env["aclr_db"][0] - pas["aclr_db"][0]) < 0.05
    assert abs(env["channel_power"] / pas["channel_power"] - 2) < 0.01
    with pytest.raises(ValueError, match="center_hz"):
        rf.aclr(y, Grid(fs=8e9, n=len(y)), **kw)


def test_aclr_refusals(clean):
    x, _ = clean
    with pytest.raises(ValueError, match="tenth of the"):
        rf.aclr(x, G, channel_bw_hz=30e6, offsets_hz=[40e6], rbw_hz=5e6)
    with pytest.raises(ValueError, match="outside the spectrum"):
        rf.aclr(x, G, channel_bw_hz=30e6, offsets_hz=[190e6])
    with pytest.raises(ValueError, match="roll-off"):
        rf.aclr(x, G, channel_bw_hz=30e6, offsets_hz=[40e6], filter="rrc")
    with pytest.raises(ValueError, match="filter='rect' has none"):
        rf.aclr(x, G, channel_bw_hz=30e6, offsets_hz=[40e6], beta=0.2)
    with pytest.raises(ValueError, match="unknown filter"):
        rf.aclr(x, G, channel_bw_hz=30e6, offsets_hz=[40e6], filter="gauss")
    with pytest.raises(ValueError, match="none of them 0"):
        rf.aclr(x, G, channel_bw_hz=30e6, offsets_hz=[0.0])
    with pytest.raises(ValueError, match="no power"):
        rf.aclr(np.zeros(N, complex), G, channel_bw_hz=30e6, offsets_hz=[40e6])


# ------------------------------------------------------------------ CCDF
def test_ccdf_of_complex_gaussian_noise_is_the_exponential_law():
    """P(|x|**2 > r * mean) = exp(-r), to 4 standard errors of a count over 2**20 samples
    (0.13 % of the probability at +0 dB, 2.3 % at +8 dB where it is 1.8e-3)."""
    levels = np.array([0.0, 3.0, 6.0, 8.0])
    c = rf.ccdf(noise(1 << 20, 2.0), levels_db=levels)
    law = np.exp(-10 ** (levels / 10))
    assert np.all(np.abs(c["probability"] - law) < 4 * np.sqrt(law * (1 - law) / c["n"]))
    assert c["n"] == 1 << 20 and abs(c["avg_power"] / 2.0 - 1) < 0.01


def test_ccdf_peak_to_average():
    """A constant envelope has no peak-to-average ratio; two equal tones have 3.01 dB (to
    0.005 dB: the largest SAMPLE is not quite the envelope's crest)."""
    one = np.exp(2j * np.pi * 1e6 * T)
    assert abs(rf.ccdf(one)["papr_db"]) < 1e-9
    two = one + np.exp(2j * np.pi * 3.7e6 * T)
    c = rf.ccdf(two)
    assert abs(c["papr_db"] - 10 * np.log10(2)) < 5e-3
    assert c["levels_db"][0] == 0 and c["levels_db"][-1] >= c["papr_db"]
    assert c["probability"][-1] == 0 and np.all(np.diff(c["probability"]) <= 0)


def test_ccdf_refuses_a_real_record_and_an_empty_one():
    with pytest.raises(ValueError, match="downconvert"):
        rf.ccdf(np.cos(2 * np.pi * 1e6 * T))
    with pytest.raises(ValueError, match="all zero"):
        rf.ccdf(np.zeros(64, complex))


# ------------------------------------------------------------------ EVM
def test_evm_of_a_clean_record_is_below_the_stated_floor(clean):
    x, sent = clean
    r = rf.evm(x, G, sent)
    assert r["evm"] < FLOOR and r["removed"] == {} and r["skip"] == 0
    assert r["n_symbols"] == len(sent) and r["reference"] == "truth"
    assert "rrc matched filter" in r["receiver"]
    assert abs(r["evm_pct"] - 100 * r["evm"]) < 1e-12
    assert abs(r["evm_db"] - 20 * np.log10(r["evm"])) < 1e-12
    # the same number the symbol-level primitive gives: one definition, not two
    assert r["evm"] == pytest.approx(IQ.evm(IQ.demodulate(x, G), sent), rel=1e-12)


@pytest.mark.parametrize("esn0_db, tol", [(30.0, 0.02), (20.0, 0.03), (10.0, 0.04)])
def test_evm_against_truth_is_the_analytic_awgn_value(clean, esn0_db, tol):
    """White complex noise of variance s2 per sample has N0 = s2/fs; a unit-power symbol has
    Es = 1/baud. A matched filter gives EVM = sqrt(N0/Es) = 10**(-EsN0/20). 4080 symbols put
    the estimate's own scatter near 1.1 %; the 30 dB case also carries the truncation floor."""
    x, sent = clean
    var = FS / (BAUD * 10 ** (esn0_db / 10))
    r = rf.evm(x + noise(N, var), G, sent, skip=SKIP)
    assert abs(r["evm"] / 10 ** (-esn0_db / 20) - 1) < tol


def test_a_decided_reference_flatters_and_the_truth_says_by_how_much(clean):
    x, sent = clean
    noisy = x + noise(N, FS / (BAUD * 10.0))               # Es/N0 = 10 dB
    truth = rf.evm(noisy, G, sent, skip=SKIP)
    blind = rf.evm(noisy, G, sent, skip=SKIP, reference="decided", constellation="qam16")
    assert blind["decision_errors"] > 500
    assert blind["evm"] < 0.9 * truth["evm"]
    quiet = rf.evm(x, G, sent, skip=SKIP, reference="decided", constellation="qam16")
    assert quiet["decision_errors"] == 0
    assert quiet["evm"] == pytest.approx(rf.evm(x, G, sent, skip=SKIP)["evm"], rel=1e-12)
    assert "decision_errors" not in rf.evm(x, G, reference="decided", constellation="qam16")
    # an alphabet given as points decides the same way as its name
    pts = rf.constellation_points("qam16")
    assert len(pts) == 16 and abs(np.mean(np.abs(pts) ** 2) - 1) < 1e-12
    again = rf.evm(noisy, G, skip=SKIP, reference="decided", constellation=pts)
    assert again["evm"] == pytest.approx(blind["evm"], rel=1e-12)


def test_gain_and_phase_are_counted_unless_named_and_returned_when_named(clean):
    x, sent = clean
    y = 0.8 * np.exp(0.3j) * x
    raw = rf.evm(y, G, sent, skip=SKIP)
    assert raw["evm"] == pytest.approx(abs(0.8 * np.exp(0.3j) - 1), rel=0.01)   # 34 %
    only_gain = rf.evm(y, G, sent, skip=SKIP, correct="gain")
    assert set(only_gain["removed"]) == {"gain_db"}
    assert only_gain["evm"] == pytest.approx(abs(np.exp(0.3j) - 1), rel=0.01)   # phase stays
    only_phase = rf.evm(y, G, sent, skip=SKIP, correct=("phase",))
    assert set(only_phase["removed"]) == {"phase_rad"}
    assert only_phase["evm"] == pytest.approx(0.2, rel=0.01)                    # gain stays
    both = rf.evm(y, G, sent, skip=SKIP, correct=("gain", "phase"))
    assert both["evm"] < FLOOR
    assert abs(both["removed"]["gain_db"] - 20 * np.log10(0.8)) < 1e-3
    assert abs(both["removed"]["phase_rad"] - 0.3) < 1e-4


def test_offset_and_frequency_are_returned(clean):
    """A carrier leak of 0.02-0.01j in the transmitter and a 1234 Hz carrier error. The
    frequency is estimated before the offset is, so the leak biases it: 1.1 Hz here, on a
    164 us record whose own resolution is 6 kHz."""
    x, sent = clean
    y = (x + (0.02 - 0.01j)) * np.exp(2j * np.pi * 1234.0 * T)
    raw = rf.evm(y, G, sent, skip=SKIP)
    assert raw["evm"] > 0.5
    r = rf.evm(y, G, sent, skip=SKIP, correct=("frequency", "offset", "phase"))
    assert abs(r["removed"]["frequency_hz"] - 1234.0) < 2.0
    assert abs(r["removed"]["offset"] - (0.02 - 0.01j)) < 2e-3
    assert r["evm"] < FLOOR
    # leave the offset in and it is still in the figure
    left = rf.evm(y, G, sent, skip=SKIP, correct=("frequency", "phase"))
    assert left["evm"] == pytest.approx(abs(0.02 - 0.01j), rel=0.05)


def _delayed(x, tau):
    return np.fft.ifft(np.fft.fft(x) * np.exp(-2j * np.pi * np.fft.fftfreq(len(x), 1 / FS) * tau))


@pytest.mark.parametrize("reference", ["truth", "decided"])
def test_timing_is_found_and_returned(clean, reference):
    """A record 0.23 symbol late: 32 % raw; with timing named, the delay comes back to 0.1 %
    of a symbol by either reference."""
    x, sent = clean
    tau = 0.23 / BAUD
    y = _delayed(x, tau)
    kw = dict(constellation="qam16") if reference == "decided" else {}
    raw = rf.evm(y, G, sent, skip=SKIP, reference=reference, **kw)
    r = rf.evm(y, G, sent, skip=SKIP, reference=reference, correct=("timing",), **kw)
    assert rf.evm(y, G, sent, skip=SKIP)["evm"] > 0.25 and raw["removed"] == {}
    assert abs(r["removed"]["timing_s"] - tau) * BAUD < 1e-3
    assert r["evm"] < FLOOR and r["delay_s"] == pytest.approx(tau, rel=5e-3)
    # the same delay stated by the caller needs no correction at all
    assert rf.evm(y, G, sent, skip=SKIP, delay_s=tau)["evm"] < FLOOR


def test_equalizer_removes_an_echo_and_returns_its_taps(clean):
    """An echo one symbol late at 0.2j: 20 % raw. The equaliser's first-order inverse is a
    tap of -0.2j one symbol late, and that is what comes back."""
    x, sent = clean
    y = x + 0.2j * np.roll(x, int(FS / BAUD))
    assert rf.evm(y, G, sent, skip=SKIP)["evm"] == pytest.approx(0.2, rel=0.01)
    r = rf.evm(y, G, sent, skip=SKIP, correct=("equalizer",), equalizer_taps=7)
    w = r["removed"]["equalizer_taps"]
    assert len(w) == 7 and r["evm"] < FLOOR
    assert abs(w[3] - 1) < 5e-3 and abs(w[2] - (-0.2j)) < 5e-3     # w[3 - 1]: one symbol back


def test_decision_directed_fit_recovers_gain_and_phase(clean):
    x, sent = clean
    y = 0.8 * np.exp(0.15j) * x
    r = rf.evm(y, G, sent, skip=SKIP, reference="decided", constellation="qam16",
               correct=("gain", "phase"))
    assert r["decision_errors"] == 0 and r["evm"] < FLOOR
    assert abs(r["removed"]["gain_db"] - 20 * np.log10(0.8)) < 1e-2
    assert abs(r["removed"]["phase_rad"] - 0.15) < 1e-3


def test_evm_through_a_passband_round_trip(clean):
    """The same measure on a real passband record through downconvert, decimated: the floor
    holds, and the grid that describes the record is the stored one."""
    g = Grid(fs=8e9, baud=BAUD, n=1 << 18, domain="iq", fc_hz=1e9)
    s = (Signal(seed=1, grid=g).iq("qam16", prbs=15, prbs_phase=100).upconvert()
         .downconvert(fc_hz=1e9, lpf=dict(pass_hz=20e6, stop_hz=60e6), fs_out_hz=250e6))
    x = s.waveform()
    sent, _ = s.transmitted()
    r = rf.evm(x, s.stored_grid(x), sent, skip=SKIP)
    assert r["evm"] < FLOOR and r["n_symbols"] == len(sent) - 2 * SKIP
    with pytest.raises(ValueError, match="stored_grid"):
        rf.evm(x, g, sent)                       # the synthesis grid does not describe x


def test_evm_refusals(clean):
    x, sent = clean
    with pytest.raises(ValueError, match="downconvert"):
        rf.evm(x.real, G, sent)
    with pytest.raises(ValueError, match="needs the transmitted symbols"):
        rf.evm(x, G)
    with pytest.raises(ValueError, match="needs constellation="):
        rf.evm(x, G, reference="decided")
    with pytest.raises(ValueError, match="does not use it"):
        rf.evm(x, G, sent, constellation="qam16")
    with pytest.raises(ValueError, match="reference is"):
        rf.evm(x, G, sent, reference="ideal")
    with pytest.raises(ValueError, match="at most once"):
        rf.evm(x, G, sent, correct=("gain", "carrier"))
    with pytest.raises(ValueError, match="BOTH"):
        rf.evm(x, G, sent, correct=("equalizer",))
    with pytest.raises(ValueError, match="BOTH"):
        rf.evm(x, G, sent, equalizer_taps=5)
    with pytest.raises(ValueError, match="odd"):
        rf.evm(x, G, sent, correct=("equalizer",), equalizer_taps=4)
    with pytest.raises(ValueError, match="nothing to measure"):
        rf.evm(x, G, sent, skip=len(sent) // 2)
    with pytest.raises(ValueError, match="unknown constellation"):
        rf.evm(x, G, reference="decided", constellation="qam32")
    with pytest.raises(ValueError, match="symbol rate"):
        rf.evm(x, Grid(fs=FS, n=N, domain="iq"), sent)


# ------------------------------------------------------------------ density, and the package
def test_constellation_density_counts_every_symbol_the_right_way_up():
    pts = np.array([0.5 + 0.5j] * 3 + [-0.5 + 0.5j] * 2 + [-0.5 - 0.5j])
    d = rf.constellation_density(pts, bins=2, extent=1.0)
    assert d["density"].tolist() == [[1.0, 0.0], [2.0, 3.0]]        # [q, i]: row 0 is Q < 0
    assert d["i_edges"].tolist() == [-1.0, 0.0, 1.0]
    assert rf.constellation_density(pts)["i_edges"][-1] == pytest.approx(0.55)
    with pytest.raises(ValueError, match="widen extent"):
        rf.constellation_density(pts, extent=0.4)
    with pytest.raises(ValueError, match="complex"):
        rf.constellation_density(pts.real)


def test_module_is_exported_and_measures_change_nothing(clean):
    x, sent = clean
    before = x.copy()
    rf.evm(x, G, sent, correct=("gain", "phase", "offset"))
    rf.ccdf(x)
    rf.aclr(x, G, channel_bw_hz=30e6, offsets_hz=[40e6])
    assert np.array_equal(x, before)
    assert ws.rf is rf and "rf" in ws.__all__


# ------------------------------------------- the measures read back what the impairment ops state
def test_the_measures_read_back_what_the_impairment_ops_state():
    """The two halves of `wfmsynth.rf` against each other: an impairment op states what it did
    (`rf_report`, `lo_phase`), and a measure that was told nothing reads the same number off
    the record. Each bound is the estimator's own standard error, stated beside it."""
    f_off, esn0_db = 1234.0, 25.0
    sig = (Signal(seed=1, grid=G).iq("qam16", prbs=15, prbs_phase=100)
           .freq_offset(f_hz=f_off).awgn(es_n0_db=esn0_db, ref_power=1.0))
    x, (sent, _) = sig.waveform(), sig.transmitted()
    stated = {r["name"]: r for r in sig.rf_report()}
    raw = rf.evm(x, G, sent, skip=8)
    fit = rf.evm(x, G, sent, skip=8, correct=("frequency", "phase"))
    # the oscillator: lo_phase() is the label, and `removed` is what a receiver would find
    t = np.arange(len(x)) / G.fs
    assert np.polyfit(t, sig.lo_phase(), 1)[0] / (2 * np.pi) == pytest.approx(f_off, abs=1e-6)
    n, span = fit["n_symbols"], fit["n_symbols"] / G.baud
    evm_stated = 10 ** (-stated["awgn"]["es_n0_db"] / 20)
    f_se = (evm_stated / np.sqrt(2)) * np.sqrt(12) / (2 * np.pi * span * np.sqrt(n))
    assert 1.0 < f_se < 4.0                                 # Hz, for this record
    assert abs(fit["removed"]["frequency_hz"] - f_off) < 4 * f_se
    # the noise: with the rotation taken out, what is left is the stated Es/N0
    assert raw["evm"] > 0.5                                 # the offset alone turns it > 60 deg
    assert fit["evm"] == pytest.approx(evm_stated, rel=4 / np.sqrt(2 * n))
    # a spur's stated level is the power `spectrum` finds at its frequency
    spur = Signal(seed=1, grid=G).iq("qam16", prbs=15).spur(f_hz=60e6, dbc=-30, ref_power=1.0)
    power = rf.band_power(rf.spectrum(spur.waveform(), G, rbw_hz=200e3), 59e6, 61e6)
    assert power == pytest.approx(spur.rf_report()[0]["amplitude"] ** 2, rel=1e-3)
    # a carrier leak is the offset `evm` removes when asked, in the symbols' units
    leak = Signal(seed=1, grid=G).iq("qam16", prbs=15).iq_offset(i=0.02, q=-0.01)
    got = rf.evm(leak.waveform(), G, leak.transmitted()[0], skip=8,
                 correct=("offset",))["removed"]["offset"]
    assert abs(got - (0.02 - 0.01j)) < 2e-3

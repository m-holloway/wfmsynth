"""GitHub #83: what a transmitter, an oscillator and a channel do to an I/Q record.

Seven ops, one effect each (`wfmsynth/rf.py`). Every one is checked here by MEASURING THE
RECORD with a method that shares no code with the op, against the number the recipe states:

  * iq_imbalance  the image tone's level against the closed form, and the passband identity
                  the rail model is defined by
  * iq_offset     the record's mean
  * freq_offset   the slope and curvature of the unwrapped phase
  * phase_noise   a periodogram of the measured phase against 2*L(f), band by band, each
                  within the estimate error its bin count gives; and the tone's own sideband
                  spectrum, which is what "dBc/Hz" means on a spectrum analyser
  * awgn          a tone's SNR read off the spectrum, and a QPSK record's Es/N0 read off the
                  error vector of its matched-filtered symbols
  * spur          the spur bin against the carrier bin
  * dac           a full-scale tone's signal-to-quantisation-noise against 6.02 N + 1.76 dB,
                  and the clipping level against the arcsine law of a sinusoid

and each refusal is provoked by name. The probes are tones that sit exactly on an FFT bin,
so a bin IS a tone and there is no window between the record and the number.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import signal as sp

import wfmsynth as ws
from wfmsynth import Grid, Signal
from wfmsynth import compose as C
from wfmsynth import domain as DOMAIN
from wfmsynth import iq as IQ
from wfmsynth import rf as RF

FS, BAUD, N = 400e6, 25e6, 1 << 16
G = Grid(fs=FS, baud=BAUD, n=N, domain="iq", fc_hz=2.4e9)
K = 1001                                   # the probe tone's bin: 6.11 MHz above centre
PROFILE = [(1e4, -70), (1e5, -90), (1e6, -120), (2e7, -130)]


def tone(k=K, grid=G, seed=3):
    n = grid.n
    return Signal(seed=seed, grid=grid).capture(values=np.exp(2j * np.pi * k * np.arange(n) / n))


def bins(y):
    return np.fft.fft(y) / len(y)


def qpsk(seed=5, n=1 << 16):
    g = Grid(fs=16 * BAUD, baud=BAUD, n=n, domain="iq", fc_hz=2.4e9)
    return Signal(seed=seed, grid=g).iq("qpsk", prbs=15, prbs_phase=100)


# ------------------------------------------------------------------ iq_imbalance
@pytest.mark.parametrize("gain_db,phase_deg", [(0.5, 0.0), (0.0, 3.0), (0.5, 3.0), (-1.0, -5.0)])
def test_imbalance_puts_the_image_where_the_closed_form_says(gain_db, phase_deg):
    """A tone at +f gains one at -f, (1 + 2g cos(phi) + g^2) / (1 - 2g cos(phi) + g^2) below
    it -- the textbook image-rejection ratio, written out here rather than imported."""
    s = tone().iq_imbalance(gain_db=gain_db, phase_deg=phase_deg)
    Y = bins(s.waveform())
    measured = 20 * np.log10(abs(Y[K]) / abs(Y[-K]))
    g, c = 10 ** (gain_db / 20), np.cos(np.deg2rad(phase_deg))
    want = 10 * np.log10((1 + 2 * g * c + g * g) / (1 - 2 * g * c + g * g))
    assert abs(measured - want) < 1e-9
    assert abs(s.rf_report()[0]["image_rejection_db"] - want) < 1e-9
    # nothing else appeared: the tone and its image are the whole record
    rest = np.delete(np.abs(Y), [K, N - K])
    assert rest.max() < 1e-12


def test_imbalance_is_the_two_rail_modulator_it_says_it_is():
    """The definition, at RF: passband = I cos(wt) - g Q sin(wt + phi). Up-converting the
    impaired envelope must give exactly that, built here from the clean rails."""
    g = Grid(fs=8e9, baud=25e6, n=1 << 12, domain="iq", fc_hz=1e9)
    clean = tone(k=3, grid=g)
    x = clean.waveform()
    y = tone(k=3, grid=g).iq_imbalance(gain_db=1.5, phase_deg=8.0).upconvert().waveform()
    wt = 2 * np.pi * 1e9 * np.arange(g.n) / g.fs
    want = x.real * np.cos(wt) - 10 ** (1.5 / 20) * x.imag * np.sin(wt + np.deg2rad(8.0))
    assert np.abs(y - want).max() < 1e-12


def test_a_stated_image_level_is_one_helper_and_stays_visible_in_the_recipe():
    """There is no `image` op: an image is what imbalance produces. `imbalance_for` returns
    the parameters, so the recipe holds the mechanism and not a second name for it."""
    s = tone().iq_imbalance(**RF.imbalance_for(30))
    Y = bins(s.waveform())
    assert abs(20 * np.log10(abs(Y[K]) / abs(Y[-K])) - 30.0) < 1e-9
    assert s.ops[-1] == {"op": "iq_imbalance", **RF.imbalance_for(30)}
    assert s.ops[-1]["phase_deg"] == 0.0 and abs(s.ops[-1]["gain_db"] - 0.5496) < 1e-4
    assert "image" not in C._EXEC
    with pytest.raises(ValueError, match="must be positive"):
        RF.imbalance_for(0)


# ------------------------------------------------------------------ iq_offset
def test_offset_is_the_records_mean_and_reports_its_leak():
    s = tone().iq_offset(i=0.02, q=-0.01)
    assert abs(s.waveform().mean() - (0.02 - 0.01j)) < 1e-15
    leak = s.rf_report()[0]["leak_dbc"]
    assert abs(leak - 10 * np.log10(0.02 ** 2 + 0.01 ** 2)) < 1e-9          # tone power is 1
    # the same leak stated in dBc is a spur at the centre frequency
    same = tone().spur(f_hz=0.0, dbc=leak, phase_rad=float(np.angle(0.02 - 0.01j)))
    assert np.abs(same.waveform() - s.waveform()).max() < 1e-12


# ------------------------------------------------------------------ freq_offset
def test_frequency_offset_and_drift_are_the_slope_and_curvature_of_the_phase():
    s = tone().freq_offset(f_hz=12345.0, drift_hz_per_s=5e6)
    t = np.arange(N) / FS
    phase = np.unwrap(np.angle(s.waveform() * np.conj(tone().waveform())))
    c2, c1, c0 = np.polyfit(t, phase, 2)
    assert abs(c1 / (2 * np.pi) - 12345.0) < 1e-6          # Hz
    assert abs(c2 / np.pi - 5e6) < 1e-2                    # Hz/s
    assert abs(c0) < 1e-9
    assert np.abs(phase - s.lo_phase()).max() < 1e-12
    r = s.rf_report()[0]
    assert r["f_start_hz"] == 12345.0 and abs(r["f_end_hz"] - (12345.0 + 5e6 * t[-1])) < 1e-9


def test_a_whole_bin_offset_moves_the_tone_and_nothing_else():
    Y = np.abs(bins(tone().freq_offset(f_hz=7 * FS / N).waveform()))
    assert abs(Y[K + 7] - 1) < 1e-12 and np.delete(Y, K + 7).max() < 1e-12


def test_ops_stated_in_hz_read_the_rate_the_record_is_at():
    """After a decimating `downconvert` the record is not on the synthesis grid's rate. An
    op that used `grid.fs` would turn the envelope 32 times too slowly."""
    g = Grid(fs=8e9, baud=25e6, n=1 << 16, domain="iq", fc_hz=1e9)

    def chain():
        return tone(k=41, grid=g).upconvert().downconvert(fc_hz=1e9, lpf=dict(pass_hz=20e6), fs_out_hz=250e6)

    x = chain().waveform()
    s = chain().freq_offset(f_hz=1e6)
    y = s.waveform()
    assert len(y) == 2048
    core = slice(200, -200)                               # the low-pass's edges
    slope = np.polyfit(np.arange(2048)[core] / 250e6,
                       np.unwrap(np.angle(y * np.conj(x)))[core], 1)[0]
    assert abs(slope / (2 * np.pi) - 1e6) < 1e-3
    assert len(s.lo_phase()) == 2048


# ------------------------------------------------------------------ phase_noise
def _phase(n=1 << 18, profile=PROFILE, seed=3):
    g = Grid(fs=FS, baud=BAUD, n=n, domain="iq", fc_hz=2.4e9)
    s = tone(grid=g, seed=seed).phase_noise(profile)
    y = s.waveform()
    return s, y, np.angle(y * np.conj(tone(grid=g).waveform()))


def test_the_phase_psd_is_the_stated_profile_band_by_band():
    """The phase is read back off the record (the angle against the clean tone) and its
    periodogram compared with 2*L(f). Each bin of a Gaussian process's periodogram is
    exponentially distributed about the true PSD, so the mean of M whitened bins has a
    standard error of 1/sqrt(M): that is the STATED estimate error, and the bound is 4 of
    them."""
    n = 1 << 18
    s, _, theta = _phase(n)
    f = np.fft.rfftfreq(n, 1 / FS)[1:]
    psd = 2 * np.abs(np.fft.rfft(theta)[1:]) ** 2 / (n * FS)          # one-sided, rad^2/Hz
    want = 2 * 10 ** (RF.profile_dbc_hz(f, PROFILE) / 10)
    inside = want > 0
    for lo, hi in ((1e4, 1e5), (1e5, 1e6), (1e6, 2e7)):
        band = (f >= lo) & (f <= hi)
        m = int(band.sum())
        ratio = float(np.mean(psd[band] / want[band]))
        assert abs(ratio - 1) < 4 / np.sqrt(m), (lo, hi, m, ratio)
    assert abs(np.mean(psd[inside] / want[inside]) - 1) < 4 / np.sqrt(inside.sum())
    # NOTHING outside the stated offsets -- not a floor, not a tail
    assert psd[~inside].max() < 1e-25 * psd[inside].max()
    assert abs(theta.mean()) < 1e-12


def test_welch_on_the_phase_agrees_too():
    """The estimate a user would actually make. Two decades in from the profile's low end a
    Hann-windowed Welch estimate has no leakage to speak of; 127 half-overlapped segments
    and >= 26 bins a band put its standard error under 2 %, and the bound is 0.3 dB."""
    _, _, theta = _phase(1 << 20)
    f, p = sp.welch(theta, fs=FS, nperseg=1 << 14, detrend=False)
    for lo, hi in ((1e6, 2e6), (2e6, 5e6), (5e6, 1.5e7)):
        band = (f >= lo) & (f <= hi)
        want = 2 * 10 ** (RF.profile_dbc_hz(f[band], PROFILE) / 10)
        assert abs(10 * np.log10(p[band].mean() / want.mean())) < 0.3


def test_dbc_per_hz_is_what_a_spectrum_analyser_reads_beside_the_carrier():
    """L(f) is defined on the phase; for a small phase it is also the tone's single sideband,
    per Hz, relative to the carrier. 0.05 rad rms here, so the two agree to second order."""
    n = 1 << 18
    _, y, _ = _phase(n)
    Y = np.abs(np.fft.fft(y)) ** 2 / n ** 2                 # power per bin; carrier at K
    carrier = Y[K]
    for lo, hi in ((1e5, 1e6), (1e6, 2e7)):
        k = np.arange(int(np.ceil(lo * n / FS)), int(hi * n / FS))
        ssb = Y[K + k] / carrier / (FS / n)                 # dBc/Hz, upper sideband
        want = 10 ** (RF.profile_dbc_hz(k * FS / n, PROFILE) / 10)
        assert abs(np.mean(ssb / want) - 1) < 4 / np.sqrt(len(k)) + 0.01


def test_integrated_phase_is_the_profiles_and_the_record_only_estimates_it():
    """`phase_rms_rad` is the closed-form integral, checked against a numerical one; a single
    record's rms is dominated by its lowest few bins and is an estimate."""
    f = np.logspace(4, np.log10(2e7), 200001)
    s_phi = 2 * 10 ** (RF.profile_dbc_hz(f, PROFILE) / 10)
    want = np.sqrt(np.sum(0.5 * (s_phi[1:] + s_phi[:-1]) * np.diff(f)))        # trapezoid
    assert abs(RF.phase_rms_rad(PROFILE) / want - 1) < 1e-6
    assert abs(RF.phase_rms_rad([(1e3, -60), (1e5, -80)]) / np.sqrt(
        2 * 1e-6 * 1e3 * np.log(100)) - 1) < 1e-12        # the 1/f segment, by hand
    s, _, theta = _phase(1 << 18)
    r = s.rf_report()[0]
    assert r["rms_rad"] == RF.phase_rms_rad(PROFILE)
    assert abs(r["realised_rms_rad"] - theta.std()) < 1e-12
    rms = [_phase(1 << 16, seed=k)[2].std() for k in range(40)]
    assert abs(np.sqrt(np.mean(np.square(rms))) / r["rms_rad"] - 1) < 0.1


def test_the_oscillator_is_ground_truth_and_its_own_role():
    s, y, theta = _phase(1 << 16)
    assert np.abs(s.lo_phase() - theta).max() < 1e-12
    s.awgn(snr_db=30, bw_hz=FS)
    assert s.roles() == ["phase_noise/1", "awgn/2"]
    a = s.waveform()
    b = s.contrast("phase_noise/1")
    clean = tone(grid=s.grid).waveform()
    # a different oscillator, the SAME channel noise sample for sample
    noise_a = a - clean * np.exp(1j * s.lo_phase())
    noise_b = b - clean * np.exp(1j * s.lo_phase(ws.Streams(s.seed).reroll("phase_noise/1")))
    assert np.abs(a - b).max() > 1e-3 and np.abs(noise_a - noise_b).max() < 1e-12
    # and the other way round
    c = s.contrast("awgn/2")
    assert np.abs(a - c).max() > 1e-3
    assert np.abs(np.angle(c * np.conj(clean)) - np.angle(a * np.conj(clean))).max() < 0.5


# ------------------------------------------------------------------ awgn
def test_snr_in_a_stated_bandwidth_is_what_the_spectrum_shows():
    """A tone on one bin, noise on all the others. Over the whole band the realised noise
    power is the stated one (it is normalised, so only the tone's own bin is missing: 1/N);
    in a sixteenth of the band it is an average of 4096 bins, standard error 1.6 %, and the
    bound is 4 of them (0.27 dB)."""
    s = tone().awgn(snr_db=20, bw_hz=FS / 16)
    Y = np.abs(bins(s.waveform())) ** 2
    f = np.fft.fftfreq(N, 1 / FS)
    noise = np.ones(N, bool)
    noise[K] = False
    inband = noise & (np.abs(f) <= FS / 32)
    assert abs(10 * np.log10(Y[K] / Y[inband].sum()) - 20.0) < 10 * np.log10(
        1 + 4 / np.sqrt(inband.sum()))
    whole = 10 * np.log10(Y[K] / Y[noise].sum())
    r = s.rf_report()[0]
    assert abs(whole - (20.0 - 10 * np.log10(16))) < 0.02 and abs(whole - r["snr_db"]) < 0.02
    assert r["ref_power"] == pytest.approx(1.0) and r["n0"] == pytest.approx(1 / (FS / 16 * 100))
    assert r["noise_power"] == pytest.approx(0.16) and r["es_n0_db"] == pytest.approx(20.0)
    # the two rails carry it equally and independently
    w = s.waveform() - tone().waveform()
    assert abs(w.real.var() / w.imag.var() - 1) < 0.05
    assert abs(np.corrcoef(w.real, w.imag)[0, 1]) < 4 / np.sqrt(N)


@pytest.mark.parametrize("request_,want_db", [(dict(es_n0_db=15), 15.0),
                                              (dict(eb_n0_db=12), 12 + 10 * np.log10(2)),
                                              (dict(snr_db=15, bw_hz=BAUD), 15.0)])
def test_es_n0_is_what_the_matched_filter_sees(request_, want_db):
    """Es/N0 read off the record by a receiver: matched-filter, sample, and the error vector
    against `transmitted()` is the noise. 4076 symbols estimate a variance to 1.6 %
    (0.07 dB); the bound is 4 of those. The reference power is STATED as 1 (the
    constellation's), so the record's own start-up and PRBS imbalance do not enter."""
    s = qpsk().awgn(ref_power=1.0, **request_)
    sent, _ = s.transmitted()
    got = IQ.demodulate(s.waveform(), s.grid)
    core = slice(10, len(sent) - 10)
    measured = -20 * np.log10(IQ.evm(got[core], sent[core]))
    assert abs(measured - want_db) < 10 * np.log10(1 + 4 / np.sqrt(len(sent) - 20))
    assert abs(s.rf_report()[0]["es_n0_db"] - want_db) < 1e-9


def test_ebn0_takes_its_bits_per_symbol_from_the_source_and_records_it():
    s = qpsk().awgn(eb_n0_db=12)
    assert s.ops[-1] == {"op": "awgn", "eb_n0_db": 12, "bits_per_symbol": 2}
    same = qpsk().awgn(es_n0_db=12 + 10 * np.log10(2))
    assert np.abs(s.waveform() - same.waveform()).max() < 1e-12
    g = Grid(fs=16 * BAUD, baud=BAUD, n=1 << 12, domain="iq")
    by_hand = Signal(seed=1, grid=g).iq(symbols=[1, 1j, -1, -1j] * 8).awgn(eb_n0_db=12)
    with pytest.raises(ValueError, match="needs bits_per_symbol="):
        by_hand.waveform()


def test_the_reference_power_is_the_record_as_it_arrives_unless_stated():
    """The default is measured, and it measures EVERYTHING there: a spur added first raises
    it. Stating `ref_power` pins it."""
    plain = tone().awgn(snr_db=10, bw_hz=FS).rf_report()[0]
    after = tone().spur(f_hz=2000 * FS / N, dbc=0.0).awgn(snr_db=10, bw_hz=FS).rf_report()[1]
    pinned = tone().spur(f_hz=2000 * FS / N, dbc=0.0).awgn(snr_db=10, bw_hz=FS,
                                                 ref_power=1.0).rf_report()[1]
    assert plain["ref_power"] == pytest.approx(1.0)
    assert after["ref_power"] == pytest.approx(2.0, rel=1e-6)
    assert pinned["noise_power"] == pytest.approx(plain["noise_power"])


# ------------------------------------------------------------------ spur
def test_spur_level_is_its_bin_against_the_carriers():
    k = -3000
    s = tone().spur(f_hz=k * FS / N, dbc=-45.0, phase_rad=0.7)
    Y = bins(s.waveform())
    assert abs(20 * np.log10(abs(Y[k]) / abs(Y[K])) + 45.0) < 1e-9
    assert abs(np.angle(Y[k]) - 0.7) < 1e-9
    assert s.rf_report()[0]["amplitude"] == pytest.approx(10 ** (-45 / 20))


# ------------------------------------------------------------------ dac
@pytest.mark.parametrize("bits", [8, 10, 12, 14])
def test_a_full_scale_tone_has_the_textbook_quantisation_noise(bits):
    """6.02 N + 1.76 dB is a full-scale sinusoid over a uniform step's q^2/12, and each rail
    of a complex tone is a full-scale sinusoid. The tone sits on a bin coprime to the record
    length, so it visits 65536 distinct phases and the error is spread, not a harmonic comb.
    Measured: within 0.05 dB from 8 bits up; the bound is 0.15 dB."""
    s = tone().dac(bits=bits)
    Y = np.abs(bins(s.waveform())) ** 2
    sqnr = 10 * np.log10(Y[K] / np.delete(Y, K).sum())
    assert abs(sqnr - (6.02 * bits + 1.76)) < 0.15
    r = s.rf_report()[0]
    assert abs(sqnr - r["sqnr_db"]) < 0.15
    assert r["clipped_fraction"] == 0.0
    assert r["lsb"] == pytest.approx(2 * r["full_scale"] / 2 ** bits)
    # every sample is on a code, and the peak is ON the last one
    codes = s.waveform().real / r["lsb"]
    assert np.abs(codes - np.rint(codes)).max() < 1e-9
    assert np.rint(codes).max() == 2 ** (bits - 1) - 1


def test_headroom_costs_exactly_its_own_decibels():
    a = tone().dac(bits=10).rf_report()[0]
    b = tone().dac(bits=10, headroom_pct=100).rf_report()[0]
    assert b["full_scale"] == pytest.approx(2 * a["full_scale"])
    assert b["backoff_db"] - a["backoff_db"] == pytest.approx(20 * np.log10(2))
    assert a["sqnr_db"] - b["sqnr_db"] == pytest.approx(20 * np.log10(2))
    # a stated range is the same thing
    c = tone().dac(bits=10, full_scale=b["full_scale"])
    assert np.array_equal(c.waveform(), tone().dac(bits=10, headroom_pct=100).waveform())


def test_crest_factor_is_the_envelopes_peak_over_its_mean_power():
    """A single complex tone has a constant envelope (0 dB); two equal tones peak at four
    times one tone's power with a mean of twice it (3.01 dB). Both by hand."""
    assert abs(tone().dac(bits=12).rf_report()[0]["crest_factor_db"]) < 1e-9
    turn = 2j * np.pi * np.arange(N) / N
    two = np.exp(1001 * turn) + np.exp(2003 * turn)
    r = Signal(grid=G).capture(values=two).dac(bits=12).rf_report()[0]
    assert abs(r["crest_factor_db"] - 10 * np.log10(2)) < 1e-6
    qam = Signal(seed=1, grid=G).iq("qam16", prbs=15).dac(bits=10, headroom_pct=5)
    x = Signal(seed=1, grid=G).iq("qam16", prbs=15).waveform()
    assert qam.rf_report()[0]["crest_factor_db"] == pytest.approx(
        10 * np.log10(np.abs(x).max() ** 2 / np.mean(np.abs(x) ** 2)))


def test_clipping_is_refused_until_it_is_asked_for_and_then_it_is_the_arcsine_law():
    """Driven 30 % past full scale, a sinusoid spends 2*acos(1/1.3)/pi of its time clipped --
    a closed form for the fraction `rf_report` counts and for the samples that sit on the
    end codes."""
    with pytest.raises(ValueError, match="would be\\s+clipped. Pass clip='hard'"):
        tone().dac(bits=12, headroom_pct=-23.0769).waveform()
    s = tone().dac(bits=12, headroom_pct=100 / 1.3 - 100, clip="hard")
    r = s.rf_report()[0]
    want = 2 * np.arccos(1 / 1.3) / np.pi
    assert abs(r["clipped_fraction"] - want) < 2e-3
    y = s.waveform()
    top = r["full_scale"] - r["lsb"]
    on_rail = np.mean((y.real >= top - 1e-9) | (y.real <= -r["full_scale"] + 1e-9))
    assert abs(on_rail - want) < 2e-3
    assert y.real.max() == pytest.approx(top) and y.real.min() == pytest.approx(-r["full_scale"])


# ------------------------------------------------------------------ the recipe, the composer
def _everything():
    return (qpsk(n=1 << 14)
            .dac(bits=10, headroom_pct=5)
            .iq_imbalance(gain_db=0.3, phase_deg=2.0)
            .iq_offset(i=0.01)
            .phase_noise(profile=[(1e5, -80), (1e6, -100), (1e7, -120)])
            .freq_offset(f_hz=12e3, drift_hz_per_s=1e6)
            .spur(f_hz=40e6, dbc=-45)
            .awgn(es_n0_db=20))


def test_the_recipe_is_data_and_replays_bit_identically():
    s = _everything()
    r = json.loads(json.dumps(s.recipe()))
    again = Signal.from_recipe(r)
    assert np.array_equal(again.waveform(), s.waveform())
    assert again.sha256() == s.sha256()
    assert r["ops"][4] == {"op": "phase_noise",
                           "profile": [[1e5, -80], [1e6, -100], [1e7, -120]]}
    assert [e["name"] for e in s.rf_report()] == list(
        ("dac", "iq_imbalance", "iq_offset", "phase_noise", "freq_offset", "spur", "awgn"))
    assert s.waveform().dtype == np.complex128


@pytest.mark.parametrize("op,knob,alt", [
    ("dac", "bits", 6), ("dac", "headroom_pct", 40), ("iq_imbalance", "gain_db", 1.0),
    ("iq_imbalance", "phase_deg", -4.0), ("iq_offset", "i", 0.1), ("freq_offset", "f_hz", 5e3),
    ("freq_offset", "drift_hz_per_s", 0.0), ("spur", "f_hz", 30e6), ("spur", "dbc", -30),
    ("awgn", "es_n0_db", 10), ("phase_noise", "profile", [[1e5, -70], [1e7, -110]]),
    ("iq_offset", "q", 0.05), ("spur", "phase_rad", 1.0), ("spur", "ref_power", 4.0),
    ("awgn", "ref_power", 4.0)])
def test_every_knob_moves_the_samples_and_the_digest(op, knob, alt):
    a, b = _everything(), _everything()
    hit = next(o for o in b.ops if o["op"] == op)
    assert hit.get(knob) != alt
    hit[knob] = alt
    assert a.sha256() != b.sha256()
    assert np.abs(a.waveform() - b.waveform()).max() > 1e-6


def test_a_toleranced_impairment_is_another_unit_with_its_label_in_the_recipe():
    s = tone().iq_imbalance(gain_db=ws.tol(0.5, tol_abs=0.3), phase_deg=2.0)
    assert s.roles() == ["unit/1/gain_db"]
    g = s.ops[-1]["gain_db"]
    assert 0.2 <= g <= 0.8 and g != 0.5
    Y = bins(s.waveform())
    assert abs(20 * np.log10(abs(Y[K]) / abs(Y[-K])) - RF.image_rejection_db(g, 2.0)) < 1e-9
    assert s.unit(9).ops[-1]["gain_db"] != g


def test_the_ops_are_iq_only_general_stages_with_a_lead_in_class():
    for op in RF.OPS:
        assert DOMAIN.accepts(op) == ("iq",)
        assert C.op_kind(op) == "general"
        assert ((op in C._LEAD_LTI) + (op in C._LEAD_SKIP) + (op in C._LEAD_REJECT)
                + (op in C._LEAD_ANALYTIC)) == 1
    real = Signal(seed=1, grid=Grid(fs=80e9, baud=10e9, n=4096)).carrier("nrz")
    for build in (lambda s: s.awgn(es_n0_db=10), lambda s: s.dac(bits=8),
                  lambda s: s.freq_offset(f_hz=1e3), lambda s: s.iq_imbalance(gain_db=1)):
        with pytest.raises(ValueError, match="real voltage record.*downconvert"):
            build(real)
    assert len(real.ops) == 1
    with pytest.raises(ValueError, match="general stage"):
        tone().iq_offset(i=0.1).canonical()
    assert tone().iq_offset(i=0.1).stage_kinds(domains=True)[-1] == ("general", "iq")


REFUSALS = [
    (lambda: Signal(grid=G).iq_offset(i=0.1), "no record yet.*iq\\(...\\) or capture"),
    (lambda: tone().iq_imbalance(), "needs gain_db=, phase_deg= or both"),
    (lambda: tone().iq_imbalance(gain_db="1"), "gain_db is one finite number"),
    (lambda: tone().iq_offset(), "needs i=, q= or both"),
    (lambda: tone().freq_offset(), "needs f_hz="),
    (lambda: tone().freq_offset(f_hz=FS / 2), "at or beyond Nyquist"),
    (lambda: tone().freq_offset(f_hz=1e6, drift_hz_per_s=2e12), "at the end\\s+of the record"),
    (lambda: tone(k=N // 2 - 10).freq_offset(f_hz=1e6), "would wrap it to the other side"),
    (lambda: tone().awgn(snr_db=10, bw_hz=FS).freq_offset(f_hz=1e6), "fold_db= accepts"),
    (lambda: tone().phase_noise([(1e4, -80)]), "two or more"),
    (lambda: tone().phase_noise([(1e5, -80), (1e4, -90)]), "positive and increasing"),
    (lambda: tone().phase_noise([(100.0, -60), (1e6, -100)]), "below 1/T\\s+has no bin"),
    (lambda: tone().phase_noise([(1e4, -80), (3e8, -100)]), "Nothing is clipped"),
    (lambda: tone().awgn(), "exactly one of"),
    (lambda: tone().awgn(snr_db=10, es_n0_db=10), "exactly one of"),
    (lambda: tone().awgn(snr_db=10), "ambiguous without bw_hz="),
    (lambda: tone().awgn(snr_db=10, bw_hz=2 * FS), "no wider than the\\s+record's band"),
    (lambda: tone().awgn(es_n0_db=10, bw_hz=1e6), "bw_hz= goes with snr_db="),
    (lambda: tone().awgn(es_n0_db=10, bits_per_symbol=2), "goes with eb_n0_db="),
    (lambda: tone().awgn(eb_n0_db=10), "needs bits_per_symbol="),
    (lambda: tone().awgn(es_n0_db=10, ref_power=-1.0), "must be\\s+positive"),
    (lambda: tone(grid=Grid(fs=FS, n=N, domain="iq")).awgn(es_n0_db=10), "needs the symbol rate"),
    (lambda: Signal(grid=G).capture(values=np.zeros(N, complex)).awgn(es_n0_db=10),
     "all zeros.*ref_power="),
    (lambda: tone().spur(f_hz=1e6), "needs f_hz=.*and dbc="),
    (lambda: tone().spur(f_hz=FS, dbc=-40), "would alias"),
    (lambda: tone().dac(bits=1), "whole number from 2 to 32"),
    (lambda: tone().dac(bits=8.5), "whole number from 2 to 32"),
    (lambda: tone().dac(bits=8, clip="soft"), "clip is 'hard'"),
    (lambda: tone().dac(bits=8, headroom_pct=5, full_scale=2.0), "not both"),
    (lambda: tone().dac(bits=8, full_scale=0.5), "Pass clip='hard'"),
    (lambda: tone().dac(bits=8, headroom_pct=-100), "no range at all"),
]


@pytest.mark.parametrize("build,match", REFUSALS, ids=[m[:28] for _, m in REFUSALS])
def test_what_cannot_be_done_honestly_is_refused_by_name(build, match):
    with pytest.raises(ValueError, match=match):
        build().waveform()


def test_a_misspelled_parameter_is_caught_where_it_is_written():
    with pytest.raises(TypeError, match="ebn0_db.*eb_n0_db="):       # 0.46.0's first spelling
        tone().awgn(ebn0_db=10)
    with pytest.raises(TypeError, match="rejection_db"):
        tone().iq_imbalance(rejection_db=30)
    with pytest.raises(ValueError, match="no freq_offset or phase_noise"):
        tone().lo_phase()


def test_a_stated_fold_lets_an_offset_follow_the_noise():
    """White noise reaches Nyquist, so an offset always wraps a sliver of it: 2*f/fs of the
    noise power, here -30 dB of the record. Refused by default; accepted when stated."""
    s = tone().awgn(snr_db=10, bw_hz=FS).freq_offset(f_hz=1e6, fold_db=-25)
    assert np.abs(bins(s.waveform())[K + int(round(1e6 * N / FS))]) > 0.5


def test_lo_phase_refuses_to_add_phases_on_two_timebases():
    g = Grid(fs=8e9, baud=25e6, n=1 << 16, domain="iq", fc_hz=1e9)
    s = (tone(k=41, grid=g).freq_offset(f_hz=1e5).upconvert()
         .downconvert(fc_hz=1e9, lpf=dict(pass_hz=20e6), fs_out_hz=250e6).freq_offset(f_hz=1e5))
    with pytest.raises(ValueError, match="not on one timebase"):
        s.lo_phase()


@pytest.mark.parametrize("profile", [
    [(1e4, -80), (1e6, -120)],                      # many bins across every segment
    [(6.2e3, -70), (1e6, -120)],                    # starts just above the first bin
    [(7e3, -60), (1.2e4, -110), (1e6, -120)],       # the close-in power sits between two bins
    [(1e4, -80), (1.2e4, -80)],                     # the whole profile is inside one bin
])
@pytest.mark.parametrize("n", [1 << 16, 1 << 17])
def test_phase_noise_delivers_the_profiles_rms_at_any_record_length(profile, n):
    """Each bin carries the profile's power over its own band, so the mean-square phase is the
    profile's integral however few bins the close-in part spans. Over 300 seeds the mean
    square of a ONE-bin profile (the widest spread: two degrees of freedom a record) is known
    to 1/sqrt(300) = 5.8 %, so its rms to 2.9 %; the bound is 3.5 sigma of that."""
    fs = 400e6                                      # bins 6.1 kHz and 3.05 kHz apart
    ms = np.mean([ws.rf.lo_noise(n, fs, profile, np.random.default_rng(s)).var()
                  for s in range(300)])
    ratio = np.sqrt(ms) / ws.rf.phase_rms_rad(profile)
    print(f"n={n}: realised / stated rms = {ratio:.4f}")
    assert abs(ratio - 1.0) < 0.10

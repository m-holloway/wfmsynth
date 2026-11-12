"""`rf.evm(reference="decided")` acquires a carrier before it decides, or refuses.

The reported failure: a QAM-16 record with a 2 kHz offset, `correct=("frequency", "phase")`,
decided reference. The record turns a third of a cycle across its 164 us, the first pass's
decisions are wrong, the decision-directed fit follows them, and the call RETURNED 24.5 %
(25.6 % in the report's record) where the truth-referenced figure is 0.32 % clean and 6.3 %
at 24 dB -- a number that is neither the signal's nor a sign that anything failed.

Now the carrier is acquired blind, from the line in the M-th power of the symbols, the
decision-directed passes refine it, and the lock is checked on the corrected symbols. Inside
the stated pull-in range the decided figure is the truth-referenced one wherever decisions
are right; outside what it can stand behind, the call raises.

The reference for every figure here is `reference="truth"`, which uses the transmitted
symbols and shares none of the blind estimator.
"""
import warnings

import numpy as np
import pytest

from wfmsynth import Grid, Signal, rf

FS, BAUD = 400e6, 25e6
SKIP = 16
BOTH = ("frequency", "phase")


def _record(f_hz, es_n0_db=None, constellation="qam16", n=1 << 16):
    g = Grid(fs=FS, baud=BAUD, n=n, fc_hz=2.4e9)
    s = Signal(seed=1, grid=g).iq(constellation, prbs=15, prbs_phase=100).freq_offset(f_hz=f_hz)
    if es_n0_db is not None:
        s = s.awgn(es_n0_db=es_n0_db, ref_power=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.waveform(), s.transmitted()[0], g


def _pair(f_hz, es_n0_db=None, constellation="qam16"):
    x, sent, g = _record(f_hz, es_n0_db, constellation)
    truth = rf.evm(x, g, sent, skip=SKIP, correct=BOTH)
    blind = rf.evm(x, g, sent, skip=SKIP, reference="decided", constellation=constellation,
                   correct=BOTH)
    return truth, blind


@pytest.mark.parametrize("es_n0_db,want_pct", [(None, 0.32), (24, 6.29)])
def test_the_reported_record_reads_what_the_truth_reads(es_n0_db, want_pct):
    truth, blind = _pair(2e3, es_n0_db)
    assert truth["evm_pct"] == pytest.approx(want_pct, abs=0.02)
    assert blind["evm_pct"] == pytest.approx(truth["evm_pct"], abs=0.01)     # it read 24.5 %
    assert blind["decision_errors"] == 0
    assert abs(blind["removed"]["frequency_hz"] - 2e3) < (1.0 if es_n0_db is None else 20.0)
    acq = blind["acquisition"]
    assert acq["order"] == 4 and acq["pull_in_hz"] == BAUD / 8
    assert acq["line"] > rf.LOCK_LINE and acq["residual_turn"] < rf.LOCK_TURN


@pytest.mark.parametrize("f_hz", [-3.0e6, -1.234e6, -2e4, 150.0, 2e3, 2e5, 2e6, 3.0e6])
def test_any_offset_inside_the_pull_in_range_is_acquired(f_hz):
    """+/- baud/8 = 3.125 MHz for QAM-16. No decision is right at the start for any of these
    but 150 Hz; the matched filter's own loss at 3 MHz is in both figures alike."""
    truth, blind = _pair(f_hz, 24)
    assert abs(blind["removed"]["frequency_hz"] - f_hz) < 25.0
    assert blind["evm_pct"] == pytest.approx(truth["evm_pct"], abs=0.02)
    assert blind["decision_errors"] == 0


@pytest.mark.parametrize("constellation,order", [("bpsk", 2), ("qpsk", 4), ("qam64", 4)])
def test_the_order_follows_the_constellation(constellation, order):
    truth, blind = _pair(1.234e6, 30, constellation)
    assert rf.symmetry_order(constellation) == order
    assert blind["acquisition"]["pull_in_hz"] == BAUD / (2 * order)
    assert abs(blind["removed"]["frequency_hz"] - 1.234e6) < 25.0
    assert blind["evm_pct"] == pytest.approx(truth["evm_pct"], abs=0.02)


def test_the_line_height_is_the_analytic_one_and_noise_has_none():
    """For n symbols of a clean constellation the line stands |E s^M| / sqrt(E |s|^2M) * sqrt(n)
    over the incoherent level: 0.3847 sqrt(n) for QAM-16, sqrt(n) for QPSK. That is the known
    answer the lock threshold is calibrated against; noise alone reads about sqrt(ln n)."""
    x, sent, g = _record(2e3)
    r = rf.evm(x, g, sent, skip=SKIP, reference="decided", constellation="qam16", correct=BOTH)
    pts = rf.constellation_points("qam16")
    c = abs(np.mean(pts ** 4)) / np.sqrt(np.mean(np.abs(pts) ** 8))
    assert c == pytest.approx(0.3847, abs=1e-4)
    assert r["acquisition"]["line"] == pytest.approx(c * np.sqrt(r["n_symbols"]), rel=0.03)
    rng = np.random.default_rng(0)
    t = np.arange(4064) / BAUD
    for _ in range(5):
        z = rng.standard_normal(4064) + 1j * rng.standard_normal(4064)
        assert rf.carrier_line(z, t, "qam16")[2] < 4.0 < rf.LOCK_LINE


def test_a_record_with_no_carrier_to_find_is_refused_not_measured():
    rng = np.random.default_rng(0)
    g = Grid(fs=FS, baud=BAUD, n=1 << 16, domain="iq")
    z = rng.standard_normal(1 << 16) + 1j * rng.standard_normal(1 << 16)
    with pytest.raises(ValueError, match=r"did not lock.*no carrier line to acquire.*"
                                         r"\+/-3\.125e\+06 Hz.*No figure is returned.*"
                                         r"reference='truth'"):
        rf.evm(z, g, reference="decided", constellation="qam16", correct=("frequency",))


def test_too_few_symbols_to_see_the_line_is_refused_with_the_numbers():
    """128 symbols of QAM-16: 0.3847 * sqrt(112) = 4.1, under the 6 the lock needs. The same
    record against the truth is measured without complaint."""
    x, sent, g = _record(2e3, n=1 << 11)
    assert len(sent) == 128
    with pytest.raises(ValueError, match=r"height \d\.\d over the incoherent level, 6 needed"
                                         r".*\(112 counted here\)"):
        rf.evm(x, g, sent, skip=8, reference="decided", constellation="qam16", correct=BOTH)
    assert rf.evm(x, g, sent, skip=8, correct=BOTH)["evm"] < 0.005


def test_outside_the_range_the_offset_comes_back_modulo_baud_over_m_as_documented():
    """3.2 MHz is past baud/8. A quarter-turn per symbol is invisible at the symbol instants,
    so the estimate is the offset less baud/4 and the EVM is still the truth's -- the matched
    filter's loss at that offset, not a failure to lock."""
    truth, blind = _pair(3.2e6, 24)
    assert abs(blind["removed"]["frequency_hz"] - (3.2e6 - BAUD / 4)) < 25.0
    assert blind["evm_pct"] == pytest.approx(truth["evm_pct"], abs=0.02)


def test_phase_is_found_at_any_angle_modulo_the_symmetry_and_errors_are_counted_there():
    x, sent, g = _record(0.0, 24)
    for turn in (0.15, 0.7, 2.0, -2.9):
        y = x * np.exp(1j * turn)
        r = rf.evm(y, g, sent, skip=SKIP, reference="decided", constellation="qam16",
                   correct=("gain", "phase"))
        want = (turn + np.pi / 4) % (np.pi / 2) - np.pi / 4          # modulo a quarter-turn
        assert abs(r["removed"]["phase_rad"] - want) < 5e-3, turn
        assert r["decision_errors"] == 0
        assert r["symmetry_steps"] == round((turn - want) / (np.pi / 2)) % 4
        assert r["evm_pct"] == pytest.approx(6.30, abs=0.03)


def test_what_was_not_asked_for_is_still_not_taken_out():
    """No correction named: nothing is acquired, the rotation counts in full, and there is no
    `acquisition` entry."""
    x, sent, g = _record(2e3)
    r = rf.evm(x, g, sent, skip=SKIP, reference="decided", constellation="qam16")
    assert r["removed"] == {} and "acquisition" not in r and r["evm"] > 0.2

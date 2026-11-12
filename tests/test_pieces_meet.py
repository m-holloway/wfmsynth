"""Where the last pieces of 0.46.0 meet each other.

Each was built and tested alone: the OFDM source and its receiver (#88), the delay and complex
gain on an envelope and `recipes.multipath` (#89), the state-dwell source (#90). These are the
checks that only exist once they are in one tree -- one piece measured by another that shares
no code with it. (A complex gain on an optical field, #89 meeting #92, is in
tests/test_optical_domain.py with the rest of the field's stages.)

Then the recipes (#91) and the swept laser (#93, #94) arrived: an enable written as a
`sequence` meets an envelope and a field, an I/Q multitone meets `multipath`, and every recipe
answers a toleranced number the same way.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Grid, Signal

SCS = 312.5e3
G = Grid(fs=20e6, n=1 << 14, fc_hz=2.4e9 + 1e6)
OFDM = dict(n_fft=64, scs_hz=SCS, cp=16, active=52, constellation="qpsk", prbs=15,
            prbs_seed=0x2F6E15)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def tx():
    return Signal(seed=1, grid=G).ofdm(**OFDM)


def test_an_ofdm_equaliser_reads_back_a_multipath_channel_carrier_phase_included():
    """The second path is 6.6 samples late (between samples, inside the 16-sample prefix), so
    subcarrier k sees H(k) = 1 + g*exp(-j*2*pi*(fc + k*scs)*tau). The OFDM receiver's
    per-subcarrier equaliser returns it; leaving the carrier's turn out of the reference is
    off by more than half the echo."""
    tau, gain = 0.33e-6, 0.4
    rx = ws.recipes.multipath(tx(), delays_s=[0.0, tau], gains=[1.0, gain])
    sent, _ = tx().transmitted()
    raw = ws.rf.evm(rx.waveform(), G, sent, ofdm=OFDM)
    eq = ws.rf.evm(rx.waveform(), G, sent, ofdm=OFDM, correct=("equalizer",))
    k = np.asarray(eq["subcarriers"])
    h = 1.0 + gain * np.exp(-2j * np.pi * (G.fc_hz + k * SCS) * tau)
    no_turn = 1.0 + gain * np.exp(-2j * np.pi * k * SCS * tau)
    err = np.abs(eq["removed"]["equalizer"] - h).max()
    print(f"raw EVM {raw['evm']:.6f} (|H - 1| = {gain}); equalised {eq['evm']:.2e}; "
          f"|equaliser - H(k)| <= {err:.2e}; without the carrier's turn "
          f"{np.abs(eq['removed']['equalizer'] - no_turn).max():.3f}")
    assert abs(raw["evm"] - gain) < 1e-3              # the echo, counted whole
    assert eq["evm"] < 3e-3 and err < 1e-3            # the fractional-delay interpolator's error
    assert np.abs(eq["removed"]["equalizer"] - no_turn).max() > 0.5 * gain
    assert rx.transmitted()[0].shape == sent.shape    # the truth survives the arrangement


def test_an_ofdm_burst_follows_a_state_schedule_and_the_intervals_say_when():
    """`gate(by=<states>)` on an OFDM envelope: inside every stated on interval the samples are
    the source's, inside every off interval nothing, to the sample."""
    sig = tx().gate(by=Signal().carrier("states", dwell=[80e-6, 30e-6]))
    x, events = sig.realize()
    spans = ws.states.intervals(events, at="gate/1/by")
    assert len(spans) > 6 and {s["state"] for s in spans} == {0, 1}
    t, ref = np.arange(G.n) / G.fs, tx().waveform()
    edge = ws.states.DEFAULT_TR_SAMPLES / G.fs
    on, inside = np.zeros(G.n, bool), np.zeros(G.n, bool)
    for s in spans:
        m = (t > s["t_start_s"] + edge) & (t < s["t_stop_s"] - edge)
        inside |= m
        on |= m & (s["state"] == 1)
    assert inside.mean() > 0.95 and 0.1 < on.mean() < 0.9
    assert np.array_equal(x[on], ref[on]) and not np.any(x[inside & ~on])
    assert sig.roles() == ["gate/1/by/states/0"]      # the schedule is the one thing drawn
    assert [e.kind for e in events if e.kind == "ofdm_symbol"]    # both truths in one list


# ------------------------------------------------------- recipes (#91) meet #89, #93 and #86
R = ws.recipes


def _closed(grid, **bounce):
    """Where `bounce_times` says the contact is closed, as a mask over the record."""
    closes, opens = R.bounce_times(**bounce)
    edges = np.round(np.sort(np.concatenate([closes, opens])) * grid.fs).astype(int)
    return np.searchsorted(edges, np.arange(grid.n), side="right") % 2 == 1


@pytest.mark.parametrize("what", ["an envelope", "a laser's field"])
def test_a_bouncing_contact_gates_an_envelope_and_a_field_to_the_sample(what):
    """`recipes.bounce` is a `sequence` of levels: a VOLTAGE wherever it sits, so it keys an
    I/Q transmitter or shutters a laser. Closed, the samples are the source's; open, nothing;
    and the instants are `bounce_times`'."""
    bounce = dict(t_s=200e-6, gap_s=40e-6)
    if what == "an envelope":
        grid, src = G, tx
    else:
        grid = Grid(fs=20e6, n=1 << 14)
        src = lambda: Signal(seed=1, grid=grid).laser(start_nm=1549.0, stop_nm=1549.01)  # noqa: E731
    ref, sig = src().waveform(), src().gate(by=R.bounce(**bounce))
    x, events = sig.realize()
    closed = _closed(grid, **bounce)
    flips = [e.t_s for e in events if e.kind == "transition"]
    truth = np.sort(np.concatenate(R.bounce_times(**bounce)))
    print(f"{what}: closed {closed.mean():.4f} of the record, {len(flips)} transitions, "
          f"largest |t - bounce_times| {np.abs(np.asarray(flips) - truth).max() * grid.fs:.2f} "
          f"samples")
    assert np.iscomplexobj(x) and 0.6 < closed.mean() < 0.7
    assert np.array_equal(x[closed], ref[closed]) and not np.any(x[~closed])
    assert len(flips) == 7 and np.abs(np.asarray(flips) - truth).max() < 0.51 / grid.fs
    assert sig.roles() == []


def test_a_bounce_is_still_refused_as_the_peer_of_an_envelope():
    """A voltage is an enable or a factor on an envelope, never a second envelope."""
    with pytest.raises(ValueError, match="REAL waveform.*peer"):
        tx().combine(R.bounce(t_s=200e-6, gap_s=40e-6)).waveform()


def test_an_iq_multitone_reads_a_multipath_channel_at_every_tone():
    """Nine tones on bins through a second path 6.6 samples late: tone k comes back scaled by
    H = 1 + g*exp(-j*2*pi*(fc + f_k)*tau), the carrier's turn included."""
    df, tau, gain = G.fs / G.n, 0.33e-6, 0.4
    plan = dict(f_lo_hz=-2048 * df, f_hi_hz=2048 * df, n_tones=9)
    comb = R.multitone(domain="iq", sig=Signal(seed=1, grid=G), **plan)
    rx = R.multipath(comb, delays_s=[0.0, tau], gains=[1.0, gain])
    f = np.asarray(R.multitone_plan(**plan)[0])
    k = np.round(f / df).astype(int) % G.n
    h = np.fft.fft(rx.waveform())[k] / np.fft.fft(comb.waveform())[k]
    err = np.abs(h - (1.0 + gain * np.exp(-2j * np.pi * (G.fc_hz + f) * tau))).max()
    no_turn = np.abs(h - (1.0 + gain * np.exp(-2j * np.pi * f * tau))).max()
    print(f"|H measured - H stated| <= {err:.2e} over 9 tones (bound 2e-3, the record's "
          f"first 7 samples have no second path); without the carrier's turn {no_turn:.3f}")
    assert err < 2e-3 and no_turn > 0.5 * gain and rx.roles() == []


LASER = Signal(seed=1, grid=Grid(fs=1e6, n=1 << 12)).laser(start_nm=1549.0, stop_nm=1551.0)
T = ws.tol(1.0, 10)


@pytest.mark.parametrize("call", [
    lambda: R.fading(doppler_hz=T),
    lambda: R.multipath(tx(), delays_s=[0.0, T], gains=[1.0, 0.5]),
    lambda: R.flicker(1e3, 1e5, rms=T),
    lambda: R.hum(50.0, amp=0.02, harmonics={3: T}),
    lambda: R.ripple(500e3, amp=T),
    lambda: R.pickup(1e6, amp=T),
    lambda: R.bounce(t_s=2e-3, gap_s=T),
    lambda: R.oscillator_startup(T, startup_s=200e-6),
    lambda: R.multitone(1e6, 9e6, 9, rms=T),
    lambda: R.buck(v_in=12.0, duty=0.5, f_sw_hz=500e3, l_h=T, c_f=22e-6, r_load_ohm=0.5),
    lambda: R.mzi(LASER, delay_s=T),
    lambda: R.stray_reflection(LASER, delay_s=40e-12, reflectivity=T),
    lambda: R.ring(LASER, wavelength_nm=1550.0, q=T),
    lambda: R.gas_cell(LASER, lines_nm=[T]),
])
def test_every_recipe_that_converts_its_numbers_refuses_a_tolerance_the_same_way(call):
    with pytest.raises(ValueError, match=r"^\w+: \w+= is a ws.tol\(\.\.\.\) or holds one.*"
                                         r"takes a number.*where the number ends up"):
        call()


def test_a_recipe_argument_that_reaches_its_op_untouched_takes_a_tolerance():
    """`intermittent`'s dwells and `ring(by=)` are handed on as they are, so the part varies
    from unit to unit under a `unit/` role."""
    gate = Signal(seed=1, grid=Grid(fs=1e6, n=1 << 12)).carrier("dc", level=1.0).gate(
        by=R.intermittent(ws.tol(5e-4, 10), 50e-6))
    ring = R.ring(LASER, wavelength_nm=1550.0, q=1e4, by=ws.tol(0.0, tol_abs=0.2))
    assert "gate/1/by/unit/0/dwell.0" in gate.roles() and ring.roles() == ["unit/1/by"]
    seen = ring.photodetect(shot=False)
    dips = {int(np.argmin(seen.unit(k).waveform())) for k in range(3)}
    assert len(dips) == 3


# ---- used together, as an agent would: what the last round of use found
def test_a_path_gain_may_be_complex_when_the_transmitter_is_a_named_chain():
    """`multipath(Signal().use("tx"), gains=[..., 0.4j])` in a `Channels` record: a chain that
    starts with `use` cannot know its domain when it is written, so the builder no longer
    calls it a voltage. The record is the one built from the transmitter directly."""
    shared = Signal().ofdm(**OFDM)
    rec = ws.Channels({"rx": R.multipath(Signal().use("tx"), [0.0, 350e-9], [1.0, 0.4j])},
                      shared={"tx": shared}, seed=1, grid=G)
    direct = R.multipath(tx(), [0.0, 350e-9], [1.0, 0.4j]).waveform()
    got = rec.waveforms()["rx"]
    assert np.max(np.abs(got - direct)) < 1e-12 * np.max(np.abs(direct))
    assert np.max(np.abs(got - tx().waveform())) > 0.1          # and the path is there


def test_a_complex_gain_on_a_named_voltage_is_still_refused_when_it_renders():
    rec = ws.Channels({"v": Signal().use("src").gain(by=0.4j)},
                      shared={"src": Signal().carrier("sine", f_hz=1e6)}, seed=1, grid=G)
    with pytest.raises(ValueError, match="complex and the record is a real voltage"):
        rec.waveforms()
    with pytest.raises(ValueError, match="complex and the record is a real voltage"):
        Signal().carrier("sine", f_hz=1e6).gain(by=0.4j)        # its own ops say: at once


def test_a_first_awgn_or_spur_is_reported_as_the_source_it_is():
    tones = R.multitone(-4e6, 4e6, 9, domain="iq", sig=Signal(seed=1, grid=G))
    assert tones.stage_kinds(domains=True)[:2] == [("source", "iq"), ("general", "iq")]
    fade = Signal().awgn(snr_db=0, bw_hz=1e3, ref_power=1.0).filter("doppler", f_hz=500.0)
    assert fade.stage_kinds() == ["source", "general"]
    assert tx().awgn(snr_db=20, bw_hz=1e6).stage_kinds() == ["source", "general"]


def test_a_recipe_handed_a_word_it_does_not_take_says_which_recipe_and_what_it_takes():
    with pytest.raises(TypeError, match=r"^hum\(\) got an unexpected keyword argument 'sig'"
                                        r".*It takes hum\(f_hz, amp"):
        R.hum(50.0, 0.02, sig=Signal(seed=1, grid=G))


def test_summary_shows_a_dwell_spec_not_only_its_keys():
    sig = Signal(seed=1, grid=Grid(fs=1e6, n=1 << 12)).carrier(
        "states", dwell={"dist": "lognormal", "median_s": 1e-4, "sigma": 1.0})
    assert "dwell={dist='lognormal', median_s=0.0001, sigma=1.0}" in sig.summary()


def test_a_chain_that_starts_with_spur_or_awgn_is_a_channel_and_a_stage_when_shared():
    """A first `spur` or `awgn` has no waveform under it, so as a CHANNEL it is the tone or
    the noise itself and other channels may `use` it; under `shared` it stays the stage it
    was (it adds to whoever uses it), and using that as a source says which to do."""
    comb = lambda: R.multitone(-4e6, 4e6, 9, domain="iq")
    rec = ws.Channels({"tx": comb(),
                       "rx": R.multipath(Signal().use("tx"), [0.0, 350e-9], [1.0, 0.4j])},
                      seed=1, grid=G)
    w = rec.waveforms()
    alone = R.multitone(-4e6, 4e6, 9, domain="iq", sig=Signal(seed=1, grid=G))
    assert np.array_equal(w["tx"], alone.waveform())
    direct = R.multipath(alone, [0.0, 350e-9], [1.0, 0.4j]).waveform()
    assert np.max(np.abs(w["rx"] - direct)) < 1e-12
    with pytest.raises(ValueError, match="under `shared` is a stage.*among the channels"):
        ws.Channels({"rx": Signal().use("tx")}, shared={"tx": comb()}, seed=1, grid=G)
    spurred = ws.Channels({"a": tx_sub().use("tone")},
                          shared={"tone": Signal().spur(f_hz=3e6, dbc=-20)}, seed=1, grid=G)
    lone = tx().spur(f_hz=3e6, dbc=-20).waveform()
    assert np.array_equal(spurred.waveforms()["a"], lone)       # the stage, as before


def tx_sub():
    return Signal().ofdm(**OFDM)

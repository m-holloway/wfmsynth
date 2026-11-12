"""The labels follow the samples. An event is on the record where its samples are the
record's, at the rate and the time the record is at when it is returned -- through a sequence,
a switch, a shared chain, an acquisition, a decimating mixer and a delay.

Each test compares the record with the SAME chain rendered without its defects and asserts the
labels cover what differs."""
import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Signal

G = ws.Grid(fs=64e9, baud=1e9, n=1 << 14)
CLOCK = dict(n_ui=256, pattern="clock")


def differs(x, clean, tol=1e-9):
    return np.flatnonzero(np.abs(x - clean) > tol)


def covered(ev, where, slack=0):
    """Every sample in `where` is under the mask, to within `slack` samples."""
    m = ev.mask()
    if slack:
        m = np.convolve(m, np.ones(2 * slack + 1), "same")
    return bool(np.all(m[where] > 0))


def test_a_sequence_segments_defects_are_on_the_record():
    def record(chain):
        return Signal(seed=1, grid=G).sequence([
            {"chain": Signal().carrier("dc", level=-1.0), "duration_s": 8192 / G.fs}, {"chain": chain}])
    x, ev = record(Signal().carrier("nrz", **CLOCK).events("glitch", count=5, amp=3.0)).realize()
    d = differs(x, record(Signal().carrier("nrz", **CLOCK)).waveform())
    kept = [e for e in ev.on_record() if e.kind == "glitch"]
    print(f"{len(d)} samples differ; {len(kept)} glitches on the record; mask {ev.mask().sum():.0f}")
    assert len(d) and len(kept) and covered(ev, d)
    assert all(e.at == "sequence/0/segments/1/chain" for e in kept)     # where it came from


def test_a_switch_inputs_defect_is_on_the_record_only_while_selected():
    sel = Signal().carrier("square", f_hz=G.fs / G.n, amp=1.0)          # one period: + then -
    data = Signal().carrier("nrz", **CLOCK)

    def record(chain):
        return Signal(seed=1, grid=G).switch(select=sel, threshold=0.0,
                                             inputs=[Signal().carrier("dc", level=0.0), chain])
    x, ev = record(data.events("glitch", count=12, amp=3.0)).realize()
    d = differs(x, record(Signal().carrier("nrz", **CLOCK)).waveform())
    glitches = [e for e in ev if e.kind == "glitch"]
    on = [e for e in glitches if e.on_record]
    print(f"{len(glitches)} glitches on the input, {len(on)} while it was selected")
    assert 0 < len(on) < len(glitches) and len(d) and covered(ev, d)
    line = sel.__class__(seed=1, grid=G); line.ops = sel.ops
    high = line.waveform() > 0.0
    assert all(high[e.sample:e.sample + e.width].any() for e in on)
    assert not any(high[e.sample:e.sample + e.width].any() for e in glitches if not e.on_record)


def test_a_shared_stage_and_a_shared_signal_put_their_defects_on_the_channel():
    hit = Signal().events("glitch", count=20, amp=2.0)
    src = Signal().carrier("nrz", **CLOCK).events("glitch", count=7, amp=2.0)
    rec = ws.Channels({"staged": Signal().carrier("nrz", **CLOCK).use("hit"),
                       "same": Signal().use("src")},
                      shared={"hit": hit, "src": src}, seed=1, grid=G)
    clean = Signal(seed=1, grid=G).carrier("nrz", **CLOCK).waveform()
    x, ev = rec.realize()
    for name in ("staged", "same"):
        d = differs(x[name], clean)
        print(f"{name}: {len(d)} samples differ, mask {ev[name].mask().sum():.0f}")
        assert len(d) and covered(ev[name], d)


def test_an_aggressors_defect_stays_off_the_record():
    s = Signal(seed=1, grid=G).carrier("nrz", **CLOCK).crosstalk(
        aggressor=Signal().carrier("nrz", n_ui=256).events("glitch", count=3, amp=2.0),
        coupling=0.1)
    ev = s.event_list()
    assert len(ev) == 3 and not ev.on_record() and ev.mask().sum() == 0
    assert all(d["on_record"] is False for d in ev.to_dicts())


def _acquired():
    return Signal(seed=1, grid=ws.Grid(fs=80e9, baud=1e9, n=1 << 15)).carrier("nrz", n_ui=400)


PROFILE = ws.AcquisitionProfile(sample_rate_hz=20e9, record_length=4096)
IQ = ws.Grid(fs=6.4e9, baud=25e6, n=1 << 16, domain="iq", fc_hz=1e9)
RATE_CHANGES = {
    "acquire": lambda ev: _acquired().events("glitch", count=40, amp=2.0, **ev).acquire(PROFILE),
    "digitize(n_out=)": lambda ev: (_acquired().events("glitch", count=40, amp=2.0, **ev)
                                    .digitize(bits=12, n_out=1 << 13)),
    "sample_clock(n_out=)": lambda ev: (_acquired().events("glitch", count=40, amp=2.0, **ev)
                                        .sample_clock(n_out=1 << 13)),
    "downconvert(fs_out_hz=)": lambda ev: (
        Signal(seed=1, grid=IQ).iq("qam16", prbs=15).upconvert()
        .events("glitch", count=6, amp=1.0, **ev)
        .downconvert(fc_hz=1e9, lpf=dict(pass_hz=40e6, stop_hz=120e6, atten_db=80),
                     fs_out_hz=400e6)),
}


@pytest.mark.parametrize("name", RATE_CHANGES)
def test_no_label_points_past_a_record_stored_at_another_rate_or_length(name):
    s = RATE_CHANGES[name]({})
    x, ev = s.realize()
    fs = s.stored_grid(x).fs
    print(f"{name}: {len(ev)} events on a {len(x)}-sample record at {fs:g} Sa/s")
    assert len(ev) and ev.n == len(x)
    for e in ev:
        assert 0 <= e.sample < len(x) and e.width >= 1
        assert abs(e.t_s - e.sample / fs) < 1.0 / fs + 1e-15       # time did not move


def test_an_acquired_records_labels_cover_the_acquired_defect():
    x, ev = RATE_CHANGES["acquire"]({}).realize()
    clean = _acquired().acquire(PROFILE).waveform()
    d = differs(x, clean, tol=0.2)                # well above the resampler's ringing
    print(f"{len(d)} acquired samples carry a glitch; {len(ev)} labels")
    # a 4:1 store floors the start and the anti-alias filter spreads the edge: 2 samples
    assert len(d) and covered(ev, d, slack=2)


def data():
    return Signal(seed=1, grid=G).carrier("nrz", n_ui=256, pattern="prbs7", seed=3)


def test_a_constant_delay_moves_the_label_with_the_defect():
    by = 1000 / G.fs
    before = data().events("glitch", count=1, amp=3.0).event_list().events[0]
    x, ev = data().events("glitch", count=1, amp=3.0).delay(by=by).realize()
    e = ev.events[0]
    assert (e.sample, e.width) == (before.sample + 1000, before.width)
    assert e.t_s == pytest.approx(before.t_s + by, abs=1e-18)
    assert covered(ev, differs(x, data().delay(by=by).waveform(), tol=0.3), slack=1)


def test_a_moving_delay_moves_each_label_by_the_delay_there():
    wobble = Signal().carrier("sine", f_hz=4 * G.fs / G.n, amp=30 / G.fs, offset=200 / G.fs)
    was = data().events("glitch", count=6, amp=3.0).event_list().events
    x, ev = data().events("glitch", count=6, amp=3.0).delay(by=wobble).realize()
    by = Signal(seed=1, grid=G); by.ops = wobble.ops
    d = by.waveform()
    now = [e for e in ev if e.kind == "glitch"]
    moved = [b.sample - a.sample for a, b in zip(was, now)]
    print("each glitch moved by [samples]:", moved)
    # the output at t is the input at t - by(t): each label is later by the delay where it LANDS
    assert moved == [int(round(d[b.sample] * G.fs)) for b in now] and len(set(moved)) > 1
    assert covered(ev, differs(x, data().delay(by=wobble).waveform(), tol=0.3), slack=2)


def test_a_label_delayed_past_the_end_is_gone_with_its_samples():
    s = data().events("glitch", indices=[250], amp=3.0)
    assert len(s.event_list()) == 1
    x, ev = s.delay(by=8 / G.baud).realize()                 # 8 UI later: off the record
    assert len(ev) == 0
    assert np.array_equal(x, data().delay(by=8 / G.baud).waveform())

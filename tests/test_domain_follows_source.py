"""A record's domain follows its source when the grid does not state one.

`Grid(domain="iq")` used to be required beside an I/Q source, so the same fact was said twice
and forgetting the second was a refusal. A grid that leaves `domain` out now takes it from the
chain's first op; a grid that STATES a domain is held to it, and a source that contradicts it
is refused by name. Real records are untouched: their recipes never carried a domain and still
do not, so their JSON and digests are what they were.
"""
import copy
import dataclasses
import json
import pickle

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import Grid, Signal

FS, BAUD, N = 64e6, 4e6, 4096
PLAIN = Grid(fs=FS, baud=BAUD, n=N)
STATED = Grid(fs=FS, baud=BAUD, n=N, domain="iq")


def _tones(n=256):
    t = np.arange(n) / FS
    return np.exp(2j * np.pi * 1e6 * t) + 0.25 * np.exp(-2j * np.pi * 3e6 * t)


def test_an_iq_source_makes_the_record_iq_and_it_is_the_record_that_said_so():
    a = Signal(seed=1, grid=PLAIN).iq("qam16", prbs=15).awgn(snr_db=25, bw_hz=BAUD)
    b = Signal(seed=1, grid=STATED).iq("qam16", prbs=15).awgn(snr_db=25, bw_hz=BAUD)
    assert a.grid.domain == "iq" and a.grid.domain_stated
    assert a.to_json() == b.to_json() and a.sha256() == b.sha256()
    x = a.waveform()
    assert np.iscomplexobj(x) and np.array_equal(x, b.waveform())
    assert np.array_equal(Signal.from_recipe(json.loads(a.to_json())).waveform(), x)
    # the measures see it the same way
    assert a.stored_grid(x).domain == "iq"


def test_a_complex_capture_makes_the_record_iq(tmp_path):
    x = _tones()
    g = Grid(fs=FS, n=256)
    by_value = Signal(seed=0, grid=g).capture(values=x)
    assert by_value.grid.domain == "iq" and np.array_equal(by_value.waveform(), x)
    np.save(tmp_path / "a.npy", x)
    np.savez(tmp_path / "a.npz", values=x)
    for name in ("a.npy", "a.npz"):
        s = Signal(seed=0, grid=g).capture(path=str(tmp_path / name))
        assert s.grid.domain == "iq" and np.array_equal(s.waveform(), x), name
        assert s.to_json() == Signal(seed=0, grid=Grid(fs=FS, n=256, domain="iq")).capture(
            path=str(tmp_path / name)).to_json()
    # a real file on the same grid is a real record, exactly as before
    np.save(tmp_path / "r.npy", x.real)
    r = Signal(seed=0, grid=g).capture(path=str(tmp_path / "r.npy"))
    assert r.grid.domain == "real" and r.waveform().dtype == np.float64
    assert "domain" not in r.recipe()["grid"]


def test_an_iq_file_brings_its_centre_frequency_with_it(tmp_path):
    x = _tones(1024).astype(np.complex64)
    ws.to_sigmf(tmp_path / "rec", x, Grid(fs=FS, n=1024, domain="iq", fc_hz=2.4e9))
    s = Signal(seed=0, grid=Grid(fs=FS, n=1024)).capture(
        path=tmp_path / "rec.sigmf-data", format="sigmf")
    assert (s.grid.domain, s.grid.fc_hz) == ("iq", 2.4e9)
    assert np.array_equal(s.waveform(), ws.read_iq(tmp_path / "rec.sigmf-data",
                                                   format="sigmf").waveform())
    # the centre it brought is the one `upconvert` needs, with nothing more said
    assert s.upconvert is not None and s.recipe()["grid"]["fc_hz"] == 2.4e9


def test_fc_hz_alone_states_an_envelope():
    g = Grid(fs=FS, baud=BAUD, n=N, fc_hz=1e9)
    assert g.domain == "iq" and g.domain_stated
    assert g == Grid(fs=FS, baud=BAUD, n=N, domain="iq", fc_hz=1e9)
    with pytest.raises(ValueError, match="carrier.*I/Q grid"):
        Signal(seed=1, grid=g).carrier("nrz", n_ui=256)


def test_a_stated_domain_that_contradicts_the_source_is_refused_by_name():
    real = Grid(fs=FS, baud=BAUD, n=N, domain="real")
    with pytest.raises(ValueError, match=r"^iq: this chain's grid STATES domain='real'.*"
                                         r"Leave domain= off the Grid"):
        Signal(seed=1, grid=real).iq("qpsk", prbs=9)
    with pytest.raises(ValueError, match=r"complex \(I/Q\) samples.*states domain='real'"):
        Signal(seed=0, grid=Grid(fs=FS, n=256, domain="real")).capture(values=_tones())
    with pytest.raises(ValueError, match=r"carrier.*I/Q grid.*source of a real voltage"):
        Signal(seed=1, grid=STATED).carrier("nrz", n_ui=256)


def test_a_real_records_grid_recipe_and_digest_are_what_they_were():
    """The recipe of a real record has never carried a domain. The digest below is of the
    recipe with its version field removed, taken on the commit before this change (the JSON
    is the same bytes on every host; the samples are compared in-process instead, since their
    last bits follow the host's LAPACK)."""
    g = Grid(fs=64e9, baud=16e9, n=4096)

    def chain(grid):
        return (Signal(seed=3, grid=grid).carrier("nrz", n_ui=1024, pattern="prbs7")
                .lossy(loss_db=6.0, loss_at_ghz=8.0).rx_noise(rms=0.01))

    sig = chain(g)
    r = sig.recipe()
    assert r["grid"] == {"fs": 64e9, "baud": 16e9, "n": 4096, "v_full": 1.0}
    assert sig.grid.domain == "real" and isinstance(sig.grid.domain, str)
    assert g == Grid(fs=64e9, baud=16e9, n=4096, domain="real")           # equal, hash equal
    assert hash(g) == hash(Grid(fs=64e9, baud=16e9, n=4096, domain="real"))
    assert "domain='real'" in repr(g)
    r.pop("wfmsynth_version")
    import hashlib
    assert hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()[:16] == \
        "48115f1befec60a3"
    # stating what was the only possibility before is the same record, recipe and samples
    stated = chain(Grid(fs=64e9, baud=16e9, n=4096, domain="real"))
    assert stated.to_json() == sig.to_json()
    assert np.array_equal(stated.waveform(), sig.waveform())


def test_not_stating_survives_replace_copy_and_pickle():
    """`dataclasses.replace` passes every field back in, so "not stated" has to be a value the
    field can hold, or a grid resized by its user would silently become a stated real one."""
    for g in (dataclasses.replace(PLAIN, n=8192), copy.deepcopy(PLAIN),
              pickle.loads(pickle.dumps(PLAIN))):
        assert not g.domain_stated and g.domain == "real"
        assert Signal(seed=1, grid=g).iq("qpsk", prbs=9).grid.domain == "iq"
    assert dataclasses.replace(PLAIN, domain="real").domain_stated


def test_a_hand_written_recipe_may_leave_the_domain_to_its_source():
    r = Signal(seed=1, grid=STATED).iq("qpsk", prbs=9).recipe()
    want = Signal.from_recipe(r).waveform()
    del r["grid"]["domain"], r["grid"]["fc_hz"]
    assert np.array_equal(Signal.from_recipe(r).waveform(), want)
    r["grid"]["domain"] = "real"
    with pytest.raises(ValueError, match="STATES domain='real'|domain 'iq' only"):
        Signal.from_recipe(r).waveform()


def test_a_sequence_of_iq_chains_is_an_iq_record():
    part = lambda **kw: Signal().iq("qpsk", **kw)                         # noqa: E731
    a = Signal(seed=2, grid=PLAIN).sequence(
        [{"chain": part(prbs=7), "duration_s": 20e-6}, {"chain": part(prbs=9)}])
    b = Signal(seed=2, grid=STATED).sequence(
        [{"chain": part(prbs=7), "duration_s": 20e-6}, {"chain": part(prbs=9)}])
    assert a.grid.domain == "iq" and a.to_json() == b.to_json()
    assert np.array_equal(a.waveform(), b.waveform())
    # and one of real chains stays real
    c = Signal(seed=2, grid=PLAIN).sequence(
        [{"chain": Signal().carrier("dc", level=0.0), "duration_s": 20e-6},
         {"chain": Signal().carrier("sine", f_hz=1e6)}])
    assert c.grid.domain == "real" and c.waveform().dtype == np.float64

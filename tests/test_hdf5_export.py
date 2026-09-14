"""HDF5 export: the samples survive, the axes are right, and the layout is the one readers expect.

An exported record has two audiences. Any HDF5 reader needs the codes plus the four numbers that
put them in volts and seconds. A bench instrument additionally looks for the identifying metadata
that marks a file as one of its own, which is specific to whoever wrote the capture and therefore
comes from a reference file rather than from this library -- `like=` is that path, and the test
below builds its own reference so it depends on nothing outside this repository.
"""
from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from wfmsynth import hdf5                                          # noqa: E402

FS = 256e9


def _signal(n=20_000, seed=0):
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    return 0.35 * np.sign(np.sin(2 * np.pi * 8e9 * t)) + 0.004 * rng.normal(size=n)


def test_the_samples_survive_to_within_half_a_code(tmp_path):
    v = _signal()
    p = hdf5.write_hdf5(tmp_path / "r.h5", v, fs=FS, full_scale=1.0)
    got, _ = hdf5.read_hdf5(p, channel=1)
    half_code = 1.0 / (2 * (2 ** 15 - 1)) / 2
    assert np.max(np.abs(got - v)) <= half_code * 1.01


def test_the_axes_are_reconstructed_from_the_stored_attributes(tmp_path):
    """A reader that never sees this library has to be able to do it from the file alone."""
    v = _signal(5_000)
    p = hdf5.write_hdf5(tmp_path / "r.h5", v, fs=FS, t0=-1e-9, full_scale=1.0)
    with h5py.File(p, "r") as f:
        a = f["Waveforms/Channel 1"].attrs
        codes = np.asarray(f["Waveforms/Channel 1/Channel 1Data"][:], dtype=np.int64)
        volts = codes * float(a["YInc"]) + float(a["YOrg"])
        t = np.arange(codes.size) * float(a["XInc"]) + float(a["XOrg"])
        assert bytes(a["YUnits"]) == b"Volt" and bytes(a["XUnits"]) == b"Second"
        assert float(a["XInc"]) == pytest.approx(1.0 / FS)
        assert t[0] == pytest.approx(-1e-9)
        assert int(a["NumPoints"]) == codes.size
    assert np.max(np.abs(volts - v)) <= 1.0 / (2 * (2 ** 15 - 1)) / 2 * 1.01


def test_samples_are_stored_as_codes_and_not_as_volts(tmp_path):
    """Half the size, and the vertical scale becomes a property of the file. A float dataset here
    would read correctly in Python and not load on an instrument."""
    p = hdf5.write_hdf5(tmp_path / "r.h5", _signal(3_000), fs=FS)
    with h5py.File(p, "r") as f:
        d = f["Waveforms/Channel 1/Channel 1Data"]
        assert d.dtype == np.dtype("int16")
        assert d.chunks is not None
        assert d.maxshape == (None,)
        assert d.fillvalue == 0


def test_the_layout_is_the_one_a_reader_walks(tmp_path):
    p = hdf5.write_hdf5(tmp_path / "r.h5", _signal(3_000), fs=FS)
    with h5py.File(p, "r") as f:
        seen = []
        f.visititems(lambda k, o: seen.append(k))
        assert "Frame/TheFrame" in seen
        assert "Waveforms/Channel 1/Channel 1Data" in seen
        assert any(k.startswith("FileType/") for k in seen)
        assert int(f["Waveforms"].attrs["NumWaveforms"]) == 1
        # the axis and scale pair, plus the fields a reader expects to find
        a = set(f["Waveforms/Channel 1"].attrs)
        assert {"XInc", "XOrg", "XUnits", "YInc", "YOrg", "YUnits", "YReference",
                "NumPoints", "Count", "WaveformType", "NumSegments"} <= a
        assert {"DataType", "RawNumPts", "StartIndex", "InfoValid"} <= set(
            f["Waveforms/Channel 1/Channel 1Data"].attrs)


def test_several_channels_are_keyed_by_number(tmp_path):
    a, b = _signal(4_000, seed=1), _signal(4_000, seed=2)
    p = hdf5.write_hdf5(tmp_path / "r.h5", {1: a, 3: b}, fs=FS, full_scale=1.0)
    got = hdf5.read_hdf5(p)
    assert sorted(got) == [1, 3]
    with h5py.File(p, "r") as f:
        assert int(f["Waveforms"].attrs["NumWaveforms"]) == 2
        assert "Waveforms/Channel 3/Channel 3Data" in f
    half = 1.0 / (2 * (2 ** 15 - 1)) / 2 * 1.01
    assert np.max(np.abs(got[1][0] - a)) <= half
    assert np.max(np.abs(got[3][0] - b)) <= half


def test_a_shared_full_scale_gives_records_one_vertical_scale(tmp_path):
    """Without it the window follows each record's own range, which spends the whole code range on
    the signal but makes two records incomparable code-for-code."""
    small = 0.05 * _signal(2_000, seed=3)
    large = _signal(2_000, seed=3)
    ps = hdf5.write_hdf5(tmp_path / "s.h5", small, fs=FS, full_scale=1.0)
    pl = hdf5.write_hdf5(tmp_path / "l.h5", large, fs=FS, full_scale=1.0)
    with h5py.File(ps, "r") as f, h5py.File(pl, "r") as g:
        assert float(f["Waveforms/Channel 1"].attrs["YInc"]) == pytest.approx(
            float(g["Waveforms/Channel 1"].attrs["YInc"]))
    auto = hdf5.write_hdf5(tmp_path / "a.h5", small, fs=FS)
    with h5py.File(auto, "r") as f, h5py.File(pl, "r") as g:
        assert float(f["Waveforms/Channel 1"].attrs["YInc"]) < float(
            g["Waveforms/Channel 1"].attrs["YInc"])


def test_identity_is_copied_from_a_reference_rather_than_stated_here(tmp_path):
    """The marker that makes a file recognisable to a particular instrument is not physics and is
    not this library's to assert. This builds its own reference, copies the identity out of it,
    and checks the result is indistinguishable in those fields."""
    ref = tmp_path / "ref.h5"
    marker_name, marker_value = "SomeVendorFileType", b"SomeVendor Waveform Format 1.0"
    with h5py.File(ref, "w") as f:
        f.create_dataset(f"FileType/{marker_name}", data=np.bytes_(marker_value),
                         dtype=h5py.string_dtype("ascii", 40))
        f.create_dataset("Frame/TheFrame",
                         data=np.array((b"MODEL", b"SERIAL", b"01-Jan-2026 00:00:00"),
                                       dtype=hdf5.FRAME_DTYPE), dtype=hdf5.FRAME_DTYPE)

    ident = hdf5.identity_from(ref)
    assert ident["filetype_name"] == marker_name
    assert ident["filetype_value"] == marker_value

    out = hdf5.write_hdf5(tmp_path / "out.h5", _signal(2_000), fs=FS, like=ref)
    with h5py.File(out, "r") as f, h5py.File(ref, "r") as r:
        (name,) = list(f["FileType"])
        assert name == marker_name
        assert bytes(np.asarray(f["FileType"][name]).item()) == marker_value
        assert f["FileType"][name].dtype == r["FileType"][marker_name].dtype
        assert np.asarray(f["Frame/TheFrame"]).item() == np.asarray(r["Frame/TheFrame"]).item()


def test_without_a_reference_the_marker_is_neutral_and_the_file_still_reads(tmp_path):
    p = hdf5.write_hdf5(tmp_path / "r.h5", _signal(2_000), fs=FS)
    with h5py.File(p, "r") as f:
        assert list(f["FileType"]) == [hdf5.DEFAULT_FILETYPE_NAME]
    volts, t = hdf5.read_hdf5(p, channel=1)
    assert volts.size == 2_000 and t.size == 2_000


def test_a_reference_with_no_marker_is_reported(tmp_path):
    empty = tmp_path / "empty.h5"
    with h5py.File(empty, "w") as f:
        f.create_group("Waveforms")
    with pytest.raises(ValueError, match="no /FileType"):
        hdf5.identity_from(empty)


def test_a_constant_record_round_trips(tmp_path):
    p = hdf5.write_hdf5(tmp_path / "r.h5", np.full(1_000, 0.25), fs=FS)
    volts, _ = hdf5.read_hdf5(p, channel=1)
    assert np.allclose(volts, 0.25, atol=1e-6)


def test_an_empty_record_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no samples"):
        hdf5.write_hdf5(tmp_path / "r.h5", np.array([]), fs=FS)

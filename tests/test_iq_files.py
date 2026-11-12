"""I/Q recordings in open formats (GitHub #82): `wfmsynth/iqfile.py`.

Every test here asserts an EFFECT -- samples compared against the samples that were written,
events against the events that were written, a refusal against the bytes that provoke it.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import wfmsynth as ws
from wfmsynth import iqfile as F

FS, FC, N = 250e6, 1e9, 1 << 13


def _grid(n=N):
    return ws.Grid(fs=FS, baud=25e6, n=n, domain="iq", fc_hz=FC)


@pytest.fixture(scope="module")
def x():
    """A QAM-16 envelope, 10 samples per symbol: peak 1.96, so not a unit-amplitude toy."""
    return ws.Signal(seed=1, grid=_grid()).iq("qam16", prbs=15).waveform()


@pytest.fixture(scope="module")
def white():
    rng = np.random.default_rng(0)
    return 0.2 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))


def _fit(x, bits=16):
    """One count such that the record's peak sits exactly at full scale."""
    return float(max(np.abs(x.real).max(), np.abs(x.imag).max())) / ((1 << (bits - 1)) - 1)


# ------------------------------------------------------------------------------ round trips
def test_float32_round_trip_is_exact(tmp_path, x):
    x32 = x.astype(np.complex64).astype(complex)
    for order in ("little", "big"):
        p = tmp_path / f"a_{order}.iq"
        ws.to_interleaved(p, x32, dtype="float32", byteorder=order)
        back = ws.read_iq(p, format="iq_interleaved", dtype="float32", byteorder=order,
                          fs_hz=FS, fc_hz=FC).waveform()
        assert back.dtype == complex and np.array_equal(back, x32)


def test_float32_rounds_a_float64_record_by_at_most_one_part_in_2_to_the_24(tmp_path, x):
    p = tmp_path / "a.iq"
    ws.to_interleaved(p, x)
    back = ws.read_iq(p, format="iq_interleaved", dtype="float32", fs_hz=FS).waveform()
    err = max(np.abs(back.real - x.real).max(), np.abs(back.imag - x.imag).max())
    assert 0 < err <= 2.0 ** -24 * np.abs(x).max()


def test_float64_holds_the_record_exactly(tmp_path, x):
    p = tmp_path / "a.iq"
    ws.to_interleaved(p, x, dtype="float64")
    assert np.array_equal(
        ws.read_iq(p, format="iq_interleaved", dtype="float64", fs_hz=FS).waveform(), x)


@pytest.mark.parametrize("dtype,bits", [("int8", 8), ("int16", 16), ("int32", 32)])
def test_integer_round_trip_is_within_half_a_count(tmp_path, x, dtype, bits):
    s = _fit(x, bits)
    p = tmp_path / "a.iq"
    out = ws.to_interleaved(p, x, dtype=dtype, scale=s)
    assert out["scale"] == s and out["dtype"] == dtype
    sig = ws.read_iq(p, format="iq_interleaved", dtype=dtype, scale=s, fs_hz=FS, fc_hz=FC)
    back = sig.waveform()
    err = max(np.abs(back.real - x.real).max(), np.abs(back.imag - x.imag).max())
    assert err <= 0.5 * s * (1 + 1e-9)             # rint: half a count, so inside one LSB
    assert err > 0.4 * s                           # and the bound is tight, not slack
    # the file's samples themselves come back EXACTLY: a second trip changes nothing
    q = tmp_path / "b.iq"
    ws.to_interleaved(q, back, dtype=dtype, scale=s)
    assert p.read_bytes() == q.read_bytes()


def test_default_integer_scale_is_full_scale_one(tmp_path):
    raw = np.array([16384, -32768, 32767, 1] * 2, "<i2")
    p = tmp_path / "a.iq"
    raw.tofile(p)
    y = F.read(p, "iq_interleaved", dtype="int16")
    assert np.array_equal(y, (raw[0::2] + 1j * raw[1::2]) / 32768.0)
    assert F.default_scale("int16") == 1 / 32768 and F.default_scale("float32") == 1.0


def test_read_iq_takes_rate_length_and_centre_from_the_file(tmp_path, x):
    ws.to_sigmf(tmp_path / "rec", x, _grid())
    sig = ws.read_iq(tmp_path / "rec", format="sigmf")
    assert (sig.grid.fs, sig.grid.n, sig.grid.fc_hz, sig.grid.domain) == (FS, N, FC, "iq")
    assert sig.stored_grid(sig.waveform()) == sig.grid and sig.grid.baud is None
    # no file states a symbol rate; baud= puts one on the grid for the ops that read it
    with pytest.raises(TypeError, match="needs the symbol rate"):
        sig.filter("rrc", beta=0.35)
    timed = ws.read_iq(tmp_path / "rec", format="sigmf", baud=25e6)
    assert timed.grid.baud == 25e6 and len(timed.filter("rrc", beta=0.35).waveform()) == N


# ------------------------------------------------------------------------------------ SigMF
def _events(x):
    return ws.EventList([
        ws.Event(sample=100, kind="burst_start", width=1, t_s=100 / FS, placement="burst/2"),
        ws.Event(sample=4000, kind="glitch", width=37, severity=0.8, direction=-1.0, ui=400,
                 t_s=4000 / FS, placement="edges", params={"f0": 1.5e9, "tau": 2e-10}),
        ws.Event(sample=250, kind="ring", width=12, t_s=250 / FS, placement="symbols",
                 at="crosstalk/1/aggressor"),
    ], n=len(x))


def test_sigmf_events_come_back_as_the_same_events(tmp_path, x):
    ev = _events(x)
    ws.to_sigmf(tmp_path / "rec", x, _grid(), events=ev, dtype="float64")
    sig = ws.read_iq(tmp_path / "rec", format="sigmf")
    y, back = sig.realize()
    assert np.array_equal(y, x)
    want = sorted(ev.to_dicts(), key=lambda d: d["sample"])       # SigMF orders by start
    assert back.to_dicts() == want
    assert sig.event_list().to_dicts() == want
    assert back.mask().sum() == 1 + 37                            # the sub-chain event is not on the record


def test_sigmf_metadata_is_what_the_spec_requires(tmp_path, x):
    out = ws.to_sigmf(tmp_path / "rec.sigmf-meta", x, _grid(), events=_events(x),
                      dtype="int16", scale=_fit(x), description="qam16")
    assert out["meta"].endswith("rec.sigmf-meta") and out["data"].endswith("rec.sigmf-data")
    meta = json.loads(open(out["meta"]).read())
    assert set(meta) == {"global", "captures", "annotations"}
    g = meta["global"]
    assert g["core:datatype"] == "ci16_le" and g["core:version"] == "1.0.0"   # the two required
    assert g["core:sample_rate"] == FS and g["core:num_channels"] == 1
    assert g["core:description"] == "qam16"
    assert g["core:sha512"] == hashlib.sha512(open(out["data"], "rb").read()).hexdigest()
    # every namespace other than core: is declared, as the spec requires of an extension
    spaces = {k.split(":")[0] for o in [g, *meta["captures"], *meta["annotations"]] for k in o}
    assert spaces - {"core"} == {e["name"] for e in g["core:extensions"]} == {"wfmsynth"}
    assert meta["captures"] == [{"core:sample_start": 0, "core:frequency": FC}]
    starts = [a["core:sample_start"] for a in meta["annotations"]]
    assert starts == sorted(starts) == [100, 250, 4000]
    assert meta["annotations"][2]["core:sample_count"] == 37
    assert meta["annotations"][2]["core:label"] == "glitch"
    assert len(open(out["data"], "rb").read()) == 2 * 2 * N
    # the scale an integer recording needs travels with it
    back = ws.read_iq(tmp_path / "rec", format="sigmf").waveform()
    assert np.abs(back - x).max() <= _fit(x) * 0.5 * np.sqrt(2) * (1 + 1e-9)


def test_a_foreign_sigmf_annotation_becomes_an_event(tmp_path, x):
    ws.to_sigmf(tmp_path / "rec", x, _grid())
    meta = json.loads((tmp_path / "rec.sigmf-meta").read_text())
    meta["annotations"] = [
        {"core:sample_start": 512, "core:sample_count": 64, "core:label": "preamble",
         "core:comment": "from the lab", "core:freq_lower_edge": 0.99e9,
         "core:freq_upper_edge": 1.01e9},
        {"core:sample_start": 900}]
    (tmp_path / "rec.sigmf-meta").write_text(json.dumps(meta))
    a, b = ws.read_iq(tmp_path / "rec", format="sigmf").event_list().events
    assert (a.sample, a.width, a.kind, a.placement) == (512, 64, "preamble", "annotation")
    assert a.t_s == 512 / FS
    assert a.params == {"comment": "from the lab", "freq_lower_edge": 0.99e9,
                        "freq_upper_edge": 1.01e9}
    assert (b.sample, b.width, b.kind) == (900, None, "annotation")


def test_annotations_live_in_the_recipe_and_replay_without_the_metadata(tmp_path, x):
    ev = _events(x)
    ws.to_sigmf(tmp_path / "rec", x, _grid(), events=ev, dtype="float64")
    sig = ws.read_iq(tmp_path / "rec", format="sigmf", embed=True)
    recipe = json.loads(json.dumps(sig.recipe()))            # the recipe is plain data
    (tmp_path / "rec.sigmf-meta").unlink()
    (tmp_path / "rec.sigmf-data").unlink()
    again = ws.Signal.from_recipe(recipe)
    y, back = again.realize()
    assert np.array_equal(y, x) and again.sha256() == sig.sha256()
    assert back.to_dicts() == sorted(ev.to_dicts(), key=lambda d: d["sample"])


def test_a_resampled_capture_moves_its_annotations_with_the_samples(tmp_path, x):
    ws.to_sigmf(tmp_path / "rec", x, _grid(), dtype="float64", events=ws.EventList(
        [ws.Event(sample=4000, kind="mark", width=40)], n=N))
    g2 = ws.Grid(fs=2 * FS, n=2 * N, domain="iq", fc_hz=FC)
    sig = ws.Signal(grid=g2).capture(path=tmp_path / "rec", format="sigmf", resample=True)
    (e,) = sig.event_list().events
    assert (e.sample, e.width, e.t_s) == (8000, 80, 8000 / (2 * FS))


# ----------------------------------------------------------------------- digest and replay
def test_the_digest_covers_the_samples_not_the_format(tmp_path, x):
    """The same samples through four formats and two byte orders: one sha256."""
    ws.to_interleaved(tmp_path / "le.iq", x, dtype="float64")
    ws.to_interleaved(tmp_path / "be.iq", x, dtype="float64", byteorder="big")
    ws.to_sigmf(tmp_path / "rec", x, _grid(), dtype="float64")
    np.savetxt(tmp_path / "a.csv", np.c_[np.arange(N) / FS, x.real, x.imag], delimiter=",",
               fmt="%.17g")
    from scipy.io import savemat
    savemat(tmp_path / "a.mat", {"x": x, "other": np.arange(3.0)})
    np.save(tmp_path / "a.npy", x)
    sigs = [
        ws.read_iq(tmp_path / "le.iq", format="iq_interleaved", dtype="float64", fs_hz=FS, fc_hz=FC),
        ws.read_iq(tmp_path / "be.iq", format="iq_interleaved", dtype="float64", byteorder="big",
                   fs_hz=FS, fc_hz=FC),
        ws.read_iq(tmp_path / "rec", format="sigmf"),
        ws.read_iq(tmp_path / "a.csv", format="csv", fs_hz=FS, fc_hz=FC),
        ws.read_iq(tmp_path / "a.mat", format="mat", variable="x", fs_hz=FS, fc_hz=FC),
        ws.Signal(grid=_grid()).capture(path=tmp_path / "a.npy"),          # today's route
    ]
    digests = {s.ops[0]["sha256"] for s in sigs}
    assert digests == {ws.capture.digest(x)}
    for s in sigs:
        assert np.array_equal(s.waveform(), x)


def test_embed_drops_the_file_and_a_path_capture_notices_a_changed_one(tmp_path, x):
    p = tmp_path / "a.iq"
    ws.to_interleaved(p, x, dtype="int16", scale=_fit(x))
    kw = dict(format="iq_interleaved", dtype="int16", scale=_fit(x), fs_hz=FS, fc_hz=FC)
    by_path, embedded = ws.read_iq(p, **kw), ws.read_iq(p, embed=True, **kw)
    assert by_path.ops[0]["path"] == str(p) and "values" not in by_path.ops[0]
    assert "path" not in embedded.ops[0] and embedded.ops[0]["sha256"] == by_path.ops[0]["sha256"]
    for key in ("format", "dtype", "scale", "fs_hz", "fc_hz"):   # how it was read stays on record
        assert embedded.ops[0][key] == by_path.ops[0][key]
    want = by_path.waveform()
    raw = np.fromfile(p, "<i2")
    raw[10] += 1
    raw.tofile(p)
    with pytest.raises(ValueError, match="does not match the recipe"):
        by_path.waveform()
    p.unlink()
    assert np.array_equal(embedded.waveform(), want)
    assert np.array_equal(ws.Signal.from_recipe(json.loads(json.dumps(embedded.recipe())))
                          .waveform(), want)


def test_csv_columns_by_header_name_and_by_number(tmp_path, x):
    p = tmp_path / "named.csv"
    np.savetxt(p, np.c_[x.imag, np.arange(N), x.real], delimiter=",", fmt="%.17g",
               header="Q,sample,I", comments="")
    by_name = ws.read_iq(p, format="csv", columns=("I", "Q"), fs_hz=FS)
    assert np.array_equal(by_name.waveform(), x) and by_name.ops[0]["columns"] == ["I", "Q"]
    assert np.array_equal(ws.read_iq(p, format="csv", columns=(2, 0), fs_hz=FS).waveform(), x)
    with pytest.raises(ValueError, match=r"no column named 'i'.*\['Q', 'sample', 'I'\]"):
        ws.read_iq(p, format="csv", columns=("i", "Q"), fs_hz=FS)
    with pytest.raises(ValueError, match="3 column"):
        ws.read_iq(p, format="csv", columns=(1, 3), fs_hz=FS)


def test_mat_reads_a_named_complex_variable(tmp_path, x):
    from scipy.io import savemat
    p = tmp_path / "a.mat"
    savemat(p, {"x": x.reshape(1, -1), "r": x.real})
    assert np.array_equal(ws.read_iq(p, format="mat", variable="x", fs_hz=FS).waveform(), x)
    with pytest.raises(ValueError, match=r"needs variable=.*\['x', 'r'\]|\['r', 'x'\]"):
        ws.read_iq(p, format="mat", fs_hz=FS)
    with pytest.raises(ValueError, match="'r' in .* is real"):
        ws.read_iq(p, format="mat", variable="r", fs_hz=FS)


# ------------------------------------------------------------------- the byte-order check
def _swapped(tmp_path, v, dtype, scale=None, name="sw.iq"):
    """Write `v` big-endian; the caller then reads it as little-endian."""
    p = tmp_path / name
    ws.to_interleaved(p, v, dtype=dtype, byteorder="big", scale=scale)
    return p


@pytest.mark.parametrize("level", [1.0, 0.5, 0.1, 0.01, 0.001])
@pytest.mark.parametrize("dtype,bits", [("int16", 16), ("int32", 32)])
def test_a_byte_swapped_integer_file_is_refused_at_any_level(tmp_path, x, dtype, bits, level):
    s = _fit(x, bits) / level
    p = _swapped(tmp_path, x, dtype, s)
    with pytest.raises(ValueError, match="probably big-endian") as e:
        ws.read_iq(p, format="iq_interleaved", dtype=dtype, scale=s, fs_hz=FS)
    assert "check=False" in str(e.value) and str(p) in str(e.value)
    # the same bytes read the right way are accepted, and so is the right file read as stated
    ws.read_iq(p, format="iq_interleaved", dtype=dtype, byteorder="big", scale=s, fs_hz=FS)
    q = tmp_path / "ok.iq"
    ws.to_interleaved(q, x, dtype=dtype, scale=s)
    ws.read_iq(q, format="iq_interleaved", dtype=dtype, scale=s, fs_hz=FS)


@pytest.mark.parametrize("level", [1.0, 0.5, 0.1, 0.01, 0.001])
def test_a_byte_swapped_noise_file_is_refused_at_any_level(tmp_path, white, level):
    """White samples have no smoothness to lose, so this is the entropy sign alone."""
    s = _fit(white) / level
    p = _swapped(tmp_path, white, "int16", s)
    with pytest.raises(ValueError, match="bits of entropy.*probably big-endian"):
        ws.read_iq(p, format="iq_interleaved", dtype="int16", scale=s, fs_hz=FS)
    q = tmp_path / "ok.iq"
    ws.to_interleaved(q, white, dtype="int16", scale=s)
    ws.read_iq(q, format="iq_interleaved", dtype="int16", scale=s, fs_hz=FS)


def test_what_the_check_cannot_see_is_as_stated(tmp_path):
    """The documented limit: white samples UNIFORM over the whole integer range read the same
    both ways. Pinned so the docstring's claim cannot quietly become false."""
    rng = np.random.default_rng(3)
    raw = rng.integers(-32768, 32768, 2 * N).astype("<i2")
    assert F.byte_order_evidence(raw, raw.dtype) is None
    assert F.byte_order_evidence(raw.byteswap(), raw.dtype) is None


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_byte_swapped_float_file_is_refused(tmp_path, x, dtype):
    p = _swapped(tmp_path, x, dtype)
    with pytest.raises(ValueError, match="not finite.*probably big-endian"):
        ws.read_iq(p, format="iq_interleaved", dtype=dtype, fs_hz=FS)
    # values that are whole numbers swap to subnormals, never to NaN: the magnitude sign
    q = _swapped(tmp_path, np.round(4 * x), dtype, name="whole.iq")
    with pytest.raises(ValueError, match="median nonzero magnitude.*probably big-endian"):
        ws.read_iq(q, format="iq_interleaved", dtype=dtype, fs_hz=FS)


def test_an_integer_file_read_as_float_is_refused_as_a_dtype_problem(tmp_path, x):
    p = tmp_path / "a.iq"
    ws.to_interleaved(p, x, dtype="int16", scale=_fit(x) * 100)
    with pytest.raises(ValueError, match="dtype is probably wrong"):
        ws.read_iq(p, format="iq_interleaved", dtype="float32", fs_hz=FS)


def test_check_false_reads_the_file_as_stated(tmp_path, x):
    s = _fit(x)
    p = _swapped(tmp_path, x, "int16", s)
    sig = ws.read_iq(p, format="iq_interleaved", dtype="int16", scale=s, fs_hz=FS, check=False)
    assert sig.ops[0]["check"] is False                       # on record, and honoured on replay
    raw = np.fromfile(p, "<i2").astype(float) * s
    assert np.array_equal(ws.Signal.from_recipe(sig.recipe()).waveform(), raw[0::2] + 1j * raw[1::2])


def test_honest_files_the_check_must_leave_alone(tmp_path, x, white):
    """False alarms are the cost of a check. These are the shapes most likely to raise one."""
    cases = {
        "left-justified 8-bit": (np.rint(x.real / _fit(x, 8)).astype("<i2") << 8),
        "left-justified 12-bit noise": (np.rint(white.real / _fit(white, 12)).astype("<i2") << 4),
        "two-level": np.where(x.real > 0, 23170, -23170).astype("<i2"),
        "full-scale ramp": (np.arange(N) * 8 % 65536 - 32768).astype("<i2"),
        "one count of noise": np.rint(white.real * 4).astype("<i2"),
        "all zero": np.zeros(N, "<i2"),
        "coarse steps": (np.rint(white.real * 30) * 1001).astype("<i2"),
    }
    for name, i in cases.items():
        raw = np.empty(2 * len(i), "<i2")
        raw[0::2], raw[1::2] = i, i[::-1]
        assert F.byte_order_evidence(raw, raw.dtype) is None, name
    for dtype in ("float32", "float64"):
        tail = (np.exp(-np.arange(2 * N) / 40.0) + 1e-3).astype(dtype)      # 0.001 .. 1
        assert F.byte_order_evidence(tail, tail.dtype) is None


def test_a_file_too_short_to_judge_is_read_as_stated(tmp_path, x):
    s = _fit(x)
    p = _swapped(tmp_path, x[:F.MIN_CHECKED // 2 - 1], "int16", s)
    ws.read_iq(p, format="iq_interleaved", dtype="int16", scale=s, fs_hz=FS)     # not refused
    p = _swapped(tmp_path, x[:F.MIN_CHECKED // 2], "int16", s)
    with pytest.raises(ValueError, match="probably big-endian"):
        ws.read_iq(p, format="iq_interleaved", dtype="int16", scale=s, fs_hz=FS)


def test_describe_shows_the_level_a_wrong_scale_would_hide(tmp_path, x):
    s = 2 * _fit(x)
    p = tmp_path / "a.iq"
    ws.to_interleaved(p, x, dtype="int16", scale=s)
    d = F.describe(p, "iq_interleaved", dtype="int16", scale=s)
    assert (d["n"], d["dtype"], d["byteorder"], d["scale"]) == (N, "int16", "little", s)
    assert d["peak_counts"] == 16384 and abs(d["full_scale_fraction"] - 0.5) < 1e-4
    assert abs(d["peak"] - np.abs(x).max()) < s and d["fs_hz"] is None
    ws.to_sigmf(tmp_path / "rec", x, _grid(), events=_events(x))
    d = F.describe(tmp_path / "rec", "sigmf")
    assert (d["fs_hz"], d["fc_hz"], d["dtype"], d["annotations"], d["peak_counts"]) == \
        (FS, FC, "float32", 3, None)


# -------------------------------------------------------------------------------- markers
def test_the_marker_channel_is_one_byte_per_sample_over_each_event(tmp_path, x):
    out = ws.to_interleaved(tmp_path / "a.iq", x, events=_events(x))
    assert out["markers"] == str(tmp_path / "a.iq") + ".markers"
    m = np.fromfile(out["markers"], np.uint8)
    want = np.zeros(N, np.uint8)
    want[100] = 1
    want[4000:4037] = 1                       # the event inside a sub-chain (sample 250) is not here
    assert np.array_equal(m, want)
    assert ws.to_interleaved(tmp_path / "b.iq", x)["markers"] is None
    assert not (tmp_path / "b.iq.markers").exists()


# -------------------------------------------------------------------------------- refusals
def test_export_refuses_to_clip_and_names_the_scale_that_fits(tmp_path, x):
    with pytest.raises(ValueError, match=r"beyond int16's \+/-32767.*pass scale=") as e:
        ws.to_interleaved(tmp_path / "a.iq", x, dtype="int16")      # peak 1.96 at full scale 1
    fit = float(str(e.value).split("pass scale=")[1].split()[0])
    out = ws.to_interleaved(tmp_path / "a.iq", x, dtype="int16", scale=fit)
    assert np.abs(np.fromfile(out["path"], "<i2")).max() == 32767
    with pytest.raises(ValueError, match="beyond int16"):
        ws.to_sigmf(tmp_path / "rec", x, _grid(), dtype="int16")


def test_export_refusals(tmp_path, x):
    with pytest.raises(ValueError, match="record is real.*downconvert"):
        ws.to_interleaved(tmp_path / "a.iq", x.real)
    with pytest.raises(ValueError, match="stored_grid"):
        ws.to_sigmf(tmp_path / "rec", x[:100], _grid())
    with pytest.raises(ValueError, match="dtype='uint8' is not one of"):
        ws.to_interleaved(tmp_path / "a.iq", x, dtype="uint8")
    with pytest.raises(ValueError, match="scale=-1 must be a positive"):
        ws.to_interleaved(tmp_path / "a.iq", x, dtype="int16", scale=-1)
    with pytest.raises(ValueError, match="byteorder='<' is not 'little' or 'big'"):
        ws.to_interleaved(tmp_path / "a.iq", x, byteorder="<")


def test_layout_refusals(tmp_path, x):
    p = tmp_path / "a.iq"
    ws.to_interleaved(p, x)
    ws.to_sigmf(tmp_path / "rec", x, _grid())
    g = _grid()
    with pytest.raises(ValueError, match="needs dtype="):
        ws.Signal(grid=g).capture(path=p, format="iq_interleaved")
    with pytest.raises(ValueError, match="unknown format 'iq'.*iq_interleaved, sigmf, csv, mat"):
        ws.Signal(grid=g).capture(path=p, format="iq")
    with pytest.raises(ValueError, match="'sigmf' does not read dtype='int16'.*states its own"):
        ws.Signal(grid=g).capture(path=tmp_path / "rec", format="sigmf", dtype="int16")
    with pytest.raises(ValueError, match="'iq_interleaved' does not read variable='x'.*'mat'"):
        ws.Signal(grid=g).capture(path=p, format="iq_interleaved", dtype="float32", variable="x")
    with pytest.raises(ValueError, match="unknown parameter 'endian'.*byteorder"):
        ws.Signal(grid=g).capture(path=p, format="iq_interleaved", dtype="float32", endian="big")
    with pytest.raises(ValueError, match="describe an I/Q FILE.*format="):
        ws.Signal(grid=g).capture(path=p, dtype="float32")
    with pytest.raises(ValueError, match="describe an I/Q FILE"):
        ws.Signal(grid=g).capture(values=x, fc_hz=FC)
    with pytest.raises(ValueError, match="dtype='uint16'.*offset binary"):
        ws.Signal(grid=g).capture(path=p, format="iq_interleaved", dtype="uint16")
    (tmp_path / "odd.iq").write_bytes(b"\0" * 10)
    with pytest.raises(ValueError, match="10 bytes.*not a whole number of I/Q pairs of int16"):
        ws.read_iq(tmp_path / "odd.iq", format="iq_interleaved", dtype="int16", fs_hz=FS)
    with pytest.raises(FileNotFoundError, match="nope.iq"):
        ws.read_iq(tmp_path / "nope.iq", format="iq_interleaved", dtype="int16", fs_hz=FS)
    with pytest.raises(ValueError, match="does not state its sample rate; pass fs_hz="):
        ws.read_iq(p, format="iq_interleaved", dtype="float32")


def test_grid_refusals(tmp_path, x):
    p = tmp_path / "a.iq"
    ws.to_interleaved(p, x)
    kw = dict(path=p, format="iq_interleaved", dtype="float32")
    with pytest.raises(ValueError, match="complex .I/Q. samples.*domain='iq'"):
        ws.Signal(grid=ws.Grid(fs=FS, n=N)).capture(**kw)
    with pytest.raises(ValueError, match="centred on fc_hz=2e\\+09 and the record's grid says "
                                         "fc_hz=1e\\+09"):
        ws.Signal(grid=_grid()).capture(fc_hz=2e9, **kw)
    with pytest.raises(ValueError, match="grid says no centre frequency"):
        ws.Signal(grid=ws.Grid(fs=FS, n=N, domain="iq")).capture(fc_hz=FC, **kw)
    with pytest.raises(ValueError, match="fs_hz=1e\\+08 does not match the record's fs"):
        ws.Signal(grid=_grid()).capture(fs_hz=1e8, **kw).waveform()
    with pytest.raises(ValueError, match="do not match the record length"):
        ws.Signal(grid=_grid(N // 2)).capture(**kw).waveform()
    # a hand-written recipe is held to the same things at render
    r = ws.Signal(grid=_grid()).capture(fc_hz=FC, **kw).recipe()
    r["grid"]["fc_hz"] = 2e9
    with pytest.raises(ValueError, match="centred on fc_hz=1e\\+09"):
        ws.Signal.from_recipe(r).waveform()


def _edit_meta(tmp_path, x, edit):
    ws.to_sigmf(tmp_path / "rec", x, _grid())
    meta = json.loads((tmp_path / "rec.sigmf-meta").read_text())
    edit(meta)
    (tmp_path / "rec.sigmf-meta").write_text(json.dumps(meta))
    return tmp_path / "rec"


@pytest.mark.parametrize("edit,match", [
    (lambda m: m["global"].update({"core:datatype": "rf32_le"}), "REAL recording"),
    (lambda m: m["global"].update({"core:datatype": "cu8"}), "unsigned samples.*offset binary"),
    (lambda m: m["global"].update({"core:datatype": "cf32"}), "no byte order"),
    (lambda m: m["global"].update({"core:datatype": "complex64"}), "not a SigMF datatype"),
    (lambda m: m["global"].pop("core:datatype"), "is not SigMF metadata"),
    (lambda m: m["global"].update({"core:num_channels": 2}), "2 channels"),
    (lambda m: m["global"].update({"core:sha512": "0" * 128}), "does not match the core:sha512"),
    (lambda m: m["global"].update({"core:trailing_bytes": 8}), "trailing bytes"),
    (lambda m: m["captures"].append({"core:sample_start": 100}), "2 capture segment"),
    (lambda m: m["captures"][0].update({"core:header_bytes": 16}), "header bytes"),
    (lambda m: m["annotations"].append({"core:label": "x"}), "no core:sample_start"),
])
def test_sigmf_this_reader_cannot_honour_is_refused_by_name(tmp_path, x, edit, match):
    with pytest.raises(ValueError, match=match):
        ws.read_iq(_edit_meta(tmp_path, x, edit), format="sigmf")


def test_sigmf_arguments_that_contradict_the_metadata_are_refused(tmp_path, x):
    ws.to_sigmf(tmp_path / "rec", x, _grid())
    with pytest.raises(ValueError, match="fs_hz=1e\\+08 was passed and the file's metadata says"):
        ws.read_iq(tmp_path / "rec", format="sigmf", fs_hz=1e8)
    with pytest.raises(ValueError, match="fc_hz=2e\\+09 was passed and the SigMF metadata says"):
        ws.Signal(grid=_grid()).capture(path=tmp_path / "rec", format="sigmf", fc_hz=2e9)
    (tmp_path / "rec.sigmf-data").unlink()
    with pytest.raises(FileNotFoundError, match="rec.sigmf-data"):
        ws.read_iq(tmp_path / "rec", format="sigmf")
    with pytest.raises(FileNotFoundError, match="base name"):
        ws.read_iq(tmp_path / "other", format="sigmf")


# ------------------------------------------------------------------- it is still a chain
def test_a_file_capture_inside_a_chain_keeps_the_events_and_takes_later_ops(tmp_path, x):
    ws.to_sigmf(tmp_path / "rec", x, _grid(), events=_events(x), dtype="float64")
    sig = ws.read_iq(tmp_path / "rec", format="sigmf").filter("butter", f_hz=40e6, order=3)
    y, ev = sig.realize()
    assert np.iscomplexobj(y) and len(ev) == 3 and not np.array_equal(y, x)
    assert sig.stage_kinds(domains=True)[0] == ("source", "iq")
    assert sig.roles() == []                              # a capture draws no randomness


def test_existing_capture_recipes_are_untouched(tmp_path, x):
    """The op a plain capture records has exactly the keys it had before this module."""
    np.save(tmp_path / "a.npy", x)
    np.save(tmp_path / "r.npy", x.real)
    assert set(ws.Signal(grid=_grid()).capture(path=tmp_path / "a.npy").ops[0]) == \
        {"op", "path", "sha256"}
    assert set(ws.Signal(grid=ws.Grid(fs=FS, n=N)).capture(path=tmp_path / "r.npy", fs_hz=FS,
                                                           embed=True).ops[0]) == \
        {"op", "values", "sha256", "fs_hz"}

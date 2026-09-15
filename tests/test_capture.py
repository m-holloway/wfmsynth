"""Phase 1's other new source: importing a real capture from disk (`Signal.capture`).

A recipe holding a `path=` is only as reproducible as the file next to it, so this pins the
two properties that make that an honest trade rather than a silent one: a missing/changed file
is a NAMED error (not zeros, not different samples under the same name), and `embed=True` /
`values=[...]` gets a consumer off the file dependency entirely -- the same trade
`Signal.pattern(embed=True)` already makes for a synthesized source.
"""
from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pytest

from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

FS = 1e9


def _npy(tmp_path, y, name="cap.npy"):
    p = tmp_path / name
    np.save(p, np.asarray(y, dtype=np.float32))     # on-disk dtype need not be float64
    return p


def _csv(tmp_path, y, fs=FS, name="cap.csv"):
    t = np.arange(len(y)) / fs
    p = tmp_path / name
    np.savetxt(p, np.column_stack([t, y]), delimiter=",")
    return p


# --------------------------------------------------------------------------------- basic load
def test_capture_from_npy_round_trips_the_values(tmp_path):
    y = np.sin(np.linspace(0, 6.28, 256))
    p = _npy(tmp_path, y)
    sig = Signal(seed=1).capture(path=str(p))
    got = sig.waveform()
    assert got.shape == y.shape
    assert np.allclose(got, y, atol=1e-6)          # float32 round-trip tolerance


def test_capture_from_two_column_csv(tmp_path):
    y = np.linspace(-1, 1, 128)
    p = _csv(tmp_path, y)
    got = Signal(seed=1).capture(path=str(p)).waveform()
    assert np.allclose(got, y, atol=1e-6)


def test_capture_from_literal_values_needs_no_file():
    y = [0.1, 0.2, -0.3, 0.4] * 8
    sig = Signal(seed=1, grid=Grid(fs=FS, n=len(y))).capture(values=y, fs_hz=FS)
    got = sig.waveform()
    assert np.allclose(got, y)
    assert sig.recipe()["ops"][0].get("path") is None


# --------------------------------------------------------------------------------- honesty gates
def test_a_missing_file_is_a_named_error_not_zeros():
    # the digest is taken from the file's CURRENT bytes at authoring time (that is what makes
    # a later mismatch meaningful), so a missing file fails as soon as it is named, not lazily
    # deferred to render time.
    with pytest.raises(Exception, match="does/not/exist"):
        Signal(seed=1).capture(path="/does/not/exist/at/all.npy")


def test_a_changed_file_is_a_named_error_not_silently_different_samples(tmp_path):
    y = np.linspace(0, 1, 64)
    p = _npy(tmp_path, y)
    sig = Signal(seed=1).capture(path=str(p), embed=False)
    r = sig.recipe()
    assert "sha256" in r["ops"][0]
    # overwrite the file with different bytes under the same path/name
    np.save(p, y * 2.0)
    with pytest.raises(Exception, match="sha256|changed|does not match"):
        Signal.from_recipe(r).waveform()


def test_embed_true_reads_the_file_now_and_drops_the_path_dependency(tmp_path):
    y = np.linspace(0, 1, 64)
    p = _npy(tmp_path, y)
    sig = Signal(seed=1).capture(path=str(p), embed=True)
    op = sig.recipe()["ops"][0]
    assert op.get("path") is None
    assert "values" in op
    # the file can now disappear entirely and the recipe still replays
    p.unlink()
    got = Signal.from_recipe(sig.recipe()).waveform()
    assert np.allclose(got, y, atol=1e-6)


def test_capture_records_a_digest_over_the_sample_values_not_the_file_bytes(tmp_path):
    """Saving the SAME values through a different codec/dtype must not move the digest --
    only the values matter, per docs/ARCHITECTURE.md's archive-digest rule."""
    # exact in both float32 and float64, so a real precision loss cannot masquerade as the
    # digest function being wrong
    y = np.array([0.0, 0.5, -0.25, 0.125, -1.0, 1.0] * 20)
    p1 = tmp_path / "a.npy"
    np.save(p1, y.astype(np.float32))
    p2 = tmp_path / "b.npy"
    np.save(p2, y.astype(np.float64))
    d1 = Signal(seed=1).capture(path=str(p1)).recipe()["ops"][0]["sha256"]
    d2 = Signal(seed=1).capture(path=str(p2)).recipe()["ops"][0]["sha256"]
    assert d1 == d2


# --------------------------------------------------------------------------------- grid/length
def test_length_mismatch_without_resample_is_a_named_error(tmp_path):
    y = np.zeros(100)
    p = _npy(tmp_path, y)
    g = Grid(fs=FS, n=50)
    with pytest.raises(Exception, match="resample"):
        Signal(seed=1, grid=g).capture(path=str(p)).waveform()


def test_resample_true_actually_resamples_onto_the_grid(tmp_path):
    y = np.sin(np.linspace(0, 20 * np.pi, 1000))
    p = _npy(tmp_path, y)
    g = Grid(fs=FS, n=500)
    got = Signal(seed=1, grid=g).capture(path=str(p), resample=True).waveform()
    assert len(got) == 500


# --------------------------------------------------------------------------------- composability
def test_fs_hz_without_a_matching_grid_warns():
    """Found by dogfooding the skill on a fresh agent (2026-09): `fs_hz=` is provenance and
    what `resample=True` resamples FROM; it does not make a downstream Hz-denominated knob
    (`lossy(loss_at_ghz=...)`, `probe(bw_hz=...)`) mean anything real, because those read
    `grid.fs`. A chain with no Grid at all renders against no real rate, silently -- so the op
    warns when `fs_hz=` is given but there is nothing for it to reconcile with."""
    y = [0.1, 0.2, -0.3, 0.4] * 8
    with pytest.warns(RuntimeWarning, match="fs_hz"):
        Signal(seed=1).capture(values=y, fs_hz=FS).waveform()


def test_fs_hz_with_a_matching_grid_does_not_warn():
    y = [0.1, 0.2, -0.3, 0.4] * 8
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Signal(seed=1, grid=Grid(fs=FS, n=len(y))).capture(values=y, fs_hz=FS).waveform()
    assert not any("fs_hz" in str(w.message) for w in caught)


def test_no_fs_hz_given_does_not_warn():
    y = [0.1, 0.2, -0.3, 0.4] * 8
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Signal(seed=1).capture(values=y).waveform()
    assert not any("fs_hz" in str(w.message) for w in caught)


def test_capture_is_a_source_that_chains_like_any_other():
    y = [0.0, 1.0] * 64
    sig = (Signal(seed=1, grid=Grid(fs=FS, n=len(y))).capture(values=y, fs_hz=FS)
           .lossy(loss_db=6.0, loss_at_ghz=1.0)
           .events("glitch", on="poisson", rate_hz=1e6))
    x = sig.waveform()
    assert x.size == len(y) and np.all(np.isfinite(x))
    assert [o["op"] for o in sig.recipe()["ops"]] == ["capture", "lossy", "events"]


def test_capture_recipe_round_trips_bit_identical(tmp_path):
    y = np.random.default_rng(0).standard_normal(200)
    p = _npy(tmp_path, y)
    sig = Signal(seed=2).capture(path=str(p)).lossy(loss_db=3.0, loss_at_ghz=2.0)
    x = sig.waveform()
    x2 = Signal.from_recipe(sig.recipe()).waveform()
    assert np.array_equal(x, x2)


def test_capture_emits_only_the_arguments_it_was_given(tmp_path):
    y = np.zeros(10)
    p = _npy(tmp_path, y)
    op = Signal(seed=1).capture(path=str(p)).recipe()["ops"][0]
    assert set(op) == {"op", "path", "sha256"}


def test_capture_has_no_public_i2c_uart_or_iso_names():
    import wfmsynth as ws
    for bad in ("i2c", "uart", "iso7637"):
        assert not hasattr(ws, bad)
        assert not hasattr(Signal, bad)


# --------------------------------------------------------------------------------- lead-in / gates
def test_capture_is_classified_for_the_lead_in():
    from wfmsynth import compose as C
    assert "capture" in C._EXEC and "capture" in C.OP_KIND
    assert C.OP_KIND["capture"] == "source"
    assert (("capture" in C._LEAD_LTI) + ("capture" in C._LEAD_SKIP)
            + ("capture" in C._LEAD_REJECT) + ("capture" in C._LEAD_ANALYTIC)) == 1


def test_capture_refuses_a_lead_in_rather_than_inventing_history():
    y = [0.0, 1.0] * 64
    sig = Signal(seed=1).capture(values=y, fs_hz=FS).with_lead_in(True)
    with pytest.raises(ValueError, match="capture"):
        sig.waveform()

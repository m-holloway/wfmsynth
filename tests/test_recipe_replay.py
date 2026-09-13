"""The standalone recipe replayer: a recipe document renders back to the same samples.

The value of this suite is the ROUND TRIP. A recipe is only ground truth if something
independent of the builder can turn it back into the identical record, so these tests build a
Signal in-process, serialise nothing but its ops/grid/seed into a document, hand that document
to the replayer as a consumer would, and demand bit-identity.

The first test is a CALIBRATION, deliberately placed first: it checks the replayer's
pattern-lowering path (carrier op -> explicit symbols op) against a pattern the kernel can
already render natively, so the lowering is proven transparent on a KNOWN answer before any
test trusts it on a registry pattern that has no independent reference.
"""
import ast
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import wfmsynth as ws                                                   # noqa: E402

_REPLAY_PY = os.path.join(_ROOT, "examples", "replay.py")


def _replay_module():
    spec = importlib.util.spec_from_file_location("_replay_under_test", _REPLAY_PY)
    mod = importlib.util.module_from_spec(spec)
    # registered before exec: `dataclass` resolves annotations through sys.modules
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _replay_module()


# ----------------------------------------------------------------- fixtures / helpers
def _grid():
    return ws.Grid(fs=50e9, baud=10e9, n=4096)


def _signal():
    """A chain with one op from every link of the chain: source (+ source jitter), shape,
    channel, instrument. Jitter and ADC noise are in it on purpose — a replay that did not
    thread the seeded streams identically would differ in exactly those ops."""
    g = _grid()
    return (ws.Signal(seed=11, grid=g)
            .carrier("nrz", n_ui=512, pattern="prbs9", tr_frac=0.3, causal=True,
                     jitter=dict(rj=0.01, dcd=0.02))
            .tx_ffe([-0.1, 1.0, -0.15], pre=1)
            .lossy(loss_db=8.0, loss_at_ghz=5.0, causal=True)
            .reflect(td_ps=90.0, gamma_s=0.15)
            .digitize(snr_db=34.0, enob=6.0))


def _record(sig, digest=True):
    r = sig.recipe()
    rec = {"ops": r["ops"], "grid": r["grid"], "seed": r["seed"]}
    if digest:
        rec["values_sha256"] = R.values_digest(sig.waveform())
    return rec


def _write(tmp_path, doc, name="recipe.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


# ----------------------------------------------------------------- 1. calibration
def test_a_symbols_op_renders_exactly_what_the_carrier_op_renders():
    """KNOWN-ANSWER CALIBRATION, and it is the measurement every pattern test below leans on.
    A registry pattern reaches the waveform through the `symbols` op; a pattern in the kernel's
    own vocabulary reaches it through the `carrier` op. In the kernel those are the same two
    lines (`carrier_symbols` then `from_symbols`), so for a pattern BOTH can express they must
    be bit-identical. If they are not, no pattern render can be trusted, because a registry
    pattern has no native twin to diff against.

    n_ui=256 on a 4096-point record is 16 samples/UI, so tr_frac=0.4 asks for 6.4 samples --
    clear of the 2-sample rise-time floor. At 8 samples/UI both this value and the kernel's
    default clamp to the floor, and the test cannot see a dropped tr_frac at all.
    """
    g = _grid()
    shape = {"tr_frac": 0.4, "causal": True, "jitter": {"rj": 0.01}, "n": g.n}
    gridspec = {"fs": g.fs, "baud": g.baud, "n": g.n}
    a = R.render({"ops": [{"op": "carrier", "kind": "nrz", "n_ui": 256, "seed": 5,
                           "pattern": "prbs9", **shape}], "grid": gridspec, "seed": 3})
    syms = ws.physics.carrier_symbols("nrz", 256, seed=5, pattern="prbs9")
    b = R.render({"ops": [{"op": "symbols", "symbols": [float(v) for v in syms], **shape}],
                  "grid": gridspec, "seed": 3})
    assert np.array_equal(a, b)


# ----------------------------------------------------------------- 1b. the pattern registry
# A conditional import, not `importorskip`: that would skip this WHOLE module on a build with no
# registry, and the round-trip tests above are the ones that must never be skipped.
try:
    import wfmsynth.patterns as PAT
except ImportError:                                                     # pragma: no cover
    PAT = None
_needs_registry = pytest.mark.skipif(PAT is None or not hasattr(ws.Signal, "pattern"),
                                     reason="this build has no pattern registry / pattern op")


@pytest.fixture
def registry_sandbox():
    """Leave the registry as it was found: a cleared or leaked entry would make the
    missing-entry cases below pass for the wrong reason."""
    saved = dict(PAT.REGISTRY)
    yield PAT.REGISTRY
    PAT.REGISTRY.clear()
    PAT.REGISTRY.update(saved)


@_needs_registry
def test_a_named_registry_pattern_replays_through_the_registry(tmp_path):
    """The replayer resolves a named pattern by executing the op, which is the composer's own
    `patterns.replay` — deliberately NOT a second lookup of its own."""
    sig = (ws.Signal(seed=4, grid=_grid())
           .pattern("prbs13", length=512, seed=9, tr_frac=0.3, causal=True)
           .lossy(loss_db=9.0, loss_at_ghz=5.0, causal=True))
    doc = {"records": {"std/prbs13/r0": _record(sig)}}
    rep = R.replay(R.load_document(_write(tmp_path, doc)), dry_run=True)
    assert rep.exit_code == 0, rep.lines()
    assert rep.results[0].pattern.startswith("prbs13")


@_needs_registry
def test_an_embedded_pattern_replays_with_the_registry_emptied(tmp_path, registry_sandbox):
    """THE STANDALONE CLAIM. A document whose patterns are embedded is replayable with this
    library and no generator code at all — which is the only guarantee that survives being
    handed to someone who will never have the registry entry."""
    sig = ws.Signal(seed=4, grid=_grid()).pattern("prbs9", length=256, seed=4, embed=True)
    doc = {"records": {"std/prbs9/r0": _record(sig)}}
    path = _write(tmp_path, doc)
    registry_sandbox.clear()
    rep = R.replay(R.load_document(path), dry_run=True)
    assert rep.exit_code == 0, rep.lines()
    assert rep.n_verified == 1
    assert not rep.results[0].needs_entry


@_needs_registry
def test_a_pattern_whose_entry_is_gone_fails_by_record_id_and_names_the_pattern(
        tmp_path, registry_sandbox):
    """The defect worth catching is the same NAME resolving to different samples elsewhere. The
    replayer must turn that into a per-record error that says what is needed."""
    @PAT.pattern("fixture_word", levels=2, period=4, source="not from a standard: a fixture")
    def _fixture_word(length, seed=1):
        return PAT.block_repeat([1.0, 1.0, -1.0, -1.0], length=length)

    sig = ws.Signal(seed=1, grid=_grid()).pattern("fixture_word", length=256)
    doc = {"records": {"ok/embedded": _record(
               ws.Signal(seed=1, grid=_grid()).pattern("fixture_word", length=256, embed=True)),
           "needs/an/entry": _record(sig)}}
    path = _write(tmp_path, doc)
    del registry_sandbox["fixture_word"]
    rep = R.replay(R.load_document(path), dry_run=True)
    assert rep.exit_code != 0
    by_id = {r.record_id: r for r in rep.results}
    assert by_id["ok/embedded"].status == "ok"                 # embedded needs no entry
    assert by_id["needs/an/entry"].status == "ERROR"
    assert "fixture_word" in by_id["needs/an/entry"].detail


@_needs_registry
def test_a_document_depending_on_an_installed_entry_says_so_even_when_it_passes(
        tmp_path, registry_sandbox):
    """A dependency that fails on the NEXT machine is invisible on this one unless the report
    names it, so the run that passes is where it has to be said. A pattern built on a generic
    mechanism carries its polynomial and depends on nothing; a caller's own generator does."""
    @PAT.pattern("caller_word", levels=2, period=3, source="not from a standard: a fixture")
    def _caller_word(length, seed=1):
        return PAT.block_repeat([1.0, -1.0, -1.0], length=length)

    doc = {"records": {
        "shipped/mechanism": _record(ws.Signal(seed=1, grid=_grid()).pattern("prbs9", length=256)),
        "callers/generator": _record(ws.Signal(seed=1, grid=_grid()).pattern("caller_word",
                                                                            length=256))}}
    rep = R.replay(R.load_document(_write(tmp_path, doc)), dry_run=True)
    assert rep.exit_code == 0, rep.lines()
    by_id = {r.record_id: r for r in rep.results}
    # the block records the polynomial, so the registry entry is not needed to render it
    assert by_id["shipped/mechanism"].needs_entry is False
    assert by_id["callers/generator"].needs_entry is True
    assert "embedded symbols" in rep.lines()


def test_the_pattern_column_names_the_sequence_for_every_source_shape():
    assert R.pattern_label([{"op": "carrier", "kind": "pam4", "pattern": "prbs13q"}]) \
        == "pam4:prbs13q"
    assert R.pattern_label([{"op": "carrier", "kind": "nrz"}]) == "nrz:default"
    assert R.pattern_label([{"op": "symbols", "symbols": [1.0, -1.0]}]) == "symbols[2]"
    named = [{"op": "symbols", "pattern": {"name": "prbs31", "digest": "abc123"}}]
    assert R.pattern_label(named) == "prbs31@abc123"
    assert R.needs_installed_pattern(named) is True
    assert R.needs_installed_pattern(
        [{"op": "symbols", "pattern": {"name": "prbs31", "generator": "lfsr"}}]) is False
    assert R.needs_installed_pattern(
        [{"op": "symbols", "symbols": [1.0], "pattern": {"name": "x"}}]) is False


# ----------------------------------------------------------------- 2. the round trip
def test_round_trip_is_bit_identical(tmp_path):
    sig = _signal()
    want = sig.waveform()
    doc = {"records": {"link/tp2/r0001": _record(sig)}}
    rep = R.replay(R.load_document(_write(tmp_path, doc)), out_dir=str(tmp_path / "out"))
    assert rep.exit_code == 0, rep.lines()
    got = np.load(str(tmp_path / "out" / "link" / "tp2" / "r0001.npy"))
    assert np.array_equal(got, want)
    assert rep.results[0].status == "ok" and rep.n_verified == 1


def test_top_level_mapping_without_a_wrapper_is_also_a_document(tmp_path):
    sig = _signal()
    doc = {"r0001": _record(sig)}
    rep = R.replay(R.load_document(_write(tmp_path, doc)), dry_run=True)
    assert rep.exit_code == 0 and rep.n_verified == 1


def test_cli_writes_records_and_exits_zero(tmp_path):
    sig = _signal()
    doc = {"records": {"a": _record(sig), "b": _record(_signal())}}
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, _REPLAY_PY, _write(tmp_path, doc), str(out)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert np.array_equal(np.load(str(out / "a.npy")), sig.waveform())
    assert json.loads((out / "replay.report.json").read_text())["records"]["a"]["status"] == "ok"


# ----------------------------------------------------------------- 3. it must catch a mismatch
def test_a_corrupted_digest_fails_and_reports_the_record_id(tmp_path):
    sig = _signal()
    rec = _record(sig)
    rec["values_sha256"] = "0" * 64
    doc = {"records": {"good": _record(_signal()), "the/bad/one": rec}}
    p = subprocess.run([sys.executable, _REPLAY_PY, _write(tmp_path, doc), "--dry-run"],
                       capture_output=True, text=True)
    assert p.returncode != 0
    assert "the/bad/one" in p.stdout + p.stderr
    assert "MISMATCH" in p.stdout + p.stderr


def test_a_record_that_fails_to_render_does_not_hide_the_others(tmp_path):
    doc = {"records": {"ok": _record(_signal()),
                       "broken": {"ops": [{"op": "no_such_op"}], "grid": {"fs": 1e9, "n": 64}}}}
    rep = R.replay(R.load_document(_write(tmp_path, doc)), dry_run=True)
    assert rep.exit_code != 0
    by_id = {r.record_id: r.status for r in rep.results}
    assert by_id == {"ok": "ok", "broken": "ERROR"}


# ----------------------------------------------------------------- 4. checked nothing is an error
def test_a_document_carrying_no_digest_is_not_a_pass(tmp_path):
    """The worst outcome is exit 0 having verified nothing, so an unverifiable document fails
    unless the caller says in as many words that it only wants the renders."""
    doc = {"records": {"r": _record(_signal(), digest=False)}}
    path = _write(tmp_path, doc)
    p = subprocess.run([sys.executable, _REPLAY_PY, path, "--dry-run"],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "verified" in (p.stdout + p.stderr).lower()
    q = subprocess.run([sys.executable, _REPLAY_PY, path, "--dry-run", "--no-verify"],
                       capture_output=True, text=True)
    assert q.returncode == 0, q.stdout + q.stderr


def test_only_with_an_id_that_is_not_in_the_document_is_an_error(tmp_path):
    doc = {"records": {"r0001": _record(_signal())}}
    p = subprocess.run([sys.executable, _REPLAY_PY, _write(tmp_path, doc), "--dry-run",
                        "--only", "r0002"], capture_output=True, text=True)
    assert p.returncode != 0 and "r0002" in p.stdout + p.stderr


def test_an_empty_document_is_an_error(tmp_path):
    p = subprocess.run([sys.executable, _REPLAY_PY, _write(tmp_path, {"records": {}}), "--dry-run"],
                       capture_output=True, text=True)
    assert p.returncode != 0


def test_only_selects_by_record_id(tmp_path):
    a, b = _signal(), _signal()
    doc = {"records": {"first": _record(a), "second": _record(b)}}
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, _REPLAY_PY, _write(tmp_path, doc), str(out),
                        "--only", "second"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "second.npy").exists() and not (out / "first.npy").exists()


def test_dry_run_writes_nothing_but_still_verifies(tmp_path):
    doc = {"records": {"r": _record(_signal())}}
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, _REPLAY_PY, _write(tmp_path, doc), str(out), "--dry-run"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert not out.exists(), "--dry-run must not create the output tree"


# ----------------------------------------------------------------- 5. the digest itself
def test_values_digest_separates_the_algorithm_from_the_data():
    x = np.linspace(-1, 1, 64)
    assert R.values_digest(x) != R.values_digest(x[::-1].copy())
    assert R.values_digest(x) == R.values_digest(x.copy())
    # a narrower stored dtype is a DIFFERENT digest, not an approximately equal one
    assert R.values_digest(x, "<f4") != R.values_digest(x, "<f8")
    assert R.VALUES_DIGEST_ALG in R.values_digest.__doc__


def test_a_declared_digest_algorithm_that_is_not_ours_is_an_error_not_a_skip(tmp_path):
    rec = _record(_signal())
    rec["values_digest_alg"] = "some-other-algorithm-v9"
    doc = {"records": {"r": rec}}
    rep = R.replay(R.load_document(_write(tmp_path, doc)), dry_run=True)
    assert rep.exit_code != 0 and rep.results[0].status == "ERROR"


def test_a_record_id_that_escapes_the_output_tree_is_refused(tmp_path):
    doc = {"records": {"../escaped": _record(_signal())}}
    with pytest.raises(ValueError, match="escaped"):
        R.load_document(_write(tmp_path, doc))


# ----------------------------------------------------------------- 6. the docstring teaches
def test_the_docstring_teaches_the_chain():
    doc = R.__doc__.lower()
    for link in ("bits", "coding", "symbol", "level", "waveform", "instrument"):
        assert link in doc, f"the docstring must place {link!r} in the chain"
    for op in ("carrier", "symbols", "lossy", "digitize"):
        assert op in doc, f"the docstring must say where the {op!r} op sits"


def test_the_replayer_imports_nothing_but_this_library_and_the_standard_one():
    """THE STANDALONE PROPERTY, gated rather than asserted in prose. The file ships next to an
    archive, so anything it imports is something the consumer must also have. numpy is in
    because this library requires it; every other name must be the standard library or this
    library itself. An `import` naming a sibling project is what this catches, and it catches
    it without needing to know that project's name."""
    allowed = set(sys.stdlib_module_names) | {"wfmsynth", "numpy"}
    tree = ast.parse(open(_REPLAY_PY).read())
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            seen |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            seen.add(node.module.split(".")[0])
    assert seen <= allowed, f"imports outside the library + stdlib: {sorted(seen - allowed)}"

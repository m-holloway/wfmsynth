"""`tools/digest_compare.py`: the fixed chains build on this tree, and `compare` says what moved.

The tool's reference is another checkout, so its real use is not a test. What is tested is
that every chain in its fixed set still builds and renders here (a chain that errors would be
compared as "equal errors" and prove nothing), and that `compare` reports a moved sample, a
recipe only one side has, and nothing when nothing moved.
"""
import importlib.util
import json
import warnings
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / "tools" / "digest_compare.py"
spec = importlib.util.spec_from_file_location("digest_compare", TOOL)
DC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DC)


def test_every_fixed_chain_builds_and_renders_on_this_tree(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chains = DC._chains(str(tmp_path))
        assert len(chains) >= 28
        for name, build in chains.items():
            s = build()
            assert len(DC._recipe_digest(s.recipe())) == 16, name
            assert len(DC._h(s.waveform())) == 16, name


def test_the_recipe_digest_ignores_the_version_and_nothing_else():
    r = {"wfmsynth_version": "0.45.0", "seed": 1, "ops": [{"op": "carrier", "kind": "nrz"}]}
    assert DC._recipe_digest(r) == DC._recipe_digest({**r, "wfmsynth_version": "0.46.0"})
    assert DC._recipe_digest(r) != DC._recipe_digest({**r, "seed": 2})


def test_compare_reports_what_moved_and_exits_nonzero_only_then(tmp_path, capsys):
    a = {"chains": {"link": {"recipe": "r", "samples": "s", "replay": "s"}},
         "renders": {"k1": "x", "k2": "y", "gone": "z"},
         "where": {"k1": "one.py", "k2": "two.py", "gone": "one.py"}}
    b = json.loads(json.dumps(a))
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(b))
    assert DC.compare(tmp_path / "a.json", tmp_path / "b.json") == 0
    assert "0 differ" in capsys.readouterr().out
    b["chains"]["link"]["samples"] = "moved"
    b["renders"]["k2"] = "moved"
    del b["renders"]["gone"]
    b["renders"]["new"] = "n"
    b["where"]["new"] = "three.py"
    (tmp_path / "b.json").write_text(json.dumps(b))
    assert DC.compare(tmp_path / "a.json", tmp_path / "b.json") == 1
    out = capsys.readouterr().out
    assert "DIFFERS  link: samples" in out
    assert "renders: 2 in common, 1 differ" in out
    assert "DIFFERS  two.py: 1 render" in out
    assert "one.py: 1 render" in out and "three.py: 1 render" in out

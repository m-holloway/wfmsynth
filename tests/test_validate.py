"""Smoke + physics-validation tests for wfmsynth."""
import subprocess
import sys

import numpy as np


def test_physics_validation_passes():
    """The hard physics-property assertions all pass (the 'don't fool yourself' gate)."""
    r = subprocess.run([sys.executable, "-m", "wfmsynth.validate"],
                       capture_output=True, text=True)
    assert r.returncode == 0, "physics validation failed:\n" + r.stdout + r.stderr


def test_generate_grammar_batch():
    import wfmsynth as ws
    g = ws.generate(8, seed=0)
    assert g["X"].shape == (8, ws.N)
    assert np.isfinite(g["X"]).all()
    assert len(g["carrier"]) == 8 and len(g["imps"]) == 8


def test_apply_impairment_and_dr():
    import wfmsynth as ws
    from wfmsynth import physics as P
    rng = np.random.default_rng(0)
    x = P.pam4(n_ui=32, seed=5)
    for name in ws.IMPAIRMENTS:
        y = ws.apply_impairment(name, x.copy(), rng)
        assert y.shape == x.shape and np.isfinite(y).all(), name
    assert np.isfinite(ws.domain_randomize(x.copy(), rng)).all()


def test_pam4_deep_capture():
    import wfmsynth as ws
    cap = ws.deep_capture(n_segments=300, needle_rate=0.05, seed=1, group_size=8)
    assert cap["X"].shape[1] == 1024
    assert np.isfinite(cap["X"]).all()
    assert len(cap["needle_idx"]) >= 1
    # group_size=1 path is the flat single-segment model
    flat = ws.deep_capture(n_segments=300, needle_rate=0.05, seed=1, group_size=1)
    assert flat["X"].shape == (300, 1024)

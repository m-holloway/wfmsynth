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


def test_rate_parameterization():
    """Primitives must honour the record length they are given, not a module global.

    The whole point of "rate-parameterizable": a real high-speed lane captured for
    tens of microseconds is millions of points, and nothing in physics/ should care.
    """
    from wfmsynth import physics as P
    for n in (512, 4096, 20000):
        x = P.pam4(n_ui=64, seed=3, n=n)
        assert len(x) == n
        for y in (P.lossy_channel(x, length_in=9.0, causal=True),
                  P.lossy_channel(x, length_in=9.0, causal=False),
                  P.multi_reflection(x, td_samples=31),
                  P.crosstalk(x, P.pam4(n_ui=64, seed=9, n=n)),
                  P.ac_couple(x),
                  P.inject_jitter(x, sigma_rj=1.5, a_pj=0.5,
                                  rng=np.random.default_rng(0))):
            assert len(y) == n and np.isfinite(y).all()


def test_default_grid_output_is_unchanged():
    """Parameterization must not perturb the legacy 4096-point behaviour."""
    from wfmsynth import physics as P
    assert len(P.nrz(seed=3)) == P.N
    assert len(P.pam4(seed=5)) == P.N
    # explicit n equal to the default must match the default path exactly
    assert np.array_equal(P.pam4(n_ui=32, seed=5), P.pam4(n_ui=32, seed=5, n=P.N))
    assert np.array_equal(P.nrz(n_ui=32, seed=3), P.nrz(n_ui=32, seed=3, n=P.N))


def test_prbs13q_is_ieee_conformant():
    """PRBS13Q against published IEEE 802.3 120.5.11.2.1 statistics.

    A non-standard degree-13 polynomial reproduces the level probabilities but is a
    different sequence, and an analyser will not pattern-lock to it -- so the
    statistics are checked, not just the shape.
    """
    from wfmsynth import physics as P
    assert P.PRBS_TAPS[13] == (13, 12, 2, 1)

    bits = P.prbs(13, 8191 * 2)
    assert np.array_equal(bits[:8191], bits[8191:])
    assert int(bits[:8191].sum()) == 4096

    q = P.prbs13q(8191)
    counts = [int(np.isclose(q, lv).sum()) for lv in (-1.0, -1 / 3, 1 / 3, 1.0)]
    assert counts == [2047, 2048, 2048, 2048]
    assert abs(float(np.mean(q[1:] != q[:-1])) - 0.7501) < 5e-4
    assert np.array_equal(P.prbs13q(16382)[:8191], P.prbs13q(16382)[8191:])


def test_pam4_pattern_selection():
    from wfmsynth import physics as P
    legacy = P.pam4(n_ui=256, seed=5)
    std = P.pam4(n_ui=256, seed=5, pattern="prbs13q")
    assert legacy.shape == std.shape
    assert not np.array_equal(legacy, std)
    try:
        P.pam4(pattern="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown pattern should raise")


def test_causal_edge_shaping_has_no_precursor_or_startup_step():
    from wfmsynth import physics as P
    step = np.concatenate([np.full(256, -1.0), np.full(256, 1.0)])
    zp = P._shape_edges(step, 8.0, causal=False)
    cz = P._shape_edges(step, 8.0, causal=True)
    pre_zp = np.abs(zp[200:256] + 1.0).max()
    pre_cz = np.abs(cz[200:256] + 1.0).max()
    assert pre_zp > 1e-3, "zero-phase shaping should show pre-cursor"
    assert pre_cz < pre_zp / 10, "causal shaping should not"
    assert abs(cz[0] + 1.0) < 1e-6, "no filter start-up transient"


def test_absolute_reflection_delay_is_record_length_independent():
    from wfmsynth import physics as P
    peaks = []
    for n in (2048, 8192):
        imp = np.zeros(n)
        imp[100] = 1.0
        r = P.multi_reflection(imp, td_samples=60, gamma_s=0.5, gamma_l=0.5,
                               n_bounce=1)
        peaks.append(int(np.argmax(r[101:])))
    assert peaks[0] == peaks[1] == 119      # 2*td - 1 relative to the slice start

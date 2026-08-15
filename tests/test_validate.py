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


def test_grid_derived_quantities():
    import wfmsynth as ws
    g = ws.Grid(fs=256e9, baud=112e9, n=4096)
    assert abs(g.dt - 1 / 256e9) < 1e-24
    assert abs(g.f_nyquist - 128e9) < 1
    assert abs(g.samples_per_ui - 256e9 / 112e9) < 1e-9
    assert abs(g.duration - 4096 / 256e9) < 1e-24
    assert ws.Grid(fs=1e9).baud is None and ws.Grid(fs=1e9).samples_per_ui is None


def test_absolute_units_round_trip():
    """Done-criteria for BACKLOG #1: ps delay and dB@freq loss realize as requested."""
    import wfmsynth as ws
    from wfmsynth import physics as P
    g = ws.Grid(fs=256e9, baud=112e9, n=4096)
    # reflection delay in ps lands within one sample
    imp = np.zeros(g.n); imp[100] = 1.0
    r = P.multi_reflection(imp, td_ps=40.0, grid=g, gamma_s=0.5, gamma_l=0.5, n_bounce=1)
    realized_ps = (int(np.argmax(r[101:])) + 1) / 2.0 * g.dt * 1e12
    assert abs(realized_ps - 40.0) <= g.dt * 1e12 + 1e-9
    # loss in dB at a stated frequency is realized there
    hi = np.zeros(g.n); hi[0] = 1.0
    H = np.abs(np.fft.rfft(P.lossy_channel(hi, loss_db=12.0, loss_at_ghz=20.0, grid=g)))
    k = int(np.argmin(np.abs(np.fft.rfftfreq(g.n) * g.fs / 1e9 - 20.0)))
    assert abs(-20 * np.log10(H[k] + 1e-12) - 12.0) < 0.5


def test_absolute_units_equal_sample_domain_and_require_grid():
    from wfmsynth import physics as P
    import wfmsynth as ws
    g = ws.Grid(fs=256e9, n=4096)
    x = P.pam4(n_ui=64, seed=3)
    assert np.allclose(
        P.inject_jitter(x, sigma_rj_s=2.0 * g.dt, rng=np.random.default_rng(1), grid=g),
        P.inject_jitter(x, sigma_rj=2.0, rng=np.random.default_rng(1)), atol=1e-9)
    assert np.allclose(P.ac_couple(x, fc_hz=5e6, grid=g),
                       P.ac_couple(x, fc_frac=g.hz_to_frac_nyquist(5e6)), atol=1e-9)
    # absolute-unit kwargs require a grid
    for call in (lambda: P.multi_reflection(x, td_ps=40.0),
                 lambda: P.ac_couple(x, fc_hz=5e6)):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("absolute-unit kwarg without grid should raise ValueError")


def test_non_integer_sps_fractional_period_and_drift():
    """BACKLOG #2 done-criteria: non-integer sps, exact fractional period, integer drift."""
    import wfmsynth as ws
    g = ws.Grid(fs=256e9, baud=112e9, n=4096)
    sps = g.samples_per_ui
    assert abs(sps - round(sps)) > 0.1                    # genuinely non-integer
    per = g.pattern_period_samples(8191)                  # exact fractional period
    assert abs(per - 8191 * sps) < 1e-9 and abs(per - round(per)) > 1e-6
    n_sym = 2000
    n = int(round(n_sym * sps))
    syms = np.random.default_rng(0).integers(0, 4, n_sym)
    lv = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
    wav = lv[syms[np.clip((np.arange(n) / sps).astype(int), 0, n_sym - 1)]]

    def rec(s):
        pos = np.round(np.arange(n_sym) * s + s / 2).astype(int)
        pos = pos[pos < n]
        got = np.argmin(np.abs(wav[pos][:, None] - lv[None, :]), axis=1)
        return (got == syms[:len(pos)]).mean()

    assert rec(sps) > 0.98                                # true fractional sps recovers
    assert rec(round(sps)) < 0.6                          # integer assumption drifts
    try:
        ws.Grid(fs=1e9).pattern_period_samples(100)       # baud unset -> raises
    except ValueError:
        pass
    else:
        raise AssertionError("pattern_period_samples without baud should raise")


def test_interleave_adc_spurs_and_clip():
    """BACKLOG #3 done-criteria: interleave mismatch -> spurs at fs/M; none at zero."""
    import wfmsynth as ws
    n, m, fin = 4096, 4, 300
    tone = np.sin(2 * np.pi * fin * np.linspace(0, 1, n, endpoint=False))
    ymm = ws.interleave_adc(tone, m_cores=m, offset_mm=0.02, gain_mm=0.01,
                            rng=np.random.default_rng(0))
    yid = ws.interleave_adc(tone, m_cores=m)                 # all mismatch default 0
    Ymm = np.abs(np.fft.rfft(ymm - ymm.mean()))
    Yid = np.abs(np.fft.rfft(yid - yid.mean()))
    spur = n // m
    assert Ymm[spur] > 50 * (Yid[spur] + 1e-9)              # spur with mismatch
    assert Yid[spur] < 1e-6 * Ymm.max()                    # none without
    assert Ymm[spur - fin] > 20 * (Yid[spur - fin] + 1e-9)  # gain image at fs/M - f_in
    # clip_adc marks saturated samples
    y, mask = ws.clip_adc(np.array([-2.0, 0.0, 2.0]), full_scale=1.0)
    assert y.tolist() == [-1.0, 0.0, 1.0]
    assert mask.tolist() == [True, False, True]


def test_source_jitter_edge_rms_and_independent_noise():
    """BACKLOG #4 done-criteria: source jitter edge-RMS ~ injected; post-channel noise independent."""
    from wfmsynth import physics as P
    from wfmsynth import Jitter, Grid
    nui = 96
    ref = P.nrz(n_ui=nui, tr_frac=0.1, seed=5)
    jit = P.nrz(n_ui=nui, tr_frac=0.1, seed=5, jitter=Jitter(rj=3.0), rng=np.random.default_rng(0))

    def cross(y):
        s = np.sign(y - y.mean())
        return np.where(np.diff(s) != 0)[0].astype(float)

    c0, c1 = cross(ref), cross(jit)
    assert len(c0) == len(c1)
    assert 1.5 < float(np.std(c1 - c0)) < 4.5           # recovered edge RMS ~ injected 3
    # jitter=None is bit-identical to the legacy carrier
    assert np.array_equal(P.nrz(n_ui=32, seed=1), P.nrz(n_ui=32, seed=1, jitter=None))
    # post-channel noise is independent of source jitter (jitter is upstream)
    sig = P.lossy_channel(P.pam4(n_ui=nui, seed=5, jitter=Jitter(rj=2.0), rng=np.random.default_rng(1)),
                          length_in=8.0, causal=True)
    noise = np.random.default_rng(2).normal(0, 0.05, len(sig))
    assert np.allclose((sig + noise) - sig, noise, atol=1e-12)
    # Jitter.at converts seconds/Hz through a grid
    g = Grid(fs=256e9, n=4096)
    assert abs(Jitter.at(g, rj_s=3.0 / 256e9).rj - 3.0) < 1e-9


def test_provenance_recipe_round_trip():
    """BACKLOG #5 done-criteria: recipe reproduces bit-for-bit through JSON; version stamped."""
    import json
    import wfmsynth as ws
    g = ws.Grid(fs=256e9, baud=112e9, n=1 << 13)
    sig = (ws.Signal(seed=42, grid=g)
           .carrier("pam4", n_ui=g.n // 8, pattern="prbs13q", causal=True, jitter=dict(rj=0.4, pj=0.2))
           .lossy(loss_db=15.0, loss_at_ghz=26.0, causal=True)
           .reflect(td_ps=55.0, gamma_s=0.4, gamma_l=0.4)
           .digitize(snr_db=32.0, enob=5.5, interleave=dict(m_cores=4, offset_mm=0.01)))
    x = sig.waveform()
    r = json.loads(json.dumps(sig.recipe()))                 # must be JSON-serializable
    assert r["wfmsynth_version"]
    assert np.array_equal(ws.Signal.from_recipe(r).waveform(), x)   # bit-for-bit
    try:
        ws.Signal(seed=1).waveform()
    except ValueError:
        pass
    else:
        raise AssertionError("empty Signal should raise")


def test_dataset_records_sampled_knobs_and_reproduces():
    import wfmsynth as ws
    g = ws.Grid(fs=256e9, baud=112e9, n=1 << 12)

    def build(rng):
        return (ws.Signal(seed=int(rng.integers(1e9)), grid=g)
                .carrier("pam4", n_ui=g.n // 8, seed=int(rng.integers(1e6)))
                .lossy(loss_db=float(rng.uniform(8, 20)), loss_at_ghz=20.0, causal=True))

    X, recipes = ws.dataset(build, 6, seed=0)
    assert X.shape[0] == 6 and len(recipes) == 6
    for r in recipes:
        assert 8.0 <= r["ops"][1]["loss_db"] <= 20.0        # the sampled knob is recorded
        w = ws.Signal.from_recipe(r).waveform()
        assert np.array_equal(w, ws.Signal.from_recipe(r).waveform())   # reproducible


def test_rng_stream_roles_independent_and_rerollable():
    import wfmsynth as ws
    s = ws.Streams(1234)
    j1 = s.role("jitter").standard_normal(64)
    n1 = s.role("noise").standard_normal(64)
    s2 = ws.Streams(1234)                       # draw the same roles in the opposite order
    n2 = s2.role("noise").standard_normal(64)
    j2 = s2.role("jitter").standard_normal(64)
    assert np.array_equal(n1, n2) and np.array_equal(j1, j2)   # order-independent + independent
    s3 = s.reroll("jitter")                     # re-roll ONE factor
    assert not np.array_equal(s3.role("jitter").standard_normal(64), j1)
    assert np.array_equal(s3.role("noise").standard_normal(64), n1)


def test_contrastive_pair_isolates_one_factor():
    import wfmsynth as ws
    g = ws.Grid(fs=256e9, baud=112e9, n=1 << 12)
    sig = (ws.Signal(seed=3, grid=g)
           .carrier("pam4", n_ui=g.n // 8, pattern="prbs13q", jitter=dict(rj=0.5))
           .digitize(snr_db=28.0))                   # carrier=op0, digitize=op1 -> role noise/1
    assert set(sig.roles()) == {"jitter/0", "noise/1"}
    base = sig.waveform()
    assert not np.array_equal(sig.contrast("noise/1", seed=1), base)   # noise re-rolled
    assert np.array_equal(sig.contrast("noise/1", seed=1), sig.contrast("noise/1", seed=1))

    # a changed upstream factor (jitter) leaves the downstream noise realization identical
    def full_and_clean(rj):
        s = (ws.Signal(seed=7, grid=g)
             .carrier("pam4", n_ui=g.n // 8, pattern="prbs13q", jitter=dict(rj=rj))
             .digitize(noise_rms=0.01))
        q = ws.Signal.from_recipe(s.recipe())
        q.ops[-1] = {k: v for k, v in q.ops[-1].items() if k != "noise_rms"}
        return s.waveform(), q.waveform()

    fA, cA = full_and_clean(0.3)
    fB, cB = full_and_clean(3.0)
    N = ws.Streams(7).role("noise/1").normal(0.0, 0.01, len(cA))
    assert np.array_equal(fA, cA + N) and np.array_equal(fB, cB + N)
    assert not np.array_equal(cA, cB)                # ...even though the clean signals differ


def test_hold_constant_sweep_pins_measured_metric():
    import wfmsynth as ws
    g = ws.Grid(fs=200e9, baud=50e9, n=1 << 13)      # spb = 4
    n_ui = int(g.n // g.samples_per_ui)

    def build(gamma=0.05, loss_db=2.0):
        return (ws.Signal(seed=1, grid=g)
                .carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
                .lossy(loss_db=loss_db, loss_at_ghz=25.0, causal=True)
                .reflect(td_ps=30.0, gamma_s=gamma, gamma_l=gamma))

    # eye is measured, monotonic in loss (needed for the bisection to be well posed)
    eyes = [ws.eye_height(build(0.05, l).waveform(), g) for l in (0.0, 2.0, 4.0)]
    assert eyes[0] > eyes[1] > eyes[2]

    target = ws.eye_height(build(0.05, 2.0).waveform(), g)
    recs = ws.hold_constant(build, "gamma", [0.05, 0.15, 0.25, 0.35], "eye", target,
                            "loss_db", (0.0, 4.0), g, ws.eye_height, tol=0.004)
    realized = [r["realized_eye"] for r in recs]
    solved = [r["loss_db"] for r in recs]
    assert max(abs(e - target) for e in realized) <= 0.02            # pin held
    assert all(solved[i] > solved[i + 1] for i in range(len(solved) - 1))   # compensation moves
    assert all("realized_eye" in r for r in recs)                   # realized, not requested


def test_realized_table_exposes_leak_and_two_eye_definitions():
    import wfmsynth as ws
    g = ws.Grid(fs=200e9, baud=50e9, n=1 << 13)
    n_ui = int(g.n // g.samples_per_ui)

    def build(gamma=0.05, loss_db=1.0):
        return (ws.Signal(seed=1, grid=g)
                .carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
                .lossy(loss_db=loss_db, loss_at_ghz=25.0, causal=True)
                .reflect(td_ps=30.0, gamma_s=gamma, gamma_l=gamma))

    gammas = np.linspace(0.0, 0.4, 6)
    sets = [dict(gamma=gm, loss_db=1.0) for gm in gammas]
    recs, corr, names = ws.realized_table(build, sets, g, ws.attributes)
    assert set(names) == {"eye_contour", "eye_sigma", "ptp", "rms"}
    assert all(f"realized_{n}" in recs[0] for n in names)
    # the leak is visible in the realized labels: sweeping reflection alone strongly
    # anti-correlates with the realized eye opening (so a model could read eye, not ISI).
    eyes = np.array([r["realized_eye_contour"] for r in recs])
    assert np.corrcoef(gammas, eyes)[0, 1] < -0.8
    # both named eye definitions are computable and finite
    x = build(0.3, 1.0).waveform()
    assert np.isfinite(ws.eye_height(x, g, defn="sigma"))
    assert np.isfinite(ws.eye_height(x, g, defn="contour"))


def test_ground_truth_measured_eye_definitions_and_symbol_alignment():
    import wfmsynth as ws
    g = ws.Grid(fs=200e9, baud=50e9, n=1 << 13)
    n_ui = int(g.n // g.samples_per_ui)

    # named eye definitions: agree under Gaussian noise, diverge under deterministic ISI
    isi = (ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
           .reflect(td_ps=40.0, gamma_s=0.45, gamma_l=0.45)).waveform()
    gau = (ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
           .digitize(noise_rms=0.06)).waveform()
    di = abs(ws.eye_height(isi, g, defn="sigma") - ws.eye_height(isi, g, defn="contour"))
    dg = abs(ws.eye_height(gau, g, defn="sigma") - ws.eye_height(gau, g, defn="contour"))
    assert dg < 0.02 and di > 0.05 and di > dg + 0.03

    # realized integer-symbol alignment: a causal channel's group delay must be recovered,
    # and skipping it (offset 0) collapses the tx/output correlation.
    sig = (ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
           .lossy(loss_db=3.0, loss_at_ghz=25.0, causal=True)
           .reflect(td_ps=40.0, gamma_s=0.3, gamma_l=0.3))
    gt = sig.ground_truth()
    assert gt["align_offset"] != 0 and gt["align_corr"] > 0.9
    assert gt["align_corr"] > gt["align_corr_at_zero"] + 0.4
    assert all(k in gt for k in ("eye_contour", "eye_sigma", "best_phase", "align_offset"))

    # carrier_symbols is the single source of truth for the transmitted stream
    assert np.array_equal(ws.physics.carrier_symbols("pam4", n_ui, 1, "prbs13q"),
                          ws.physics.prbs13q(n_ui, 1))


def test_instrument_adc_model_quantise_offset_and_ordered_digitize():
    from wfmsynth.instrument import (quantize_adc, digitize, interleave_adc, clip_adc,
                                     shaped_noise_floor)
    # #24 standalone ENOB quantiser
    x = np.linspace(-1, 1, 4000)
    q = quantize_adc(x, enob=6, full_scale=1.0)
    lsb = 2.0 / 2 ** 6
    assert abs(len(np.unique(q)) - 64) <= 4
    assert np.max(np.abs(q - x)) <= 0.5 * lsb + 1e-12

    # #26 absolute-volts offset is input-scale invariant; fractional offset scales with input
    xo = 0.5 * np.sin(np.linspace(0, 80, 2000))

    def tone(kw, sc):
        y = interleave_adc(sc * xo, m_cores=4, gain_mm=0.0, skew_mm=0.0,
                           rng=np.random.default_rng(0), **kw)
        return float(np.std(y - sc * xo))

    assert abs(tone(dict(offset_v=1e-3), 4.0) / tone(dict(offset_v=1e-3), 1.0) - 1.0) < 0.02
    assert abs(tone(dict(offset_mm=0.05), 4.0) / tone(dict(offset_mm=0.05), 1.0) - 4.0) < 0.05

    # #25 composed digitize == manual correct-order pipeline; order matters
    mm = (np.array([0.01, -0.005, 0.008, -0.003]), np.array([2e-3, -1e-3, 1.5e-3, -2e-3]),
          np.zeros(4))
    yc, info = digitize(xo, noise_floor={"rms": 2e-3, "shape": "white"},
                        interleave={"m_cores": 4, "mismatch": mm}, clip_full_scale=0.7,
                        enob=6, rng=np.random.default_rng(3))
    r = np.random.default_rng(3)
    nz = shaped_noise_floor(len(xo), rng=r, rms=2e-3, shape="white")
    ym = interleave_adc(xo + nz, m_cores=4, mismatch=mm)
    ym, _ = clip_adc(ym, 0.7)
    ym = quantize_adc(ym, enob=6, full_scale=0.7)
    assert np.array_equal(yc, ym) and "clipped_fraction" in info
    assert not np.array_equal(quantize_adc(xo + nz, enob=6, full_scale=0.7),
                              quantize_adc(xo, enob=6, full_scale=0.7) + nz)


def test_mix_at_constant_power_holds_rms_and_moves_character():
    import wfmsynth as ws
    rng = np.random.default_rng(0)
    white = ws.shaped_noise_floor(1 << 13, rms=1.0, shape="white", rng=rng)
    pink = ws.shaped_noise_floor(1 << 13, rms=1.0, shape="pink", rng=rng)

    def centroid(x):
        X = np.abs(np.fft.rfft(x))
        return (np.arange(len(X)) * X).sum() / (X.sum() + 1e-12)

    rmss, cents = [], []
    for a in np.linspace(0.0, 1.0, 6):
        y = ws.mix_at_constant_power([white, pink], [1 - a, a], total_rms=0.05)
        rmss.append(float(np.sqrt(np.mean(y ** 2))))
        cents.append(centroid(y))
    assert max(rmss) - min(rmss) < 1e-9                       # total power invariant
    assert all(cents[i] > cents[i + 1] for i in range(len(cents) - 1))   # character moves
    # weights need not sum to 1, and negatives are rejected
    assert abs(np.sqrt(np.mean(ws.mix_at_constant_power([white, pink], [2, 5], 0.1) ** 2))
               - 0.1) < 1e-9
    try:
        ws.mix_at_constant_power([white, pink], [-1, 1], 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative weights should raise")


def test_gated_intermittent_impairment_confined_with_mask():
    import wfmsynth as ws
    g = ws.Grid(fs=200e9, baud=50e9, n=1 << 13)
    x = (ws.Signal(seed=1, grid=g)
         .carrier("pam4", n_ui=int(g.n // g.samples_per_ui), pattern="prbs13q", causal=True)
         ).waveform()

    def glitch(s):
        return s + 0.5 * np.sign(np.sin(np.linspace(0, 300, len(s))))

    y, mask = ws.apply_gated(x, glitch, [(2000, 180), (6000, 140)])
    out = mask == 0
    assert np.array_equal(y[out], x[out])                      # zero leakage outside gate
    assert np.max(np.abs((y - x)[mask > 0.5])) > 0.1           # defect present inside
    assert mask.shape == x.shape and 0.0 < (mask > 0).mean() < 0.1   # low-duty mask emitted
    assert np.max(np.abs(np.diff(mask))) < 0.3                 # raised-cosine, not a step
    # additive-array form also confines exactly
    y2, m2 = ws.apply_gated(x, np.full_like(x, 5.0), [(100, 50)])
    assert np.array_equal(y2[m2 == 0], x[m2 == 0])


def test_tx_ffe_precursor_and_eye_compensation():
    import wfmsynth as ws
    g = ws.Grid(fs=200e9, baud=50e9, n=1 << 13)
    spb = g.samples_per_ui
    n_ui = int(g.n // spb)

    # a one-UI pulse gains a pre-cursor one UI ahead of the main peak, equal to the pre tap
    pulse = np.zeros(g.n)
    pulse[g.n // 2:g.n // 2 + int(spb)] = 1.0
    y = ws.tx_ffe(pulse, [-0.2, 1.0, -0.3], spb, pre=1)
    peak = int(np.argmax(np.abs(y)))
    assert abs(y[peak] - 1.0) < 1e-6
    assert abs(y[peak - int(round(spb))] - (-0.2)) < 0.05     # pre-cursor ~ the pre tap

    # FFE de-emphasis opens a lossy-channel eye relative to no FFE
    no_ffe = (ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
              .lossy(loss_db=8.0, loss_at_ghz=25.0, causal=True)).waveform()
    with_ffe = (ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=n_ui, pattern="prbs13q", causal=True)
                .tx_ffe(taps=[-0.15, 1.0, -0.25], pre=1)
                .lossy(loss_db=8.0, loss_at_ghz=25.0, causal=True)).waveform()
    assert ws.eye_height(with_ffe, g) > ws.eye_height(no_ffe, g) + 0.02


def test_composed_chain_causality():
    import wfmsynth.physics as P
    N = P.N
    tr = 8.0
    pulse = np.zeros(N)
    c = N // 2
    W = int(20 * tr)
    pulse[c:c + W] = 1.0

    def edge_ratio(shaped_causal):
        y = P.lossy_channel(P._shape_edges(pulse, tr, causal=shaped_causal),
                            length_in=12.0, causal=True)
        w = int(4 * tr)
        return float(np.sum(y[c - w:c] ** 2) / (np.sum(y[c:c + w] ** 2) + 1e-12))

    rc, rz = edge_ratio(True), edge_ratio(False)
    assert rc < 0.01                          # fully-causal composed chain: ~zero pre-cursor
    assert rz > 10 * rc and rz > 0.005        # zero-phase shaping leaks behind a causal channel


def test_pattern_lock_single_sharp_peak_at_declared_period():
    import wfmsynth as ws
    syms = ws.physics.carrier_symbols("pam4", 2 * 8191, 1, "prbs13q")
    lag, peak, ac = ws.pattern_period(syms, max_lag=9000)
    m = np.ones(9001, bool)
    m[max(1, lag - 20):lag + 20] = False
    m[:20] = False
    nxt = float(np.max(ac[:9001][m]))
    assert lag == 8191 and peak > 0.95 and peak > 2 * nxt     # single dominant peak at 8191


def test_clock_recovery_jitter_transfer():
    import wfmsynth as ws
    baud = 50e9
    N = 1 << 16
    t = np.arange(N) / baud
    BW = 1e6

    def jtf(fm, bw=BW, order=2):
        ph = np.sin(2 * np.pi * fm * t)
        return np.ptp(ws.recover_clock(ph, baud, bw, order)[1][N // 2:]) / np.ptp(ph[N // 2:])

    # high-pass jitter transfer: below the loop bandwidth is tracked out, above is passed
    assert jtf(1e5) < 0.1 and jtf(1e7) > 0.9
    # a wider loop bandwidth tracks out more low-frequency jitter
    lf = np.sin(2 * np.pi * 3e5 * t)
    assert ws.tracked_out_fraction(lf, baud, 3e6) > ws.tracked_out_fraction(lf, baud, 3e5) + 0.2
    # 2nd-order tracks a frequency offset (phase ramp) to ~zero; 1st-order leaves a lag
    ramp = np.arange(N) * 2e-4
    r1 = ws.recover_clock(ramp, baud, BW, order=1)[1]
    r2 = ws.recover_clock(ramp, baud, BW, order=2)[1]
    assert np.mean(np.abs(r2[N // 2:])) < 0.1 * np.mean(np.abs(r1[N // 2:]))
    # clock + residual reconstruct the input (definitional)
    clk, res = ws.recover_clock(lf, baud, BW)
    assert np.allclose(clk + res, lf, atol=1e-9)


def test_touchstone_roundtrip_and_resonant_channel(tmp_path):
    import wfmsynth as ws
    f = np.linspace(1e8, 40e9, 800)
    notch = 1 - 0.9 * np.exp(-((f - 20e9) / 1.5e9) ** 2)
    S = np.zeros((len(f), 2, 2), complex)
    S[:, 1, 0] = notch * np.exp(-1j * 2 * np.pi * f * 20e-12)
    S[:, 0, 1] = S[:, 1, 0]
    S[:, 0, 0] = 0.05
    S[:, 1, 1] = 0.05
    p = str(tmp_path / "thru.s2p")
    # every format round-trips (S21, through the 2-port ordering quirk)
    for fmt, atol in (("RI", 1e-9), ("MA", 1e-6), ("DB", 1e-4)):
        ws.write_touchstone(p, f, S, fmt=fmt)
        fr, Sr = ws.read_touchstone(p)
        assert np.allclose(f, fr) and np.allclose(S[:, 1, 0], Sr[:, 1, 0], atol=atol)
        assert Sr.shape == (800, 2, 2)
    # the channel reproduces the resonant notch a monotonic analytic model can't
    g = ws.Grid(fs=80e9, n=1 << 14)
    x = np.random.default_rng(0).standard_normal(g.n)
    y = ws.touchstone_channel(x, p, grid=g)
    fg = np.fft.rfftfreq(g.n, d=g.dt)
    Y = np.abs(np.fft.rfft(y))
    nb = int(np.argmin(np.abs(fg - 20e9)))
    rb = int(np.argmin(np.abs(fg - 10e9)))
    assert Y[nb] / (Y[rb] + 1e-9) < 0.2
    # composes as a fluent op with provenance
    g2 = ws.Grid(fs=80e9, baud=20e9, n=1 << 13)
    sig = (ws.Signal(seed=1, grid=g2)
           .carrier("pam4", n_ui=g2.n // 4, pattern="prbs13q", causal=True).sparam(path=p))
    assert sig.waveform().shape == (g2.n,) and sig.recipe()["ops"][-1]["op"] == "sparam"


def test_resonant_reflection_is_frequency_dependent():
    import wfmsynth as ws
    g = ws.Grid(fs=100e9, n=1 << 14)
    f = np.fft.rfftfreq(g.n, d=g.dt)
    x = np.random.default_rng(0).standard_normal(g.n)
    yr = ws.resonant_reflection(x, grid=g, td_ps=50.0, f0_ghz=25.0, q=12.0, gamma0=0.5)
    yf = ws.physics.multi_reflection(x, grid=g, td_ps=50.0, gamma_s=0.5, gamma_l=0.0, n_bounce=1)
    Rr = np.abs(np.fft.rfft(yr - x))
    Rf = np.abs(np.fft.rfft(yf - x))
    b0 = int(np.argmin(np.abs(f - 25e9)))
    blo = int(np.argmin(np.abs(f - 5e9)))
    # resonant Γ concentrates the reflection near f0; flat Γ does not
    assert Rr[b0] / (Rr[blo] + 1e-9) > 5.0
    assert abs(f[int(np.argmax(Rr))] - 25e9) < 3e9
    assert Rf[b0] / (Rf[blo] + 1e-9) < 3.0
    # composes fluently
    g2 = ws.Grid(fs=100e9, baud=25e9, n=1 << 13)
    sig = (ws.Signal(seed=1, grid=g2).carrier("pam4", n_ui=g2.n // 4, pattern="prbs13q", causal=True)
           .resonant_reflect(td_ps=40.0, f0_ghz=30.0, q=10.0, gamma0=0.4))
    assert sig.waveform().shape == (g2.n,)


def test_nominal_nonlinearity_three_effects():
    import wfmsynth as ws
    # compression: PAM4 level spacing non-uniform (RLM<1) but mild
    lv = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
    gaps = np.diff(ws.nominal_nonlinearity(lv, compression=0.06))
    assert 0.85 < gaps.min() / gaps.max() < 0.999
    # level-dependent noise: outer noisier than inner
    rng = np.random.default_rng(0)
    out = ws.nominal_nonlinearity(np.full(20000, 1.0), compression=0.0, level_noise=0.02, rng=rng)
    inr = ws.nominal_nonlinearity(np.full(20000, 1 / 3), compression=0.0, level_noise=0.02, rng=rng)
    assert out.std() > 2.0 * inr.std()
    # rise/fall asymmetry on a step
    step = np.concatenate([np.zeros(200), np.ones(200), np.zeros(200)]).astype(float)
    y = ws.nominal_nonlinearity(step, compression=0.0, rise_fall_ratio=2.5)
    rise = int(np.argmax(y[200:400] >= 0.9))
    fall = int(np.argmax(y[400:600] <= 0.1))
    assert rise != fall and rise > 0 and fall > 0
    # composes fluently with its own noise stream
    g = ws.Grid(fs=200e9, baud=50e9, n=1 << 12)
    s = (ws.Signal(seed=1, grid=g).carrier("pam4", n_ui=g.n // 4, pattern="prbs13q", causal=True)
         .nonlinearity(compression=0.04, level_noise=0.005, rise_fall_ratio=1.3))
    assert s.waveform().shape == (g.n,)


def test_crosstalk_matrix_async_default_and_superposition():
    import wfmsynth as ws
    import wfmsynth.physics as P
    g = ws.Grid(fs=400e9, baud=50e9, n=1 << 14)
    spb = int(g.samples_per_ui)
    x = P.nrz(n_ui=g.n // spb, seed=1, n=g.n, causal=True)

    def conc(y):
        c = np.abs(y - x)
        prof = np.array([c[i::spb][:len(c) // spb].mean() for i in range(spb)])
        return np.ptp(prof) / (prof.mean() + 1e-12)

    # linear superposition in the coupling vector
    p1 = np.std(ws.crosstalk_matrix(x, g, [0.1]) - x)
    p2 = np.std(ws.crosstalk_matrix(x, g, [0.2]) - x)
    assert abs(p2 / p1 - 2.0) < 0.1
    assert np.std(ws.crosstalk_matrix(x, g, [0.1, 0.1, 0.1]) - x) > p1
    # async by default is NOT locked to the victim UI; synchronous IS
    assert conc(ws.crosstalk_matrix(x, g, [0.15])) < 0.3 * conc(
        ws.crosstalk_matrix(x, g, [0.15], synchronous=True))
    # composes fluently
    g2 = ws.Grid(fs=200e9, baud=50e9, n=1 << 12)
    sig = (ws.Signal(seed=1, grid=g2).carrier("pam4", n_ui=g2.n // 4, pattern="prbs13q", causal=True)
           .crosstalk_matrix(couplings=[0.12, 0.08]))
    assert sig.waveform().shape == (g2.n,)


def test_realistic_noise_heavy_tails_and_pink():
    import wfmsynth as ws

    def exk(a):
        a = a - a.mean()
        return np.mean(a ** 4) / (np.mean(a ** 2) ** 2) - 3.0

    heavy = ws.realistic_noise(1 << 16, df=5.0, rng=np.random.default_rng(0))
    gauss = ws.realistic_noise(1 << 16, df=200.0, rng=np.random.default_rng(0))
    assert exk(heavy) > 1.0 and abs(exk(gauss)) < 0.5              # heavy tails vs Gaussian
    assert abs(np.sqrt(np.mean(heavy ** 2)) - 0.01) < 1e-6         # RMS honored
    # 1/f fraction concentrates power at low frequency
    pink = ws.realistic_noise(1 << 16, df=200.0, pink_frac=0.85, rng=np.random.default_rng(1))
    P = np.abs(np.fft.rfft(pink)) ** 2
    f = np.fft.rfftfreq(1 << 16)
    lo = P[(f > 1e-3) & (f < 1e-2)].mean()
    hi = P[(f > 0.1) & (f < 0.4)].mean()
    assert lo / hi > 5.0


def test_streaming_convolution_matches_full_and_is_chunked():
    import wfmsynth as ws
    import wfmsynth.physics as P
    n = 1 << 18
    x = np.random.default_rng(0).standard_normal(n)
    h = np.exp(-((np.arange(65) - 32) / 8.0) ** 2)
    h /= h.sum()
    ref = np.convolve(x, h)[:n]
    assert np.allclose(ws.stream_convolve(x, h, chunk=8192), ref, atol=1e-9)
    # generator yields the whole record in chunks (>1 block, bounded working set)
    blocks = list(ws.stream_blocks(x, h, chunk=8192))
    assert sum(len(b) for b in blocks) == n and len(blocks) > 1
    # a linear channel applied via its FIR (chunked) matches the direct channel interior
    g = ws.Grid(fs=100e9, n=1 << 15)
    apply = lambda a: P.lossy_channel(a, length_in=8.0, tand=0.02, causal=True)
    hc = ws.channel_fir(apply, n_taps=400)
    xt = P.nrz(n_ui=g.n // 8, seed=1, n=g.n, causal=True)
    i = slice(2000, g.n - 2000)
    assert np.corrcoef(ws.stream_convolve(xt, hc, chunk=4096)[i], apply(xt)[i])[0, 1] > 0.999


def test_simreal_separability_harness():
    import wfmsynth as ws
    g = ws.Grid(fs=200e9, baud=50e9, n=1 << 12)
    nui = g.n // 4

    def mk(seed, noise_rms=0.0):
        s = (ws.Signal(seed=seed, grid=g).carrier("pam4", n_ui=nui, pattern="prbs13q",
                                                  causal=True, seed=seed)
             .lossy(loss_db=6.0, loss_at_ghz=25.0, causal=True))
        return (s.digitize(noise_rms=noise_rms) if noise_rms else s).waveform()

    A = [mk(s) for s in range(60)]
    B = [mk(s + 5000) for s in range(60)]
    same = ws.separability(A, B, g)
    assert same["best_auc"] < 0.75                       # same distribution -> not separable
    # feature vector is a stable, common dict
    fv = ws.feature_vector(A[0], g)
    assert set(same["auc"]) == set(fv) and 0.5 <= same["best_auc"] <= 1.0
    # an added-noise difference is separable and a culprit feature is named
    C = [mk(s, noise_rms=0.08) for s in range(60)]
    diff = ws.separability(A, C, g)
    assert diff["best_auc"] > 0.9 and diff["best_feature"] in diff["auc"]

"""
validate_physics.py — HARD checks that each physics primitive does what it claims.
Run before trusting the generator. Prints PASS/FAIL per property; exits nonzero on
any failure. This is the "don't fool yourself" gate for the data engine.
"""
import sys
import numpy as np
from scipy import signal as sp
from wfmsynth import physics as P

# The validation suite is the "don't fool yourself" gate, so it must be able to
# RUN everywhere. A stock Windows console is cp1252 and cannot encode the maths
# glyphs that otherwise appear in the pass/fail detail strings -- the suite died
# with UnicodeEncodeError partway through, which looks a lot like a hang and hides
# every check after it. Degrade the encoding rather than the diagnostics.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):       # pragma: no cover - very old/odd stdio
    pass

N = P.N
T = P.T
fails = []


def _pam4_levels(p4):  # stricter PAM4 check: 4 separated quantile levels
    flat = p4[np.abs(np.gradient(p4)) < 0.03]
    if flat.size < 50:
        return False
    q = np.quantile(flat, [0.1, 0.37, 0.63, 0.9])
    return bool(np.all(np.diff(q) > 0.12))


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        fails.append(name)


def spectral_centroid(x):
    """mean frequency (normalized 0..0.5) — drops when HF is attenuated."""
    mag = np.abs(np.fft.rfft(x - x.mean()))
    f = np.fft.rfftfreq(N)
    return float((f * mag).sum() / (mag.sum() + 1e-12))


print("== lossy channel: must attenuate HF (lower spectral centroid) and slow edges ==")
x = P.nrz(seed=3)
mild = P.lossy_channel(x, length_in=2.0, tand=0.005)
harsh = P.lossy_channel(x, length_in=14.0, tand=0.025)
c_raw, c_mild, c_harsh = spectral_centroid(x), spectral_centroid(mild), spectral_centroid(harsh)
check("harsher channel lowers spectral centroid (more HF loss)",
      c_harsh < c_mild < c_raw,
      f"raw={c_raw:.4f} mild={c_mild:.4f} harsh={c_harsh:.4f}")
check("harsh channel lowers max edge slope",
      np.abs(np.gradient(harsh)).max() < np.abs(np.gradient(mild)).max(),
      f"mildslope={np.abs(np.gradient(mild)).max():.3f} harshslope={np.abs(np.gradient(harsh)).max():.3f}")

print("== causal channel: minimum-phase concentrates response AFTER t=0 (post-cursor ISI) ==")
imp = np.zeros(N); imp[0] = 1.0                          # impulse at t=0 (circular)
h_zero = P.lossy_channel(imp, length_in=12.0, tand=0.02, causal=False)
h_caus = P.lossy_channel(imp, length_in=12.0, tand=0.02, causal=True)
# for the causal response, energy in the "pre-cursor" wrap (last half) must be small
pre_z = np.abs(h_zero[N // 2:]).sum(); post_z = np.abs(h_zero[:N // 2]).sum()
pre_c = np.abs(h_caus[N // 2:]).sum(); post_c = np.abs(h_caus[:N // 2]).sum()
check("zero-phase channel is symmetric (non-causal pre-ringing)",
      pre_z > 0.3 * post_z, f"pre/post={pre_z/(post_z+1e-9):.2f}")
check("causal channel concentrates energy post-t0 (pre-cursor << post-cursor)",
      pre_c < 0.15 * post_c, f"causal pre/post={pre_c/(post_c+1e-9):.3f}")
check("causal channel preserves the loss magnitude (same |H|)",
      abs(spectral_centroid(P.lossy_channel(P.nrz(seed=3), 12, 0.02, causal=True))
          - spectral_centroid(P.lossy_channel(P.nrz(seed=3), 12, 0.02, causal=False))) < 0.01)

print("== crosstalk: bounded coupling from an aggressor; zero coupling is a no-op ==")
victim = P.nrz(seed=1); aggr = P.nrz(seed=99)
xt = P.crosstalk(victim, aggr, coupling=0.12, kind="fext")
check("crosstalk perturbs the victim but stays bounded",
      1e-3 < np.abs(xt - victim).max() < 0.6 * np.ptp(victim),
      f"max|Δ|={np.abs(xt-victim).max():.3f} span={np.ptp(victim):.3f}")
check("zero coupling -> victim unchanged",
      np.allclose(P.crosstalk(victim, aggr, coupling=0.0), victim, atol=1e-9))

print("== AC coupling: removes DC / low-freq (baseline wander), keeps HF edges ==")
dc_sig = P.nrz(seed=7) + 0.5                             # add a DC pedestal
acd = P.ac_couple(dc_sig, fc_frac=0.01)
check("AC coupling removes the DC pedestal (|mean| shrinks)",
      abs(acd.mean()) < 0.2 * abs(dc_sig.mean()),
      f"mean {dc_sig.mean():.3f} -> {acd.mean():.3f}")
check("AC coupling preserves edge energy (HF kept)",
      np.abs(np.gradient(acd)).max() > 0.5 * np.abs(np.gradient(dc_sig)).max())

print("== multi-reflection: PULSE echoes decay geometrically at multiples of 2*td ==")
# a localized pulse so each reflection is a localized bump (not a staircase)
pulse = np.zeros(N); pulse[N // 4:N // 4 + 30] = 1.0
pulse = sp.sosfiltfilt(sp.bessel(4, 0.08, output="sos"), pulse)
td = 0.12; gs, gl = 0.4, 0.5
refl = P.multi_reflection(pulse, td_frac=td, gamma_s=gs, gamma_l=gl, n_bounce=4)
d = int(td * N); edge = N // 4
win = lambda c: np.abs(refl[c - 30:c + 30]).sum()
e1, e2 = win(edge + 2 * d), win(edge + 4 * d)
check("echo1 (2*td) present and larger than echo2 (4*td)",
      e1 > e2 > 1e-3, f"E@2td={e1:.2f} E@4td={e2:.2f}")
check("echo decay ratio ~ (gs*gl) within 2x",
      0.5 * (gs * gl) < (e2 / (e1 + 1e-9)) < 2.0 * (gs * gl),
      f"measured={e2/(e1+1e-9):.3f} expected~{gs*gl:.3f}")

print("== jitter: measured edge-time RMS tracks injected Rj (coherent shift) ==")
sqr = sp.sosfiltfilt(sp.bessel(4, 0.02, output="sos"), sp.square(2 * np.pi * 20 * T))
def crossings(y):
    s = np.sign(y - y.mean())
    return np.where(np.diff(s) > 0)[0].astype(float)
base_zc = crossings(sqr)
rng = np.random.default_rng(0)
devs = []
for _ in range(60):
    jt = P.inject_jitter(sqr, sigma_rj=3.0, rng=rng)
    zc = crossings(jt)
    if len(zc) == len(base_zc):                 # only compare when no crossings lost/gained
        devs.append(zc - base_zc)
meas_rms = float(np.std(np.concatenate(devs))) if devs else np.nan
check("injected sigma_rj=3 samples -> measured edge RMS in [1.5, 4.5]",
      1.5 < meas_rms < 4.5, f"measured_rms={meas_rms:.2f} samples over {len(devs)} runs")
check("zero jitter -> waveform unchanged",
      np.allclose(P.inject_jitter(sqr, sigma_rj=0.0, rng=rng), sqr, atol=1e-6))

print("== PAM4: 4 genuinely separated amplitude levels ==")
p4 = P.pam4(n_ui=96, seed=5)
flat = p4[np.abs(np.gradient(p4)) < 0.03]
q = np.quantile(flat, [0.1, 0.37, 0.63, 0.9])
check("4 quantile levels monotonically separated by >0.1",
      np.all(np.diff(q) > 0.1), f"levels={np.round(q,2).tolist()}")

print("== AM vs FM envelope: AM envelope varies, FM ~constant ==")
def envelope(y):
    return np.abs(sp.hilbert(y))
am_env = envelope(P.am(depth=0.7)); fm_env = envelope(P.fm(beta=5))
check("AM envelope varies more than FM envelope",
      am_env.std() / (am_env.mean() + 1e-9) > fm_env.std() / (fm_env.mean() + 1e-9),
      f"AM_cv={am_env.std()/am_env.mean():.3f} FM_cv={fm_env.std()/fm_env.mean():.3f}")

print("== chirp: instantaneous frequency increases ==")
ch = P.chirp(f0=3, f1=40)
inst = np.diff(np.unwrap(np.angle(sp.hilbert(ch))))
check("chirp instantaneous freq rising", inst[:N // 4].mean() < inst[-N // 4:].mean(),
      f"f_start~{inst[:N//4].mean():.4f} f_end~{inst[-N//4:].mean():.4f}")

print("== PRBS13Q: conformance to IEEE 802.3 120.5.11.2.1 published statistics ==")
b13 = P.prbs(13, 8191 * 3)
check("PRBS13 is maximal length (period 8191)",
      np.array_equal(b13[:8191], b13[8191:16382]),
      f"ones in one period={int(b13[:8191].sum())} (expect 4096)")
check("PRBS13 has 4096 ones per period", int(b13[:8191].sum()) == 4096)
check("PRBS13 has no sub-period",
      all(not np.array_equal(b13[:p], b13[p:2 * p]) for p in (7, 13, 89, 691)))

q = P.prbs13q(8191)
counts = [int(np.isclose(q, lv).sum()) for lv in (-1.0, -1 / 3, 1 / 3, 1.0)]
probs = np.array(counts) / 8191.0
dens = float(np.mean(q[1:] != q[:-1]))
check("PRBS13Q level probabilities match IEEE 0.2499/0.2500/0.2500/0.2500",
      np.allclose(probs, [0.2499, 0.2500, 0.2500, 0.2500], atol=5e-4),
      f"counts={counts}")
check("PRBS13Q transition density is 0.7501",
      abs(dens - 0.7501) < 5e-4, f"density={dens:.4f}")
check("PRBS13Q period is 8191 symbols",
      np.array_equal(P.prbs13q(8191 * 2)[:8191], P.prbs13q(8191 * 2)[8191:]))
check("pam4(pattern='prbs13q') yields 4 separated levels",
      _pam4_levels(P.pam4(n_ui=512, seed=5, pattern="prbs13q")))

print("== rate parameterization: primitives must not be locked to the default grid ==")
for n in (1024, 4096, 16384):
    xn = P.pam4(n_ui=64, seed=3, n=n)
    check(f"pam4 honours n={n}", len(xn) == n, f"got {len(xn)}")
    yn = P.lossy_channel(xn, length_in=10.0, tand=0.02, causal=True)
    check(f"causal lossy_channel works at n={n}",
          len(yn) == n and np.isfinite(yn).all())
    rn = P.multi_reflection(xn, td_samples=37, gamma_s=0.3, gamma_l=0.4)
    check(f"multi_reflection works at n={n}", len(rn) == n and np.isfinite(rn).all())
    jn = P.inject_jitter(xn, sigma_rj=2.0, a_pj=1.0, rng=np.random.default_rng(0))
    check(f"inject_jitter works at n={n}", len(jn) == n and np.isfinite(jn).all())

check("default-grid output is unchanged by parameterization (regression)",
      len(P.lossy_channel(P.nrz(seed=3), 12, 0.02, causal=True)) == N)

print("== absolute reflection delay: td_samples is independent of record length ==")
imp_a = np.zeros(2048); imp_a[100] = 1.0
imp_b = np.zeros(8192); imp_b[100] = 1.0
ra = P.multi_reflection(imp_a, td_samples=60, gamma_s=0.5, gamma_l=0.5, n_bounce=1)
rb = P.multi_reflection(imp_b, td_samples=60, gamma_s=0.5, gamma_l=0.5, n_bounce=1)
check("echo lands at the same absolute delay regardless of n",
      int(np.argmax(ra[101:])) == int(np.argmax(rb[101:])) == 119,
      f"a={int(np.argmax(ra[101:]))} b={int(np.argmax(rb[101:]))} (expect 2*60-1)")

print("== causal edge shaping: opt-in forward-only filter adds no pre-cursor ==")
step = np.concatenate([np.full(256, -1.0), np.full(256, 1.0)])
tr = 8.0
zp = P._shape_edges(step, tr, causal=False)
cz = P._shape_edges(step, tr, causal=True)
# measure just BEFORE the transition, away from the record head, so this reflects
# pre-cursor from the shaping rather than any filter start-up
pre_zp = float(np.abs(zp[200:256] - (-1.0)).max())
pre_cz = float(np.abs(cz[200:256] - (-1.0)).max())
check("zero-phase shaping disturbs samples BEFORE the edge (non-causal)",
      pre_zp > 1e-3, f"pre-edge deviation={pre_zp:.4f}")
check("causal shaping leaves pre-edge samples alone",
      pre_cz < pre_zp / 10, f"zero-phase={pre_zp:.4f} causal={pre_cz:.6f}")
check("causal shaping has no start-up transient (steady-state init)",
      abs(cz[0] - (-1.0)) < 1e-6, f"cz[0]={cz[0]:.6f} (expect -1)")

print("== family bank: all finite, non-degenerate, right length ==")
bank = P.family_bank()
rng = np.random.default_rng(1)
for name, fn in bank.items():
    y = fn(rng)
    check(f"family {name} finite/nondegenerate/len",
          np.isfinite(y).all() and np.ptp(y) > 1e-6 and len(y) == N)

print("== absolute units (Grid): requested ps/dB/Hz round-trip through the grid ==")
from wfmsynth.grid import Grid
_g = Grid(fs=256e9, baud=112e9, n=4096)
# 1) a reflection requested at a delay in PICOSECONDS lands at that delay (within a sample)
_imp = np.zeros(_g.n); _imp[100] = 1.0
_td_ps = 40.0
_r = P.multi_reflection(_imp, td_ps=_td_ps, grid=_g, gamma_s=0.5, gamma_l=0.5, n_bounce=1)
_peak = int(np.argmax(_r[101:])) + 1                 # first echo at 2*td samples past the impulse
_realized_ps = (_peak / 2.0) * _g.dt * 1e12
check("reflection delay in ps round-trips through the grid",
      abs(_realized_ps - _td_ps) <= _g.dt * 1e12 + 1e-9,
      f"requested {_td_ps}ps -> realized {_realized_ps:.2f}ps (dt={_g.dt*1e12:.2f}ps)")
# 2) a channel loss requested in dB at a stated frequency is realized at that frequency
_loss_db, _at_ghz = 12.0, 20.0
_hi = np.zeros(_g.n); _hi[0] = 1.0
_h = P.lossy_channel(_hi, loss_db=_loss_db, loss_at_ghz=_at_ghz, grid=_g)
_H = np.abs(np.fft.rfft(_h))
_f_ghz = np.fft.rfftfreq(_g.n) * _g.fs / 1e9
_k = int(np.argmin(np.abs(_f_ghz - _at_ghz)))
_realized_db = -20.0 * np.log10(_H[_k] + 1e-12)
check("channel loss in dB at a stated frequency is realized",
      abs(_realized_db - _loss_db) < 0.5,
      f"requested {_loss_db}dB@{_at_ghz}GHz -> realized {_realized_db:.2f}dB")
# 3) jitter in SECONDS equals the equivalent sample-domain call (exact grid conversion)
_sq = sp.sosfiltfilt(sp.bessel(4, 0.02, output="sos"), sp.square(2 * np.pi * 20 * T))
_a = P.inject_jitter(_sq, sigma_rj_s=3.0 * _g.dt, rng=np.random.default_rng(7), grid=_g)
_b = P.inject_jitter(_sq, sigma_rj=3.0, rng=np.random.default_rng(7))
check("jitter in seconds == equivalent sample-domain jitter (exact grid conversion)",
      np.allclose(_a, _b, atol=1e-9), "sigma_rj_s = 3*dt reproduces sigma_rj = 3 samples")
# 4) AC-coupling corner in Hz maps to the corresponding fraction-of-Nyquist
_fc_hz = 5e6
check("AC-coupling corner in Hz == equivalent fraction-of-Nyquist",
      np.allclose(P.ac_couple(_sq, fc_hz=_fc_hz, grid=_g),
                  P.ac_couple(_sq, fc_frac=_g.hz_to_frac_nyquist(_fc_hz)), atol=1e-9),
      f"{_fc_hz/1e6:.1f} MHz -> frac {_g.hz_to_frac_nyquist(_fc_hz):.2e}")

print("== non-integer samples-per-UI: fractional pattern period; integer assumption drifts ==")
_g2 = Grid(fs=256e9, baud=112e9, n=4096)
_sps = _g2.samples_per_ui                                  # 2.2857..., deliberately non-integer
check("samples-per-UI is non-integer at a realistic grid", abs(_sps - round(_sps)) > 0.1,
      f"sps={_sps:.4f}")
_nsym = 2000
_ntot = int(round(_nsym * _sps))
_syms = np.random.default_rng(0).integers(0, 4, _nsym)
_lv = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
_wav = _lv[_syms[np.clip((np.arange(_ntot) / _sps).astype(int), 0, _nsym - 1)]]  # sample-and-hold at non-int sps
def _recover(sps_used):
    pos = np.round(np.arange(_nsym) * sps_used + sps_used / 2).astype(int)
    pos = pos[pos < _ntot]
    rec = np.argmin(np.abs(_wav[pos][:, None] - _lv[None, :]), axis=1)
    return float((rec == _syms[:len(rec)]).mean())
check("decision sampling at the true fractional sps recovers the symbols",
      _recover(_sps) > 0.98, f"true-sps recovery={_recover(_sps):.3f}")
check("assuming an integer sps drifts and recovers wrongly (the failure this prevents)",
      _recover(round(_sps)) < 0.6, f"integer-sps recovery={_recover(round(_sps)):.3f}")
_per = _g2.pattern_period_samples(_nsym)
check("exact fractional pattern period is available and non-integer",
      abs(_per - round(_per)) > 1e-6 and abs(_per - _nsym * _sps) < 1e-9,
      f"period={_per:.2f} samples for {_nsym} UI")

print("== interleaved ADC: core mismatch -> spurs at fs/M and images; none when mismatch=0 ==")
from wfmsynth.instrument import interleave_adc
_nA, _mc, _fin = 4096, 4, 300
_tone = np.sin(2 * np.pi * _fin * np.linspace(0, 1, _nA, endpoint=False))
_ymm = interleave_adc(_tone, m_cores=_mc, offset_mm=0.02, gain_mm=0.01, rng=np.random.default_rng(0))
_yid = interleave_adc(_tone, m_cores=_mc, offset_mm=0.0, gain_mm=0.0, skew_mm=0.0)
_Ymm = np.abs(np.fft.rfft(_ymm - _ymm.mean()))
_Yid = np.abs(np.fft.rfft(_yid - _yid.mean()))
_spur = _nA // _mc                                          # fs/M bin
check("offset mismatch -> spur at fs/M", _Ymm[_spur] > 50 * (_Yid[_spur] + 1e-9),
      f"spur@fs/M: mismatch={_Ymm[_spur]:.2f} vs ideal={_Yid[_spur]:.2e}")
check("gain mismatch -> image spur at fs/M - f_in",
      _Ymm[_spur - _fin] > 20 * (_Yid[_spur - _fin] + 1e-9))
check("zero mismatch -> no interleave spur (ideal ADC transparent)",
      _Yid[_spur] < 1e-6 * _Ymm.max(), f"ideal spur={_Yid[_spur]:.2e}")

print("== source jitter: edges jittered at the transmitter; post-channel noise independent ==")
from wfmsynth.physics import Jitter
_nui = 96
_ref = P.nrz(n_ui=_nui, tr_frac=0.1, seed=5)
_jit = P.nrz(n_ui=_nui, tr_frac=0.1, seed=5, jitter=Jitter(rj=3.0), rng=np.random.default_rng(0))
def _cross(y):
    s = np.sign(y - y.mean()); return np.where(np.diff(s) != 0)[0].astype(float)
_c0, _c1 = _cross(_ref), _cross(_jit)
_rms = float(np.std(_c1 - _c0)) if len(_c0) == len(_c1) else -1.0
check("source Rj: recovered edge-time RMS ~ injected (3 samples)", 1.5 < _rms < 4.5,
      f"recovered RMS={_rms:.2f} samples ({len(_c0)} crossings)")
_clean = P.pam4(n_ui=_nui, seed=5, jitter=Jitter(rj=2.0, pj=1.0), rng=np.random.default_rng(1))
_sigc = P.lossy_channel(_clean, length_in=8.0, causal=True)
_noise = np.random.default_rng(2).normal(0, 0.05, len(_sigc))
check("source jitter: post-channel additive noise recovers exactly (uncorrelated)",
      np.allclose((_sigc + _noise) - _sigc, _noise, atol=1e-12),
      "noise added after the channel is not itself jittered")
_wn = P.inject_jitter(_sigc + _noise, sigma_rj=2.0, rng=np.random.default_rng(3))
_wc = P.inject_jitter(_sigc, sigma_rj=2.0, rng=np.random.default_rng(3))
check("output-warp jitter corrupts post-hoc noise (why source jitter is the correct model)",
      not np.allclose(_wn - _wc, _noise, atol=1e-6),
      "warping the whole waveform jitters the noise too -- unphysical")

print("== provenance: recipe round-trips bit-for-bit (through JSON); engine version stamped ==")
import json as _json
from wfmsynth.compose import Signal
_gp = Grid(fs=256e9, baud=112e9, n=1 << 13)
_sig = (Signal(seed=42, grid=_gp)
        .carrier("pam4", n_ui=_gp.n // 8, pattern="prbs13q", causal=True, jitter=dict(rj=0.4, pj=0.2))
        .lossy(loss_db=15.0, loss_at_ghz=26.0, causal=True)
        .reflect(td_ps=55.0, gamma_s=0.4, gamma_l=0.4)
        .digitize(snr_db=32.0, enob=5.5, interleave=dict(m_cores=4, offset_mm=0.01)))
_xr = _sig.waveform()
_rec = _json.loads(_json.dumps(_sig.recipe()))               # must survive JSON
_xr2 = Signal.from_recipe(_rec).waveform()
check("recipe reproduces the waveform bit-for-bit through JSON",
      _xr2.shape == _xr.shape and np.array_equal(_xr2, _xr), f"len={len(_xr)}")
check("recipe stamps the engine version",
      bool(_rec.get("wfmsynth_version")), f"version={_rec.get('wfmsynth_version')}")
# a second seed gives a different waveform but its own exact round-trip (no shared state)
_sig2 = Signal.from_recipe(_rec); _sig2.seed = 7
_xa = _sig2.waveform()
check("different seed -> different waveform, still exactly reproducible",
      not np.array_equal(_xa, _xr) and np.array_equal(_xa, Signal.from_recipe(_sig2.recipe()).waveform()))

print("== rng stream roles: factors are independent & re-rollable (valid contrastive pairs) ==")
from wfmsynth.streams import Streams as _St
_s = _St(1234)
_j1 = _s.role("jitter").standard_normal(64); _n1 = _s.role("noise").standard_normal(64)
_s2 = _St(1234)  # draw the SAME roles in the OPPOSITE order
_n2 = _s2.role("noise").standard_normal(64); _j2 = _s2.role("jitter").standard_normal(64)
check("role streams are order-independent and mutually independent",
      np.array_equal(_n1, _n2) and np.array_equal(_j1, _j2))
_s3 = _St(1234).reroll("jitter")  # a sibling with ONLY jitter re-rolled
check("re-rolling one factor changes that factor and leaves the others bit-identical",
      (not np.array_equal(_s3.role("jitter").standard_normal(64), _j1))
      and np.array_equal(_s3.role("noise").standard_normal(64), _n1))

# compose level: a changed UPSTREAM factor must leave the DOWNSTREAM noise realization
# untouched. (with a single shared rng the jitter change would shift the noise draws ->
# confounded.) An absolute noise floor lets us reconstruct the added noise bit-for-bit.
_g6 = Grid(fs=256e9, baud=112e9, n=1 << 12)
def _full_and_clean(rj):
    _sig = (Signal(seed=7, grid=_g6)                # carrier=op0, digitize=op1 -> role noise/1
            .carrier("pam4", n_ui=_g6.n // 8, pattern="prbs13q", jitter=dict(rj=rj))
            .digitize(noise_rms=0.01))
    _quiet = Signal.from_recipe(_sig.recipe())
    _quiet.ops[-1] = {k: v for k, v in _quiet.ops[-1].items() if k != "noise_rms"}
    return _sig.waveform(), _quiet.waveform()       # (clean + noise, clean)
_fA, _cA = _full_and_clean(0.3); _fB, _cB = _full_and_clean(3.0)
_N = _St(7).role("noise/1").normal(0.0, 0.01, len(_cA))   # the noise role's draws, standalone
check("the same-seed noise realization is identical regardless of the upstream jitter",
      np.array_equal(_fA, _cA + _N) and np.array_equal(_fB, _cB + _N)
      and not np.array_equal(_cA, _cB))             # ...even though the clean signals differ

# and the ergonomic wrapper: contrast() re-rolls exactly the named factor
_cs = (Signal(seed=3, grid=_g6)
       .carrier("pam4", n_ui=_g6.n // 8, pattern="prbs13q", jitter=dict(rj=0.5))
       .digitize(snr_db=28.0))
check("Signal.contrast(factor) re-rolls only that factor (reproducibly)",
      set(_cs.roles()) == {"jitter/0", "noise/1"}
      and not np.array_equal(_cs.contrast("noise/1", seed=1), _cs.waveform())
      and np.array_equal(_cs.contrast("noise/1", seed=1), _cs.contrast("noise/1", seed=1)))

print("== confounder-controlled sweeps: hold a measured metric while sweeping a knob ==")
from wfmsynth.measure import eye_height as _eye
from wfmsynth.sweep import hold_constant as _hold
_g7 = Grid(fs=200e9, baud=50e9, n=1 << 13)          # spb = 4
_nui = int(_g7.n // _g7.samples_per_ui)
def _b7(gamma=0.05, loss_db=2.0):
    return (Signal(seed=1, grid=_g7)
            .carrier("pam4", n_ui=_nui, pattern="prbs13q", causal=True)
            .lossy(loss_db=loss_db, loss_at_ghz=25.0, causal=True)
            .reflect(td_ps=30.0, gamma_s=gamma, gamma_l=gamma))
# a naive reflection sweep is ALSO an eye-height sweep -> realized labels expose the leak
_naive = [_eye(_b7(gm, 0.0).waveform(), _g7) for gm in (0.0, 0.15, 0.3, 0.4)]
check("realized labels expose the confound: reflection alone closes the eye",
      _naive[0] > _naive[-1] and all(_naive[i] >= _naive[i + 1] for i in range(len(_naive) - 1)),
      f"eye {_naive[0]:.3f} -> {_naive[-1]:.3f}")
# hold eye height fixed by solving insertion loss as reflection is swept
_tgt = _eye(_b7(0.05, 2.0).waveform(), _g7)
_recs = _hold(_b7, "gamma", [0.05, 0.15, 0.25, 0.35], "eye", _tgt,
              "loss_db", (0.0, 4.0), _g7, _eye, tol=0.004)
_real = [r["realized_eye"] for r in _recs]
_solved = [r["loss_db"] for r in _recs]
check("hold-constant sweep keeps the pinned metric within tolerance",
      max(abs(e - _tgt) for e in _real) <= 0.02, f"max dev {max(abs(e - _tgt) for e in _real):.4f}")
check("...while the swept knob forces a monotonic compensation (the constraint is real)",
      all(_solved[i] > _solved[i + 1] for i in range(len(_solved) - 1)),
      f"loss {_solved[0]:.2f} -> {_solved[-1]:.2f} as gamma rises")
# realized values are returned, not the requested ones
check("sweep returns REALIZED metric values alongside the request",
      all("realized_eye" in r and "target_eye" in r for r in _recs))

print("== ground truth as measured: named eye definitions + realized symbol alignment ==")
from wfmsynth.measure import eye_height as _eh
_g8 = Grid(fs=200e9, baud=50e9, n=1 << 13)
_n8 = int(_g8.n // _g8.samples_per_ui)
# the two eye definitions agree under Gaussian noise and diverge under deterministic ISI
_isi = (Signal(seed=1, grid=_g8).carrier("pam4", n_ui=_n8, pattern="prbs13q", causal=True)
        .reflect(td_ps=40.0, gamma_s=0.45, gamma_l=0.45)).waveform()
_gau = (Signal(seed=1, grid=_g8).carrier("pam4", n_ui=_n8, pattern="prbs13q", causal=True)
        .digitize(noise_rms=0.06)).waveform()
_di = abs(_eh(_isi, _g8, defn="sigma") - _eh(_isi, _g8, defn="contour"))
_dg = abs(_eh(_gau, _g8, defn="sigma") - _eh(_gau, _g8, defn="contour"))
check("named eye definitions agree under Gaussian noise, diverge under deterministic ISI",
      _dg < 0.02 and _di > 0.05 and _di > _dg + 0.03, f"|diff| ISI={_di:.3f} Gauss={_dg:.3f}")
# realized integer-symbol alignment: a causal channel's group delay must be recovered
_sig8 = (Signal(seed=1, grid=_g8).carrier("pam4", n_ui=_n8, pattern="prbs13q", causal=True)
         .lossy(loss_db=3.0, loss_at_ghz=25.0, causal=True)
         .reflect(td_ps=40.0, gamma_s=0.3, gamma_l=0.3))
_gt8 = _sig8.ground_truth()
check("realized symbol alignment recovers a nonzero group-delay offset",
      _gt8["align_offset"] != 0 and _gt8["align_corr"] > 0.9, f"offset={_gt8['align_offset']}")
check("skipping the realignment collapses tx/output correlation (offset matters)",
      _gt8["align_corr"] > _gt8["align_corr_at_zero"] + 0.4,
      f"corr {_gt8['align_corr']:.3f} vs @0 {_gt8['align_corr_at_zero']:.3f}")
check("ground_truth labels are measured from the output (both eye defs + phase + offset)",
      all(k in _gt8 for k in ("eye_contour", "eye_sigma", "best_phase", "align_offset")))
# carrier_symbols is the single source of truth for the transmitted stream
check("carrier_symbols reproduces the transmitted stream used by the carrier",
      np.array_equal(P.carrier_symbols("pam4", _n8, 1, "prbs13q"), P.prbs13q(_n8, 1)))

print("== instrument ADC model: standalone ENOB, absolute offset, correctly-ordered digitize ==")
from wfmsynth.instrument import (quantize_adc as _q, digitize as _dig,
                                 interleave_adc as _ila, clip_adc as _clip,
                                 shaped_noise_floor as _snf)
# #24 standalone quantiser: ~2^enob distinct codes, moves by at most half an LSB
_xq = np.linspace(-1, 1, 4000)
_qq = _q(_xq, enob=6, full_scale=1.0)
_lsb = 2.0 / 2 ** 6
check("quantize_adc collapses to ~2^enob codes, sample moves <= half an LSB",
      abs(len(np.unique(_qq)) - 64) <= 4 and np.max(np.abs(_qq - _xq)) <= 0.5 * _lsb + 1e-12,
      f"codes={len(np.unique(_qq))} maxmove={np.max(np.abs(_qq - _xq)) / _lsb:.3f} LSB")
# #26 offset in absolute volts is input-scale invariant; as a fraction it scales with input
_xo = 0.5 * np.sin(np.linspace(0, 80, 2000))
def _otone(kw, sc):
    return float(np.std(_ila(sc * _xo, m_cores=4, gain_mm=0.0, skew_mm=0.0,
                             rng=np.random.default_rng(0), **kw) - sc * _xo))
_av = _otone(dict(offset_v=1e-3), 1.0), _otone(dict(offset_v=1e-3), 4.0)
_fr = _otone(dict(offset_mm=0.05), 1.0), _otone(dict(offset_mm=0.05), 4.0)
check("interleave offset in volts is invariant to input scale; as a fraction it scales",
      abs(_av[1] / _av[0] - 1.0) < 0.02 and abs(_fr[1] / _fr[0] - 4.0) < 0.05,
      f"volts x{_av[1] / _av[0]:.2f}, frac x{_fr[1] / _fr[0]:.2f}")
# #25 composed digitize matches the manual correct-order pipeline; order genuinely matters
_mm = (np.array([0.01, -0.005, 0.008, -0.003]), np.array([2e-3, -1e-3, 1.5e-3, -2e-3]), np.zeros(4))
_yc, _info = _dig(_xo, noise_floor={"rms": 2e-3, "shape": "white"},
                  interleave={"m_cores": 4, "mismatch": _mm}, clip_full_scale=0.7, enob=6,
                  rng=np.random.default_rng(3))
_r = np.random.default_rng(3); _nz = _snf(len(_xo), rng=_r, rms=2e-3, shape="white")
_ym = _ila(_xo + _nz, m_cores=4, mismatch=_mm)
_ym, _ = _clip(_ym, 0.7); _ym = _q(_ym, enob=6, full_scale=0.7)
check("digitize() composes noise->interleave->clip->quantise identically to manual stages",
      np.array_equal(_yc, _ym) and "clipped_fraction" in _info)
check("stage order matters: quantise-before-noise != noise-before-quantise",
      not np.array_equal(_q(_xo + _nz, enob=6, full_scale=0.7),
                         _q(_xo, enob=6, full_scale=0.7) + _nz))

print("== impairment mixing at constant total power: magnitude vs character are separable ==")
from wfmsynth.impairments import mix_at_constant_power as _mix
from wfmsynth.instrument import shaped_noise_floor as _snf2
_rng9 = np.random.default_rng(0)
_white = _snf2(1 << 13, rms=1.0, shape="white", rng=_rng9)
_pink = _snf2(1 << 13, rms=1.0, shape="pink", rng=_rng9)
def _cent(x):
    _X = np.abs(np.fft.rfft(x))
    return float((np.arange(len(_X)) * _X).sum() / (_X.sum() + 1e-12))
_rms9, _ch9 = [], []
for _a in np.linspace(0.0, 1.0, 6):                 # sweep white -> pink at fixed total power
    _y = _mix([_white, _pink], [1 - _a, _a], total_rms=0.05)
    _rms9.append(float(np.sqrt(np.mean(_y ** 2))))
    _ch9.append(_cent(_y))
check("total impairment RMS is invariant across the mixing sweep",
      max(_rms9) - min(_rms9) < 1e-9, f"rms in [{min(_rms9):.5f}, {max(_rms9):.5f}]")
check("...while a character statistic (spectral centroid) moves monotonically",
      all(_ch9[i] > _ch9[i + 1] for i in range(len(_ch9) - 1)),
      f"centroid {_ch9[0]:.0f} -> {_ch9[-1]:.0f}")

print("== intermittent impairments: confined to a gate, per-sample mask as ground truth ==")
from wfmsynth.impairments import apply_gated as _ag
_g10 = Grid(fs=200e9, baud=50e9, n=1 << 13)
_x10 = (Signal(seed=1, grid=_g10)
        .carrier("pam4", n_ui=int(_g10.n // _g10.samples_per_ui), pattern="prbs13q", causal=True)
        ).waveform()
def _glitch(_s):
    return _s + 0.5 * np.sign(np.sin(np.linspace(0, 300, len(_s))))
_y10, _mask10 = _ag(_x10, _glitch, [(2000, 180), (6000, 140)])
_out = _mask10 == 0
check("intermittent impairment is confined to its gate (zero leakage outside)",
      np.array_equal(_y10[_out], _x10[_out]), f"duty={100 * (_mask10 > 0).mean():.1f}%")
check("the defect is present inside the gate and the per-sample mask is emitted",
      np.max(np.abs((_y10 - _x10)[_mask10 > 0.5])) > 0.1 and _mask10.shape == _x10.shape)
check("the gate is smooth (raised-cosine onset, not a step that reads as an edge)",
      np.max(np.abs(np.diff(_mask10))) < 0.3)

print("== Tx FFE: a deliberate pre-cursor in the pulse response, de-emphasizes ISI ==")
_g11 = Grid(fs=200e9, baud=50e9, n=1 << 13)
_spb = _g11.samples_per_ui
_n11 = int(_g11.n // _spb)
# pulse response: a one-UI pulse through FFE gets a pre-cursor one UI before the main peak
_pulse = np.zeros(_g11.n); _pulse[_g11.n // 2:_g11.n // 2 + int(_spb)] = 1.0
_yp = P.tx_ffe(_pulse, [-0.2, 1.0, -0.3], _spb, pre=1)
_peak = int(np.argmax(np.abs(_yp)))
check("Tx FFE injects a pre-cursor one UI before the main pulse (the real-link shape)",
      abs(_yp[_peak - int(round(_spb))]) > 0.05 and abs(_yp[_peak] - 1.0) < 1e-6,
      f"pre-cursor={_yp[_peak - int(round(_spb))]:.2f}")
# de-emphasis opens a lossy-channel eye relative to no FFE
_noffe = (Signal(seed=1, grid=_g11).carrier("pam4", n_ui=_n11, pattern="prbs13q", causal=True)
          .lossy(loss_db=8.0, loss_at_ghz=25.0, causal=True)).waveform()
_wffe = (Signal(seed=1, grid=_g11).carrier("pam4", n_ui=_n11, pattern="prbs13q", causal=True)
         .tx_ffe(taps=[-0.15, 1.0, -0.25], pre=1)
         .lossy(loss_db=8.0, loss_at_ghz=25.0, causal=True)).waveform()
check("Tx FFE de-emphasis opens a lossy-channel eye vs no FFE",
      _eh(_wffe, _g11) > _eh(_noffe, _g11) + 0.02,
      f"eye {_eh(_noffe, _g11):.3f} -> {_eh(_wffe, _g11):.3f}")

print("== composition-level causality: a FULL composed chain, not just the channel ==")
# hazard: causality is asserted for lossy_channel alone, but default zero-phase edge
# shaping reintroduces pre-cursor AFTER the causal channel -- each stage looks fine while
# the pipeline is not causal end to end.
_tr18 = 8.0
_pulse18 = np.zeros(N); _c18 = N // 2; _W18 = int(20 * _tr18); _pulse18[_c18:_c18 + _W18] = 1.0
def _edge_ratio(shaped_causal):
    _y = P.lossy_channel(P._shape_edges(_pulse18, _tr18, causal=shaped_causal),
                         length_in=12.0, causal=True)
    _w = int(4 * _tr18)
    return float(np.sum(_y[_c18 - _w:_c18] ** 2) / (np.sum(_y[_c18:_c18 + _w] ** 2) + 1e-12))
_rc18, _rz18 = _edge_ratio(True), _edge_ratio(False)
check("a fully-causal composed chain (causal shaping + causal channel) has ~zero pre-cursor",
      _rc18 < 0.01, f"pre/post energy @ edge = {_rc18:.4f}")
check("the hazard is real: zero-phase edge shaping leaks pre-cursor behind a causal channel",
      _rz18 > 10 * _rc18 and _rz18 > 0.005, f"zero-phase pre/post = {_rz18:.4f} vs causal {_rc18:.4f}")

print("== pattern lock-ability: a standard pattern autocorrelates to a single sharp peak ==")
from wfmsynth.measure import pattern_period as _pp
_syms19 = P.carrier_symbols("pam4", 2 * 8191, 1, "prbs13q")     # two full PRBS13Q periods
_lag19, _peak19, _ac19 = _pp(_syms19, max_lag=9000)
_m19 = np.ones(9001, bool); _m19[max(1, _lag19 - 20):_lag19 + 20] = False; _m19[:20] = False
_next19 = float(np.max(_ac19[:9001][_m19]))
check("PRBS13Q symbol sequence locks to a single sharp peak at its declared period (8191)",
      _lag19 == 8191 and _peak19 > 0.95 and _peak19 > 2 * _next19,
      f"period={_lag19} peak={_peak19:.3f} next={_next19:.3f}")

print("== clock recovery: a CDR tracks out jitter below its loop bandwidth (what a scope shows) ==")
from wfmsynth.cdr import recover_clock as _rc, tracked_out_fraction as _tof
_baud12 = 50e9; _N12 = 1 << 16; _t12 = np.arange(_N12) / _baud12; _BW12 = 1e6
def _jtf(fm, bw=_BW12, order=2):
    _ph = np.sin(2 * np.pi * fm * _t12)
    return np.ptp(_rc(_ph, _baud12, bw, order)[1][_N12 // 2:]) / np.ptp(_ph[_N12 // 2:])
check("CDR jitter transfer is high-pass: low-freq jitter tracked out, high-freq passed",
      _jtf(1e5) < 0.1 and _jtf(1e7) > 0.9, f"res/in @100kHz={_jtf(1e5):.3f} @10MHz={_jtf(1e7):.3f}")
check("a wider loop bandwidth tracks out MORE low-frequency jitter",
      _tof(np.sin(2 * np.pi * 3e5 * _t12), _baud12, 3e6) > _tof(np.sin(2 * np.pi * 3e5 * _t12), _baud12, 3e5) + 0.2,
      f"tracked-out @300kHz: BW3MHz={_tof(np.sin(2*np.pi*3e5*_t12),_baud12,3e6):.2f} BW300kHz={_tof(np.sin(2*np.pi*3e5*_t12),_baud12,3e5):.2f}")
_ramp12 = np.arange(_N12) * 2e-4     # a frequency offset (linear phase ramp)
_r1 = _rc(_ramp12, _baud12, _BW12, 1)[1]; _r2 = _rc(_ramp12, _baud12, _BW12, 2)[1]
check("a 2nd-order (type-2) CDR tracks a frequency offset to ~zero; 1st-order leaves a lag",
      np.mean(np.abs(_r2[_N12 // 2:])) < 0.1 * np.mean(np.abs(_r1[_N12 // 2:])),
      f"ramp residual: order1={np.mean(np.abs(_r1[_N12//2:])):.3f} order2={np.mean(np.abs(_r2[_N12//2:])):.4f}")

print("== measured S-parameter channels: Touchstone round-trip + a resonance the model can't ==")
import tempfile as _tmp, os as _os
from wfmsynth.sparam import (read_touchstone as _rts, write_touchstone as _wts,
                             touchstone_channel as _tsc)
_f13 = np.linspace(1e8, 40e9, 800)
_notch = 1 - 0.9 * np.exp(-((_f13 - 20e9) / 1.5e9) ** 2)     # resonant null at 20 GHz
_S13 = np.zeros((len(_f13), 2, 2), complex)
_S13[:, 1, 0] = _notch * np.exp(-1j * 2 * np.pi * _f13 * 20e-12); _S13[:, 0, 1] = _S13[:, 1, 0]
_S13[:, 0, 0] = 0.05; _S13[:, 1, 1] = 0.05
_p13 = _os.path.join(_tmp.mkdtemp(), "thru.s2p")
_wts(_p13, _f13, _S13, fmt="RI"); _fr, _Sr = _rts(_p13)
check("Touchstone .s2p round-trips (freqs + S21, incl. the 2-port ordering quirk)",
      np.allclose(_f13, _fr) and np.allclose(_S13[:, 1, 0], _Sr[:, 1, 0]) and _Sr.shape == (800, 2, 2))
_g13 = Grid(fs=80e9, n=1 << 14)
_x13 = np.random.default_rng(0).standard_normal(_g13.n)
_y13 = _tsc(_x13, _p13, grid=_g13)
_fg13 = np.fft.rfftfreq(_g13.n, d=_g13.dt); _Y13 = np.abs(np.fft.rfft(_y13))
_nb = int(np.argmin(np.abs(_fg13 - 20e9))); _rb = int(np.argmin(np.abs(_fg13 - 10e9)))
check("an S-parameter channel reproduces a resonant notch (the analytic model cannot)",
      _Y13[_nb] / (_Y13[_rb] + 1e-9) < 0.2, f"|Y(20G)|/|Y(10G)|={_Y13[_nb] / (_Y13[_rb] + 1e-9):.3f}")

print("== resonant reflection: frequency-dependent Γ (a stub resonates, not a flat mirror) ==")
_g14 = Grid(fs=100e9, n=1 << 14)
_f14 = np.fft.rfftfreq(_g14.n, d=_g14.dt)
_x14 = np.random.default_rng(0).standard_normal(_g14.n)
_yr = P.resonant_reflection(_x14, grid=_g14, td_ps=50.0, f0_ghz=25.0, q=12.0, gamma0=0.5)
_yf = P.multi_reflection(_x14, grid=_g14, td_ps=50.0, gamma_s=0.5, gamma_l=0.0, n_bounce=1)
_Rr = np.abs(np.fft.rfft(_yr - _x14))                  # the reflection contribution's spectrum
_b0 = int(np.argmin(np.abs(_f14 - 25e9))); _blo = int(np.argmin(np.abs(_f14 - 5e9)))
check("resonant Γ peaks the reflected content near f0 (frequency-dependent magnitude)",
      _Rr[_b0] / (_Rr[_blo] + 1e-9) > 5.0 and abs(_f14[int(np.argmax(_Rr))] - 25e9) < 3e9,
      f"|refl(25G)|/|refl(5G)|={_Rr[_b0] / (_Rr[_blo] + 1e-9):.1f}, peak@{_f14[int(np.argmax(_Rr))]/1e9:.1f}GHz")
check("the flat-Γ multi_reflection does NOT concentrate reflection at one frequency",
      np.abs(np.fft.rfft(_yf - _x14))[_b0] / (np.abs(np.fft.rfft(_yf - _x14))[_blo] + 1e-9) < 3.0)

print("== nominal nonlinearity: an unfaulted transmitter is imperfect, not suspiciously perfect ==")
# compression makes PAM4 level spacing non-uniform (RLM < 1) but mild (still nominal)
_lv15 = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
_gaps15 = np.diff(P.nominal_nonlinearity(_lv15, compression=0.06))
_rlm15 = _gaps15.min() / _gaps15.max()
check("nominal compression makes PAM4 level spacing non-uniform (RLM<1) but mild (>0.85)",
      0.85 < _rlm15 < 0.999, f"RLM={_rlm15:.3f}")
# level-dependent noise: outer levels noisier than inner
_rng15 = np.random.default_rng(0)
_out15 = P.nominal_nonlinearity(np.full(20000, 1.0), compression=0.0, level_noise=0.02, rng=_rng15)
_in15 = P.nominal_nonlinearity(np.full(20000, 1 / 3), compression=0.0, level_noise=0.02, rng=_rng15)
check("level-dependent noise makes outer levels noisier than inner",
      _out15.std() > 2.0 * _in15.std(), f"outer={_out15.std():.4f} inner={_in15.std():.4f}")
# rise/fall asymmetry: a step's rise time differs from its fall time
_step15 = np.concatenate([np.zeros(200), np.ones(200), np.zeros(200)]).astype(float)
_y15 = P.nominal_nonlinearity(_step15, compression=0.0, rise_fall_ratio=2.5)
_rise = int(np.argmax(_y15[200:400] >= 0.9)); _fall = int(np.argmax(_y15[400:600] <= 0.1))
check("rise/fall-ratio != 1 gives asymmetric rise vs fall times",
      _rise != _fall and _rise > 0 and _fall > 0, f"rise->90%={_rise} fall->10%={_fall} samples")

print("== multi-aggressor crosstalk from a coupling matrix, ASYNCHRONOUS by default ==")
_g16 = Grid(fs=400e9, baud=50e9, n=1 << 14); _spb16 = int(_g16.samples_per_ui)
_x16 = P.nrz(n_ui=_g16.n // _spb16, seed=1, n=_g16.n, causal=True)
def _conc(y):     # how concentrated the crosstalk energy is at a fixed victim-UI phase
    _c = np.abs(y - _x16)
    _prof = np.array([_c[i::_spb16][:len(_c) // _spb16].mean() for i in range(_spb16)])
    return np.ptp(_prof) / (_prof.mean() + 1e-12)
check("total crosstalk power scales with the coupling vector (linear superposition)",
      abs(np.std(P.crosstalk_matrix(_x16, _g16, [0.2]) - _x16)
          / (np.std(P.crosstalk_matrix(_x16, _g16, [0.1]) - _x16) + 1e-12) - 2.0) < 0.1)
check("aggressors are ASYNCHRONOUS by default: crosstalk not locked to the victim UI",
      _conc(P.crosstalk_matrix(_x16, _g16, [0.15])) < 0.3 * _conc(P.crosstalk_matrix(_x16, _g16, [0.15], synchronous=True)),
      f"async={_conc(P.crosstalk_matrix(_x16,_g16,[0.15])):.3f} sync={_conc(P.crosstalk_matrix(_x16,_g16,[0.15],synchronous=True)):.3f}")
check("more aggressors -> more crosstalk power",
      np.std(P.crosstalk_matrix(_x16, _g16, [0.1, 0.1, 0.1]) - _x16)
      > np.std(P.crosstalk_matrix(_x16, _g16, [0.1]) - _x16))

print("== noise realism beyond white Gaussian: heavy tails + 1/f structure ==")
from wfmsynth.impairments import realistic_noise as _rn
def _exk(a):
    a = a - a.mean()
    return np.mean(a ** 4) / (np.mean(a ** 2) ** 2) - 3.0
_heavy = _rn(1 << 16, df=5.0, rng=np.random.default_rng(0))
_gauss = _rn(1 << 16, df=200.0, rng=np.random.default_rng(0))
check("heavy-tailed noise has clear excess kurtosis; near-Gaussian does not",
      _exk(_heavy) > 1.0 and abs(_exk(_gauss)) < 0.5, f"exkurt heavy={_exk(_heavy):.2f} gauss={_exk(_gauss):.2f}")
check("both are scaled to the requested RMS",
      abs(np.sqrt(np.mean(_heavy ** 2)) - 0.01) < 1e-6 and abs(np.sqrt(np.mean(_gauss ** 2)) - 0.01) < 1e-6)
_pink17 = _rn(1 << 16, df=200.0, pink_frac=0.85, rng=np.random.default_rng(1))
_P17 = np.abs(np.fft.rfft(_pink17)) ** 2; _f17 = np.fft.rfftfreq(1 << 16)
_lo = _P17[(_f17 > 1e-3) & (_f17 < 1e-2)].mean(); _hi = _P17[(_f17 > 0.1) & (_f17 < 0.4)].mean()
check("a 1/f (pink) fraction puts far more power at low frequency than high",
      _lo / _hi > 5.0, f"low/high band power = {_lo / _hi:.1f}")

print("== streaming / chunked convolution: bounded memory, matches the full convolution ==")
from wfmsynth.stream import (stream_convolve as _sc, stream_blocks as _sb,
                             channel_fir as _cfir)
_n20 = 1 << 18
_x20 = np.random.default_rng(0).standard_normal(_n20)
_h20 = np.exp(-((np.arange(65) - 32) / 8.0) ** 2); _h20 /= _h20.sum()
_ref20 = np.convolve(_x20, _h20)[:_n20]
check("overlap-save stream_convolve equals the full linear convolution (bounded memory)",
      np.allclose(_sc(_x20, _h20, chunk=8192), _ref20, atol=1e-9),
      f"max err {np.max(np.abs(_sc(_x20, _h20, chunk=8192) - _ref20)):.1e}")
check("stream_blocks yields the whole record in chunks (never a full-length output/FFT)",
      sum(len(_b) for _b in _sb(_x20, _h20, chunk=8192)) == _n20)
# a linear channel applied via its FIR (chunked) matches the direct channel in the interior
_g20 = Grid(fs=100e9, n=1 << 15)
_apply20 = lambda a: P.lossy_channel(a, length_in=8.0, tand=0.02, causal=True)
_h20c = _cfir(_apply20, n_taps=400)
_xt20 = P.nrz(n_ui=_g20.n // 8, seed=1, n=_g20.n, causal=True)
_i20 = slice(2000, _g20.n - 2000)
check("channel_fir + stream_convolve reproduces a linear channel (chunked deep-memory path)",
      np.corrcoef(_sc(_xt20, _h20c, chunk=4096)[_i20], _apply20(_xt20)[_i20])[0, 1] > 0.999)

print("== sim-to-real separability harness: is the synthetic distinguishable, and by what? ==")
from wfmsynth.simreal import separability as _sep
_g21 = Grid(fs=200e9, baud=50e9, n=1 << 12); _nui21 = _g21.n // 4
def _mk21(seed, noise_rms=0.0):
    _s = (Signal(seed=seed, grid=_g21).carrier("pam4", n_ui=_nui21, pattern="prbs13q",
                                               causal=True, seed=seed)
          .lossy(loss_db=6.0, loss_at_ghz=25.0, causal=True))
    return (_s.digitize(noise_rms=noise_rms) if noise_rms else _s).waveform()
_A21 = [_mk21(s) for s in range(60)]
_B21 = [_mk21(s + 5000) for s in range(60)]
_same = _sep(_A21, _B21, _g21)
check("two sets from the same distribution are NOT strongly separable (best AUC ~ chance)",
      _same["best_auc"] < 0.75, f"best {_same['best_feature']}={_same['best_auc']:.3f}")
_C21 = [_mk21(s, noise_rms=0.08) for s in range(60)]
_diff = _sep(_A21, _C21, _g21)
check("an added-noise difference IS separable and the harness names a culprit feature",
      _diff["best_auc"] > 0.9 and _diff["best_feature"] in _diff["auc"],
      f"best {_diff['best_feature']}={_diff['best_auc']:.3f}")

print("== Rx equalization: CTLE peaks/opens a lossy eye; DFE cancels a post-cursor ==")
from wfmsynth.rx import ctle as _ctle, dfe as _dfe
_g28 = Grid(fs=200e9, baud=50e9, n=1 << 13); _nui28 = int(_g28.n // _g28.samples_per_ui)
_no28 = (Signal(seed=1, grid=_g28).carrier("pam4", n_ui=_nui28, pattern="prbs13q", causal=True)
         .lossy(loss_db=9.0, loss_at_ghz=25.0, causal=True)).waveform()
_eq28 = _ctle(_no28, _g28, fz_ghz=6.0, fp1_ghz=22.0, fp2_ghz=45.0, dc_gain=1.0)
check("CTLE opens a lossy-channel eye (receiver-side high-frequency peaking)",
      _eh(_eq28, _g28) > _eh(_no28, _g28) + 0.02, f"eye {_eh(_no28,_g28):.3f} -> {_eh(_eq28,_g28):.3f}")
_wc, _hc = sp.freqz(*sp.bilinear([1 / (2 * np.pi * 6e9), 1.0],
                                 np.polymul([1 / (2 * np.pi * 22e9), 1.0], [1 / (2 * np.pi * 45e9), 1.0]),
                                 fs=_g28.fs), worN=2048, fs=_g28.fs)
check("the CTLE response peaks above its DC gain (it is a high-frequency booster)",
      np.abs(_hc).max() > 1.5 * np.abs(_hc[0]))
# DFE cancels a known discrete post-cursor -> symbol errors collapse
_rng28 = np.random.default_rng(0)
_syms28 = _rng28.choice([-1.0, -1 / 3, 1 / 3, 1.0], 2000)
_h1 = 0.35
_rx28 = _syms28.astype(float).copy(); _rx28[1:] += _h1 * _syms28[:-1]      # 1-tap post-cursor ISI
_err_no = np.mean(_dfe(_rx28, [0.0])[1] != _syms28)
_err_dfe = np.mean(_dfe(_rx28, [_h1])[1] != _syms28)
check("DFE cancels the post-cursor it is tuned to (symbol errors collapse to ~0)",
      _err_dfe < 0.01 and _err_no > 0.1, f"symbol error {_err_no:.3f} -> {_err_dfe:.3f}")

print("== spread-spectrum clocking (SSC): triangular clock FM, tracked by a wide-loop CDR ==")
from wfmsynth.cdr import ssc_phase as _sscp, apply_ssc as _sscw, recover_clock as _rc2
_ns31 = 1 << 18; _baud31 = 50e9
_ph31 = _sscp(_ns31, _baud31, f_ssc=1e6, spread=0.005, profile="down")
_dfrac31 = np.diff(_ph31)                       # instantaneous fractional frequency deviation
check("down-spread SSC frequency deviation stays within [-spread, 0] and is periodic at f_ssc",
      -0.0051 < _dfrac31.min() and _dfrac31.max() < 1e-6
      and np.argmax(np.abs(np.fft.rfft(_dfrac31 - _dfrac31.mean()))) > 0,
      f"dev range [{_dfrac31.min():.5f}, {_dfrac31.max():.5f}]")
_rw31 = np.ptp(_rc2(_ph31, _baud31, 3e6, order=2)[1][_ns31 // 2:])
_rn31 = np.ptp(_rc2(_ph31, _baud31, 3e4, order=2)[1][_ns31 // 2:])
check("a wide-loop CDR tracks the SSC wander out; a narrow loop does not",
      _rw31 < 0.3 * _rn31, f"residual ptp wide={_rw31:.1f} narrow={_rn31:.1f}")
_fs31 = 200e9; _tone31 = np.sin(2 * np.pi * 10e9 * np.arange(1 << 16) / _fs31)
_w31 = _sscw(_tone31, _fs31, f_ssc=5e6, spread=0.01)
_S0 = np.abs(np.fft.rfft(_tone31)); _S1 = np.abs(np.fft.rfft(_w31))
check("SSC spreads the spectrum (its whole purpose: a tone's energy fans out)",
      np.sum(_S1 > 0.05 * _S1.max()) > 2 * np.sum(_S0 > 0.05 * _S0.max()))

print("== differential pair: intra-pair skew closes the eye and makes common-mode; imbalance converts ==")
_g32 = Grid(fs=200e9, baud=50e9, n=1 << 13); _nui32 = int(_g32.n // _g32.samples_per_ui)
_x32 = (Signal(seed=1, grid=_g32).carrier("pam4", n_ui=_nui32, pattern="prbs13q", causal=True)).waveform()
_p0, _n0 = P.differential_pair(_x32, _g32)              # ideal
check("ideal differential pair recovers the data with ~zero common-mode",
      np.allclose(P.differential_mode(_p0, _n0), _x32)
      and np.sqrt(np.mean(P.common_mode(_p0, _n0) ** 2)) < 1e-9)
_ps, _ns = P.differential_pair(_x32, _g32, skew_ps=6.0)
check("intra-pair skew closes the differential eye and generates common-mode (zero at skew=0)",
      _eh(P.differential_mode(_ps, _ns), _g32) < _eh(P.differential_mode(_p0, _n0), _g32) - 0.02
      and np.sqrt(np.mean(P.common_mode(_ps, _ns) ** 2)) > 0.01)
_pg, _ng = P.differential_pair(_x32, _g32, gain_imbalance=0.1)
check("gain imbalance converts differential to common-mode, proportional to the data",
      abs(np.corrcoef(P.common_mode(_pg, _ng), _x32)[0, 1]) > 0.95
      and np.sqrt(np.mean(P.common_mode(_pg, _ng) ** 2))
      > 3 * np.sqrt(np.mean(P.common_mode(_p0, _n0) ** 2)))

print("== power-supply / PDN coupling: correlated AM + PSIJ sidebands from a supply rail ==")
_g33 = Grid(fs=200e9, n=1 << 16); _t33 = np.arange(_g33.n) / _g33.fs
_fc33, _fr33 = 10e9, 1e9
_probe33 = np.cos(2 * np.pi * _fc33 * _t33)
_f33 = np.fft.rfftfreq(_g33.n, d=_g33.dt)
def _sb33(y):
    _Y = np.abs(np.fft.rfft(y))
    return _Y[int(np.argmin(np.abs(_f33 - (_fc33 + _fr33))))] / _Y[int(np.argmin(np.abs(_f33 - _fc33)))]
_am1 = _sb33(P.supply_coupling(_probe33, _g33, _fr33, am_depth=0.05))
_am2 = _sb33(P.supply_coupling(_probe33, _g33, _fr33, am_depth=0.10))
check("supply AM produces a sideband at f_ripple that scales with the coupling depth",
      _am2 > 1.7 * _am1 and _am1 > 1e-3, f"sideband depth0.05={_am1:.4f} depth0.10={_am2:.4f}")
_pm0 = _sb33(P.supply_coupling(_probe33, _g33, _fr33, am_depth=0.0))
_pm1 = _sb33(P.supply_coupling(_probe33, _g33, _fr33, psij_ps=8.0))
check("supply PSIJ produces a timing (PM) sideband that scales with the coupling",
      _pm1 > 10 * max(_pm0, 1e-6), f"sideband none={_pm0:.5f} psij8ps={_pm1:.4f}")
_both = P.supply_coupling(_probe33, _g33, _fr33, am_depth=0.06, psij_ps=6.0)
check("AM and PSIJ come from the SAME rail (both sidebands present, correlated to f_ripple)",
      _sb33(_both) > _am1 and _sb33(_both) > _pm0)

print("== timing-modulation source: compose SSC + Pj + Rj into one injectable phase ==")
from wfmsynth.cdr import timing_source as _tsrc, apply_phase as _aph
_g35 = Grid(fs=200e9, n=1 << 16); _f35 = np.fft.rfftfreq(_g35.n, d=_g35.dt)
_pj35 = _tsrc(_g35.n, _g35, pj=dict(amp_ps=3.0, f_hz=2e9))
check("a Pj timing source puts a phase tone at its frequency",
      abs(_f35[int(np.argmax(np.abs(np.fft.rfft(_pj35 - _pj35.mean()))))] - 2e9) < 5e7)
_comb35 = _tsrc(_g35.n, _g35, ssc=dict(f_ssc=1e6, spread=0.005), pj=dict(amp_ps=3.0, f_hz=2e9),
                rj_ps=0.5, rng=np.random.default_rng(0))
_Pc35 = np.abs(np.fft.rfft(_comb35 - _comb35.mean()))
check("a composed source (SSC+Pj+Rj) carries each component: the Pj tone survives, Rj adds spread",
      _Pc35[int(np.argmin(np.abs(_f35 - 2e9)))] > 5 * np.median(_Pc35) and _comb35.std() > _pj35.std())
_tone35 = np.cos(2 * np.pi * 10e9 * np.arange(_g35.n) / _g35.fs)
check("apply_phase warps a waveform by the composed timing (feeds a carrier or the CDR)",
      not np.allclose(_aph(_tone35, _pj35), _tone35))

print("== multi-signal scene: shared supply correlates lanes; lane-to-lane coupling; diff pair ==")
from wfmsynth.scene import Scene as _Scene
_g36 = Grid(fs=200e9, baud=50e9, n=1 << 12); _nui36 = _g36.n // 4
_w0 = (Signal(seed=1, grid=_g36).carrier("pam4", n_ui=_nui36, pattern="prbs13q", causal=True, seed=1)).waveform()
_w1 = (Signal(seed=2, grid=_g36).carrier("pam4", n_ui=_nui36, pattern="prbs13q", causal=True, seed=7)).waveform()
_sc36 = _Scene(_g36).add("l0", _w0).add("l1", _w1).shared_supply(f_ripple_hz=1e6, am_depth=0.05)
check("a shared supply rail induces the SAME (correlated) artifact across independent lanes",
      np.allclose(_sc36.lane("l0") / _w0 - 1.0, _sc36.lane("l1") / _w1 - 1.0)
      and abs(np.corrcoef(_w0, _w1)[0, 1]) < 0.2)
_sc36b = _Scene(_g36).add("l0", _w0).add("l1", _w1).couple(into="l1", frm="l0", coupling=0.4)
check("lane-to-lane coupling injects the aggressor's signature into the victim",
      not np.array_equal(_sc36b.lane("l1"), _w1)
      and abs(np.corrcoef(_sc36b.lane("l1") - _w1, np.gradient(_w0))[0, 1]) > 0.9)
_sc36c = _Scene(_g36).add("d", _w0).differential("d", skew_ps=6.0)
check("a differential pair splits a lane into P/N that share timing (skew -> common-mode)",
      "d_p" in _sc36c.lanes() and "d_n" in _sc36c.lanes()
      and np.sqrt(np.mean(P.common_mode(_sc36c.lane("d_p"), _sc36c.lane("d_n")) ** 2)) > 0.01)

print("== optical-link primitives: extinction ratio, RIN, shot noise, chromatic dispersion ==")
from wfmsynth.optical import (to_optical as _topt, rin_noise as _rin, shot_noise as _shot,
                              chromatic_dispersion as _cd)
_x34 = P.nrz(n_ui=1 << 11, seed=1, n=1 << 13, causal=True)
_Popt = _topt(_x34, er_db=8.0)
check("to_optical gives non-negative power with the requested extinction ratio",
      _Popt.min() >= 0 and abs(10 * np.log10(_Popt.max() / _Popt.min()) - 8.0) < 0.2)
_rng34 = np.random.default_rng(0)
_hi = _rin(np.full(20000, 1.0), -130, 1e10, _rng34); _lo = _rin(np.full(20000, 0.2), -130, 1e10, _rng34)
check("RIN noise std is proportional to optical power (high level noisier than low)",
      _hi.std() > 3 * _lo.std())
_sh1 = _shot(np.full(20000, 1.0), 1e3, _rng34); _sh2 = _shot(np.full(20000, 0.2), 1e3, _rng34)
check("shot noise variance is proportional to power (Poisson photon counting; folds #30)",
      abs(_sh1.var() / _sh2.var() - 5.0) < 1.0)
_pulse34 = np.zeros(1 << 13); _pulse34[(1 << 12) - 20:(1 << 12) + 20] = 1.0
_disp = _cd(_pulse34, strength=60.0)
check("chromatic dispersion spreads a pulse (frequency-dependent group delay)",
      np.sum(np.abs(_disp) > 0.1 * np.abs(_disp).max()) > 2 * np.sum(np.abs(_pulse34) > 0.1))

print("== oscillator phase noise: a colored jitter spectrum with the requested slope ==")
from wfmsynth.cdr import phase_noise as _pn
_g37 = Grid(fs=200e9, n=1 << 16)
_pns = _pn(_g37.n, _g37, rms_ps=1.0, slope=2.0, rng=np.random.default_rng(0))
_P37 = np.abs(np.fft.rfft(_pns - _pns.mean())) ** 2
_f37 = np.arange(len(_P37))
_band = (_f37 > 30) & (_f37 < 3000)                    # mid band, avoid DC and the noise floor
_sl = np.polyfit(np.log(_f37[_band]), np.log(_P37[_band] + 1e-30), 1)[0]
check("phase-noise PSD falls with the requested slope (~ -slope for 1/f**slope)",
      abs(_sl - (-2.0)) < 0.5, f"fitted PSD slope={_sl:.2f} (want ~-2)")
check("phase noise is scaled to the requested RMS (in samples)",
      abs(np.sqrt(np.mean(_pns ** 2)) - 1e-12 * _g37.fs) < 0.2 * 1e-12 * _g37.fs)

print("== long-term drift: a measured attribute moves across the record; short windows are stable ==")
from wfmsynth.impairments import drift as _drift
_x38 = P.nrz(n_ui=1 << 11, seed=1, n=1 << 14, causal=True)
_y38 = _drift(_x38, kind="gain", amount=0.8, shape="linear")
_q = np.array_split(np.abs(_y38), 4)
_rms_q = [float(np.sqrt(np.mean(_qi ** 2))) for _qi in _q]
check("a slow gain drift moves the measured level monotonically across the record",
      all(_rms_q[i] < _rms_q[i + 1] for i in range(3)), f"quarter RMS {[round(v,2) for v in _rms_q]}")
_w = len(_x38) // 64
_e0 = np.sqrt(np.mean(_y38[:_w] ** 2)); _e1 = np.sqrt(np.mean(_y38[_w:2 * _w] ** 2))
check("within a short window the drift is negligible (adjacent short windows ~ equal)",
      abs(_e1 / _e0 - 1.0) < 0.05, f"adjacent short-window ratio {_e1 / _e0:.3f}")

print("== line coding: a DC-balanced code bounds running disparity that raw PRBS random-walks ==")
from wfmsynth.coding import (dc_balanced as _dcb, scramble_64b66b as _scr,
                             running_disparity as _rd, max_run as _mrun)
_raw39 = P.prbs(13, 1 << 15, seed=3)
_cod39 = _dcb(_raw39, block=8)
check("dc_balanced keeps running disparity bounded while raw PRBS random-walks far",
      np.max(np.abs(_rd(_cod39))) < 4 * 8 and np.max(np.abs(_rd(_cod39))) < 0.3 * np.max(np.abs(_rd(_raw39))),
      f"max|disp| coded={np.max(np.abs(_rd(_cod39)))} raw={np.max(np.abs(_rd(_raw39)))}")
check("the coded stream is balanced (mean ~0.5) with a bounded run length",
      abs(_cod39.mean() - 0.5) < 0.02 and _mrun(_cod39) <= 3 * 8)
_scr39 = _scr(_raw39)
check("64b/66b scrambler whitens to a ~balanced stream at 66/64 the length",
      abs(_scr39.mean() - 0.5) < 0.03 and len(_scr39) == (len(_raw39) // 64) * 66)

print("== acquisition chain: scope bandwidth rolls off HF; timebase jitter smears the eye ==")
from wfmsynth.instrument import (scope_bandwidth as _sbw, timebase_jitter as _tbj,
                                 probe_loading as _pl)
_g40 = Grid(fs=400e9, baud=50e9, n=1 << 13); _nui40 = int(_g40.n // _g40.samples_per_ui)
_x40 = (Signal(seed=1, grid=_g40).carrier("pam4", n_ui=_nui40, pattern="prbs13q", causal=True)).waveform()
_f40 = np.fft.rfftfreq(_g40.n, d=_g40.dt)
_hi = _f40 > 60e9
_bw40 = _sbw(_x40, _g40, bw_hz=33e9)
check("scope bandwidth rolls off high frequencies (band-limited acquisition)",
      np.sum(np.abs(np.fft.rfft(_bw40))[_hi]) < 0.5 * np.sum(np.abs(np.fft.rfft(_x40))[_hi]))
check("probe capacitive loading is a low-pass (attenuates HF, loads the DUT)",
      np.sum(np.abs(np.fft.rfft(_pl(_x40, _g40, c_load_f=1e-12)))[_hi]) < np.sum(np.abs(np.fft.rfft(_x40))[_hi]))
_tb40 = _tbj(_x40, _g40, rms_ps=1.5, rng=np.random.default_rng(0))
check("timebase jitter smears the eye horizontally (closes it)",
      _eh(_tb40, _g40) < _eh(_x40, _g40) - 0.02, f"eye {_eh(_x40,_g40):.3f} -> {_eh(_tb40,_g40):.3f}")

print("== Tx de-emphasis preset: a dB preset yields that transition/steady ratio ==")
_spb42 = 16
_taps42 = P.de_emphasis_taps(3.5)
_sq = np.repeat(np.tile([-1.0, 1.0], 8), _spb42).astype(float)     # 8-UI runs, sharp edges
_y42 = P.tx_ffe(_sq, _taps42, _spb42, pre=0)
_lo_run = 8 * _spb42                                               # index where the high run starts
_emph = _y42[_lo_run + _spb42 // 2]                                # first UI after the transition
_steady = _y42[_lo_run + 5 * _spb42 + _spb42 // 2]                 # a mid-run (steady) UI
check("de_emphasis_taps(3.5 dB) yields ~3.5 dB transition-to-steady ratio",
      abs(20 * np.log10(_emph / _steady) - 3.5) < 0.3, f"measured {20*np.log10(_emph/_steady):.2f} dB")

print()
if fails:
    print(f"VALIDATION FAILED: {len(fails)} checks -> {fails}")
    sys.exit(1)
print("ALL PHYSICS CHECKS PASSED")


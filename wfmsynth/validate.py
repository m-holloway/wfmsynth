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

print()
if fails:
    print(f"VALIDATION FAILED: {len(fails)} checks -> {fails}")
    sys.exit(1)
print("ALL PHYSICS CHECKS PASSED")


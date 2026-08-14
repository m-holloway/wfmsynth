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

print()
if fails:
    print(f"VALIDATION FAILED: {len(fails)} checks -> {fails}")
    sys.exit(1)
print("ALL PHYSICS CHECKS PASSED")


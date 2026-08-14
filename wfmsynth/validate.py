"""
validate_physics.py — HARD checks that each physics primitive does what it claims.
Run before trusting the generator. Prints PASS/FAIL per property; exits nonzero on
any failure. This is the "don't fool yourself" gate for the data engine.
"""
import sys
import numpy as np
from scipy import signal as sp
from wfmsynth import physics as P

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


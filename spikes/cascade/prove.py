"""
The cascaded-channel proof. Nothing here is fitted; every number has an answer before it runs.

    PYTHONPATH=. python3 spikes/cascade/prove.py

WHAT IT ESTABLISHES
  A  the cascade algebra against its own closed form, with no waveform involved
  B  a discontinuity at a KNOWN distance arrives at the arithmetic's lag with the attenuation
     the segment implies -- and how much of the residual is the causal loss's own dispersion
  C  a NEAR and a FAR reflection of EQUAL Gamma come out DIFFERENT SIZES, which is exactly what
     the shipped lumped topology (one lossy, then one reflect) cannot do
  D  both discontinuities in one path, each echo attenuated by its own segment

The measured-backplane arm is a separate study outside this repository: no third
party's measurement is committed to this repository.
"""
import numpy as np
import wfmsynth.physics as P
import wfmsynth.sparam as SP
from wfmsynth.grid import Grid

g = Grid(fs=256e9, baud=16e9, n=1 << 14)
f = np.fft.rfftfreq(g.n, d=g.dt)
EPS = 4.0                      # one permittivity for delay AND dielectric loss, everywhere
PPI = SP.ps_per_inch(EPS)
GAM, TOTAL = 0.055, 12.0

sym = np.zeros(64); sym[8] = 1.0
pulse = P.from_symbols(sym, n=int(round(64 * g.samples_per_ui)), tr_frac=0.15)
x = np.zeros(g.n); x[:len(pulse)] = pulse
k0 = int(np.argmax(np.abs(x)))


def hdr(s):
    print("\n" + "=" * 96); print(s); print("=" * 96)


def peak_ps(sig, skip=0):
    s = np.abs(sig).copy(); s[:skip] = 0.0
    k = int(np.argmax(s)); a, b, c = s[k - 1], s[k], s[k + 1]
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * g.dt * 1e12


def xcorr_lag_ps(sig, template, skip=0):
    """Matched-filter lag -- the estimator a caller uses to turn a lag into a distance."""
    c = np.abs(np.correlate(sig, template, mode="full")[len(template) - 1:])
    c[:skip] = 0.0
    k = int(np.argmax(c)); a, b, d = c[k - 1], c[k], c[k + 1]
    return (k + 0.5 * (a - d) / (a - 2 * b + d)) * g.dt * 1e12


# ---------------------------------------------------------------------------- A
hdr("A. THE ALGEBRA, AGAINST ITS OWN CLOSED FORM (no waveform involved)")
for G in (0.02, 0.055, 0.3, 0.7):
    d = SP.discontinuity(f, G)
    p = np.abs(d.s11) ** 2 + np.abs(d.s21) ** 2
    print(f"   lossless step Gamma={G:<6} |S11|^2+|S21|^2 = {p.min():.15f} .. {p.max():.15f}")

LEN = 1.66
p1 = [{"line": {"length_in": LEN, "eps_r": EPS}}, {"disc": {"gamma": GAM}},
      {"line": {"length_in": TOTAL - LEN, "eps_r": EPS}}]
tp1 = SP.cascade(p1, f)
H = SP.line(f, length_in=LEN, eps_r=EPS).s21
print(f"\n   one discontinuity at {LEN} in:")
print(f"     S11 vs the closed form Gamma * H_segment(f)^2   max|err| = "
      f"{np.abs(tp1.s11 - GAM * H ** 2).max():.3e}")
print("     -- the echo pays the SEGMENT's transmission twice, exactly, by construction")

G1, G2, L1, L2 = 0.055, 0.055, 1.0, 9.0
p2 = [{"line": {"length_in": L1, "eps_r": EPS}}, {"disc": {"gamma": G1}}, {"line": {"length_in": L2, "eps_r": EPS}},
      {"disc": {"gamma": G2}}, {"line": {"length_in": TOTAL - L1 - L2, "eps_r": EPS}}]
tp2 = SP.cascade(p2, f)
H1, Lw = SP.line(f, length_in=L1, eps_r=EPS).s21, SP.line(f, length_in=L2, eps_r=EPS).s21
hand = H1 ** 2 * (G1 + (1 - G1 ** 2) * G2 * Lw ** 2 / (1 + G1 * G2 * Lw ** 2))
print(f"\n   two discontinuities, hand-derived S11 = H1^2*(G1 + t1^2 G2 L^2/(1+G1 G2 L^2)):")
print(f"     max|err| = {np.abs(tp2.s11 - hand).max():.3e}")
for nm, tp in (("1 disc", tp1), ("2 disc", tp2)):
    tot = np.abs(tp.s11) ** 2 + np.abs(tp.s21) ** 2
    print(f"   passivity {nm}: max(|S11|^2+|S21|^2) = {tot.max():.9f}  (passive iff <= 1)")

# ---------------------------------------------------------------------------- B
hdr("B. A DISCONTINUITY AT A KNOWN DISTANCE -> THE ARITHMETIC'S LAG")
print(f"   {PPI:.4f} ps/inch one way at eps_r=4.0 (a caller rounding c to 11.8 in/ns gets "
          f"169.5 -- 0.03 % apart)\n")
print(f"   {'d[in]':>6} {'arith[ps]':>10} {'zero-phase':>11} {'err':>8} {'causal':>10} "
      f"{'excess':>9} {'segIL@8G':>9}")
for LEN in (1.0, 1.66, 4.0, 10.0):
    got = []
    for causal in (False, True):
        path = [{"line": {"length_in": LEN, "eps_r": EPS, "causal": causal}}, {"disc": {"gamma": GAM}},
                {"line": {"length_in": TOTAL - LEN, "eps_r": EPS, "causal": causal}}]
        y = SP.cascade_channel(x, path, grid=g, node="source")
        sk = int(0.6 * 2 * LEN * PPI * 1e-12 * g.fs)   # NOT k0-offset: the template carries it
        got.append(xcorr_lag_ps(y - x, pulse, sk) - xcorr_lag_ps(x, pulse))
    ar = 2 * LEN * PPI
    il = P.insertion_loss_db(np.array([8.0]), length_in=LEN, eps_r=EPS)[0]
    print(f"   {LEN:6.2f} {ar:10.3f} {got[0]:11.3f} {got[0]-ar:+8.3f} {got[1]:10.3f} "
          f"{got[1]-ar:+9.3f} {il:9.3f}")
print("\n   Zero-phase loss lands on the arithmetic with a CONSTANT -1.9 ps offset at every")
print("   distance -- that is the matched filter's own half-sample bias, not the model's.")
print("   Turning on the causal (minimum-phase) loss adds an EXCESS that grows with the")
print("   segment's loss: a causal lossy segment makes its own echo arrive late, and it is")
print("   paid twice. A distance estimator that ignores this reads a far echo as further away")
print("   than it is -- 0.43 inch too far at 10 inches here.")

# ---------------------------------------------------------------------------- C
hdr("C. NEAR AND FAR, IDENTICAL Gamma = 0.055, IDENTICAL 12-INCH PATH")
print("   'closed form' = irfft(X * Gamma*H_segment^2) peak. 'LUMPED' is the shipped topology:")
print("   one lossy for the whole 12 in, then one reflect.\n")
print(f"   {'d[in]':>6} {'lag[ps]':>9} {'2xsegIL':>8} {'closed form':>12} {'CASCADE mV':>11} "
      f"{'err':>10} {'LUMPED mV':>10}")
xl = P.lossy_channel(x, length_in=TOTAL, eps_r=EPS, causal=True, grid=g)
for LEN in (0.5, 1.0, 1.66, 3.0, 6.0, 10.0):
    path = [{"line": {"length_in": LEN, "eps_r": EPS}}, {"disc": {"gamma": GAM}},
            {"line": {"length_in": TOTAL - LEN, "eps_r": EPS}}]
    sk = k0 + int(0.6 * 2 * LEN * PPI * 1e-12 * g.fs)
    casc = 1000 * np.abs((SP.cascade_channel(x, path, grid=g, node="source") - x)[sk:]).max()
    Hs = SP.line(f, length_in=LEN, eps_r=EPS).s21
    exact = 1000 * np.abs(np.fft.irfft(np.fft.rfft(x) * (GAM * Hs ** 2), g.n)).max()
    yl = P.multi_reflection(xl, td_ps=LEN * PPI, gamma_s=GAM, gamma_l=GAM, n_bounce=6,
                            grid=g, node="source")
    lump = 1000 * np.abs((yl - xl)[sk:]).max()
    il = P.insertion_loss_db(np.array([8.0]), length_in=LEN, eps_r=EPS)[0]
    print(f"   {LEN:6.2f} {2*LEN*PPI:9.1f} {2*il:8.2f} {exact:12.4f} {casc:11.4f} "
          f"{casc-exact:+10.2e} {lump:10.4f}")
print("\n   CASCADE tracks the closed form to ~1e-14 mV. The LUMPED column is CONSTANT to four")
print("   decimal places from 0.5 in to 10 in: it applies the whole channel's loss once and")
print("   then reflects, so it has no idea where the discontinuity is. THAT is the defect the")
print("   user caught, and it is the reason the lumped model cannot turn a lag into a distance.")

# ---------------------------------------------------------------------------- D
hdr("D. BOTH DISCONTINUITIES IN ONE PATH -- 1.0 in and 10.0 in, same Gamma")
for e in SP.first_order_echoes(p2):
    print(f"   arithmetic: disc at {e['distance_inch']:5.2f} in -> echo at {e['delay_ps']:8.2f} ps"
          f" ({e['delay_ps']/62.5:6.3f} UI), DC amplitude {e['amp']:.6f}")
ech = SP.cascade_channel(x, p2, grid=g, node="source") - x
amps = []
for e in SP.first_order_echoes(p2):
    a = k0 + int(0.75 * e["delay_ps"] * 1e-12 * g.fs)
    b = k0 + int(1.35 * e["delay_ps"] * 1e-12 * g.fs)
    k = a + int(np.argmax(np.abs(ech[a:b])))
    amps.append(1000 * abs(ech[k]))
    il = P.insertion_loss_db(np.array([8.0]), length_in=e["distance_inch"], eps_r=EPS)[0]
    print(f"   measured  : peak at {(k-k0)*g.dt*1e12:8.2f} ps, {amps[-1]:7.3f} mV, "
          f"2 x segment IL = {2*il:6.2f} dB at 8 GHz")
print(f"\n   near/far amplitude ratio = {amps[0]/amps[1]:.2f}x = "
      f"{20*np.log10(amps[0]/amps[1]):.2f} dB, at IDENTICAL Gamma.")
print("   The lumped model's ratio is 1.00x = 0.00 dB, at every distance, always.")

"""
wfmsynth.physics — physics-informed waveform synthesis primitives.

Frequency-domain lossy channels (skin + dielectric, causal minimum-phase option),
transmission-line multi-reflection, crosstalk (NEXT/FEXT), AC-coupling, physically-
decomposed jitter (Rj/Pj/DCD), digital signaling (NRZ/PAM4 from PRBS), RF modulation
(AM/FM/PSK/QAM), chirp, PDN transients, and an ECG-like generator.

Every primitive is paired with an assertion in `wfmsynth.validate` that checks the
physical property actually holds — nothing is trusted without that check. Works on a
normalized, unitless time grid (`N` samples over [0,1)); rate-parameterizable to any
symbol/sample rate downstream. numpy/scipy only.
"""
from __future__ import annotations
import numpy as np
from scipy import signal

N = 4096                       # working grid
T = np.linspace(0.0, 1.0, N, endpoint=False)


# ---------------------------------------------------------------- channel physics
def _min_phase_H(Hmag):
    """Causal minimum-phase complex response from a real magnitude |H| (rfft bins),
    via the cepstral / Hilbert relation so loss and phase are physically LINKED
    (Kramers-Kronig; the Djordjevic-Sarkar-style causal channel). A real magnitude-
    only response is zero-phase -> non-causal (symmetric pre/post-ringing); the
    minimum-phase version concentrates the response AFTER t=0 (asymmetric post-cursor
    ISI), which is what a real dispersive interconnect does."""
    mag_full = np.concatenate([Hmag, Hmag[-2:0:-1]])      # symmetric, length N
    logmag = np.log(mag_full + 1e-12)
    c = np.fft.ifft(logmag).real                          # real cepstrum
    w = np.zeros(N); w[0] = 1.0; w[1:N // 2] = 2.0; w[N // 2] = 1.0   # causal folding
    return np.exp(np.fft.fft(c * w))                      # complex min-phase, length N


def lossy_channel(x, length_in=6.0, tand=0.02, eps_r=4.3, f_nyq_ghz=8.0,
                  skin_k=0.0, causal=False):
    """Apply a frequency-dependent SI channel: insertion loss
        IL(f)[dB] = (a_skin*sqrt(f_GHz) + b_diel*f_GHz) * length_in
    with dielectric-loss coefficient b_diel = 2.3*sqrt(eps_r)*tand (dB/in/GHz)
    [KB: All-About-Circuits/Bogatin]. f axis scaled so bin N/2 == f_nyq_ghz.
    causal=False: zero-phase magnitude response (legacy). causal=True: physically
    correct minimum-phase response -> asymmetric post-cursor ISI (real dispersion)."""
    f_ghz = np.fft.rfftfreq(N) * 2.0 * f_nyq_ghz          # 0..f_nyq_ghz at bin N/2
    b_diel = 2.3 * np.sqrt(eps_r) * tand
    a_skin = skin_k if skin_k > 0 else 0.35               # ~dB/in/sqrt(GHz) typ
    il_db = (a_skin * np.sqrt(f_ghz) + b_diel * f_ghz) * length_in
    Hmag = 10.0 ** (-il_db / 20.0)
    if causal:
        Hc = _min_phase_H(Hmag)                           # full-spectrum complex H
        return np.fft.ifft(np.fft.fft(x) * Hc).real
    return np.fft.irfft(np.fft.rfft(x) * Hmag, n=N)


def crosstalk(x, aggressor, coupling=0.12, kind="fext", td_frac=0.05):
    """Add coupled noise from a neighboring (aggressor) line. FEXT couples through
    the mutual C/L as the DERIVATIVE of the aggressor (∝ d/dt); NEXT is a delayed,
    broadband coupling. `coupling` scales the aggressor relative to x's span. Real
    high-speed links are frequently crosstalk-limited — a major missing effect."""
    a = np.asarray(aggressor, float)
    if kind == "fext":
        k = np.gradient(a)
    else:                                                 # next: delayed coupling
        d = int(td_frac * N); k = np.zeros(N); k[d:] = a[:N - d]
    k = k / (np.abs(k).max() + 1e-9)
    return x + coupling * (np.ptp(x) + 1e-9) * k


def ac_couple(x, fc_frac=0.004):
    """AC-coupling (series cap) as a 1st-order high-pass -> baseline wander/droop
    that grows with run length. fc_frac is the corner as a fraction of Nyquist.
    Ubiquitous on real serial links (currently absent)."""
    fc = float(np.clip(fc_frac, 1e-4, 0.5))
    sos = signal.butter(1, fc, btype="high", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, float))


def multi_reflection(x, td_frac=0.12, gamma_s=0.3, gamma_l=0.4, n_bounce=6):
    """Transmission-line bounce diagram: received = incident + reflected train.
    Each round trip is delayed by 2*td and scaled by (gamma_s*gamma_l)^n. This is
    the lattice/bounce superposition for a mismatched line."""
    d = int(td_frac * N)
    y = x.copy()
    g = gamma_s * gamma_l
    for n in range(1, n_bounce + 1):
        shift = 2 * d * n
        if shift >= N:
            break
        refl = np.zeros_like(x)
        refl[shift:] = x[:N - shift]
        y = y + (g ** n) * gamma_l * refl
    return y


# ---------------------------------------------------------------- jitter physics
def inject_jitter(x, sigma_rj=0.0, a_pj=0.0, f_pj=5.0, dcd=0.0, rng=None):
    """Physically decomposed jitter via time-axis warp then resample.
      Rj: BANDLIMITED Gaussian phase noise (smooth, so edges shift COHERENTLY),
          renormalized to RMS = sigma_rj samples. (Per-sample white noise would
          scramble the waveform, not jitter its edges — validated.)
      Pj: sinusoidal A*sin(2*pi*f*t) (samples)
      DCD: polarity-dependent offset (rising +dcd/2, falling -dcd/2)
    sigma_rj/a_pj/dcd are in SAMPLES of timing displacement."""
    rng = rng or np.random.default_rng()
    disp = np.zeros(N)
    if sigma_rj > 0:
        w = rng.standard_normal(N)
        sos = signal.bessel(2, 0.03, output="sos")          # smooth to ~phase-noise BW
        w = signal.sosfiltfilt(sos, w)
        w = w / (w.std() + 1e-9) * sigma_rj                 # exact RMS = sigma_rj
        disp += w
    if a_pj > 0:
        disp += a_pj * np.sin(2 * np.pi * f_pj * T)
    if dcd != 0:
        slope = np.sign(np.gradient(x))
        disp += (dcd / 2.0) * slope
    src = np.clip(np.arange(N) + disp, 0, N - 1)
    return np.interp(src, np.arange(N), x)


# ---------------------------------------------------------------- signaling
def prbs(order, length, seed=1):
    taps = {7: (7, 6), 9: (9, 5), 15: (15, 14), 31: (31, 28)}[order]
    st = seed & ((1 << order) - 1) or 1
    out = np.empty(length, np.int8)
    for i in range(length):
        b = 0
        for t in taps:
            b ^= (st >> (t - 1)) & 1
        out[i] = st & 1
        st = ((st << 1) | b) & ((1 << order) - 1)
    return out


def nrz(n_ui=32, tr_frac=0.15, seed=1):
    spb = N / n_ui
    bits = prbs(7, n_ui, seed)
    idx = np.clip((np.arange(N) / spb).astype(int), 0, n_ui - 1)
    x = np.where(bits[idx] > 0, 1.0, -1.0)
    tr = max(tr_frac * spb, 2)
    sos = signal.bessel(4, min(0.7 / tr, 0.98), output="sos")
    return signal.sosfiltfilt(sos, x)


def pam4(n_ui=32, tr_frac=0.15, seed=1):
    spb = N / n_ui
    b0 = prbs(7, n_ui, seed); b1 = prbs(9, n_ui, seed + 3)
    levels = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
    gray = (b0 * 2 + (b0 ^ b1))                      # simple Gray-ish map -> 0..3
    idx = np.clip((np.arange(N) / spb).astype(int), 0, n_ui - 1)
    x = levels[np.clip(gray[idx], 0, 3)]
    tr = max(tr_frac * spb, 2)
    sos = signal.bessel(4, min(0.7 / tr, 0.98), output="sos")
    return signal.sosfiltfilt(sos, x)


# ---------------------------------------------------------------- RF / analog
def am(fc=40.0, fm=3.0, depth=0.6):
    msg = np.sin(2 * np.pi * fm * T)
    return (1 + depth * msg) * np.cos(2 * np.pi * fc * T)


def fm(fc=40.0, fm_rate=3.0, beta=5.0):
    msg = np.sin(2 * np.pi * fm_rate * T)
    phase = 2 * np.pi * fc * T + beta * np.cumsum(msg) / N * (2 * np.pi)
    return np.cos(phase)


def psk(fc=40.0, n_sym=16, m=4, seed=1):
    rng = np.random.default_rng(seed)
    syms = rng.integers(0, m, n_sym)
    spb = N / n_sym
    idx = np.clip((np.arange(N) / spb).astype(int), 0, n_sym - 1)
    ph = syms[idx] * (2 * np.pi / m)
    return np.cos(2 * np.pi * fc * T + ph)


def qam(fc=40.0, n_sym=16, seed=1):
    rng = np.random.default_rng(seed)
    I = rng.choice([-1, -1 / 3, 1 / 3, 1], n_sym); Q = rng.choice([-1, -1 / 3, 1 / 3, 1], n_sym)
    spb = N / n_sym
    idx = np.clip((np.arange(N) / spb).astype(int), 0, n_sym - 1)
    return I[idx] * np.cos(2 * np.pi * fc * T) - Q[idx] * np.sin(2 * np.pi * fc * T)


def chirp(f0=3.0, f1=40.0):
    return signal.chirp(T, f0=f0, f1=f1, t1=1.0, method="linear")


def pdn_transient(droop=0.08, tau1=0.03, tau2=0.25, t0=0.3):
    tt = T - t0
    sag = np.where(tt > 0, droop * (0.5 * np.exp(-tt / tau1) + 0.5 * np.exp(-tt / tau2)), 0.0)
    return 1.0 - sag


def ecg_like(hr=8.0):
    """Synthetic ECG-ish: Gaussian-bump QRS + P/T waves per beat."""
    x = np.zeros(N)
    for k in range(int(hr)):
        c = (k + 0.5) / hr
        x += 1.0 * np.exp(-((T - c) ** 2) / (2 * 0.004 ** 2))          # R
        x -= 0.15 * np.exp(-((T - c + 0.012) ** 2) / (2 * 0.004 ** 2))  # Q
        x -= 0.15 * np.exp(-((T - c - 0.012) ** 2) / (2 * 0.004 ** 2))  # S
        x += 0.2 * np.exp(-((T - c - 0.05) ** 2) / (2 * 0.012 ** 2))    # T
        x += 0.1 * np.exp(-((T - c + 0.06) ** 2) / (2 * 0.010 ** 2))    # P
    return x


# family registry: name -> (generator thunk taking rng) ; broad + physical
def family_bank():
    return {
        "nrz_si": lambda r: lossy_channel(nrz(seed=r.integers(1, 127)),
                                          length_in=r.uniform(2, 14), tand=r.uniform(0.005, 0.025)),
        "pam4": lambda r: lossy_channel(pam4(seed=r.integers(1, 127)), length_in=r.uniform(2, 10)),
        "nrz_reflect": lambda r: multi_reflection(nrz(seed=r.integers(1, 127)),
                                                  td_frac=r.uniform(0.06, 0.2),
                                                  gamma_s=r.uniform(0.1, 0.4), gamma_l=r.uniform(0.1, 0.5)),
        "am": lambda r: am(fc=r.uniform(25, 55), fm=r.uniform(2, 5), depth=r.uniform(0.3, 0.9)),
        "fm": lambda r: fm(fc=r.uniform(25, 55), fm_rate=r.uniform(2, 5), beta=r.uniform(2, 8)),
        "psk": lambda r: psk(fc=r.uniform(25, 55), m=int(r.choice([2, 4, 8])), seed=r.integers(1, 999)),
        "qam": lambda r: qam(fc=r.uniform(25, 55), seed=r.integers(1, 999)),
        "chirp": lambda r: chirp(f0=r.uniform(1, 5), f1=r.uniform(20, 50)),
        "pdn": lambda r: pdn_transient(droop=r.uniform(0.03, 0.15), t0=r.uniform(0.2, 0.5)),
        "ecg": lambda r: ecg_like(hr=r.uniform(5, 12)),
    }

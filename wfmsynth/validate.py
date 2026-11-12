"""
validate_physics.py — HARD checks that each physics primitive does what it claims.
Run before trusting the generator. Prints PASS/FAIL per property; exits nonzero on
any failure. This is the "don't fool yourself" gate for the data engine.
"""
import sys
import warnings

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
    """mean frequency (normalized 0..0.5) — drops when HF is attenuated.

    WINDOWED, and it has to be. An unwindowed rfft of a record whose first and last samples
    differ sees a step in the periodic extension and spreads it across the whole band, which is
    not the record's spectrum. Our records used to be exactly periodic — a frequency-domain
    stage applied its response CIRCULARLY, so the wrap made them so — and a rectangular window
    was harmless. With those stages applying a linear convolution (U-16) a record begins on a
    quiescent line and ends without its own tail, and unwindowed this statistic then reports a
    HARSHER channel as having MORE high frequency (0.0084 against 0.0040 for the raw record),
    which is the edge step, not the channel. Under a Hann window the measurement is identical
    on the circular and the linear path to every digit printed here."""
    z = (x - x.mean()) * np.hanning(len(x))
    mag = np.abs(np.fft.rfft(z))
    f = np.fft.rfftfreq(len(x))
    return float((f * mag).sum() / (mag.sum() + 1e-12))


print("== lossy channel: must attenuate HF (lower spectral centroid) and slow edges ==")
x = P.nrz(seed=3)
mild = P.lossy_channel(x, length_in=2.0, tand=0.005)
harsh = P.lossy_channel(x, length_in=14.0, tand=0.025)
c_raw, c_mild, c_harsh = spectral_centroid(x), spectral_centroid(mild), spectral_centroid(harsh)
check("harsher channel lowers spectral centroid (more HF loss)",
      c_harsh < c_mild < c_raw,
      f"raw={c_raw:.4f} mild={c_mild:.4f} harsh={c_harsh:.4f}")
# Read on the record's INTERIOR. A zero-phase channel is not causal, so the response to the
# record's last samples reaches for samples that are not there: under a linear convolution the
# tail rolls off, and its final sample is the steepest thing in the record. That is an edge of
# the record, not an edge rate.
_edge = slice(N // 16, -N // 16)
_sl = lambda y: float(np.abs(np.gradient(y[_edge])).max())
check("harsh channel lowers max edge slope",
      _sl(harsh) < _sl(mild), f"mildslope={_sl(mild):.3f} harshslope={_sl(harsh):.3f}")

print("== causal channel: minimum-phase concentrates response AFTER t=0 (post-cursor ISI) ==")
# The impulse goes in the MIDDLE of the record, so "before t=0" is a real place in the array
# rather than the far end of a wrap. That is the only way to read pre-cursor ringing off a
# LINEAR convolution: with the impulse at sample 0 a zero-phase channel's pre-ringing falls
# before the record starts and is truncated away, so it would measure as zero and the check
# would pass for the wrong reason.
imp = np.zeros(N); imp[N // 2] = 1.0
h_zero = P.lossy_channel(imp, length_in=12.0, tand=0.02, causal=False)
h_caus = P.lossy_channel(imp, length_in=12.0, tand=0.02, causal=True)
pre_z = np.abs(h_zero[:N // 2]).sum(); post_z = np.abs(h_zero[N // 2:]).sum()
pre_c = np.abs(h_caus[:N // 2]).sum(); post_c = np.abs(h_caus[N // 2:]).sum()
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
# `sosfiltfilt` here is deliberate and stays: this is a SOURCE FIXTURE -- a smooth pulse / a
# smooth square for the stage under test to act on -- not an instrument stage. It states no
# corner and makes no causality claim, so zero phase costs nothing. The stages that DID claim
# a corner (`instrument.scope_bandwidth`, `probe_loading`) are single-pass causal; BACKLOG #57
# lists the remaining calls and which category each is in.
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
# a source fixture, not an instrument stage -- see the note above and BACKLOG #57
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

print("== PRBS tap table: every offered polynomial is PRIMITIVE (maximal length) ==")
# A tap set that is not primitive yields a sequence that looks random, has short
# sub-periods, and is NOT the sequence the standard names -- and orders 23 and 31 have
# periods (8.4e6, 2.1e9) far too long to demonstrate by generating them. So check the
# algebra instead: G is primitive iff the order of x in GF(2)[x]/G is exactly 2^order-1,
# i.e. x^(2^n-1) == 1 and x^((2^n-1)/p) != 1 for every prime p dividing 2^n-1. Exact,
# and it runs in microseconds for every order in the table.
def _prime_factors(m):
    f, d = set(), 2
    while d * d <= m:
        while m % d == 0:
            f.add(d); m //= d
        d += 1
    if m > 1:
        f.add(m)
    return f

def _polymulmod(a, b, g, deg):                 # (a*b) mod g over GF(2)
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if (a >> deg) & 1:
            a ^= g
    return r

def _x_pow_mod(e, g, deg):                     # x^e mod g over GF(2)
    r, base = 1, 2
    while e:
        if e & 1:
            r = _polymulmod(r, base, g, deg)
        base = _polymulmod(base, base, g, deg)
        e >>= 1
    return r

def _is_primitive(order, taps):
    g = 1
    for t in taps:                             # G(x) = 1 + sum x^t, matching prbs()'s feedback
        g ^= 1 << t
    m = (1 << order) - 1
    if _x_pow_mod(m, g, order) != 1:
        return False
    return all(_x_pow_mod(m // p, g, order) != 1 for p in _prime_factors(m))

for _o, _t in sorted(P.PRBS_TAPS.items()):
    _poly = " + ".join(["1"] + [f"x^{t}" for t in sorted(_t)])
    check(f"PRBS{_o} polynomial is primitive (period 2^{_o}-1 = {(1 << _o) - 1})",
          _is_primitive(_o, _t), _poly)
check("PRBS_TAPS offers only orders whose standard polynomial is stated",
      set(P.PRBS_TAPS) == {7, 9, 11, 13, 15, 23, 31}, f"orders={sorted(P.PRBS_TAPS)}")

print("== maximal-length properties hold over a FULL period (balance and longest run) ==")
# Two properties every m-sequence has and no near-miss polynomial has: exactly 2^(n-1)
# ones per period, and a longest run of exactly n. Orders 23/31 are covered by the
# algebraic check above -- materializing their periods is not worth the seconds.
for _o in (7, 9, 11, 13, 15):
    _p = (1 << _o) - 1
    _b = P.prbs(_o, _p * 2)
    _mr = _c = 1
    for _i in range(1, _p):
        _c = _c + 1 if _b[_i] == _b[_i - 1] else 1
        _mr = max(_mr, _c)
    check(f"PRBS{_o}: period {_p}, {1 << (_o - 1)} ones, longest run {_o}",
          np.array_equal(_b[:_p], _b[_p:2 * _p]) and int(_b[:_p].sum()) == (1 << (_o - 1))
          and _mr == _o,
          f"ones={int(_b[:_p].sum())} longest_run={_mr}")

print("== NRZ pattern: the carrier HONOURS `pattern`, and the default is PRBS7 bit-for-bit ==")
# The default is a COMPATIBILITY default: this kernel is pinned by SHA and its output is
# diffed sample-for-sample downstream, so an NRZ carrier with no pattern given must stay
# the PRBS7 stream it has always been. That is asserted here, not assumed.
_nd = 4096
check("carrier_symbols('nrz') with no pattern is PRBS7, sample for sample",
      np.array_equal(P.carrier_symbols("nrz", _nd, 3),
                     np.where(P.prbs(7, _nd, 3) > 0, 1.0, -1.0)))
check("'legacy' and 'prbs7' are the same NRZ stream (the alias is exact)",
      np.array_equal(P.carrier_symbols("nrz", _nd, 3, "legacy"),
                     P.carrier_symbols("nrz", _nd, 3, "prbs7")))
check("nrz() default waveform == nrz(pattern='prbs7') waveform, sample for sample",
      np.array_equal(P.nrz(n_ui=512, seed=3, n=8192),
                     P.nrz(n_ui=512, seed=3, n=8192, pattern="prbs7")))
check("every exposed NRZ pattern produces a two-level stream of the requested length",
      all(len(_s) == 777 and set(np.unique(_s)) == {-1.0, 1.0}
          for _s in (P.carrier_symbols("nrz", 777, 5, _p) for _p in P.NRZ_PATTERNS)),
      f"patterns={list(P.NRZ_PATTERNS)}")
check("pattern actually changes the stream (PRBS7 != PRBS31 != PRBS13)",
      not np.array_equal(P.carrier_symbols("nrz", _nd, 1, "prbs7"),
                         P.carrier_symbols("nrz", _nd, 1, "prbs31"))
      and not np.array_equal(P.carrier_symbols("nrz", _nd, 1, "prbs13"),
                             P.carrier_symbols("nrz", _nd, 1, "prbs31")))

print("== pattern REPEAT length: PRBS7 repeats inside a record; PRBS31 cannot ==")
# The substance of the defect. 127 bits inside a 3 M UI record is 23622 repetitions --
# the channel sees the same 127-bit history over and over, so the record contains far
# less distinct pattern history than its length suggests.
_s7 = P.carrier_symbols("nrz", 4064, 1, "prbs7")
check("PRBS7 repeats every 127 UI (32 whole repetitions in a 4064 UI record)",
      all(np.array_equal(_s7[:127], _s7[k * 127:(k + 1) * 127]) for k in range(1, 32)),
      "127-bit period; 23622 repetitions inside a 3 M UI record")
_s31 = P.carrier_symbols("nrz", 4064, 1, "prbs31")
check("PRBS31 does not repeat at ANY lag inside the same record (period 2147483647)",
      not any(np.array_equal(_s31[:_L], _s31[_L:2 * _L]) for _L in range(1, 2033)))

print("== pattern HISTORY closes the eye: PRBS7 renders a lossy channel too optimistic ==")
# Channel ISI is a function of pattern history: long runs and low-frequency content are
# what actually shut an eye. PRBS7's longest run is 7 UI, PRBS31's is ~27 in this record.
# Same channel, same seed, same everything else -- only the pattern differs.
# NOTE the record must be long enough to SAMPLE PRBS31's run distribution: PRBS31 is a
# capture-length segment of a 2.1e9-symbol sequence, and in a short (~3 k UI) record
# whether a long run falls inside the window is luck of the seed. 8192 UI is not.
from wfmsynth.grid import Grid as _GridP
_gpat = _GridP(fs=80e9, baud=10e9, n=8192 * 8)
from wfmsynth.measure import eye_height as _eh
def _lossy_eye(pattern, seed):
    _x = P.nrz(n_ui=8192, seed=seed, n=_gpat.n, causal=True, tr_frac=0.4, pattern=pattern)
    return float(_eh(P.lossy_channel(_x, length_in=14.0, tand=0.02, causal=True),
                     _gpat, levels=2))
for _sd in (1, 3, 11):
    _e7, _e31, _eck = (_lossy_eye("prbs7", _sd), _lossy_eye("prbs31", _sd),
                       _lossy_eye("clock", _sd))
    # 1.05, not 1.10. The margin shrank when the edge shaper started delivering the rise time
    # it is asked for: a sharper edge carries less ISI, so the eye depends less on which pattern
    # drove it. The direction of the claim is what matters and it is unchanged -- a short pattern
    # overstates the eye.
    check(f"seed {_sd}: PRBS7 eye is OPTIMISTIC vs PRBS31 through the same 14in channel",
          _e7 > _e31 * 1.05,
          f"prbs7={_e7:.4f} prbs31={_e31:.4f} ({100 * (_e7 / _e31 - 1):.0f}% overstated)")
    check(f"seed {_sd}: the clock pattern is the ISI-free contrast (most open of the three)",
          _eck > _e7 > 0, f"clock={_eck:.4f} prbs7={_e7:.4f} prbs31={_e31:.4f}")

print("== clock pattern: alternating 1010..., one transition per UI, no runs ==")
_ck = P.carrier_symbols("nrz", 1001, 5, "clock")
check("clock alternates every UI (transition density 1.0, longest run 1)",
      np.all(_ck[1:] != _ck[:-1]) and set(np.unique(_ck)) == {-1.0, 1.0},
      f"density={float(np.mean(_ck[1:] != _ck[:-1])):.1f}")
check("clock is seed-independent (there is nothing random in it)",
      np.array_equal(P.carrier_symbols("nrz", 1001, 5, "clock"),
                     P.carrier_symbols("nrz", 1001, 99, "clock")))
check("clock is DC-balanced over an even number of UI",
      abs(float(P.carrier_symbols("nrz", 1000, 1, "clock").sum())) < 1e-12)

print("== pattern errors stay HONEST: no silent coercion across carrier kinds ==")
def _err(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    return ""
_e_q_on_nrz = _err(lambda: P.carrier_symbols("nrz", 64, 1, "prbs13q"))
check("a QUATERNARY pattern on an NRZ carrier raises and says so",
      "quaternary" in _e_q_on_nrz and "pam4" in _e_q_on_nrz, _e_q_on_nrz)
_e_b_on_pam4 = _err(lambda: P.carrier_symbols("pam4", 64, 1, "prbs31"))
check("a BINARY pattern on a PAM4 carrier raises and says so",
      "binary" in _e_b_on_pam4 and "nrz" in _e_b_on_pam4, _e_b_on_pam4)
_e_ck_on_pam4 = _err(lambda: P.carrier_symbols("pam4", 64, 1, "clock"))
check("the clock pattern is rejected on a PAM4 carrier (it is a binary sequence)",
      "binary" in _e_ck_on_pam4, _e_ck_on_pam4)
_e_unk = _err(lambda: P.carrier_symbols("nrz", 64, 1, "prbs42"))
check("an unknown NRZ pattern raises a message NAMING what is accepted",
      all(_p in _e_unk for _p in ("prbs7", "prbs31", "clock", "legacy")), _e_unk)
_e_unk4 = _err(lambda: P.carrier_symbols("pam4", 64, 1, "prbs42"))
check("an unknown PAM4 pattern raises a message NAMING what is accepted",
      all(_p in _e_unk4 for _p in ("legacy", "prbs13q", "prbs31q")), _e_unk4)
check("nrz() rejects a bad pattern too (not just carrier_symbols)",
      _err(lambda: P.nrz(n_ui=64, pattern="prbs13q")) != "")


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

print("== reverse-wave channel: a reflection is visible at the SOURCE plane, one bounce before the load ==")
_imp = np.zeros(4096); _imp[100] = 1.0
_d = 300
_load_ms = P.multi_reflection(_imp, gamma_s=0.0, gamma_l=0.4, td_samples=_d, node="load")
_src_ms = P.multi_reflection(_imp, gamma_s=0.0, gamma_l=0.4, td_samples=_d, node="source")
check("matched source: the load tap sees no reflection train (the single bounce is absorbed at source)",
      np.allclose(_load_ms, _imp))
check("reverse wave: the source tap shows the returning echo at 2*td (return loss visible upstream)",
      abs(_src_ms[100 + 2 * _d] - 0.4) < 1e-9 and abs(_src_ms[100] - 1.0) < 1e-9)
check("matched load: source and load taps are both bit-identical to the incident (forward-only regression)",
      np.array_equal(P.multi_reflection(_imp, gamma_s=0.3, gamma_l=0.0, td_samples=_d, node="source"), _imp)
      and np.array_equal(P.multi_reflection(_imp, gamma_s=0.3, gamma_l=0.0, td_samples=_d, node="load"), _imp))
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
# a source fixture, not an instrument stage -- see BACKLOG #57
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

print("== provenance identity: canonical JSON + sha256 stable; annotations round-trip, samples unchanged ==")
from wfmsynth.compose import rederive_anchor as _rda
check("sha256 is stable across a canonical-JSON round-trip",
      _sig.sha256() == Signal.from_recipe(_json.loads(_sig.to_json())).sha256(), _sig.sha256()[:12])
# annotating an op with provenance changes the recipe but leaves the samples bit-identical
_siga = Signal.from_recipe(_rec).annotate(stage="channel", node="TP2")
check("provenance annotation round-trips (recipe carries it) and samples stay bit-identical",
      np.array_equal(Signal.from_recipe(_json.loads(_siga.to_json())).waveform(), _xr)
      and _siga.recipe()["ops"][-1]["_prov"]["node"] == "TP2")
# rederive_anchor turns a symbolic 'when' into a sample index on the grid
check("rederive_anchor maps ui / t / sample to a consistent sample index",
      _rda({"ui": 10}, _gp) == int(round(10 * _gp.samples_per_ui))
      and _rda({"t": 1e-9}, _gp) == int(round(_gp.to_samples(1e-9)))
      and _rda({"sample": 123}, _gp) == 123)

print("== fabric homing (opt-in): stage-kind canonical order; commute-by-construction; default unchanged ==")
from wfmsynth.compose import KIND_RANK as _KR, op_kind as _opk
# the provenance recipe is already canonical -> canonical() is a no-op -> bit-identical waveform
check("canonical() is bit-identical on an already-canonical recipe (default path unchanged)",
      np.array_equal(_sig.canonical().waveform(), _sig.waveform()))
# a shape op (glitch) and a channel op (reflection) authored in EITHER order home to the same fabric
_gk = Grid(fs=64e9, baud=16e9, n=1 << 12)
_fa = (Signal(seed=3, grid=_gk).carrier("nrz", n_ui=128)
       .reflect(td_ps=40, gamma_s=0.3, gamma_l=0.3).events("glitch", on="symbols", count=1))
_fb = (Signal(seed=3, grid=_gk).carrier("nrz", n_ui=128)
       .events("glitch", on="symbols", count=1).reflect(td_ps=40, gamma_s=0.3, gamma_l=0.3))
check("homing makes cross-kind authoring order commute (identical op order and waveform)",
      [o["op"] for o in _fa.canonical().ops] == [o["op"] for o in _fb.canonical().ops]
      and np.array_equal(_fa.canonical().waveform(), _fb.canonical().waveform()))
_ks = _fa.canonical().stage_kinds()
check("canonical stage-kind order is nondecreasing (source->shape->supply->channel->instrument)",
      all(_KR[_ks[i]] <= _KR[_ks[i + 1]] for i in range(len(_ks) - 1)))

print("== piecewise timebase: a segmented grid resolves anchors WITHIN their segment (CAN-FD BRS) ==")
_gseg = Grid(fs=32e9, segments=(("arb", 500e6, 20), ("data", 2e9, 40)))
# arbitration bits are 32e9/500e6 = 64 samples wide; data bits are 32e9/2e9 = 16 samples wide
check("segment anchors resolve within-segment (dual-rate bit widths)",
      _gseg.resolve({"segment": "arb", "bit": 3}) == 3 * 64
      and _gseg.resolve({"segment": "data", "bit": 2}) == 20 * 64 + 2 * 16)
_bounds = _gseg.segment_bounds()
check("segment bounds are contiguous and dual-rate (data bits narrower than arbitration bits)",
      _bounds[0] == ("arb", 0, 20 * 64) and _bounds[1] == ("data", 20 * 64, 20 * 64 + 40 * 16))
# a single-segment grid is unchanged: resolve matches the uniform ui math
check("single-segment grid: resolve('ui') matches uniform samples_per_ui (regression)",
      Grid(fs=256e9, baud=112e9, n=4096).resolve({"ui": 7})
      == int(round(7 * (256e9 / 112e9))))

print("== multi-driver combine laws: N drivers on one shared net resolved by a declared law ==")
from wfmsynth.bus import combine_drivers as _cd, open_drain as _od
_A, _B = np.array([1, 1, 0, 1, 1.]), np.array([1, 0, 0, 1, 1.])
check("wired_and: the line is low wherever ANY driver pulls (I2C multi-master; matches open_drain)",
      np.array_equal(_cd([_A, _B], "wired_and"), _od([_A, _B]))
      and np.array_equal(_cd([_A, _B], "wired_and"), np.array([1, 0, 0, 1, 1.])))
_L1, _L2 = np.array([1.0, 0.2, -0.5]), np.array([0.3, 0.4, 0.4])
check("dominant_min: the lowest (dominant) level wins pointwise (CAN arbitration)",
      np.array_equal(_cd([_L1, _L2], "dominant_min"), np.array([0.3, 0.2, -0.5])))
check("superpose: co-propagating drivers on a shared medium add linearly",
      np.array_equal(_cd([_L1, _L2], "superpose"), _L1 + _L2))

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
_naive = [_eye(_b7(gm, 0.0).waveform(), _g7, levels=4) for gm in (0.0, 0.15, 0.3, 0.4)]
check("realized labels expose the confound: reflection alone closes the eye",
      _naive[0] > _naive[-1] and all(_naive[i] >= _naive[i + 1] for i in range(len(_naive) - 1)),
      f"eye {_naive[0]:.3f} -> {_naive[-1]:.3f}")
# hold eye height fixed by solving insertion loss as reflection is swept
_tgt = _eye(_b7(0.05, 2.0).waveform(), _g7, levels=4)
# the metric reaches `hold_constant` as a callback it calls with (x, grid), so the level count
# is bound here rather than defaulted inside the measurement
_recs = _hold(_b7, "gamma", [0.05, 0.15, 0.25, 0.35], "eye", _tgt,
              "loss_db", (0.0, 4.0), _g7, lambda x, g: _eye(x, g, levels=4), tol=0.004)
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
_di = abs(_eh(_isi, _g8, levels=4, defn="sigma") - _eh(_isi, _g8, levels=4, defn="contour"))
_dg = abs(_eh(_gau, _g8, levels=4, defn="sigma") - _eh(_gau, _g8, levels=4, defn="contour"))
# The separation is the claim: the two definitions disagree 5.6x more under ISI than under
# noise. The Gaussian bound is 0.035 rather than 0.02 because a correctly-sharp edge leaves more
# of the record in transition, where the two definitions read a noisy eye slightly differently.
check("named eye definitions agree under Gaussian noise, diverge under deterministic ISI",
      _dg < 0.035 and _di > 0.05 and _di > _dg + 0.03, f"|diff| ISI={_di:.3f} Gauss={_dg:.3f}")
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
      _eh(_wffe, _g11, levels=4) > _eh(_noffe, _g11, levels=4) + 0.02,
      f"eye {_eh(_noffe, _g11, levels=4):.3f} -> {_eh(_wffe, _g11, levels=4):.3f}")

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
# 0.003, not 0.005: the zero-phase path's corner is now 1.39x wider (it has to be, to deliver the
# rise time asked for), so the filter is shorter in time and leaks proportionally less. The hazard
# is unchanged in kind -- the ratio to the causal path is still enormous.
check("the hazard is real: zero-phase edge shaping leaks pre-cursor behind a causal channel",
      _rz18 > 10 * _rc18 and _rz18 > 0.003, f"zero-phase pre/post = {_rz18:.4f} vs causal {_rc18:.4f}")

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

print("== 4D-PAM5 / 8B1Q4: the symbol statistics are the CODE\'s, not a uniform PAM5 draw ==")
from wfmsynth import quinary as _Q
_con = _Q.quartet_constellation()
_cnt = {_s: int((_con == _s).sum()) for _s in _Q.QUINARY_SYMBOLS}
check("the 512-point constellation partitions into 8 subsets of 64, over 625 quartets",
      _con.shape == (8, 64, 4) and sum(len(_x) for _x in _Q.quartet_subsets()) == 625)
check("the marginal level distribution is {21,28,30,28,21}/128, NOT 1/5 each",
      _cnt == {-2: 336, -1: 448, 0: 480, 1: 448, 2: 336},
      f"counts {_cnt} of {_con.size}; uniform would be {_con.size // 5} each")
check("it is balanced (no DC of its own) and costs 0.58 dB less power than uniform PAM5",
      sum(_s * _cnt[_s] for _s in _cnt) == 0
      and abs(sum(_s * _s * _cnt[_s] for _s in _cnt) / _con.size - 1.75) < 1e-12)
_alpha = _Q.quinary_volts(np.array(_Q.QUINARY_SYMBOLS))
_lv = lambda _t: len({round(_t[0] * _a + _t[1] * _b, 12) for _a in _alpha for _b in _alpha})
check("partial response 3/4 + 1/4 z^-1 gives 17 observed levels; (1+D)/2 would give 9",
      _lv(_Q.PARTIAL_RESPONSE_TAPS) == 17 and _lv((0.5, 0.5)) == 9,
      f"{_lv(_Q.PARTIAL_RESPONSE_TAPS)} vs {_lv((0.5, 0.5))} levels")
check("test mode 1 is the clause's own 2048 symbols: four isolated pulses, then 1024 zeros",
      len(_Q.test_mode_symbols(1)) == 2048
      and [_Q.test_mode_symbols(1)[_i] for _i in (0, 128, 256, 384)] == [2, -2, 1, -1]
      and (_Q.test_mode_symbols(1)[1024:] == 0).all())

print("== the hybrid echo is a copy of a DIFFERENT stream, which `reflect` cannot be ==")
_gq = Grid(fs=125e6 * 16, baud=125e6, n=1 << 13)
_lane = lambda _k, _ro: P.from_symbols(_Q.pam5_symbols(pair=_k, n_ui=_gq.n // 16, role=_ro),
                                       n=_gq.n, tr_frac=0.15, causal=True)
_want, _own = _lane(0, "slave"), _lane(0, "master")
check("the two ENDS of the link are independent: they run different scrambler polynomials",
      abs(float(np.corrcoef(_want, _own)[0, 1])) < 0.06,
      f"master vs slave corr={float(np.corrcoef(_want, _own)[0, 1]):+.4f} against the "
      f"1/sqrt(512)={512 ** -0.5:.3f} of two independent streams; two SEEDS of one 33-bit LFSR "
      f"would be phases of ONE sequence and correlate at 0.58")
check("a reflection of silence is silence; an echo of own transmit is not",
      np.abs(P.multi_reflection(np.zeros(_gq.n), grid=_gq, td_ps=200.0)).max() == 0.0
      and np.std(P.hybrid_echo(np.zeros(_gq.n), _own, grid=_gq, isolation_db=20.0)) > 0.0)
_term = P.hybrid_echo(_want, _own, grid=_gq, isolation_db=14.0, f_hp_frac=1e-4) - _want
_cr = lambda _a, _b: float(np.corrcoef(_a, _b)[0, 1])
check("the echo tracks own transmit, not the wanted signal",
      abs(_cr(_term, _own)) > 0.9 and abs(_cr(_term, _want)) < 0.06,
      f"own={_cr(_term, _own):.3f} wanted={_cr(_term, _want):.3f}")
check("infinite isolation returns the wanted samples BIT-FOR-BIT",
      np.array_equal(P.hybrid_echo(_want, _own, grid=_gq, isolation_db=np.inf, td_ps=200.0), _want))
check("6 dB more isolation halves the echo amplitude (the dB convention is delivered)",
      abs(20 * np.log10(np.std(P.hybrid_echo(_want, _own, grid=_gq, isolation_db=14.0) - _want)
                        / np.std(P.hybrid_echo(_want, _own, grid=_gq, isolation_db=20.0) - _want))
          - 6.0) < 0.05)

print("== one observed channel: wanted + echo + NEXT + FEXT, and it couples the DERIVATIVE ==")
_near = [_lane(_k, "master") for _k in (1, 2, 3)]
_far = [_lane(_k, "slave") for _k in (1, 2, 3)]
_obs = lambda **_kw: P.single_pair_observation(_want, _own, _near, _far, grid=_gq, **_kw)
check("every coupling off -> the wanted signal, bit for bit", np.array_equal(_obs(), _want))
check("... and an explicitly infinite budget is the same exact zero",
      np.array_equal(_obs(echo_db=np.inf, next_db=np.inf, fext_db=np.inf), _want))
_full = _obs(**_Q.CLAUSE_40_BUDGET) - _want
_parts = sum(_obs(**{_k: _v}) - _want for _k, _v in _Q.CLAUSE_40_BUDGET.items())
check("the four mechanisms superpose, so a budget is auditable term by term",
      np.abs(_full - _parts).max() < 1e-12 * np.abs(_full).max())
check("Clause 40's budget puts the aggression well below the wanted signal but far above noise",
      0.02 < np.std(_full) / np.std(_want) < 0.5,
      f"aggression {np.std(_full) / np.std(_want):.4f} of the wanted signal")
_fx = P.crosstalk_sum(_want, _far, [0.1] * 3, kinds="fext") - _want
_dd = sum(np.gradient(_a) for _a in _far)
check("FEXT couples the aggressors' DERIVATIVE, not their level",
      abs(_cr(_fx, _dd)) > 0.99 and abs(_cr(_fx, sum(_far))) < 0.15,
      f"d/dt={_cr(_fx, _dd):.4f} level={_cr(_fx, sum(_far)):.4f}")

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
      _eh(_eq28, _g28, levels=4) > _eh(_no28, _g28, levels=4) + 0.02, f"eye {_eh(_no28,_g28, levels=4):.3f} -> {_eh(_eq28,_g28, levels=4):.3f}")
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
      _eh(P.differential_mode(_ps, _ns), _g32, levels=4) < _eh(P.differential_mode(_p0, _n0), _g32, levels=4) - 0.02
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

print("== level coding: the level map is exact, Gray costs one bit, precoding bounds the burst ==")
from wfmsynth import coding as _CD
_ulp = float(np.spacing(2.0 / 3.0))
_lv4 = P.pam_levels(4)
check("the four PAM4 levels are equally spaced to 1 ulp (not linspace, which is 5.6e-17 off)",
      np.ptp(np.diff(_lv4)) <= _ulp and float(np.max(np.abs(np.linspace(-1, 1, 4) - _lv4))) > 5e-17,
      f"spacing spread {np.ptp(np.diff(_lv4)):.2e} <= 1 ulp {_ulp:.2e}; "
      f"linspace differs by {float(np.max(np.abs(np.linspace(-1, 1, 4) - _lv4))):.2e}")
check("the millivolt map is the same map scaled: 266.67 mV between adjacent levels",
      np.array_equal(_CD.PAM4_LEVELS_MV, _CD.PAM4_OUTER_MV * _lv4)
      and abs(float(np.diff(_CD.PAM4_LEVELS_MV)[0]) - 800.0 / 3.0) <= np.spacing(800.0 / 3.0),
      f"{np.round(_CD.PAM4_LEVELS_MV, 2).tolist()} mV")
_gb = _CD.GRAY_PAM4_BITS
check("Gray coding: every ADJACENT level pair differs in exactly one bit, skip-one pairs in two",
      [int(np.sum(_gb[i] != _gb[i + 1])) for i in range(3)] == [1, 1, 1]
      and int(np.sum(_gb[0] != _gb[2])) == 2 and int(np.sum(_gb[1] != _gb[3])) == 2,
      f"map {[tuple(int(v) for v in r) for r in _gb]}")
# TWO full PRBS13 periods: 8191 symbols, i.e. the whole PRBS13Q period, whose level and
# transition statistics are exact by construction rather than sampled. At 2048 symbols the
# 1/sqrt(n) spread is 0.02 and a 1e-3 claim about the histogram is not measurable.
_bits41 = P.prbs(13, 2 * 8191, seed=5).astype(int)
_y41 = _CD.precode(_bits41)
check("precoding is exactly invertible (GF(2), no tolerance)",
      np.array_equal(_CD.precode_inverse(_y41), _bits41))
_s41 = _CD.pam4_gray_encode(_y41)
_r41 = _CD.pam4_gray_encode(_bits41)
_p41 = np.bincount(_s41, minlength=4) / _s41.size
check("precoding leaves the eye alone: same level histogram and transition density, "
      "different sequence",
      float(np.max(np.abs(_p41 - 0.25))) < 1e-3
      and abs(float(np.mean(np.diff(_s41) != 0)) - float(np.mean(np.diff(_r41) != 0))) < 1e-3
      and float(np.mean(_s41 != _r41)) > 0.5,
      f"levels {np.round(_p41, 4).tolist()}, transitions {np.mean(np.diff(_s41) != 0):.4f} vs "
      f"{np.mean(np.diff(_r41) != 0):.4f}, symbols differing {np.mean(_s41 != _r41):.3f}")
_bad41 = _s41.copy(); _k41 = 512
_bad41[_k41] = _s41[_k41] + (1 if _s41[_k41] < 3 else -1)
_burst41 = int(np.sum(_CD.precode_inverse(_CD.pam4_gray_decode(_bad41)) != _bits41))
_rec41 = _CD.precode(_CD.precode_inverse(_bits41) ^ np.eye(1, len(_bits41), _k41, dtype=int)[0])
check("one adjacent-level error decodes to a BOUNDED 2-bit burst; the recursive receiver's "
      "is unbounded",
      _burst41 == 2 and int(np.sum(_rec41 != _bits41)) == len(_bits41) - _k41,
      f"precoded {_burst41} bits, recursive {int(np.sum(_rec41 != _bits41))} bits "
      f"(to the end of a {len(_bits41)}-bit record)")
_w42 = np.arange(1 << _CD.PAM3_BLOCK_BITS)
_b42 = ((_w42[:, None] >> np.arange(_CD.PAM3_BLOCK_BITS - 1, -1, -1)) & 1).ravel()
_s42 = _CD.pam3_encode(_b42)
check("PAM3 11b/7t round-trips over all 2048 codewords onto exactly three levels",
      np.array_equal(_CD.pam3_decode(_s42), _b42)
      and np.array_equal(np.unique(_s42), P.pam_levels(3)),
      f"{_CD.PAM3_BLOCK_BITS}b/{_CD.PAM3_BLOCK_SYMBOLS}t = "
      f"{_CD.PAM3_BITS_PER_SYMBOL:.4f} bits/symbol (log2 3 = {np.log2(3):.4f})")
check("7 is the shortest ternary block that carries 11 bits (3^6 < 2^11 <= 3^7)",
      3 ** _CD.PAM3_BLOCK_SYMBOLS >= 2 ** _CD.PAM3_BLOCK_BITS > 3 ** (_CD.PAM3_BLOCK_SYMBOLS - 1),
      f"729 < 2048 <= 2187, 139 spare codewords")

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

# THE CHECK THAT REPLACED "ROLLS OFF HF AT ALL". That one passed on a response twice as steep as
# the physics, and for months it did: the analog kinds ran their filter forwards AND backwards
# (`sosfiltfilt`), which squares |H|, on top of designing the Bessel with scipy's default
# `norm='phase'`, whose own -3 dB point is at 0.68 of Wn. A front end asked for 32 GHz realised
# 15.90 GHz. Nothing in this file noticed, because nothing here asked WHERE the corner was.
_gfe = Grid(fs=1024e9, n=1 << 16)


def _realised_corner_hz(fn, fc_hz, rising=False):
    """The -3 dB point of a stage, bisected on its OWN realised output."""
    def gain(f_hz):
        k = max(1, round(f_hz * _gfe.n / _gfe.fs))
        t = np.sin(2 * np.pi * k * np.arange(_gfe.n) / _gfe.n)
        y = np.asarray(fn(t), float)
        return abs(np.fft.rfft(y[:_gfe.n])[k] / np.fft.rfft(t)[k])
    lo, hi = fc_hz / 60.0, min(0.45 * _gfe.fs, fc_hz * 60.0)
    for _ in range(36):
        mid = np.sqrt(lo * hi)
        if (gain(mid) < 0.70710678) == rising:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


_fe_ratio, _fe_ratio0 = [], []
for _fb in (12e9, 20e9, 32e9, 40e9):
    for _kind in ("bessel", "gaussian"):
        _fe_ratio.append(_realised_corner_hz(
            lambda x, b=_fb, k=_kind: _sbw(x, _gfe, b, kind=k), _fb) / _fb)
        _fe_ratio0.append(_realised_corner_hz(
            lambda x, b=_fb, k=_kind: _sbw(x, _gfe, b, kind=k, causal=False), _fb) / _fb)
check("the analog front end realises the -3 dB point it was ASKED for (not half of it)",
      max(abs(np.array(_fe_ratio) - 1.0)) < 0.01,
      f"realised/requested {min(_fe_ratio):.4f}..{max(_fe_ratio):.4f} over 4 bandwidths x "
      f"{{bessel, gaussian}}")
check("  ... and the GATE OBSERVED FAILING: the zero-phase opt-out realises 0.43-0.83x of it",
      max(_fe_ratio0) < 0.90,
      f"realised/requested {min(_fe_ratio0):.4f}..{max(_fe_ratio0):.4f} -- a 32 GHz bessel-4 "
      f"front end behaving like a {32.0 * _fe_ratio0[4]:.1f} GHz one")

# The 10-90 % step rise of a gently-rolled-off low-pass is 0.35/BW. It is the spec sheet's own
# arithmetic, it needs no reference waveform, and it is where the halving shows up as a number a
# reader recognises.
_step = np.concatenate([np.zeros(_gfe.n // 2), np.ones(_gfe.n - _gfe.n // 2)])


def _rise_mid_ps(y):
    y = (y - y[:1000].mean()) / (y[-1000:].mean() - y[:1000].mean())
    _t = (np.arange(len(y)) - _gfe.n // 2) / _gfe.fs * 1e12
    def cr(lv):
        i = int(np.argmax(y > lv))
        return float(np.interp(lv, [y[i - 1], y[i]], [_t[i - 1], _t[i]]))
    return cr(0.9) - cr(0.1), cr(0.5)


_r32, _m32 = _rise_mid_ps(_sbw(_step, _gfe, 32e9, kind="bessel", order=4))
_r32o, _m32o = _rise_mid_ps(_sbw(_step, _gfe, 32e9, kind="bessel", order=4, causal=False))
_gd_cf = 2.1139 / (2 * np.pi * 32e9) * 1e12                 # analog Bessel-4 group delay at DC
check("a 32 GHz analog front end has the 0.35/BW rise time and the group delay of one",
      abs(_r32 * 32e9 / 1e12 - 0.35) < 0.01 and 0.6 < _m32 / _gd_cf < 1.0,
      f"rise {_r32:.2f} ps (rise*BW = {_r32 * 32e9 / 1e12:.4f}), 50 % at {_m32:+.2f} ps against "
      f"the closed form's {_gd_cf:.2f} ps")
check("  ... and the opt-out is 0.35/BW for HALF the bandwidth, with no delay at all",
      abs(_r32o * 32e9 / 1e12 - 0.728) < 0.01 and abs(_m32o) < 1.0,
      f"rise {_r32o:.2f} ps (rise*BW = {_r32o * 32e9 / 1e12:.4f}, i.e. 0.35/{32 / 2.08:.1f} GHz), "
      f"50 % at {_m32o:+.2f} ps")

# The probe's loading pole, in MAGNITUDE AND PHASE, against `1/(1 + j*f/fc)`. A zero-phase
# filter can carry the right rolloff shape and still not be an RC pole, which is exactly what
# the old default was.
_gpr = Grid(fs=400e9, baud=25e9, n=1 << 15)
from wfmsynth.instrument import rc_pole_hz as _rc_pole_hz
_fc_pr = _rc_pole_hz(50.0, 0.5e-12)
_pdb, _pph, _pdb0 = [], [], []
for _mult in (0.25, 0.5, 1.0, 2.0, 4.0):
    _kp = int(round(_fc_pr * _mult * _gpr.n / _gpr.fs)); _fp = _kp * _gpr.fs / _gpr.n
    _tp = np.sin(2 * np.pi * _kp * np.arange(_gpr.n) / _gpr.n)
    _Hp = np.fft.rfft(_pl(_tp, _gpr, c_load_f=0.5e-12))[_kp] / np.fft.rfft(_tp)[_kp]
    _Hp0 = np.fft.rfft(_pl(_tp, _gpr, c_load_f=0.5e-12, causal=False))[_kp] / np.fft.rfft(_tp)[_kp]
    _want_db = 10 * np.log10(1 + (_fp / _fc_pr) ** 2)
    _pdb.append(abs(-20 * np.log10(abs(_Hp)) - _want_db))
    _pph.append(abs(np.degrees(np.angle(_Hp)) + np.degrees(np.arctan(_fp / _fc_pr))))
    _pdb0.append((-20 * np.log10(abs(_Hp0))) / _want_db)
check("probe capacitive loading IS the RC pole 1/(1 + j*f/fc), magnitude AND phase",
      max(_pdb) < 1e-9 and max(_pph) < 1e-9,
      f"fc = {_fc_pr / 1e9:.4f} GHz; worst deviation {max(_pdb):.2e} dB and {max(_pph):.2e} deg "
      f"over f/fc = 0.25..4")
check("  ... and the GATE OBSERVED FAILING: the zero-phase opt-out is that closed form SQUARED",
      1.99 < min(_pdb0) and max(_pdb0) < 2.02,
      f"{min(_pdb0):.4f}..{max(_pdb0):.4f}x the closed form's dB at every frequency")
_tb40 = _tbj(_x40, _g40, rms_ps=1.5, rng=np.random.default_rng(0))
check("timebase jitter smears the eye horizontally (closes it)",
      _eh(_tb40, _g40, levels=4) < _eh(_x40, _g40, levels=4) - 0.02, f"eye {_eh(_x40,_g40, levels=4):.3f} -> {_eh(_tb40,_g40, levels=4):.3f}")

print("== Tx de-emphasis preset: a dB preset yields that transition/steady ratio ==")
_spb42 = 16
_taps42 = P.de_emphasis_taps(-3.5)   # -3.5 dB is 3.5 dB of DE-emphasis, the PCIe convention
_sq = np.repeat(np.array([-1.0]*8 + [1.0]*8), _spb42).astype(float)     # 8-UI runs, sharp edges
_y42 = P.tx_ffe(_sq, _taps42, _spb42, pre=0)
_lo_run = 8 * _spb42                                               # index where the high run starts
_emph = _y42[_lo_run + _spb42 // 2]                                # first UI after the transition
_steady = _y42[_lo_run + 5 * _spb42 + _spb42 // 2]                 # a mid-run (steady) UI
check("de_emphasis_taps(-3.5 dB) yields ~3.5 dB transition-to-steady ratio",
      abs(20 * np.log10(_emph / _steady) - 3.5) < 0.3, f"measured {20*np.log10(_emph/_steady):.2f} dB")

print("== electrical idle / LFPS: idle carries no data energy; LFPS is a low-freq burst ==")
from wfmsynth.impairments import electrical_idle as _eidle
_g43 = Grid(fs=200e9, baud=50e9, n=1 << 13)
_x43 = (Signal(seed=1, grid=_g43).carrier("pam4", n_ui=_g43.n // 4, pattern="prbs13q", causal=True)).waveform()
_iv = [(2000, 3000)]
_idle = _eidle(_x43, _iv)
_seg = slice(2200, 4800)
check("an electrical-idle interval carries ~no data energy vs the active signal",
      np.sqrt(np.mean(_idle[_seg] ** 2)) < 0.02 * np.sqrt(np.mean(_x43[_seg] ** 2)))
_lf = _eidle(_x43, _iv, grid=_g43, lfps_hz=500e6, amp=0.3)
_S = np.abs(np.fft.rfft(_lf[2000:5000])); _fseg = np.fft.rfftfreq(3000, d=_g43.dt)
check("LFPS fills the idle with a low-frequency periodic burst (peak near f_lfps)",
      _fseg[int(np.argmax(_S))] < 2e9 and abs(_fseg[int(np.argmax(_S))] - 500e6) < 2e8)

print("== laser chirp: a transition-edge frequency excursion scaling with alpha ==")
from wfmsynth.optical import to_optical as _to, laser_chirp as _chirp
_g44 = Grid(fs=200e9, baud=50e9, n=1 << 12)
_Popt44 = _to(P.nrz(n_ui=_g44.n // 8, seed=1, n=_g44.n, causal=True), er_db=10.0)
_c1 = _chirp(_Popt44, alpha=2.0, grid=_g44)
_c2 = _chirp(_Popt44, alpha=4.0, grid=_g44)
# chirp energy concentrates at transitions (where |dP| is large), not on steady levels
_edges = np.abs(np.gradient(_Popt44)) > 0.2 * np.abs(np.gradient(_Popt44)).max()
check("chirp is concentrated at intensity transitions (near-zero on steady levels)",
      np.mean(np.abs(_c1[_edges])) > 5 * np.mean(np.abs(_c1[~_edges])))
check("chirp magnitude scales with the alpha (linewidth-enhancement) parameter",
      abs(np.max(np.abs(_c2)) / (np.max(np.abs(_c1)) + 1e-30) - 2.0) < 0.1)

print("== low-speed buses: open-drain wired-AND, UART framing ==")
from wfmsynth.bus import open_drain as _od, uart_frame as _uf, uart_decode as _ud
_d1 = np.array([1, 1, 0, 1, 1, 1])
_d2 = np.array([1, 0, 1, 1, 0, 1])
_bus = _od([_d1, _d2])
check("open-drain wired-AND pulls low when ANY driver is active, else floats to the pull-up",
      np.array_equal(_bus, np.array([1., 0., 0., 1., 0., 1.]))
      and np.array_equal(_od([np.ones(5), np.ones(5)]), np.ones(5)))
_msg = [0x55, 0xA3, 0x00, 0xFF]
_wave = _uf(_msg, samples_per_bit=16)
check("UART framing is idle-high with start/stop bits and decodes back to the bytes",
      _ud(_wave, 16) == _msg and _wave[0] == 0.0 and _wave[9 * 16 + 8] == 1.0)

print("== carrier from arbitrary symbols: a coded/scrambled stream drives synthesis, round-trips ==")
from wfmsynth.coding import dc_balanced as _dcb2
_bits45 = _dcb2(P.prbs(7, 512, seed=2), block=8)             # a real coded stream
_lev45 = 2.0 * _bits45 - 1.0                                 # NRZ levels
_n45 = len(_lev45) * 16
_w45 = P.from_symbols(_lev45, n=_n45, causal=True)
_spb45 = _n45 / len(_lev45)
_recov = (_w45[(np.arange(len(_lev45)) * _spb45 + _spb45 / 2).astype(int)] > 0).astype(int)
check("a carrier built from an arbitrary (coded) symbol stream reproduces those bits",
      np.array_equal(_recov, _bits45))

print("== waveform-level clock-recovery fold: recovered clock opens a low-freq-jittered eye ==")
from wfmsynth.cdr import recover_and_fold as _fold, timing_source as _ts2, apply_phase as _ap2
_g29 = Grid(fs=400e9, baud=50e9, n=1 << 15); _nui29 = int(_g29.n // _g29.samples_per_ui)
_base29 = (Signal(seed=1, grid=_g29).carrier("pam4", n_ui=_nui29, pattern="prbs13q", causal=True)).waveform()
_lf29 = _ap2(_base29, _ts2(_g29.n, _g29, pj=dict(amp_ps=8.0, f_hz=2e6)))     # slow wander
_hf29 = _ap2(_base29, _ts2(_g29.n, _g29, pj=dict(amp_ps=8.0, f_hz=2e9)))     # fast jitter
check("the CDR-folded eye is more OPEN than the fixed-grid eye for low-frequency jitter",
      _fold(_lf29, _g29) > _eh(_lf29, _g29, levels=4) + 0.02,
      f"fixed {_eh(_lf29,_g29, levels=4):.3f} -> folded {_fold(_lf29,_g29):.3f}")
check("high-frequency jitter is NOT tracked out (folded ~ fixed-grid eye)",
      abs(_fold(_hf29, _g29) - _eh(_hf29, _g29, levels=4)) < 0.05)

print("== generalized two-rate acquisition (issue #51): sim grid -> acquisition -> stored record ==")
import json as _json51
from wfmsynth.acquire import AcquisitionProfile as _AP, record_decimation as _rdec
_gsim = Grid(fs=200e9, baud=25e9, n=1 << 14)
_prof = _AP(sample_rate_hz=2.5e9, record_length=1024, input_bandwidth_hz=800e6, enob=7,
            sample_clock_jitter_rms_s=0.5e-12, noise_floor=dict(rms=1e-3, shape="pink"))
_asig = (Signal(seed=1, grid=_gsim).carrier("nrz", n_ui=256, causal=True)
         .lossy(loss_db=6.0, loss_at_ghz=12.0, causal=True).acquire(_prof))
_stored = _asig.waveform()
check("two-rate: fine sim grid >= acquisition grid, and digitized length == record_length",
      _gsim.fs >= _prof.sample_rate_hz and len(_stored) == 1024)
_taps = (Signal(seed=1, grid=_gsim).carrier("nrz", n_ui=256, causal=True)
         .lossy(loss_db=6.0, loss_at_ghz=12.0, causal=True)).acquire_taps(_prof)
check("taps: 'simulated' and 'digitized' differ; info reports realized sim vs acq rates",
      _taps["simulated"].shape != _taps["digitized"].shape
      and _taps["info"]["sim_fs"] > _taps["info"]["acq_fs"]
      and _taps["info"]["record_length"] == 1024)
_rec51 = _json51.loads(_json51.dumps(_asig.recipe()))
check("acquire recipe round-trips bit-for-bit (front end + jitter + noise via role streams)",
      np.array_equal(Signal.from_recipe(_rec51).waveform(), _stored))
_pulse51 = np.zeros(4096); _pulse51[2001:2007] = 1.0        # a 6-sample narrow pulse
_ph = _rdec(_pulse51, mode="peak_hold", depth=512); _sm = _rdec(_pulse51, mode="sample", depth=512)
check("record decimation: a narrow pulse survives peak_hold (2-ch) but is lost under naive sample",
      _ph.shape == (2, 512) and _ph.max() > 0.9 and _sm.max() < 0.5)
_a51 = (Signal(seed=1, grid=_gsim).carrier("nrz", n_ui=256, causal=True).scope(bw_hz=5e9)).waveform()
_b51 = (Signal(seed=1, grid=_gsim).carrier("nrz", n_ui=256, causal=True).input_bandwidth(bw_hz=5e9)).waveform()
check("backward-compat: .input_bandwidth() / .sample_clock_jitter() alias scope/timebase bit-identically",
      np.array_equal(_a51, _b51))

print("== localized events: identity outside support; 2nd-order overshoot; placement + labels ==")
from wfmsynth.events import (place_events as _pe, apply_events as _ae, label_windows as _lw,
                             nominal_ui_windows as _nuw, second_order_step as _s2,
                             step_overshoot_fraction as _sof, damped_sinusoid as _dsin,
                             slope_reversals as _srev)
from wfmsynth.compose import Signal as _SigEv
from wfmsynth.grid import Grid as _GridEv
_gev = _GridEv(fs=10e9, baud=1e9, n=4096)
_nui_ev = int(_gev.n // _gev.samples_per_ui)
_xev = _SigEv(seed=1, grid=_gev).carrier("nrz", n_ui=_nui_ev, causal=True, tr_frac=0.2).waveform()
_eev = _pe(len(_xev), kind="glitch", on="symbols", grid=_gev, rng=np.random.default_rng(0),
           x=_xev, indices=[20, 80], severity=0.8)
_yev, _mev, _eev = _ae(_xev, _eev, grid=_gev)
check("apply_events is bit-identical outside the mask",
      np.array_equal(_yev[_mev == 0], _xev[_mev == 0]))
check("apply_events modifies the support and emits a low-duty mask",
      np.max(np.abs(_yev - _xev)) > 0.05 and 0.0 < (_mev > 0).mean() < 0.15)
_t2 = np.linspace(0, 8e-9, 2000)
_z2 = 0.3
_s2y = _s2(_t2, wn=2 * np.pi * 1e9, zeta=_z2)
_meas_os = (_s2y.max() - 1.0) / 1.0
check("2nd-order step peak overshoot matches exp(-pi zeta / sqrt(1-zeta^2))",
      abs(_meas_os - _sof(_z2)) < 0.02, f"meas={_meas_os:.4f} theory={_sof(_z2):.4f}")
_kern = _dsin(512, 1 / _gev.fs, f0=500e6, tau=6e-9, amp=1.0)
_K = np.abs(np.fft.rfft(_kern))
_fk = np.fft.rfftfreq(len(_kern), d=1 / _gev.fs)
check("damped-sinusoid kernel is causal at t=0 and peaks near f0",
      abs(_kern[0]) < 1e-12 and abs(_fk[int(np.argmax(_K[1:])) + 1] - 500e6) / 500e6 < 0.25)
_txev = P.carrier_symbols("nrz", _nui_ev, seed=1)
_runt_ui = int((np.where(np.diff(_txev) != 0)[0] + 1)[4])
_erunt = _pe(len(_xev), kind="runt", on="symbols", grid=_gev, x=_xev, n_ui=_nui_ev,
             indices=[_runt_ui], severity=0.9, floor=0.2)
_yrunt, _mrunt, _erunt = _ae(_xev, _erunt, grid=_gev)
_sl = slice(*_erunt.events[0].span(len(_xev)))
_prev = _xev[_sl.start - 1] if _sl.start else _xev[_sl.start]
_exc = lambda w: np.max(np.abs(w[_sl] - _prev))
_other = _mrunt == 0
check("runt lowers the target-UI peak and leaves the rest identical",
      _exc(_yrunt) < 0.85 * _exc(_xev) and np.array_equal(_yrunt[_other], _xev[_other]))
_counts = []
for _si in range(40):
    _ep = _pe(len(_xev), kind="glitch", on="poisson", grid=_gev,
              rng=np.random.default_rng(_si), rate_hz=2e7, severity=0.5)
    _counts.append(len(_ep))
check("poisson placement mean count tracks rate * duration",
      abs(np.mean(_counts) - (2e7 * _gev.duration)) < 3.0,
      f"mean={np.mean(_counts):.2f} expected={2e7 * _gev.duration:.2f}")
_wins = _nuw(len(_xev), _gev, half_ui=1.0)
_rows = _lw(_wins, _eev, x=_yev)
_hit = [r for r in _rows if r["events"]]
check("label_windows attaches each event only to overlapping UI windows",
      len(_hit) >= 1 and all(any(e["sample"] >= r["start"] and e["sample"] < r["stop"]
                                 for e in r["events"]) for r in _hit))
_sig_ev = (_SigEv(seed=7, grid=_gev).carrier("nrz", n_ui=_nui_ev, causal=True)
           .events("runt", on="symbols", count=3, severity=0.6, floor=0.3))
_w1 = _sig_ev.waveform()
_w2, _el2 = _sig_ev.realize()
_rec_ev = _sig_ev.recipe()
check("Signal.events recipe round-trips and realize() matches waveform()",
      np.array_equal(_w1, _w2) and np.array_equal(_SigEv.from_recipe(_rec_ev).waveform(), _w1)
      and len(_el2) == 3)
_syms_d = np.array([-1.0] * 8 + [1.0] * 24 + [-1.0] * 8)
_wdroop = P.from_symbols(_syms_d, n=2048, causal=True, tr_frac=0.15)
_edroop = _pe(len(_wdroop), kind="droop", on="pattern", grid=_GridEv(fs=10e9, baud=1e9, n=2048),
              x=_wdroop, symbols=_syms_d, min_run=10, count=1, severity=0.8, depth=0.4)
_ydroop, _mdroop, _ = _ae(_wdroop, _edroop, grid=_GridEv(fs=10e9, baud=1e9, n=2048))
_lo_d, _hi_d = _edroop.events[0].span(len(_wdroop))
_third = max(1, (_hi_d - _lo_d) // 3)
check("droop sags |level| through a long run",
      np.mean(np.abs(_ydroop[_hi_d - _third:_hi_d])) < 0.85 * np.mean(np.abs(_ydroop[_lo_d:_lo_d + _third])))
_eslow = _pe(len(_xev), kind="slow_edge", on="edges", grid=_gev, x=_xev, symbols=_txev,
             which="rising", indices=[2], severity=0.9, tr_factor=6.0, n_ui=_nui_ev)
_yslow, _mslow, _eslow = _ae(_xev, _eslow, grid=_gev)
_t0s = int(_eslow.events[0].sample)
_nb = slice(max(0, _t0s - 3), min(len(_xev), _t0s + int(_gev.samples_per_ui)))
check("slow_edge reduces max |dv/dt| on the targeted edge",
      np.max(np.abs(np.diff(_yslow[_nb]))) < 0.90 * np.max(np.abs(np.diff(_xev[_nb]))))
_enm = _pe(len(_xev), kind="nonmonotonic", on="edges", grid=_gev, x=_xev, which="rising",
           indices=[3], severity=0.9)
_ynm, _, _enm = _ae(_xev, _enm, grid=_gev)
_n0, _n1 = _enm.events[0].span(len(_xev))
# BACKWARD TRAVEL, not a count of slope reversals. A causal Bessel edge rings, so it already has
# reversals of its own and the count stopped discriminating once the edge was sharp enough for its
# window to be a handful of samples. Backward travel is what "nonmonotonic" actually asserts --
# the edge moves away from where it is going -- and it separates by four orders of magnitude:
# 1e-5 for the plain edge against 0.615 for the perturbed one.
def _backward(_v):
    _d = np.diff(np.asarray(_v, float))
    return float(-_d[_d < 0].sum())
check("nonmonotonic edge travels backwards inside its window",
      _backward(_ynm[_n0:_n1]) > 100 * _backward(_xev[_n0:_n1]) + 0.05,
      f"backward travel {_backward(_ynm[_n0:_n1]):.5f} vs plain {_backward(_xev[_n0:_n1]):.5f}")

from wfmsynth import optical as _OPT

print("== optical E/O/E: square-law photodetection closes the loop (i = R·|E|^2, real) ==")
_Efield = np.array([0.5 + 0.5j, 1 + 0j, 0.3 - 0.2j])
_iphoto = _OPT.photodetect(_Efield, responsivity=2.0, shot=False)
check("square-law detection: photocurrent = R·|E|^2, real-valued (O->E closes E/O/E)",
      np.isrealobj(_iphoto) and np.allclose(_iphoto, 2.0 * np.abs(_Efield) ** 2))

print("== MZM: cos^2 electro-optic transfer (a real modulator, not a linear intensity ramp) ==")
_drv = np.linspace(-1.0, 1.0, 401)
_Pmzm = np.abs(_OPT.modulate_field(_drv, kind="mzm", bias=0.5, p_avg=1.0)) ** 2
_argm = (np.pi / 2.0) * ((_drv - _drv.mean()) / (np.max(np.abs(_drv - _drv.mean())) + 1e-12) + 0.5)
check("MZM intensity is cos^2 of the drive (nonlinear transfer, not a linear ramp)",
      np.corrcoef(_Pmzm, np.cos(_argm) ** 2)[0, 1] > 0.999 and np.corrcoef(_Pmzm, _drv)[0, 1] ** 2 < 0.99)

print("== optical fibre: chromatic dispersion spreads a pulse (physical beta2*L on the field) ==")
_gfib = Grid(fs=1e12, baud=25e9, n=8192)
_fpulse = np.zeros(8192, complex); _fpulse[4000:4010] = 1.0
_wid_in = int(np.sum(np.abs(_fpulse) > 0.1))
_fout = _OPT.fiber(_fpulse, length_km=20.0, D_ps_nm_km=17.0, grid=_gfib)
_wid_out = int(np.sum(np.abs(_fout) > 0.1 * np.abs(_fout).max()))
check("fibre chromatic dispersion broadens an optical pulse", _wid_out > 2 * _wid_in)

print("== chirp x dispersion: DML laser chirp materially changes the DETECTED signal after fibre ==")
_gd = Grid(fs=2560e9, baud=80e9, n=32 * 1500)
def _detected(alpha):
    return np.asarray(Signal(seed=1, grid=_gd).carrier("pam4", n_ui=1500, pattern="prbs13q", causal=True)
                      .eo(kind="dml", alpha=alpha, er_db=8).fiber(length_km=4.0, D_ps_nm_km=17.0)
                      .photodetect(shot=False).tia().waveform(), float)
_w_nochirp, _w_chirp = _detected(0.0), _detected(4.0)
check("chirp x dispersion interaction is modeled (chirped field detects differently post-fibre)",
      np.corrcoef(_w_nochirp, _w_chirp)[0, 1] < 0.98)

print("== laser linewidth: a finite linewidth adds random-walk phase noise to the field ==")
_glw = Grid(fs=100e9, baud=10e9, n=4096)
_cw = np.ones(4096)                                            # steady drive: chirp-free, phase ~ const
_f0 = _OPT.modulate_field(_cw, kind="dml", linewidth_hz=0.0, grid=_glw)
_flw = _OPT.modulate_field(_cw, kind="dml", linewidth_hz=1e7, grid=_glw, rng=np.random.default_rng(0))
check("finite laser linewidth adds random-walk phase noise (var grows vs zero-linewidth)",
      np.var(np.unwrap(np.angle(_flw))) > 10.0 * np.var(np.unwrap(np.angle(_f0))) + 1e-6)

print("== optical MPI: a delayed field copy beats COHERENTLY (the real source of the MPI penalty) ==")
# A CW field carries flat power (|E|^2 == 1) but a random-walk phase from finite linewidth; a delayed
# copy has a decorrelated phase, so the field-domain ghost beats in TIME while the intensity ghost stays
# flat. (This is exactly why laser linewidth governs the MPI penalty.) Measure in the overlap region.
_cwm = _OPT.modulate_field(np.ones(4096), kind="dml", linewidth_hz=5e8, grid=_glw,
                           rng=np.random.default_rng(0))
_d = 20
_ov = slice(_d, None)
_det_field = np.abs(_OPT.field_mpi(_cwm, delay_samples=_d, reflectivity=0.2)) ** 2     # coherent beat
_det_int = _OPT.mpi(np.abs(_cwm) ** 2, delay_samples=_d, reflectivity=0.2)             # flat intensity ghost
check("field-domain optical MPI produces a coherent detected beat (an intensity ghost does not)",
      np.var(_det_field[_ov]) > 1e-3 and np.var(_det_int[_ov]) < 1e-9)

# ===================================================================================================
# The loss TREND, and the storage lattice — the two realism gaps, each measured by an instrument
# that was first made to recover an answer we constructed.
# ===================================================================================================
from wfmsynth import instrument as INST

print("== insertion-loss instrument: does it recover a loss curve we constructed? ==")
_gt = Grid(fs=64e9, baud=16e9, n=1 << 14)


def _il_db(f_ghz_want, **kw):
    """Measure a channel's insertion loss in dB at a stated frequency, by rendering an impulse
    through it and reading the rfft back. The ONLY loss instrument used below."""
    imp = np.zeros(_gt.n); imp[0] = 1.0
    # `linear=False` -- the pinned-length transform, where an impulse in gives H out exactly.
    # The default path applies the SAME H as a linear convolution and truncates the response at
    # the record, which loses its tail: this channel's impulse response is longer than 16384
    # samples, and reading its rfft back would put the constructed 8.00 dB at 9.50. That is a
    # property of the measurement, not of the channel; `tests/test_linear_convolution.py` is
    # what checks the padded path, against `np.convolve` and a hand-written two-tap echo.
    H = np.abs(np.fft.rfft(P.lossy_channel(imp, grid=_gt, linear=False, **kw)))
    f = np.fft.rfftfreq(_gt.n) * _gt.fs / 1e9
    k = int(np.argmin(np.abs(f - f_ghz_want)))
    return -20.0 * np.log10(H[k] + 1e-300)


# CONSTRUCTED CASE: a trend whose loss at 4 and 8 GHz is arithmetic we can do by hand.
# trend=(a,b,c) means |S21|dB = a*sqrt(f)+b*f+c, so IL(f) = -(a*sqrt(f)+b*f+c).
# (a,b,c) = (0, -2, 0) is exactly 2 dB per GHz: 8 dB at 4 GHz, 16 dB at 8 GHz, ratio 0.5.
_known = (0.0, -2.0, 0.0)
check("loss instrument recovers a CONSTRUCTED trend at 4 GHz (8.00 dB by hand)",
      abs(_il_db(4.0, trend=_known) - 8.0) < 0.02, f"measured {_il_db(4.0, trend=_known):.4f} dB")
check("loss instrument recovers a CONSTRUCTED trend at 8 GHz (16.00 dB by hand)",
      abs(_il_db(8.0, trend=_known) - 16.0) < 0.02, f"measured {_il_db(8.0, trend=_known):.4f} dB")
# and the sqrt term alone: (a,b,c)=(-4,0,0) is 4*sqrt(f) dB -> 8.00 at 4 GHz, 11.3137 at 8 GHz
check("loss instrument recovers a CONSTRUCTED sqrt-only trend (4*sqrt(f): 8.000 / 11.314 dB)",
      abs(_il_db(4.0, trend=(-4.0, 0.0, 0.0)) - 8.0) < 0.02
      and abs(_il_db(8.0, trend=(-4.0, 0.0, 0.0)) - 4.0 * np.sqrt(8.0)) < 0.02)
# NEGATIVE CONTROL: the instrument must not report a loss a channel does not have.
check("loss instrument reports ~0 dB on a constructed transparent channel (no false positive)",
      abs(_il_db(8.0, trend=(0.0, 0.0, 0.0))) < 1e-9)

print("== the fixed skin/dielectric mix: one anchor cannot set a shape (the measured defect) ==")
# The anchored model's L(4)/L(8) is a CONSTANT, whatever loss is requested — that is the whole bug.
_ratios = [_il_db(4.0, loss_db=d, loss_at_ghz=8.0) / _il_db(8.0, loss_db=d, loss_at_ghz=8.0)
           for d in (7.18, 15.07, 21.75, 26.05, 34.06)]
check("anchored loss_db has ONE hard-wired L(4)/L(8) for every channel it can make",
      max(_ratios) - min(_ratios) < 1e-6 and abs(_ratios[0] - 0.617) < 0.005,
      f"ratio {_ratios[0]:.4f} across 7.2-34.1 dB, spread {max(_ratios) - min(_ratios):.2e}")
# Real measured backplanes (from measured IEEE 802.3ap backplane-channel S-parameter files) run
# 0.468-0.678. `trend` reaches that range; the anchored model cannot leave 0.617.
_r_lo = _il_db(4.0, trend=(0.0, -2.0, 0.0)) / _il_db(8.0, trend=(0.0, -2.0, 0.0))     # pure dielectric
_r_hi = _il_db(4.0, trend=(-4.0, 0.0, 0.0)) / _il_db(8.0, trend=(-4.0, 0.0, 0.0))     # pure skin
check("trend spans the measured 0.468-0.678 band of real boards (0.500 dielectric -> 0.707 skin)",
      abs(_r_lo - 0.5) < 0.002 and abs(_r_hi - 0.7071) < 0.002,
      f"L4/L8 {_r_lo:.4f} .. {_r_hi:.4f}, vs the anchored model's fixed {_ratios[0]:.4f}")
# Passivity: a fit whose c > 0 would otherwise deliver GAIN at DC.
check("a trend that fits to GAIN is clamped to a passive channel (IL >= 0 everywhere)",
      _il_db(0.5, trend=(-1.0, 1.0, 20.0)) >= -1e-9 and _il_db(4.0, trend=(-1.0, 1.0, 20.0)) >= -1e-9)
# `trend` ignores the per-inch knobs rather than scaling by them: a fitted curve is not dB/in.
check("trend ignores length_in/tand (a fitted curve is a whole channel, not a per-inch coefficient)",
      abs(_il_db(8.0, trend=_known, length_in=1.0, tand=0.001)
          - _il_db(8.0, trend=_known, length_in=20.0, tand=0.03)) < 1e-9)

print("== the storage lattice: a stored record is int16 codes, and that IS its noise floor ==")
_fs_store = 256e9
# CONSTRUCTED CASE 1 — the lattice detector. A signal we quantised ourselves to a pitch we chose.
_rs = np.random.default_rng(4)
_cont = np.cumsum(_rs.normal(0, 1e-3, 1 << 16))                      # a continuous wander
_qknown = 4.88e-4
_lat = np.round(_cont / _qknown) * _qknown


def _on_lattice(y, q):
    return float(np.mean(np.abs(y / q - np.round(y / q)) < 1e-9))


check("lattice detector recovers a CONSTRUCTED pitch (on-lattice fraction 1.0000)",
      _on_lattice(_lat, _qknown) > 0.9999, f"{_on_lattice(_lat, _qknown):.4f}")
check("lattice detector does NOT fire on the same signal unquantised (negative control)",
      _on_lattice(_cont, _qknown) < 0.01, f"{_on_lattice(_cont, _qknown):.4f}")


def _stopband_psd_db_per_hz(y, fs_hz, lo_frac=0.80, hi_frac=0.98):
    """One-sided PSD averaged over an empty band, in dB/Hz. The floor instrument.

    HANN-WINDOWED, with the window's noise power divided back out, because the thing being
    measured is 17 dB below what a rectangular window leaks. One step of the record's own
    amplitude in the periodic extension puts A^2/(2*n*fs) of white leakage across the band:
    for A ~ 0.9 at n = 65536 and fs = 256 GSa/s that is -166 dB/Hz, and an 11-bit lattice's
    q^2/12/(fs/2) is -182. Our records used to have no such step -- a frequency-domain stage
    wrapped, which made them exactly periodic -- and now they do (U-16), so this instrument
    would read the edge instead of the floor. The window costs nothing here: it still recovers
    both CONSTRUCTED quantisation floors above to 0.03 dB."""
    n = len(y)
    w = np.hanning(n)
    Y = np.fft.rfft((y - y.mean()) * w)
    psd = (np.abs(Y) ** 2) * (2.0 / (n * n)) * (n / fs_hz) / float((w * w).mean())
    f = np.fft.rfftfreq(n, 1.0 / fs_hz)
    band = (f > lo_frac * fs_hz / 2) & (f < hi_frac * fs_hz / 2)
    return float(10.0 * np.log10(np.mean(psd[band]) + 1e-300))


# CONSTRUCTED CASE 2 — the floor instrument, on pure quantisation error of a known step.
# A quantiser of step q contributes q^2/12 spread flat over fs/2: that is the closed form the
# three real captures matched to under 0.1 dB, and it is what this must recover.
for _qk in (2.1227e-4, 4.8904e-4):
    _err = _rs.uniform(-_qk / 2, _qk / 2, 1 << 18)
    _pred = INST.quantisation_floor_db_per_hz(_qk, _fs_store)
    _meas = _stopband_psd_db_per_hz(_err, _fs_store)
    check(f"floor instrument recovers q^2/12/(fs/2) for a CONSTRUCTED q={_qk * 1e6:.1f} uV",
          abs(_meas - _pred) < 0.2, f"predicted {_pred:.2f}, measured {_meas:.2f} dB/Hz")

# THE MECHANISM. Same recipe, rendered twice: without the export step and with it.
# (`n_ui=32` is stated here and in the lead-in and filter checks below because that is what
# these chains always rendered: the default 32 symbols, on a grid that says 16 GBd. Nothing
# they assert -- a lattice, a lead-in, a filter's response -- depends on the symbol rate, and
# stating it keeps their numbers and says so instead of tripping `carrier`'s warning.)
_gst = Grid(fs=_fs_store, baud=16e9, n=1 << 16, v_full=0.846)
_base = (Signal(seed=17, grid=_gst).carrier(kind="nrz", n_ui=32, pattern="prbs13")
         .lossy(loss_db=12.0, loss_at_ghz=8.0, causal=True)
         .scope(bw_hz=110e9).digitize(noise_rms=0.0032, enob=10.0).scope(bw_hz=32e9))
_smeared = _base.waveform()
_stored = (Signal(seed=17, grid=_gst).carrier(kind="nrz", n_ui=32, pattern="prbs13")
           .lossy(loss_db=12.0, loss_at_ghz=8.0, causal=True)
           .scope(bw_hz=110e9).digitize(noise_rms=0.0032, enob=10.0).scope(bw_hz=32e9)
           .store(bits=11, full_scale=1.0)).waveform()
_q11 = 2.0 / 2 ** 11
check("without the export step the DSP filter smears the converter lattice away (the defect)",
      _on_lattice(_smeared, _q11) < 0.02 and len(np.unique(_smeared)) > len(_smeared) // 2,
      f"{len(np.unique(_smeared))} distinct values, on-lattice {_on_lattice(_smeared, _q11):.4f}")
check("`store` puts the record back on a lattice, fully (occupancy, not a sparse set)",
      _on_lattice(_stored, _q11) > 0.9999
      and 1500 < len(np.unique(_stored)) < 2600,
      f"{len(np.unique(_stored))} distinct values on a {_q11 * 0.5 * _gst.v_full * 1e6:.0f} uV lattice "
      f"(real captures: 1851-2035 in 8 M samples)")
_pred_floor = INST.quantisation_floor_db_per_hz(_q11, _fs_store)
_meas_floor = _stopband_psd_db_per_hz(_stored, _fs_store)
check("the stored record's stop band sits at its own lattice's q^2/12/(fs/2)",
      abs(_meas_floor - _pred_floor) < 1.0,
      f"predicted {_pred_floor:.2f}, measured {_meas_floor:.2f} dB/Hz "
      f"(unstored: {_stopband_psd_db_per_hz(_smeared, _fs_store):.1f})")
check("the export step is a rounding, not a rewrite: no sample moves by more than half an LSB",
      float(np.max(np.abs(_stored - _smeared))) <= _q11 / 2 + 1e-12,
      f"max move {float(np.max(np.abs(_stored - _smeared))) / _q11:.4f} LSB")
check("`store` is idempotent — storing a stored record changes nothing",
      np.array_equal(_stored, INST.store_record(_stored, bits=11, full_scale=1.0)))
check("`store` clips to the representable code range (a real export has no room beyond it)",
      float(np.max(np.abs(INST.store_record(np.array([-4.0, 4.0]), bits=8, full_scale=1.0))))
      <= 1.0 + 1e-12)
# The vertical is a SETTING, made once per acquisition. Ranged to the record with 5 % of
# headroom -- the middle of the 0.6-10.7 % the three real captures' own code counts imply at
# 11 bits -- the count falls out of the arithmetic rather than being chosen: 2**11/1.05 = 1950.
_ranged = INST.store_record(_smeared, bits=11)             # full_scale=None: range to this record
_n_ranged = len(np.unique(_ranged))
check("a vertical ranged to the record at 11 bits lands on the real captures' code count",
      1851 <= _n_ranged <= 2035,
      f"{_n_ranged} distinct (2**11/1.05 = 1950 predicted; real 1851 / 1880 / 2035)")
check("a ranged vertical clips nothing (real captures show no rail pile-up)",
      float(np.max(np.abs(_ranged))) < 1.05 * float(np.max(np.abs(_smeared))))
check("the ranged pitch is constant across the record (a setting, not per-sample rounding)",
      _on_lattice(_ranged, float(np.min(np.diff(np.unique(_ranged))))) > 0.9999)


# ---------------------------------------------------------------------------------------------
# QUANTISATION AND ENOB ARE TWO MECHANISMS. Quantisation is the converter's real bit depth: a
# uniform lattice, discrete, `q = 2*FS/2**bits`. ENOB is a SINAD figure: noise AND distortion,
# continuous, and the reason ten physical bits behave like about six. Modelling the second as if
# it were the first (rounding to a `2**enob` lattice) gets the noise POWER roughly right and the
# record's structure entirely wrong. Everything below is a constructed case with a known answer.
print()
print("== converter vs ENOB: two mechanisms, checked apart ==")
_FSV = 0.846; _A = _FSV / 2; _FSAMP = 256e9; _NYQ = _FSAMP / 2      # a 10-bit, 256 GSa/s DSO
_gE = Grid(fs=_FSAMP, n=1 << 20)
_NE = _gE.n; _tE = np.arange(_NE); _kE = int(round(2.0e9 * _NE / _FSAMP))
_sine = _A * np.sin(2 * np.pi * _kE * _tE / _NE)                    # full-scale, on an exact bin


def _enob_meas(y, guard=2):
    """ENOB from SINAD by coherent-sampling FFT: (SINAD_dB - 1.76)/6.02, DC and the
    fundamental +/- guard removed. The signal sits on an exact FFT bin, so no window is
    needed and none is used -- a window would leak the carrier into the noise sum."""
    P = np.abs(np.fft.rfft(np.asarray(y, float) - np.mean(y))) ** 2
    k = int(np.argmax(P)); sig = P[max(k - guard, 1):k + guard + 1].sum()
    return float((10.0 * np.log10(sig / (P[1:].sum() - sig)) - 1.76) / 6.02)


# CONSTRUCTED CASE — the ENOB instrument itself, before it is used to judge anything.
for _et in (5.0, 6.5, 9.0, 11.0):
    _n = _rs.normal(0.0, INST.sinad_noise_rms(_et, _A), _NE)
    check(f"ENOB instrument recovers a CONSTRUCTED SINAD of {_et} bits",
          abs(_enob_meas(_sine + _n) - _et) < 0.05, f"{_enob_meas(_sine + _n):.2f}")

# ONE fixed converter floor, sized ONCE from the widest published setting, RENDERED through the
# whole chain at every other setting. A 110 GHz four-channel real-time oscilloscope's published
# data sheet gives these 13 bandwidth/ENOB pairs for its 10-bit converter; a white
# converter-referred floor shaped only by the selected-bandwidth filter has to reproduce all of
# them or the split between quantisation and converter noise is wrong.
_BW_ENOB = [(10, 7.0), (13, 6.8), (16, 6.7), (20, 6.5), (25, 6.2), (32, 5.9), (40, 5.8),
            (50, 5.6), (59, 5.5), (67, 5.4), (80, 5.3), (90, 5.1), (110, 5.0)]
_sig_c = INST.converter_noise_rms(5.0, _A, 110e9, _NYQ, bits=10)    # anchored at 110 GHz -> 5.0
_dev = []
for _B, _pub in _BW_ENOB:
    _y = INST.scope_bandwidth(_sine, _gE, 110e9, kind="bessel")     # analog front end
    _y = _y + np.random.default_rng(11).normal(0.0, _sig_c, _NE)    # the converter's own noise
    _y, _ = INST.clip_adc(_y, _A)
    _y = INST.quantize_adc(_y, bits=10, full_scale=_A)              # the converter's real lattice
    _y = INST.scope_bandwidth(_y, _gE, _B * 1e9, kind="brickwall")  # the DSP filter, AFTER it
    _dev.append(_enob_meas(_y) - _pub)
_dev = np.array(_dev)
check("ONE converter noise floor + a 10-bit lattice renders all 13 published bandwidth/ENOB "
      "settings", np.abs(_dev).max() < 0.35,
      f"{_dev.min():+.2f}..{_dev.max():+.2f} bits, rms {np.sqrt((_dev ** 2).mean()):.3f} "
      f"(floor {_sig_c * 1e3:.2f} mV rms wideband, sized once at 110 GHz)")
_q10 = 2.0 * _A / 2 ** 10
check("the converter's lattice is far BELOW the noise that sets its ENOB (so it is dithered, "
      "not visible)", 8.0 < _sig_c / _q10 < 12.0 and (_q10 ** 2 / 12) / _sig_c ** 2 < 0.002,
      f"{_sig_c / _q10:.2f} LSB rms; the lattice is {(_q10 ** 2 / 12) / _sig_c ** 2 * 100:.3f} % "
      f"of the noise power, and alone would be worth "
      f"{(10 * np.log10((_A ** 2 / 2) / (_q10 ** 2 / 12)) - 1.76) / 6.02:.2f} bits")

# THE REFUTATION. The legacy knob rounds to a 2**enob lattice. A real DSO filters AFTER its
# converter, and that filter throws away most of a lattice's noise power -- so the ENOB-as-lattice
# model does not even reproduce the number it was handed.
#
# MEASURED AT SIX TONES, NOT ONE, AND THAT IS THE POINT. This check used to read one 2.0 GHz
# tone and demand >1.5 bits of overshoot. A coarse lattice on a pure tone produces DETERMINISTIC
# HARMONICS, not white error, so the reading depends on exactly where the tone's peaks land
# between codes: on the pre-fix zero-phase front end the six tones below read
# +1.91 / +1.05 / +0.68 / +1.08 / +0.76 / +0.81 bits. The 1.5 threshold was passing on the ONE
# tone that happened to read 1.91 and would have failed on four of the other five. It is now the
# WORST of the six, and the contrast that carries the argument is measured on the same tones:
# the honest two-mechanism path (a 10-bit lattice plus `converter_noise_rms`) lands within
# 0.046 bits of the published figure at every one of them, while the legacy lattice overshoots
# by 0.77 to 1.05.
_ENOB_TONES = (2.0e9, 2.1e9, 3.0e9, 5.0e9, 7.0e9, 11.0e9)


def _legacy_lattice_overshoot(f_hz):
    _k = int(round(f_hz * _NE / _FSAMP))
    _s = _A * np.sin(2 * np.pi * _k * np.arange(_NE) / _NE)
    _yl = INST.quantize_adc(INST.scope_bandwidth(_s, _gE, 110e9, kind="bessel"), enob=5.9,
                            full_scale=_A)
    return _enob_meas(INST.scope_bandwidth(_yl, _gE, 32e9, kind="brickwall")) - 5.9, _yl


_over = np.array([_legacy_lattice_overshoot(_f)[0] for _f in _ENOB_TONES])
_yl = _legacy_lattice_overshoot(2.0e9)[1]
check("ENOB-as-a-lattice does NOT survive the DSP filter a real DSO has after its converter",
      _over.min() > 0.6,
      f"asked for 5.9, reads {_over.min() + 5.9:.2f}..{_over.max() + 5.9:.2f} after the 32 GHz "
      f"filter at {len(_ENOB_TONES)} tones ({_enob_meas(_yl):.2f} before it at 2 GHz); its "
      f"lattice is {2 * _A / 2 ** 5.9 * 1e6:.0f} uV against the converter's real "
      f"{_q10 * 1e6:.0f} uV")
try:
    INST.converter_noise_rms(11.0, _A, 110e9, _NYQ, bits=10)
    _guard = False
except ValueError:
    _guard = True
check("converter_noise_rms REFUSES an ENOB a converter's depth cannot reach", _guard)

from wfmsynth.compose import _lead_extent
print()
print("== the record's head: a rendered-and-discarded lead-in (U-16's turn-on, removed) ==")
# A linear convolution (U-16) means the samples before index 0 are a quiescent line, so a record
# that begins mid-pattern begins with a TURN-ON EDGE no running link has, every stage with memory
# answers that edge, and a still-circular stage rings on it. `Signal(lead_in=...)` renders extra
# samples of the same pattern before AND after the record, runs the chain on that longer record and
# delivers only the middle. These checks are the ground truth for that mechanism, and they are built
# from an EXACTLY PERIODIC record, because a periodic record is the one case where the answer for a
# link that has been running forever can be written down: it is the steady-state response, and
# `np.convolve` on the tiled source gives it with nothing of ours in the computation.
_LP, _LSPB = 508, 16                                   # 508 symbols x 16 samples/UI, exactly periodic
_LN = _LP * _LSPB
_LSYM = list(np.where(np.random.default_rng(11).random(_LP) > 0.5, 1.0, -1.0))
_LG = Grid(fs=64e9, baud=64e9 / _LSPB, n=_LN)
_LTAPS, _LPRE = [-0.10, 1.00, -0.25, 0.05], 1          # a hand-written FIR: 4 taps, 1 pre-cursor


def _lsrc(n_rec, syms, **kw):
    return Signal(seed=3, grid=Grid(fs=_LG.fs, baud=_LG.baud, n=n_rec), **kw).symbols(syms)


# One period of the SHAPED source, taken from the interior of a three-period render so no filtfilt
# edge padding is inside it. Tiling this is the source a link that never started would carry.
_lper = _lsrc(3 * _LN, _LSYM * 3).waveform()[_LN:2 * _LN]
_lh = np.zeros((len(_LTAPS) - 1) * _LSPB + 1)
_lh[np.arange(len(_LTAPS)) * _LSPB] = _LTAPS
_lconv = np.convolve(np.tile(_lper, 7), _lh)           # numpy's convolution, not ours
_LTRUTH = _lconv[3 * _LN + _LPRE * _LSPB:4 * _LN + _LPRE * _LSPB]
check("the CONSTRUCTED steady state is stationary (so it is the answer for a link that never "
      "started)",
      np.abs(_LTRUTH - _lconv[4 * _LN + _LPRE * _LSPB:5 * _LN + _LPRE * _LSPB]).max() < 1e-15,
      f"{np.abs(_LTRUTH - _lconv[4 * _LN + _LPRE * _LSPB:5 * _LN + _LPRE * _LSPB]).max():.2e}")


def _lffe(**kw):
    s = _lsrc(_LN, _LSYM, **kw)
    s.tx_ffe(_LTAPS, pre=_LPRE)
    return s.waveform()


def _lerr(y, tol=1e-9):
    """(max |error| against the constructed steady state, where the wrong samples are)."""
    e = np.abs(np.asarray(y, float) - _LTRUTH)
    bad = np.nonzero(e > tol)[0]
    head = int(np.count_nonzero(bad < _LN // 2))
    where = ("none" if bad.size == 0 else
             f"{bad.size} ({head} in the head, {bad.size - head} in the tail; "
             f"first {bad.min()}, last {_LN - 1 - bad.max()} from the end)")
    return float(e.max()), where


_e5, _w5 = _lerr(_lffe(lead_in=5 * _LSPB))
check("a 5-UI lead-in RECOVERS np.convolve's steady-state answer for the whole record",
      _e5 < 1e-12, f"max |err| {_e5:.2e}, corrupted: {_w5}")
_e0, _w0 = _lerr(_lffe())
check("  ... and WITHOUT it the record is wrong, at both ends (negative control)",
      _e0 > 0.5, f"max |err| {_e0:.3f} of a +/-1 signal, corrupted: {_w0}")
# GATE OBSERVED FAILING: a guard one UI short of the FIR's own post-cursor reach.
_e1, _w1 = _lerr(_lffe(lead_in=1 * _LSPB))
check("  ... a guard 1 UI SHORT of the FIR's 2-UI post-cursor reach is still wrong (the length is "
      "load-bearing, not decorative)", 1e-3 < _e1 < 0.5, f"max |err| {_e1:.3f}, corrupted: {_w1}")
_e2, _w2 = _lerr(_lffe(lead_in=2 * _LSPB))
check("  ... and 2 UI, its exact reach, leaves only the tail (which is what lead_out is for)",
      _e2 < 1e-5, f"max |err| {_e2:.2e}, corrupted: {_w2}")
_et, _wt = _lerr(_lffe(lead_in=5 * _LSPB, lead_out=0))
check("the LEAD-OUT is load-bearing too: lead_out=0 leaves the record's TAIL wrong, because a "
      "pre-cursor tap (and zero-phase edge shaping) reach FORWARD into samples that are not there",
      _et > 0.5, f"max |err| {_et:.3f}, corrupted: {_wt}")

# THE SOURCE'S OWN EDGES, before any channel at all. `physics._shape_edges` is `sosfiltfilt`, whose
# padding invents the samples beyond the record: the record's last edge is simply MISSING.
_s0, _s1 = _lsrc(_LN, _LSYM).waveform(), _lsrc(_LN, _LSYM, lead_in=5 * _LSPB).waveform()
_d0, _d1 = np.abs(_s0 - _lper), np.abs(_s1 - _lper)
check("the CARRIER ITSELF is wrong at both edges without a lead-in -- its last edge is missing, "
      "not slow", _d0.max() > 0.5 and _d1.max() < 1e-12,
      f"no lead-in: max {_d0.max():.3f} in {np.count_nonzero(_d0 > 1e-9)} samples; last four "
      f"samples read {np.round(_s0[-4:], 3).tolist()} where the running link's are "
      f"{np.round(_lper[-4:], 3).tolist()}. With a lead-in: {_d1.max():.1e}")

# THE SIZER, against a CLOSED FORM. A lumped reflection's impulse response is a delta plus echoes at
# 2*td*k with weight gamma_l*(gamma_s*gamma_l)**k, so the extent at the -100 dB threshold is exactly
# 2*td*k_max+1 for the largest k whose weight is still >= 1e-5. Nothing is approximated here.
_ltd, _lgs, _lgl, _lnb = 72, 0.25, 0.2, 4
_lops = [dict(op="symbols", symbols=_LSYM),
         dict(op="reflect", td_samples=_ltd, gamma_s=_lgs, gamma_l=_lgl, n_bounce=_lnb)]
_lkmax = max(k for k in range(_lnb + 1) if _lgl * (_lgs * _lgl) ** k >= P.RESPONSE_REL)
_lmeas = _lead_extent(_lops, _LG, 3, _LN)
check("the lead-in SIZER recovers a CONSTRUCTED impulse-response extent exactly",
      _lmeas == 2 * _ltd * _lkmax + 1,
      f"measured {_lmeas}, closed form {2 * _ltd * _lkmax + 1} (last echo above "
      f"{P.RESPONSE_REL:g} is k={_lkmax} at weight {_lgl * (_lgs * _lgl) ** _lkmax:.1e}; k="
      f"{_lkmax + 1} would be {_lgl * (_lgs * _lgl) ** (_lkmax + 1):.1e})")


# GATE OBSERVED FAILING: the probe's impulse MUST sit at index 0. Placed anywhere else, a causal
# response runs from the impulse to the end of the buffer, the longest quiet run is the part BEFORE
# it, and `response_extent`'s "does it fit in half the probe" test passes at every probe length
# while reporting half of it. This is the estimator failure mode that would have under-sized every
# lead-in by 10x, and it is checked rather than commented.
def _lext_at(pos, n0):
    def make_H(m):
        m = int(m)
        d = np.zeros(m)
        d[int(pos * m)] = 1.0
        return np.fft.rfft(P.lossy_channel(d, grid=Grid(fs=_FSAMP, baud=16e9, n=m),
                                          loss_db=12.0, loss_at_ghz=8.0, causal=True))
    return P.response_extent(make_H, rel=P.RESPONSE_REL, n0=n0, probe_max=1 << 22, warn=False)


_lz = [_lext_at(0.0, m) for m in (4096, 8192, 16384)]
_lm = [_lext_at(0.5, m) for m in (4096, 8192, 16384)]
check("the sizer's probe puts its impulse at index 0, and a mid-buffer impulse is REJECTED by "
      "this check: it reports half the probe at every probe length",
      len(set(_lz)) == 1 and _lm == [m // 2 for m in (4096, 8192, 16384)] and _lm[-1] < _lz[0],
      f"impulse at 0: {_lz} (stable); impulse at the middle: {_lm} (= half the probe, every time)")

# THE FULL CHAIN, on the shipped recipe, with NO TRIM ANYWHERE. This is what replaced a 1024-sample
# edge trim in this file: the trim measured the interior of a spoiled record, the lead-in delivers a
# record that is not spoiled.
_lchain = dict(seed=17, grid=Grid(fs=_FSAMP, baud=16e9, n=1 << 20, v_full=0.8))


def _lfull(**kw):
    return (Signal(**_lchain, **kw)
            .carrier(kind="nrz", n_ui=32, pattern="prbs13").lossy(loss_db=12.0, loss_at_ghz=8.0, causal=True)
            .scope(bw_hz=110e9).digitize(noise_rms=_sig_c, bits=10, full_scale=_A))


_lsig = _lfull(lead_in=True).scope(bw_hz=32e9, kind="brickwall")
_lplan = _lsig.lead_plan()
_lwith = _lsig.waveform()
_lwout = _lfull().scope(bw_hz=32e9, kind="brickwall").waveform()
_lpk = (np.abs(_lwout).max() / np.abs(_lwout[1024:-1024]).max() - 1) * 100
check("a lead-in removes the ringing the STILL-CIRCULAR brickwall puts on the record's edges",
      abs(np.abs(_lwith).max() / np.abs(_lwith[1024:-1024]).max() - 1) < 1e-9 and _lpk > 5.0,
      f"with: peak {np.abs(_lwith).max():.4f} = interior peak; without: {np.abs(_lwout).max():.4f}, "
      f"{_lpk:+.2f} % above its own interior. Plan: {_lplan.summary()}")
for _dth, _want, _lbl in ((0.0, 0.0, "the stored record's floor is its own lattice's q^2/12, on the "
                                     "WHOLE record with no trim"),
                          (1 / np.sqrt(12), 3.01, "  ... and dither still moves it by 3 dB")):
    _lr = INST.store_record(_lwith, bits=11, dither_lsb=_dth, rng=np.random.default_rng(5))
    _lu = np.unique(_lr)
    _lq = float(np.min(np.diff(_lu)))
    _ld = _stopband_psd_db_per_hz(_lr, _FSAMP) - INST.quantisation_floor_db_per_hz(_lq, _FSAMP)
    check(_lbl, abs(_ld - _want) < 0.4 and 1851 <= len(_lu) <= 2035,
          f"{_ld:+.2f} dB above q^2/12/(fs/2), {len(_lu)} distinct codes (real: 1851/1880/2035), "
          f"on-lattice {_on_lattice(_lr, _lq):.4f}")
_lru = np.unique(INST.store_record(_lwout, bits=11, rng=np.random.default_rng(5)))
check("  ... and the same recipe WITHOUT a lead-in is outside the real captures' code band "
      "(negative control)", len(_lru) < 1851,
      f"{len(_lru)} distinct codes against 1851-2035, because the edge ringing takes "
      f"{(np.abs(_lwout).max() / np.abs(_lwith).max() - 1) * 100:.1f} % of the vertical range")

# THE VERTICAL MUST BE RANGED TO THE DELIVERED WINDOW, NOT TO THE GUARD. The guard is where the
# ringing lives; a `store` ranged to it would spend the record's codes on samples nobody receives,
# which is the same defect wearing a different hat. `rendered_lead()` hands over the whole render so
# this can be checked on the artifact rather than on the intention.
_lxe, _lpl = _lfull(lead_in=True).scope(bw_hz=32e9, kind="brickwall").rendered_lead()
_lwin_codes = len(np.unique(INST.store_record(_lxe[_lpl.window], bits=11,
                                             rng=np.random.default_rng(5))))
_lgrd_codes = len(np.unique(INST.store_record(_lxe, bits=11,
                                              rng=np.random.default_rng(5))[_lpl.window]))
check("the export ranges to the WINDOW, and ranging it to the guard instead is observed to lose "
      "the codes again", 1851 <= _lwin_codes <= 2035 and _lwin_codes - _lgrd_codes > 50,
      f"window-ranged {_lwin_codes} codes, guard-ranged {_lgrd_codes} "
      f"({_lwin_codes - _lgrd_codes} lost); the guard peaks at "
      f"{np.abs(_lxe).max():.4f} against the window's {np.abs(_lxe[_lpl.window]).max():.4f}")

# BACKLOG #54: `instrument.scope`'s frequency-domain kinds are STILL circular. The lead-in does not
# fix them -- it moves their wrap into the samples that are thrown away. Measured, not asserted: the
# same record through the same brickwall applied circularly and applied as a linear convolution.
_LSB11 = 2 * 1.05 / 2 ** 11                            # one code of an 11-bit record ranged to +/-1
_lbw, _lwn = 32e9, (32e9 / _NYQ) / 2.0


def _lbrick(x, nfft=None):
    n = len(x)
    xp = x if nfft is None else np.concatenate([x, np.zeros(int(nfft) - n)])
    X = np.fft.rfft(xp)
    X[np.fft.rfftfreq(len(xp)) > _lwn] = 0.0
    return np.fft.irfft(X, len(xp))[:n]


_lpre_e, _lpre_pl = _lfull(lead_in=True).rendered_lead()
_lpre_0 = _lfull().waveform()
_lguard = P.response_extent(lambda m: np.fft.rfft(_lbrick(np.eye(1, int(m), 0).ravel())),
                            rel=P.RESPONSE_REL, probe_max=1 << 22, warn=False)
_lw_in = np.abs(_lbrick(_lpre_e) - _lbrick(_lpre_e, P.linear_fft_length(len(_lpre_e), _lguard)))
_lw_no = np.abs(_lbrick(_lpre_0) - _lbrick(_lpre_0, P.linear_fft_length(len(_lpre_0), _lguard)))
check("#54: a lead-in makes the STILL-CIRCULAR brickwall's wrap harmless in the DELIVERED window "
      "(it does not fix the stage -- the wrap is still there, in the guard)",
      _lw_in[_lpre_pl.window].max() / _LSB11 < 0.1 and _lw_no.max() / _LSB11 > 10,
      f"circular vs linear, inside the delivered record: {_lw_in[_lpre_pl.window].max() / _LSB11:.3f} "
      f"LSB with a lead-in against {_lw_no.max() / _LSB11:.1f} LSB without; the wrap in the guard "
      f"itself is still {_lw_in.max() / _LSB11:.0f} LSB, and the brickwall's own sinc is "
      f"{_lguard} samples long at {P.RESPONSE_REL:g}")

# THE RECIPE IS THE GROUND TRUTH, so it has to carry the lead-in.
_lrs = _lsrc(_LN, _LSYM, lead_in=True)
_lrs.lossy(loss_db=8.0, loss_at_ghz=8.0, causal=True)
_lrr = _lrs.recipe()
check("a lead-in round-trips through the recipe bit-for-bit, and a Signal without one adds no "
      "recipe key (so every existing recipe hashes exactly as before)",
      np.array_equal(Signal.from_recipe(_lrr).waveform(), _lrs.waveform())
      and _lrr.get("lead_in") is True and "lead_in" not in _lsrc(_LN, _LSYM).recipe(),
      f"recipe lead_in={_lrr.get('lead_in')!r}, plan: {_lrs.lead_plan().summary()}")

# REFUSALS OBSERVED FIRING. A knob that is a fraction of the record, an op that changes the record's
# length, and an op that places something at an absolute position in it all mean something different
# once the record is longer. Refusing is the honest answer; silently moving them is not.
_lref = []
for _lname, _lmk in (("reflect(td_frac=)", lambda: _lsrc(_LN, _LSYM, lead_in=True)
                      .reflect(td_frac=0.1, gamma_l=0.2)),
                     ("ac_couple(fc_frac=)", lambda: _lsrc(_LN, _LSYM, lead_in=True)
                      .ac_couple(fc_frac=1e-4)),
                     ("digitize(n_out=)", lambda: _lsrc(_LN, _LSYM, lead_in=True)
                      .digitize(n_out=_LN // 2)),
                     ("acquire", lambda: _lsrc(_LN, _LSYM, lead_in=True)
                      .acquire(dict(sample_rate_hz=1e9, record_length=1024))),
                     ("events", lambda: _lsrc(_LN, _LSYM, lead_in=True).events("runt", count=2))):
    try:
        _lmk().waveform()
        _lref.append(f"{_lname}: NOT REFUSED")
    except ValueError:
        pass
check("a lead-in REFUSES every op whose meaning it would change (5 of 5 observed refusing)",
      not _lref, "; ".join(_lref) or "reflect(td_frac), ac_couple(fc_frac), digitize(n_out), "
                                     "acquire, events")
from wfmsynth.compose import (_LEAD_LTI, _LEAD_SKIP, _LEAD_REJECT, _LEAD_ANALYTIC,
                              _EXEC as _LEXEC)
_lunc = sorted(set(_LEXEC) - (_LEAD_LTI | _LEAD_SKIP | set(_LEAD_REJECT) | set(_LEAD_ANALYTIC)))
check("every composable op is CLASSIFIED for the lead-in (has an impulse response / has none / "
      "has one that is known in closed form / cannot be guarded), so a new op cannot default into "
      "the safe-looking pile",
      not _lunc, f"{len(_LEXEC)} ops: {len(_LEAD_LTI)} sized, {len(_LEAD_SKIP)} skipped, "
                 f"{len(_LEAD_ANALYTIC)} stated, {len(_LEAD_REJECT)} refused"
                 + (f"; UNCLASSIFIED {_lunc}" if _lunc else ""))
# A STATED extent has to be a real quantity, not a placeholder. Two things make it one: it scales
# with the physics it came from, and at the capacitance where such a bus FAILS it exceeds a symbol --
# which is the same statement as the failure, so the lead-in and the failure mode agree.
#
# Note what this check does NOT claim. At the specified limit (885 ohm, 400 pF) 5 tau is 0.7 UI at
# 400 kbit/s -- SHORTER than one symbol, and that is the compliant condition, not a shortfall: the
# specification caps the rise at 300 ns inside a 2.5 us bit precisely so the edge finishes well
# within the bit. An earlier version of this check asserted 5 tau > 1 UI and failed here; the check
# was wrong, not the extent.
from wfmsynth import Grid as _LGrid
_lg = _LGrid(fs=100e6, baud=400e3, n=1 << 14, v_full=3.3)
def _lstated(r, c):
    return _LEAD_ANALYTIC["open_drain"]({"r_pullup_ohm": r, "c_bus_f": c}, _lg)
_l1 = _lstated(885.0, 400e-12)
# ---- a shared net: several drivers, switched terminations, one solve (GitHub #72) ----
# The claim is EXACTNESS, so these are equalities against closed forms and against the op this
# generalises -- not agreement to a tolerance someone picked.
from wfmsynth import net as _NET

_nrp, _ncb, _nvdd, _nrs, _nfs = 885.0, 400e-12, 3.3, 20.0, 1e9
_nspec = {"nodes": ["bus"], "rails": {"vdd": _nvdd},
          "elements": [{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": _nrp},
                       {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": _ncb}],
          "drivers": [{"id": "d0", "node": "bus", "volts": 0.0, "ohm": _nrs, "state": "hiz"}]}

_nn = 4000
_nt = np.arange(1, _nn + 1) / _nfs
_nrc = _NET.solve(_nspec, fs=_nfs, n=_nn, probes="bus", v0={"bus": 0.0})
_nwant = _nvdd * (1.0 - np.exp(-_nt / (_nrp * _ncb)))
check("a net's RC charge IS V(1 - exp(-t/RC)), not an integration of it",
      float(np.abs(_nrc - _nwant).max()) < 1e-12,
      f"max |solved - closed form| = {float(np.abs(_nrc - _nwant).max()):.2e} V over "
      f"{_nn} samples; tau = {_nrp * _ncb:.4g} s")

_nx = np.asarray(P.nrz(n_ui=_nn // 50, n=_nn, seed=3, causal=True), float)
_nsink = _nx >= 0.5 * (_nx.min() + _nx.max())
_nref = P.open_drain_line(_nsink, _nfs, _nrp, _ncb, v_dd=_nvdd, r_sink_ohm=_nrs)
_ngot = _NET.solve(_nspec, fs=_nfs, n=_nn, probes="bus",
                   schedule=_NET.schedule_from_waveform(_nx, "d0"))
check("the one-node net IS open_drain_line -- the general solver reproduces the special case",
      float(np.abs(_ngot - _nref).max()) < 1e-13,
      f"max |net - open_drain_line| = {float(np.abs(_ngot - _nref).max()):.2e} V across "
      f"{int(np.count_nonzero(np.diff(_nsink.astype(int)))) + 1} driver state intervals")

# a weaker driver holding a level against a stronger one -- the thing a 2-port chain cannot say
_nspec2 = {"nodes": ["bus"], "rails": {"vdd": _nvdd},
           "elements": [{"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": _ncb}],
           "drivers": [{"id": "strong", "node": "bus", "volts": 0.0, "ohm": 20.0},
                       {"id": "weak", "node": "bus", "volts": _nvdd, "ohm": 200.0}]}
_ncont = _NET.solve(_nspec2, fs=_nfs, n=2000, probes="bus")
_nwant2 = _nvdd * (1.0 / 200.0) / (1.0 / 20.0 + 1.0 / 200.0)
check("two drivers contending settle to the conductance-weighted average, not to either one",
      abs(float(_ncont[-1]) - _nwant2) < 1e-12,
      f"strong 20 ohm at 0 V against weak 200 ohm at {_nvdd} V -> {float(_ncont[-1]):.6f} V, "
      f"closed form {_nwant2:.6f} V (the strong driver wins but does not win outright)")

# ---- the scenario layer on top of it: holds + agents -> that schedule (GitHub #73) ----
# This layer adds no physics, so the claim is that it is a faithful TRANSCRIPTION: the same
# events a careful person would have written, and therefore the same samples.
from wfmsynth import scenario as _SC

_sh = [_SC.Hold("a", 1.0e-6, 2.0e-6, bit=0), _SC.Hold("a", 2.0e-6, 3.0e-6, bit=1)]
_sa = {"a": _SC.Agent(during={"d0": "driving"}, after={"d0": "hiz"}, lead=5e-8, lag=1e-7)}
_sev = _SC.schedule(_sh, _sa)
_sgot = _NET.solve(_nspec, fs=_nfs, n=_nn, probes="bus", schedule=_sev)
_shand = _NET.solve(_nspec, fs=_nfs, n=_nn, probes="bus", schedule=[
    {"t_s": 0.0, "id": "d0", "state": "hiz"},
    {"t_s": 0.95e-6, "id": "d0", "state": "driving"},
    {"t_s": 3.1e-6, "id": "d0", "state": "hiz"}])
check("a compiled scenario IS the hand-written schedule -- bit for bit, not to a tolerance",
      np.array_equal(_sgot, _shand) and len(_sev) == 3,
      f"two abutting holds merged to one interval, widened by lead 50 ns and lag 100 ns -> "
      f"{len(_sev)} events; max |compiled - hand-written| = "
      f"{float(np.abs(_sgot - _shand).max()):.2e} V")

# Two events at one instant are ONE configuration change. Applied in sequence instead, a
# hand-over passes through a moment with nobody driving, and that is worth real volts.
_shandover = {"nodes": ["bus"], "rails": {"vdd": _nvdd},
              "elements": [{"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": _ncb}],
              "drivers": [{"id": "p", "node": "bus", "volts": 0.0, "ohm": _nrs},
                          {"id": "q", "node": "bus", "volts": 0.0, "ohm": _nrs,
                           "state": "hiz"}]}
_soff = {"t_s": 1e-6, "id": "p", "state": "hiz"}
_son = {"t_s": 1e-6, "id": "q", "state": "driving"}
_sfwd = _NET.solve(_shandover, fs=_nfs, n=3000, probes="bus", schedule=[_soff, _son])
_srev = _NET.solve(_shandover, fs=_nfs, n=3000, probes="bus", schedule=[_son, _soff])
check("events sharing a sample are applied as one step, so a hand-over has no order and no seam",
      np.array_equal(_sfwd, _srev) and float(np.ptp(_sfwd)) < 1e-12,
      f"both listings identical; total excursion across the hand-over "
      f"{float(np.ptp(_sfwd)):.2e} V (a sequenced application would leave the node floating "
      f"for one step and ring)")

# A released driver is REMOVED from the stamp, so its parked level cannot pull the node.
_spark = {"nodes": ["bus"], "rails": {"vdd": _nvdd},
          "elements": [{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": _nrp},
                       {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": _ncb}],
          "drivers": [{"id": "d0", "node": "bus", "volts": -7.5, "ohm": _nrs, "state": "hiz"}]}
check("a hi-Z driver's level is absent from the solve, not merely small",
      np.array_equal(_NET.solve(_spark, fs=_nfs, n=1000, probes="bus"),
                     _NET.solve(_nspec, fs=_nfs, n=1000, probes="bus")),
      "a driver parked at -7.5 V while released gives the same samples as one parked at 0 V")

# ---- drivers whose output follows their own chain (GitHub #72, follow-on) ----
# A driver's Thevenin level may be a WAVEFORM. The claim is still exactness, so: the reduction
# is bit-for-bit, the one-driver case is a closed form, and a released driver contributes
# nothing whatever its own chain is doing -- which is the baseline-drift bug this answered.
_wspec = {"nodes": ["bus"], "rails": {"vdd": _nvdd},
          "elements": [{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": _nrp},
                       {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": _ncb}],
          "drivers": [{"id": "d0", "node": "bus", "ohm": _nrs, "state": "driving"}]}
_wdecl = {**_wspec, "drivers": [{**_wspec["drivers"][0], "volts": 0.4}]}
check("a driver source that never moves IS a declared level -- bit for bit, not to a tolerance",
      np.array_equal(_NET.solve(_wdecl, fs=_nfs, n=2000, probes="bus"),
                     _NET.solve(_wspec, fs=_nfs, n=2000, probes="bus",
                                sources={"d0": np.full(2000, 0.4)})),
      "a constant source array is routed back to the closed form, so an idle interval cannot "
      "pick up a different last bit from the tier that has no sources at all")

_wn, _wrd = 2000, 90.0
_wt = np.arange(_wn) / _nfs
_wsrc = 1.25 + np.sin(2 * np.pi * 3e6 * _wt) + 0.3 * np.sin(2 * np.pi * 11e6 * _wt)
_wonly = {"nodes": ["bus"],
          "elements": [{"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": _ncb}],
          "drivers": [{"id": "d0", "node": "bus", "ohm": _wrd, "state": "driving"}]}
_wgot = _NET.solve(_wonly, fs=_nfs, n=_wn, probes="bus", sources={"d0": _wsrc})
_walpha, _wv = np.exp(-1.0 / (_nfs * _wrd * _ncb)), _wsrc[0]
_wref = np.empty(_wn)
for _wk in range(_wn):                     # exponential stepping -- open_drain_line's method
    _wv = _wsrc[_wk] + (_wv - _wsrc[_wk]) * _walpha
    _wref[_wk] = _wv
check("one driver with its own chain IS that chain through its own R and C",
      float(np.abs(_wgot - _wref).max()) < 1e-14 * float(np.ptp(_wref)),
      f"max |net - exponential stepping| = {float(np.abs(_wgot - _wref).max()):.2e} V over a "
      f"{float(np.ptp(_wref)):.3f} V swing; tau = {_wrd * _ncb:.4g} s")

_whold = [{"t_s": 0.0, "id": "d0", "state": "driving"},
          {"t_s": 1e-6, "id": "d0", "state": "hiz"}]
_wk1 = _NET.sample_at(1e-6, _nfs)
_wloud = _wsrc.copy()
_wloud[_wk1:] = -37.0 + 9.0 * np.sin(2 * np.pi * 7e6 * _wt[_wk1:])   # absurd, while released
check("a released driver contributes nothing, whatever its own chain is then doing",
      np.array_equal(_NET.solve(_wspec, fs=_nfs, n=_wn, probes="bus", schedule=_whold,
                                sources={"d0": _wsrc}),
                     _NET.solve(_wspec, fs=_nfs, n=_wn, probes="bus", schedule=_whold,
                                sources={"d0": _wloud})),
      "the level a driver was holding when it let go is ABSENT from the stamp, so it cannot be "
      "carried into the next owner's baseline -- the drift this feature was asked for")

# ---- inductance in the shared net, so that a hand-over rings (GitHub #74) ----
# An R-C net can only SETTLE. Overshoot and ringing need an inductance, and they are what a
# scope shows at a driver turning on or letting go. The solve is a different decomposition from
# the R-C tier's (a zero-order-hold pair back-substituted through a Schur form, because an
# inductor costs the symmetry the eigensolver needs), so exactness is re-established here rather
# than inherited -- against the textbook series-RLC step response, in all three damping regimes.
_iL, _iC, _iVS = 100e-9, 100e-12, 1.0
_ifs, _in = 2e10, 4000
_it = np.arange(1, _in + 1) / _ifs


def _irlc(ohm):
    spec = {"nodes": ["bus"], "rails": {"vdd": _iVS},
            "elements": [{"id": "lp", "kind": "l", "a": "vdd", "b": "bus",
                          "henry": _iL, "ohm": ohm},
                         {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": _iC}]}
    return _NET.solve(spec, fs=_ifs, n=_in, probes="bus", v0={"bus": 0.0, "i(lp)": 0.0})


def _ianalytic(ohm):
    al, w0 = ohm / (2.0 * _iL), 1.0 / np.sqrt(_iL * _iC)
    if al < w0 * (1 - 1e-12):
        wd = np.sqrt(w0 ** 2 - al ** 2)
        return _iVS * (1 - np.exp(-al * _it) * (np.cos(wd * _it) + al / wd * np.sin(wd * _it)))
    if al > w0 * (1 + 1e-12):
        wd = np.sqrt(al ** 2 - w0 ** 2)
        return _iVS * (1 - np.exp(-al * _it) * (np.cosh(wd * _it) + al / wd * np.sinh(wd * _it)))
    return _iVS * (1 - np.exp(-al * _it) * (1 + al * _it))


_icrit = 2.0 * np.sqrt(_iL / _iC)
for _iohm, _ilbl in ((10.0, "underdamped"), (_icrit, "critically damped"),
                     (4 * np.sqrt(_iL / _iC), "overdamped")):
    _ierr = float(np.abs(_irlc(_iohm) - _ianalytic(_iohm)).max())
    check(f"a series RLC IS the analytic step response, {_ilbl}",
          _ierr < 1e-12,
          f"max |net - closed form| = {_ierr:.2e} V on a {_iVS:.1f} V step at "
          f"R = {_iohm:.4g} ohm (R_crit = {_icrit:.4g})")

_iv = _irlc(10.0)
_ia, _iw0 = 10.0 / (2.0 * _iL), 1.0 / np.sqrt(_iL * _iC)
_iwd = np.sqrt(_iw0 ** 2 - _ia ** 2)
_ios, _iwant = float(_iv.max() - _iVS), _iVS * float(np.exp(-_ia * np.pi / _iwd))
check("and the three numbers a measurement reports off it are the circuit's: ring frequency, "
      "damping, overshoot",
      abs(_ios / _iwant - 1.0) < 1e-4,
      f"first peak {_ios * 1e3:.2f} mV over the level against exp(-alpha pi / wd) = "
      f"{_iwant * 1e3:.2f} mV; ring at {_iwd / (2 * np.pi) / 1e6:.3f} MHz, "
      f"tau = {1 / _ia * 1e9:.1f} ns")
check("  ... and critical damping really is the fastest settle without overshoot, which is why "
      "the defective case is one somebody designs to and not a corner",
      _irlc(_icrit * 0.98).max() > _iVS and _irlc(_icrit).max() <= _iVS + 1e-12
      and int(np.flatnonzero(_irlc(_icrit) > 0.99)[0])
      < int(np.flatnonzero(_irlc(_icrit * 1.02) > 0.99)[0]),
      f"R = {_icrit:.4g} ohm reaches 99 % in "
      f"{int(np.flatnonzero(_irlc(_icrit) > 0.99)[0]) / _ifs * 1e9:.2f} ns with no overshoot; "
      f"2 % more resistance takes "
      f"{int(np.flatnonzero(_irlc(_icrit * 1.02) > 0.99)[0]) / _ifs * 1e9:.2f} ns")

# A SWITCH while current flows. The acceptance criterion is continuity, and the sharp statement
# of it is an identity: split the record at the switch, re-enter the solve with the voltages AND
# the inductor current it handed over, and the two halves must be the one record bit for bit.
_ibrd = {"nodes": ["pad", "bus"], "rails": {"vdd": 3.3},
         "elements": [{"id": "lp", "kind": "l", "a": "pad", "b": "bus", "henry": 4e-9,
                       "ohm": 0.2},
                      {"id": "cp", "kind": "c", "a": "pad", "b": "gnd", "farad": 4e-12},
                      {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 20e-12},
                      {"id": "rt", "kind": "r", "a": "bus", "b": "vdd", "ohm": 150.0}],
         "drivers": [{"id": "d0", "node": "pad", "volts": 0.0, "ohm": 5.0, "state": "driving"}]}
_ifs2, _in2, _iksw = 4e10, 4000, 2000
_ipr = ["pad", "bus", "i(lp)"]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)     # the 4 pF pad settles inside a sample
    _iwhole = _NET.solve(_ibrd, fs=_ifs2, n=_in2, probes=_ipr,
                         schedule=[{"t_s": _iksw / _ifs2, "id": "d0", "state": "hiz"}])
    _ifirst = _NET.solve(_ibrd, fs=_ifs2, n=_iksw, probes=_ipr)
    _irel = {**_ibrd, "drivers": [{**_ibrd["drivers"][0], "state": "hiz"}]}
    _isecond = _NET.solve(_irel, fs=_ifs2, n=_in2 - _iksw, probes=_ipr,
                          v0={nm: _ifirst[nm][-1] for nm in _ipr})
check("an inductor current is continuous across a switch -- as an IDENTITY, not a tolerance: the "
      "state handed over at the switch reproduces the record",
      all(np.array_equal(_iwhole[nm][_iksw:], _isecond[nm]) for nm in _ipr)
      and all(np.array_equal(_iwhole[nm][:_iksw], _ifirst[nm]) for nm in _ipr),
      f"the driver lets go sinking {abs(_ifirst['i(lp)'][-1]) * 1e3:.1f} mA, and the pad then "
      f"overshoots to {_iwhole['pad'][_iksw:].max():.3f} V against a {3.3:.1f} V rail")

# AND A LINE, as a lumped ladder -- the one approximation in this module, so its own error is
# measured against what it is approximating rather than asserted.
_ild, _iltr, _ilz = 1e-9, 500e-12, 50.0
_ilfs = 4e11
_iln = int(round((10 * _iltr + 3 * _ild) * _ilfs))
_ilu = np.clip((np.arange(_iln) / _ilfs - 5 * _iltr) / (_iltr / 0.8) + 0.5, 0.0, 1.0)
_ilvs = np.where((_ilu > 0) & (_ilu < 1), 0.5 - 0.5 * np.cos(np.pi * _ilu), _ilu)
_ilnodes, _ilels = _NET.line_ladder("a", "b", _ilz, _ild, _iltr)
_ilspec = {"nodes": ["a", "b"] + _ilnodes,
           "elements": _ilels + [{"id": "rl", "kind": "r", "a": "b", "b": "gnd", "ohm": _ilz}],
           "drivers": [{"id": "d", "node": "a", "ohm": _ilz, "state": "driving"}]}
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    _ilgot = _NET.solve(_ilspec, fs=_ilfs, n=_iln, probes="b", sources={"d": _ilvs})
_ilk = int(round(_ild * _ilfs))
_ilref = 0.5 * np.concatenate([np.zeros(_ilk), _ilvs[:_iln - _ilk]])
_ilerr = float(np.abs(_ilgot - _ilref).max()) / 0.5
check("a lumped ladder at its own measured section count delivers the edge delayed and halved, "
      "to the error the rule promises",
      _ilerr < 0.02 and abs(float(_ilgot[-1]) - 0.5) < 1e-3,
      f"{len(_ilnodes) + 1} sections for a line {_ild / _iltr:.3g} rise times long: "
      f"{_ilerr * 100:.2f} % of the swing against an ideal delay, and a matched divider at DC "
      f"({float(_ilgot[-1]):.4f} of 0.5)")


check("a lead-in extent STATED in closed form scales with the physics it came from (open_drain: "
      "5 tau is proportional to Rp and to Cb)",
      _l1 > 0 and abs(_lstated(1770.0, 400e-12) / _l1 - 2.0) < 0.02
      and abs(_lstated(885.0, 800e-12) / _l1 - 2.0) < 0.02,
      f"5 tau = {_l1} samples at 885 ohm / 400 pF; {_lstated(1770.0, 400e-12)} at 2x Rp; "
      f"{_lstated(885.0, 800e-12)} at 2x Cb")
_lbad = _lstated(885.0, 4700e-12)
check("and it exceeds one symbol exactly where such a bus fails (11x over the 400 pF limit, the "
      "line no longer reaches the input-high threshold inside a bit)",
      _lbad > int(_lg.samples_per_ui),
      f"5 tau = {_lbad} samples = {_lbad / _lg.samples_per_ui:.1f} UI at 400 kbit/s, against "
      f"{_l1 / _lg.samples_per_ui:.1f} UI at the compliant limit")


# THE TERMINAL STORE'S FLOOR. Three real exports land on their own lattice's q^2/12/(fs/2) --
# a BARE rounder, no dither. An earlier revision of this file asserted +2.68 / +3.00 / +3.08 dB
# above it; that was spectral leakage from an in-band signal through a rectangular window, and a
# Hann window on the same band of the same records reads q^2/12 to 0.00 / 0.00 / -0.03 dB.
# NOTHING IS TRIMMED HERE ANY MORE, and what the trim was for is worth keeping in view. The
# channel applies a LINEAR convolution (U-16), so a record without a lead-in starts on a quiescent
# line -- and `scope(kind="brickwall")` after it is still a CIRCULAR frequency-domain stage
# (BACKLOG #54), so it saw that turn-on as a step in the periodic extension and rang on it. The
# overshoot landed on the record's last samples 6.2 % above anything the link itself does, which
# stole vertical range and dropped the ranged 11-bit code count from 1949 to 1840 -- outside the
# 1851-2035 band the three real captures set. This file used to trim 1024 samples off each end and
# measure the interior. It now renders a LEAD-IN and discards it (`Signal(lead_in=True)`, sized by
# the chain's own measured impulse-response extent), so the record delivered here is a window on a
# link that was already running and there is no spoiled edge to trim: 1950 codes, and the floor on
# q^2/12 across the WHOLE record. The section above is the ground truth for that mechanism.
_ys = (Signal(seed=17, grid=Grid(fs=_FSAMP, baud=16e9, n=1 << 20, v_full=0.8), lead_in=True)
       .carrier(kind="nrz", pattern="prbs13").lossy(loss_db=12.0, loss_at_ghz=8.0, causal=True)
       .scope(bw_hz=110e9).digitize(noise_rms=_sig_c, bits=10, full_scale=_A)
       .scope(bw_hz=32e9, kind="brickwall")).waveform()
for _dth, _want, _lbl in ((0.0, 0.0, "a BARE store lands on q^2/12 -- where the real captures are"),
                          (1 / np.sqrt(12), 3.01, "and dither is a real mechanism, worth 3 dB when a"
                                                  " chain actually has it")):
    _r = INST.store_record(_ys, bits=11, dither_lsb=_dth, rng=np.random.default_rng(5))
    _u = np.unique(_r); _qs = float(np.min(np.diff(_u)))
    _d = _stopband_psd_db_per_hz(_r, _FSAMP) - INST.quantisation_floor_db_per_hz(_qs, _FSAMP)
    check(_lbl, abs(_d - _want) < 0.4, f"{_d:+.2f} dB above q^2/12/(fs/2) (real, windowed: +0.00/+0.00/-0.03)")
    check(f"  ... and it is still a lattice ({_dth:.4f} LSB of dither does not smear it)",
          _on_lattice(_r, _qs) > 0.9999 and 1851 <= len(_u) <= 2035,
          f"{len(_u)} distinct, on-lattice {_on_lattice(_r, _qs):.4f}")

# ---- a sub-chain is a value: one nested-signal form for every op that takes one (GitHub #75) ----
# `crosstalk(aggressor=)`, `modulate(message=)`, `open_drain(second=)`, `crosstalk_matrix(
# aggressors=)` and a `net` driver's `source` all take {"ops": [...]}, {"from": <recipe>} or a
# Signal, through ONE resolver (`wfmsynth.subchain`). The claims are that the second signal is
# exactly the chain it says it is, that its randomness is its own and separately re-rollable, and
# that nothing is fitted to the record behind the caller's back. Each is checked against something
# that does not go through the resolver.
from wfmsynth import subchain as _SUB
from wfmsynth.compose import _carrier as _scar
_sg = Grid(fs=64e9, baud=4e9, n=1 << 13)
_sj = dict(rj=0.4)
_sat = "crosstalk/1/aggressor"


def _svic():
    return Signal(seed=1, grid=_sg).carrier("nrz", n_ui=512, jitter=_sj)


def _slane(loss_db, noise_rms=0.02):
    s = Signal(grid=_sg).carrier("nrz", n_ui=512, seed=9, jitter=_sj)
    if loss_db:
        s = s.lossy(loss_db=loss_db, loss_at_ghz=2.0)
    return s.digitize(noise_rms=noise_rms) if noise_rms else s


def _salone(ops, at=_sat, seed=1):
    """A sub-chain's ops rendered WITHOUT the resolver, under the role names it should have."""
    s = Signal(grid=_sg)
    s.ops = [dict(o) for o in ops]
    return s._run(_SUB.Namespaced(_St(seed), at + "/"))


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _ssig = _svic().crosstalk(aggressor=_slane(12.0), coupling=0.1).digitize(noise_rms=0.01)
    _sx, _sv = _ssig.waveform(), _svic().waveform()
    _sa = _salone(_slane(12.0).ops)
    _sown = _sx - P.crosstalk(_sv, _sa, coupling=0.1)          # the record's own digitizer noise
    _sbare = _svic().crosstalk(aggressor=_slane(12.0), coupling=0.1).waveform()
    check("an aggressor given as a sub-chain IS that chain rendered separately and coupled by hand",
          np.array_equal(_sbare, P.crosstalk(_sv, _sa, coupling=0.1))
          and np.array_equal(_ssig.subchain(_sat), _sa),
          f"max |op - by hand| = {float(np.abs(_sbare - P.crosstalk(_sv, _sa, coupling=0.1)).max()):.1e} "
          f"on a coupled term of {float(np.abs(_sbare - _sv).max()):.3f} peak; `subchain()` returns "
          f"the same {len(_sa)} samples")

    # THE EFFECT, not the call: the aggressor's own channel must be visible in what the victim
    # receives. FEXT couples the aggressor's derivative, so an aggressor that has been through
    # 12 dB of loss -- slower edges -- couples a lower-frequency term than the same lane raw.
    _sc = [spectral_centroid(_svic().crosstalk(aggressor=_slane(_l, 0), coupling=0.1).waveform()
                             - _sv) for _l in (0.0, 12.0)]
    check("the aggressor's OWN channel reaches the victim: a lane after 12 dB of loss couples a "
          "lower-frequency term than the same lane raw",
          _sc[1] < 0.8 * _sc[0],
          f"spectral centroid of the coupled term {_sc[0]:.4f} -> {_sc[1]:.4f} of fs "
          f"({_sc[1] / _sc[0]:.2f}x)")

    # ONE factor inside the sub-chain re-rolls and nothing else moves.
    _srole = f"{_sat}/noise/2"
    _sa1 = _ssig.subchain(_sat, streams=_St(1).reroll(_srole, seed=11))
    _sclean = _salone(_slane(12.0).ops[:2])
    _sd = float(np.std(_sa1 - _sa)) / (np.sqrt(2.0) * 0.02)
    _sres = float(np.abs(_ssig.contrast(_srole, seed=11)
                         - (P.crosstalk(_sv, _sa1, coupling=0.1) + _sown)).max())
    check("contrast() reaches inside: re-rolling the aggressor's noise changes the aggressor's "
          "noise and nothing else",
          _srole in _ssig.roles() and abs(_sd - 1.0) < 0.05 and _sres < 1e-12
          and abs(float(np.std(_sa1 - _sclean)) / 0.02 - 1.0) < 0.05,
          f"roles {_ssig.roles()}; the aggressor moved by {_sd:.3f}x the sqrt(2)*20 mV two "
          f"independent draws differ by (8192 samples estimate an rms to 0.8 %), and the record equals "
          f"victim + coupling of the NEW aggressor + its own unchanged noise to {_sres:.1e}")

    _sother = (Signal(seed=4, grid=Grid(fs=_sg.fs, baud=2.5e9, n=_sg.n))
               .carrier("pam4", n_ui=320, jitter=_sj).lossy(loss_db=3.0, loss_at_ghz=2.0))
    _sfrom = _svic().crosstalk(aggressor={"from": _sother}, coupling=0.2)
    _srt = Signal.from_recipe(_json.loads(_sfrom.to_json()))
    _sknob = _slane(12.0)
    _sknob.ops[1]["loss_db"] = 12.5
    check("a recipe under `from` is that record bit for bit -- its own seed, its own symbol rate "
          "-- and the nested recipe replays from JSON alone, under a digest that covers it",
          np.array_equal(_sfrom.subchain(_sat), _sother.waveform())
          and np.array_equal(_srt.waveform(), _sfrom.waveform())
          and np.array_equal(Signal.from_recipe(_json.loads(_ssig.to_json())).waveform(), _sx)
          and _svic().crosstalk(aggressor=_sknob).sha256()
          != _svic().crosstalk(aggressor=_slane(12.0)).sha256(),
          f"a 2.5 GBd PAM4 lane inside a 4 GBd record on one {_sg.fs / 1e9:.0f} GSa/s grid; "
          f"0.5 dB of loss inside the sub-chain changes the record's sha256")

    _srefused = []
    for _sbad in ({"from": Signal(grid=Grid(fs=32e9, baud=4e9, n=_sg.n)).carrier("nrz", n_ui=1024)
                   .recipe()},
                  {"ops": [{"op": "carrier", "kind": "nrz", "n_ui": 512},
                           {"op": "digitize", "n_out": _sg.n // 2}]},
                  {"ops": [{"op": "carrier", "kind": "nrz", "n_ui": 512},
                           {"op": "acquire", "profile": {"sample_rate_hz": 32e9,
                                                         "record_length": _sg.n}}]},
                  {"ops": [{"op": "carrier", "kind": "nrz", "n_ui": 512}, {"op": "eo"}]}):
        _sb = Signal(seed=1, grid=_sg)
        _sb.ops = [{"op": "carrier", "kind": "nrz", "n_ui": 512},
                   {"op": "crosstalk", "aggressor": _sbad}]
        try:
            _sb.waveform()
            _srefused.append("RENDERED")
        except ValueError as _se:
            _srefused.append(next((w for w in ("SAME grid", "different timebase", "COMPLEX",
                                               "never resampled")
                                   if w in str(_se)), "UNNAMED"))
    check("a sub-chain at another rate, another length, another timebase or complex is REFUSED "
          "by name -- never resampled, cut or stripped to fit",
          _srefused == ["never resampled", "SAME grid", "different timebase", "COMPLEX"],
          f"{_srefused}")

    _sspec = dict(n_ui=512, seed=9, jitter=_sj)
    _sleg = _svic().crosstalk(aggressor=_sspec, coupling=0.2)
    _swant = P.crosstalk(_sv, _scar({"kind": "nrz", **_sspec}, _St(1), _sg, "xtalk1"),
                         coupling=0.2)
    check("the carrier-spec shorthand is untouched: same recipe, same role stream, same samples",
          np.array_equal(_sleg.waveform(), _swant)
          and _sleg.ops[1] == {"op": "crosstalk", "aggressor": _sspec, "coupling": 0.2}
          and _sleg.roles() == ["jitter/0", "jitter/xtalk1"],
          f"max |op - primitive on role 'jitter/xtalk1'| = "
          f"{float(np.abs(_sleg.waveform() - _swant).max()):.1e}")

# ---------------------------------------------------------------------------------------------
# #80: TWO GENERAL STAGES. `law` is a named transfer law and `filter` a stated linear filter, for
# a shape that has no op of its own. Each law is checked against its analytic curve by a property
# of a SINE through it (a harmonic, a mean, a duty, a noise floor) rather than by re-evaluating
# the formula it was written with, and each filter by its magnitude MEASURED on an impulse sent
# down a chain, against the design's closed form at the bilinear transform's own frequency map.
from scipy import special as _zsp
from scipy.optimize import brentq as _zbrentq
from wfmsynth import stages as _ZS

_zfs, _zn = 1e9, 1 << 14
_zg = Grid(fs=_zfs, n=_zn)
_zcyc = 37                                        # whole cycles in the record: bin 37, no leakage
_zf0 = _zcyc * _zfs / _zn


def _zsine(amp, **law):
    s = Signal(seed=1, grid=_zg).carrier("sine", cycles=_zcyc, amp=amp)
    return (s.law(**law) if law else s).waveform()


def _zharm(y, k):
    return float(2.0 * np.abs(np.fft.rfft(y)[k * _zcyc]) / _zn)


_zA = 0.05
_zy = _zsine(_zA, kind="tanh", gain=3.0, sat=0.8)
_zwant = (3.0 * _zA) ** 3 / (12.0 * 0.8 ** 2)
check("law tanh: the small-signal gain is `gain` and the first distortion product is the cubic "
      "term's, (gain A)^3 / (12 sat^2)",
      abs(_zharm(_zy, 1) / (3.0 * _zA) - 1.0) < 0.01 and abs(_zharm(_zy, 3) / _zwant - 1.0) < 0.02
      and abs(_zsine(50.0, kind="tanh", gain=3.0, sat=0.8)).max() <= 0.8,
      f"fundamental {_zharm(_zy, 1) / _zA:.4f}x against gain 3 (compressed "
      f"{(1 - _zharm(_zy, 1) / (3 * _zA)) * 100:.2f} %), HD3 {_zharm(_zy, 3) * 1e6:.2f} uV "
      f"against {_zwant * 1e6:.2f} uV, and a 50x overdrive stays inside +/-0.8")

_zy = _zsine(0.5, kind="poly", coeffs=[0.0, 1.0, 0.2, -0.4])
check("law poly: 0.2 x^2 - 0.4 x^3 on a sine makes exactly the second and third harmonics the "
      "identities give, c2 A^2 / 2 and c3 A^3 / 4",
      abs(_zharm(_zy, 2) - 0.2 * 0.25 / 2) < 1e-12 and abs(_zharm(_zy, 3) - 0.4 * 0.125 / 4) < 1e-12
      and _zharm(_zy, 4) < 1e-12,
      f"HD2 {_zharm(_zy, 2):.6f} (0.025000), HD3 {_zharm(_zy, 3):.6f} (0.012500), "
      f"HD4 {_zharm(_zy, 4):.1e}")

_zy = _zsine(1.0, kind="limiter", lo=-0.3, hi=0.6)
_zwant = (0.5 - np.arcsin(0.6) / np.pi, 0.5 - np.arcsin(0.3) / np.pi)
_zgot = (float(np.mean(_zy == 0.6)), float(np.mean(_zy == -0.3)))
check("law limiter: a sine spends the arcsine's share of its time on each rail, and is untouched "
      "between them",
      abs(_zgot[0] - _zwant[0]) < 2e-3 and abs(_zgot[1] - _zwant[1]) < 2e-3
      and np.array_equal(_zy[(_zy > -0.3) & (_zy < 0.6)],
                         _zsine(1.0)[(_zy > -0.3) & (_zy < 0.6)]),
      f"{_zgot[0] * 100:.2f} % at +0.6 (analytic {_zwant[0] * 100:.2f} %), "
      f"{_zgot[1] * 100:.2f} % at -0.3 ({_zwant[1] * 100:.2f} %)")

_zy = _zsine(1.0, kind="deadzone", width=0.5)
_zwant = 2.0 * np.arcsin(0.25) / np.pi
check("law deadzone: the output is zero for the arcsine's share of a cycle and the peak is "
      "moved in by half the width",
      abs(float(np.mean(_zy == 0.0)) - _zwant) < 2e-3 and abs(_zy.max() - 0.75) < 1e-6
      and abs(_zy.min() + 0.75) < 1e-6,
      f"{np.mean(_zy == 0.0) * 100:.2f} % of the record at zero (analytic {_zwant * 100:.2f} %), "
      f"peaks {_zy.min():+.4f} / {_zy.max():+.4f}")

_zh, _zf = _zsine(1.0, kind="rectify", mode="half"), _zsine(1.0, kind="rectify", mode="full")
check("law rectify: the mean of a rectified sine is A/pi (half) and 2A/pi (full), and a "
      "full-wave rectifier doubles the frequency",
      abs(_zh.mean() - 1 / np.pi) < 1e-6 and abs(_zf.mean() - 2 / np.pi) < 1e-6
      and _zharm(_zf, 1) < 1e-9 and abs(_zharm(_zf, 2) - 4 / (3 * np.pi)) < 1e-6,
      f"means {_zh.mean():.6f} (0.318310) and {_zf.mean():.6f} (0.636620); full-wave fundamental "
      f"{_zharm(_zf, 1):.1e}, second harmonic {_zharm(_zf, 2):.6f} (4/3pi = 0.424413)")

_zb = 10
_zstep = 2.0 / 2 ** _zb
_zx = _zsine(1.0 - _zstep)
_zy = _zsine(1.0 - _zstep, kind="quantize", bits=_zb, lo=-1.0, hi=1.0)
_zsqnr = 10 * np.log10(np.mean(_zx ** 2) / np.mean((_zy - _zx) ** 2))
_zover = _zsine(3.0, kind="quantize", bits=_zb, lo=-1.0, hi=1.0)
check("law quantize: a full-scale sine comes out 6.02 N + 1.76 dB above its own quantisation "
      "error, on 2^N codes, and an overdrive SATURATES at the end codes",
      abs(_zsqnr - (6.02 * _zb + 1.76)) < 0.3
      and len(np.unique(_zover)) == 2 ** _zb
      and _zover.min() == -1.0 and _zover.max() == 1.0 - _zstep,
      f"SQNR {_zsqnr:.2f} dB at {_zb} bits (ideal {6.02 * _zb + 1.76:.2f}); a 3x overdrive "
      f"uses {len(np.unique(_zover))} codes from {_zover.min():+.4f} to {_zover.max():+.6f}")

_zy = _zsine(2.0, kind="diode", r_load_ohm=1e3)
_zx = _zsine(2.0)
_zfw = _zy > 1e-6
_zres = float(np.abs(_zy[_zfw] + 0.025852 * np.log(_zy[_zfw] / (1e3 * 1e-12) + 1.0)
                     - _zx[_zfw]).max())
_zbig = _ZS.apply_law(np.array([5.0, 50.0, 500.0]), "diode", r_load_ohm=1e3)
_zbres = float(np.abs(_zbig + 0.025852 * np.log(_zbig / 1e-9 + 1.0)
                      - np.array([5.0, 50.0, 500.0])).max())
check("law diode: the output satisfies the circuit it claims to be -- v_in = v_diode + v_load "
      "with the Shockley current in the load -- and blocks in reverse",
      _zres < 1e-12 and _zbres < 1e-10 and abs(_zy.min() + 1e3 * 1e-12) < 1e-15,
      f"Kirchhoff residual {_zres:.1e} V over the forward half of a 2 V sine and {_zbres:.1e} V "
      f"at 5 / 50 / 500 V; peak {_zy.max():.4f} V (a {2.0 - _zy.max():.3f} V drop at "
      f"{_zy.max():.2f} mA), reverse {_zy.min():.1e} V = -R * I_sat")

# THE COMPARATOR. It is the bridge from an analog signal to a digital one, so what it must get
# right is the COUNT and the TIME of its transitions, against a signal whose crossings are known.
_zthr, _zhys, _zrms = 0.2, 0.3, 0.02
_zclean = Signal(seed=5, grid=_zg).carrier("sine", cycles=_zcyc, amp=1.0)
_znoisy = (Signal(seed=5, grid=_zg).carrier("sine", cycles=_zcyc, amp=1.0)
           .rx_noise(rms=_zrms))
_zpeak = float(np.abs(_znoisy.waveform() - _zclean.waveform()).max())
_zwith = _znoisy.law("comparator", threshold=_zthr, hysteresis=_zhys).event_list()
_znone = (Signal(seed=5, grid=_zg).carrier("sine", cycles=_zcyc, amp=1.0).rx_noise(rms=_zrms)
          .law("comparator", threshold=_zthr).event_list())
check("law comparator: with hysteresis wider than the noise there is EXACTLY one transition per "
      "crossing, and without it the same record chatters",
      _zhys > 2 * _zpeak and len(_zwith) == 2 * _zcyc and len(_znone) > 2 * _zcyc
      and all(a.direction != b.direction for a, b in zip(_zwith.events, _zwith.events[1:])),
      f"{len(_zwith)} transitions for {2 * _zcyc} crossings at {_zhys:.2f} of hysteresis against "
      f"{2 * _zpeak:.3f} peak-to-peak of noise; {len(_znone)} with none")
_zev = _zclean.law("comparator", threshold=_zthr, hysteresis=_zhys).event_list().events
_zw = 2 * np.pi * _zf0
_zwant = np.array([((np.arcsin((_zthr + 0.5 * _zhys * e.direction)) if e.direction > 0
                     else np.pi - np.arcsin(_zthr - 0.5 * _zhys)) + 2 * np.pi * (i // 2)) / _zw
                   for i, e in enumerate(_zev)])
_zterr = float(np.abs(np.array([e.t_s for e in _zev]) - _zwant).max())
_zbound = (_zw / _zfs) ** 2 / 8.0 / (_zw * np.sqrt(1 - 0.35 ** 2))     # chord error / slope
check("  ... and each transition's time is the INPUT's crossing of the trip level, to the error "
      "of a straight line between two samples and not to a sample",
      len(_zev) == 2 * _zcyc and _zterr < _zbound,
      f"worst |t_event - arcsin| = {_zterr * 1e12:.3f} ps against a chord bound of "
      f"{_zbound * 1e12:.3f} ps, on a {1e12 / _zfs:.0f} ps grid")

# THE FILTERS. One impulse down a chain, and the ratio of the two spectra.
_zimp = np.zeros(_zn)
_zimp[_zn // 8] = 1.0
_zfreq = np.fft.rfftfreq(_zn, 1 / _zfs)


def _zmeas(**flt):
    y = Signal(seed=1, grid=_zg).capture(values=_zimp, fs_hz=_zfs).filter(**flt).waveform()
    return np.abs(np.fft.rfft(y) / np.fft.rfft(_zimp))


_zfc = 60e6
_zr = np.tan(np.pi * _zfreq[:-1] / _zfs) / np.tan(np.pi * _zfc / _zfs)   # the bilinear map
_zjr = 1j * _zr
_zth4 = lambda s: s ** 4 + 10 * s ** 3 + 45 * s ** 2 + 105 * s + 105     # noqa: E731
_zk = _zbrentq(lambda k: abs(105 / _zth4(1j * k)) - np.sqrt(0.5), 1.0, 3.0)
_zeps2 = 10 ** (1.0 / 10) - 1
_zez, _zep, _zek = sp.ellipap(5, 0.5, 60.0)
_zclosed = {
    "butter": (dict(order=5), 1 / np.sqrt(1 + _zr ** 10)),
    "bessel": (dict(order=4), np.abs(105 / _zth4(_zjr * _zk))),
    "cheby1": (dict(order=5, ripple_db=1.0),
               1 / np.sqrt(1 + _zeps2 * _zsp.eval_chebyt(5, _zr) ** 2)),
    "elliptic": (dict(order=5, ripple_db=0.5, stop_db=60.0),
                 np.abs(sp.freqs_zpk(_zez, _zep, _zek, _zr)[1])),
}
for _zkind, (_zkw, _zwant) in _zclosed.items():
    _zgot = _zmeas(kind=_zkind, f_hz=_zfc, **_zkw)[:-1]
    _zerr = float(np.abs(_zgot - _zwant).max())
    check(f"filter {_zkind}: the magnitude measured on an impulse through the chain is the "
          f"design's closed form, at every frequency on the grid",
          _zerr < 1e-9,
          f"max ||H| - closed form| = {_zerr:.1e} over {len(_zwant)} bins; "
          f"{20 * np.log10(_zgot[np.argmin(abs(_zfreq[:-1] - _zfc))]):.3f} dB at the "
          f"{_zfc / 1e6:.0f} MHz corner")
_zdb = 20 * np.log10(_zmeas(kind="elliptic", f_hz=_zfc, order=5, ripple_db=0.5, stop_db=60.0))
_zstop = int(np.argmax(_zdb <= -60.0))            # where the stopband begins
check("  ... and the elliptic design is equiripple where it says: the passband dips to exactly "
      "-ripple_db and the stopband comes back up to exactly -stop_db",
      abs(_zdb[_zfreq <= _zfc].min() + 0.5) < 1e-3 and abs(_zdb[_zstop:].max() + 60.0) < 1e-3,
      f"passband minimum {_zdb[_zfreq <= _zfc].min():.4f} dB; stopband from "
      f"{_zfreq[_zstop] / 1e6:.1f} MHz, maximum {_zdb[_zstop:].max():.4f} dB")
for _zbt, _zfh, _zwant in (
        ("high", _zfc, np.sqrt(_zr ** 6 / (1 + _zr ** 6))),
        ("bandpass", [40e6, 90e6], None), ("bandstop", [40e6, 90e6], None)):
    _zgot = _zmeas(kind="butter", f_hz=_zfh, order=3, btype=_zbt)
    if _zwant is None:
        _zedge = [float(_zgot[np.argmin(abs(_zfreq - f))]) for f in (40e6, 90e6)]
        _zmid = float(_zgot[np.argmin(abs(_zfreq - 60e6))])
        check(f"filter butter, btype={_zbt}: both stated edges are the -3 dB points and the band "
              f"between them is {'passed' if _zbt == 'bandpass' else 'removed'}",
              all(abs(20 * np.log10(e) + 3.0103) < 0.05 for e in _zedge)
              and (_zmid > 0.999 if _zbt == "bandpass" else _zmid < 1e-3),
              f"{20 * np.log10(_zedge[0]):.3f} / {20 * np.log10(_zedge[1]):.3f} dB at 40 / 90 "
              f"MHz, {20 * np.log10(_zmid):.2f} dB at 60 MHz")
    else:
        _zerr = float(np.abs(_zgot[:-1] - _zwant).max())
        check(f"filter butter, btype={_zbt}: the mirrored closed form", _zerr < 1e-9,
              f"max error {_zerr:.1e}")

_zbaud, _zbeta = 100e6, 0.35
_zgot = _zmeas(kind="rrc", beta=_zbeta, baud=_zbaud, span_ui=64)
_zu = np.abs(_zfreq) / _zbaud
_zwant = np.sqrt(np.where(_zu <= (1 - _zbeta) / 2, 1.0, np.where(
    _zu >= (1 + _zbeta) / 2, 0.0,
    0.5 * (1 + np.cos(np.pi / _zbeta * (_zu - (1 - _zbeta) / 2))))))
_zerr = float(np.abs(_zgot - _zwant).max())
_zh = _ZS.design_filter(_zfs, "rrc", beta=_zbeta, baud=_zbaud, span_ui=64)["taps"]
_zrc = np.convolve(_zh, _zh)
_zisi = np.abs(_zrc[len(_zrc) // 2 % 10::10] / _zrc.max())
_zisi = float(np.sort(_zisi)[-2])                 # the largest sample at a symbol instant but one
check("filter rrc: the magnitude is the square root of a raised cosine, and two of them in "
      "cascade put zeros on every other symbol's instant",
      _zerr < 0.01 and _zisi < 1e-3 and abs(_zh.sum() - 1.0) < 1e-12,
      f"max ||H| - sqrt(RC)| = {_zerr:.1e} at 64 symbols of span and "
      f"{float(np.abs(_zmeas(kind='rrc', beta=_zbeta, baud=_zbaud) - _zwant).max()):.1e} at the "
      f"default 8 (the truncation, which is what span buys); worst inter-symbol sample "
      f"{_zisi:.1e} of the peak, DC gain {_zh.sum():.6f}")

_zw0, _zq = 2 * np.pi * 40e6, 4.0
_zpole = complex(-_zw0 / (2 * _zq), _zw0 * np.sqrt(1 - 1 / (4 * _zq ** 2)))
_zgot = _zmeas(kind="zpk", poles=[_zpole, _zpole.conjugate()], zeros=[-2 * np.pi * 150e6],
               gain=_zw0 ** 2 / (2 * np.pi * 150e6))[:-1]
_zs = 2j * _zfs * np.tan(np.pi * _zfreq[:-1] / _zfs)
_zwant = np.abs(_zw0 ** 2 / (2 * np.pi * 150e6) * (_zs + 2 * np.pi * 150e6)
                / ((_zs - _zpole) * (_zs - _zpole.conjugate())))
_zerr = float(np.abs(_zgot - _zwant).max())
check("filter zpk: an analog resonance and a zero, stated in rad/s, come out as that transfer "
      "function at the bilinear transform's frequency map",
      _zerr < 1e-9 and abs(_zgot.max() / _zq - 1.0) < 0.05,
      f"max ||H| - |H(j 2 fs tan(pi f / fs))|| = {_zerr:.1e}; the Q = {_zq:g} peak measures "
      f"{_zgot.max():.3f}x at {_zfreq[np.argmax(_zgot)] / 1e6:.2f} MHz")

_zint = (Signal(seed=1, grid=_zg).carrier("sine", cycles=_zcyc, amp=1.0, phase_rad=np.pi / 2)
         .filter("integrate", tau_s=1 / _zw).waveform())
_zdif = (Signal(seed=1, grid=_zg).carrier("sine", cycles=_zcyc, amp=1.0)
         .filter("differentiate", tau_s=1 / _zw).waveform())
_zt = np.arange(_zn) / _zfs
_zhalf = _zw / (2 * _zfs)
_zie = float(np.abs(_zint - _zhalf / np.tan(_zhalf) * np.sin(_zw * _zt)).max())
_zde = float(np.abs(_zdif[1:] - np.sin(_zhalf) / _zhalf
                    * np.cos(_zw * (_zt[1:] - 0.5 / _zfs))).max())
check("filter integrate / differentiate: a cosine integrates to a sine and a sine differentiates "
      "to a cosine, each at its rule's own closed-form gain, the difference half a sample late",
      _zie < 1e-10 and _zde < 1e-10,
      f"integrate: {_zie:.1e} from sin * (wT/2)/tan(wT/2), {(1 - _zhalf / np.tan(_zhalf)) * 100:.4f}"
      f" % under the ideal 1/w; differentiate: {_zde:.1e} from cos(w(t - T/2)) * sinc, "
      f"{(1 - np.sin(_zhalf) / _zhalf) * 100:.4f} % under the ideal w")

# THE FRONT END, AT THE SAME SETTINGS. `scope` stays the named op; the general stage must not be
# a second, slightly different, Bessel.
_zbg = Grid(fs=256e9, baud=32e9, n=1 << 13)
_zsc = Signal(seed=3, grid=_zbg).carrier("nrz", n_ui=32, pattern="prbs7").scope(bw_hz=32e9).waveform()
_zfl = (Signal(seed=3, grid=_zbg).carrier("nrz", n_ui=32, pattern="prbs7")
        .filter("bessel", order=4, f_hz=32e9).waveform())
check("filter bessel, order 4, IS the existing front-end filter at the same settings -- bound "
      "zero, the same samples",
      np.array_equal(_zsc, _zfl),
      f"max |filter - scope| = {np.abs(_zsc - _zfl).max():.1e} over {len(_zsc)} samples of a "
      f"32 GHz Bessel-4 on PRBS7")

# THE DELAY IT REPORTS IS THE DELAY IT HAS. A ramp through a low-pass comes out the same ramp,
# later by the group delay at DC -- so a comparator before and after the filter times it.
_zramp = np.clip((np.arange(_zn) - 2000) / 6000.0, 0.0, 1.0)
_zpre = (Signal(seed=1, grid=_zg).capture(values=_zramp, fs_hz=_zfs)
         .law("comparator", threshold=0.5).event_list().events[0].t_s)
_zchain = (Signal(seed=1, grid=_zg).capture(values=_zramp, fs_hz=_zfs)
           .filter("butter", f_hz=5e6, order=4).law("comparator", threshold=0.5))
_zpost = _zchain.event_list().events[0].t_s
_zrep = _zchain.filter_delays()[0]["group_delay_s"]
check("a filter REPORTS its group delay, and adding it to an upstream event's time lands on the "
      "event as delivered",
      abs((_zpost - _zpre) - _zrep) < 1e-3 / _zfs,
      f"the 50 % crossing moves {(_zpost - _zpre) * 1e9:.4f} ns through a 5 MHz Butterworth-4; "
      f"filter_delays() reports {_zrep * 1e9:.4f} ns "
      f"(off by {abs(_zpost - _zpre - _zrep) * _zfs:.1e} of a sample)")

# ... AND THAT IS THE SLOW-EDGE NUMBER. An edge FAST against the filter arrives at the filter's
# own 50 % step delay, a different number, which it reports beside it. Both regimes, four designs.
_zstep = (np.arange(_zn) >= 2000).astype(float)
_zfast = []
for _zk, _zkw in (("butter", dict(order=1)), ("butter", dict(order=4)), ("bessel", dict(order=4)),
                  ("fir", dict(taps=[0.1, 0.2, 0.4, 0.2, 0.1]))):
    _zkw = _zkw if _zk == "fir" else dict(f_hz=5e6, **_zkw)
    _zc = (Signal(seed=1, grid=_zg).capture(values=_zstep, fs_hz=_zfs)
           .filter(_zk, **_zkw).law("comparator", threshold=0.5))
    _zd = _zc.filter_delays()[0]
    _zfast.append((_zk, (_zc.event_list().events[0].t_s - 1999.5 / _zfs) * _zfs,
                   _zd["step_delay_s"] * _zfs, _zd["group_delay_s"] * _zfs))
check("a filter reports the delay of a FAST edge too: the 50 % point of its own step response, "
      "which is not its group delay",
      all(abs(m - s) < 1e-6 for _k, m, s, _g in _zfast)
      and abs(_zfast[0][2] / _zfast[0][3] - np.log(2.0)) < 2e-3      # one pole: ln 2 of 1/(2 pi fc)
      and abs(_zfast[1][1] - _zfast[1][3]) > 5.0,
      "a step's midpoint moved / step_delay_s / group_delay_s, in samples: "
      + "; ".join(f"{k} {m:.3f} / {s:.3f} / {g:.3f}" for k, m, s, g in _zfast))

_zzp = _zmeas(kind="butter", f_hz=_zfc, order=5, zero_phase=True)[:-1]
_zunit = (np.arange(_zn) >= _zn // 2).astype(float)
_zzs = (Signal(seed=1, grid=_zg).capture(values=_zunit, fs_hz=_zfs)
        .filter("butter", f_hz=_zfc, order=5, zero_phase=True))
_zys = _zzs.waveform()
_zsym = float(np.abs(_zys[_zn // 2:_zn // 2 + 500] + _zys[_zn // 2 - 1:_zn // 2 - 501:-1]
                     - 1.0).max())
check("zero_phase=True keeps the STATED magnitude (a forward-backward pass would square it), "
      "has no delay, and says so in the recipe",
      float(np.abs(_zzp - 1 / np.sqrt(1 + _zr ** 10)).max()) < 1e-6 and _zsym < 1e-9
      and _zzs.recipe()["ops"][-1]["zero_phase"] is True
      and _zzs.filter_delays()[0]["group_delay_s"] == 0.0,
      f"max ||H| - closed form| = {np.abs(_zzp - 1 / np.sqrt(1 + _zr ** 10)).max():.1e} "
      f"({20 * np.log10(_zzp[np.argmin(abs(_zfreq[:-1] - _zfc))]):.3f} dB at the corner, where "
      f"filtfilt gives -6.02); a step comes out odd-symmetric about its own instant to "
      f"{_zsym:.1e}")

print("== complex baseband: the mixers are the textbook identities, the source has ground "
      "truth, and nothing drops an imaginary part (#81) ==")
import json as _qjson                                                # noqa: E402
from wfmsynth import iq as _IQ                                              # noqa: E402
from wfmsynth.acquire import AcquisitionProfile as _QProfile                 # noqa: E402
_qfs, _qbd, _qfc = 8e9, 25e6, 1e9
_qt = np.arange(1 << 15) / _qfs
_qtones = ((-19.7e6, 0.30, 0.1), (-7.1e6, 0.20, 1.0), (0.9e6, 0.25, 2.0), (4.3e6, 0.15, -1.0),
           (18.9e6, 0.10, 0.5))
_qx = sum(_a * np.exp(1j * (2 * np.pi * _f * _qt + _p)) for _f, _a, _p in _qtones)
_qg = Grid(fs=_qfs, n=len(_qx), domain="iq", fc_hz=_qfc)
_qone = 0.7 * np.exp(1j * (2 * np.pi * 3.3e6 * _qt + 0.4))
_qup = Signal(seed=0, grid=_qg).capture(values=_qone).upconvert().waveform()
_querr = float(np.abs(_qup - 0.7 * np.cos(2 * np.pi * (_qfc + 3.3e6) * _qt + 0.4)).max())
check("a complex tone at +f0 up-converts to cos(2 pi (fc + f0) t): one sideband, the upper one "
      "-- so Q is used, with the right sign",
      _qup.dtype == np.float64 and _querr < 1e-10,
      f"max |passband - closed form| = {_querr:.2e} on a 0.7 peak carrier at 1.0033 GHz")
_qlpf = dict(pass_hz=20e6, stop_hz=60e6)
_qrt = (Signal(seed=0, grid=_qg).capture(values=_qx).upconvert()
        .downconvert(fc_hz=_qfc, lpf=_qlpf).waveform())
_qhalf = len(_IQ.lowpass(_qlpf, _qfs)) // 2
_qpe, _qsd = _IQ.lowpass_error(_qlpf, _qfs)
_qbound = (_qpe + 10 ** (_qsd / 20)) * sum(_a for _, _a, _ in _qtones)
_qrterr = float(np.abs(_qrt - _qx)[_qhalf:-_qhalf].max())
check("downconvert(upconvert(x)) is x to within the STATED low-pass's own error -- its measured "
      "passband deviation on the tones plus its measured stopband on their images",
      _qrt.dtype == np.complex128 and 1e-3 * _qbound < _qrterr < _qbound,
      f"{_qrterr:.2e} against a bound of {_qbound:.2e} ({2 * _qhalf + 1} taps: passband "
      f"|H-1| <= {_qpe:.2e}, stopband {_qsd:.1f} dB) on five tones inside +/-20 MHz")

_qgs = Grid(fs=_qfs, baud=_qbd, n=1 << 16, domain="iq", fc_hz=_qfc)
_qs = Signal(seed=1, grid=_qgs).iq("qam16", prbs=15, prbs_phase=100)
_qsent, _ = _qs.transmitted()
_qevm = {_sp: _IQ.evm(_IQ.demodulate(
    Signal(seed=1, grid=_qgs).iq("qam16", prbs=15, prbs_phase=100, span_ui=_sp).waveform(),
    _qgs, span_ui=_sp)[16:-16], _qsent[16:-16]) for _sp in (8, 16, 32)}
check("an I/Q source's ground truth is what the record holds: a matched filter returns "
      "transmitted(), and the residual is the pulse's truncation and falls with its span",
      _qevm[8] > 2 * _qevm[16] > 2 * _qevm[32] and _qevm[32] < 0.002,
      "QAM-16, root-raised-cosine beta 0.35: EVM "
      + " / ".join(f"{100 * _qevm[_sp]:.3f} %" for _sp in (8, 16, 32))
      + " at span_ui = 8 / 16 / 32")
_qh = _IQ.rrc(np.arange(-6400, 6401) / 16.0, 0.35)
_qisi = max(abs(float(np.sum(_qh[:len(_qh) - _k * 16] * _qh[_k * 16:]) / 16)) for _k in (1, 2, 3))
check("  ... and the pulse is a ROOT raised cosine: unit energy, and an autocorrelation that is "
      "zero at every other symbol centre",
      abs(float(np.sum(_qh * _qh) / 16) - 1.0) < 1e-8 and _qisi < 1e-8,
      f"energy {float(np.sum(_qh * _qh) / 16):.10f} per symbol, ISI {_qisi:.1e}")

_qgr = Grid(fs=16e9, baud=_qbd, n=1 << 18, domain="iq", fc_hz=_qfc)
_qrf = (Signal(seed=1, grid=_qgr).iq("qam16", prbs=15, prbs_phase=100).upconvert()
        .acquire(_QProfile(sample_rate_hz=4e9, record_length=1 << 16,
                           input_bandwidth_hz=3e9, bits=8))
        .downconvert(fc_hz=_qfc, lpf=_qlpf, fs_out_hz=250e6))
_qy = _qrf.waveform()
_qsg = _qrf.stored_grid(_qy)
_qrs, _ = _qrf.transmitted()
_qgot = _IQ.demodulate(_qy, _qsg)[16:-16]
_qfit = _IQ.evm(_qgot, _qrs[16:-16], fit=True)
check("an instrument sampling RF: I/Q -> 1 GHz passband -> an 8-bit 4 GSa/s acquisition -> "
      "down-converted at the STORED rate returns the transmitted symbols",
      (_qsg.fs, _qsg.domain) == (250e6, "iq") and _qfit < 0.01
      and _IQ.evm(_qgot, _qrs[16:-16]) > 0.3,
      f"EVM {100 * _qfit:.2f} % once the front end's carrier phase "
      f"({np.angle(_IQ.complex_gain(_qgot, _qrs[16:-16])):+.3f} rad) is taken out, "
      f"{100 * _IQ.evm(_qgot, _qrs[16:-16]):.0f} % before; stored at {_qsg.fs / 1e6:.0f} MSa/s")


def _qrefusal(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    return ""


_qmsg = _qrefusal(lambda: Signal(seed=1, grid=_qgs).iq("qpsk", prbs=9)
                  .lossy(loss_db=3.0, loss_at_ghz=1.0))
check("a real-only op handed an I/Q record refuses BY NAME, saying which op converts; complex "
      "samples offered to a real record are refused, not cast",
      _qmsg.startswith("lossy") and "'iq'" in _qmsg and "upconvert(" in _qmsg
      and "complex (I/Q)" in _qrefusal(
          lambda: Signal(seed=0, grid=Grid(fs=_qfs, n=len(_qx), domain="real"))
          .capture(values=_qx)),
      _qmsg[:118] + "...")
_qfollow = Signal(seed=0, grid=Grid(fs=_qfs, n=len(_qx))).capture(values=_qx)
_qstated = Signal(seed=0, grid=Grid(fs=_qfs, n=len(_qx), domain="iq")).capture(values=_qx)
_qsrc = Signal(seed=1, grid=Grid(fs=_qgs.fs, baud=_qgs.baud, n=_qgs.n)).iq("qpsk", prbs=9)
_qsrc_stated = Signal(seed=1, grid=Grid(fs=_qgs.fs, baud=_qgs.baud, n=_qgs.n,
                                         domain="iq")).iq("qpsk", prbs=9)
check("a grid that does not state its domain takes its source's: an `iq` source and a complex "
      "capture are I/Q records with the recipe and samples of one that said domain='iq', a "
      "real source's recipe is unchanged, and a grid that STATES 'real' refuses `iq` by name",
      _qfollow.grid.domain == "iq" and _qfollow.to_json() == _qstated.to_json()
      and np.array_equal(_qfollow.waveform(), _qx)
      and _qsrc.to_json() == _qsrc_stated.to_json()
      and np.array_equal(_qsrc.waveform(), _qsrc_stated.waveform())
      and "domain" not in Signal(seed=1, grid=Grid(fs=80e9, baud=10e9, n=4096))
      .carrier("nrz", n_ui=512).recipe()["grid"]
      and "STATES domain='real'" in _qrefusal(
          lambda: Signal(seed=1, grid=Grid(fs=_qgs.fs, baud=_qgs.baud, n=_qgs.n,
                                           domain="real")).iq("qpsk", prbs=9)),
      f"inferred and stated I/Q digests equal: capture {_qfollow.sha256()[:12]} == "
      f"{_qstated.sha256()[:12]}, iq {_qsrc.sha256()[:12]} == {_qsrc_stated.sha256()[:12]}")
_qr = Signal.from_recipe(_qjson.loads(_qrf.to_json()))
_qreal = Signal(seed=1, grid=Grid(fs=80e9, baud=10e9, n=4096)).carrier("nrz", n_ui=512)
check("waveform() of an I/Q record is complex128 and replays bit for bit from its recipe; a "
      "real record's recipe has no new key, so its content address did not move",
      np.array_equal(_qr.waveform(), _qy) and _qr.sha256() == _qrf.sha256()
      and _qrf.recipe()["grid"]["domain"] == "iq"
      and sorted(_qreal.recipe()["grid"]) == ["baud", "fs", "n", "v_full"]
      and _qreal.waveform().dtype == np.float64,
      f"{_qy.dtype}, {len(_qy)} samples; real grid keys {sorted(_qreal.recipe()['grid'])}")


# ---- unit-to-unit variation: parameters drawn from stated tolerances (GitHub #86) ----
# A toleranced parameter is drawn once per record under a `unit/...` role and the NUMBER is
# written into the recipe. Three things have to be true for that number to be worth having: it
# follows the distribution that was stated (including what "normal" means -- tol = 3 sigma,
# truncated), it is the value the waveform was actually made with, and re-rolling the unit
# moves it and nothing else. The populations are seeds 0..N-1, so every figure is deterministic;
# a distribution passes at p > 1e-3 and its negative control must fail below 1e-9.
from scipy import stats as _ust                                   # noqa: E402
from wfmsynth import tolerance as _TOL                            # noqa: E402

_un, _ug = 10000, Grid(fs=100e9, n=8000)
_up = np.array([[_o[_k] for _k in ("r_source", "c_load_f", "atten", "bw_hz")] for _o in (
    Signal(seed=_s, grid=_ug).probe(
        r_source=_TOL.tol(1e3, 5), c_load_f=_TOL.tol(1e-11, 10, dist="normal"),
        atten=_TOL.tol(0.1, 2, dist="corner"),
        bw_hz=_TOL.corners({"slow": 1e9, "typ": 2e9, "fast": 3e9})).recipe()["ops"][0]
    for _s in range(_un))])
_uu = (_up[:, 0] - 950.0) / 100.0
_uz = (_up[:, 1] - 1e-11) / (1e-12 / 3.0)
_utn = _ust.truncnorm(-3, 3)
_upu, _upn = _ust.kstest(_uu, "uniform").pvalue, _ust.kstest(_uz, _utn.cdf).pvalue
check(f"a 'uniform' tolerance is uniform over nom +/- tol ({_un} records)",
      _upu > 1e-3 and _uu.min() >= 0 and _uu.max() <= 1
      and _ust.kstest((_up[:, 0] - 1e3) / (50.0 / 3), _utn.cdf).pvalue < 1e-9,
      f"KS p = {_upu:.3f}; the same values against the normal form: "
      f"p = {_ust.kstest((_up[:, 0] - 1e3) / (50.0 / 3), _utn.cdf).pvalue:.0e}")
check("a 'normal' tolerance is tol = 3 sigma, truncated at +/- tol",
      _upn > 1e-3 and np.abs(_uz).max() <= 3.0 and abs(_uz.std() / _utn.std() - 1) < 0.03
      and _ust.kstest(_uz / 3.0, _ust.truncnorm(-1, 1).cdf).pvalue < 1e-9,
      f"KS p = {_upn:.3f}; sd = {_uz.std():.4f} sigma against {_utn.std():.4f}; max |z| = "
      f"{np.abs(_uz).max():.3f}; read as tol = 1 sigma: p = "
      f"{_ust.kstest(_uz / 3.0, _ust.truncnorm(-1, 1).cdf).pvalue:.0e}")
_uc2 = np.unique(_up[:, 2], return_counts=True)[1]
_uc3 = np.unique(_up[:, 3], return_counts=True)[1]
check("'corner' is the two extremes and a named corner set its names, each equally likely",
      len(_uc2) == 2 and len(_uc3) == 3 and _ust.chisquare(_uc2).pvalue > 1e-3
      and _ust.chisquare(_uc3).pvalue > 1e-3
      and _ust.chisquare(_uc3, f_exp=_un * np.array([0.25, 0.5, 0.25])).pvalue < 1e-9,
      f"{_uc2.tolist()} (p = {_ust.chisquare(_uc2).pvalue:.3f}), {_uc3.tolist()} "
      f"(p = {_ust.chisquare(_uc3).pvalue:.3f})")
check("two toleranced parameters of one record are independent draws",
      abs(np.corrcoef(_uu, _uz)[0, 1]) < 0.04,
      f"r = {np.corrcoef(_uu, _uz)[0, 1]:+.4f} over {_un} records (4 standard errors = 0.04)")


def _urec(seed, noise=0.0):
    return (Signal(seed=seed, grid=_ug)
            .carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=25e6, duty=0.5, tr_s=1e-10)
            .probe(r_source=_TOL.tol(1e3, 20), c_load_f=_TOL.tol(2e-12, 20, dist="normal"),
                   **({"noise_rms": noise} if noise else {})))


def _urise(y, fs, lo=0.0, hi=3.3):
    a, b = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
    i = np.flatnonzero((y[:-1] < a) & (y[1:] >= a))[0]
    j = np.flatnonzero((y[:-1] < b) & (y[1:] >= b))
    j = j[j > i][0]
    return ((j + (b - y[j]) / (y[j + 1] - y[j])) - (i + (a - y[i]) / (y[i + 1] - y[i]))) / fs


_ut = []
for _s in range(8):
    _usig = _urec(_s)
    _uo = _usig.recipe()["ops"][1]
    _ut.append((_urise(_usig.waveform(), _ug.fs), np.log(9.0) * _uo["r_source"] * _uo["c_load_f"]))
_ut = np.array(_ut)
check("the resolved R and C in the recipe ARE the ones in the waveform (rise = ln 9 * R * C)",
      np.abs(_ut[:, 0] / _ut[:, 1] - 1).max() < 1e-3 and _ut[:, 0].max() / _ut[:, 0].min() > 1.2,
      f"worst {np.abs(_ut[:, 0] / _ut[:, 1] - 1).max():.1e} over 8 units spanning "
      f"{_ut[:, 0].min() * 1e9:.2f}..{_ut[:, 0].max() * 1e9:.2f} ns")

_us, _uclean = _urec(3, 2e-3), _urec(3)
_uy = _us.waveform()
_unoise = _uy - _uclean.waveform()
_uyu = _us.contrast("unit/*", seed=40)
_udn = float(np.abs((_uyu - _uclean.unit(40).waveform()) - _unoise).max())
_uon = _us.contrast("probe/1") - _uclean.waveform()
check("re-rolling unit/* is another unit under the SAME noise; re-rolling the noise, the reverse",
      np.array_equal(_uyu, _us.unit(40).waveform()) and float(np.abs(_uyu - _uy).max()) > 0.02
      and _udn < 1e-12 and abs(np.corrcoef(_uon, _unoise)[0, 1]) < 0.05
      and abs(_uon.std() / 2e-3 - 1) < 0.05,
      f"unit re-rolled: waveform moves {np.abs(_uyu - _uy).max() * 1e3:.1f} mV, its noise by "
      f"{_udn:.1e} V; noise re-rolled: r = {np.corrcoef(_uon, _unoise)[0, 1]:+.4f} with the old")
_ura, _urb = _us.recipe()["ops"], _us.unit(40).recipe()["ops"]
check("  ... and only the toleranced parameters differ between the two recipes",
      _ura[0] == _urb[0] and {k for k in _ura[1] if _ura[1][k] != _urb[1][k]}
      == {"r_source", "c_load_f", "_unit"},
      f"changed in the probe op: {sorted(k for k in _ura[1] if _ura[1][k] != _urb[1][k])}")
import json as _ujson                                             # noqa: E402
_ub = Signal.from_recipe(_ujson.loads(_us.to_json()))
check("a recipe with resolved values replays bit-identically",
      _ub.sha256() == _us.sha256() and np.array_equal(_ub.waveform(), _uy),
      f"digest {_us.sha256()[:12]}, max |diff| = {np.abs(_ub.waveform() - _uy).max():.1e}")

print("== the pieces compose: a matched filter on an envelope, a tolerance inside a sub-chain ==")
_jg = Grid(fs=8e9, baud=25e6, n=1 << 16, domain="iq", fc_hz=1e9)


def _jsrc():
    return Signal(seed=1, grid=_jg).iq("qam16", prbs=15, prbs_phase=100)


_jsent, _jt = _jsrc().transmitted()
_jk = np.rint(_jt * _jg.fs).astype(int)[10:-10]         # symbol centres; the ends lack neighbours
_jsent = _jsent[10:-10]
_jm = _IQ.evm(_jsrc().filter("rrc", beta=0.35, zero_phase=True).waveform()[_jk], _jsent)
_jr = _IQ.evm(_jsrc().waveform()[_jk], _jsent, fit=True)
check("filter('rrc') on an I/Q record is the source's matched filter: the record AT the symbol "
      "centres is the sent symbols, nothing fitted", _jm < 0.006 and _jr > 0.10,
      f"EVM {100 * _jm:.3f} % after the filter (bound 0.6 %, set by the 16-symbol truncation) "
      f"against {100 * _jr:.1f} % for the shaped source alone")
from wfmsynth import tolerance as _JT                                       # noqa: E402
_jgr = Grid(fs=20e9, n=4000)
_jlane = Signal(grid=_jgr).carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=25e6,
                                   tr_s=_JT.tol(1e-9, 20))
_jrec = Signal(seed=4, grid=_jgr).carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=10e6,
                                          tr_s=1e-10).crosstalk(aggressor=_jlane, coupling=0.1)
_jtr = _jrec.ops[1]["aggressor"]["ops"][0]["tr_s"]
_jagg = _jrec.subchain("crosstalk/1/aggressor")

def _jcross(level):                       # the first FALLING crossing of `level`, in samples
    i = int(np.argmax(_jagg < level))
    return i - 1 + (_jagg[i - 1] - level) / (_jagg[i - 1] - _jagg[i])


_jmeas = float((_jcross(0.33) - _jcross(2.97)) / _jgr.fs)     # 90 % to 10 % of 0..3.3 V
_jother = _jrec.unit(9)
check("a tolerance inside a sub-chain is drawn by the record it joins: the aggressor's edge is "
      "the recipe's number, and unit() moves it",
      _jrec.roles() == ["crosstalk/1/aggressor/unit/0/tr_s"] and abs(_jmeas / _jtr - 1) < 0.02
      and _jother.ops[1]["aggressor"]["ops"][0]["tr_s"] != _jtr
      and np.array_equal(_jrec.contrast("unit/*", seed=9), _jother.waveform()),
      f"recipe tr_s {_jtr * 1e12:.1f} ps, measured 10-90 % {_jmeas * 1e12:.1f} ps (bound 2 %); "
      f"unit 9 has {_jother.ops[1]['aggressor']['ops'][0]['tr_s'] * 1e12:.1f} ps")

print("== joins: combine, and gain / offset / gate / delay driven by a signal (GitHub #76, #79) ==")
from wfmsynth import joins as _JN                                           # noqa: E402
_cg = Grid(fs=1e6, baud=1e3, n=1 << 14)
_ct = np.arange(_cg.n) / _cg.fs


def _csin(f, amp=1.0, offset=0.0, phase_rad=0.0):
    return Signal(grid=_cg).carrier("sine", f_hz=f, amp=amp, offset=offset, phase_rad=phase_rad)


def _csq01():
    return Signal(grid=_cg).carrier("square", f_hz=500.0, amp=0.5, offset=0.5)


_cy = _csin(1e3).combine(_csin(7e3, amp=0.3, phase_rad=0.4)).waveform()
_cw = np.sin(2 * np.pi * 1e3 * _ct) + 0.3 * np.sin(2 * np.pi * 7e3 * _ct + 0.4)
check("combine: the sum of two sines is the analytic sum", np.array_equal(_cy, _cw),
      f"max |diff| = {np.abs(_cy - _cw).max():.1e} (bound 0)")
_ca = _csin(50e3).combine([_csq01()], how="product").waveform()
_cb = _csin(50e3).modulate(kind="ook", suppressed=True, message=_csq01()).waveform()
_cc = _csin(50e3).combine([_csq01()], how="product", gains=[2.0]).waveform()
_cd = _csin(50e3).modulate(kind="ook", message=Signal(grid=_cg).carrier("square", f_hz=500.0)).waveform()
check("combine: a carrier times a 0/1 square is modulate's OOK, suppressed and not",
      np.array_equal(_ca, _cb) and np.abs(_cc - _cd).max() <= 4 * np.finfo(float).eps,
      f"max |diff| = {np.abs(_ca - _cb).max():.1e} suppressed (bound 0), "
      f"{np.abs(_cc - _cd).max():.1e} with the carrier kept and a gain of 2 (bound 8.9e-16)")
_cgt = _csin(50e3).gate(by=_csq01()).waveform()
_cbu = _csin(50e3).burst(t_on_s=1e-3, t_off_s=1e-3, edge_frac=0.0).waveform()
_cdf = np.flatnonzero(_cgt != _cbu)
check("gate by a square is burst at that cadence, but for burst's own one-sample ramps",
      len(_cdf) <= 2 * 9 and set(_cdf % 2000) <= {0, 999} and not _cbu[_cdf].any(),
      f"{len(_cdf)} of {_cg.n} samples differ, each the first or last of a burst, which "
      f"burst zeroes and an ideal switch lets through")
_cev = _csin(50e3).gate(by=_csin(500.0)).event_list()
_cwant = np.sort(np.concatenate([(np.arange(9) + 1 / 12) / 500.0, (np.arange(9) + 5 / 12) / 500.0]))
_cwant = _cwant[_cwant < _cg.n / _cg.fs]
_cgot = np.array([e.t_s for e in _cev])
check("gate emits its switching instants: the times a sine enable crosses the threshold",
      len(_cgot) == len(_cwant) and np.abs(_cgot - _cwant).max() < 2e-9,
      f"{len(_cgot)} transitions, worst {np.abs(_cgot - _cwant).max() * 1e9:.2f} ns from "
      f"asin(0.5) (bound 2 ns: linear interpolation across a 1 us sample)")
_cfj, _camp, _ck = 300.0, 2e-6, 4
_cd0 = _ck / _cg.fs
_cerr = []
for _cf0 in (2e3, 50e3, 200e3, 390e3):
    _cyd = _csin(_cf0).delay(by=_csin(_cfj, amp=_camp, offset=_cd0)).waveform()
    _cex = np.sin(2 * np.pi * _cf0 * (_ct - _cd0 - _camp * np.sin(2 * np.pi * _cfj * _ct)))
    _cerr.append((np.abs(_cyd - _cex)[24:-16].max(), _JN.delay_error(_cf0, _cg.fs)))
check("delay: a tone under a moving delay is within the error the op reports, across the band",
      all(e <= b + 1e-13 for e, b in _cerr) and _JN.describe_delay(_cg.fs)["error"] < 2.5e-5,
      "measured / reported at 0.002, 0.05, 0.2, 0.39 fs: "
      + ", ".join(f"{e:.1e} / {b:.1e}" for e, b in _cerr)
      + f"; stated for the band below 0.4 fs: {_JN.describe_delay(_cg.fs)['error']:.2e}")
_cyd = _csin(2e3).delay(by=_csin(_cfj, amp=_camp, offset=_cd0,
                                  phase_rad=-2 * np.pi * _cfj * _cd0)).waveform()
_cyt = _csin(2e3).timing(pj=dict(amp_ps=_camp * 1e12, f_hz=_cfj)).waveform()
_cdiff = np.abs(_cyd[_ck:] - _cyt[:-_ck])[16:-16].max()
_clin = (2 * np.pi * 2e3 / _cg.fs) ** 2 / 8
check("delay by a sine reproduces timing's periodic jitter at that amplitude and frequency",
      _cdiff <= _JN.delay_error(2e3, _cg.fs) + _clin
      and np.ptp(_cyd[_ck:] - np.sin(2 * np.pi * 2e3 * _ct)[:-_ck]) > 0.02,
      f"max |diff| = {_cdiff:.2e}; bound {_JN.delay_error(2e3, _cg.fs):.1e} (delay, reported) + "
      f"{_clin:.2e} (timing's linear interpolation, (2*pi*f/fs)^2/8)")
_cng = Grid(fs=64e9, baud=1e9, n=1 << 13)
_cns = (Signal(seed=2, grid=_cng).carrier("nrz", n_ui=128)
        .combine([Signal(grid=_cng).carrier("nrz", n_ui=128).events(kind="glitch", count=3)],
                 gains=[0.2]))
_cne = _cns.event_list()
check("a combined sub-chain's events are in the record's event list under its namespace",
      len(_cne) == 3 and {e.at for e in _cne} == {"combine/1/others/0"} and not _cne.on_record(),
      f"{len(_cne)} events, at = {sorted({e.at for e in _cne})}")
# ---------------------------------------------------------------------------------------------
print("\n-- sequence and switch: chains joined in time, or selected by a signal (GitHub #77) --")
from wfmsynth import sequence as _QS                                        # noqa: E402
_qg = Grid(fs=1e9, baud=50e6, n=8000)
_qna, _qm = 2000, 200


def _qtone(f=5e6):
    return Signal(grid=_qg).carrier("sine", f_hz=f)


def _qlane():
    return (Signal(seed=7, grid=_qg).carrier("nrz", n_ui=400, jitter=dict(rj=0.03))
            .lossy(loss_db=3.0, loss_at_ghz=0.2).rx_noise(rms=0.01))


def _qseq(join="cut", t_join_s=None, state="restart"):
    return Signal(seed=3, grid=_qg).sequence(
        [{"chain": {"from": _qtone().recipe()}, "duration_s": _qna / _qg.fs},
         {"chain": {"from": _qlane().recipe()}, "state": state}], join=join, t_join_s=t_join_s)


_qa, _qb = _qtone().waveform(), _qlane().waveform()
_qcut = _qseq().waveform()
check("a two-segment sequence IS its chains rendered alone and spliced (array_equal)",
      np.array_equal(_qcut, np.concatenate([_qa[:_qna], _qb[:_qg.n - _qna]])),
      f"{_qna} + {_qg.n - _qna} samples, a jittered noisy lane after a tone, 0 differ")
_qx, _qxe = _qseq("crossfade", _qm / _qg.fs).realize()
_qr = 0.5 * (1 - np.cos(np.pi * (np.arange(_qm) + 0.5) / _qm))
_qin = float(np.max(np.abs(_qx[_qna:_qna + _qm]
                           - ((1 - _qr) * _qa[_qna:_qna + _qm] + _qr * _qb[:_qm]))))
check("a crossfade is exact outside its stated window and the stated raised cosine inside it",
      np.array_equal(_qx[:_qna], _qcut[:_qna])
      and np.array_equal(_qx[_qna + _qm:], _qcut[_qna + _qm:]) and _qin < 1e-15,
      f"outside the {_qm}-sample window 0 samples differ from the cut; inside, max error "
      f"against the closed form {_qin:.1e}")
_qe = _qxe.events[0]
check("the boundary event is at the requested time and carries the segment index",
      len(_qxe) == 1 and _qe.kind == "segment" and _qe.sample == _qna
      and abs(_qe.t_s * _qg.fs - _qna) < 1 and _qe.params["segment"] == 1,
      f"requested {_qna / _qg.fs * 1e6:g} us, event at sample {_qe.sample} "
      f"(t_s {_qe.t_s * 1e6:g} us), segment {_qe.params['segment']}, width {_qe.width}")
_qt = _qtone(5.3e6)
_qtw = _qt.waveform()


def _qstate(state):
    return Signal(grid=_qg).sequence([{"chain": _qt, "duration_s": _qna / _qg.fs},
                                      {"chain": _qt, "state": state}]).waveform()


_qjump = float(abs(_qstate("restart")[_qna] - _qstate("restart")[_qna - 1]))
check("'continue' carries a tone's phase across the boundary; 'restart' starts it again",
      np.array_equal(_qstate("continue"), _qtw)
      and np.array_equal(_qstate("restart")[_qna:], _qtw[:_qg.n - _qna]) and _qjump > 0.3,
      f"5.3 MHz continued over a 2 us boundary is the uninterrupted tone (0 samples differ); "
      f"restarted it steps {_qjump:.3f} at the boundary")
_qstep = Signal(grid=_qg).sequence(
    [{"chain": Signal(grid=_qg).carrier("dc", level=0.0), "duration_s": _qna / _qg.fs},
     {"chain": Signal(grid=_qg).carrier("dc", level=1.0)}],
    join="crossfade", t_join_s=_qm / _qg.fs).waveform()


def _qcross(level):
    i = int(np.argmax(_qstep > level))
    return i - 1 + (level - _qstep[i - 1]) / (_qstep[i] - _qstep[i - 1])


_q1090 = (_qcross(0.9) - _qcross(0.1)) / _qg.fs
_qan = (np.arccos(-0.8) - np.arccos(0.8)) / np.pi * _qm / _qg.fs
check("a crossfade between two levels rises 10-90 % in 0.5903 of t_join_s",
      abs(_q1090 - _qan) < 0.01 / _qg.fs,
      f"measured {_q1090 * 1e9:.3f} ns, analytic {_qan * 1e9:.3f} ns (bound 0.01 sample)")
for _qbad, _qword in (
        (lambda: Signal(grid=_qg).sequence([{"chain": _qt, "duration_s": 2.0005e-6}, {"chain": _qt}]),
         "not on the grid"),
        (lambda: Signal(grid=_qg).sequence([{"chain": _qt, "duration_s": 2e-6},
                                            {"chain": _qt, "duration_s": 5e-6}]), "fill it exactly")):
    try:
        _qbad()
        _qmsg = ""
    except ValueError as _qerr:
        _qmsg = str(_qerr)
    check(f"a sequence whose durations are '{_qword}' is refused, never rounded",
          _qword in _qmsg, _qmsg[:96] + "...")


def _qsw(select, **kw):
    return Signal(seed=3, grid=_qg).switch(
        select=select, inputs=[{"from": _qtone().recipe()}, {"from": _qlane().recipe()}], **kw)


def _qdc(level):
    return Signal(grid=_qg).carrier("dc", level=level)


check("a switch with a constant selector IS that input (array_equal), switching time or not",
      all(np.array_equal(_qsw(_qdc(lv), threshold=0.5, t_switch_s=ts).waveform(), w)
          for lv, w in ((0.0, _qa), (1.0, _qb)) for ts in (0.0, 50e-9)),
      "selector 0 -> inputs[0], selector 1 -> inputs[1], at t_switch_s 0 and 50 ns")
_qf = 1.3e6
_qy, _qev = _qsw(_qtone(_qf), threshold=0.25, t_switch_s=40e-9).realize()
_qw, _qas = 2 * np.pi * _qf, np.arcsin(0.25)
_qk = np.arange(12)
_qwant = np.sort(np.concatenate([(_qas + 2 * np.pi * _qk) / _qw,
                                 (np.pi - _qas + 2 * np.pi * _qk) / _qw]))
_qwant = _qwant[_qwant < _qg.n / _qg.fs]
_qgot = np.array([e.t_s for e in _qev])
_qterr = float(np.max(np.abs(_qgot - _qwant)) * _qg.fs) if len(_qgot) == len(_qwant) else np.inf
_qbound = (_qw / _qg.fs) / (8 * np.cos(_qas))           # a chord's sag under a sine, as time
check("switch events are the selector's analytic threshold crossings, between samples",
      len(_qgot) == len(_qwant) and _qterr < _qbound
      and [e.sample for e in _qev] == list(np.ceil(_qwant * _qg.fs).astype(int)),
      f"{len(_qgot)} changes of a 1.3 MHz sine across 0.25; worst time error {_qterr:.1e} "
      f"sample (bound {_qbound:.1e}, linear interpolation)")
_qbusy = np.zeros(_qg.n, bool)
for _qe in _qev:
    _qbusy[_qe.sample:_qe.sample + _qe.width] = True
_qideal = np.where(_qtone(_qf).waveform() > 0.25, _qb, _qa)
check("a switch is each input exactly wherever no change is in progress",
      np.array_equal(_qy[~_qbusy], _qideal[~_qbusy]) and _qbusy.sum() == 40 * len(_qev),
      f"{int((~_qbusy).sum())} samples outside the {len(_qev)} switching windows of 40 "
      f"samples: 0 differ from the selected input")
print("== several correlated channels from one recipe (GitHub #78) ==")
import json as _cjson                                                       # noqa: E402
import wfmsynth as _cws                                                     # noqa: E402
_cg = Grid(fs=320e9, baud=10e9, n=32 * 512)
_cramp = np.arange(_cg.n, dtype=float)


def _cclk():
    return Signal().carrier("square", f_hz=5e9, tr_frac=0.3)


def _cdat():
    return Signal().carrier("nrz", n_ui=512, pattern="prbs7", tr_frac=0.3)


def _ccross(y):
    s = np.signbit(y)
    i = np.nonzero(s[1:] != s[:-1])[0]
    return i + y[i] / (y[i] - y[i + 1])


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _crec = _cws.Channels(
        {"clk": _cclk().use("pll"), "data": _cdat().use("pll"),
         "ramp": Signal().capture(values=_cramp).use("pll"),
         "ramp3": Signal().capture(values=3.0 * _cramp + 7.0).use("pll"), "quiet": _cdat()},
        shared={"pll": Signal().timing(phase_noise=dict(rms_ps=2.0))}, seed=1, grid=_cg)
    _cx = _crec.waveforms()
    _cideal = _cws.Channels({"clk": _cclk(), "data": _cdat()}, seed=1, grid=_cg).waveforms()
    _cy = _crec.contrast("pll/timing/0")
    _cback = _cws.Channels.from_recipe(_cjson.loads(_crec.to_json()))
    _cz = _cback.waveforms()
# a ramp through a timing warp reads back the displacement, so two ramps of different slope
# recover the one shared source twice
_cd1 = (_cramp - _cx["ramp"])[8:-8]
_cd3 = (_cramp - (_cx["ramp3"] - 7.0) / 3.0)[8:-8]
check("two channels through one shared timing source are displaced by the same samples",
      abs(_cd1.std() / _cg.fs * 1e12 - 2.0) < 0.02 and np.abs(_cd1 - _cd3).max() < 1e-10,
      f"{_cd1.std() / _cg.fs * 1e12:.4f} ps rms asked 2.0 (bound 1 %); the two recovered "
      f"displacements differ by {np.abs(_cd1 - _cd3).max():.1e} samples (bound 1e-10)")
_ctc, _ctd = _ccross(_cideal["clk"]), _ccross(_cideal["data"])
_cjc, _cjd = _ccross(_cx["clk"]) - _ctc, _ccross(_cx["data"]) - _ctd
_cdisp = _cramp - _cx["ramp"]
_cerr = max(np.abs(_cjc - np.interp(_ctc + _cjc, _cramp, _cdisp)).max(),
            np.abs(_cjd - np.interp(_ctd + _cjd, _cramp, _cdisp)).max())
check("a clock and its data sharing a timing source carry the same jitter: every edge of "
      "both sits on the one applied displacement",
      _cerr < 0.01 * _cjc.std() and len(_ctc) == 511 and len(_ctd) > 200,
      f"{len(_ctc)} clock and {len(_ctd)} data edges, TIE {_cjc.std():.3f} samples rms; "
      f"worst edge is {_cerr:.4f} samples off the displacement (bound 1 % of rms)")
_cmoved = {k: float(np.abs(_cy[k] - _cx[k]).max()) for k in ("clk", "data", "quiet")}
check("re-rolling the shared role moves both channels that use it and not the one that does "
      "not", _crec.roles() == ["pll/timing/0"] and _cmoved["clk"] > 0.1
      and _cmoved["data"] > 0.1 and _cmoved["quiet"] == 0.0,
      f"largest change clk {_cmoved['clk']:.3f}, data {_cmoved['data']:.3f}, "
      f"quiet {_cmoved['quiet']:.1f}")
check("a several-channel record replays bit-identically from its recipe, under one digest",
      all(np.array_equal(_cx[k], _cz[k]) for k in _cx) and _cback.sha256() == _crec.sha256(),
      f"{len(_cx)} channels, sha256 {_crec.sha256()[:16]}")


def _clane():
    return Signal().carrier("nrz", n_ui=512, pattern="prbs9", seed=9).lossy(loss_db=6.0,
                                                                            loss_at_ghz=5.0)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _cxt = _cws.Channels({"victim": _cdat().crosstalk(aggressor=Signal().use("lane"),
                                                      coupling=0.1), "lane": _clane()},
                         seed=1, grid=_cg).waveforms()
    _cref = Signal(seed=1, grid=_cg)
    _cref.ops = _cdat().crosstalk(aggressor=_clane(), coupling=0.1).ops
    _crefx, _crefa = _cref.waveform(), _cref.subchain("crosstalk/1/aggressor")
check("a channel used as another's crosstalk aggressor is that channel: the victim equals "
      "the sub-chain build of the same lane, and the aggressor is the lane's own output",
      np.array_equal(_cxt["victim"], _crefx) and np.array_equal(_cxt["lane"], _crefa),
      f"victim max difference {np.abs(_cxt['victim'] - _crefx).max():.1e}; the coupling "
      f"moved it by up to {np.abs(_crefx - _cideal['data']).max():.3f}")
try:
    _cws.Channels({"a": _cdat().crosstalk(aggressor=Signal().use("b"), coupling=0.1),
                   "b": _cdat().crosstalk(aggressor=Signal().use("a"), coupling=0.1)}, grid=_cg)
    _cmsg = ""
except ValueError as _ce:
    _cmsg = str(_ce)
check("two channels that name each other are refused naming the cycle",
      "a -> b -> a is a cycle" in _cmsg, _cmsg[:34])
# ===================================================================================================
print("== a triggered acquisition: render long, find the condition, keep the window (#85) ==")
# ===================================================================================================
# A trigger cuts the window a scope would out of a long render. Each claim is measured against
# something the trigger did not compute: the long render itself, the transitions `stages`'
# comparator reports for it (a separate implementation), a brute-force arming loop, and 1/sqrt(n).
import re as _trg_re                                                          # noqa: E402
from wfmsynth import trigger as _TRG, stages as _trg_stages                   # noqa: E402

_trg_g = Grid(fs=2e9, baud=2e9 / 128, n=2048 * 128)
_trg_prof = dict(sample_rate_hz=500e6, record_length=_trg_g.n // 4, input_bandwidth_hz=150e6)


def _trg_link():
    return (Signal(seed=3, grid=_trg_g).carrier("nrz", n_ui=2048, pattern="prbs7",
                                                jitter=dict(rj=1.0))
            .lossy(loss_db=3.0, loss_at_ghz=0.02).rx_noise(rms=0.02).acquire(_trg_prof))


def _trg_cross(y, level, near, rising=True):
    """Sub-sample crossing of `level` nearest sample `near`, by np.interp on the local edge."""
    k = np.arange(near - 3, near + 3)
    seg = y[k] if rising else y[k][::-1]
    return float(np.interp(level, seg, k if rising else k[::-1]))


_trg_fs = 500e6
_trg_long = _trg_link().waveform()
_trg_sig = _trg_link().trigger(pre_s=200e-9, post_s=600e-9, level=0.0, hysteresis=0.2)
_trg_x, _trg_ev = _trg_sig.realize()
_trg_e = _trg_ev.events[-1]
_trg_meas = _trg_cross(_trg_x, 0.0, _trg_e.sample) / _trg_fs
check("the window is a slice of the long render and the triggering edge sits at pre_s to "
      "half a sample, where the event says",
      np.array_equal(_trg_x, _trg_long[_trg_e.params["start"]:][:len(_trg_x)])
      and abs(_trg_meas - 200e-9) * _trg_fs <= 0.5 and abs(_trg_meas - _trg_e.t_s) * _trg_fs < 1e-6
      and _trg_sig.stored_grid(_trg_x).fs == _trg_fs,
      f"edge measured at {_trg_meas * 1e9:.4f} ns, event {_trg_e.t_s * 1e9:.4f} ns, pre_s 200 ns, "
      f"half a sample {0.5 / _trg_fs * 1e9:.1f} ns")

_trg_sj = 3e-9
_trg_xs, _trg_evs = _trg_link().trigger(pre_s=200e-9, post_s=200e-9, level=0.0, hysteresis=0.2,
                                        jitter_s=_trg_sj, n_segments=180).realize()
_trg_rows = _trg_xs.reshape(180, -1)
_trg_pos = np.array([_trg_cross(_trg_rows[k], 0.0, e.sample - k * 200) / _trg_fs
                     for k, e in enumerate(_trg_evs)])
_trg_jit = np.array([e.params["jitter_s"] for e in _trg_evs])
_trg_want = float(np.sqrt(_trg_sj ** 2 + (1 / _trg_fs) ** 2 / 12))
check("with trigger jitter the edge sits at pre_s - jitter to half a sample in every segment, "
      "and its spread is the stated rms",
      np.abs(_trg_pos - (200e-9 - _trg_jit)).max() * _trg_fs <= 0.5 + 1e-6
      and abs(_trg_pos.std() / _trg_want - 1) < 0.15,
      f"180 segments: position rms {_trg_pos.std() * 1e9:.3f} ns against "
      f"sqrt(jitter^2 + dt^2/12) = {_trg_want * 1e9:.3f} ns (bound 15 %), "
      f"worst residual "
      f"{np.abs(_trg_pos - (200e-9 - _trg_jit)).max() * _trg_fs:.3f} samples")

# holdoff and timestamps: the comparator's rising transitions ARE the event times in the long
# render, and a loop written out here is the scope's arming
_trg_seen = []
_trg_stages.apply_law(_trg_long, "comparator", fs=_trg_fs, events=_trg_seen, threshold=0.0,
                      hysteresis=0.2)
# the comparator trips at the band's edge; the instant the same edge crossed 0 is found here
_trg_edges = np.array([_trg_cross(_trg_long, 0.0, e.sample - 1) / _trg_fs
                       for e in _trg_seen if e.direction > 0 and e.sample > 3])
_trg_pre, _trg_post, _trg_hold = 200e-9, 600e-9, 5e-6
_trg_take, _trg_ready = [], _trg_pre
for _trg_t in _trg_edges:
    if _trg_t >= _trg_ready:
        _trg_take.append(_trg_t)
        _trg_ready = _trg_t + max(_trg_hold, _trg_pre + _trg_post)
_, _trg_evh = _trg_link().trigger(pre_s=_trg_pre, post_s=_trg_post, level=0.0, hysteresis=0.2,
                                  holdoff_s=_trg_hold, n_segments=20).realize()
_trg_stamps = np.array([e.params["timestamp_s"] for e in _trg_evh])
_trg_skipped = int(np.searchsorted(_trg_edges, _trg_stamps[-1] + 1e-12)) - 20
_trg_dt = float(np.abs(_trg_stamps - np.array(_trg_take[:20])).max())
check("holdoff skips exactly the edges it should, and every segment's timestamp is that "
      "edge's time in the long render",
      _trg_dt * _trg_fs < 1e-6 and np.diff(_trg_stamps).min() >= _trg_hold and _trg_skipped > 20,
      f"20 segments, {_trg_skipped} edges passed over, timestamps against the independent "
      f"edge list: worst {_trg_dt * _trg_fs:.1e} samples, closest pair "
      f"{np.diff(_trg_stamps).min() * 1e6:.3f} us (holdoff 5 us)")

# averaging: a clock, so every trigger sees the same waveform and only the noise differs
_trg_cg = Grid(fs=1e9, n=1000 * 300)
_trg_n, _trg_rms = 64, 0.05


def _trg_clock(**kw):
    return (Signal(seed=7, grid=_trg_cg).carrier("cmos", v_lo=0.0, v_hi=1.0, f_hz=1e6,
                                                 tr_s=5e-9).rx_noise(rms=_trg_rms)
            .trigger(pre_s=50e-9, post_s=400e-9, level=0.5, hysteresis=0.4, **kw))


_trg_one = _trg_clock(n_segments=_trg_n).waveform().reshape(_trg_n, -1)
_trg_avg = _trg_clock(average=_trg_n).waveform()
_trg_flat = slice(100, 440)                                   # the settled high level
_trg_s1 = float(_trg_one[:, _trg_flat].std(1).mean())
_trg_sn = float(_trg_avg[_trg_flat].std())
check("averaging n triggers is the mean of those n windows and cuts uncorrelated noise by "
      "sqrt(n)",
      np.allclose(_trg_avg, _trg_one.mean(0), rtol=0, atol=1e-15)
      and abs(_trg_s1 / _trg_rms - 1) < 0.05
      and abs(_trg_sn / (_trg_rms / np.sqrt(_trg_n)) - 1) < 0.15,
      f"noise {_trg_s1 * 1e3:.2f} mV single, {_trg_sn * 1e3:.3f} mV after {_trg_n} "
      f"(sqrt(n) predicts {_trg_rms / np.sqrt(_trg_n) * 1e3:.3f} mV, bound 15 %)")

# trigger jitter MOVES a record and does not smear it; only averaging smears (BACKLOG #50)
_trg_tr, _trg_sig_j = 20e-9, 6e-9
_trg_pg = Grid(fs=1e9, n=1000 * 300)


def _trg_edge(**kw):
    return (Signal(seed=1, grid=_trg_pg).carrier("cmos", v_lo=0.0, v_hi=1.0, f_hz=1e6,
                                                 tr_s=_trg_tr)
            .trigger(pre_s=100e-9, post_s=100e-9, level=0.5, jitter_s=_trg_sig_j, **kw))


def _trg_rise(y):
    k = np.arange(len(y))
    m = (k > 20) & (k < 180)
    return float(np.interp(0.9, y[m], k[m]) - np.interp(0.1, y[m], k[m])) / 1e9


_trg_src = Signal(seed=1, grid=_trg_pg).carrier("cmos", v_lo=0.0, v_hi=1.0, f_hz=1e6,
                                               tr_s=_trg_tr).waveform()
_trg_i0 = 200 + int(np.argmax((_trg_src[200:-1] < 0.5) & (_trg_src[201:] >= 0.5)))
_trg_r0 = _trg_rise(_trg_src[_trg_i0 - 100:_trg_i0 + 100])     # a rising edge of the source
_trg_r1 = _trg_rise(_trg_edge().waveform())
_trg_shots = _trg_edge(n_segments=256).waveform().reshape(256, -1)
_trg_ra = _trg_rise(_trg_edge(average=256).waveform())
# the average is the single edge convolved with the jitter's Gaussian (and the half-sample
# rounding): do that convolution numerically on the first shot
_trg_u = np.linspace(-6, 6, 2401)
_trg_w = np.exp(-_trg_u ** 2 / 2); _trg_w /= _trg_w.sum()
_trg_k = np.arange(200.0)
_trg_pred = np.array([(_trg_w * np.interp(_trg_k[i] - _trg_u * _trg_sig_j * 1e9, _trg_k,
                                          _trg_shots[0])).sum() for i in range(200)])
_trg_rp = _trg_rise(_trg_pred)
check("trigger jitter moves the window without smearing the record; averaging jittered "
      "triggers smears the edge by the jitter's Gaussian",
      abs(_trg_r1 / _trg_r0 - 1) < 1e-9 and _trg_rp > 1.15 * _trg_r0
      and abs(_trg_ra / _trg_rp - 1) < 0.10,
      f"10-90 % rise: source {_trg_r0 * 1e9:.2f} ns, one jittered record {_trg_r1 * 1e9:.2f} ns, "
      f"average of 256 {_trg_ra * 1e9:.2f} ns against {_trg_rp * 1e9:.2f} ns predicted "
      f"(bound 10 %)")

# no silent untriggered record: the refusal's render length is enough, and one period less is not
_trg_ck = Signal(grid=Grid(fs=1e9, n=20000)).carrier("sine", f_hz=5e6).waveform()
try:
    _TRG.apply(_trg_ck, 1e9, pre_s=100e-9, post_s=300e-9, level=0.0, n_segments=60)
    _trg_msg = ""
except ValueError as _trg_err:
    _trg_msg = str(_trg_err)
_trg_m = _trg_re.search(r"needs about (\d+) samples", _trg_msg)
_trg_need = int(_trg_m.group(1)) if _trg_m else 0
_trg_more = Signal(grid=Grid(fs=1e9, n=max(_trg_need, 1))).carrier("sine", f_hz=5e6).waveform()
_trg_ok = len(_TRG.apply(_trg_more, 1e9, pre_s=100e-9, post_s=300e-9, level=0.0,
                         n_segments=60)) == 60 * 400 if _trg_need else False
try:
    _TRG.apply(_trg_more[:_trg_need - 400], 1e9, pre_s=100e-9, post_s=300e-9, level=0.0,
               n_segments=60)
    _trg_tight = False
except ValueError:
    _trg_tight = True
check("too few triggers is refused with a render length, and that length is enough and "
      "not generous",
      _trg_ok and _trg_tight,
      f"20000 samples held 50 of 60; the refusal asked for {_trg_need}, which triggers 60, "
      f"and {_trg_need - 400} does not")

# events of the long render are re-timed into the window; the rest are dropped and counted
_trg_evg = Grid(fs=2e9, baud=2e9 / 128, n=1024 * 128)


def _trg_marked():
    return (Signal(seed=3, grid=_trg_evg).carrier("nrz", n_ui=1024, pattern="prbs7")
            .events("glitch", on="symbols", count=12).lossy(loss_db=3.0, loss_at_ghz=0.02))


_trg_lx, _trg_lev = _trg_marked().realize()
_trg_wx, _trg_wev = _trg_marked().trigger(pre_s=2e-6, post_s=6e-6, level=0.0,
                                          n_segments=3, holdoff_s=20e-6).realize()
_trg_tr_ev = [e for e in _trg_wev if e.kind == "trigger"]
_trg_kept = [e for e in _trg_wev if e.kind == "glitch"]
_trg_nw = len(_trg_wx) // 3
_trg_want_ev = sorted((t.params["segment"] * _trg_nw + e.sample - t.params["start"])
                      for e in _trg_lev for t in _trg_tr_ev
                      if t.params["start"] <= e.sample < t.params["start"] + _trg_nw)
check("an event of the long render is re-timed into the window that holds it, and those in "
      "no window are dropped and counted rather than left pointing past the record",
      sorted(e.sample for e in _trg_kept) == _trg_want_ev and len(_trg_kept) >= 1
      and _trg_tr_ev[0].params["dropped"] == len(_trg_lev) - len(_trg_kept) >= 1
      and all(0 <= e.sample < len(_trg_wx) for e in _trg_wev)
      and float(_trg_wev.mask().sum()) == float(sum(
          min(e.sample + e.width, len(_trg_wx)) - e.sample for e in _trg_kept)),
      f"{len(_trg_lev)} glitches in the long render: {len(_trg_kept)} inside the 3 windows, "
      f"{_trg_tr_ev[0].params['dropped']} dropped; the 3 trigger events mark no samples")
# ============================================================ I/Q recordings (GitHub #82)
print("\n== I/Q recordings in open formats: interleaved binary and SigMF, in and out ==")
import hashlib as _iqf_hl, json as _iqf_json, tempfile as _iqf_tmp                # noqa: E402
import wfmsynth as _iqf_ws                                                        # noqa: E402
from wfmsynth import iqfile as _IQF                                               # noqa: E402
_iqf_g = Grid(fs=250e6, baud=25e6, n=1 << 13, domain="iq", fc_hz=1e9)
_iqf_x = Signal(seed=1, grid=_iqf_g).iq("qam16", prbs=15).waveform()
_iqf_d = _iqf_tmp.mkdtemp()
_iqf_fs = dict(fs_hz=250e6, fc_hz=1e9)


def _iqf_parts(z):
    return np.c_[z.real, z.imag].ravel()


# float32, against numpy alone in BOTH directions: our writer read by np.fromfile, and a file
# np.tofile wrote read by our reader
_iqf_x32 = _iqf_x.astype(np.complex64).astype(complex)
_IQF.to_interleaved(f"{_iqf_d}/f.iq", _iqf_x32)
_iqf_a = np.array_equal(np.fromfile(f"{_iqf_d}/f.iq", "<f4"), _iqf_parts(_iqf_x32))
_iqf_parts(_iqf_x32).astype(">f4").tofile(f"{_iqf_d}/g.iq")
_iqf_b = np.array_equal(_iqf_ws.read_iq(f"{_iqf_d}/g.iq", format="iq_interleaved",
                                        dtype="float32", byteorder="big",
                                        **_iqf_fs).waveform(), _iqf_x32)
_IQF.to_interleaved(f"{_iqf_d}/h.iq", _iqf_x)
_iqf_e32 = np.abs(_iqf_parts(_iqf_ws.read_iq(f"{_iqf_d}/h.iq", format="iq_interleaved",
                                             dtype="float32", **_iqf_fs).waveform()
                             - _iqf_x)).max() / np.abs(_iqf_x).max()
check("a float32 I/Q file round-trips exactly, checked against numpy's own reader and writer",
      _iqf_a and _iqf_b and _iqf_e32 <= 2.0 ** -24,
      f"float32 samples: 0 error both ways; a float64 record written as float32 moves by "
      f"{_iqf_e32:.1e} of its peak (bound 2**-24 = {2.0 ** -24:.1e})")

_iqf_s = float(np.abs(_iqf_parts(_iqf_x)).max()) / 32767
_IQF.to_interleaved(f"{_iqf_d}/i.iq", _iqf_x, dtype="int16", scale=_iqf_s)
_iqf_cnt = np.fromfile(f"{_iqf_d}/i.iq", "<i2")
_iqf_back = _iqf_ws.read_iq(f"{_iqf_d}/i.iq", format="iq_interleaved", dtype="int16",
                            scale=_iqf_s, **_iqf_fs).waveform()
_iqf_e16 = np.abs(_iqf_parts(_iqf_back - _iqf_x)).max() / _iqf_s
check("an int16 I/Q file holds the record within one LSB (half a count, by rounding)",
      np.array_equal(_iqf_cnt, np.rint(_iqf_parts(_iqf_x) / _iqf_s))
      and _iqf_e16 <= 0.5 + 1e-9 and int(np.abs(_iqf_cnt).max()) == 32767,
      f"worst error {_iqf_e16:.4f} LSB over {_iqf_cnt.size} values (bound 0.5), "
      f"peak {int(np.abs(_iqf_cnt).max())} counts")

# SigMF: the metadata is read back with json alone, and the events through the chain
_iqf_ev = _iqf_ws.EventList([
    _iqf_ws.Event(sample=100, kind="burst_start", width=1, t_s=100 / 250e6, placement="burst/2"),
    _iqf_ws.Event(sample=4000, kind="glitch", width=37, severity=0.8, direction=-1.0,
                  t_s=4000 / 250e6, placement="edges", params={"f0": 1.5e9})], n=len(_iqf_x))
_iqf_o = _IQF.to_sigmf(f"{_iqf_d}/rec", _iqf_x, _iqf_g, events=_iqf_ev, dtype="int16",
                       scale=_iqf_s)
_iqf_m = _iqf_json.load(open(_iqf_o["meta"]))
_iqf_sig = _iqf_ws.read_iq(f"{_iqf_d}/rec", format="sigmf")
_iqf_y, _iqf_got = _iqf_sig.realize()
check("a SigMF recording carries the required fields, and its rate, centre and annotations "
      "come back as the record's grid and events",
      _iqf_m["global"]["core:datatype"] == "ci16_le" and _iqf_m["global"]["core:version"] == "1.0.0"
      and _iqf_m["global"]["core:sha512"] == _iqf_hl.sha512(open(_iqf_o["data"], "rb").read()).hexdigest()
      and [a["core:sample_start"] for a in _iqf_m["annotations"]] == [100, 4000]
      and _iqf_m["annotations"][1]["core:sample_count"] == 37
      and (_iqf_sig.grid.fs, _iqf_sig.grid.fc_hz, _iqf_sig.grid.n) == (250e6, 1e9, len(_iqf_x))
      and _iqf_got.to_dicts() == _iqf_ev.to_dicts() and np.array_equal(_iqf_y, _iqf_back),
      f"{len(_iqf_got)} events equal to the {len(_iqf_ev)} written, field for field; samples "
      f"identical to the interleaved int16 file's")

# the byte-order check, both ways round, at five levels and on two kinds of record
_iqf_rng = np.random.default_rng(0)
_iqf_w = 0.2 * (_iqf_rng.standard_normal(1 << 13) + 1j * _iqf_rng.standard_normal(1 << 13))
_iqf_caught, _iqf_false, _iqf_n = [], 0, 0
for _iqf_v in (_iqf_x, _iqf_w):                    # QAM-16 at 10 samples/symbol; white noise
    for _iqf_db in (0, -6, -20, -40, -60):
        _iqf_c = np.rint(_iqf_parts(_iqf_v) / np.abs(_iqf_parts(_iqf_v)).max()
                         * 32767 * 10 ** (_iqf_db / 20)).astype("<i2")
        _iqf_n += 1
        _iqf_false += _IQF.byte_order_evidence(_iqf_c, _iqf_c.dtype) is not None
        if _IQF.byte_order_evidence(_iqf_c.byteswap(), _iqf_c.dtype) is not None:
            _iqf_caught.append(_iqf_db)
try:
    _iqf_cnt.byteswap().tofile(f"{_iqf_d}/sw.iq")
    _iqf_ws.read_iq(f"{_iqf_d}/sw.iq", format="iq_interleaved", dtype="int16", scale=_iqf_s,
                    **_iqf_fs)
    _iqf_msg = ""
except ValueError as _iqf_err:
    _iqf_msg = str(_iqf_err)
check("a byte-swapped int16 file is refused with the likely cause, and a correctly read one "
      "never is", _iqf_false == 0 and "probably big-endian" in _iqf_msg
      and len(_iqf_caught) == _iqf_n == 10,
      f"{len(_iqf_caught)} of {_iqf_n} swapped files caught (full scale to -60 dB, a shaped "
      f"signal and white noise), {_iqf_false} of {_iqf_n} correct files flagged")

# the digest is over the samples: two formats, one sha256; and embed needs no file afterwards
_IQF.to_interleaved(f"{_iqf_d}/e.iq", _iqf_x, dtype="float64", byteorder="big")
_IQF.to_sigmf(f"{_iqf_d}/e", _iqf_x, _iqf_g, dtype="float64")
_iqf_p = _iqf_ws.read_iq(f"{_iqf_d}/e.iq", format="iq_interleaved", dtype="float64",
                         byteorder="big", embed=True, **_iqf_fs)
_iqf_q = _iqf_ws.read_iq(f"{_iqf_d}/e", format="sigmf")
_iqf_r = _iqf_json.loads(_iqf_json.dumps(_iqf_p.recipe()))
import shutil as _iqf_sh                                                          # noqa: E402
_iqf_sh.rmtree(_iqf_d)
check("the capture digest covers the samples, not the file: two formats give one sha256, and "
      "an embedded recording replays with every file deleted",
      _iqf_p.ops[0]["sha256"] == _iqf_q.ops[0]["sha256"] == _iqf_ws.capture.digest(_iqf_x)
      and np.array_equal(Signal.from_recipe(_iqf_r).waveform(), _iqf_x),
      f"sha256 {_iqf_p.ops[0]['sha256'][:12]}... for big-endian float64 and SigMF cf64_le alike")
# ============================================================ I/Q impairments (GitHub #83)
# Seven ops on a complex baseband record, one effect each (wfmsynth/rf.py). Each is measured
# back off the record by a method that shares no code with the op. The probe is a complex tone
# exactly on an FFT bin, so a bin IS a tone and no window sits between the record and the
# number. Where the measurement is an estimate, its standard error is stated and printed.
print("\nI/Q impairments (GitHub #83)")
import json as _rjson                                                       # noqa: E402
from wfmsynth import Grid as _RGrid, Signal as _RSig, Streams as _RSt       # noqa: E402
from wfmsynth import iq as _RIQ, rf as _RRF                                 # noqa: E402
_rfs, _rn, _rk = 400e6, 1 << 16, 1001
_rg = _RGrid(fs=_rfs, baud=25e6, n=_rn, domain="iq", fc_hz=2.4e9)


def _rtone(grid=_rg):
    return _RSig(seed=3, grid=grid).capture(
        values=np.exp(2j * np.pi * _rk * np.arange(grid.n) / grid.n))


def _rbins(y):
    return np.fft.fft(y) / len(y)


def _rrefused(build):
    try:
        build().waveform()
    except ValueError as e:
        return str(e)
    return ""


_rY = _rbins(_rtone().iq_imbalance(gain_db=0.5, phase_deg=3.0).waveform())
_rimg = 20 * np.log10(abs(_rY[_rk]) / abs(_rY[-_rk]))
_rgl, _rc = 10 ** (0.5 / 20), np.cos(np.deg2rad(3.0))
_rwant = 10 * np.log10((1 + 2 * _rgl * _rc + _rgl ** 2) / (1 - 2 * _rgl * _rc + _rgl ** 2))
_rY = _rbins(_rtone().iq_imbalance(**_RRF.imbalance_for(30)).waveform())
_rimg30 = 20 * np.log10(abs(_rY[_rk]) / abs(_rY[-_rk]))
check("iq_imbalance: the image tone sits where the closed-form rejection ratio puts it, and "
      "imbalance_for(30) makes a 30 dB image",
      abs(_rimg - _rwant) < 1e-9 and abs(_RRF.image_rejection_db(0.5, 3.0) - _rwant) < 1e-9
      and abs(_rimg30 - 30) < 1e-9,
      f"0.5 dB / 3 deg: image {_rimg:.4f} dB down, closed form {_rwant:.4f} dB; "
      f"imbalance_for(30) -> {_rimg30:.6f} dB")

_rcfo = _rtone().freq_offset(f_hz=12345.0, drift_hz_per_s=5e6)
_rph = np.unwrap(np.angle(_rcfo.waveform() * np.conj(_rtone().waveform())))
_rc2, _rc1, _ = np.polyfit(np.arange(_rn) / _rfs, _rph, 2)
_rdc = _rtone().iq_offset(i=0.02, q=-0.01).waveform().mean()
check("freq_offset is the slope and curvature of the phase, lo_phase() is that phase, and "
      "iq_offset is the record's mean",
      abs(_rc1 / (2 * np.pi) - 12345.0) < 1e-6 and abs(_rc2 / np.pi - 5e6) < 1e-2
      and np.abs(_rph - _rcfo.lo_phase()).max() < 1e-12 and abs(_rdc - (0.02 - 0.01j)) < 1e-15,
      f"asked 12345 Hz + 5e6 Hz/s: fitted {_rc1 / (2 * np.pi):.6f} Hz, {_rc2 / np.pi:.4f} Hz/s; "
      f"mean {_rdc:.5f}")

_rprof = [(1e4, -70), (1e5, -90), (1e6, -120), (2e7, -130)]
_rg2 = _RGrid(fs=_rfs, baud=25e6, n=1 << 18, domain="iq", fc_hz=2.4e9)
_rth = np.angle(_rtone(_rg2).phase_noise(_rprof).waveform() * np.conj(_rtone(_rg2).waveform()))
_rf = np.fft.rfftfreq(_rg2.n, 1 / _rfs)[1:]
_rpsd = 2 * np.abs(np.fft.rfft(_rth)[1:]) ** 2 / (_rg2.n * _rfs)          # one-sided, rad^2/Hz
_rl = 2 * 10 ** (_RRF.profile_dbc_hz(_rf, _rprof) / 10)
_rbands = []
for _rlo, _rhi in ((1e4, 1e5), (1e5, 1e6), (1e6, 2e7)):
    _rm = (_rf >= _rlo) & (_rf <= _rhi)
    _rbands.append((int(_rm.sum()), float(np.mean(_rpsd[_rm] / _rl[_rm]))))
check("phase_noise: a periodogram of the phase read off the record is 2*L(f) in every decade, "
      "within 4 standard errors (1/sqrt(bins) each), and empty outside the stated offsets",
      all(abs(r - 1) < 4 / np.sqrt(m) for m, r in _rbands)
      and _rpsd[_rl == 0].max() < 1e-25 * _rpsd[_rl > 0].max(),
      "; ".join(f"{m} bins {10 * np.log10(r):+.2f} dB (s.e. {10 * np.log10(1 + 1 / np.sqrt(m)):.2f})"
                for m, r in _rbands)
      + f"; this record's rms {_rth.std():.4f} rad, the profile integrates to "
        f"{_RRF.phase_rms_rad(_rprof):.4f}")

_rP = np.abs(_rbins(_rtone().awgn(snr_db=20, bw_hz=_rfs / 16).waveform())) ** 2
_rin = np.abs(np.fft.fftfreq(_rn, 1 / _rfs)) <= _rfs / 32
_rin[_rk] = False
_rsnr = 10 * np.log10(_rP[_rk] / _rP[_rin].sum())
_rq = _RSig(seed=5, grid=_rg).iq("qpsk", prbs=15, prbs_phase=100).awgn(es_n0_db=15,
                                                                      ref_power=1.0)
_rsent, _ = _rq.transmitted()
_resn0 = -20 * np.log10(_RIQ.evm(_RIQ.demodulate(_rq.waveform(), _rg)[10:-10], _rsent[10:-10]))
_rse = 10 * np.log10(1 + 1 / np.sqrt(len(_rsent) - 20))
check("awgn hits the stated ratio: SNR in the stated bandwidth read off a tone's spectrum, "
      "Es/N0 read off the error vector of matched-filtered QPSK, each within 4 standard errors",
      abs(_rsnr - 20) < 10 * np.log10(1 + 4 / np.sqrt(_rin.sum())) and abs(_resn0 - 15) < 4 * _rse,
      f"asked 20 dB in fs/16: {_rsnr:.3f} dB over {int(_rin.sum())} bins; asked Es/N0 15 dB: "
      f"{_resn0:.3f} dB over {len(_rsent) - 20} symbols (s.e. {_rse:.3f} dB)")
check("awgn refuses an SNR with no bandwidth, and two statements of one noise density",
      "ambiguous without bw_hz=" in _rrefused(lambda: _rtone().awgn(snr_db=20))
      and "exactly one of" in _rrefused(lambda: _rtone().awgn(snr_db=20, bw_hz=1e6, es_n0_db=3)))

_rY = _rbins(_rtone().spur(f_hz=-3000 * _rfs / _rn, dbc=-45.0).waveform())
_rsp = 20 * np.log10(abs(_rY[-3000]) / abs(_rY[_rk]))
check("spur: its bin against the carrier's is the stated dBc", abs(_rsp + 45.0) < 1e-9,
      f"asked -45 dBc, measured {_rsp:.6f}")

_rdq = []
for _rb in (8, 10, 12, 14):
    _rP = np.abs(_rbins(_rtone().dac(bits=_rb).waveform())) ** 2
    _rdq.append((_rb, 10 * np.log10(_rP[_rk] / np.delete(_rP, _rk).sum())))
_rclip = _rtone().dac(bits=12, headroom_pct=100 / 1.3 - 100, clip="hard").rf_report()[0]
_rasin = 2 * np.arccos(1 / 1.3) / np.pi
check("dac: a full-scale tone's quantisation noise is 6.02 N + 1.76 dB down (bound 0.15 dB), "
      "and one driven 30 % over clips for the arcsine law's share of the time, once asked to",
      all(abs(q - (6.02 * b + 1.76)) < 0.15 for b, q in _rdq)
      and abs(_rclip["clipped_fraction"] - _rasin) < 2e-3
      and "Pass clip='hard'" in _rrefused(lambda: _rtone().dac(bits=12, headroom_pct=-20)),
      "; ".join(f"{b} bits {q:.2f} dB ({q - (6.02 * b + 1.76):+.2f})" for b, q in _rdq)
      + f"; clipped {_rclip['clipped_fraction']:.4f}, arcsine law {_rasin:.4f}")

_rall = (_RSig(seed=5, grid=_RGrid(fs=400e6, baud=25e6, n=1 << 14, domain="iq", fc_hz=2.4e9))
         .iq("qam16", prbs=15).dac(bits=10, headroom_pct=5)
         .iq_imbalance(gain_db=0.3, phase_deg=2.0).iq_offset(i=0.01)
         .phase_noise(profile=[(1e5, -80), (1e6, -100), (1e7, -120)])
         .freq_offset(f_hz=12e3).spur(f_hz=40e6, dbc=-45).awgn(es_n0_db=20))
_rre = _RSig.from_recipe(_rjson.loads(_rjson.dumps(_rall.recipe())))
_ra, _rbb = _rall.waveform(), _rall.contrast("phase_noise/4")
check("the whole impaired chain replays bit-identically from its recipe, and its oscillator "
      "and its noise are separate roles",
      np.array_equal(_rre.waveform(), _ra) and _rre.sha256() == _rall.sha256()
      and _rall.roles() == ["phase_noise/4", "awgn/7"] and np.abs(_ra - _rbb).max() > 1e-3
      and np.array_equal(_rall.contrast("awgn/7", seed=5), _ra),
      f"roles {_rall.roles()}; re-rolling the oscillator moved the record by "
      f"{np.abs(_ra - _rbb).max():.3f}")
print("== RF measures against ground truth: spectrum, CCDF, ACLR and an EVM that hides "
      "nothing (GitHub #84) ==")
from wfmsynth import rf as _RF                                              # noqa: E402
_wg = Grid(fs=400e6, baud=25e6, n=1 << 16, domain="iq", fc_hz=1e9)
_wt = np.arange(_wg.n) / _wg.fs
_wsig = Signal(seed=1, grid=_wg).iq("qam16", prbs=15, prbs_phase=100)
_wx, _wsent = _wsig.waveform(), _wsig.transmitted()[0]
_wrng = np.random.default_rng(84)


def _wnoise(n, var):
    return np.sqrt(var / 2) * (_wrng.standard_normal(n) + 1j * _wrng.standard_normal(n))


_wa, _wf0 = 0.7, 12.3456e6
_wsp = _RF.spectrum(_wa * np.exp(2j * np.pi * _wf0 * _wt), _wg, rbw_hz=100e3, window="flattop")
_wk = int(np.argmax(_wsp["psd_per_hz"]))
_wpk = _wsp["psd_per_hz"][_wk] * _wsp["rbw_hz"] / _wa ** 2
_wbp = _RF.band_power(_wsp, _wf0 - 1e6, _wf0 + 1e6) / _wa ** 2
check("a tone's spectrum peaks at its frequency, and its power is its amplitude squared -- "
      "integrated over a band, and read off the peak as density times resolution bandwidth",
      abs(_wsp["f_hz"][_wk] - _wf0) <= _wsp["bin_hz"] and abs(_wbp - 1) < 1e-6
      and abs(_wpk - 1) < 2e-3,
      f"peak at {_wsp['f_hz'][_wk] / 1e6:.4f} MHz (tone {_wf0 / 1e6:.4f}, bin "
      f"{_wsp['bin_hz'] / 1e3:.1f} kHz); band power / A^2 = 1{_wbp - 1:+.1e} (bound 1e-6); "
      f"peak x rbw / A^2 = {_wpk:.4f} (bound 2e-3, flat-top, rbw {_wsp['rbw_hz'] / 1e3:.1f} kHz)")
_wlv = np.array([0.0, 3.0, 6.0, 8.0])
_wcc = _RF.ccdf(_wnoise(1 << 20, 1.0), levels_db=_wlv)
_wlaw = np.exp(-10 ** (_wlv / 10))
_wse = np.sqrt(_wlaw * (1 - _wlaw) / _wcc["n"])            # a count's own standard error
_wce = float((np.abs(_wcc["probability"] - _wlaw) / _wse).max())
check("the CCDF of complex Gaussian noise is the exponential law, P(power > r x mean) = exp(-r)",
      _wce < 4.0,
      f"at +0/3/6/8 dB: measured {np.array2string(_wcc['probability'], precision=5)} against "
      f"{np.array2string(_wlaw, precision=5)}; worst miss {_wce:.2f} standard errors of the "
      f"count (bound 4; 2^20 samples, so {100 * _wse[-1] / _wlaw[-1]:.1f} % at +8 dB)")
_wother = Signal(seed=1, grid=_wg).iq("qam16", prbs=15, prbs_phase=5000).waveform()
_winj = 10 ** (-40 / 20) * _wother * np.exp(2j * np.pi * 40e6 * _wt)
_wwant = 10 * np.log10(np.mean(np.abs(_wx) ** 2) / np.mean(np.abs(_winj) ** 2))
_wac = _RF.aclr(_wx + _winj, _wg, channel_bw_hz=1.35 * 25e6, offsets_hz=[-40e6, 40e6])
check("ACLR reads a neighbour injected at a known level, and the empty side reads the source's "
      "own skirt", abs(_wac["aclr_db"][1] - _wwant) < 0.1 and _wac["aclr_db"][0] > 55,
      f"injected {_wwant:.2f} dB below the channel, measured {_wac['aclr_db'][1]:.2f} dB "
      f"(bound 0.1 dB); the other side {_wac['aclr_db'][0]:.1f} dB; rbw "
      f"{_wac['rbw_hz'] / 1e3:.0f} kHz")
_wclean = _RF.evm(_wx, _wg, _wsent)
check("EVM of a clean record against the transmitted symbols is below the stated floor, with "
      "nothing fitted and no symbol left out",
      _wclean["evm"] < 0.005 and _wclean["removed"] == {} and _wclean["skip"] == 0,
      f"{_wclean['evm_pct']:.3f} % over {_wclean['n_symbols']} symbols (floor 0.5 %: the "
      f"16-symbol truncation of the pulse, and the burst's two ends)")
_wawgn = []
for _wdb in (30.0, 20.0, 10.0):
    _wn = _wnoise(_wg.n, _wg.fs / (_wg.baud * 10 ** (_wdb / 10)))
    _wawgn.append((_wdb, _RF.evm(_wx + _wn, _wg, _wsent, skip=8),
                   _RF.evm(_wx + _wn, _wg, _wsent, skip=8, reference="decided",
                           constellation="qam16")))
check("EVM against truth in white noise is the analytic 10^(-EsN0/20), at 30, 20 and 10 dB",
      all(abs(_t["evm"] / 10 ** (-_d / 20) - 1) < 0.04 for _d, _t, _ in _wawgn),
      "; ".join(f"{_d:.0f} dB: {_t['evm_pct']:.2f} % against {100 * 10 ** (-_d / 20):.2f} %"
                for _d, _t, _ in _wawgn) + " (bound 4 %, 4080 symbols)")
_wd10, _wt10, _wb10 = _wawgn[-1]
check("  ... and a DECIDED reference reads low once decisions fail, by an amount only the "
      "truth can state", _wb10["decision_errors"] > 500 and _wb10["evm"] < 0.9 * _wt10["evm"]
      and _wawgn[0][2]["decision_errors"] == 0
      and abs(_wawgn[0][2]["evm"] - _wawgn[0][1]["evm"]) < 1e-12,
      f"at 10 dB: decided {_wb10['evm_pct']:.2f} % against truth {_wt10['evm_pct']:.2f} %, "
      f"{_wb10['decision_errors']} of {_wb10['n_symbols']} decisions wrong; at 30 dB none, and "
      f"the two agree")
_wimp = 0.8 * np.exp(0.3j) * _wx
_wraw = _RF.evm(_wimp, _wg, _wsent, skip=8)
_wfit = _RF.evm(_wimp, _wg, _wsent, skip=8, correct=("gain", "phase"))
check("EVM does not hide an impairment: a gain of 0.8 and a 0.3 rad rotation count in full "
      "unless named, and naming them returns what was removed",
      abs(_wraw["evm"] / abs(0.8 * np.exp(0.3j) - 1) - 1) < 0.01 and _wraw["removed"] == {}
      and _wfit["evm"] < 0.005 and abs(_wfit["removed"]["gain_db"] - 20 * np.log10(0.8)) < 1e-3
      and abs(_wfit["removed"]["phase_rad"] - 0.3) < 1e-4,
      f"raw {_wraw['evm_pct']:.2f} % (|0.8 e^(0.3j) - 1| = "
      f"{100 * abs(0.8 * np.exp(0.3j) - 1):.2f} %); corrected {_wfit['evm_pct']:.3f} % with "
      f"removed gain {_wfit['removed']['gain_db']:+.4f} dB and phase "
      f"{_wfit['removed']['phase_rad']:+.5f} rad")
_wtau = 0.23 / _wg.baud
_wlate = np.fft.ifft(np.fft.fft(_wx) * np.exp(-2j * np.pi * np.fft.fftfreq(_wg.n, 1 / _wg.fs)
                                              * _wtau))
_wtim = _RF.evm(_wlate, _wg, _wsent, skip=8, correct=("timing",))
_wtraw = _RF.evm(_wlate, _wg, _wsent, skip=8)
check("  ... and a record 0.23 symbol late is found late by that much",
      _wtraw["evm"] > 0.25 and _wtim["evm"] < 0.005
      and abs(_wtim["removed"]["timing_s"] - _wtau) * _wg.baud < 1e-3,
      f"raw {_wtraw['evm_pct']:.1f} %; with timing named {_wtim['evm_pct']:.3f} %, removed "
      f"{_wtim['removed']['timing_s'] * 1e9:.4f} ns of {_wtau * 1e9:.4f} ns (bound 1e-3 symbol)")
_wpg = Grid(fs=8e9, baud=25e6, n=1 << 18, domain="iq", fc_hz=1e9)
_wps = (Signal(seed=1, grid=_wpg).iq("qam16", prbs=15, prbs_phase=100).upconvert()
        .downconvert(fc_hz=1e9, lpf=dict(pass_hz=20e6, stop_hz=60e6), fs_out_hz=250e6))
_wpx = _wps.waveform()
_wpe = _RF.evm(_wpx, _wps.stored_grid(_wpx), _wps.transmitted()[0], skip=8)
check("the same EVM on a real passband record, through downconvert at a decimated rate",
      _wpe["evm"] < 0.005 and _wpe["removed"] == {},
      f"{_wpe['evm_pct']:.3f} % over {_wpe['n_symbols']} symbols at "
      f"{_wps.stored_grid(_wpx).fs / 1e6:.0f} MSa/s, nothing fitted (floor 0.5 %)")

print("\n== the seams between the pieces: one trigger, one oscillator, an envelope and a "
      "voltage in one record ==")
import wfmsynth as _sws
from wfmsynth import trigger as _STRIG
_sg = Grid(fs=320e9, baud=10e9, n=1 << 14)
_slanes = {"clk_line": Signal().carrier("square", f_hz=5e9).lossy(loss_db=2.0, loss_at_ghz=5.0),
           "tx": Signal().carrier("nrz", n_ui=512, seed=5).lossy(loss_db=4.0, loss_at_ghz=5.0)}
_srec = _sws.Channels(
    {"clk": Signal().use("clk_line").use("scope"), "data": Signal().use("tx").use("scope")},
    shared=dict(_slanes, scope=Signal().trigger(on=Signal().use("clk_line"), pre_s=50e-12,
                                                post_s=350e-12, level=0.0, n_segments=6,
                                                holdoff_s=4e-9)), seed=3, grid=_sg)
_sx, _sev = _srec.realize()
_slong = _sws.Channels({"clk": Signal().use("clk_line"), "data": Signal().use("tx")},
                       shared=_slanes, seed=3, grid=_sg).waveforms()
_sfire = _STRIG.find(_slong["clk"], _sg.fs, level=0.0)["t_s"]
_sst = [(e.params["start"], e.params["timestamp_s"]) for e in _sev["data"] if e.kind == "trigger"]
_snw = len(_sx["data"]) // 6
_sworst = max(float(np.min(np.abs(_sfire - t))) for _a, t in _sst) * _sg.fs
check("a trigger taken from the clock cuts the DATA at the clock's crossings: every window is "
      "the long render of that channel, sliced where an independent search finds a clock edge",
      len(_sst) == 6 and _sworst < 1e-9
      and all(np.array_equal(_sx[k][j * _snw:(j + 1) * _snw], _slong[k][a:a + _snw])
              for k in _sx for j, (a, _t) in enumerate(_sst)),
      f"6 windows of {_snw} samples in each of 2 channels equal the long render exactly; "
      f"trigger times are clock crossings to {_sworst:.1e} samples")

_sgi = Grid(fs=400e6, baud=5e6, n=1 << 15, domain="iq", fc_hz=60e6)
_sprof = [(2e4, -70.0), (2e5, -90.0), (2e6, -110.0)]
_slo = _sws.Channels(
    {"a": Signal().iq("qpsk", prbs=9, prbs_seed=3).use("lo"),
     "b": Signal().iq("qpsk", prbs=9, prbs_seed=5).use("lo")},
    shared={"lo": Signal().phase_noise(profile=_sprof)}, seed=2, grid=_sgi).waveforms()
_sown = _sws.Channels(
    {"a": Signal().iq("qpsk", prbs=9, prbs_seed=3).phase_noise(profile=_sprof),
     "b": Signal().iq("qpsk", prbs=9, prbs_seed=5).phase_noise(profile=_sprof)},
    seed=2, grid=_sgi).waveforms()
_scl = {k: Signal(seed=2, grid=_sgi).iq("qpsk", prbs=9, prbs_seed=v).waveform()
        for k, v in (("a", 3), ("b", 5))}
_sbusy = (np.abs(_scl["a"]) > 0.2) & (np.abs(_scl["b"]) > 0.2)
_sturn = {k: np.angle(_slo[k][_sbusy] / _scl[k][_sbusy]) for k in "ab"}
_sind = {k: np.angle(_sown[k][_sbusy] / _scl[k][_sbusy]) for k in "ab"}
_srms = float(_sturn["a"].std())
_sdiff = float(np.abs(_sturn["a"] - _sturn["b"]).max())
_sdind = float((_sind["a"] - _sind["b"]).std())
check("one oscillator shared by name turns two different envelopes by the same phase, and an "
      "oscillator each does not",
      _sdiff < 1e-9 * _srms and _sdind > 0.5 * _srms,
      f"shared: {_srms * 1e3:.2f} mrad rms on each channel, largest difference between the "
      f"two {_sdiff:.1e} rad; an oscillator each: the two differ by {_sdind * 1e3:.2f} mrad "
      f"rms (bound: more than half the shared rms)")

_senv = Signal(seed=1, grid=_sgi).iq("qpsk", prbs=9, prbs_seed=3)
_snb = Signal(seed=1, grid=_sgi).iq("qpsk", prbs=9, prbs_seed=11).freq_offset(f_hz=20e6)
_srip = Signal().carrier("sine", f_hz=40e3, amp=0.2, offset=1.0)
_sboth = Signal(seed=1, grid=_sgi).iq("qpsk", prbs=9, prbs_seed=3).combine(
    _snb, gains=[0.1]).gain(by=_srip)
_sy = _sboth.waveform()
from wfmsynth import iq as _SIQ
_srr = _sboth.subchain("gain/2/by")
_swant = (_SIQ.upconvert(_senv.waveform(), 60e6, 400e6)
          + 0.1 * _SIQ.upconvert(_snb.waveform(), 60e6, 400e6)) * _srr
_serr = float(np.abs(_SIQ.upconvert(_sy, 60e6, 400e6) - _swant).max())
_ssp = _sws.rf.spectrum(_senv.combine(_snb, gains=[0.1]).waveform(), _sgi, rbw_hz=200e3)
_sratio = 10 * np.log10(_sws.rf.band_power(_ssp, 16e6, 24e6) / _sws.rf.band_power(_ssp, -4e6, 4e6))
check("on an I/Q record a `combine` sum and a `gain` by a real supply are what they are on the "
      "passband: an interferer 20 MHz up and 20 dB down, the whole scaled by a 20 % ripple",
      _serr < 1e-12 and _srr.dtype == np.float64 and abs(_sratio + 20.0) < 0.5,
      f"upconverted envelope against the same sum and gain done on the real passband records: "
      f"max difference {_serr:.1e}; neighbour measured {_sratio:.2f} dB against the channel "
      f"(asked -20, bound 0.5 dB)")

_sug = Grid(fs=20e9, n=1 << 13)


def _slane(group=None):
    return (Signal().carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=5e6, tr_s=0.2e-9)
            .probe(r_source=_sws.tol(1e3, 20, group=group), c_load_f=2e-12))


def _srise(x):
    a = np.flatnonzero((x[:-1] < 0.33) & (x[1:] >= 0.33))[0]
    b = a + np.flatnonzero(x[a:] >= 2.97)[0]
    return ((b - 1 + (2.97 - x[b - 1]) / (x[b] - x[b - 1]))
            - (a + (0.33 - x[a]) / (x[a + 1] - x[a]))) / _sug.fs


_surec = _sws.Channels({"a": _slane(), "b": _slane(), "c": _slane("pack"), "d": _slane("pack")},
                       seed=7, grid=_sug)
_sur = {k: _surec.chains[k][1]["r_source"] for k in "abcd"}
_suw = {u: r.waveforms() for u, r in (("this", _surec), ("other", _surec.unit(8)))}
_sur2 = {k: _surec.unit(8).chains[k][1]["r_source"] for k in "abcd"}
_suerr = max(abs(_srise(_suw[u][k]) / (np.log(9.0) * rr[k] * 2e-12) - 1)
             for u, rr in (("this", _sur), ("other", _sur2)) for k in "abcd")
check("a toleranced part in a several-channel record: the drawn resistor is the one in each "
      "channel's edge, a group is ONE part across channels, and unit() draws them all again",
      _suerr < 5e-3 and _sur["c"] == _sur["d"] and len({_sur["a"], _sur["b"], _sur["c"]}) == 3
      and all(_sur2[k] != _sur[k] for k in "abcd") and _sur2["c"] == _sur2["d"],
      f"10-90 % rise against ln(9) R C from the recipe, 4 channels x 2 units: worst "
      f"{_suerr:.1e} (bound 5e-3); this unit {', '.join(f'{_sur[k]:.0f}' for k in 'abcd')} ohm "
      f"(c and d share group 'pack'), the other "
      f"{', '.join(f'{_sur2[k]:.0f}' for k in 'abcd')} ohm")

# ---- `square` / `cmos`: every edge at its own time, to a fraction of a sample (0.46.0)
# The instrument: a degree-7 polynomial through the 8 samples about each crossing, which shares
# nothing with how the edge was generated. It is calibrated on the known answer first -- the
# train as it was built before 0.46.0, `where(phase < duty)` on the samples, which must read one
# whole sample peak to peak.
_efs, _ef, _en = 64e9, 0.997e9, 1 << 15
_espp = _efs / _ef


def _etie(x, level):
    s = np.asarray(x, float) - level
    i = np.flatnonzero((s[:-1] < 0) != (s[1:] < 0))
    i = i[(i > 0.1 * s.size) & (i < 0.9 * s.size)]
    w = np.arange(-3, 5)
    pp = []
    for sign in (1, -1):
        t = []
        for k in i[np.sign(s[i + 1] - s[i]) == sign]:
            r = np.polynomial.Polynomial.fit(w, s[k + w], 7).convert().roots()
            r = r[np.abs(r.imag) < 1e-9].real
            t.append(k + r[(r >= -1e-6) & (r <= 1 + 1e-6)][0])
        t = np.asarray(t)
        e = t - np.round((t - t[0]) / _espp) * _espp
        pp.append(np.ptp(e) / _efs * 1e12)
    return max(pp)


_eg = Grid(fs=_efs, n=_en)
_eold = np.where((_ef * np.arange(_en) / _efs) % 1.0 < 0.5, 1.0, -1.0)
_ek = 0.15 * _espp
_ebefore = _etie(P._shape_edges(_eold, _ek), 0.0)
_esq = _etie(Signal(seed=1, grid=_eg).carrier("square", f_hz=_ef, tr_frac=0.15).waveform(), 0.0)
_ecm = _etie(Signal(seed=1, grid=_eg).carrier("cmos", v_lo=0.0, v_hi=3.3, f_hz=_ef,
                                              tr_s=_ek / _efs).waveform(), 1.65)
_eduty = P.square(f_hz=1e9, duty=0.3, phase_rad=1.5 * np.pi, tr_frac=0.05, n=8192, fs=256e9)
_eon = P.square(f_hz=10e6, n=4096, fs=10e9)
_eon_old = P._shape_edges(np.where((10e6 * np.arange(4096) / 10e9) % 1.0 < 0.5, 1.0, -1.0),
                          0.05 * 1000)
check("a square and a CMOS clock that do not fit the grid carry no jitter of the grid's: each "
      "edge is at its own time between samples, the duty is the one asked for, and a clock "
      "whose edges are on samples is the record it was",
      14.0 < _ebefore < 16.5 and _esq < 1e-3 and _ecm < 1e-3
      and abs(_eduty.mean() + 0.4) < 1e-9 and np.array_equal(_eon, _eon_old),
      f"0.997 GHz at 64 GSa/s, edge-time error peak to peak by polynomial crossing: "
      f"{_ebefore:.2f} ps as built before 0.46.0 (one sample is 15.63), square {_esq:.1e} ps, "
      f"cmos {_ecm:.1e} ps (bound 1e-3); duty 0.3 over 32 periods reads "
      f"{(_eduty.mean() + 1) / 2:.12f}; 10 MHz at 10 GSa/s is bit-identical to the old "
      f"construction")

# ---- roles(): the list is what the record draws from, in both directions (0.46.0)
from wfmsynth.streams import Streams as _RSt
from wfmsynth.acquire import AcquisitionProfile as _RAP
_rchain = (Signal(seed=3, grid=Grid(fs=64e9, baud=16e9, n=4096))
           .carrier("nrz", n_ui=1024, jitter=dict(rj=0.4)).nonlinearity(level_noise=0.05)
           .crosstalk(coupling=0.1)                          # default aggressor: draws nothing
           .acquire(_RAP(sample_rate_hz=32e9, record_length=2048, input_bandwidth_hz=13e9,
                         noise_floor=dict(rms=0.01)))
           .store(bits=8, dither_lsb=0.5).store(bits=8))     # the second has no dither
_rst = _RSt(_rchain.seed)
_rx = _rchain.waveform(streams=_rst)
_rmoved = [r for r in _rchain.roles() if not np.array_equal(_rx, _rchain.contrast(r, seed=9))]
check("roles() lists exactly what the record draws from: every listed role was drawn from and "
      "moves samples when re-rolled, and nothing was drawn under a name that is not listed",
      _rchain.roles() == ["jitter/0", "nl_noise/1", "acquire/3", "store/4"]
      and _rst.drawn() == set(_rchain.roles()) and _rmoved == _rchain.roles(),
      f"listed {_rchain.roles()}; drawn (generators that moved) {sorted(_rst.drawn())}; "
      f"asked for and not drawn {sorted(_rst.asked - _rst.drawn())}; "
      f"{len(_rmoved)} of {len(_rchain.roles())} re-rolls changed samples")

# ---- rf.evm(reference="decided"): a carrier is acquired before deciding, or refused (0.46.0)
# The reference is the truth-referenced EVM, which uses the transmitted symbols and none of
# the blind estimator. Before this, the 2 kHz record below returned 24.5 % / 24.9 %.
_vg = Grid(fs=400e6, baud=25e6, n=1 << 16, fc_hz=2.4e9)


def _vacq(f_hz, es_n0_db):
    s = Signal(seed=1, grid=_vg).iq("qam16", prbs=15, prbs_phase=100).freq_offset(f_hz=f_hz)
    if es_n0_db is not None:
        s = s.awgn(es_n0_db=es_n0_db, ref_power=1.0)
    x, sent = s.waveform(), s.transmitted()[0]
    kw = dict(skip=16, correct=("frequency", "phase"))
    return (_sws.rf.evm(x, _vg, sent, **kw),
            _sws.rf.evm(x, _vg, sent, reference="decided", constellation="qam16", **kw))


_vrep = [_vacq(2e3, e) for e in (None, 24)]
_vsweep = [(f, _vacq(f, 24)) for f in (-3.0e6, -1.234e6, -2e4, 150.0, 2e5, 2e6, 3.0e6)]
_vworst = max(abs(b["evm_pct"] - t["evm_pct"]) for _, (t, b) in _vsweep)
_vferr = max(abs(b["removed"]["frequency_hz"] - f) for f, (_, b) in _vsweep)
_vrng = np.random.default_rng(0)
_vnoise = _vrng.standard_normal(1 << 16) + 1j * _vrng.standard_normal(1 << 16)
_vref = _qrefusal(lambda: _sws.rf.evm(_vnoise, _vg, reference="decided", constellation="qam16",
                                      correct=("frequency",)))
check("a decided-reference EVM acquires its carrier blind before deciding: a QAM-16 record "
      "2 kHz off reads what the truth-referenced EVM reads, any offset within +/- baud/8 is "
      "found, and a record with no carrier to find is refused, not measured",
      all(abs(b["evm_pct"] - t["evm_pct"]) < 0.01 and b["decision_errors"] == 0
          for t, b in _vrep)
      and _vworst < 0.02 and _vferr < 25.0 and "did not lock" in _vref
      and "No figure is returned" in _vref,
      f"2 kHz offset, decided against truth: {_vrep[0][1]['evm_pct']:.2f} % / "
      f"{_vrep[0][0]['evm_pct']:.2f} % clean, {_vrep[1][1]['evm_pct']:.2f} % / "
      f"{_vrep[1][0]['evm_pct']:.2f} % at Es/N0 24 dB (carrier line "
      f"{_vrep[1][1]['acquisition']['line']:.1f}, {_sws.rf.LOCK_LINE:g} needed); offsets "
      f"-3 .. +3 MHz at 24 dB: worst EVM difference {_vworst:.3f} points, worst frequency "
      f"error {_vferr:.1f} Hz (pull-in +/-{_vrep[0][1]['acquisition']['pull_in_hz'] / 1e6:.3f}"
      f" MHz); noise alone: refused")

print()
if fails:
    print(f"VALIDATION FAILED: {len(fails)} checks -> {fails}")
    sys.exit(1)
print("ALL PHYSICS CHECKS PASSED")


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
    check(f"seed {_sd}: PRBS7 eye is OPTIMISTIC vs PRBS31 through the same 14in channel",
          _e7 > _e31 * 1.10,
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
      _eh(_tb40, _g40) < _eh(_x40, _g40) - 0.02, f"eye {_eh(_x40,_g40):.3f} -> {_eh(_tb40,_g40):.3f}")

print("== Tx de-emphasis preset: a dB preset yields that transition/steady ratio ==")
_spb42 = 16
_taps42 = P.de_emphasis_taps(3.5)
_sq = np.repeat(np.array([-1.0]*8 + [1.0]*8), _spb42).astype(float)     # 8-UI runs, sharp edges
_y42 = P.tx_ffe(_sq, _taps42, _spb42, pre=0)
_lo_run = 8 * _spb42                                               # index where the high run starts
_emph = _y42[_lo_run + _spb42 // 2]                                # first UI after the transition
_steady = _y42[_lo_run + 5 * _spb42 + _spb42 // 2]                 # a mid-run (steady) UI
check("de_emphasis_taps(3.5 dB) yields ~3.5 dB transition-to-steady ratio",
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
      _fold(_lf29, _g29) > _eh(_lf29, _g29) + 0.02,
      f"fixed {_eh(_lf29,_g29):.3f} -> folded {_fold(_lf29,_g29):.3f}")
check("high-frequency jitter is NOT tracked out (folded ~ fixed-grid eye)",
      abs(_fold(_hf29, _g29) - _eh(_hf29, _g29)) < 0.05)

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
check("nonmonotonic edge increases slope reversals in its window",
      _srev(_ynm[_n0:_n1]) > _srev(_xev[_n0:_n1]))

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
# Real measured backplanes (wfmplan spike `real_channel`, from the 802.3ap/Molex/TE files) run
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
_gst = Grid(fs=_fs_store, baud=16e9, n=1 << 16, v_full=0.846)
_base = (Signal(seed=17, grid=_gst).carrier(kind="nrz", pattern="prbs13")
         .lossy(loss_db=12.0, loss_at_ghz=8.0, causal=True)
         .scope(bw_hz=110e9).digitize(noise_rms=0.0032, enob=10.0).scope(bw_hz=32e9))
_smeared = _base.waveform()
_stored = (Signal(seed=17, grid=_gst).carrier(kind="nrz", pattern="prbs13")
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
            .carrier(kind="nrz", pattern="prbs13").lossy(loss_db=12.0, loss_at_ghz=8.0, causal=True)
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
from wfmsynth.compose import _LEAD_LTI, _LEAD_SKIP, _LEAD_REJECT, _EXEC as _LEXEC
_lunc = sorted(set(_LEXEC) - (_LEAD_LTI | _LEAD_SKIP | set(_LEAD_REJECT)))
check("every composable op is CLASSIFIED for the lead-in (has an impulse response / has none / "
      "cannot be guarded), so a new op cannot default into the safe-looking pile",
      not _lunc, f"{len(_LEXEC)} ops: {len(_LEAD_LTI)} sized, {len(_LEAD_SKIP)} skipped, "
                 f"{len(_LEAD_REJECT)} refused" + (f"; UNCLASSIFIED {_lunc}" if _lunc else ""))


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

print()
if fails:
    print(f"VALIDATION FAILED: {len(fails)} checks -> {fails}")
    sys.exit(1)
print("ALL PHYSICS CHECKS PASSED")


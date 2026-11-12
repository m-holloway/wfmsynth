"""A swept-wavelength record (`wfmsynth.swept`, GitHub #93 and #94).

A swept laser's field is held in the frame of its own moving carrier; `delay` on it turns
the carrier by its integral over the delay, so an interferometer and a stray reflection are
chains (`wfmsynth.recipes`); a resonator is solved in that frame. Each effect is asserted
against something that does not come from the code under test: a closed form written out
here, scipy's ODE integrator, the Faddeeva function, the textbook ring formula.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import wofz

import wfmsynth as ws
from wfmsynth import Grid, Signal
from wfmsynth import measure as M
from wfmsynth import swept as SW

R = ws.recipes
C_NM = 299792458.0e9                                  # c in nm/s: nu = C_NM / lambda_nm
FS, N = 1e6, 1 << 17                                  # 0.131 s at 1 MSa/s
G = Grid(fs=FS, n=N)
T = np.arange(N) / FS
DUR = N / FS
RATE = (C_NM / 1549.0 - C_NM / 1551.0) / DUR          # mean |d nu/dt| of the sweep below


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def laser(grid=G, **kw):
    kw = kw or dict(start_nm=1549.0, stop_nm=1551.0)
    return Signal(seed=1, grid=grid).laser(**kw)


def lam_of(sig):
    return SW.wavelength_map(sig)["wavelength_nm"]


def pd(sig):
    return sig.copy().photodetect(shot=False).waveform()


def lorentzian(lam, lam0, ki, ke):
    """|1 - ke/(k/2 + j*D)|^2, written out: a dip `ki + ke` Hz wide."""
    d = C_NM / lam - C_NM / lam0
    return np.abs(1.0 - ke / (0.5 * (ki + ke) + 1j * d)) ** 2


def delay_of_fringe(x, nu):
    """The delay a fringe in `x` against optical frequency `nu` stands for: the peak of its
    spectrum on a uniform frequency axis, in seconds."""
    order = np.argsort(nu)
    un = np.linspace(nu.min(), nu.max(), len(nu))
    xi = np.interp(un, nu[order], x[order])
    sp = np.abs(np.fft.rfft((xi - xi.mean()) * np.hanning(len(xi))))
    k = int(sp.argmax())
    k = k + 0.5 * (sp[k - 1] - sp[k + 1]) / (sp[k - 1] - 2 * sp[k] + sp[k + 1])
    return k / (len(xi) * (un[1] - un[0]))


# ---------------------------------------------------------------- the laser and its map
def test_a_laser_is_a_constant_field_and_the_sweep_is_in_its_carrier():
    sig = laser()
    x = sig.waveform()
    assert x.dtype == np.complex128 and np.array_equal(x, np.ones(N, complex))
    assert sig.grid.domain == "optical" and not G.domain_stated
    assert sig.stage_kinds(domains=True) == [("source", "optical")]
    assert sig.roles() == []                                   # it draws nothing
    m = SW.wavelength_map(sig)
    assert np.array_equal(m["t_s"], T)
    assert m["wavelength_nm"][0] == 1549.0
    assert abs(m["wavelength_nm"][-1] - (1551.0 - 2.0 / N)) < 1e-9     # stop is at t = N/fs
    assert np.allclose(m["f_hz"], C_NM / m["wavelength_nm"], rtol=1e-15)
    assert np.allclose(np.abs(laser(start_nm=1550.0, power=4.0).waveform()), 2.0)


def test_the_map_is_the_stated_sweep_term_by_term():
    lin = 1549.0 + 2.0 * T / DUR
    assert np.allclose(lam_of(laser(start_nm=1549.0, speed_nm_per_s=2.0 / DUR)), lin,
                       atol=1e-9)
    assert np.all(lam_of(laser(start_nm=1550.0)) == 1550.0)            # standing still
    # stop and speed: the sweep ends early and the laser parks
    park = lam_of(laser(start_nm=1549.0, stop_nm=1550.0, speed_nm_per_s=20.0))
    k = int(round(0.05 * FS))
    assert np.allclose(park[:k], 1549.0 + 20.0 * T[:k], atol=1e-9)
    assert np.all(park[k:] == 1550.0)
    # the bow peaks at mid-sweep by bow_nm; the ripple is its own sine
    bow = lam_of(laser(start_nm=1549.0, stop_nm=1551.0, bow_nm=0.02)) - lin
    assert abs(bow.max() - 0.02) < 1e-9 and abs(int(bow.argmax()) - N // 2) <= 1
    assert abs(bow[0]) < 1e-12
    rip = lam_of(laser(start_nm=1549.0, stop_nm=1551.0, ripple_nm=3e-3, ripple_hz=40.0))
    assert np.allclose(rip - lin, 3e-3 * np.sin(2 * np.pi * 40.0 * T), atol=1e-9)
    # a downward sweep
    assert np.allclose(lam_of(laser(start_nm=1551.0, stop_nm=1549.0)), 1551.0 - 2.0 * T / DUR,
                       atol=1e-9)


def test_a_mode_hop_is_a_step_in_the_map_and_an_event():
    sig = laser(start_nm=1549.0, stop_nm=1551.0, hops=[{"t_s": 0.05, "by_nm": 0.03},
                                                       {"t_s": 0.09, "by_nm": -0.01}])
    step = lam_of(sig) - (1549.0 + 2.0 * T / DUR)
    k1, k2 = int(0.05 * FS), int(0.09 * FS)
    assert np.all(step[:k1] == 0) and np.allclose(step[k1:k2], 0.03, atol=1e-12)
    assert np.allclose(step[k2:], 0.02, atol=1e-12)
    x, ev = sig.realize()
    assert np.array_equal(x, np.ones(N, complex))             # the field itself is continuous
    hops = [e for e in ev if e.kind == "mode_hop"]
    assert [(e.sample, e.t_s, e.params["by_nm"], e.direction, e.placement) for e in hops] == [
        (k1, 0.05, 0.03, 1.0, "laser/0"), (k2, 0.09, -0.01, -1.0, "laser/0")]
    assert abs(hops[0].params["wavelength_nm"] - lam_of(sig)[k1]) < 1e-12


LASER_KNOBS = [dict(start_nm=1549.2), dict(stop_nm=1550.9), dict(bow_nm=0.01),
               dict(ripple_nm=1e-3, ripple_hz=30.0), dict(power=2.0),
               dict(hops=[{"t_s": 0.03, "by_nm": 0.02}])]


@pytest.mark.parametrize("knob", LASER_KNOBS, ids=lambda k: next(iter(k)))
def test_every_laser_parameter_moves_the_record_and_its_digest(knob):
    base = dict(start_nm=1549.0, stop_nm=1551.0)
    a = R.mzi(laser(**base), delay_s=50e-12).photodetect(shot=False)
    b = R.mzi(laser(**dict(base, **knob)), delay_s=50e-12).photodetect(shot=False)
    assert a.sha256() != b.sha256()
    assert np.abs(a.waveform() - b.waveform()).max() > 1e-3
    again = Signal.from_recipe(json.loads(json.dumps(b.recipe())))
    assert again.sha256() == b.sha256() and np.array_equal(again.waveform(), b.waveform())
    # the speed is its own knob: the same span in half the record, then parked
    c = R.mzi(laser(start_nm=1549.0, stop_nm=1551.0, speed_nm_per_s=4.0 / DUR), delay_s=50e-12)
    assert np.abs(pd(c) - a.waveform()).max() > 1e-3


@pytest.mark.parametrize("kw,says", [
    (dict(stop_nm=1551.0), "needs start_nm="),
    (dict(start_nm=1549.0, stop_nm=1551.0, speed_nm_per_s=-5.0), "does not lead from"),
    (dict(start_nm=1549.0, stop_nm=1551.0, speed_nm_per_s=1.0), "would not be reached"),
    (dict(start_nm=1549.0, ripple_nm=1e-3), "ripple_nm= (its peak) AND ripple_hz="),
    (dict(start_nm=1549.0, hops=[{"t_s": 9.0, "by_nm": 0.1}]), "inside the record"),
    (dict(start_nm=1549.0, hops=[{"t": 0.01, "by": 0.1}]), "one per mode hop"),
    (dict(start_nm=-1.0), "finite positive number"),
])
def test_a_sweep_that_means_nothing_is_refused_where_it_is_written(kw, says):
    with pytest.raises(ValueError) as e:
        Signal(grid=G).laser(**kw)
    assert says in str(e.value)


def test_the_words_another_op_uses_are_pointed_at_this_ops_own():
    with pytest.raises(TypeError, match="laser says start_nm="):
        Signal(grid=G).laser(start_nm=1549.0, wavelength_nm=1550.0)
    with pytest.raises(TypeError, match="resonator says intrinsic_hz= and coupling_hz="):
        laser().resonator(wavelength_nm=1550.0, intrinsic_hz=1e9, coupling_hz=1e9, q=1e5)


def test_a_laser_is_a_source_and_says_so():
    with pytest.raises(ValueError, match="laser is a source and starts a chain"):
        laser().laser(start_nm=1550.0)
    with pytest.raises(ValueError, match="laser is a source"):
        Signal(grid=G).carrier("sine", f_hz=1e3).laser(start_nm=1550.0)
    with pytest.raises(ValueError, match="STATES domain='real'.*source of an optical field"):
        Signal(grid=Grid(fs=FS, n=N, domain="real")).laser(start_nm=1550.0)
    with pytest.raises(ValueError, match="sweep is counted from the record's start"):
        laser().with_lead_in(64).waveform()
    hand = Signal(grid=Grid(fs=FS, n=N, domain="optical"))
    hand.ops = [{"op": "capture", "values": [[1.0, 0.0]] * N}, {"op": "laser", "start_nm": 1550.0}]
    with pytest.raises(ValueError, match="laser \\(op 1\\).*already a field"):
        hand.waveform()


# ---------------------------------------------------------------- a delay on the field
@pytest.mark.parametrize("tau", [20e-12, 50e-12, 2e-9])
def test_a_delayed_copy_is_turned_by_the_carrier_and_beats_at_sweep_rate_times_delay(tau):
    """E + E(t - tau) detected is (1 + cos(2*pi*nu(t - tau/2)*tau))/2: the carrier's integral
    over the delay is nu at its middle, to third order in tau."""
    las = laser()
    x = pd(las.copy().combine([las.copy().delay(by=tau)]).gain(by=0.5))
    nu_mid = C_NM / (1549.0 + 2.0 * (T - tau / 2) / DUR)
    want = 0.5 * (1.0 + np.cos(2 * np.pi * nu_mid * tau))
    assert np.abs(x - want)[1:].max() < 1e-8                   # measured 4e-12 .. 3e-10
    if tau >= 1e-9:                                            # enough fringes to count
        sp = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(N)))
        beat = sp.argmax() * FS / N
        assert abs(beat / (RATE * tau) - 1.0) < 2e-3           # the beat is rate x delay


def test_a_delay_of_several_samples_shifts_the_envelope_as_well():
    """A slow sweep, a long line: the same closed form, and the copy arrives 2.5 samples
    late (seen on a laser whose power steps)."""
    tau = 2.5e-6
    las = laser(start_nm=1550.0, stop_nm=1550.001)
    x = pd(las.copy().combine([las.copy().delay(by=tau)]).gain(by=0.5))
    nu_mid = C_NM / (1550.0 + 0.001 * (T - tau / 2) / DUR)
    want = 0.5 * (1.0 + np.cos(2 * np.pi * nu_mid * tau))
    # nu * tau is 4.8e8 turns here, and a double's last digits are 1e-7 of one
    assert np.abs(x - want)[4:].max() < 5e-6                       # measured 8e-7
    step = Signal().carrier("step", t_step_s=0.05, tr_s=40e-6, amp=0.25, offset=0.75)
    a = np.abs(las.copy().gain(by=step).waveform())
    b = np.abs(las.copy().gain(by=step).delay(by=tau).waveform())
    k = int(0.05 * FS)
    lag = np.interp(0.75, b[k - 60:k + 60], T[k - 60:k + 60]) - np.interp(
        0.75, a[k - 60:k + 60], T[k - 60:k + 60])
    assert abs(lag - tau) < 2e-9


def test_a_delay_that_moves_phase_modulates_the_fringe():
    """An arm that vibrates by 1 fs (0.3 um of path): the fringe follows nu * tau(t)."""
    wob = Signal().carrier("sine", f_hz=200.0, amp=1e-15, offset=20e-12)
    las = laser()
    x = pd(las.copy().combine([las.copy().delay(by=wob)]).gain(by=0.5))
    tau = 20e-12 + 1e-15 * np.sin(2 * np.pi * 200.0 * T)
    want = 0.5 * (1.0 + np.cos(2 * np.pi * (C_NM / (1549.0 + 2.0 * T / DUR)) * tau))
    still = pd(las.copy().combine([las.copy().delay(by=20e-12)]).gain(by=0.5))
    assert np.abs(x - want)[1:].max() < 1e-6                   # measured 2e-9
    assert np.abs(x - still).max() > 0.5                       # ... and it is not the still arm


def test_mzi_fringe_period_and_extinction():
    tau = 50e-12
    las = laser()
    nu = SW.wavelength_map(las)["f_hz"]
    ideal = pd(R.mzi(las, delay_s=tau))
    assert abs(ideal.max() - 1.0) < 1e-6 and ideal.min() < 1e-6
    assert abs(delay_of_fringe(ideal, nu) / tau - 1.0) < 2e-3      # fringes 1/tau apart
    for er in (10.0, 20.0, 30.0):
        x = pd(R.mzi(las, delay_s=tau, extinction_db=er))
        assert abs(x.max() - 1.0) < 1e-6
        assert abs(10 * np.log10(x.max() / x.min()) - er) < 0.01
    ops = R.mzi(las, delay_s=tau, extinction_db=20.0).ops          # nothing hidden: a chain
    assert [o["op"] for o in ops] == ["laser", "gain", "combine"]
    assert [o["op"] for o in ops[2]["others"][0]["ops"]] == ["laser", "delay", "gain"]


@pytest.mark.parametrize("tau,r", [(40e-12, 0.02), (150e-12, 0.005)])
def test_a_stray_reflection_makes_a_fringe_at_the_spacing_its_delay_predicts(tau, r):
    las = laser()
    x = pd(R.stray_reflection(las, delay_s=tau, reflectivity=r))
    assert abs(np.ptp(x) / (4 * r) - 1.0) < 1e-3                   # 4r peak to peak
    assert abs(x.mean() - (1 + r * r)) < 2 * r / 10
    got = delay_of_fringe(x, SW.wavelength_map(las)["f_hz"])
    assert abs(got / tau - 1.0) < 2e-3                             # measured 4e-4
    # on top of a device it multiplies the device's field, as light going round twice does
    ring = R.ring(las, wavelength_nm=1550.0, q=5e4, extinction_db=10.0)
    both = pd(R.stray_reflection(ring, delay_s=tau, reflectivity=r))
    assert abs(both.min() / pd(ring).min() - 1.0) < 4.5 * r


def test_what_a_field_delay_cannot_do_is_refused_by_name():
    las = laser()
    with pytest.raises(ValueError, match="beats at up to .* past this grid's Nyquist"):
        las.copy().delay(by=1e-6).waveform()
    hop = laser(start_nm=1550.0, stop_nm=1550.001, hops=[{"t_s": 0.05, "by_nm": 0.01}])
    with pytest.raises(ValueError, match="mode hop and this delay is 2.5 samples"):
        hop.copy().delay(by=2.5e-6).waveform()
    hop.copy().delay(by=20e-12).waveform()                         # under a sample: fine
    # a field that states no carrier is refused as before, now naming the source that does
    eo = Signal(seed=1, grid=Grid(fs=80e9, baud=10e9, n=4096)).carrier("nrz", n_ui=512).eo()
    for build, said in ((lambda s: s.delay(by=1e-11), "laser(start_nm=...)"),
                        (lambda s: s.resonator(wavelength_nm=1550.0, intrinsic_hz=1e9,
                                               coupling_hz=1e9), "laser(start_nm=...)")):
        with pytest.raises(ValueError) as e:
            build(eo.copy())
        assert "carrier was never stated" in str(e.value) and said in str(e.value)
    # the copy without the carrier's turn is wrong on a moving carrier
    with pytest.raises(ValueError, match="optical_mpi.*recipes.stray_reflection"):
        las.copy().optical_mpi(delay_samples=3)
    with pytest.raises(ValueError, match="delay_s is how much longer"):
        R.mzi(las, delay_s=0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        R.stray_reflection(las, delay_s=1e-11, reflectivity=1.5)
    with pytest.raises(ValueError, match="the chain the light arrives on"):
        R.mzi(Signal(), delay_s=1e-11)


def test_fields_about_two_carriers_do_not_combine():
    las = laser()
    other = Signal().laser(start_nm=1549.0, stop_nm=1551.5)
    with pytest.raises(ValueError, match="another carrier.*stop_nm=1551.5.*SAME carrier"):
        las.copy().combine([other]).waveform()
    eo = Signal().carrier("sine", f_hz=1e3).eo()
    with pytest.raises(ValueError, match="another carrier \\(none stated\\)"):
        las.copy().combine([eo]).waveform()
    las.copy().combine([Signal().laser(start_nm=1549.0, stop_nm=1551.0)]).waveform()
    # a PINNED record is a field about the carrier its own recipe states
    with pytest.raises(ValueError, match="others\\[0\\] is a field about another carrier"):
        las.copy().combine({"from": laser(start_nm=1540.0).recipe()}).waveform()
    with pytest.raises(ValueError, match="another carrier \\(none stated\\)"):
        las.copy().combine({"from": Signal(seed=1, grid=G).carrier("sine", f_hz=1e3).eo()
                           .recipe()}).waveform()
    both = las.copy().combine({"from": laser().recipe()}).photodetect(shot=False).waveform()
    assert np.allclose(both, 4.0)                    # the same laser twice: twice the field


# ---------------------------------------------------------------- the resonator
def test_a_slow_sweep_traces_the_stated_lorentzian():
    las = laser()
    ki, ke = 1.2e9, 0.7e9
    x = pd(las.copy().resonator(wavelength_nm=1550.0, intrinsic_hz=ki, coupling_hz=ke))
    want = lorentzian(lam_of(las), 1550.0, ki, ke)
    assert np.abs(x - want).max() < 1e-5                           # measured 1.3e-6
    assert abs(x.min() - ((ki - ke) / (ki + ke)) ** 2) < 1e-5
    # the settled response the library states is the same closed form
    assert np.allclose(np.abs(SW.steady(lam_of(las), dict(
        wavelength_nm=1550.0, intrinsic_hz=ki, coupling_hz=ke))) ** 2, want, atol=1e-12)
    # critical coupling: the dip reaches zero
    crit = pd(las.copy().resonator(wavelength_nm=1550.0, intrinsic_hz=1e9, coupling_hz=1e9))
    assert crit.min() < 1e-5


def test_a_ring_measured_on_a_slow_sweep_has_its_stated_wavelength_q_extinction_and_fsr():
    las = laser()
    for coupling in ("under", "over"):
        ring = R.ring(las, wavelength_nm=1550.0, q=1e5, extinction_db=15.0, fsr_nm=0.8,
                      coupling=coupling)
        wl, t = M.transmission(pd(ring), lam_of(las))
        dips = M.resonances(wl, t)
        mid = min(dips, key=lambda d: abs(d["wavelength_nm"] - 1550.0))
        assert len(dips) == 3
        assert abs(mid["wavelength_nm"] - 1550.0) < 2e-5           # 0.02 pm; measured 1e-8 nm
        assert abs(mid["q"] / 1e5 - 1.0) < 5e-3                    # measured +0.2 %
        # the other orders' skirts are in the stated response: -0.003 and -0.034 dB
        assert abs(mid["extinction_db"] - 15.0) < 0.05
        up = [d["wavelength_nm"] for d in dips if d["wavelength_nm"] > 1550.4][0]
        assert abs((up - mid["wavelength_nm"]) - 0.8) < 1e-4       # the next order: +0.8 nm
        assert abs(mid["fwhm_nm"] - 1550.0 / 1e5) < 1e-4
    op = R.ring(las, wavelength_nm=1550.0, q=1e5, extinction_db=15.0, coupling="over").ops[1]
    assert op["coupling_hz"] > op["intrinsic_hz"]
    assert abs((op["coupling_hz"] + op["intrinsic_hz"]) / (C_NM / 1550.0 / 1e5) - 1) < 1e-12
    # the two couplings are one dip in power and opposite in phase on resonance
    under = las.copy().resonator(wavelength_nm=1550.0, intrinsic_hz=1.5e9, coupling_hz=0.5e9)
    over = las.copy().resonator(wavelength_nm=1550.0, intrinsic_hz=0.5e9, coupling_hz=1.5e9)
    k = N // 2
    assert under.waveform()[k].real > 0.4 and over.waveform()[k].real < -0.4


@pytest.mark.parametrize("coupling, q, ext, top", [("under", 19566, 12.20, 0.966),
                                                   ("over", 19050, 11.67, 0.943)])
def test_a_low_finesse_ring_reads_off_its_stated_q_and_extinction_by_the_stated_amount(
        coupling, q, ext, top):
    """`ring(q=2e4, fsr_nm=0.8)` is a finesse of 10: the numbers are one mode's, and the
    neighbouring orders move what a sweep reads by about (pi*coupling_hz/fsr_hz)**2 -- by
    different amounts under- and over-coupled. The record is still the stated response."""
    g = Grid(fs=1e6, n=1 << 18)
    ring = R.ring(Signal(seed=1, grid=g).laser(start_nm=1549.5, stop_nm=1551.5),
                  wavelength_nm=1550.0, q=2e4, extinction_db=12.0, fsr_nm=0.8, coupling=coupling)
    lam = SW.wavelength_map(ring)["wavelength_nm"]
    x = pd(ring)
    assert np.abs(x - np.abs(SW.steady(lam, ring.ops[-1])) ** 2).max() < 1e-6
    dip = min(M.resonances(*M.transmission(x, lam), level=1.0),
              key=lambda d: abs(d["wavelength_nm"] - 1550.0))
    print(f"{coupling}: Q {dip['q']:.0f} (stated 20000), extinction {dip['extinction_db']:.2f} "
          f"dB (stated 12), {x.max():.3f} between resonances")
    assert abs(dip["q"] / q - 1) < 2e-3 and abs(dip["extinction_db"] - ext) < 0.02
    assert abs(x.max() - top) < 2e-3
    rho2 = (np.pi * ring.ops[-1]["coupling_hz"] / ring.ops[-1]["fsr_hz"]) ** 2
    assert abs(dip["q"] / 2e4 - 1) < 6 * rho2 and abs(dip["extinction_db"] - 12.0) < 0.5


@pytest.mark.parametrize("ki,ke,fsr,bound", [(1e9, 0.5e9, 100e9, 5e-4),
                                              (2e9, 2e9, 50e9, 3e-2)])
def test_every_order_of_the_ring_is_counted_and_it_is_the_ring_formula(ki, ke, fsr, bound):
    """Against (r - a*e^{j phi}) / (1 - r*a*e^{j phi}): equal to first order in the coupling
    per round trip rho = pi*ke/fsr, so the difference is about rho^2 (2.5e-4 and 1.6e-2)."""
    las = laser()
    op = dict(wavelength_nm=1550.0, intrinsic_hz=ki, coupling_hz=ke, fsr_hz=fsr)
    x = pd(las.copy().resonator(**op))
    nu = SW.wavelength_map(las)["f_hz"]
    r, a = np.exp(-np.pi * ke / fsr), np.exp(-np.pi * ki / fsr)
    e = np.exp(-2j * np.pi * (nu - C_NM / 1550.0) / fsr)
    exact = np.abs((r - a * e) / (1 - r * a * e)) ** 2
    rho = np.pi * ke / fsr
    assert np.abs(x - exact).max() < bound and np.abs(x - exact).max() < 2.5 * rho ** 2
    # the record is the stated settled response to 1e-5, far orders included
    assert np.abs(x - np.abs(SW.steady(lam_of(las), op)) ** 2).max() < 1e-5
    # ... which a sum over the nearest orders alone misses by more
    near = np.abs(1 - 2 * np.pi * ke * sum(
        1.0 / (np.pi * (ki + ke) + 2j * np.pi * (nu - C_NM / 1550.0 - m * fsr))
        for m in range(-4, 5))) ** 2
    assert np.abs(near - x).max() > 20 * np.abs(x - np.abs(SW.steady(lam_of(las), op)) ** 2).max()


def _fast(speed=50.0, fs=50e6, n=100000, width=1e6):
    g = Grid(fs=fs, n=n)
    las = Signal(seed=1, grid=g).laser(start_nm=1550.0, speed_nm_per_s=speed)
    lam = lam_of(las)
    lam0 = float(lam[n // 2])
    ring = las.copy().resonator(wavelength_nm=lam0, intrinsic_hz=0.3 * width,
                                coupling_hz=0.7 * width)
    d = 2 * np.pi * (C_NM / lam - C_NM / lam0)                     # detuning, rad/s
    return g, ring, d, 2 * np.pi * width, 2 * np.pi * 0.7 * width


@pytest.mark.parametrize("speed,fs", [(50.0, 50e6), (-200.0, 200e6), (2.0, 2e6)])
def test_a_fast_sweep_rings_down_as_the_swept_lorentzian_says(speed, fs):
    """A linear chirp through one mode has a closed form in the Faddeeva function:
    y = sqrt(pi/p)/2 * w(j*b/(2*sqrt(p))), p = -j*chirp/2, b = kappa/2 + j*D(t)."""
    g, ring, d, kappa, ke = _fast(speed, fs)
    t = np.arange(g.n) / g.fs
    chirp = np.gradient(d, t)[g.n // 2]
    p = -0.5j * chirp
    want = 1.0 - ke * 0.5 * np.sqrt(np.pi / p) * wofz(1j * (0.5 * kappa + 1j * d)
                                                      / (2 * np.sqrt(p)))
    x = ring.waveform()
    assert np.abs(x - want).max() < 3e-5                           # measured 4e-6 .. 9e-6
    still = 1.0 - ke / (0.5 * kappa + 1j * d)
    if abs(chirp) > 0.5 * kappa ** 2:                              # crossed within a lifetime
        assert np.abs(x - still).max() > 0.5                       # nothing like the Lorentzian
        assert (np.abs(x) ** 2).max() > 1.3                        # the ring-down beats above 1
        assert int(np.argmin(np.abs(x))) > g.n // 2                # ... and the dip comes late


def test_the_ring_down_agrees_with_an_independent_ode_integration():
    g, ring, d, kappa, ke = _fast()
    t = np.arange(g.n) / g.fs
    i0, i1 = g.n // 2 - 6000, g.n // 2 + 12000
    sol = solve_ivp(lambda s, y: -(0.5 * kappa + 1j * np.interp(s, t, d)) * y + 1.0,
                    (t[i0], t[i1]), [1.0 / (0.5 * kappa + 1j * d[i0]) + 0j],
                    t_eval=t[i0:i1 + 1], method="DOP853", rtol=1e-10, atol=1e-16)
    x = ring.waveform()[i0:i1 + 1]
    assert np.abs(x - (1.0 - ke * sol.y[0])).max() < 3e-5          # measured 6e-6


def _hop_ring(df_hz, frac, fs=50e6, n=2000, k=500):
    """A laser standing on a critically coupled 2 MHz ring that hops `df_hz` at sample
    `k + frac`: (the rendered field, the closed form). After the hop the detuning is constant,
    so y relaxes from 2/kappa to 1/z as exp(-z*(t - t_hop)), with z = kappa/2 + j*2*pi*df."""
    lam, kappa = 1550.0, 2 * np.pi * 2e6
    th = (k + frac) / fs
    hop = [{"t_s": th, "by_nm": C_NM / (C_NM / lam + df_hz) - lam}]
    sig = Signal(seed=1, grid=Grid(fs=fs, n=n)).laser(start_nm=lam, hops=hop).resonator(
        wavelength_nm=lam, intrinsic_hz=1e6, coupling_hz=1e6)
    t = np.arange(n) / fs
    z = 0.5 * kappa + 2j * np.pi * df_hz
    y = np.where(t < th, 2.0 / kappa,
                 (2.0 / kappa) * np.exp(-z * (t - th)) - np.expm1(-z * (t - th)) / z)
    return sig, 1.0 - 0.5 * kappa * y


@pytest.mark.parametrize("df_hz, frac", [(3.3e6, 0.37), (20e6, 0.37), (24e6, 0.5),
                                         (-20e6, 0.999), (20e6, 0.001), (20e6, 0.0)])
def test_a_mode_hop_between_two_samples_rings_the_resonator_from_its_own_instant(df_hz, frac):
    sig, exact = _hop_ring(df_hz, frac)
    x = sig.waveform()
    assert np.abs(x - exact).max() < 1e-11                         # measured 8e-14
    # ... and it is the hop's INSTANT that is being checked: half a sample out is far off
    late = _hop_ring(df_hz, frac + 0.5)[1]
    assert np.abs(late - exact).max() > 0.05


def test_a_hop_that_would_ring_past_nyquist_is_refused_and_one_off_resonance_is_not():
    with pytest.raises(ValueError, match=r"mode hop at t_s=.*still holds light.*alias"):
        _hop_ring(123.4e6, 0.37)[0].waveform()                     # Nyquist is 25 MHz
    with pytest.raises(ValueError, match="Raise Grid.fs=. above 1.2e\\+08 Sa/s"):
        _hop_ring(60e6, 0.5)[0].waveform()
    # the same 60 MHz hop on a grid that holds it is solved
    sig, exact = _hop_ring(60e6, 0.5, fs=200e6, n=8000, k=2000)
    assert np.abs(sig.waveform() - exact).max() < 1e-11
    # a hop of gigahertz far from every resonance leaves nothing ringing: accepted
    far = Signal(seed=1, grid=G).laser(start_nm=1549.0, hops=[{"t_s": 0.0500004, "by_nm": 0.03}])
    x = far.resonator(wavelength_nm=1550.0, intrinsic_hz=1e6, coupling_hz=1e6).waveform()
    assert np.abs(np.abs(x) - 1.0).max() < 1e-4


@pytest.mark.parametrize("kdt", [0.1, 1.0, 6.0, 30.0])
def test_an_input_that_ramps_across_a_sample_is_integrated_exactly(kdt):
    """A shuttered or power-ramped field into a ring. For a standing laser and an input linear
    across each sample the equation has an exact step, written out here."""
    fs, n, lam = 1e6, 200, 1550.0
    width = kdt * fs / (2 * np.pi)
    for det_hz in (0.0, 0.3 * fs):
        z = 0.5 * 2 * np.pi * width + 2j * np.pi * det_hz
        x = (1.0 + 0.5 * np.interp(np.arange(n), [0, 50, 60, 200], [0, 0, 1, 1])).astype(complex)
        e = np.exp(-z / fs)
        p1 = (1 - e) / (z / fs)
        p2 = (z / fs - 1 + e) / (z / fs) ** 2
        y = np.empty(n, complex)
        y[0] = x[0] / z
        for k in range(n - 1):
            y[k + 1] = e * y[k] + (x[k] * (p1 - p2) + x[k + 1] * p2) / fs
        got = SW.resonator(x, fs, dict(start_nm=C_NM / (C_NM / lam + det_hz)),
                           dict(wavelength_nm=lam, intrinsic_hz=width / 2, coupling_hz=width / 2))
        assert np.abs(got - (x - np.pi * width * y)).max() < 1e-12    # measured 2e-13


def test_detector_bandwidth_and_sweep_speed_interact():
    """The same ring, the same detector: a slow sweep keeps the dip, a fast one loses depth
    by what the detector's response does to the dip's shape in TIME."""
    depth = {}
    for name, n in (("slow", 1 << 17), ("fast", 1 << 8)):          # 0.13 s and 0.26 ms
        g = Grid(fs=FS, n=n)
        las = Signal(seed=1, grid=g).laser(start_nm=1549.9, stop_nm=1550.1)
        ring = las.copy().resonator(wavelength_nm=1550.0, intrinsic_hz=1e9, coupling_hz=1e9)
        seen = ring.copy().photodetect(shot=False).tia(bw_hz=10e3).waveform()
        assert pd(ring).min() < 1e-6                               # the ring itself: to zero
        # the dip is 2 GHz wide: 10 ms of the slow sweep, 20 us of the fast one, against a
        # 10 kHz detector. The same detector on the Lorentzian written out in time:
        want = ws.optical.tia(lorentzian(lam_of(las), 1550.0, 1e9, 1e9), bw_hz=10e3, grid=g)
        assert np.abs(seen - want).max() < 5e-5                    # measured 3e-6
        depth[name] = 1.0 - seen.min()
    assert depth["slow"] > 0.9999 and 0.5 < depth["fast"] < 0.62   # measured 0.99999, 0.562


def test_backscatter_splits_the_resonance_by_split_hz():
    las = laser()
    x = pd(las.copy().resonator(wavelength_nm=1550.0, intrinsic_hz=0.6e9, coupling_hz=0.4e9,
                                split_hz=5e9))
    nu = SW.wavelength_map(las)["f_hz"]
    wl, t = M.transmission(x, lam_of(las))
    dips = M.resonances(wl, t)
    assert len(dips) == 2
    gap = C_NM / dips[0]["wavelength_nm"] - C_NM / dips[1]["wavelength_nm"]
    assert abs(gap / 5e9 - 1.0) < 0.01                             # measured -0.4 %
    mean = 0.5 * (dips[0]["wavelength_nm"] + dips[1]["wavelength_nm"])
    assert abs(mean - 1550.0) < 1e-4
    d = nu - C_NM / 1550.0
    want = np.abs(1 - 0.4e9 * 0.5 * (1 / (0.5e9 + 1j * (d - 2.5e9))
                                     + 1 / (0.5e9 + 1j * (d + 2.5e9)))) ** 2
    assert np.abs(x - want).max() < 1e-5


def test_a_resonance_that_moves_during_the_sweep_is_a_composition():
    """`by` in nm: a number moves the dip by that much; a chain makes it drift, and the dip
    is found where the laser meets the moving resonance."""
    las = laser()
    lam = lam_of(las)
    kw = dict(wavelength_nm=1550.0, intrinsic_hz=1e9, coupling_hz=1e9)
    assert abs(lam[pd(las.copy().resonator(by=0.01, **kw)).argmin()] - 1550.01) < 2e-5
    warm = Signal().carrier("sine", f_hz=3.0, amp=0.02)            # +/- 20 pm at 3 Hz
    sig = las.copy().resonator(by=warm, **kw)
    assert "resonator/1/by" in sig.summary()
    res = 1550.0 + 0.02 * np.sin(2 * np.pi * 3.0 * T)
    meet = int(np.argmin(np.abs(lam - res)))
    assert abs(int(pd(sig).argmin()) - meet) <= 2
    assert abs(lam[meet] - 1550.0) > 0.01                          # it did move: 18 pm
    assert np.abs(pd(sig) - lorentzian(lam, res, 1e9, 1e9)).max() < 1e-4


RESONATOR_KNOBS = [dict(wavelength_nm=1550.2), dict(intrinsic_hz=2e9), dict(coupling_hz=0.2e9),
                   dict(fsr_hz=60e9), dict(split_hz=3e9), dict(by=0.02)]


@pytest.mark.parametrize("knob", RESONATOR_KNOBS, ids=lambda k: next(iter(k)))
def test_every_resonator_parameter_moves_the_record_and_its_digest(knob):
    base = dict(wavelength_nm=1550.0, intrinsic_hz=1e9, coupling_hz=0.5e9, fsr_hz=100e9)
    a = laser().resonator(**base).photodetect(shot=False)
    b = laser().resonator(**dict(base, **knob)).photodetect(shot=False)
    assert a.sha256() != b.sha256() and np.abs(a.waveform() - b.waveform()).max() > 1e-2
    again = Signal.from_recipe(json.loads(json.dumps(b.recipe())))
    assert np.array_equal(again.waveform(), b.waveform())


def test_a_resonator_that_cannot_be_solved_honestly_is_refused():
    las = laser()
    with pytest.raises(ValueError, match="crosses the resonance in .* samples.*alias"):
        las.copy().resonator(wavelength_nm=1550.0, intrinsic_hz=2e6, coupling_hz=2e6).waveform()
    with pytest.raises(ValueError, match="needs coupling_hz=.*ws.recipes.ring"):
        las.copy().resonator(wavelength_nm=1550.0, intrinsic_hz=1e9)
    with pytest.raises(ValueError, match="not both zero"):
        las.copy().resonator(wavelength_nm=1550.0, intrinsic_hz=0.0, coupling_hz=0.0)
    with pytest.raises(ValueError, match="over 4 linewidths"):
        las.copy().resonator(wavelength_nm=1550.0, intrinsic_hz=5e9, coupling_hz=5e9,
                             fsr_hz=30e9)
    with pytest.raises(ValueError, match="ring: coupling is 'under'"):
        R.ring(las, wavelength_nm=1550.0, q=1e5, coupling="critical")
    with pytest.raises(ValueError, match="takes a number.*tolerance where the number ends up"):
        R.ring(las, wavelength_nm=1550.0, q=ws.tol(1e5, 10))


# ---------------------------------------------------------------- a campaign, two channels
def bench(laser_kw=None, ref_delay_s=2e-9, noise=0.0, **ring_kw):
    """A swept bench: one laser, the device on one detector, an MZI reference on another."""
    src = lambda: Signal().use("laser")                            # noqa: E731
    ring_kw = ring_kw or dict(wavelength_nm=1550.0, q=1e5, extinction_db=15.0, fsr_nm=0.8)
    ref = R.mzi(src(), delay_s=ref_delay_s).photodetect(shot=False)
    return ws.Channels(
        {"dut": R.ring(src(), **ring_kw).photodetect(shot=False),
         "ref": ref.tia(thermal_rms=noise) if noise else ref},
        shared={"laser": Signal().laser(**(laser_kw or dict(start_nm=1549.0, stop_nm=1551.0)))},
        seed=1, grid=G)


def test_the_map_recovered_from_the_reference_channel_matches_the_truth():
    """Fringes 4 pm apart: recovered to a few thousandths of a fringe, the ends included,
    on a sweep that is not linear; and to a few hundredths with 1 % detector noise.
    Measured, in fringes: 2e-4 linear, 6e-3 bowed and rippled, 1.6e-2 noisy, 1.3e-3 falling."""
    fringe_nm = 1550.0 ** 2 / C_NM / 2e-9
    for laser_kw, noise, tol in (
            (dict(start_nm=1549.0, stop_nm=1551.0), 0.0, 3e-3),
            (dict(start_nm=1549.0, stop_nm=1551.0, bow_nm=0.02, ripple_nm=3e-3,
                  ripple_hz=40.0), 0.0, 0.02),
            (dict(start_nm=1549.0, stop_nm=1551.0), 0.01, 0.06)):
        rec = bench(laser_kw, noise=noise)
        w = rec.waveforms()
        truth = lam_of(rec.signal("ref"))
        got = M.swept_map(w["ref"], G, 2e-9, at_s=0.0, wavelength_nm=1549.0)
        assert np.abs(got - truth).max() < tol * fringe_nm
    # the linear sweep one would otherwise assume is 23 pm out on the second of those
    assert np.abs(truth - lam_of(bench().signal("ref"))).max() < 1e-12
    rec = bench(dict(start_nm=1549.0, stop_nm=1551.0, bow_nm=0.02, ripple_nm=3e-3,
                     ripple_hz=40.0))
    assert np.abs(lam_of(rec.signal("ref")) - (1549.0 + 2.0 * T / DUR)).max() > 0.02
    # a falling sweep says so
    down = bench(dict(start_nm=1551.0, stop_nm=1549.0))
    got = M.swept_map(down.waveforms()["ref"], G, 2e-9, at_s=0.0, wavelength_nm=1551.0,
                      rising=False)
    assert np.abs(got - lam_of(down.signal("ref"))).max() < 3e-3 * fringe_nm


@pytest.mark.parametrize("ripple_hz, bow", [(150.0, 0.02), (150.0, 0.0), (200.0, 0.004),
                                            (60.0, 0.02)])
def test_the_part_fringe_at_each_end_is_read_where_the_sweep_rate_is_changing(ripple_hz, bow):
    """A ripple makes the rate at a record's end differ from the whole fringe beside it by
    tens of percent, so the part fringe is read from its own shape (its crest and trough),
    not from an extrapolation. Four starts put the record's first sample at four places in
    a fringe; an error in the FIRST part fringe would move every sample, the anchor being
    there. Measured: 0.02 fringe at worst (it was 0.7)."""
    g, tau = Grid(fs=1e6, n=1 << 17), 2e-9
    fringe_nm = 1550.0 ** 2 / C_NM / tau
    worst = 0.0
    for start in (1549.8, 1549.801, 1549.802, 1549.803):
        las = Signal(seed=1, grid=g).laser(start_nm=start, stop_nm=1550.2, bow_nm=bow,
                                           ripple_nm=1e-3, ripple_hz=ripple_hz)
        truth = lam_of(las)
        assert np.all(np.diff(truth) > 0)                          # it never reverses
        ref = pd(R.mzi(las, delay_s=tau))
        got = M.swept_map(ref, g, tau, at_s=0.0, wavelength_nm=start)
        worst = max(worst, float(np.abs(got - truth).max()) / fringe_nm)
    print(f"ripple {ripple_hz:g} Hz, bow {bow:g} nm: worst map error {worst:.4f} fringe")
    assert worst < 0.04


@pytest.mark.parametrize("noise, bound", [(0.01, 0.05), (0.02, 0.08)])
def test_the_part_fringe_is_read_through_detector_noise(noise, bound):
    """A record that ends a few samples after a counted rise, still climbing: with noise on
    it one of those samples is the highest, and read as a crest it put the last ones a
    quarter of a fringe out (5 draws in 60 at 1 % of the fringe's height, 17 at 2 %). The
    turn is taken of the samples averaged, so the ends are read as well as the whole
    fringes are. Measured over 24 draws: 0.023 fringe at worst at 1 %, 0.039 at 2 %; the
    same sweep without noise 0.013."""
    g, tau = Grid(fs=1e6, n=1 << 15), 2e-9
    las = Signal(seed=1, grid=g).laser(start_nm=1550.0051208875166, stop_nm=1548.991121344502,
                                       ripple_nm=0.012982360639786781,
                                       ripple_hz=88.40705818175192, bow_nm=0.054589788878561835)
    truth = lam_of(las)
    assert np.all(np.diff(truth) < 0)                              # it never reverses
    ref, mid = pd(R.mzi(las, delay_s=tau)), g.n // 2

    def worst_end(rec):
        got = M.swept_map(rec, g, tau, at_s=mid / g.fs, wavelength_nm=float(truth[mid]),
                          rising=False)
        err = np.abs(C_NM / got - C_NM / truth) * tau              # in fringes
        return float(max(err[:400].max(), err[-400:].max()))

    clean = worst_end(ref)
    ends = [worst_end(ref + np.random.default_rng(k).normal(0, noise, g.n) * np.ptp(ref))
            for k in range(24)]
    print(f"noise {noise:g} of the fringe: end error {max(ends):.4f} fringe at worst, "
          f"{np.median(ends):.4f} median; without noise {clean:.4f}")
    assert clean < 0.02 and max(ends) < bound


def test_transmission_on_the_recovered_axis_gives_the_stated_ring():
    rec = bench(dict(start_nm=1549.0, stop_nm=1551.0, bow_nm=0.02, ripple_nm=3e-3,
                     ripple_hz=40.0))
    w = rec.waveforms()
    lam = M.swept_map(w["ref"], G, 2e-9, at_s=0.0, wavelength_nm=1549.0)
    wl, t = M.transmission(w["dut"], lam)
    op = [o for o in rec.signal("dut").ops if o["op"] == "resonator"][0]
    assert np.all(np.diff(wl) >= 0)
    assert np.abs(t - np.abs(SW.steady(wl, op)) ** 2).max() < 5e-3      # measured 1.4e-3
    mid = min(M.resonances(wl, t), key=lambda d: abs(d["wavelength_nm"] - 1550.0))
    assert abs(mid["wavelength_nm"] - 1550.0) < 5e-5                    # 0.05 pm
    assert abs(mid["q"] / 1e5 - 1.0) < 5e-3 and abs(mid["extinction_db"] - 15.0) < 0.02
    # on the axis a linear sweep would assume, the same ring is misplaced by picometres
    wl2, t2 = M.transmission(w["dut"], 1549.0 + 2.0 * T / DUR)
    off = min(M.resonances(wl2, t2), key=lambda d: abs(d["wavelength_nm"] - 1550.0))
    assert abs(off["wavelength_nm"] - 1550.0) > 0.015


def test_a_mode_hop_breaks_the_fringe_count_by_the_hop_less_whole_fringes():
    hop = dict(start_nm=1549.0, stop_nm=1551.0, hops=[{"t_s": 0.07, "by_nm": 0.03}])
    rec = bench(hop)
    w, ev = rec.realize()
    truth = lam_of(rec.signal("ref"))
    got = M.swept_map(w["ref"], G, 2e-9, at_s=0.0, wavelength_nm=1549.0)
    k = int(0.07 * FS)
    fringe_nm = 1550.0 ** 2 / C_NM / 2e-9
    assert np.abs(got - truth)[:k - 500].max() < 3e-3 * fringe_nm
    err = (got - truth)[k + 500:] / fringe_nm
    assert np.ptp(err) < 0.02 and abs(err.mean() + 7.0) < 0.05     # 30 pm is 7.49 fringes
    assert [e.t_s for e in ev["dut"] if e.kind == "mode_hop"] == [0.07]
    # the hop is ground truth in the map, so on the TRUE axis the ring is where it was put
    wl, t = M.transmission(w["dut"], truth)
    mid = min(M.resonances(wl, t), key=lambda d: abs(d["wavelength_nm"] - 1550.0))
    assert abs(mid["wavelength_nm"] - 1550.0) < 5e-5


def test_a_gas_cell_line_is_an_absolute_anchor():
    src = lambda: Signal().use("laser")                            # noqa: E731
    rec = ws.Channels(
        {"ref": R.mzi(src(), delay_s=2e-9).photodetect(shot=False),
         "gas": R.gas_cell(src(), lines_nm=[1549.73, 1550.52], depth=0.4,
                           width_hz=2e9).photodetect(shot=False)},
        shared={"laser": Signal().laser(start_nm=1549.0, stop_nm=1551.0, bow_nm=0.02)},
        seed=1, grid=G)
    w = rec.waveforms()
    truth = lam_of(rec.signal("gas"))
    wl, t = M.transmission(w["gas"], truth)
    lines = M.resonances(wl, t)
    assert [round(d["wavelength_nm"], 4) for d in lines] == [1549.73, 1550.52]
    for d in lines:
        assert abs(d["t_min"] - 0.6) < 1e-3                        # takes 0.4 of the power
        assert abs(C_NM / d["wavelength_nm"] ** 2 * d["fwhm_nm"] / 2e9 - 1.0) < 5e-3
    # anchor the fringe count on the time the first line is crossed: no start wavelength
    at = float(np.argmin(np.where(T < 0.07, w["gas"], 1.0))) / FS
    got = M.swept_map(w["ref"], G, 2e-9, at_s=at, wavelength_nm=1549.73)
    assert np.abs(got - truth).max() < 2e-5                        # 0.02 pm; measured 0.01


def test_a_shared_stage_is_handed_the_carrier_of_the_field_it_acts_on():
    kw = dict(wavelength_nm=1550.0, intrinsic_hz=1e9, coupling_hz=0.5e9)
    rec = ws.Channels({"out": Signal().use("laser").use("device").photodetect(shot=False)},
                      shared={"laser": Signal().laser(start_nm=1549.0, stop_nm=1551.0),
                              "device": Signal().resonator(**kw)}, seed=1, grid=G)
    assert np.array_equal(rec.waveforms()["out"], pd(laser().resonator(**kw)))
    again = ws.Channels.from_recipe(json.loads(json.dumps(rec.recipe())))
    assert np.array_equal(again.waveforms()["out"], rec.waveforms()["out"])
    assert "resonator" in rec.summary() and "-> optical" in rec.summary()


def test_die_to_die_variation_is_a_tolerance_on_the_resonator():
    """Nothing added: `ws.tol` where the number goes, `unit()` for another die, and each
    die's drawn resonance is in its recipe and is where the sweep finds it."""
    design = laser().resonator(wavelength_nm=ws.tol(1550.0, tol_abs=0.3),
                               intrinsic_hz=ws.tol(1e9, 20), coupling_hz=ws.tol(1e9, 20),
                               fsr_hz=100e9).photodetect(shot=False)
    assert sorted(design.roles()) == ["unit/1/coupling_hz", "unit/1/intrinsic_hz",
                                      "unit/1/wavelength_nm"]
    seen = []
    for die in range(4):
        unit = design.unit(die)
        op = unit.recipe()["ops"][1]
        wl, t = M.transmission(unit.waveform(), lam_of(unit))
        near = min(M.resonances(wl, t), key=lambda d: abs(d["wavelength_nm"] - op["wavelength_nm"]))
        assert abs(near["wavelength_nm"] - op["wavelength_nm"]) < 5e-5
        q = (C_NM / op["wavelength_nm"]) / (op["intrinsic_hz"] + op["coupling_hz"])
        assert abs(near["q"] / q - 1.0) < 0.01
        seen.append(op["wavelength_nm"])
    assert np.ptp(seen) > 0.05 and all(abs(s - 1550.0) <= 0.3 for s in seen)


def test_a_range_switch_is_a_gain_by_a_step_after_the_detector():
    """Detector range switching composed: the photocurrent's scale steps at a stated time."""
    rng = Signal().carrier("step", t_step_s=0.06, tr_s=20e-6, amp=4.5, offset=5.5)   # x1 -> x10
    ring = laser().resonator(wavelength_nm=1550.0, intrinsic_hz=1e9, coupling_hz=0.5e9)
    x, y = pd(ring.copy()), ring.copy().photodetect(shot=False).gain(by=rng).waveform()
    k = int(0.06 * FS)
    assert np.allclose(y[:k - 100], x[:k - 100], rtol=1e-6)
    assert np.allclose(y[k + 100:], 10.0 * x[k + 100:], rtol=1e-6)


# ---------------------------------------------------------------- the measures alone
def test_resonances_reads_a_lorentzian_it_is_handed():
    lam = np.linspace(1549.5, 1550.5, 20001)
    t = lorentzian(lam, 1550.1234, 3e9, 1e9)
    width = 4e9 * 1550.1234 ** 2 / C_NM
    (d,) = M.resonances(lam, t, level=1.0)
    assert abs(d["wavelength_nm"] - 1550.1234) < 1e-6
    assert abs(d["fwhm_nm"] / width - 1.0) < 1e-4
    assert abs(d["q"] / (1550.1234 / width) - 1.0) < 1e-4
    assert abs(d["extinction_db"] - 10 * np.log10(1.0 / 0.25)) < 1e-4
    assert abs(d["t_min"] - 0.25) < 1e-6
    # left to find the level itself it takes the median, which the dip's skirts pull down
    (m,) = M.resonances(lam, t)
    assert -6e-3 < m["fwhm_nm"] / width - 1.0 < -2e-3              # measured -0.41 %
    assert M.resonances(lam, t, depth=0.9) == []


def test_the_swept_measures_refuse_what_they_cannot_read():
    with pytest.raises(ValueError, match="swept_map.*complex"):
        M.swept_map(laser().waveform(), G, 2e-9, at_s=0.0, wavelength_nm=1549.0)
    with pytest.raises(ValueError, match="under two whole fringes"):
        M.swept_map(pd(R.mzi(laser(start_nm=1550.0), delay_s=2e-9)) + 0.0, G, 2e-9,
                    at_s=0.0, wavelength_nm=1550.0)
    with pytest.raises(ValueError, match="at_s=.*inside the record"):
        M.swept_map(np.cos(np.arange(1000.0)), G, 2e-9, at_s=1.0, wavelength_nm=1550.0)
    with pytest.raises(ValueError, match="one wavelength per sample"):
        M.transmission(np.ones(10), np.ones(9))
    # a stated baseline the record never comes half way up to: 6 dB of loss, level=1.0
    ring = R.ring(laser(start_nm=1549.9, stop_nm=1550.1), wavelength_nm=1550.0, q=1e5,
                  extinction_db=15.0)
    lam = lam_of(ring)
    wl, t = M.transmission(pd(ring) * 10 ** -0.6, lam)
    with pytest.raises(ValueError, match=r"level=1.0 is the transmission away from resonance"
                                         r".*never rises above 0.2497.*power=.*leave level= out"):
        M.resonances(wl, t, level=1.0)
    # a smaller loss is NOT refused (a ring of low finesse never comes up to its baseline
    # either), and is read wrong by the stated amount: 1 dB left in, on the REFERENCE bench
    g17 = Grid(fs=1e6, n=1 << 17)
    wide = R.ring(laser(g17), wavelength_nm=1550.0, q=1e5, extinction_db=15.0)
    left = M.resonances(*M.transmission(pd(wide) * 10 ** -0.1, lam_of(wide)), level=1.0)[0]
    print(f"1 dB left in, level=1.0: Q {left['q']:.0f} for 1e5, "
          f"{left['extinction_db']:.2f} dB for 15")
    assert abs(1e5 / left["q"] - 1.31) < 0.02 and abs(left["extinction_db"] - 16.0) < 0.02
    own = M.resonances(wl, t)[0]                # its own median: 13 widths of span, 2 % narrow
    assert abs(own["q"] / 1e5 - 1) < 0.04 and abs(own["extinction_db"] - 15.0) < 0.2
    out = M.resonances(*M.transmission(pd(ring) * 10 ** -0.6, lam, power=10 ** -0.6),
                       level=1.0)[0]            # the loss divided out: the stated ring
    assert abs(out["q"] / 1e5 - 1) < 5e-3 and abs(out["extinction_db"] - 15.0) < 0.05
    with pytest.raises(ValueError, match="no `laser`"):
        SW.wavelength_map(Signal(grid=G).carrier("sine", f_hz=1e3))

"""One driver, one package inductance, and the ringing an R-C model cannot show.

Run from the repository root after ``pip install -e .``:

    python examples/inductive_handover.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
An `{"kind": "l"}` element in a shared net -- package and lead inductance -- which is what turns
a hand-over from a settle into an overshoot and a ring:

    elements=[{"id": "lp", "kind": "l", "a": "pad", "b": "bus", "henry": 8e-9, "ohm": 0.3},
              {"id": "cp", "kind": "c", "a": "pad", "b": "gnd", "farad": 2e-12},
              ...]

An R-C net can only settle toward a level. What a scope shows when a fast driver takes a bus,
changes it, or lets go of it is overshoot at a frequency set by that inductance against the node
capacitance, decaying at a rate set by the termination -- and for a waveform-level comparison of
a synthetic record against a measured one, that was the largest remaining visible difference.
This file prints how large: the same board, the same driver and the same edge, with the
inductance in and with it replaced by its own series resistance.

THE PART AND THE BOARD
----------------------
An ordinary 3.3 V bus transceiver in a package with long leads, on a lightly loaded bus:

    output stage      7 ohm, 300 ps edges        a strong driver, which is the problem
    package           8 nH with 0.3 ohm of DC    bond wire plus lead frame
    pad               2 pF                       inside the package, where nobody probes
    bus              20 pF, 100 ohm to 1.65 V    few loads, so little damping

Two numbers from the receiver's data sheet decide whether the record is a pass: an input may not
go above V_DD + 0.3 V nor below -0.3 V, because past those the input protection diodes conduct.
They are the caller's numbers; the library has no opinion about either.

WHAT THE LIBRARY DOES NOT DO HERE
---------------------------------
It has no protection diode. A diode is a VOLTAGE-DEPENDENT element and this tier solves linear
ones, so an excursion past -0.3 V is rendered in full rather than clamped. That is the useful
answer and not a shortcoming to work around: what the record is telling you is that the diode
conducts, which is the finding. What it cannot tell you is how hard, or how hot.

It also has no opinion about which series resistor you should fit. The last section sweeps one
and prints the margin and the edge rate it costs; choosing between them is engineering.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth.compose import Signal

V_DD, R_OUT, TR = 3.3, 7.0, 300e-12            # the output stage
L_PKG, R_PKG, C_PAD = 8e-9, 0.3, 2e-12         # the package: bond wire, lead frame, pad
R_TERM, V_BIAS, C_BUS = 100.0, 1.65, 20e-12    # the board
V_ABSMAX, V_ABSMIN = V_DD + 0.3, -0.3          # the receiver's data sheet, not the library's

F_SIG = 1.0 / 120e-9                           # 60 ns high, 60 ns low
T_ON, T_RELEASE = 10e-9, 200e-9                # when the output stage conducts, and lets go

FS, N = 2e11, 1 << 16                          # 5 ps grid, 328 ns record


def board(extra_ohm=0.0, inductive=True):
    """The net. `extra_ohm` is a series damping resistor fitted in the same place the package
    inductance is -- on the same piece of wire, which is why it is the inductor's own `ohm`
    rather than a separate element: the node between a series R and a series L has no
    capacitance, so it is not a state and this tier would refuse it.

    `inductive=False` is the R-C model of the identical board -- the same resistances and the
    same total capacitance, with the inductance left out, which is the model this library could
    write before this one. It is the comparison the first finding is about. There is no pad node
    in it, because without an inductance there is nothing between the pad and the bus: that is
    the whole of the difference, and it is worth 57 % of the swing."""
    if not inductive:
        return dict(
            nodes=["bus"], rails={"vb": V_BIAS},
            elements=[{"id": "cb", "kind": "c", "a": "bus", "b": "gnd",
                       "farad": C_PAD + C_BUS},
                      {"id": "rt", "kind": "r", "a": "bus", "b": "vb", "ohm": R_TERM}],
            drivers=[{"id": "d", "node": "bus", "ohm": R_OUT + R_PKG + extra_ohm,
                      "state": "hiz"}])
    return dict(
        nodes=["pad", "bus"], rails={"vb": V_BIAS},
        elements=[{"id": "lp", "kind": "l", "a": "pad", "b": "bus",
                   "henry": L_PKG, "ohm": R_PKG + extra_ohm},
                  {"id": "cp", "kind": "c", "a": "pad", "b": "gnd", "farad": C_PAD},
                  {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": C_BUS},
                  {"id": "rt", "kind": "r", "a": "bus", "b": "vb", "ohm": R_TERM}],
        # no `volts`: this driver's level is its own chain, below
        drivers=[{"id": "d", "node": "pad", "ohm": R_OUT, "state": "hiz"}])


def source(grid):
    """The output stage's own waveform: a CMOS square wave in real volts with a 300 ps edge.
    `cmos` is the volts-native carrier kind, so `volts=True` says the chain already IS volts --
    the alternative, a `[lo, hi]` pair, maps a +-1 carrier onto two levels."""
    return {"ops": Signal(grid=grid).carrier("cmos", v_lo=0.0, v_hi=V_DD, f_hz=F_SIG,
                                             duty=0.5, tr_s=TR).ops,
            "volts": True}


def schedule():
    """The output stage conducts, then lets go while current is still flowing in the package."""
    return [{"t_s": T_ON, "id": "d", "state": "driving"},
            {"t_s": T_RELEASE, "id": "d", "state": "hiz"}]


def rendered(grid):
    """The driver's level as an array. `Signal.net` renders the nested `source` spec itself; it
    is done by hand here so that every variant of the board below is driven by the SAME samples,
    rather than by a waveform re-rendered each time and assumed to be identical."""
    op = dict(source(grid)["ops"][0])
    op.pop("op")
    return Signal(grid=grid).carrier(**op).waveform()


def edges(vs, fs):
    """Where the output stage's own level crosses its midpoint, read off the waveform rather
    than assumed from the request -- the same rule the net uses to threshold a `follow=` driver,
    and the only honest way to index a record by its edges."""
    on = vs >= 0.5 * (vs.min() + vs.max())
    k = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    return [(int(i), bool(on[i])) for i in k]


def peak(v, k0, window, going_high):
    """The worst excursion in the direction of travel, and when it happened. Taken over the
    whole window including the old level, which is safe BECAUSE it is the direction of travel:
    a rising edge cannot overshoot downward past where it started."""
    seg = v[k0:k0 + window]
    i = int(np.argmax(seg) if going_high else np.argmin(seg))
    return float(seg[i]), k0 + i


def first_turn(v, k0, going_up):
    """Index of the first turning point after `k0`, by the slope that ends the excursion.

    A kick is the FIRST turning point, not the largest excursion in some window: a released node
    also drifts toward its new resting level, and over a long enough window that drift is bigger
    than the kick and has nothing to do with it."""
    d = np.diff(v[k0:])
    sign = -1 if going_up else 1
    hit = np.flatnonzero(np.sign(d) == sign)
    return k0 + int(hit[0]) if hit.size else k0


def half_periods(v, k0, going_up, how_many=4):
    """Successive turning points after `k0`, alternating direction. The interval between them is
    half the ring period, which is how the frequency comes off the record rather than out of the
    request."""
    out, k, up = [], k0, going_up
    for _ in range(how_many):
        k = first_turn(v, k + 1, up)
        out.append(k)
        up = not up
    return out


def transition_time(v, k0, window, frac=(0.1, 0.9)):
    """10-90 % of a transition, interpolated, with both levels read off the record -- the start
    level from the sample before it and the settled level from the end of the window. The edge
    rate is what damping buys and what it costs, so neither end can come from the request."""
    seg = v[k0 - 1:k0 + window]
    lo, hi = float(seg[0]), float(v[k0 + window - 1])
    at = []
    for f in frac:
        lv = lo + f * (hi - lo)
        hit = np.flatnonzero((seg >= lv) if hi > lo else (seg <= lv))
        if not hit.size:
            return float("nan")
        i = int(hit[0])
        at.append(0.0 if i == 0 else i - 1 + (lv - seg[i - 1]) / (seg[i] - seg[i - 1]))
    return abs(at[1] - at[0])


def main():
    grid = ws.Grid(fs=FS, baud=F_SIG, n=N)
    vs = rendered(grid)
    out = ws.net.solve(board(), fs=FS, n=N, schedule=schedule(),
                       probes=["pad", "bus", "i(lp)"], sources={"d": vs})
    rc = ws.net.solve(board(inductive=False), fs=FS, n=N, schedule=schedule(),
                      probes="bus", sources={"d": vs})
    v, pad, il = out["bus"], out["pad"], out["i(lp)"]
    k_rel = ws.net.sample_at(T_RELEASE, FS)
    f0 = 1.0 / (2 * np.pi * np.sqrt(L_PKG * C_BUS))
    zeta = (R_OUT + R_PKG) / (2 * np.sqrt(L_PKG / C_BUS))
    window = int(round(6 / f0 * FS))             # six ring periods, about 7 time constants
    swings = [(k, g) for k, g in edges(vs, FS) if T_ON < k / FS < T_RELEASE - window / FS]
    # one of each polarity: the bus does the same thing at every edge, and saying so twice
    # demonstrates nothing the first line did not
    swings = [next(e for e in swings if e[1] is want) for want in (True, False)]
    swings.sort()
    settled = {g: float(v[k + window - 1]) for k, g in swings}

    print()
    print("THE SAME BOARD WITH AND WITHOUT ITS PACKAGE INDUCTANCE")
    print(f"  a {R_OUT:.0f} ohm output stage, {TR * 1e12:.0f} ps edges and "
          f"{L_PKG * 1e9:.0f} nH of package, into {C_BUS * 1e12:.0f} pF and "
          f"{R_TERM:.0f} ohm to {V_BIAS} V")
    print(f"  the bus rings at {f0 / 1e6:.0f} MHz with zeta = {zeta:.3f}: critical damping "
          f"would want {2 * np.sqrt(L_PKG / C_BUS):.0f} ohm in series and there is "
          f"{R_OUT + R_PKG:.1f}")
    print(f"  levels {settled[True]:.3f} V high, {settled[False]:.3f} V low, idle "
          f"{float(v[-1]):.3f} V   [a divider against the termination, not the supply]")
    print()
    print("  what the receiver sees at each edge, against its own absolute maxima:")
    for k, going_high in swings:
        lim = V_ABSMAX if going_high else V_ABSMIN
        worst, k_w = peak(v, k, window, going_high)
        flat, _ = peak(rc, k, window, going_high)
        over = (worst - lim) if going_high else (lim - worst)
        verdict = (f"EXCEEDS {lim:+.1f} V by {over * 1e3:.0f} mV" if over > 0
                   else f"{abs(over) * 1e3:.0f} mV inside {lim:+.1f} V")
        print(f"    {'rising ' if going_high else 'falling'} at {k / FS * 1e9:6.1f} ns: "
              f"{worst:+7.3f} V after {(k_w - k) / FS * 1e12:4.0f} ps   [{verdict}]")
        print(f"                       the R-C model of the same board: {flat:+7.3f} V   "
              f"[{abs(flat - lim) * 1e3:.0f} mV inside {lim:+.1f} V]")
    d = float(np.abs(v - rc).max())
    print(f"  -> the two records differ by up to {d:.3f} V, {d / np.ptp(rc) * 100:.0f} % of the "
          f"{np.ptp(rc):.3f} V swing.")
    print("     The R-C model passes both limits at every edge and is wrong about every edge.")
    print()

    # ---- letting go while current is still flowing ------------------------------------
    print("LETTING GO WHILE THE PACKAGE IS STILL CARRYING CURRENT")
    c_eff = C_PAD * C_BUS / (C_PAD + C_BUS)
    z_loop = float(np.sqrt(L_PKG / c_eff))
    f_loop = 1.0 / (2 * np.pi * np.sqrt(L_PKG * c_eff))
    i_rel = float(il[k_rel - 1])
    v_pad0 = float(pad[k_rel - 1])
    # the inductor's `a` end is the pad, so a POSITIVE current leaves the pad: a negative one is
    # charging the pad capacitance, and the kick is upward
    up = i_rel < 0
    ks = half_periods(pad, k_rel, up)
    kick = float(pad[ks[0]])
    f_meas = 0.5 / (np.diff(ks).mean() / FS)
    print(f"  the output stage lets go while carrying {abs(i_rel) * 1e3:.1f} mA. That current is "
          f"a STATE and cannot stop,")
    print(f"  so the pad capacitance takes it: {v_pad0:+.3f} V -> {kick:+.3f} V in "
          f"{(ks[0] - k_rel) / FS * 1e12:.0f} ps, a {abs(kick - v_pad0):.3f} V kick")
    print(f"  against the {abs(i_rel) * z_loop:.3f} V that {abs(i_rel) * 1e3:.1f} mA into the "
          f"loop's own {z_loop:.0f} ohm would give undamped")
    print(f"  the loop is {L_PKG * 1e9:.0f} nH against {c_eff * 1e12:.2f} pF -- the pad in "
          f"series with the bus -- so it should ring at {f_loop / 1e6:.0f} MHz:")
    print(f"  measured off the record's own turning points, {f_meas / 1e6:.0f} MHz "
          f"({(f_meas / f_loop - 1) * 100:+.1f} %)")
    lim, ov = ((V_ABSMAX, kick - V_ABSMAX) if up else (V_ABSMIN, V_ABSMIN - kick))
    other = float(pad[ks[1]])
    print(f"  the kick itself is {abs(ov) * 1e3:.0f} mV {'past' if ov > 0 else 'inside'} the "
          f"{lim:+.1f} V limit, and the ring then carries the pad")
    print(f"  back to {other:+.3f} V, {'past' if other < V_ABSMIN else 'inside'} "
          f"{V_ABSMIN:+.1f} V at the other end -- all of it INSIDE the package, where no probe")
    print(f"  reaches. The bus, which is where a probe does reach, turns at "
          f"{float(v[first_turn(v, k_rel, True)]):+.3f} V on its way to the "
          f"{V_BIAS} V bias.")
    print("  An R-C model of the same release decays to the bias monotonically: no kick, no "
          "ring, no finding.")
    print()

    # ---- what fixing it costs --------------------------------------------------------
    print("WHAT A SERIES DAMPING RESISTOR COSTS")
    print("  fitted on the same piece of wire as the inductance, so it is the inductor's own")
    print("  `ohm` rather than an element of its own:")
    print("    series       low level   worst     margin to -0.3 V   fall time (10-90 %)")
    k_fall = [k for k, g in swings if not g][0]
    rows = []
    for extra in (0.0, 4.0, 8.0, 12.0, 20.0, 2 * np.sqrt(L_PKG / C_BUS) - R_OUT - R_PKG):
        w = ws.net.solve(board(extra_ohm=extra), fs=FS, n=N, schedule=schedule(),
                         probes="bus", sources={"d": vs})
        worst, _ = peak(w, k_fall, window, False)
        tf = transition_time(w, k_fall, window) / FS
        rows.append((extra, float(w[k_fall + window - 1]), worst,
                     worst - V_ABSMIN, tf))
        tag = "  <- critically damped" if extra > 30 else ""
        print(f"    {extra:5.1f} ohm   {rows[-1][1]:+7.3f} V  {worst:+7.3f} V   "
              f"{(worst - V_ABSMIN) * 1e3:+8.1f} mV        {tf * 1e12:7.1f} ps{tag}")
    passing = [r for r in rows if r[3] > 0]
    least, crit = passing[0], rows[-1]
    print(f"  -> {least[0]:.0f} ohm is the whole fix: it clears the limit by "
          f"{least[3] * 1e3:.0f} mV and costs "
          f"{(least[4] - rows[0][4]) * 1e12:.0f} ps of fall time "
          f"({rows[0][4] * 1e12:.0f} -> {least[4] * 1e12:.0f} ps).")
    print(f"     Critical damping -- {crit[0]:.0f} ohm, no ringing at all -- costs "
          f"{(crit[4] - rows[0][4]) * 1e12:.0f} ps instead, for "
          f"{(crit[3] - least[3]) * 1e3:.0f} mV more")
    print("     margin than the data sheet asks for. The limit is what has to be met, not the "
          "absence of a ring.")
    print()

    # ---- and the whole thing as a recipe ---------------------------------------------
    # The driver carries its own chain as a nested `source` spec, so the net needs nothing
    # before it: it is the FIRST op, and the record above is reproducible from JSON alone.
    spec = board()
    spec["drivers"] = [{**spec["drivers"][0], "source": source(grid)}]
    sig = Signal(grid=grid).net(**spec, schedule=schedule(), probe="bus")
    assert np.array_equal(sig.waveform(), v), "the recipe has to be the record"
    assert Signal.from_recipe(sig.recipe()).sha256() == sig.sha256()
    print(f"REPLAYABLE: the same record as a recipe -- the net is the first op, because the "
          f"driver\n  brings its own chain -- digest {sig.sha256()[:16]}, "
          f"{len(json.dumps(sig.recipe()))} bytes of JSON")
    print()


if __name__ == "__main__":
    main()

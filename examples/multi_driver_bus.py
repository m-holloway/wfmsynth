"""Three transceivers on one bus, each with its OWN output stage: driver sources.

Run from the repository root after ``pip install -e .``:

    python examples/multi_driver_bus.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
A driver on a shared net whose Thevenin level is a WAVEFORM rather than a number -- its own
supply, its own edge rate, its own timing impairments -- rendered through its own chain and
injected through its own scheduled output conductance:

    drivers=[{"id": "ctrl", "node": "bus", "ohm": 20.0, "state": "hiz",
              "source": {"ops": Signal(grid=g).carrier("cmos", v_hi=3.3, tr_s=2e-9)
                                              .timing(rj_ps=12.0),
                         "volts": True}}, ...]

The alternative is to synthesize each party separately and add the contributions by hand. That
is easy to get wrong in one particular way: a driver that has let go is still in your sum, so
the level it happened to be holding at the moment of release is carried into the next owner's
baseline, and the bus drifts by one held offset at every change of hands. Inside the solve it
cannot happen -- a released driver is removed from the stamp, so its level is ABSENT rather
than small. This file prints both numbers: where the bus actually sits between talkers, and
the offset the by-hand route would have left there.

THE BOARD
---------
A single-ended backplane bus with a Thevenin termination -- 56 ohm to a 1.5 V bias rail, the
classic parallel-terminated arrangement -- and 60 pF of stub and input capacitance. Three
parties take it in turn, and they are deliberately not the same part:

    ctrl   3.3 V, 20 ohm output,  2 ns edges     the controller
    n1     2.5 V, 35 ohm output,  6 ns edges     a mid-range node
    n2     1.8 V, 55 ohm output, 10 ns edges     a cheap node, and the one with the problem

A driver's swing on this bus is a DIVIDER against the termination, not its own supply, so the
weakest part is penalised twice -- it starts lower and it divides harder.

WHAT THE LIBRARY DOES NOT DO HERE
---------------------------------
It has no opinion about backplanes, LVTTL thresholds or who gets to talk. The arbitration is
nine lines of holds at the bottom of this file, the receiver thresholds are two constants, and
both are the caller's. The library supplies the net, the scenario compiler and the solve.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth.scenario import Hold, Agent, hold_windows

V_BIAS, R_TERM, C_BUS = 1.5, 56.0, 60e-12
V_IH, V_IL = 2.0, 0.8              # LVTTL receiver thresholds -- the caller's, not the library's
F_BUS = 20e6                       # what each party sends while it holds the bus

# supply, output resistance, 10-90 % edge of the output stage itself
PARTIES = {"ctrl": (3.3, 20.0, 2e-9),
           "n1": (2.5, 35.0, 6e-9),
           "n2": (1.8, 55.0, 10e-9)}

T_SLOT, T_GAP, T_START = 1.2e-6, 0.8e-6, 0.4e-6
T_LEAD, T_LAG = 15e-9, 40e-9       # an output stage conducts a little either side of its slot


# ----------------------------------------------------------------- the board, and who is on it
def source(name, grid):
    """One party's output stage as a nested source spec.

    `ctrl` gets the long form -- its own `Signal` chain, so its transmitter and timing ops are
    its own -- and the other two the compact one-carrier form. Both are rendered on the net's
    grid and both end in volts, which is what `volts=True` asserts: a net is in volts, and a
    carrier is normalised amplitude of about +/-1, so a source handed over unconverted would be
    a driver a thousand times too weak and the solve would be perfectly correct about it.
    """
    v_hi, _ohm, tr = PARTIES[name]
    edge = dict(kind="cmos", v_lo=0.0, v_hi=v_hi, f_hz=F_BUS, tr_s=tr, duty=0.5)
    if name != "ctrl":
        return {**edge, "volts": True}
    # No seed on the chain: a driver's chain is part of the record and draws under the
    # record's seed by its own name (`net/0/ctrl/timing/1`), so a seed here would be ignored
    # -- and since 0.46.0 the builder says so.
    chain = (ws.Signal(grid=grid)
             .carrier(**edge)
             .timing(rj_ps=12.0))          # its own clock, with its own jitter on it
    return {"ops": chain, "volts": True}


def board(grid):
    return {"nodes": ["bus"], "rails": {"vbias": V_BIAS},
            "elements": [{"id": "rt", "kind": "r", "a": "bus", "b": "vbias", "ohm": R_TERM},
                         {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": C_BUS}],
            "drivers": [{"id": nm, "node": "bus", "ohm": ohm, "state": "hiz",
                         "source": source(nm, grid)}
                        for nm, (_v, ohm, _tr) in PARTIES.items()]}


def slots():
    """Who holds the bus, when. This is the arbitration, and it is the caller's: nine lines,
    no electricity in them, and the same list is the label for every window below."""
    return [Hold(nm, T_START + k * (T_SLOT + T_GAP), T_START + k * (T_SLOT + T_GAP) + T_SLOT,
                 talker=nm, slot=k)
            for k, nm in enumerate(PARTIES)]


def agents():
    """What holding the bus DOES: this party's output stage conducts, and nothing else moves.
    Device behaviour, with no arbitration in it."""
    return {nm: Agent(during={nm: "driving"}, after={nm: "hiz"}, lead=T_LEAD, lag=T_LAG)
            for nm in PARTIES}


# ------------------------------------------------------------------------- render and measure
def rise_time(v, t):
    """10-90 % of this window's own excursion, by interpolation between the straddling samples
    -- counting whole samples would quantise the answer to the grid (the library's rule 12).

    Taken on the first COMPLETE rising edge in the window: the first sample above the 90 %
    level that has a sample below the 10 % level somewhere before it.
    """
    lo, hi = float(v.min()), float(v.max())
    a, b = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
    up = np.flatnonzero(v >= b)
    for k in up:
        below = np.flatnonzero(v[:k] <= a)
        if not below.size:
            continue
        j = int(below[-1])

        def cross(level, i0, i1):
            i = i0 + int(np.argmax(v[i0:i1 + 1] >= level))
            f = (level - v[i - 1]) / (v[i] - v[i - 1])
            return float(t[i - 1] + f * (t[i] - t[i - 1]))

        return cross(b, j, int(k)) - cross(a, j, int(k))
    raise RuntimeError("no complete rising edge in this window")


def divider(v_drive, ohm):
    """Where the bus settles when one party drives `v_drive` through `ohm` against the
    termination. A driver's swing on a terminated bus is this, not its own supply."""
    return V_BIAS + (v_drive - V_BIAS) * R_TERM / (R_TERM + ohm)


def main():
    # Size the grid from the FASTEST edge on the board, not from the bit rate: the controller's
    # 2 ns output stage at the library's k >= 8 needs 4 GSa/s, and the quickest electrical mode
    # -- the termination in parallel with that 20 ohm output into 60 pF, 0.88 ns -- needs 2.3.
    # 5 GSa/s clears both, and the solver says so if you get it wrong.
    fs, n = 5e9, 32768
    grid = ws.Grid(fs=fs, n=n, v_full=3.3)
    holds = slots()

    # The net is the FIRST op: every driver brings its own chain, so nothing has to arrive at
    # it and its length comes from the grid. The recipe stores the holds and the agents as they
    # were authored -- and each driver's source chain as ITS ops -- so what is recorded is what
    # was meant, down to the controller's jitter.
    sig = ws.Signal(seed=1, grid=grid).net(
        **board(grid), holds=holds, agents=agents(), probe="bus")
    v = sig.waveform()
    t = np.arange(n) / fs
    assert np.array_equal(ws.Signal.from_recipe(sig.recipe()).waveform(), v)

    print(f"A terminated backplane bus: {R_TERM:.0f} ohm to {V_BIAS} V, {C_BUS * 1e12:.0f} pF, "
          f"three parties in turn")
    print(f"  {len(sig.recipe()['ops'][0]['drivers'])} drivers, each with its own chain; "
          f"{n} samples at {fs / 1e9:.1f} GSa/s")
    print()
    print("  party     supply  R_out   V_OH             V_OL             t_r "
          "(stage -> bus)")
    worst = []
    for w in hold_windows(holds, fs, n):
        nm = w["who"]
        v_hi, ohm, tr = PARTIES[nm]
        # measure inside the hold only, and after the output stage has had a slot to settle
        k0, k1 = w["start"] + int(0.1e-6 * fs), w["stop"]
        seg, ts = v[k0:k1], t[k0:k1]
        voh, vol = float(seg.max()), float(seg.min())
        got = rise_time(seg, ts)
        # a ramp into a single pole: the two 10-90 % times add in quadrature
        want = np.hypot(tr, 2.2 * (R_TERM * ohm / (R_TERM + ohm)) * C_BUS)
        print(f"  {nm:<8}  {v_hi:4.1f} V  {ohm:4.0f}   "
              f"{voh:5.3f} V ({divider(v_hi, ohm):5.3f})  "
              f"{vol:5.3f} V ({divider(0.0, ohm):5.3f})  "
              f"{tr * 1e9:5.2f} -> {got * 1e9:5.2f} ns ({want * 1e9:5.2f})")
        worst.append((nm, voh, vol))
    print(f"  (closed forms in brackets: the divider against the termination, and the stage's "
          f"own edge in quadrature with 2.2*R_out||R_term*C)")
    print()

    bad = []
    for nm, voh, vol in worst:
        note = []
        if voh < V_IH:
            note.append(f"FAILS V_IH by {(V_IH - voh) * 1e3:.0f} mV")
            bad.append(nm)
        else:
            note.append(f"{(voh - V_IH) * 1e3:.0f} mV over V_IH")
        if vol > V_IL:
            note.append(f"FAILS V_IL by {(vol - V_IL) * 1e3:.0f} mV")
            bad.append(nm)
        else:
            note.append(f"{(V_IL - vol) * 1e3:.0f} mV under V_IL")
        print(f"  {nm:<8}  high {voh:5.3f} V / low {vol:5.3f} V   "
              f"[{note[0]}, {note[1]}]")
    print(f"  -> an LVTTL input on this termination cannot read "
          f"{', '.join(dict.fromkeys(bad)) or 'nobody'}. A weak part is penalised twice here: "
          f"it starts lower AND divides harder.")
    print()

    # ---- the hand-over, which is what the driver sources are for ------------------------
    # Every party is holding SOME level when its slot ends, and it is not the idle one.
    # Inside the solve that costs nothing: a released driver is out of the stamp, so its level
    # is absent rather than small. Added up by hand outside it, that level stays in the sum.
    print("  at each hand-over:")
    naive, resid = 0.0, []
    for w in hold_windows(holds, fs, n):
        nm = w["who"]
        _v_hi, ohm, _tr = PARTIES[nm]
        # the output stage releases T_LAG after the slot ends, and the next one takes over
        # T_LEAD before the next slot begins -- so the bus is nobody's in between
        k_rel = min(n - 1, ws.net.sample_at(w["t1"] + T_LAG, fs))
        k_idle = min(n - 1, ws.net.sample_at(w["t1"] + T_GAP - T_LEAD, fs) - 1)
        held = float(v[k_rel - 1])
        idle = float(v[k_idle])
        resid.append(abs(idle - V_BIAS))
        # what a by-hand sum would leave behind: this driver's contribution, still in it
        offset = (held - V_BIAS) * R_TERM / (R_TERM + ohm)
        naive += offset
        print(f"    {nm:<8} let go holding {held:5.3f} V; the bus then idles at "
              f"{idle:9.6f} V   [a by-hand sum would leave {offset:+.3f} V behind]")
    print(f"  the bus is back at the {V_BIAS} V bias every time, to within "
          f"{max(resid):.1e} V. Summed by hand instead, the three held levels accumulate to "
          f"{naive:+.3f} V of baseline drift -- which is the bug this exists to make "
          f"unwriteable.")


if __name__ == "__main__":
    main()

"""An I2C bus across a level translator: a protocol encoder built ON this library.

Run from the repository root after ``pip install -e .``:

    python examples/i2c_bus.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
Not I2C. The library has no opinion about I2C and does not want one -- which bits a standard
puts on a wire has a revision date and a document behind it. What it has is the layer under
one: `wfmsynth.net` solves a shared net (several parties, switched elements, one solve) and
`wfmsynth.scenario` turns "who held the line when" into the net's event schedule.

This file is the layer ABOVE, the one you write. It is deliberately small, and the shape is
the point:

    bytes --> [ your protocol encoder ] --> holds --> [ scenario ] --> events --> [ net ] --> volts
                     ~40 lines, here                    library             library

The encoder knows start conditions and acknowledge slots and nothing about electricity. The
agents know about output stages and nothing about I2C. Neither knows about the other, so the
same encoder drives a different board by editing a dict, and the same board runs SPI or
1-Wire or a bespoke backplane by swapping the encoder. A fault is an edit to the holds -- and
because the holds are also the labels, the edit labels itself.

THE BOARD
---------
A 3.3 V host and a 1.8 V target, bridged by the standard single-MOSFET bidirectional
translator (gate tied to the low rail, source low side, drain high side). Each side has its
own pull-up and its own bus capacitance, which is where the interesting physics is.

TWO APPROXIMATIONS, NAMED
-------------------------
1. The real FET switches on its own terminal voltage. This tier solves lumped R and C, so it
   does not. But the scenario already knows who is pulling, so the FET's state is SCHEDULED
   from the holds -- exact for the levels and the falling edge. Its turn-OFF is the part that
   genuinely depends on a voltage (it conducts until the low side rises past Vgs(th)), and
   that is modelled as the agent's release reaction: a `then` row that opens it once the low
   side has had time to cross the gate threshold. That time is not guessed -- it is read off
   the net's own release (`fet_turn_off_delay`). A solver that knew voltages would apply the
   threshold directly at every switch -- GitHub #72 tier 2.
2. Rds_on is 60 ohm, not the 3.5 ohm on the front of a BSS138 datasheet. That figure is at
   Vgs = 4.5 V; this FET runs at Vgs = 1.8 V, a few hundred millivolts above threshold, where
   it is an order of magnitude worse. Using the headline number here would be optimistic
   about exactly the thing this example ends up finding a problem with.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth.scenario import Hold, Agent, merge_holds, hold_windows

F_SCL = 400e3          # Fast-mode. Its rise-time limit is 300 ns and its V_OL limit is 0.4 V.
VDD_A, VDD_B = 3.3, 1.8
RP_A, RP_B = 1.0e3, 2.2e3          # pull-ups, one per side
CB_A, CB_B = 100e-12, 250e-12      # bus capacitance, ditto -- the target side is the loaded one
R_PAD = 50.0                       # an open-drain output stage, pulled down
R_FET = 60.0                       # translator Rds_on AT Vgs = 1.8 V (see the docstring)
V_TH = 1.3                         # its gate threshold; what REALLY decides when it lets go

# An output stage conducts a little before the logic edge that asked for it and a little after
# the one that released it. Both are properties of the device, which is why they live on the
# agent and not in the encoder.
T_LEAD, T_LAG = 20e-9, 60e-9


# --------------------------------------------------------------------- your protocol encoder
class I2C:
    """Bytes in, holds out. Knows I2C; knows nothing about volts, nodes or time constants.

    A hold is "this party pulled this line low over this interval", tagged with whatever you
    want to label the record by. Nothing here is library API -- this is the forty lines you
    would write for your own bus, and the only thing it has to produce is `Hold`s.
    """

    def __init__(self, f_scl, t0=0.0):
        self.T = 1.0 / f_scl
        self.t = t0
        self.holds = []

    def _low(self, who, t0, t1, **tag):
        self.holds.append(Hold(who, t0, t1, **tag))

    def _clock(self, **tag):
        """One bit period: SCL low for the first half, released for the second. Data changes
        on the falling edge and is read while SCL is high -- so an SDA hold spans a whole bit
        period and two consecutive 0 bits abut exactly. `merge_holds` is what makes that one
        hold rather than a release and re-acquire in the middle of a byte."""
        self._low("host/scl", self.t, self.t + self.T / 2, **tag)
        self.t += self.T

    def start(self):
        """SDA falls while SCL is high. SCL is not clocked here, so it idles high."""
        self._low("host/sda", self.t + self.T / 2, self.t + self.T, field="start")
        self.t += self.T
        return self

    def byte(self, value, field="data"):
        for i in range(8):
            bit = (value >> (7 - i)) & 1
            if not bit:                       # open drain: a 1 is "let go", not "drive high"
                self._low("host/sda", self.t, self.t + self.T,
                          field=field, bit=i, value=0)
            self._clock(field=field, bit=i)
        return self

    def ack(self, who="target"):
        """The host lets go of SDA and the target pulls it low: the hand-over this whole
        machinery exists for. Two parties, overlapping once lead and lag are applied, which is
        two drivers conducting at once -- and that resolves in the nodal solve, not by a rule."""
        self._low(f"{who}/sda", self.t, self.t + self.T, field="ack")
        self._clock(field="ack")
        return self

    def stop(self):
        """SDA rises while SCL is high -- so SDA must be low first, then released."""
        self._low("host/sda", self.t, self.t + self.T / 2, field="stop")
        self.t += self.T
        return self


def mirror(holds, line, who):
    """One agent whose holds are the UNION of everybody else's on a line.

    The translator conducts whenever anyone is pulling, so its holds are not its own -- they
    are the merge of the host's and the target's. Doing it this way is what makes an ACK
    hand-over a single continuous hold for the FET: the two parties overlap, the union has no
    gap, and so the FET never sees a release to react to.
    """
    return merge_holds([Hold(who, h.t0, h.t1) for h in holds if h.who.endswith(line)])


# --------------------------------------------------------------------------------- the board
def board():
    """Two domains, two pull-ups, two bus capacitances, two translator FETs. Plain data."""
    elements, drivers = [], []
    for ln in ("sda", "scl"):
        for side, rail, rp, cb in (("a", "vdd_a", RP_A, CB_A), ("b", "vdd_b", RP_B, CB_B)):
            elements += [
                {"id": f"rp_{ln}_{side}", "kind": "r", "a": f"{ln}_{side}", "b": rail, "ohm": rp},
                {"id": f"cb_{ln}_{side}", "kind": "c", "a": f"{ln}_{side}", "b": "gnd",
                 "farad": cb},
            ]
        # The FET starts OPEN: an idle bus is both sides high, and nothing is conducting.
        elements.append({"id": f"q_{ln}", "kind": "r", "a": f"{ln}_a", "b": f"{ln}_b",
                         "ohm": R_FET, "state": "open"})
    # Open-drain output stages. The host's are on the 3.3 V side, the target's on the 1.8 V side.
    for did, node in (("pad_host_sda", "sda_a"), ("pad_host_scl", "scl_a"),
                      ("pad_targ_sda", "sda_b")):
        drivers.append({"id": did, "node": node, "volts": 0.0, "ohm": R_PAD, "state": "hiz"})
    return {"nodes": ["sda_a", "sda_b", "scl_a", "scl_b"],
            "rails": {"vdd_a": VDD_A, "vdd_b": VDD_B},
            "elements": elements, "drivers": drivers}


def fet_turn_off_delay(fs):
    """How long the translator keeps conducting after the line is let go.

    The FET's gate sits at VDD_B, so it conducts while ``VDD_B - v(sda_b) > V_TH``: it stops
    at a VOLTAGE, not at a time. This tier schedules switches, so that voltage is converted
    into a time ONCE, here, by solving the net's own release with the FET held closed and
    reading off the crossing. A voltage-aware solver (GitHub #72 tier 2) would do exactly this
    at every switching instant rather than once up front, and the error left behind is the
    spread between release conditions -- a few nanoseconds on this board.

    Calibrating it beats picking a round number: the delay sets how long the 3.3 V side drags
    the 1.8 V side up with it, and that is the effect the measurements below are about.
    """
    spec = board()
    for d in spec["drivers"]:
        d["state"] = "driving" if d["id"] == "pad_targ_sda" else "hiz"
    for e in spec["elements"]:
        if e["id"].startswith("q_"):
            e["state"] = "closed"
    n = int(round(4e-6 * fs))
    k = n // 4
    v = ws.net.solve(spec, fs=fs, n=n, probes="sda_b",
                     schedule=[{"t_s": k / fs, "id": "pad_targ_sda", "state": "hiz"}])
    above = np.flatnonzero(v[k:] >= VDD_B - V_TH)
    if not above.size:
        raise RuntimeError("the low side never reaches the FET threshold; it cannot turn off")
    return float(above[0]) / fs


def agents(t_fet_off):
    """What holding a line DOES. Device behaviour, with no protocol in it."""
    pad = dict(lead=T_LEAD, lag=T_LAG)
    out = {
        "host/sda": Agent(during={"pad_host_sda": "driving"},
                          after={"pad_host_sda": "hiz"}, **pad),
        "host/scl": Agent(during={"pad_host_scl": "driving"},
                          after={"pad_host_scl": "hiz"}, **pad),
        "target/sda": Agent(during={"pad_targ_sda": "driving"},
                            after={"pad_targ_sda": "hiz"}, **pad),
    }
    for ln in ("sda", "scl"):
        # `after` keeps the FET conducting; `then` opens it `t_fet_off` later. That ordering
        # is the physics: the gate does not know the line was released until the low side has
        # risen past its threshold. It also means the two sides rise TOGETHER until then --
        # which is the effect the measurements below are about.
        out[f"xlate/{ln}"] = Agent(during={f"q_{ln}": "closed"},
                                   after={f"q_{ln}": "closed"},
                                   then=[(t_fet_off, f"q_{ln}", "open")])
    return out


# ------------------------------------------------------------------------- render and measure
def rise_time(v, t, lo_frac=0.1, hi_frac=0.9):
    """10-90 % of the excursion actually present in this window, by linear interpolation
    between the straddling samples -- not by counting whole samples, which quantises the
    answer to the grid (see the library's own rule 12)."""
    v0, v1 = float(v[0]), float(v.max())
    lo, hi = v0 + lo_frac * (v1 - v0), v0 + hi_frac * (v1 - v0)

    def cross(level):
        k = int(np.argmax(v >= level))
        if k == 0:
            return float(t[0])
        f = (level - v[k - 1]) / (v[k] - v[k - 1])
        return float(t[k - 1] + f * (t[k] - t[k - 1]))

    return cross(hi) - cross(lo)


def main():
    bus = I2C(F_SCL, t0=2e-6)
    bus.start().byte(0x3C << 1, field="addr").ack("target").stop()
    holds = bus.holds + mirror(bus.holds, "/sda", "xlate/sda") \
                      + mirror(bus.holds, "/scl", "xlate/scl")

    # Size the grid from the FASTEST thing on the board, not from the bit rate. The quickest
    # mode here is charge redistribution through the FET -- Rds_on against the two bus
    # capacitances in series, about 2.4 ns -- so a bit-rate-derived grid would be out by three
    # orders of magnitude and every edge measured off it would be the grid's. The solver says
    # so if you get this wrong, rather than quietly returning a staircase.
    fs, n = 1.5e9, 54000
    grid = ws.Grid(fs=fs, n=n, v_full=VDD_A)
    t_fet_off = fet_turn_off_delay(fs)
    sched = ws.scenario.schedule(holds, agents(t_fet_off))
    v = ws.net.solve(board(), fs=fs, n=n, schedule=sched,
                     probes=["sda_a", "sda_b", "scl_a", "scl_b"])
    t = np.arange(n) / fs

    # The same chain through `Signal`, so the record carries a recipe. `holds` and `agents` go
    # in as they were authored and the recipe stores THAT -- not the 60-odd events they expand
    # to -- so what is recorded is what was meant.
    sig = (ws.Signal(seed=1, grid=grid)
           .carrier("dc", level=0.0)
           .net(**board(), holds=[h.to_dict() for h in holds],
                agents={k: a.to_dict() for k, a in agents(t_fet_off).items()},
                probe="sda_a"))
    assert np.allclose(sig.waveform(), v["sda_a"], rtol=0, atol=0)

    # ---- what the record says about the board -------------------------------------------
    # Measure the rising edge at the STOP, where SDA is released and the bus goes idle. NOT
    # the one after the ACK: a STOP needs SDA low first, so the host pulls it down again
    # immediately and that edge never completes. An edge measured over a window something
    # else interrupts reports the window.
    rel = max(h.t1 for h in bus.holds if h.who == "host/sda")
    w = slice(ws.net.sample_at(rel + T_LAG, fs), n)
    tr_a, tr_b = rise_time(v["sda_a"][w], t[w]), rise_time(v["sda_b"][w], t[w])

    # V_OL on the 3.3 V side while the 1.8 V TARGET is the one pulling, read in the middle of
    # the ACK bit: the target's output stage and Rds_on are in series against the host side's
    # pull-up, and only the pad's own 50 ohm would be in the budget if you forgot the FET was
    # in the path at all.
    ack = max(h.t1 for h in bus.holds if h.who == "target/sda")
    mid = ws.net.sample_at(ack - 0.5 / F_SCL, fs)
    vol_a, vol_b = float(v["sda_a"][mid]), float(v["sda_b"][mid])

    print(f"I2C write to 0x3C at {F_SCL/1e3:.0f} kHz, 3.3 V host / 1.8 V target, "
          f"{len(sched)} events from {len(holds)} holds")
    print(f"  translator lets go {t_fet_off*1e9:.0f} ns after a release "
          f"(calibrated: the 1.8 V side reaching {VDD_B - V_TH:.1f} V)")
    print(f"  t_r after the ACK   side A (3.3 V) {tr_a*1e9:7.1f} ns   "
          f"side B (1.8 V) {tr_b*1e9:7.1f} ns   [Fast-mode limit 300 ns]")
    print(f"  V_OL, target driving side A (3.3 V) {vol_a:7.3f} V   "
          f"side B (1.8 V) {vol_b:7.3f} V   [Fast-mode limit 0.400 V]")
    for name, val, lim in (("t_r side A", tr_a, 300e-9), ("t_r side B", tr_b, 300e-9),
                           ("V_OL side A", vol_a, 0.4), ("V_OL side B", vol_b, 0.4)):
        print(f"    {name:12s} {'PASS' if val <= lim else 'FAIL'}  "
              f"({100*val/lim:.0f} % of the limit)")

    # ---- the holds are also the labels ---------------------------------------------------
    # One list produced the waveform and describes it, so they cannot disagree. These windows
    # go straight into `events.label_windows` beside measured attributes; the row it returns
    # keeps `i`, which is how you get `who` and the tags back.
    win = hold_windows(bus.holds, fs, n)
    print(f"  {len(win)} labelled windows, e.g. "
          f"{ {k: win[9][k] for k in ('i', 'start', 'stop', 'who', 'field')} }")
    drove = {}
    for wd in win:
        drove[wd["who"]] = drove.get(wd["who"], 0) + (wd["stop"] - wd["start"])
    print("  bus occupancy: " + ", ".join(f"{k} {100*s/n:.1f} %" for k, s in sorted(drove.items())))


if __name__ == "__main__":
    main()

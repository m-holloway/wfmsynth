"""One capture, composed: why the eye is clean on the preamble and closed on the payload.

Run from the repository root after ``pip install -e .``:

    python examples/composed_capture.py

WHAT THIS IS AN EXAMPLE OF
--------------------------
The composition ops used TOGETHER, on the kind of record a lab actually takes: a forwarded-clock
link on two scope channels, the scope triggered from the clock.

    sequence     the transmitter idles, sends a clock-like preamble, then the payload
    use          clock and data go through ONE PLL, ONE supply and ONE scope, by name
    gain(by=)    the supply's ripple rides on both output stages
    delay(by=)   the data trace is longer than the clock trace
    combine      a neighbouring 133 MHz clock leaks onto the data line
    ws.tol       the board's loss is a toleranced part, and both lines are on that board
    trigger(on=) the scope triggers on the clock and cuts both channels at the same instants

THE QUESTION
------------
The PLL is specified at 8 ps rms. Triggered on the clock, the scope shows the data edges about
6 ps wide during the preamble and more than twice that during the payload. Same PLL, same
line, same scope, half a microsecond apart. Which of the things in this record is it?

Every candidate is one named thing in the recipe, so each is tested by changing exactly that
and measuring the edges again. Nothing is fitted. The answer is not the PLL, and it is a
different thing in each part: the neighbour on the preamble, the channel on the payload.

WHAT THE LIBRARY DOES NOT DO HERE
---------------------------------
It has no opinion about the remedy. Section 4 shows what two taps of transmit de-emphasis do
to the number, including the setting that makes it worse; whether to spend them is engineering.
It also does not decide what "the jitter" is: the clock against an ideal time base and the data
against the clock are different measurements, and both are printed.
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfmsynth as ws
from wfmsynth import Signal

BAUD = 1e9
UI = 1.0 / BAUD
FS, N = 64e9, 1 << 18                           # 9.6 samples on the edge; 4096 UI of record
N_UI = int(N / FS * BAUD)
GRID = ws.Grid(fs=FS, baud=BAUD, n=N)

T_IDLE, T_PREAMBLE = 64 * UI, 448 * UI          # then the payload to the end of the record
PLL_PS = 8.0                                    # the PLL's data-sheet jitter, rms
SKEW_S = 300e-12                                # the data trace is this much longer
PRE_S, POST_S = 1 * UI, 3 * UI                  # the scope's window about each trigger
SEGMENTS, HOLDOFF_S = 192, 20 * UI


def clock():
    """A clock as a 1010 PATTERN, so its edges sit on the symbol clock exactly. The analog
    `square` and `cmos` carriers put each edge on the nearest sample, which at this rate is
    15.6 ps peak to peak of jitter that is the grid's and not the link's -- more than anything
    this example is trying to measure."""
    return Signal().carrier("nrz", n_ui=N_UI, pattern="clock")


def link(loss_db=6.0, hum=0.03, ripple=0.04, noise_rms=0.01, pll_ps=PLL_PS, skew_s=SKEW_S,
         one_pll=True, ffe=None, segments=SEGMENTS):
    """The whole capture as one record. Every argument is one thing in it, so that a variant
    differs from the capture in exactly that."""
    tx = Signal().sequence([
        {"chain": Signal().carrier("dc", level=-1.0), "duration_s": T_IDLE},
        {"chain": clock(), "duration_s": T_PREAMBLE,
         "state": "continue"},
        {"chain": Signal().carrier("nrz", n_ui=N_UI, pattern="prbs7"), "state": "continue"}])
    if ffe:
        tx = tx.tx_ffe(ffe, pre=0)

    def pll():
        return Signal().timing(phase_noise=dict(rms_ps=pll_ps))

    def board(db):                              # both lines are etched on the same board
        return ws.tol(db, 15, group="board")

    shared = {
        "tx": tx,
        "vdd": Signal().carrier("sine", f_hz=700e3, amp=ripple, offset=1.0),
        "pll": pll(),
        # the clock as it arrives at the scope's trigger input
        "clk_line": (clock().gain(by=Signal().use("vdd"))
                     .use("pll").lossy(loss_db=board(loss_db / 2), loss_at_ghz=0.5)),
        "scope": Signal().trigger(on=Signal().use("clk_line"), pre_s=PRE_S, post_s=POST_S,
                                  level=0.0, n_segments=segments, holdoff_s=HOLDOFF_S),
    }
    if not one_pll:
        shared["pll_data"] = pll()              # the same design, a second oscillator
    data = (Signal().use("tx").gain(by=Signal().use("vdd"))
            .use("pll" if one_pll else "pll_data")
            .delay(by=skew_s)
            .lossy(loss_db=board(loss_db), loss_at_ghz=0.5)
            .combine(Signal().carrier("sine", f_hz=133e6, amp=hum))
            .rx_noise(rms=noise_rms).use("scope"))
    clk = Signal().use("clk_line").rx_noise(rms=noise_rms).use("scope")
    return ws.Channels({"clk": clk, "data": data}, shared=shared, seed=1, grid=GRID)


def data_edges(rec):
    """Every data edge the scope captured, from the record and its ground truth alone:
    ``(part of the transmission, seconds after the clock trigger)``. The trigger events say
    where each window came from; the sequence's own layout says which part of the transmission
    that was. Also returns the instants the clock fired, in the long render."""
    x, events = rec.realize()
    fires = [e for e in events["data"] if e.kind == "trigger"]
    n_win = len(x["data"]) // len(fires)
    plan = ws.sequence.layout(rec.chains["tx"][0]["segments"], GRID)
    skew_s = next(o["by"] for o in rec.chains["data"] if o["op"] == "delay")
    out = []
    for e in fires:
        k = e.params["segment"]
        row = x["data"][k * n_win:(k + 1) * n_win]
        t_trigger = e.t_s - k * n_win / FS                  # where the clock fired, in the window
        # when that part left the transmitter (before the line has anything, it idles)
        t_sent = max(e.params["timestamp_s"] - skew_s, 0.0)
        part = next(s["segment"] for s in plan if s["t_start_s"] <= t_sent < s["t_stop_s"])
        for t in ws.trigger.find(row, FS, level=0.0, slope="either")["t_s"]:
            out.append((part, t - t_trigger))
    return np.array(out), np.array([e.params["timestamp_s"] for e in fires])


def spread_ps(t_s, period_s=UI):
    """rms of edge times about their own grid of `period_s`, in ps."""
    phase = np.angle(np.mean(np.exp(2j * np.pi * t_s / period_s))) / (2 * np.pi) * period_s
    d = (t_s - phase + period_s / 2) % period_s - period_s / 2
    return 1e12 * float(np.sqrt(np.mean(d ** 2)))


def measured(rec):
    """``{"preamble": ps, "payload": ps, "clock": ps, "edges": (n, n)}``: the data edges of
    each part against the clock trigger, and the clock's own trigger instants against an ideal
    time base."""
    edges, fired = data_edges(rec)
    pre, pay = (edges[edges[:, 0] == k, 1] for k in (1, 2))
    return {"preamble": spread_ps(pre), "payload": spread_ps(pay),
            "clock": spread_ps(fired, 2 * UI), "edges": (len(pre), len(pay))}


VARIANTS = [
    ("PLL jitter removed", dict(pll_ps=1e-6)),
    ("a PLL each, not one shared", dict(one_pll=False)),
    ("trace mismatch 0.3 -> 30 ns", dict(skew_s=30e-9)),
    ("supply ripple removed", dict(ripple=0.0)),
    ("scope noise removed", dict(noise_rms=1e-9)),
    ("133 MHz neighbour removed", dict(hum=0.0)),
    ("channel loss 6 -> 0.5 dB", dict(loss_db=0.5)),
]


def main():
    warnings.simplefilter("ignore")
    rec = link()

    print("THE CAPTURE")
    print(rec.summary())
    print("\nre-rollable:", ", ".join(rec.roles()))

    base = measured(rec)
    print(f"\n1. WHAT THE SCOPE SHOWS  ({SEGMENTS} windows, triggered on the clock)")
    print(f"   clock, its trigger instants against an ideal time base   {base['clock']:6.2f} ps rms"
          f"   <- the PLL ({PLL_PS:.0f} ps)")
    for part, n in zip(("preamble", "payload"), base["edges"]):
        print(f"   data edges in the {part:8s} against the clock trigger   {base[part]:6.2f} ps rms"
              f"   ({n} edges)")

    print("\n2. ONE THING AT A TIME  (data edges against the clock trigger, ps rms)")
    print(f"   {'':30s}{'preamble':>10s}{'payload':>10s}")
    print(f"   {'the capture':30s}{base['preamble']:10.2f}{base['payload']:10.2f}")
    found = {}
    for what, change in VARIANTS:
        found[what] = m = measured(link(**change))
        print(f"   {what:30s}{m['preamble']:10.2f}{m['payload']:10.2f}")
    n_pre, n_pay = base["edges"]
    print(f"   ({n_pre} and {n_pay} edges: an rms is known to about "
          f"{100 / np.sqrt(2 * n_pre):.0f} % and {100 / np.sqrt(2 * n_pay):.0f} %.)\n"
          "   The PLL moves clock and data together, so a scope triggered on the clock does not\n"
          "   see it; give each its own and it is back. The preamble's edges are spread by the\n"
          "   NEIGHBOUR more than by anything else, by as much as the channel has slowed the edge\n"
          "   it lands on. The payload's are spread by the DATA: each edge arrives on the tail of\n"
          "   the bits before it. A clock pattern gives every edge the same history, so the same\n"
          "   channel leaves the preamble alone once it is past the idle level's own tail.")

    print("\n3. UNIT TO UNIT  (the board's loss is +/-15 %: one part, both lines)")
    units = []
    for seed in range(1, 6):
        unit = rec.unit(seed)
        part = {r["chain"]: r["value"] for r in ws.tolerance.resolved(unit.recipe())}
        m = measured(unit)
        units.append((part["data"], m["payload"]))
        print(f"   unit {seed}: data line {part['data']:.2f} dB, clock line "
              f"{part['clk_line']:.2f} dB  ->  payload {m['payload']:6.2f} ps, "
              f"preamble {m['preamble']:5.2f} ps")

    print("\n4. A REMEDY, MEASURED  (transmit de-emphasis, main and one post-cursor tap)")
    ffe = {}
    for taps in ([1.0], [0.9, -0.1], [0.8, -0.2]):
        ffe[taps[-1]] = m = measured(link(ffe=taps))
        print(f"   taps {str(taps):13s} payload {m['payload']:6.2f} ps, "
              f"preamble {m['preamble']:5.2f} ps")
    print("   A little helps and more over-corrects: the right amount is the channel's, and the\n"
          "   channel is the part that varies from unit to unit.")

    print("\n5. WHAT IS REFUSED: more windows than the record holds")
    try:
        link(segments=400).waveforms()
    except ValueError as e:
        print("   " + str(e).split(". The long render")[0] + ". [...] "
              + str(e)[str(e).index("It needs about"):].split(";")[0] + ".")

    again = ws.Channels.from_recipe(json.loads(rec.to_json()))
    same = all(np.array_equal(a, b) for a, b in zip(again.waveforms().values(),
                                                    rec.waveforms().values()))
    print(f"\nrecipe {rec.sha256()[:16]}: {len(rec.to_json())} bytes of JSON, replayed "
          f"{'bit-identically' if same else 'DIFFERENTLY'}")
    return {"capture": base, "variants": found, "units": units, "ffe": ffe, "replayed": same}


if __name__ == "__main__":
    main()

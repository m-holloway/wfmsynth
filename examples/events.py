"""
Localized events — needles in a long record, labelled per window.

Placement (when) is independent of mechanism (what). This example plants
data-locked runts, a run-locked droop, edge-excited ringing, and an
asynchronous glitch, then joins realized event times to ±1 UI windows.

An external recovered-clock folder should pass its sampling instants to
``windows_from_centers`` instead of ``nominal_ui_windows``. This library does
not cut segments or run CDR here.

    python examples/events.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import wfmsynth as ws

g = ws.Grid(fs=10e9, baud=1e9, n=1 << 14)
n_ui = int(g.n // g.samples_per_ui)

sig = (ws.Signal(seed=11, grid=g)
       .carrier("nrz", n_ui=n_ui, causal=True)
       .events("runt", on="symbols", count=4, severity=0.65, floor=0.28)
       .events("droop", on="pattern", min_run=6, count=1, severity=0.55, depth=0.35)
       .events("ring", on="edges", which="rising", count=2, severity=0.4,
               f0_hz=250e6, tau_s=8e-9)
       .events("glitch", on="poisson", rate_hz=2e6, severity=0.5, amp=0.35))

x, events = sig.realize()
print(f"record: {len(x)} samples, {n_ui} UI, {len(events)} realized events")
for e in events.to_dicts():
    t_ns = 1e9 * e["t_s"] if e.get("t_s") is not None else float("nan")
    ui = e.get("ui", "-")
    print(f"  {e['kind']:12s}  ui={str(ui):>4}  sample={e['sample']:<6}  t={t_ns:8.1f} ns")

# ±1 UI windows on the known baud (not a recovered clock)
windows = ws.nominal_ui_windows(len(x), g, half_ui=1.0)
rows = ws.label_windows(windows, events, x=x, eye_low=0.12)
n_hit = sum(1 for r in rows if r["events"])
n_eye = sum(1 for r in rows if "eye_violation" in r["labels"])
print(f"\n{len(rows)} UI windows; {n_hit} overlap a requested event "
      f"(a ring tail spans many UIs); {n_eye} fail the height mask")

print("sample labelled windows (requested kinds + measured height):")
shown = 0
for r in rows:
    if not r["events"]:
        continue
    kinds = ",".join(r["kinds"])
    h = r["measured"]["height"]
    print(f"  UI {r['i']:<5}  kinds={kinds:<16}  height={h:+.3f}  labels={r['labels']}")
    shown += 1
    if shown == 6:
        break

# recovered-clock hook: same join, your instants
centers = [r["center"] for r in windows[100:104]]
alt = ws.windows_from_centers(centers, len(x), half=g.samples_per_ui)
print(f"\nwindows_from_centers: {len(alt)} windows from 4 external instants")

# source-level runt (mutate symbols, then shape) — cleaner than painting a pulse
tx = ws.physics.carrier_symbols("nrz", 64, seed=2)
ui = int(np.flatnonzero(tx[1:] != tx[:-1])[3] + 1)
ev = ws.place_events(1024, kind="runt", on="symbols", n_ui=64, symbols=tx,
                     indices=[ui], severity=0.8, floor=0.2)
tx_r = ws.defect_symbols(tx, ev)
print(f"defect_symbols: UI {ui}  {tx[ui]:+.2f} -> {tx_r[ui]:+.2f}  (from {tx[ui-1]:+.2f})")

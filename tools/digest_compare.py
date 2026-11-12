"""Did a change move anything that already existed? Digest a tree, then compare two digests.

    python tools/digest_compare.py write before.json --tree ../wfmsynth-main
    python tools/digest_compare.py write after.json                  # this checkout
    python tools/digest_compare.py compare before.json after.json    # exit 1 if anything moved

`write` renders, on the tree it is pointed at:

  * a fixed set of chains written only with calls every release since 0.42 has had -- each
    signal-taking parameter in its bare-spec form, the instrument and acquisition path, the
    capture forms, a lead-in, a `net` -- and records for each the digest of its recipe (the
    version field removed), samples, JSON replay, event list and one re-rolled sibling;
  * every example in `examples/`, with `Signal._run` wrapped so each render the example makes
    is recorded as recipe digest -> sample digest.

`compare` reports what the two files have in common, what differs, and what only one has,
the example renders grouped by the example that made them. A key is a chain's name or a
recipe's digest, so a recipe that itself changed shows up as "only in", not as a difference:
read both lists. (An example that embeds a temporary path in a recipe is "only in" both.)

It is a tool and not a test because the reference is ANOTHER CHECKOUT (`git worktree add`, or
`git archive <rev> | tar -x -C dir`). Run the tree's own `egg_info` first if its version
reads stale. `tests/test_byte_identity.py` is the in-tree pin of the default path.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import runpy
import sys
import tempfile
import warnings
from pathlib import Path

SKIP_EXAMPLES = ("fill_zarr.py", "replay.py")      # need an archive / a recipe file as input


def _h(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


def _recipe_digest(recipe):
    r = {k: v for k, v in recipe.items() if k != "wfmsynth_version"}
    return hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _chains(tmp):
    import numpy as np
    from wfmsynth import AcquisitionProfile, Grid, Signal
    g = Grid(fs=64e9, baud=16e9, n=4096)
    gs = Grid(fs=1e8, baud=1e6, n=4096)
    ga = Grid(fs=80e9, baud=10e9, n=1 << 14)
    J = dict(rj=0.4)
    cap = os.path.join(tmp, "cap.npy")
    np.save(cap, np.sin(np.arange(4096) * 0.01))
    net = dict(nodes=["bus"], rails={"vdd": 3.3}, probe="bus",
               elements=[{"id": "rp", "kind": "r", "a": "bus", "b": "vdd", "ohm": 885.0},
                         {"id": "cb", "kind": "c", "a": "bus", "b": "gnd", "farad": 40e-12}])

    def base():
        return Signal(seed=3, grid=g).carrier("nrz", n_ui=1024, pattern="prbs7", jitter=J)

    def link():
        return Signal(seed=1, grid=ga).carrier("nrz", n_ui=ga.n // 8, pattern="prbs13")

    def sine():
        return Signal(seed=1, grid=g).carrier("sine", f_hz=4e9)

    return {
        "crosstalk default": lambda: base().crosstalk(coupling=0.1),
        "crosstalk spec": lambda: base().crosstalk(
            coupling=0.2, kind="next", aggressor=dict(n_ui=512, seed=9, jitter=J)),
        "crosstalk pam4": lambda: base().lossy(loss_db=6.0, loss_at_ghz=8.0).crosstalk(
            aggressor=dict(kind="pam4", n_ui=1024, jitter=J)).digitize(snr_db=30.0),
        "crosstalk_matrix": lambda: base().crosstalk_matrix(couplings=[0.05, 0.03]),
        "modulate am": lambda: sine().modulate(message=dict(f_hz=1e8), depth=0.5),
        "modulate noise": lambda: sine().modulate(message=dict(kind="noise", rms=0.2), depth=0.5),
        "modulate fsk": lambda: sine().modulate(
            kind="fsk", fc_hz=4e9, dev_hz=5e8, message=dict(kind="nrz", n_ui=64, jitter=J)),
        "open_drain": lambda: Signal(seed=2, grid=gs).carrier("nrz", n_ui=40).open_drain(
            r_pullup_ohm=1e3, c_bus_f=2e-9),
        "open_drain second": lambda: Signal(seed=2, grid=gs).carrier("nrz", n_ui=40).open_drain(
            r_pullup_ohm=1e3, c_bus_f=2e-9, second=dict(n_ui=32, seed=9, jitter=J),
            r_sink_b_ohm=40.0),
        "hybrid_echo": lambda: base().hybrid_echo(
            isolation_db=20.0, td_samples=5, own=dict(n_ui=1024, seed=4, jitter=J)),
        "pass_fet": lambda: Signal(seed=2, grid=gs).carrier("sine", f_hz=1e6).pass_fet(
            gate=dict(f_hz=2e5)),
        "events aggressor": lambda: base().events(
            "glitch", on="aggressor", aggressor=dict(n_ui=64, seed=2), count=4),
        "events runt": lambda: link().events("runt", on="symbols", count=3).rx_noise(rms=0.01),
        "net": lambda: Signal(seed=5, grid=gs).net(**net, drivers=[
            {"id": "tgt", "node": "bus", "ohm": 30.0, "state": "driving",
             "source": {"kind": "nrz", "n_ui": 64, "seed": 5, "jitter": J,
                        "volts": [0.0, 1.8]}}]),
        "link": lambda: link().tx_ffe([-0.1, 0.8, -0.1]).nonlinearity(compression=0.1)
        .lossy(loss_db=8, loss_at_ghz=5, causal=True)
        .reflect(td_ps=55, gamma_s=0.3, gamma_l=0.3).ctle(fz_ghz=2, fp1_ghz=6, fp2_ghz=12),
        "instrument": lambda: link().probe(bw_hz=20e9, noise_rms=0.01).scope(bw_hz=16e9)
        .digitize(snr_db=35, bits=8).store(bits=8),
        "interleave": lambda: Signal(seed=2, grid=ga).carrier(
            "pam4", n_ui=ga.n // 8, pattern="prbs13q").digitize(
            snr_db=32, enob=5.5, interleave=dict(m_cores=4, offset_mm=0.01)),
        "acquire": lambda: link().acquire(AcquisitionProfile(
            sample_rate_hz=40e9, record_length=ga.n // 2, input_bandwidth_hz=13e9, bits=8)),
        "digitize n_out": lambda: link().digitize(n_out=6000, bits=8),
        "lead_in": lambda: Signal(seed=11, grid=ga, lead_in=True).carrier(
            "nrz", n_ui=ga.n // 8, pattern="prbs13").lossy(loss_db=8, loss_at_ghz=5, causal=True),
        "coded": lambda: Signal(seed=7, grid=ga).coded("64b66b", prbs=13, n_bits=1024),
        "optical": lambda: Signal(seed=9, grid=ga).carrier("pam4", n_ui=ga.n // 8).eo(kind="mzm")
        .fiber(length_km=2.0).edfa().photodetect().tia(bw_hz=8e9),
        "square": lambda: Signal(seed=1, grid=Grid(fs=10e9, n=4096)).carrier("square", f_hz=10e6),
        "cmos": lambda: Signal(seed=5, grid=Grid(fs=10e9, n=4096)).carrier(
            "cmos", v_lo=0.0, v_hi=3.3, duty=0.5, f_hz=10e6, tr_s=2e-9),
        "capture values": lambda: Signal(seed=6, grid=Grid(fs=1e9, n=64)).capture(
            values=list(np.cos(np.arange(64) * 0.2)), fs_hz=1e9).probe(bw_hz=2e8),
        "capture embed": lambda: Signal(seed=6, grid=Grid(fs=1e9, n=4096)).capture(
            path=cap, embed=True),
        "capture resample": lambda: Signal(seed=6, grid=Grid(fs=2e9, n=8192)).capture(
            path=cap, fs_hz=1e9, resample=True, embed=True),
        "no grid": lambda: Signal(seed=12).carrier("nrz", n_ui=64, n=2048).ac_couple(
            fc_frac=0.001),
    }


def write(out, tree):
    tree = Path(tree).resolve()
    sys.path.insert(0, str(tree))
    warnings.simplefilter("ignore")
    import wfmsynth
    from wfmsynth.compose import Signal
    if Path(wfmsynth.__file__).resolve().parent.parent != tree:
        sys.exit(f"wfmsynth was imported from {wfmsynth.__file__}, not from {tree}")
    got = {"tree": str(tree), "version": wfmsynth.__version__, "chains": {}, "renders": {},
           "where": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for name, build in _chains(tmp).items():
            try:
                s = build()
                roles = [r for r in s.roles() if not r.startswith("unit/")]
                got["chains"][name] = {
                    "recipe": _recipe_digest(s.recipe()),
                    "samples": _h(s.waveform()),
                    "replay": _h(Signal.from_recipe(json.loads(s.to_json())).waveform()),
                    "events": hashlib.sha256(json.dumps(
                        s.event_list().to_dicts(), sort_keys=True,
                        default=float).encode()).hexdigest()[:16],
                    "sibling": _h(s.contrast(roles[0], seed=11)) if roles else "-"}
            except Exception as e:                   # an older tree may not have the call
                got["chains"][name] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
        run = Signal._run

        def recording(self, *a, **k):
            y = run(self, *a, **k)
            try:
                key = _recipe_digest(self.recipe())
            except Exception:                        # one channel of several: no recipe of its own
                return y
            got["renders"][key] = _h(y[0] if isinstance(y, tuple) else y)
            got["where"][key] = os.path.basename(sys.argv[0])
            return y

        Signal._run = recording
        argv, cwd = sys.argv, os.getcwd()
        os.chdir(tmp)                                # examples that write files write them here
        try:
            for ex in sorted((tree / "examples").glob("*.py")):
                if ex.name in SKIP_EXAMPLES:
                    continue
                sys.argv = [str(ex)]
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        runpy.run_path(str(ex), run_name="__main__")
                except SystemExit:
                    pass
                except Exception as e:
                    print(f"example {ex.name} failed: {e!r}"[:200], file=sys.stderr)
        finally:
            Signal._run, sys.argv = run, argv
            os.chdir(cwd)
    Path(out).write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
    print(f"{wfmsynth.__version__} at {tree}: {len(got['chains'])} chains, "
          f"{len(got['renders'])} example renders -> {out}")


def compare(a, b):
    A, B = (json.loads(Path(p).read_text()) for p in (a, b))
    moved = 0
    for kind in ("chains", "renders"):
        x, y = A[kind], B[kind]
        both = sorted(set(x) & set(y))
        differ = [k for k in both if x[k] != y[k]]
        moved += len(differ)
        print(f"{kind}: {len(both)} in common, {len(differ)} differ, "
              f"{len(set(x) - set(y))} only in {a}, {len(set(y) - set(x))} only in {b}")
        if kind == "chains":
            for k in differ:
                print(f"  DIFFERS  {k}: "
                      f"{', '.join(sorted(f for f in x[k] if x[k].get(f) != y[k].get(f)))}")
            for k in sorted(set(x) ^ set(y)):
                print(f"  only in {a if k in x else b}: {k}")
        else:                                        # by the example that made the render
            where = {**B.get("where", {}), **A.get("where", {})}
            for label, keys in (("DIFFERS", differ), (f"only in {a}", set(x) - set(y)),
                                (f"only in {b}", set(y) - set(x))):
                by = {}
                for k in keys:
                    by[where.get(k, "?")] = by.get(where.get(k, "?"), 0) + 1
                for ex in sorted(by):
                    print(f"  {label}  {ex}: {by[ex]} render{'s' if by[ex] > 1 else ''}")
    return 1 if moved else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="digest a tree to a JSON file")
    w.add_argument("out")
    w.add_argument("--tree", default=str(Path(__file__).resolve().parent.parent),
                   help="the checkout to digest (default: the one this file is in)")
    c = sub.add_parser("compare", help="compare two digest files; exit 1 if anything differs")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "write":
        write(args.out, args.tree)
    else:
        sys.exit(compare(args.a, args.b))

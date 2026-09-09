"""Load the SHIPPING eye reduction -- the demo app's engine kernel -- into this
process, unmodified, so anything measured here is measured against the code that
actually runs and never against a restatement of it.

PINNED, and deliberately. The reduction lives in the demo repo as a Python template
literal inside src/engine/kernel.ts, and that file is under active development by
another unit: it changed twice while this benchmark was being written, and one of the
changes moved a returned array by 1667 elements. A baseline that moves is not a
baseline. So the comparison is made against ``BASELINE_REV`` -- the commit whose
ground-truth and jitter-transfer evidence this work must not disturb -- materialised
read-only into the scratch directory with ``git archive``. Nothing here writes to the
demo checkout. Set WFMSYNTH_EYE_BASELINE to compare against a different revision.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DEMO = Path(os.environ.get("WFMSYNTH_DEMO", str(Path.home() / "dev" / "wfmsynth_demo")))
BASELINE_REV = os.environ.get("WFMSYNTH_EYE_BASELINE", "b37483b")
SCRATCH = Path(os.environ.get("WFMSYNTH_SCRATCH", "/tmp"))


def _materialise():
    """The demo's src/ and spikes/ at BASELINE_REV, in the scratch directory."""
    rev = subprocess.run(["git", "-C", str(DEMO), "rev-parse", BASELINE_REV],
                         capture_output=True, text=True, check=True).stdout.strip()
    root = SCRATCH / ("eye3m_baseline_" + rev[:12])
    stamp = root / ".complete"
    if not stamp.exists():
        root.mkdir(parents=True, exist_ok=True)
        tar = subprocess.run(["git", "-C", str(DEMO), "archive", rev, "src", "spikes"],
                             capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(root)], input=tar, check=True)
        # One constant in the engine kernel is COMPUTED by the TypeScript beside it,
        # and kernel_load evaluates it with the repo's own esbuild rather than
        # restating it in Python. Point the materialised tree at the checkout's
        # node_modules -- a symlink here, nothing written there.
        link = root / "node_modules"
        if not link.exists():
            link.symlink_to(DEMO / "node_modules")
        stamp.write_text(rev + "\n")
    return root, rev


def load():
    """The shipping kernels in one namespace, as the browser has them."""
    root, rev = _materialise()
    loader = root / "spikes" / "cdr"
    if not (loader / "kernel_load.py").exists():
        raise SystemExit("no kernel_load.py at %s in %s" % (BASELINE_REV, DEMO))
    sys.path.insert(0, str(loader))
    import kernel_load                                   # noqa: E402
    return kernel_load.load_kernels()


def truth_builder():
    """`build` and `bits_lfsr` from the demo's spikes/cdr/fold_truth.py at the same
    revision -- the closed-form record the sibling unit's evidence is written
    against. Imported, not copied."""
    root, _rev = _materialise()
    sys.path.insert(0, str(root / "spikes" / "cdr"))
    import fold_truth                                    # noqa: E402
    return fold_truth.build, fold_truth.bits_lfsr


def revision():
    return _materialise()[1]

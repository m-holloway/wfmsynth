"""Assertions against a CLASS of defect: an op that fabricates or destroys information.

The defect that motivated this file: `dfe` evaluated one equalised value per symbol and rebuilt
a waveform from those values alone. It passed every test it had — the decisions were right, the
symbol-error rate was right — while DELETING 30.8 dB of its input's noise and every trace of the
15 samples in 16 that fall between decision instants. Nothing caught it because every existing
assertion looked at the decisions, and the defect was in the waveform.

Four general properties, each applied to every op that claims to be a waveform-to-waveform stage.
They are cheap, they need no reference output, and each one would have caught that defect.
"""
import numpy as np
import pytest

from wfmsynth.compose import Signal, _EXEC
from wfmsynth.grid import Grid
from wfmsynth.streams import Streams

FS, BAUD, N_UI = 256e9, 16e9, 4000
SPB = int(FS / BAUD)


def _grid():
    return Grid(fs=FS, baud=BAUD, n=N_UI * SPB, v_full=0.8)


def _input(seed=3):
    g = _grid()
    sym = np.random.default_rng(seed).choice([-1.0, 1.0], N_UI)
    return (Signal(seed=1, grid=g).symbols(sym.tolist(), kind="nrz", tr_frac=0.35, causal=True)
            .lossy(causal=True, loss_db=10.0, loss_at_ghz=8.0)
            .digitize(noise_rms=0.01)).waveform(), g


# Ops under test: waveform -> waveform stages a chain can place after a channel. Sources
# (`carrier`, `symbols`) and multi-signal/optical stages are excluded because they do not take a
# meaningful electrical waveform in.
STAGES = {
    "lossy": dict(loss_db=6.0, loss_at_ghz=8.0, causal=True),
    "reflect": dict(td_ps=40.0, gamma_s=0.06, gamma_l=0.06, n_bounce=3),
    "resonant_reflect": dict(td_ps=90.0, f0_ghz=10.7, q=9.0, gamma0=0.2),
    "ctle": dict(fz_ghz=2.0, fp1_ghz=8.0, fp2_ghz=16.0, dc_gain=0.65),
    "rx_ffe": dict(taps=[-0.05, 1.0, -0.18], spacing_ui=1.0, pre=1),
    "dfe": dict(taps=[0.12, 0.05], levels=[-1.0, 1.0]),
    "tx_ffe": dict(taps=[-0.1, 0.75, -0.15], pre=1),
    "de_emphasis": dict(db=3.0),
    "scope": dict(bw_hz=32e9, kind="bessel", order=4),
    "ac_couple": dict(fc_hz=2e9),
    "supply_coupling": dict(f_ripple_hz=2.5e6, am_depth=0.02),
    "nonlinearity": dict(compression=0.04),
    "drift": dict(kind="gain", amount=0.004, shape="linear"),
    "probe": dict(c_load_f=0.45e-12, r_source=50.0),
}


def _run(op, params, x, g, seed=0):
    return np.asarray(_EXEC[op](x, dict(params, op=op), Streams(seed), g, 0), float)


@pytest.mark.parametrize("op", sorted(STAGES))
def test_every_input_sample_influences_the_output(op):
    """THE assertion that catches a decimator masquerading as a waveform stage.

    Perturb ONE interior sample and the output must move. An op that reads one sample per symbol
    and rebuilds the rest ignores 15 of every 16 samples, so perturbing one of those 15 changes
    nothing at all — which is what `dfe` did before it returned its summing node.
    """
    x, g = _input()
    base = _run(op, STAGES[op], x, g)
    # probe several offsets within one UI so a decimator cannot pass by luck of alignment
    unmoved = []
    for off in range(SPB):
        k = (N_UI // 2) * SPB + off
        y = x.copy()
        y[k] += 0.05                                 # 5 % of full scale: far above any tolerance
        moved = float(np.max(np.abs(_run(op, STAGES[op], y, g) - base)))
        if moved < 1e-9:
            unmoved.append(off)
    assert not unmoved, (f"{op}: perturbing the sample at these offsets within a UI changed "
                         f"NOTHING in the output: {unmoved} of {SPB}. The op is ignoring part "
                         f"of its input.")


LTI = ["lossy", "reflect", "resonant_reflect", "ctle", "rx_ffe", "tx_ffe", "de_emphasis",
       "scope", "ac_couple", "probe"]


@pytest.mark.parametrize("op", LTI)
def test_an_op_that_claims_to_be_linear_obeys_superposition(op):
    """``op(a + b) == op(a) + op(b)`` and ``op(k*a) == k*op(a)``.

    Superposition is the definition of linearity, it needs no spectral division, and it is
    immune to how an op pads or truncates. Anything that decides, clamps, quantises or
    reconstructs breaks it immediately.

    TWO EARLIER VERSIONS OF THIS TEST DID NOT WORK, and both failures are worth recording.
    Watching the noise floor cannot separate honest filtering from fabrication: measured, the
    `dfe` re-render deleted 11.8 dB of the top of the band while `lossy` legitimately took
    33.8 dB, `scope` 32.1 and `probe` 23.8 — the defect sat BELOW three honest stages. And
    measuring |H| on one record to predict another fails for a different reason: the channel
    now applies a LINEAR convolution, so the output is a truncated linear convolution and the
    DFT relation ``Y = H*X`` does not hold at all. Superposition survives both problems.
    """
    g = _grid()
    rng = np.random.default_rng(5)
    a = rng.standard_normal(g.n) * 0.3
    b = rng.standard_normal(g.n) * 0.3
    ya, yb, yab = (_run(op, STAGES[op], v, g) for v in (a, b, a + b))
    m = min(len(ya), len(yb), len(yab))
    add = float(np.max(np.abs(yab[:m] - (ya[:m] + yb[:m])))) / (float(np.std(yab[:m])) + 1e-30)
    y3 = _run(op, STAGES[op], 3.0 * a, g)
    m2 = min(len(y3), len(ya))
    homo = float(np.max(np.abs(y3[:m2] - 3.0 * ya[:m2]))) / (float(np.std(y3[:m2])) + 1e-30)
    assert add < 1e-9 and homo < 1e-9, (
        f"{op}: additivity {add:.2e}, homogeneity {homo:.2e} (relative to output rms). It is "
        f"listed as linear but is not — it decides, clamps, quantises or reconstructs.")


@pytest.mark.parametrize("op,fc_key,fc", [
    ("ac_couple", "fc_hz", 2e9),
    pytest.param("scope", "bw_hz", 32e9, marks=pytest.mark.xfail(strict=True, reason=(
        "MEASURED DEFECT, not a tolerance: scope_bandwidth's IIR branch applies a 4th-order "
        "Bessel through sosfiltfilt, i.e. TWICE, so the realised -3 dB point is HALF the "
        "requested bandwidth. Asked 32 GHz: |H| = 0.7039 at 16 GHz and 0.1747 (-15.1 dB) at "
        "32 GHz. A front end set to 32 GHz therefore behaves like a 16 GHz one, and every "
        "chain that states an instrument bandwidth is stating twice what it gets. The fix is "
        "a single-pass causal design, which also gives the stage the group delay it currently "
        "lacks (measured +0.000 ps).")))])
def test_a_filter_realises_the_corner_it_was_asked_for(op, fc_key, fc):
    """Catches a silent clamp and a doubly-applied pole. `ac_couple` clamped any corner below
    1e-4 of Nyquist — a 50 kHz request became 12.8 MHz, 256x off, with nothing said. Assert the
    realised -3 dB point is within 2x of the request, which a squared response (its -3 dB at
    1.55x the stated corner) still passes but a 256x clamp does not.
    """
    g = _grid()
    n = len(np.zeros(g.n))
    def gain(f_hz):
        k = max(1, round(f_hz * n / FS))
        t = np.sin(2 * np.pi * k * np.arange(n) / n)
        y = _run(op, {fc_key: fc}, t, g)
        return abs(np.fft.rfft(y[:n])[k] / np.fft.rfft(t)[k]), k * FS / n
    lo, hi = fc / 60.0, min(0.45 * FS, fc * 60.0)
    for _ in range(34):                                  # bisect the |H| = 1/sqrt(2) point
        mid = np.sqrt(lo * hi)
        gn, _fa = gain(mid)
        rising = op == "ac_couple"
        if (gn < 0.70710678) == rising:
            lo = mid
        else:
            hi = mid
    realised = np.sqrt(lo * hi)
    assert 0.5 < realised / fc < 2.0, (f"{op}: asked for {fc:.4g} Hz, realised "
                                       f"{realised:.4g} Hz ({realised / fc:.1f}x)")

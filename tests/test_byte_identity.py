"""The default path is pinned, byte for byte.

The kernel is consumed downstream by SHA and the outputs are diffed, so "I added a feature and
nothing else moved" is a claim that has to be checkable rather than asserted. Every hash below is
of the raw float64 bytes of a realised waveform. They were captured BEFORE the cascaded-channel
work landed and must not change when a new op is added beside the old ones.

If one of these fails, a default changed. That is either a bug or a deliberate break that needs
a version bump and a note downstream -- it is never a test to relax.
"""
import hashlib

import numpy as np
import pytest

import wfmsynth.physics as P
import wfmsynth.sparam as SP
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid


def h(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, float)).tobytes()).hexdigest()[:32]


G = Grid(fs=256e9, baud=16e9, n=1 << 16)
G8 = Grid(fs=256e9, baud=16e9, n=8192)
X = P.nrz(n_ui=256, n=8192, seed=3, tr_frac=0.15)


def _gen4():
    """The shipped Gen4-shaped recipe, in the op order that motivated the cascade work:
    carrier -> ssc -> lossy -> reflect -> timing -> supply -> crosstalk -> scope -> digitize."""
    return (Signal(seed=7, grid=G)
            .carrier("nrz", n_ui=16384, tr_frac=0.15, pattern="prbs13")
            .ssc(spread=0.0025, f_ssc=32e3)
            .lossy(loss_db=12.5, loss_at_ghz=8.0, causal=True)
            .reflect(td_ps=281.25, gamma_s=0.055, gamma_l=0.055, n_bounce=6, node="source")
            .timing(rj_ps=0.4)
            .supply_coupling(f_ripple_hz=1e6, am_depth=0.01)
            .crosstalk(coupling=0.05, kind="fext")
            .scope(bw_hz=110e9)
            .digitize(enob=11, snr_db=45)).waveform()


def _sparam():
    f = np.linspace(0, 40e9, 401)
    s21 = 10 ** (-(0.35 * np.sqrt(f / 1e9) + 0.2 * f / 1e9) / 20) * np.exp(-1j * 2 * np.pi * f * 500e-12)
    return SP.sparam_channel(X, f, s21, grid=G8)


PINNED = {
    "gen4_recipe":    (_gen4,                                                    "184472a8eb37ea8644c2a67948f10acf"),
    "nrz":            (lambda: X,                                                "3a16d7331a8403333831abc60b7af629"),
    "pam4":           (lambda: P.pam4(n_ui=256, n=8192, seed=5),                 "b349e50a01a83d549c61e1143692eea0"),
    "lossy_legacy":   (lambda: P.lossy_channel(X),                               "944b7440b1d1f292c455b0a709d36080"),
    "lossy_causal":   (lambda: P.lossy_channel(X, causal=True, loss_db=12.5, loss_at_ghz=8.0, grid=G),
                                                                                 "8b5ac2eee58de632aad80fafc9fc6d1d"),
    "lossy_trend":    (lambda: P.lossy_channel(X, trend=(1.335, -2.965, -2.16), causal=True, grid=G8),
                                                                                 "1e13bdeb6476254cfd4547249f5d8d53"),
    "reflect_load":   (lambda: P.multi_reflection(X),                            "82b5d03c59f413e409fe07ec9a4349b1"),
    "reflect_source": (lambda: P.multi_reflection(X, td_samples=144, gamma_s=0.055, gamma_l=0.055, node="source"),
                                                                                 "5a645d25d39ea89a06df6364a17297ef"),
    "resonant":       (lambda: P.resonant_reflection(X),                         "17d7e5c6aef78ee13ec364019dc059a4"),
    "multi_bounce":   (lambda: P.multi_reflection(X, td_frac=0.05, gamma_s=0.3, gamma_l=0.4, n_bounce=6),
                                                                                 "b4f33f55e1223e37ac3f6dc5e5e0ab43"),
    "crosstalk":      (lambda: P.crosstalk(X, P.nrz(n_ui=256, n=8192, seed=9)),   "18cc0fcb180ad6ef2744a95bc8980b75"),
    "ac_couple":      (lambda: P.ac_couple(X),                                   "214573f1e149175fac4660b6e087dd07"),
    "minphase":       (lambda: np.abs(P._min_phase_H(np.linspace(1.0, 0.01, 4097), 8192)),
                                                                                 "5ddb840540db647b27bc72281daffd64"),
    "sparam_channel": (_sparam,                                                  "43cac30c930815cfcbb8257072081a0c"),
    "jitter":         (lambda: P.inject_jitter(X, sigma_rj=0.01, a_pj=0.02, f_pj=5.0, rng=np.random.default_rng(1)),
                                                                                 "6650fa61a8012402c8b8846d30dcfe72"),
}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_the_default_path_is_byte_identical(name):
    make, want = PINNED[name]
    assert h(make()) == want, f"{name}: the default path moved"


def test_extracting_the_loss_kernel_did_not_move_the_lumped_channel():
    """`physics.insertion_loss_db` was lifted out of `lossy_channel` so a cascade SECTION and a
    lumped channel share one loss law. It is a pure extraction: the lumped result must be the
    same floats it was before, on every argument form."""
    f_ghz = np.fft.rfftfreq(8192) * 2.0 * 128.0
    for kw in ({}, {"length_in": 3.0, "tand": 0.01},
               {"loss_db": 12.5, "loss_at_ghz": 8.0},
               {"trend": (1.335, -2.965, -2.16)},
               {"trend": (13.920, -6.120, -10.78), "trend_floor_db": 40.0},
               {"skin_k": 0.5, "eps_r": 3.6}):
        il = P.insertion_loss_db(f_ghz, **kw)
        y = P.lossy_channel(X, f_nyq_ghz=128.0, **kw)
        assert np.array_equal(y, np.fft.irfft(np.fft.rfft(X) * 10.0 ** (-il / 20.0), n=8192))
        assert (il >= 0).all()                     # passive on every form

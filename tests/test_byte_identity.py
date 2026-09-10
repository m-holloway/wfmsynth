"""The default path is pinned, byte for byte.

The kernel is consumed downstream by SHA and the outputs are diffed, so "I added a feature and
nothing else moved" is a claim that has to be checkable rather than asserted. Every hash below is
of the raw float64 bytes of a realised waveform.

If one of these fails, a default changed. That is either a bug or a deliberate break that needs
a version bump and a note downstream -- it is never a test to relax.

THE SECOND SUCH BREAK. The first was the cascaded-channel work, before which these were
captured. The second is U-16: every frequency-domain stage now applies its response as a LINEAR
convolution rather than a circular one, so the response to a record's tail no longer wraps onto
its head, and the transform length is chosen (the next 5-smooth length at or above the
linear-convolution length) rather than pinned to the record. Six entries moved on that change --
`gen4_recipe`, `lossy_legacy`, `lossy_causal`, `lossy_trend`, `resonant` and `sparam_channel` --
and every one of them contains a frequency-domain stage. The nine that did not move
(`nrz`, `pam4`, the three `reflect`/`multi_bounce` entries, `crosstalk`, `ac_couple`,
`minphase`, `jitter`) are the ops that were already time-domain or already linear, which is the
check that the change reached exactly what it claimed to reach and nothing else.

How far each moved, against the same record's peak-to-peak: 7.1 % for a causal analytic
channel, 11.4 % zero-phase, 2.9 % for a cascade read at the driver plane and 29.3 % at the load,
and it is the record's HEAD that moves -- see `tests/test_linear_convolution.py`, which pins the
new path against `np.convolve` and against a two-tap echo whose closed form is written by hand.

THE THIRD SUCH BREAK, and it moved exactly one entry: `gen4_recipe`, the only pinned record
with an analog INSTRUMENT stage in it. `scope`'s analog kinds ran a Bessel forwards AND
backwards (`sosfiltfilt`), which squared the magnitude and zeroed the group delay, so a front
end asked for 32 GHz realised 16.0 GHz. It is now a single forward pass designed with
`norm='mag'`, so the realised -3 dB point is the stated one (measured 32.00 GHz, |H| = 0.7071)
and the stage delays (measured 10.00 ps at DC for bessel-4 at 32 GHz, against the analog closed
form's 10.51 ps). MEASURED move on this record, whose front end is at 110 GHz on a 128 GHz
Nyquist grid and therefore barely bites: 1.59 % of peak-to-peak at worst, 0.44 % rms.
`gen4_zero_phase` pins the OLD hash through the `causal=False` opt-out, so the claim that the
old behaviour is still exactly reachable is a hash, not a sentence.
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


def _gen4(**scope_kw):
    """The shipped Gen4-shaped recipe, in the op order that motivated the cascade work:
    carrier -> ssc -> lossy -> reflect -> timing -> supply -> crosstalk -> scope -> digitize.

    `scope_kw` reaches the front end alone, which is how the zero-phase opt-out is pinned
    against the hash this recipe had before the front end became causal."""
    return (Signal(seed=7, grid=G)
            .carrier("nrz", n_ui=16384, tr_frac=0.15, pattern="prbs13")
            .ssc(spread=0.0025, f_ssc=32e3)
            .lossy(loss_db=12.5, loss_at_ghz=8.0, causal=True)
            .reflect(td_ps=281.25, gamma_s=0.055, gamma_l=0.055, n_bounce=6, node="source")
            .timing(rj_ps=0.4)
            .supply_coupling(f_ripple_hz=1e6, am_depth=0.01)
            .crosstalk(coupling=0.05, kind="fext")
            .scope(bw_hz=110e9, **scope_kw)
            .digitize(enob=11, snr_db=45)).waveform()


def _sparam():
    f = np.linspace(0, 40e9, 401)
    s21 = 10 ** (-(0.35 * np.sqrt(f / 1e9) + 0.2 * f / 1e9) / 20) * np.exp(-1j * 2 * np.pi * f * 500e-12)
    return SP.sparam_channel(X, f, s21, grid=G8)


PINNED = {
    "gen4_recipe":    (_gen4,                                                    "694f0a4f7a85f0df0e47045afe765837"),
    # The pre-fix hash of the SAME recipe, reached by the documented opt-out. This is the
    # entry that makes "the old behaviour is still available, exactly" checkable rather than
    # asserted -- and it is what caught `_op_scope` dropping `causal` from the kwargs it
    # forwarded, which had made `causal=False` render the causal path and hash identically.
    "gen4_zero_phase": (lambda: _gen4(causal=False),                             "3447e5da7cc90b76bd22cbbad22983ec"),
    "nrz":            (lambda: X,                                                "3a16d7331a8403333831abc60b7af629"),
    "pam4":           (lambda: P.pam4(n_ui=256, n=8192, seed=5),                 "b349e50a01a83d549c61e1143692eea0"),
    "lossy_legacy":   (lambda: P.lossy_channel(X),                               "37ce9b15660008866402d55597b5aa25"),
    "lossy_causal":   (lambda: P.lossy_channel(X, causal=True, loss_db=12.5, loss_at_ghz=8.0, grid=G),
                                                                                 "99f1e28639e65bd65fcb4b5b9df80eac"),
    "lossy_trend":    (lambda: P.lossy_channel(X, trend=(1.335, -2.965, -2.16), causal=True, grid=G8),
                                                                                 "ea6f737e271bd577995d9b1c747a90de"),
    "reflect_load":   (lambda: P.multi_reflection(X),                            "82b5d03c59f413e409fe07ec9a4349b1"),
    "reflect_source": (lambda: P.multi_reflection(X, td_samples=144, gamma_s=0.055, gamma_l=0.055, node="source"),
                                                                                 "5a645d25d39ea89a06df6364a17297ef"),
    "resonant":       (lambda: P.resonant_reflection(X),                         "1dab2a355a841caad16168fb3ad010f3"),
    "multi_bounce":   (lambda: P.multi_reflection(X, td_frac=0.05, gamma_s=0.3, gamma_l=0.4, n_bounce=6),
                                                                                 "b4f33f55e1223e37ac3f6dc5e5e0ab43"),
    "crosstalk":      (lambda: P.crosstalk(X, P.nrz(n_ui=256, n=8192, seed=9)),   "18cc0fcb180ad6ef2744a95bc8980b75"),
    "ac_couple":      (lambda: P.ac_couple(X),                                   "214573f1e149175fac4660b6e087dd07"),
    "minphase":       (lambda: np.abs(P._min_phase_H(np.linspace(1.0, 0.01, 4097), 8192)),
                                                                                 "5ddb840540db647b27bc72281daffd64"),
    "sparam_channel": (_sparam,                                                  "95bb048015c39b4dcbd9ed14342ce113"),
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
    same floats it was before, on every argument form.

    Stated on the pinned-length path (`linear=False`), which is the arithmetic this test was
    written against. The claim is about the LOSS LAW -- that one function computes it for both
    callers -- and holding the transform length fixed is what keeps it a statement about the
    loss and not about the padding. What the padded default does with that same |H| is
    `tests/test_linear_convolution.py`'s business."""
    f_ghz = np.fft.rfftfreq(8192) * 2.0 * 128.0
    for kw in ({}, {"length_in": 3.0, "tand": 0.01},
               {"loss_db": 12.5, "loss_at_ghz": 8.0},
               {"trend": (1.335, -2.965, -2.16)},
               {"trend": (13.920, -6.120, -10.78), "trend_floor_db": 40.0},
               {"skin_k": 0.5, "eps_r": 3.6}):
        il = P.insertion_loss_db(f_ghz, **kw)
        y = P.lossy_channel(X, f_nyq_ghz=128.0, linear=False, **kw)
        assert np.array_equal(y, np.fft.irfft(np.fft.rfft(X) * 10.0 ** (-il / 20.0), n=8192))
        assert (il >= 0).all()                     # passive on every form

"""The default path is pinned to a tolerance, not to bytes.

The kernel is consumed downstream and the outputs are diffed, so "I added a feature and
nothing else moved" has to be checkable rather than asserted. Every entry below pins a
realised waveform: its length exactly, and its peak-to-peak, rms, mean and forty-eight evenly
spaced samples to within TOL_FRAC of the record's own peak-to-peak.

WHY A TOLERANCE AND NOT A HASH. `physics._shape_edges` calls `scipy.signal.bessel`, which is
filter DESIGN -- polynomial root-finding, and therefore LAPACK. A build against Accelerate and
a build against OpenBLAS return coefficients that agree to about 1e-15 relative, not to the
bit, and every entry here is built on a shaped carrier. A sha256 of the float64 bytes cannot
tell "a default changed" from "a different LAPACK", so it reported the second as the first on
any host but the one that captured it: fifteen of seventeen entries failed on a fresh clone
while the library was provably unchanged -- no commit had touched `physics.py` since the
hashes were taken.

The two that did pass are the proof of that diagnosis rather than an exception to it.
`minphase` is the only entry with no filter design in it, pure FFT on a deterministic ramp,
and it matched to the bit -- so numpy's FFT is identical between the capture host and any
other, and scipy's filter design is not.

THE TOLERANCE IS MEASURED, NOT CHOSEN. All three numbers below were swept on this pipeline,
and it is the gap between them that the tolerance lives in:

    3.8e-13 relative   what a LAPACK difference costs, modelled by perturbing the Bessel
                       section coefficients by 1e-14 -- an order of magnitude LOOSER than
                       two builds actually differ
    1e-5    relative   where this file starts catching a change
    1.6e-2  relative   the smallest behaviour change it has ever had to record (the largest
                       was 29.3 %)

Seven orders of margin under the detection threshold, three over it. Two tests keep both ends
honest: `test_the_tolerance_is_far_above_the_numerical_floor` fails if the pipeline is ever
made chaotically sensitive, and `test_a_real_change_is_caught` fails if the fingerprint stops
discriminating.

If one of these fails, a default changed. That is either a bug or a deliberate break that
needs a version bump and a note downstream. The tolerance exists to absorb the host's
arithmetic, and nothing else -- widening it to make a failure go away discards the only thing
this file is for.

WHAT MOVED, AND WHEN. The linear-convolution change: every frequency-domain stage applies its
response as a linear rather than a circular convolution, so the response to a record's tail no
longer wraps onto its head, and the transform length is the next 5-smooth length at or above
the linear-convolution length. It moved the six entries that contain a frequency-domain stage
and none of the nine that were already time-domain or already linear, which is how the change
was shown to reach exactly what it claimed. `tests/test_linear_convolution.py` pins the new
path against `np.convolve` and against a two-tap echo whose closed form is written by hand.

The causal front-end change: `scope`'s analog kinds ran a Bessel forwards AND backwards
(`sosfiltfilt`), squaring the magnitude and zeroing the group delay, so a front end asked for
32 GHz realised 16.0 GHz. It is now a single forward pass designed with `norm='mag'`: the
realised -3 dB point is the stated one (measured 32.00 GHz, |H| = 0.7071) and the stage delays
(measured 10.00 ps at DC for bessel-4 at 32 GHz, against the analog closed form's 10.51 ps).
It moved exactly one entry, `gen4_recipe`, the only pinned record with an analog instrument
stage in it. `gen4_zero_phase` pins the OLD behaviour through the `causal=False` opt-out, so
"the old path is still exactly reachable" is a test rather than a sentence.
"""
import numpy as np
import pytest
from scipy import signal

import wfmsynth.physics as P
import wfmsynth.sparam as SP
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid

# of the record's own peak-to-peak. See the module docstring: measured numerical floor is
# 3.8e-11 %, smallest real behaviour change on record is 1.59 %.
TOL_FRAC = 1e-6
N_PROBE = 48


G = Grid(fs=256e9, baud=16e9, n=1 << 16)
G8 = Grid(fs=256e9, baud=16e9, n=8192)
X = P.nrz(n_ui=256, n=8192, seed=3, tr_frac=0.15)


def _gen4(**scope_kw):
    """The shipped Gen4-shaped recipe, in the op order that motivated the cascade work:
    carrier -> ssc -> lossy -> reflect -> timing -> supply -> crosstalk -> scope -> digitize.

    `scope_kw` reaches the front end alone, which is how the zero-phase opt-out is pinned
    against the behaviour this recipe had before the front end became causal."""
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


PATHS = {
    "gen4_recipe":     _gen4,
    # The pre-change behaviour of the SAME recipe, reached by the documented opt-out. This is
    # the entry that makes "the old path is still available, exactly" checkable -- and it is
    # what caught `_op_scope` dropping `causal` from the kwargs it forwarded, which had made
    # `causal=False` render the causal path.
    "gen4_zero_phase": lambda: _gen4(causal=False),
    "nrz":             lambda: X,
    "pam4":            lambda: P.pam4(n_ui=256, n=8192, seed=5),
    "lossy_legacy":    lambda: P.lossy_channel(X),
    "lossy_causal":    lambda: P.lossy_channel(X, causal=True, loss_db=12.5, loss_at_ghz=8.0,
                                               grid=G),
    "lossy_trend":     lambda: P.lossy_channel(X, trend=(1.335, -2.965, -2.16), causal=True,
                                               grid=G8),
    "reflect_load":    lambda: P.multi_reflection(X),
    "reflect_source":  lambda: P.multi_reflection(X, td_samples=144, gamma_s=0.055,
                                                  gamma_l=0.055, node="source"),
    "resonant":        lambda: P.resonant_reflection(X),
    "multi_bounce":    lambda: P.multi_reflection(X, td_frac=0.05, gamma_s=0.3, gamma_l=0.4,
                                                  n_bounce=6),
    "crosstalk":       lambda: P.crosstalk(X, P.nrz(n_ui=256, n=8192, seed=9)),
    "ac_couple":       lambda: P.ac_couple(X),
    "minphase":        lambda: np.abs(P._min_phase_H(np.linspace(1.0, 0.01, 4097), 8192)),
    "sparam_channel":  _sparam,
    "jitter":          lambda: P.inject_jitter(X, sigma_rj=0.01, a_pj=0.02, f_pj=5.0,
                                               rng=np.random.default_rng(1)),
}

REFERENCE = {
    "ac_couple": dict(
        n=8192, pp=2.926898307519e+00, rms=7.920702434993e-01,
        mean=+1.224983905382e-03,
        probe=[+5.991008056693e-01, +1.202456998837e+00, -6.045619721007e-01, -8.406493624635e-01, +9.979563034951e-01, +7.898420586296e-01, -1.058793804981e+00, +5.291158625553e-01, -7.709662492581e-01, +9.525314835906e-01, -7.055516755063e-01, +8.627356244767e-01, +7.912999929542e-01, +5.178027434678e-01, +8.206536593208e-01, +9.467404005764e-01, -8.546078975834e-01, +6.283645637430e-01, +8.927685307173e-01, +9.247479742666e-01, +4.055271103166e-01, -8.825475502731e-01, -3.020112992890e-01, -5.188074772609e-01, -5.239812359848e-01, -7.237713575402e-01, +8.833014058002e-01, -6.996179571351e-01, -1.012180360076e+00, -4.271727496205e-01, +7.290099439978e-01, -9.013094732862e-01, +1.038742292004e+00, -6.863126652117e-01, -8.568081106142e-01, +7.597055541304e-01, -6.398727931682e-01, -1.265765716658e+00, -2.505945714542e-01, -9.187717791415e-01, +4.068663296974e-01, -1.024089762099e+00, +9.892863911571e-01, +4.780055317367e-01, +6.230115969148e-01, -3.589054822164e-01, -3.082517720123e-01, -5.445780897153e-02]),
    "crosstalk": dict(
        n=8192, pp=2.483717322560e+00, rms=9.351643894227e-01,
        mean=+7.515799613522e-03,
        probe=[+1.000023099533e+00, +1.002223609789e+00, -4.606348760937e-01, -9.996343964753e-01, +9.173311261371e-01, +1.004608645834e+00, -1.006468843020e+00, +1.000041196355e+00, -9.989233207386e-01, +1.241858659514e+00, -1.001448552034e+00, +1.000033373757e+00, +1.007763742807e+00, +1.000160404188e+00, +1.043138348981e+00, +1.013390981692e+00, -7.549846101655e-01, +1.001694837238e+00, +1.241826266011e+00, +9.996465826336e-01, +1.000000000776e+00, -1.007030206016e+00, -8.497209931859e-01, -1.024466580981e+00, -9.997560698430e-01, -8.664975750313e-01, +1.006369672869e+00, -9.999956918556e-01, -9.991557793109e-01, -5.249364506945e-01, +1.003507680166e+00, -8.498861328288e-01, +1.012374505185e+00, -9.880940025417e-01, -1.098799772957e+00, +1.003506976215e+00, -2.970305062424e-01, -9.989006072364e-01, -2.969590059524e-01, -9.993524369761e-01, +7.548328176037e-01, -1.011469441601e+00, +1.012410552343e+00, +9.483797573703e-01, +9.989061600371e-01, -7.975964039983e-01, -1.001483321655e+00, -9.999810261703e-01]),
    "gen4_recipe": dict(
        n=65536, pp=1.547838252887e+00, rms=2.577550971576e-01,
        mean=+2.208164571950e-04,
        probe=[-2.244328061218e-03, +1.593472923465e-01, -1.226899340132e-01, -2.917626479583e-01, +2.206922593531e-01, +3.875206452370e-01, +1.331634649656e-01, -9.201745050994e-02, +9.725421598611e-03, -3.456265214276e-01, +1.428888865642e-01, +5.939988268690e-01, -1.593472923465e-01, +1.107201843534e-01, -6.209307636036e-02, +1.256823714282e-01, -2.618382738088e-02, -9.052123180246e-02, +9.800232533985e-02, +2.663269299312e-01, -1.630878391152e-01, +1.167050591833e-01, -1.817905729587e-01, +2.408912119041e-01, -7.406282602019e-02, -4.989889389441e-01, -3.216870221079e-02, +1.122164030609e-01, -1.773019168362e-01, -1.144607311221e-01, -1.870273384348e-01, +2.917626479583e-01, +2.708155860536e-01, +2.251809154755e-01, -4.309109877539e-01, +5.236765476175e-02, +1.144607311221e-01, -1.196974965983e-02, +1.339115743193e-01, -2.580977270401e-01, +2.895183198971e-01, +5.236765476175e-03, +9.351366921742e-02, -3.366492091827e-02, +4.361477532300e-01, +4.787899863932e-01, -2.872739918359e-01, -2.753042421761e-01]),
    "gen4_zero_phase": dict(
        n=65536, pp=1.548936742256e+00, rms=2.577063032666e-01,
        mean=+2.202726917477e-04,
        probe=[-2.244835858342e-03, +1.571385100839e-01, -1.234659722088e-01, -2.978148905400e-01, +2.117628493036e-01, +3.816220959181e-01, +1.406763804561e-01, -9.128999157257e-02, +7.482786194473e-04, -3.367253787513e-01, +1.421729376950e-01, +5.918883879828e-01, -1.571385100839e-01, +9.727622052815e-02, -5.462433921965e-02, +1.234659722088e-01, -3.142770201679e-02, -9.952105638649e-02, +1.077521212004e-01, +2.731216960983e-01, -1.623764604201e-01, +1.129900715365e-01, -1.810834259062e-01, +2.394491582231e-01, -8.305892675865e-02, -5.050880681269e-01, -3.666565235292e-02, +1.114935142976e-01, -1.743489183312e-01, -1.159831860143e-01, -1.893144907202e-01, +2.850941540094e-01, +2.753665319566e-01, +2.282249789314e-01, -4.369947137572e-01, +4.863811026407e-02, +1.159831860143e-01, -9.727622052815e-03, +1.414246590755e-01, -2.648906312843e-01, +2.865907112483e-01, +2.993114477789e-03, +9.577966328925e-02, -3.142770201679e-02, +4.257705344655e-01, +4.848845454018e-01, -2.895838257261e-01, -2.544147306121e-01]),
    "jitter": dict(
        n=8192, pp=2.015470171247e+00, rms=9.315026921586e-01,
        mean=+7.828328686262e-03,
        probe=[+1.000002385499e+00, +1.000113044001e+00, -6.273224590924e-01, -9.996282153322e-01, +9.179671549545e-01, +9.619537592957e-01, -1.006812197195e+00, +9.999999992883e-01, -9.987312658711e-01, +1.000002011406e+00, -1.001439208951e+00, +1.000032892717e+00, +1.007003903699e+00, +1.000150978273e+00, +1.000122858944e+00, +1.001845511641e+00, -7.511165210701e-01, +9.987319433303e-01, +9.999944584879e-01, +9.989005290272e-01, +1.000000003185e+00, -1.007690415731e+00, -8.466615363141e-01, -1.000033479953e+00, -1.000001095478e+00, -9.999932950334e-01, +1.005966510180e+00, -1.000019093134e+00, -9.999024277330e-01, -2.965193630726e-01, +1.005984496513e+00, -8.484649951689e-01, +9.882785924412e-01, -9.880851979689e-01, -1.000137593148e+00, +1.005975380931e+00, -2.917882542599e-01, -9.989004717588e-01, -2.949998122403e-01, -9.989937628266e-01, +7.525292433611e-01, -9.998741954351e-01, +9.880634454827e-01, +8.477386605313e-01, +1.000000882138e+00, -1.000000003185e+00, -9.999999999978e-01, -1.000002059281e+00]),
    "lossy_causal": dict(
        n=8192, pp=1.757843624844e+00, rms=5.673817656067e-01,
        mean=+1.008920028180e-02,
        probe=[+7.631331923040e-04, -8.998312890760e-02, -8.116570153519e-01, -5.539092464572e-01, +3.024386560502e-01, -4.009054532893e-01, -3.086788194550e-01, +7.143323710652e-01, -1.896658981807e-01, +6.779119728487e-01, -1.583293384894e-01, +3.983947021884e-01, -2.763679902817e-03, +4.727328788834e-01, +6.798829784940e-01, +6.614855773214e-01, +3.504314784459e-01, +3.215563266737e-01, +5.193600756858e-01, +1.332963018517e-01, +7.289187326610e-01, +3.088864948105e-01, -8.324326039871e-01, -8.606156270999e-01, -8.000546318074e-01, -6.224796884902e-01, +7.782939022548e-01, -7.503007129400e-01, -5.685480380342e-02, -4.027090859131e-01, -1.169826656263e-01, -2.970095852840e-01, -4.215494042297e-01, -7.609505694588e-01, -6.351407716244e-01, +6.854657379660e-01, +6.737122528720e-01, +1.362950632063e-02, -7.324127408625e-01, -6.261176840095e-01, +7.881245695432e-01, -5.179374321249e-01, +3.359375794235e-01, -3.566618323680e-01, +8.736123137695e-01, -7.828114386661e-01, -8.069726310058e-01, -5.174667274225e-01]),
    "lossy_legacy": dict(
        n=8192, pp=1.927916437912e+00, rms=8.501203200130e-01,
        mean=+8.016177416688e-03,
        probe=[+7.116785940546e-01, +8.736871143695e-01, -5.291284268770e-01, -9.247532849205e-01, +7.751422914441e-01, +8.388211715306e-01, -8.789383249091e-01, +9.569113761713e-01, -9.264943681714e-01, +9.314748836230e-01, -9.193980459471e-01, +9.282890775286e-01, +8.952487593182e-01, +9.470811320107e-01, +9.276634920501e-01, +8.875008993267e-01, -6.256579128031e-01, +9.329089936181e-01, +9.265197500036e-01, +8.906902949227e-01, +9.605501363837e-01, -9.053642450301e-01, -7.501627580742e-01, -9.433838648443e-01, -9.548004658105e-01, -9.440879914478e-01, +9.078608379243e-01, -9.415621749707e-01, -9.060630375429e-01, -2.381080899045e-01, +9.147502422215e-01, -7.131667598025e-01, +8.605079110592e-01, -8.782074647480e-01, -9.255808068844e-01, +9.151276669477e-01, -2.228832923459e-01, -8.698161227248e-01, -2.470444554343e-01, -9.174259192090e-01, +6.576462927031e-01, -9.148449146477e-01, +8.518577090251e-01, +7.418607220116e-01, +9.492348904924e-01, -9.625502179482e-01, -9.623237366498e-01, -7.214960862332e-01]),
    "lossy_trend": dict(
        n=8192, pp=1.563831005778e+00, rms=4.957672950209e-01,
        mean=+7.820890989054e-03,
        probe=[+2.813435544899e-04, -5.269523365682e-01, -7.272015486783e-01, -3.465232466984e-01, -1.048404283000e-01, -5.825029394232e-01, +8.980093322967e-02, +6.245496662971e-01, +1.904674210555e-01, +5.872540508049e-01, +1.845017362463e-01, +5.729968436340e-02, -2.412298639639e-01, +9.784738212383e-02, +4.364927446831e-01, +5.202199160115e-01, +2.308001687537e-01, -8.817459485332e-02, +1.786577204078e-01, -2.481060805605e-01, +6.196443951366e-01, +6.384309395363e-01, -7.732000733846e-01, -7.736312117974e-01, -7.060124794672e-01, -3.484224768727e-01, +7.344959484099e-01, -6.487227464845e-01, +4.131996139456e-01, -6.790926720106e-02, -5.060395178368e-01, +9.001490689554e-02, -6.787101696174e-01, -6.922743947327e-01, -3.863572954694e-01, +5.577335915552e-01, +6.377165200593e-01, +4.648952180942e-01, -7.027839174037e-01, -4.598080366325e-01, +7.159298652377e-01, -3.278515353802e-01, -1.203205940051e-01, -3.188378081023e-01, +7.786024646302e-01, -7.447418804323e-01, -7.175086979054e-01, -1.582580375768e-01]),
    "minphase": dict(
        n=8192, pp=9.900000000000e-01, rms=5.802585714468e-01,
        mean=+5.050000000010e-01,
        probe=[+1.000000000001e+00, +9.579443359385e-01, +9.158886718760e-01, +8.738330078135e-01, +8.315356445323e-01, +7.894799804698e-01, +7.474243164072e-01, +7.053686523448e-01, +6.630712890635e-01, +6.210156250010e-01, +5.789599609385e-01, +5.366625976573e-01, +4.946069335948e-01, +4.525512695323e-01, +4.104956054698e-01, +3.681982421885e-01, +3.261425781260e-01, +2.840869140635e-01, +2.420312500010e-01, +1.997338867197e-01, +1.576782226572e-01, +1.156225585947e-01, +7.332519531350e-02, +3.126953125100e-02, +3.078613281350e-02, +7.284179687600e-02, +1.151391601572e-01, +1.571948242197e-01, +1.992504882822e-01, +2.415478515635e-01, +2.836035156260e-01, +3.256591796885e-01, +3.677148437510e-01, +4.100122070323e-01, +4.520678710947e-01, +4.941235351573e-01, +5.361791992197e-01, +5.784765625010e-01, +6.205322265635e-01, +6.625878906260e-01, +7.048852539073e-01, +7.469409179698e-01, +7.889965820322e-01, +8.310522460947e-01, +8.733496093760e-01, +9.154052734385e-01, +9.574609375010e-01, +9.997583007823e-01]),
    "multi_bounce": dict(
        n=8192, pp=2.123830208672e+00, rms=9.327393722746e-01,
        mean=+8.710177006391e-03,
        probe=[+1.000002071821e+00, +1.000128458900e+00, -6.300794461779e-01, -9.996343700959e-01, +9.173426659841e-01, +9.136120754409e-01, -9.765566624457e-01, +1.047999994427e+00, -1.003580850134e+00, +1.048071601779e+00, -9.592090075695e-01, +9.459012280267e-01, +1.060818782053e+00, +9.463911351708e-01, +9.485110681559e-01, +1.032331682194e+00, -8.031634187799e-01, +9.971288631749e-01, +1.045061041917e+00, +9.444285486949e-01, +1.041829683026e+00, -9.531936915649e-01, -7.963844231035e-01, -1.051679762966e+00, -9.695415250073e-01, -9.570189031341e-01, +9.635289142939e-01, -1.053170777803e+00, -9.454690805000e-01, -2.530715831586e-01, +9.716030884894e-01, -9.031851703190e-01, +1.031177718860e+00, -9.529507215912e-01, -1.046024526375e+00, +1.060193876396e+00, -3.428537060910e-01, -9.458683972387e-01, -3.402966151194e-01, -9.572963365868e-01, +8.092868712142e-01, -9.583615602565e-01, +1.030257380907e+00, +8.989784263620e-01, +9.711177461960e-01, -9.545471496095e-01, -9.467540404767e-01, -9.584327243071e-01]),
    "nrz": dict(
        n=8192, pp=2.015471592765e+00, rms=9.315387979231e-01,
        mean=+7.812500001242e-03,
        probe=[+1.000002071821e+00, +1.000128458900e+00, -6.300794461779e-01, -9.996343700959e-01, +9.173426659841e-01, +9.615944713077e-01, -1.006804655471e+00, +9.999999992908e-01, -9.987326656964e-01, +1.000002071281e+00, -1.001448552034e+00, +1.000033371328e+00, +1.007058152460e+00, +1.000151141532e+00, +1.000122721672e+00, +1.001798492422e+00, -7.549421438744e-01, +9.987326657318e-01, +9.999944002599e-01, +9.989005290272e-01, +1.000000003204e+00, -1.007735796382e+00, -8.497179503871e-01, -1.000036817576e+00, -1.000001096896e+00, -9.999933798170e-01, +1.006010809070e+00, -1.000019372157e+00, -9.998929118897e-01, -2.970160124388e-01, +1.006010809070e+00, -8.498597645099e-01, +9.880947730085e-01, -9.880940152895e-01, -1.000137965449e+00, +1.006010105117e+00, -2.970044422871e-01, -9.989004717588e-01, -2.969850678771e-01, -9.989935471536e-01, +7.548293813495e-01, -9.998769722217e-01, +9.881308329136e-01, +8.497179500867e-01, +1.000000889611e+00, -1.000000003204e+00, -9.999999999978e-01, -1.000002059281e+00]),
    "pam4": dict(
        n=8192, pp=2.015471553820e+00, rms=6.943702827720e-01,
        mean=+1.562500195356e-02,
        probe=[+1.000002071763e+00, +9.995600040539e-01, +6.301601881239e-01, +3.357234056632e-01, +9.724459391898e-01, -3.204906229444e-01, +3.334563099120e-01, -3.333074782602e-01, -3.324882557936e-01, +4.006661858698e-01, -1.000568473717e+00, -4.760816049126e-01, +1.002352678633e+00, -1.000100229366e+00, +3.205724670797e-01, +9.999176208547e-01, +9.183892199172e-01, -3.317057674626e-01, +3.366863779419e-02, +9.999643000553e-01, -5.079689353807e-01, -3.361369432729e-01, +1.000090693776e+00, +9.881193185943e-01, -1.001199736201e+00, -9.183450020108e-01, +3.346650845252e-01, -1.353497457469e-01, -3.329668430105e-01, +5.676651871270e-01, -3.373401531793e-01, +3.332847881311e-01, -3.253960091067e-01, +9.960563880592e-01, +2.832878680730e-01, +1.000996529296e+00, +7.656707482897e-01, +3.335927659124e-01, +5.313466099912e-01, +3.373401531283e-01, +9.183472165675e-01, +3.338497352665e-01, +9.960313368402e-01, -2.330967489805e-01, -3.359113390755e-01, -8.253793895282e-01, -9.999999887229e-01, -3.333347060130e-01]),
    "reflect_load": dict(
        n=8192, pp=2.123472028700e+00, rms=9.318640772150e-01,
        mean=+7.515096261280e-03,
        probe=[+1.000002071821e+00, +1.000128458900e+00, -6.300794461779e-01, -9.996343700959e-01, +9.173426659841e-01, +9.615944713077e-01, -1.006804655471e+00, +9.999999992908e-01, -9.987326656964e-01, +1.000002071281e+00, -1.001448552034e+00, +1.000033371328e+00, +9.590587089618e-01, +9.521511016342e-01, +1.044147836539e+00, +9.537967250986e-01, -8.029362385411e-01, +1.034969888270e+00, +1.047933563357e+00, +1.013157297665e+00, +1.047994866286e+00, -1.030586111711e+00, -8.976696366619e-01, -9.650103591037e-01, -1.042536243221e+00, -1.042313802839e+00, +1.041036874304e+00, -1.042547062522e+00, -1.008354506395e+00, -3.507710644284e-01, +1.026026961169e+00, -8.073312088218e-01, +1.030334585160e+00, -1.031745586000e+00, -1.041845872649e+00, +9.630781489664e-01, -2.423122207450e-01, -9.560264103045e-01, -3.483755478137e-01, -9.445419810395e-01, +8.040415760425e-01, -1.052943568533e+00, +1.039782142097e+00, +8.079996472031e-01, +9.455500197665e-01, -9.567565084717e-01, -9.720589524156e-01, -9.550869419630e-01]),
    "reflect_source": dict(
        n=8192, pp=2.126656747188e+00, rms=9.325818907499e-01,
        mean=+8.912082026327e-03,
        probe=[+1.000002071821e+00, +1.000128458900e+00, -6.850837128212e-01, -9.442601159277e-01, +9.721762917510e-01, +9.085408981865e-01, -9.516566900881e-01, +1.054838193279e+00, -1.053829649974e+00, +1.055018940764e+00, -1.056281427932e+00, +9.449514502022e-01, +1.062614563406e+00, +9.495442814978e-01, +1.052844532941e+00, +1.056964722235e+00, -8.100668803143e-01, +1.053695576260e+00, +1.055011285036e+00, +1.053680235443e+00, +9.450776114183e-01, -9.525143689004e-01, -7.945760503231e-01, -1.054871918606e+00, -1.054827390026e+00, -9.451534156802e-01, +1.061119099543e+00, -9.838514155459e-01, -1.054672640566e+00, -2.805142615049e-01, +1.061173526855e+00, -8.964352241590e-01, +1.043260652334e+00, -1.043262926399e+00, -9.449633406645e-01, +9.508995892133e-01, -2.421702363575e-01, -9.437942462843e-01, -2.420335674773e-01, -9.438820329517e-01, +7.965117499813e-01, -1.055135941051e+00, +1.043296762563e+00, +9.048923417053e-01, +1.055556244404e+00, -9.451666241269e-01, -1.054779724676e+00, -1.005390843259e+00]),
    "resonant": dict(
        n=8192, pp=2.038001512609e+00, rms=9.315872873775e-01,
        mean=+7.818702374015e-03,
        probe=[+1.000004357447e+00, +1.000129199911e+00, -6.300799328287e-01, -9.996362845028e-01, +9.173471272633e-01, +9.616083842119e-01, -1.010038841120e+00, +9.997898287669e-01, -9.986873179794e-01, +9.996479737378e-01, -1.001571279544e+00, +1.000000407758e+00, +1.007027348538e+00, +1.000153912824e+00, +1.000123110968e+00, +1.001918211710e+00, -7.553242180785e-01, +1.003553295614e+00, +1.000362247106e+00, +9.988984243493e-01, +1.000115056859e+00, -1.007535814617e+00, -8.496355965821e-01, -9.999472041118e-01, -9.998888639022e-01, -1.000340623384e+00, +1.006011055925e+00, -1.000049329171e+00, -9.998882207623e-01, -2.971048899406e-01, +1.005969036733e+00, -8.420316536528e-01, +9.879639244867e-01, -9.998406466801e-01, -1.000527074489e+00, +1.012758381026e+00, -2.969760102585e-01, -9.990098614684e-01, -2.970117131257e-01, -9.988917050119e-01, +7.548400048442e-01, -9.995280433257e-01, +9.881264187344e-01, +8.496403636024e-01, +9.913730504353e-01, -9.999915224239e-01, -9.999027858407e-01, -1.000009320157e+00]),
    "sparam_channel": dict(
        n=8192, pp=1.927493434505e+00, rms=8.263906647926e-01,
        mean=+8.085409225543e-03,
        probe=[+2.674184139308e-03, -9.048151243988e-01, -4.879393677633e-01, +9.408711672729e-01, +7.132034969623e-01, +7.961270411984e-01, +9.423337509085e-01, -4.909505575574e-01, +9.478050082903e-01, -9.498867259137e-01, -8.998874603083e-01, -3.799909084921e-01, -8.371702362506e-01, -7.160247594498e-01, +9.534718568076e-01, -9.170171308043e-01, +5.814397139131e-01, +9.148672365103e-01, +9.391625521099e-01, -8.981300416857e-01, -3.550813883139e-01, +9.610059722284e-01, -9.539378891168e-01, -8.561212727726e-01, +8.918952620659e-01, +5.560944289641e-01, -8.859816925347e-01, -9.211831351922e-01, +8.605905077241e-01, -2.175621173740e-01, -8.571619490080e-01, +9.476481827104e-01, -8.367057338958e-01, +9.256210287507e-01, -6.877186490814e-01, +9.050842591876e-01, +2.344813705122e-01, +9.197679938827e-01, +9.218186820540e-01, +9.076073519409e-01, +5.850109480662e-01, +8.706211795952e-01, +9.150130709179e-01, +6.693314558188e-01, +9.588854580297e-01, -3.646405793305e-01, -9.192960486580e-01, -9.446364150999e-01]),
}

def _fingerprint(y):
    y = np.asarray(y, float).ravel()
    idx = np.linspace(0, y.size - 1, N_PROBE).astype(int)
    return dict(n=y.size, pp=float(np.ptp(y)), rms=float(np.sqrt((y * y).mean())),
                mean=float(y.mean()), probe=y[idx])


@pytest.mark.parametrize("name", sorted(PATHS))
def test_the_default_path_is_unchanged(name):
    want = REFERENCE[name]
    got = _fingerprint(PATHS[name]())
    # LENGTH is exact. A transform length or a padding rule is a discrete choice, and there is
    # no host arithmetic that rounds it -- so nothing here should be tolerant of it moving.
    assert got["n"] == want["n"], f"{name}: record length moved"
    tol = TOL_FRAC * want["pp"]
    for key in ("pp", "rms", "mean"):
        d = abs(got[key] - want[key])
        assert d <= tol, (f"{name}: {key} moved {d:.3e} = "
                          f"{100 * d / want['pp']:.3e} % of peak-to-peak, over "
                          f"{100 * TOL_FRAC:.1e} %")
    d = np.abs(got["probe"] - np.asarray(want["probe"]))
    worst = int(np.argmax(d))
    assert d.max() <= tol, (
        f"{name}: the waveform moved at probe {worst} of {N_PROBE} by {d.max():.3e} = "
        f"{100 * d.max() / want['pp']:.3e} % of peak-to-peak, over {100 * TOL_FRAC:.1e} %. "
        f"{int((d > tol).sum())} of {N_PROBE} probes are over.")


def test_the_tolerance_is_far_above_the_numerical_floor():
    """What TOL_FRAC has to absorb, measured rather than assumed.

    Two LAPACK builds return Bessel coefficients agreeing to roughly 1e-15 relative. This
    perturbs them by 1e-14 -- deliberately looser -- and requires the realised waveform to move
    by less than a thousandth of TOL_FRAC. If that margin ever closes, the tolerance has stopped
    being a statement about the host's arithmetic and this file needs rethinking, not widening.
    """
    n, n_ui = 8192, 256
    spb = n / n_ui
    tr, _ = P.resolve_rise_time(0.15, spb, floor_samples=P.TR_DEFAULT_FLOOR_SAMPLES)
    x = P._place_symbols(P.carrier_symbols("nrz", n_ui, 3, "legacy"), n, spb, None, None)
    sos = signal.bessel(4, min(0.7 / tr, 0.98), output="sos")
    base = signal.sosfiltfilt(sos, x)
    pp = float(np.ptp(base))
    rng = np.random.default_rng(0)
    free = [0, 1, 2, 4, 5]                    # column 3 of an SOS row must stay exactly 1.0
    worst = 0.0
    for _ in range(8):
        s2 = sos.copy()
        s2[:, free] *= 1.0 + 1e-14 * rng.standard_normal((sos.shape[0], len(free)))
        worst = max(worst, float(np.abs(signal.sosfiltfilt(s2, x) - base).max()))
    assert worst < TOL_FRAC * pp / 1000.0, (
        f"a 1e-14 coefficient perturbation moves the waveform {worst:.3e} = "
        f"{100 * worst / pp:.3e} % of peak-to-peak, which is no longer far below "
        f"TOL_FRAC ({100 * TOL_FRAC:.1e} %)")


# 1e-6 is deliberately absent: a pure gain of exactly TOL_FRAC moves peak-to-peak by exactly
# TOL_FRAC * pp (measured 2.0155e-6 against a tolerance of 2.0155e-6), so that point sits on
# the comparison boundary by construction. Which side it lands is a rounding question, not a
# statement about the test.
@pytest.mark.parametrize("rel,must_catch", [(1e-7, False), (1e-5, True), (1e-4, True),
                                            (1.59e-2, True)])
def test_a_real_change_is_caught(rel, must_catch):
    """The fingerprint discriminates, and where. A check that has never failed is untested.

    `rel` is a pure gain on the pinned carrier. 1.59e-2 is the smallest behaviour change this
    file has had to record, and it must be caught with room to spare; 1e-7 is inside the band
    the tolerance deliberately absorbs and must NOT be.
    """
    want = REFERENCE["nrz"]
    got = _fingerprint(np.asarray(PATHS["nrz"](), float) * (1.0 + rel))
    tol = TOL_FRAC * want["pp"]
    # bool(), not `is`: the probe comparison yields np.bool_, and `np.bool_(False) is False`
    # is False, which would make this assertion fail on its own machinery rather than on the
    # measurement.
    moved = bool(abs(got["pp"] - want["pp"]) > tol
                 or abs(got["rms"] - want["rms"]) > tol
                 or np.abs(got["probe"] - np.asarray(want["probe"])).max() > tol)
    assert moved == must_catch, (
        f"a gain of 1+{rel:.0e} was {'missed' if must_catch else 'caught'}; "
        f"peak-to-peak moved {abs(got['pp'] - want['pp']) / want['pp']:.3e} relative")


def test_extracting_the_loss_kernel_did_not_move_the_lumped_channel():
    """`physics.insertion_loss_db` was lifted out of `lossy_channel` so a cascade SECTION and a
    lumped channel share one loss law. It is a pure extraction: the lumped result must be the
    same floats, on every argument form.

    Byte-exact rather than tolerant, and legitimately so: both sides are computed here, in one
    process, by the same library. No host arithmetic differs between them, so there is nothing
    for a tolerance to absorb.

    Stated on the pinned-length path (`linear=False`), which is the arithmetic this test was
    written against. The claim is about the LOSS LAW -- that one function computes it for both
    callers -- and holding the transform length fixed keeps it a statement about the loss and
    not about the padding. What the padded default does with that same |H| is
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

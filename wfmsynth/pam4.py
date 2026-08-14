"""
wfmsynth.pam4 — synthesize a realistic PAM4 PHY "deep memory" capture: mostly
nominal/typical segments (with realistic channel + noise) plus a small fraction of
segments with injected SI defects. Ground truth (which segment, which defect) is
returned so downstream analysis / detection can be scored against it.

Models a frontier PAM4 lane as SEGMENTS (a "segmented deep memory"):
  * internal generation at N=4096 (256 UI x 16 samples/UI), reusing the validated
    physics primitives (causal channel, reflections, crosstalk, wander, jitter);
  * then DIGITIZED to a scope capture of SEG=1024 samples (= 4 samples/UI), with
    thermal noise + finite-ENOB quantization — i.e. what a high-end real-time scope
    at ~256 GSa/s would actually record for a ~64 GBaud (128 Gb/s) PAM4 lane.
    (Baud/GSa/s are just labels; the pipeline is rate-parameterizable.)

Injected defects, grounded in real PAM4 SI failure modes:
  rlm_compression  driver nonlinearity compresses the outer levels (RLM << 1)
  reflection       impedance discontinuity (connector/via) -> echo/ISI
  crosstalk        aggressor (NEXT/FEXT) coupling burst -> level smear
  jitter_event     PLL/CDR disturbance -> Rj+Pj -> horizontal eye closure
  lossy_marginal   excessive channel insertion loss -> ISI, vertical+horiz closure
  baseline_wander  AC-coupling droop event -> baseline shift

CAPTURE MODEL — segment-groups sharing a fixed channel:
A real scope deep-memory capture is ONE contiguous acquisition of ONE link, so the
CHANNEL (loss, reflection, crosstalk coupling, AC-coupling corner) is the SAME across
all of a capture's segments; only the transmitted PRBS data pattern and per-segment
noise vary segment-to-segment. The 4 CHANNEL-class defects (reflection, crosstalk,
lossy_marginal, baseline_wander) are therefore drawn as a GROUP of `group_size`
contiguous segments sharing one fixed channel-param instance; the 2 non-channel
defects (rlm_compression: TX driver nonlinearity, jitter_event: PLL/CDR event) stay
single-segment regardless of group_size. This is what makes a persistent-but-weak
defect (e.g. a reflection echo far below one segment's noise floor) recoverable by
coherent averaging across a group's segments, the way real SI defects actually get
found. group_size=1 (the default) reproduces a flat, single-segment-instance capture
(same code path, bit-identical output).
"""
import numpy as np
from scipy import signal as sg

from wfmsynth import physics as P

N = P.N                     # 4096 internal grid
UI = 256                    # unit intervals per segment
SPS_INT = N // UI           # 16 internal samples per UI
SEG = 1024                  # scope samples per stored segment
SCOPE_SPS = SEG // UI       # 4 samples/UI at the scope

PATHOLOGIES = ["rlm_compression", "reflection", "crosstalk",
               "jitter_event", "lossy_marginal", "baseline_wander"]

# channel-class pathologies: the SI defect lives in the channel, not the data/event,
# so it is physically persistent across every segment of one capture/group.
CHANNEL_CLASS = {"reflection", "crosstalk", "lossy_marginal", "baseline_wander"}


def _pam4(rng, rlm=1.0, tr=0.4):
    """One 256-UI PAM4 burst at the internal rate. rlm<1 compresses outer levels."""
    lv = np.array([-1.0, -1 / 3, 1 / 3, 1.0]) if rlm >= 1.0 else np.array([-rlm, -1 / 3, 1 / 3, rlm])
    syms = rng.integers(0, 4, UI)
    x = np.repeat(lv[syms], SPS_INT).astype(float)
    trs = max(tr * SPS_INT, 2.0)
    return sg.sosfiltfilt(sg.bessel(4, min(0.7 / trs, 0.95), output="sos"), x), syms


def _digitize(x, rng, enob=5.5, snr_db=32.0):
    """Internal high-res waveform -> scope capture: resample to SEG, add thermal
    noise at snr_db, quantize at finite ENOB (a real scope's noise floor)."""
    xr = sg.resample_poly(x, SEG, N)
    span = np.ptp(xr) + 1e-9
    xr = xr + rng.normal(0.0, span / 10 ** (snr_db / 20), SEG)     # thermal noise
    lsb = span / 2 ** enob
    xr = lsb * np.round(xr / lsb)                                  # ENOB quantization
    return xr.astype(np.float32)


def _draw_channel_params(kind, rng):
    """Draw the CHANNEL-class params for one group: shared across all segments in
    the group (the physical link doesn't change segment-to-segment)."""
    p = {}
    if kind == "lossy_marginal":
        p["length_in"] = rng.uniform(11.0, 15.0)
    else:
        p["base_len"] = rng.uniform(3.0, 6.0)
        if kind == "reflection":
            p["td_frac"] = rng.uniform(0.05, 0.12)
            p["gamma_s"] = rng.uniform(0.30, 0.45)
        elif kind == "crosstalk":
            p["coupling"] = rng.uniform(0.12, 0.22)
        elif kind == "baseline_wander":
            p["fc_frac"] = rng.uniform(0.010, 0.020)
            p["wander_scale"] = rng.uniform(0.6, 0.9)
    return p


def _segment(kind, rng, chan=None):
    """Build one segment of the given kind ('nominal' or a pathology). `chan`, if
    given, supplies pre-drawn CHANNEL-class params (shared across a group) instead
    of drawing them fresh here — everything else (PRBS data, noise) is still drawn
    fresh per call, i.e. per segment."""
    x, _ = _pam4(rng, rlm=1.0)
    base_len = chan["base_len"] if chan and "base_len" in chan else rng.uniform(3.0, 6.0)
    if kind == "rlm_compression":
        x, _ = _pam4(rng, rlm=rng.uniform(0.72, 0.82))
        x = P.lossy_channel(x, length_in=base_len, tand=0.015, causal=True)
    elif kind == "lossy_marginal":
        length_in = chan["length_in"] if chan else rng.uniform(11.0, 15.0)
        x = P.lossy_channel(x, length_in=length_in, tand=0.024, causal=True)
    else:
        x = P.lossy_channel(x, length_in=base_len, tand=0.015, causal=True)   # nominal baseline
        if kind == "reflection":
            td_frac = chan["td_frac"] if chan else rng.uniform(0.05, 0.12)
            gamma_s = chan["gamma_s"] if chan else rng.uniform(0.30, 0.45)
            x = P.multi_reflection(x, td_frac=td_frac, gamma_s=gamma_s, gamma_l=0.4)
        elif kind == "crosstalk":
            aggr, _ = _pam4(rng)
            coupling = chan["coupling"] if chan else rng.uniform(0.12, 0.22)
            x = P.crosstalk(x, aggr, coupling=coupling, kind="fext")
        elif kind == "jitter_event":
            x = P.inject_jitter(x, sigma_rj=rng.uniform(3.0, 6.0), a_pj=rng.uniform(2.0, 5.0),
                                f_pj=rng.uniform(3.0, 8.0), rng=rng)
        elif kind == "baseline_wander":
            fc_frac = chan["fc_frac"] if chan else rng.uniform(0.010, 0.020)
            wander_scale = chan["wander_scale"] if chan else rng.uniform(0.6, 0.9)
            lf = x - P.ac_couple(x, fc_frac=fc_frac)
            x = x - wander_scale * lf
        # kind == "nominal": baseline channel only
    # ambient (label-preserving) crosstalk on everything, for realism -- genuinely
    # per-segment (independent aggressor activity), stays per-segment in a group too
    if rng.uniform() < 0.5:
        aggr, _ = _pam4(rng)
        x = P.crosstalk(x, aggr, coupling=rng.uniform(0.01, 0.03), kind="next",
                        td_frac=rng.uniform(0.02, 0.10))
    return _digitize(x, rng)


def _deep_capture_flat(n_segments, needle_rate, seed):
    """Original flat capture: every needle is an independent single-segment
    instance (no cross-segment structure). This is EXACTLY the pre-refactor
    code path, bit-identical output — the group_size=1 default routes here."""
    rng = np.random.default_rng(seed)
    X = np.empty((n_segments, SEG), np.float32)
    labels = np.empty(n_segments, dtype=object)
    n_needle = max(1, int(round(n_segments * needle_rate)))
    needle_idx = set(int(i) for i in rng.choice(n_segments, n_needle, replace=False))
    for i in range(n_segments):
        kind = PATHOLOGIES[rng.integers(len(PATHOLOGIES))] if i in needle_idx else "nominal"
        X[i] = _segment(kind, rng)
        labels[i] = kind
    needle_idx = np.array(sorted(needle_idx))
    group_id = np.full(n_segments, -1, dtype=int)
    group_id[needle_idx] = np.arange(len(needle_idx))     # each needle is its own size-1 group
    return dict(X=X, labels=labels, needle_idx=needle_idx, group_id=group_id,
                sps=SCOPE_SPS, ui=UI, seg=SEG)


def _deep_capture_grouped(n_segments, needle_rate, seed, group_size):
    """Grouped capture: channel-class needle instances (reflection/crosstalk/
    lossy_marginal/baseline_wander) occupy a CONTIGUOUS block of `group_size`
    segments sharing one fixed channel-param draw; rlm_compression/jitter_event
    instances stay single-segment. Non-needle segments fill the remaining gaps."""
    rng = np.random.default_rng(seed)
    target = max(1, int(round(n_segments * needle_rate)))
    instances = []                                    # list of (kind, size)
    placed = 0
    while placed < target:
        kind = PATHOLOGIES[rng.integers(len(PATHOLOGIES))]
        size = group_size if kind in CHANNEL_CLASS else 1
        instances.append((kind, size))
        placed += size
    order = rng.permutation(len(instances))
    instances = [instances[i] for i in order]
    total_needle = sum(s for _, s in instances)
    n_nominal = max(0, n_segments - total_needle)
    gaps = rng.multinomial(n_nominal, np.ones(len(instances) + 1) / (len(instances) + 1))

    X = np.empty((n_segments, SEG), np.float32)
    labels = np.empty(n_segments, dtype=object)
    group_id = np.full(n_segments, -1, dtype=int)
    needle_idx = []
    pos = 0
    for gi, (kind, size) in enumerate(instances):
        for _ in range(gaps[gi]):
            X[pos] = _segment("nominal", rng); labels[pos] = "nominal"; pos += 1
        chan = _draw_channel_params(kind, rng) if (kind in CHANNEL_CLASS and size > 1) else None
        for _ in range(size):
            if pos >= n_segments:
                break
            X[pos] = _segment(kind, rng, chan=chan)
            labels[pos] = kind
            group_id[pos] = gi
            needle_idx.append(pos)
            pos += 1
    for _ in range(gaps[-1]):
        if pos >= n_segments:
            break
        X[pos] = _segment("nominal", rng); labels[pos] = "nominal"; pos += 1
    assert pos == n_segments, f"placement bug: pos={pos} != n_segments={n_segments}"
    return dict(X=X, labels=labels, needle_idx=np.array(sorted(needle_idx)), group_id=group_id,
                sps=SCOPE_SPS, ui=UI, seg=SEG)


def deep_capture(n_segments=2000, needle_rate=0.02, seed=0, group_size=1):
    """Return a segmented deep-memory capture: X (n,1024) scope segments, per-segment
    labels ('nominal' or a pathology), needle indices (ground truth), and group_id
    (which needle-instance group each needle segment belongs to; -1 for nominal).

    group_size=1 (default): every needle is an independent single-segment instance,
    identical to the original capture model — bit-identical output, all existing
    callers unaffected.
    group_size>1: channel-class needle instances (reflection/crosstalk/lossy_marginal/
    baseline_wander) become GROUPS of `group_size` contiguous segments sharing one
    fixed channel (see module docstring); rlm_compression/jitter_event stay
    single-segment. Total needle-segment count may overshoot `needle_rate*n_segments`
    by up to (group_size-1) since group instances aren't split.
    """
    if group_size <= 1:
        return _deep_capture_flat(n_segments, needle_rate, seed)
    return _deep_capture_grouped(n_segments, needle_rate, seed, group_size)

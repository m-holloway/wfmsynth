"""
wfmsynth — physics-informed synthetic waveform generation.

Generate realistic voltage-vs-time signals grounded in real signal-integrity physics:
frequency-dependent (causal) channels, transmission-line reflections, crosstalk,
decomposed jitter, AC-coupling, plus digital/RF carriers and full "deep-memory" scope
captures. Everything is validated (`wfmsynth.validate`) and depends only on numpy/scipy.

Modules / public API:
  physics       low-level primitives — lossy_channel, multi_reflection, crosstalk,
                ac_couple, inject_jitter, nrz, pam4, am, fm, psk, qam, chirp, ...
  impairments   apply_impairment(name, x, rng), domain_randomize(x, rng), IMPAIRMENTS
  grammar       carrier(), envelope(), sample(), generate() — compositional signals
  pam4          deep_capture() — realistic segmented PAM4 scope captures with defects
  validate      run as `python -m wfmsynth.validate` — hard physics-property assertions
"""
from . import physics, impairments, grammar, pam4
from .physics import N, T
from .impairments import IMPAIRMENTS, apply_impairment, domain_randomize
from .grammar import sample, generate, CARRIERS, ENVELOPES
from .pam4 import deep_capture, PATHOLOGIES

__version__ = "0.1.0"
__all__ = [
    "physics", "impairments", "grammar", "pam4",
    "N", "T", "IMPAIRMENTS", "apply_impairment", "domain_randomize",
    "sample", "generate", "CARRIERS", "ENVELOPES", "deep_capture", "PATHOLOGIES",
    "__version__",
]

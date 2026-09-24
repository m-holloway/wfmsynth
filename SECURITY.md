# Security

## Reporting

Report a suspected vulnerability privately through
[GitHub's security advisories](https://github.com/m-holloway/wfmsynth/security/advisories/new)
rather than a public issue. Please include a reproduction — for this library that is usually a
small file or a recipe plus the call that loads it.

## What this library is, for threat-modelling purposes

`wfmsynth` is a numerical library. It performs no network I/O, spawns no subprocesses, and
contains no `eval`, `exec` or dynamic import on any input-driven path. Its attack surface is
therefore **the files and recipes it is asked to read**, and the realistic scenario is a capture
file, an S-parameter file or a recipe that came from somewhere other than the person running it.

### Capture files — pickling is refused

`.npy` and `.npz` can carry a pickled object array, and unpickling one executes arbitrary code.
`Signal.capture()` exists to read files the user did not necessarily produce, so this is the one
genuinely hostile-input path in the library.

`capture.load_values` loads with `allow_pickle=False` explicitly — which is also numpy's default
since 1.16.3 — and reports a pickled file as a named refusal rather than letting numpy's generic
message through. `tests/test_capture_pickle_refused.py` attempts a real code-execution payload
through both `.npy` and `.npz` and asserts it is refused, so the flag cannot quietly go away in a
refactor or a future numpy default.

### Recipes — data, not code

A recipe is JSON: a grid, a seed, and an ordered list of ops with their parameters.
`Signal.from_recipe` does not import, resolve or call anything named in the document. Op names
are looked up in a fixed table at render time, and every parameter goes through the same
`opkeys` gate as the builder API, so an unknown key is refused rather than recorded.

Two honest limits on that:

- **Resource exhaustion.** A recipe states its own record length. A hostile one can ask for a
  record large enough to exhaust memory. If you replay recipes you did not author — a public
  dataset, a submitted archive — bound `grid.n` before rendering.
- **File references.** A `capture` op names a path, so replaying an untrusted recipe can cause a
  read of any file the process can already reach. It can only be interpreted as numeric samples
  (`.npy`/`.npz`/`.csv`, pickling refused), but the values could end up in the rendered output.
  `Signal.capture(embed=True)` drops the file reference entirely, and a recorded sha256 over the
  sample values means a file that changed is a named error rather than a different waveform under
  the same recipe.

### Text-format parsers

Touchstone (`.sNp`) and MDIF (`.mdf`) are parsed with ordinary numeric conversion — no `eval`.
Malformed input raises. These parsers are not hardened against adversarial input designed to be
slow; if you ingest such files at scale from untrusted sources, run it where a hang or an
allocation failure is contained.

### Optional formats

HDF5 (`h5py`) and Zarr are optional extras and are not part of a default install. Their own
security properties are their projects' to state.

## Supported versions

Pre-1.0. Fixes land on `main` and in the next release; there are no maintained back-branches.
See [`CHANGELOG.md`](CHANGELOG.md) and the `v*` tags.

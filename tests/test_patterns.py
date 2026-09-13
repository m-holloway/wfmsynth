"""The pattern registry: a NAME resolves to symbols, and the recipe can be replayed without it.

The design decision this file enforces: MECHANISMS IN THE LIBRARY, STANDARDS KNOWLEDGE OUT.
A PRBS is a polynomial, a seed and a length -- physics, and it lives here. "Which pattern a
given standard names for a given lane rate" is a fact that changes with the revision of a
document and is the CALLER's to state, via `register`.

Four properties, and the last two are the ones that bite:

  * THE MECHANISM IS THE ONE ALREADY IN THE KERNEL. The general arbitrary-polynomial LFSR must
    reproduce `physics.prbs` bit-for-bit on every order in `physics.PRBS_TAPS`, and the
    block-repeat generator must reproduce `physics.clock_pattern`. That is the calibration:
    the arbitrary-polynomial claim is only worth as much as the known-answer case underneath it.
  * A RECIPE RECORDS THE NAME **AND** THE RESOLVED PARAMETERS. Name alone is unreplayable
    without the registry; polynomial alone is unreadable. Both, or the contract is broken.
  * A CONSUMER WITHOUT THE REGISTRY ENTRY STILL RENDERS THE RECORD -- through the generic
    mechanism the block names, or through embedded symbols-as-data, which needs no code at all.
  * A USER GENERATOR THAT IS MISSING OR HAS CHANGED IS NAMED, NOT SILENTLY DIFFERENT. The
    consumer is told `needs pattern 'x' @ <hash>` instead of rendering other samples under the
    same name.
"""
import json

import numpy as np
import pytest

import wfmsynth.patterns as PAT
import wfmsynth.physics as P
from wfmsynth.compose import Signal
from wfmsynth.grid import Grid
from wfmsynth.measure import pattern_period

GRID = Grid(fs=256e9, baud=16e9, n=1 << 13)


@pytest.fixture(autouse=True)
def _registry_sandbox():
    """Every test gets the shipped registry and leaves it as it found it: a registration that
    leaked between tests would make the missing-entry cases pass for the wrong reason."""
    saved = dict(PAT.REGISTRY)
    yield
    PAT.REGISTRY.clear()
    PAT.REGISTRY.update(saved)


# --------------------------------------------------------------- calibration: known answers
@pytest.mark.parametrize("order", sorted(P.PRBS_TAPS))
def test_the_general_lfsr_reproduces_the_kernels_prbs_bit_for_bit(order):
    """KNOWN ANSWER FIRST. The arbitrary-polynomial LFSR is only trustworthy on a polynomial
    nobody has checked if it is identical to the checked one on the polynomials the kernel
    already ships. Bit-for-bit, not statistically."""
    taps = P.PRBS_TAPS[order]
    bits = PAT.lfsr_bits(taps, 4096, seed=7)
    assert (bits == P.prbs(order, 4096, seed=7)).all()
    assert (PAT.lfsr_bits(taps, 100, seed=7, phase=13)
            == P.prbs(order, 100, seed=7, phase=13)).all()


def test_the_general_lfsr_is_maximal_length_on_a_polynomial_the_kernel_does_not_ship():
    """x^10 + x^7 + 1 is primitive, so its period is 2^10-1 = 1023 and every non-zero state
    appears once. A period of anything less would mean the tap convention is being read
    differently from `physics.prbs` -- which the calibration above cannot show, because every
    order in PRBS_TAPS has its second tap in a different place."""
    bits = PAT.lfsr_bits((10, 7), 1023 * 2 + 40, seed=1)
    assert (bits[:1023] == bits[1023:2046]).all()
    assert not (bits[:511] == bits[512:1023]).all()
    assert pattern_period(bits)[0] == 1023


def test_block_repeat_reproduces_the_kernels_clock_pattern():
    """The second mechanism, calibrated the same way: the 1010... clock is a two-symbol block
    repeated, so block-repeat must give exactly what `physics.clock_pattern` gives."""
    assert (PAT.resolve("clock", length=1000) == P.clock_pattern(1000)).all()
    x = PAT.block_repeat([1.0, 0.5, -1.0], length=7)
    assert (x == np.array([1.0, 0.5, -1.0, 1.0, 0.5, -1.0, 1.0])).all()
    assert len(PAT.block_repeat([1.0, -1.0], repeat=5)) == 10


def test_registered_prbs_names_resolve_to_the_kernels_carrier_symbols():
    """The registry's `prbs<order>` entries and `physics.carrier_symbols('nrz', ...)` must be
    the same stream: two ways to say a polynomial that disagreed would be the worst kind of
    defect, because both look right in isolation."""
    for order in (7, 13, 31):
        got = PAT.resolve(f"prbs{order}", length=2048, seed=5)
        assert (got == P.carrier_symbols("nrz", 2048, seed=5, pattern=f"prbs{order}")).all()


def test_symbol_statistics_of_a_prbs_are_the_textbook_values():
    """A maximal-length PRBS has transition density 0.5 and equal mark/space density to within
    one symbol per period. This is the instrument the caller checks a STANDARD's published
    conformance numbers with, so it is calibrated here on a sequence whose answer is closed
    form."""
    st = PAT.symbol_stats(PAT.resolve("prbs9", length=511))
    assert abs(st["transition_density"] - 0.5) < 2 / 511
    assert st["levels"] == 2
    assert abs(st["level_probability"][1.0] - 0.5) < 2 / 511


# --------------------------------------------------------------- the recipe contract
def test_a_block_records_the_name_and_the_resolved_parameters_both():
    """THE CONTRACT. Name alone cannot be replayed without the registry; parameters alone
    cannot be read. A block carries both, plus the metadata that makes it self-describing."""
    b = PAT.describe("prbs13", length=4096, seed=9)
    assert b["name"] == "prbs13"
    assert b["params"]["taps"] == list(P.PRBS_TAPS[13])      # resolved: the polynomial
    assert b["params"]["seed"] == 9 and b["params"]["length"] == 4096
    assert b["levels"] == 2 and b["period"] == 8191
    assert b["generator"] == "lfsr"
    assert json.loads(json.dumps(b)) == b                    # JSON-native, hand-editable


def test_a_block_does_not_restate_a_default_it_was_not_given():
    """The block is a content address. Restating a generator default would change the hash of
    every record carrying the pattern the day that default changes, so only the registry's own
    resolution and the caller's own arguments are recorded."""
    b = PAT.describe("prbs7", length=64)
    assert "phase" not in b["params"] and "seed" not in b["params"]
    assert (PAT.replay(b) == PAT.resolve("prbs7", length=64)).all()


def test_a_block_replays_from_the_mechanism_with_the_registry_entry_gone():
    """The consumer's side of the contract: delete the entry and the block still renders,
    because the resolved parameters name a mechanism that ships in the library."""
    b = PAT.describe("prbs13", length=4096, seed=9)
    want = PAT.replay(b)
    del PAT.REGISTRY["prbs13"]
    assert (PAT.replay(b) == want).all()


# --------------------------------------------------------------- user extension, no lambdas
def test_a_decorated_generator_registers_and_records_a_source_hash():
    @PAT.pattern("two_up_two_down", levels=2, period=4,
                 source="not from a standard: a fixture")
    def _two_up_two_down(length, seed=1):
        return PAT.block_repeat([1.0, 1.0, -1.0, -1.0], length=length)

    b = PAT.describe("two_up_two_down", length=16)
    assert b["digest"] and len(b["digest"]) == 12
    assert (PAT.replay(b) == np.tile([1.0, 1.0, -1.0, -1.0], 4)).all()


def test_a_missing_user_generator_is_named_with_its_hash():
    @PAT.pattern("needs_me", levels=2)
    def _needs_me(length, seed=1):
        return PAT.block_repeat([1.0, -1.0, -1.0], length=length)

    b = PAT.describe("needs_me", length=9)
    del PAT.REGISTRY["needs_me"]
    with pytest.raises(PAT.PatternUnavailable) as e:
        PAT.replay(b)
    assert f"needs pattern 'needs_me' @ {b['digest']}" in str(e.value)


def test_a_changed_user_generator_is_a_named_mismatch_not_different_samples():
    """The defect this exists to prevent: the same NAME resolving to a different sequence on
    the consumer's machine, silently. The hash makes it an error that says what is needed."""
    @PAT.pattern("drifted", levels=2)
    def _drifted(length, seed=1):
        return PAT.block_repeat([1.0, -1.0, -1.0], length=length)

    b = PAT.describe("drifted", length=9)
    del PAT.REGISTRY["drifted"]

    @PAT.pattern("drifted", levels=2)
    def _drifted_v2(length, seed=1):
        return PAT.block_repeat([1.0, 1.0, -1.0], length=length)     # a different sequence

    with pytest.raises(PAT.PatternMismatch) as e:
        PAT.replay(b)
    assert f"needs pattern 'drifted' @ {b['digest']}" in str(e.value)
    assert PAT.REGISTRY["drifted"].digest in str(e.value)            # and what is installed


def test_registering_a_name_twice_raises_rather_than_shadowing():
    PAT.register("mine", lambda length, seed=1: PAT.block_repeat([1.0, -1.0], length=length),
                 levels=2, period=2)
    with pytest.raises(ValueError, match="already registered"):
        PAT.register("mine", lambda length: np.ones(length), levels=2)
    PAT.register("mine", lambda length: np.ones(length), levels=2, replace=True)   # explicit


def test_a_generator_must_take_length_as_the_contract_says():
    with pytest.raises(ValueError, match="length"):
        PAT.register("wrong_signature", lambda n_symbols: np.ones(n_symbols), levels=2)


# --------------------------------------------------------------- round-trip through a recipe
def _recipe_round_trip(sig):
    """Render, serialize to JSON, rebuild from the parsed JSON, render again. Anything that
    survives only in the live object and not in the JSON shows up here as a difference."""
    x = sig.waveform()
    r = json.loads(sig.to_json())
    return x, Signal.from_recipe(r).waveform()


def test_a_named_pattern_round_trips_through_json_bit_for_bit():
    sig = Signal(seed=3, grid=GRID).pattern("prbs13", length=512, seed=9, tr_frac=0.15)
    x, y = _recipe_round_trip(sig)
    assert (x == y).all()
    op = sig.recipe()["ops"][0]
    assert op["op"] == "symbols" and op["pattern"]["name"] == "prbs13"
    assert op["pattern"]["params"]["taps"] == list(P.PRBS_TAPS[13])


def test_a_named_pattern_renders_with_the_registry_entry_gone():
    sig = Signal(seed=3, grid=GRID).pattern("prbs13", length=512, seed=9)
    x = sig.waveform()
    r = json.loads(sig.to_json())
    del PAT.REGISTRY["prbs13"]
    assert (Signal.from_recipe(r).waveform() == x).all()


def test_symbols_as_data_round_trips_with_an_empty_registry():
    """The always-works fallback. `embed=True` puts the symbols in the recipe, so the record is
    reproducible with NO generator code on the consumer's side at all -- which is the only
    guarantee that holds for a pattern whose generator the consumer will never have."""
    sig = Signal(seed=3, grid=GRID).pattern("prbs9", length=256, seed=4, embed=True)
    x = sig.waveform()
    r = json.loads(sig.to_json())
    assert len(r["ops"][0]["symbols"]) == 256
    assert r["ops"][0]["pattern"]["name"] == "prbs9"          # still self-describing
    PAT.REGISTRY.clear()
    assert (Signal.from_recipe(r).waveform() == x).all()


def test_a_user_pattern_round_trips_through_json_and_embeds():
    @PAT.pattern("fixture_word", levels=2, period=6, source="not from a standard: a fixture")
    def _fixture_word(length, seed=1):
        return PAT.block_repeat([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], length=length)

    named = Signal(seed=1, grid=GRID).pattern("fixture_word", length=192)
    embedded = Signal(seed=1, grid=GRID).pattern("fixture_word", length=192, embed=True)
    x, y = _recipe_round_trip(named)
    assert (x == y).all()
    r = json.loads(embedded.to_json())
    PAT.REGISTRY.clear()
    assert (Signal.from_recipe(r).waveform() == x).all()      # same samples, no generator


def test_an_embedded_recipe_is_identical_to_the_named_one_in_samples():
    a = Signal(seed=2, grid=GRID).pattern("prbs7", length=128, seed=3)
    b = Signal(seed=2, grid=GRID).pattern("prbs7", length=128, seed=3, embed=True)
    assert (a.waveform() == b.waveform()).all()
    assert a.sha256() != b.sha256()          # different recipes: one needs code, one does not


def test_a_pattern_op_is_a_source_and_can_carry_a_lead_in():
    """The pattern op is the `symbols` op, so it inherits the source's classification: it is a
    source for stage-kind homing and it is renderable with a lead-in. A new op would have had
    to be reasoned about in both places; this one already has been."""
    sig = Signal(seed=1, grid=GRID, lead_in=True).pattern("prbs9", length=512)
    assert sig.stage_kinds() == ["source"]
    assert len(sig.lossy(length_in=1.0).waveform()) == GRID.n


# --------------------------------------------------------------- markers and metadata
def test_a_marker_locates_the_pattern_position():
    """A marker is the alignment key: a symbol subsequence that occurs once per period, so a
    consumer can find WHERE in the pattern a capture starts. Declared by the caller, because
    which word a standard designates is the caller's knowledge."""
    PAT.register("marked", lambda length, seed=1: PAT.block_repeat(
        [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0], length=length),
        levels=2, period=8, marker=[-1.0, 1.0, -1.0, -1.0])
    syms = PAT.resolve("marked", length=8 * 5)
    pos = PAT.find_marker(syms, PAT.get("marked").marker)
    assert list(pos) == [4, 12, 20, 28, 36]
    assert PAT.describe("marked", length=8)["marker"] == [-1.0, 1.0, -1.0, -1.0]


def test_an_unregistered_name_says_what_is_registered():
    with pytest.raises(PAT.PatternUnavailable) as e:
        PAT.resolve("no_such_pattern", length=8)
    assert "prbs13" in str(e.value)


def test_resolve_needs_a_length_and_says_so():
    with pytest.raises(TypeError, match="length"):
        PAT.resolve("prbs7")


def test_the_arbitrary_carrier_wart_is_documented_and_points_at_the_registry():
    """`physics.arbitrary(fn=...)` takes a callable, so a recipe carrying it cannot be replayed
    from JSON. That is not fixed here -- it is NAMED, in the place someone reaching for it
    reads, with the reproducible route next to it."""
    doc = P.arbitrary.__doc__
    assert "cannot" in doc and "patterns" in doc

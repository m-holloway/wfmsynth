"""The skill is one canonical copy, its examples run, and its version moves when it changes.

The skill is shipped documentation that an agent acts on, so a wrong line in it becomes wrong code
in someone else's project. Two independent agents reviewing an earlier version found four claims
that were false, each of which would have been caught by running the examples.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO / ".claude" / "skills" / "wfmsynth"
SKILL = SKILL_DIR / "SKILL.md"
REFERENCE = SKILL_DIR / "REFERENCE.md"


def test_the_skill_and_its_reference_exist_where_an_agent_looks():
    assert SKILL.is_file(), "an agent working in a clone finds the skill at .claude/skills/"
    assert REFERENCE.is_file()


def test_there_is_exactly_one_copy_of_the_skill_in_the_tree():
    """A second copy drifts, and the stale one is the one somebody reads."""
    found = sorted(p.relative_to(REPO) for p in REPO.rglob("SKILL.md")
                   if ".git" not in p.parts and ".venv" not in p.parts)
    assert found == [SKILL.relative_to(REPO)], f"more than one SKILL.md: {found}"


def test_the_frontmatter_carries_a_name_and_an_integer_version():
    head = SKILL.read_text().split("---")[1]
    assert re.search(r"^name:\s*wfmsynth\s*$", head, re.M)
    assert re.search(r"^description:\s*\S", head, re.M)
    m = re.search(r"^version:\s*(\d+)\s*$", head, re.M)
    assert m, "no integer version in the frontmatter; an installed copy cannot be compared"
    assert int(m.group(1)) >= 1


def test_the_installer_points_at_the_canonical_location():
    sh = (REPO / "skills" / "install.sh").read_text()
    assert ".claude/skills" in sh
    assert "REFERENCE.md" in sh, "the reference has to travel with the skill or its links dangle"


def test_the_skill_names_the_reference_it_relies_on():
    assert "REFERENCE.md" in SKILL.read_text()


def test_the_skill_does_not_repeat_claims_the_library_refuses():
    """Each of these appeared in a published version of the skill and is false. They are pinned
    here by name so they cannot come back through a well-meaning edit."""
    both = SKILL.read_text() + REFERENCE.read_text()
    # The op has no `symbols` parameter. Naming the wrong form in order to warn about it is
    # fine; recommending it is not, so every mention has to sit on a line that denies it.
    for m in re.finditer(r"[^\n]*carrier\([^\n]*symbols[^\n]*", both):
        line = m.group(0)
        assert "does not exist" in line or "not take" in line, \
            f"carrier(symbols=) presented as usable: {line.strip()}"
    assert "grid.to_volts" not in both               # no such method

    # `n_out` is a record length. The published wording that made it a rate, verbatim:
    assert "set by `sample_clock(n_out=...)`" not in both
    # and the correction has to be present, not merely the absence of the error
    assert "record LENGTH, not a rate" in SKILL.read_text()
    assert "AcquisitionProfile" in SKILL.read_text()

    # the other three corrections, each pinned by the fact it asserts
    assert "normalised amplitude" in SKILL.read_text()
    assert "does not seed the pattern phase" in SKILL.read_text()
    assert "Signal.pattern(name, length=...)" in SKILL.read_text()


def test_every_python_block_is_syntactically_valid():
    for path in (SKILL, REFERENCE):
        blocks = re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
        for i, b in enumerate(blocks):
            compile(b, f"{path.name}:block{i}", "exec")

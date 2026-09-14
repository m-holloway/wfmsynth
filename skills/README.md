# Agent skills

```
skills/
  install.sh          idempotent installer; --check, --local, --all
  wfmsynth/
    SKILL.md          the skill, versioned in its own frontmatter
```

## Install

From a clone:

```bash
./skills/install.sh
```

Without one:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/m-holloway/wfmsynth/main/skills/install.sh)
```

It installs into the user-scope skills directory of each agent CLI it finds (Claude Code, Codex,
Cursor). `--local` installs into `./.claude/skills` for one project instead, and `--all` installs
for every supported target whether or not it is detected. Re-running is safe: an unchanged file is
left alone and the script says so.

## Versions

Each skill carries `version:` in its frontmatter, and the copy in this repository is the source of
truth. `./skills/install.sh --check` prints the repository version, the published version and
whatever is installed, and changes nothing.

Raise `version` whenever you change a skill. An unchanged number on changed content is what makes
an installed copy undetectably stale.

## Adding a skill

One directory per skill, named as the skill is invoked, holding a `SKILL.md` whose frontmatter
carries `name`, `description` and `version`. `install.sh` currently installs `wfmsynth` by name;
extend `SKILL_NAME` to a list when there is a second one.

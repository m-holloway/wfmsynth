# Agent skills

The skill itself lives at `.claude/skills/wfmsynth/`, which is where an agent working **inside a
clone of this repository** finds it with no installation at all. This directory holds the installer
for everyone else.

```
.claude/skills/wfmsynth/
  SKILL.md            the skill: the shape, and what people get wrong. Versioned in its frontmatter
  REFERENCE.md        the worked examples, read on demand
skills/
  install.sh          idempotent installer; --check, --local, --all
  README.md           this file
```

One canonical copy, so there is nothing to keep in sync.

## Install

If you are working inside a clone, you already have it. To use the skill from your own project,
where wfmsynth is a dependency rather than the working directory:

```bash
./skills/install.sh
```

Or without a clone:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/m-holloway/wfmsynth/main/skills/install.sh)
```

It installs into the user-scope skills directory of each agent CLI it finds (Claude Code, Codex,
Cursor), carrying both files. `--local` installs into `./.claude/skills` for one project, and
`--all` installs for every supported target whether or not it is detected. Re-running is safe: an
unchanged pair is left alone and the script says so.

## Windows

`install.sh` needs bash, so it runs under Git Bash or WSL but not in a plain Windows shell. The
skill is two files in a directory, so fetching them directly is equivalent. In PowerShell:

```powershell
$dir = "$HOME\.claude\skills\wfmsynth"
$src = "https://raw.githubusercontent.com/m-holloway/wfmsynth/main/.claude/skills/wfmsynth"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
curl.exe -fsSL "$src/SKILL.md"     -o "$dir\SKILL.md"
curl.exe -fsSL "$src/REFERENCE.md" -o "$dir\REFERENCE.md"
```

`curl.exe` ships with Windows 10 and later. `Invoke-WebRequest -Uri "$src/SKILL.md" -OutFile
"$dir\SKILL.md"` does the same if you would rather not shell out.

Both files have to land, not just `SKILL.md`: the skill points at `REFERENCE.md` for every worked
example, and the link dangles without it. To check what you have:

```powershell
Select-String -Path "$dir\SKILL.md" -Pattern '^version:' | Select-Object -First 1
```

## Versions

`SKILL.md` carries `version:` in its frontmatter, and the copy in this repository is the source of
truth. `./skills/install.sh --check` prints the repository version, the published version and
whatever is installed, and changes nothing.

Raise `version` whenever you change either file. An unchanged number on changed content is what
makes an installed copy undetectably stale.

## What goes where

`SKILL.md` is loaded whenever the skill triggers, so it stays lean: what the library is, the shape
of a chain, the two numbers that decide whether a record means anything, and the list of things
that go wrong silently. `REFERENCE.md` holds the worked examples and is read when needed.

Every code block in both files is executed against the library with warnings as errors before
either is published. Several of the entries in the "what people get wrong" list are there because
a snippet in an earlier version of this skill was wrong in exactly that way.

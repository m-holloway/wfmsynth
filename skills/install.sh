#!/usr/bin/env bash
# Install the wfmsynth agent skill. Idempotent: re-running it is safe, and it reports what it did.
#
#   ./install.sh              install or update for every agent CLI it finds
#   ./install.sh --check      compare installed / repository / published versions, change nothing
#   ./install.sh --local      install into ./.claude/skills of the current project instead
#   ./install.sh --all        install for every supported target whether or not it is detected
#
# Also works with no clone:
#   bash <(curl -fsSL https://raw.githubusercontent.com/m-holloway/wfmsynth/main/skills/install.sh)
set -euo pipefail

SKILL_NAME="wfmsynth"
RAW_BASE="https://raw.githubusercontent.com/m-holloway/wfmsynth/main/.claude/skills"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"
SRC="${HERE:+$HERE/../.claude/skills/$SKILL_NAME/SKILL.md}"

MODE="install"; SCOPE="user"; FORCE_ALL=0
for a in "$@"; do
  case "$a" in
    --check) MODE="check" ;;
    --local) SCOPE="local" ;;
    --all)   FORCE_ALL=1 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]:-$0}"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

version_of() {                       # the frontmatter version of a SKILL.md, or empty
  [ -f "$1" ] || return 0
  sed -n 's/^version: *//p' "$1" | head -1
}

# The source: the sibling copy in this clone, or the published one fetched to a temp file.
TMP=""
cleanup() { [ -n "$TMP" ] && rm -rf "$TMP" || true; }
trap cleanup EXIT
if [ -z "$SRC" ] || [ ! -f "$SRC" ]; then
  command -v curl >/dev/null || { echo "no local skill file and no curl to fetch one" >&2; exit 1; }
  TMP="$(mktemp -d)"
  curl -fsSL "$RAW_BASE/$SKILL_NAME/SKILL.md" -o "$TMP/SKILL.md"
  curl -fsSL "$RAW_BASE/$SKILL_NAME/REFERENCE.md" -o "$TMP/REFERENCE.md"
  SRC="$TMP/SKILL.md"
  echo "source: published copy (no local clone found)"
else
  echo "source: $SRC"
fi
SRC_V="$(version_of "$SRC")"

# Where each agent CLI keeps user-scope skills. Only these are handled; anything else is left
# alone rather than guessed at.
declare -a TARGETS=()
add() { TARGETS+=("$1"); }
if [ "$SCOPE" = "local" ]; then
  add "$PWD/.claude/skills"
else
  { [ -d "$HOME/.claude" ] || [ "$FORCE_ALL" = 1 ]; } && add "$HOME/.claude/skills"
  { [ -d "$HOME/.codex" ]  || [ "$FORCE_ALL" = 1 ]; } && add "$HOME/.codex/skills"
  { [ -d "$HOME/.cursor" ] || [ "$FORCE_ALL" = 1 ]; } && add "$HOME/.cursor/skills"
fi

if [ "${#TARGETS[@]}" -eq 0 ]; then
  echo "no agent CLI directory found. Use --all to install anyway, or --local for this project." >&2
  exit 1
fi

if [ "$MODE" = "check" ]; then
  echo "repository version: ${SRC_V:-unknown}"
  if command -v curl >/dev/null; then
    pub="$(curl -fsSL "$RAW_BASE/$SKILL_NAME/SKILL.md" 2>/dev/null \
           | sed -n 's/^version: *//p' | head -1 || true)"
    echo "published version:  ${pub:-unreachable}"
  fi
  for t in "${TARGETS[@]}"; do
    f="$t/$SKILL_NAME/SKILL.md"
    if [ -f "$f" ]; then
      iv="$(version_of "$f")"
      if [ "$iv" = "$SRC_V" ]; then echo "installed at $f: ${iv:-unknown} (current)"
      else echo "installed at $f: ${iv:-unknown} (source is ${SRC_V:-unknown} -- run install.sh)"; fi
    else
      echo "installed at $t: not installed"
    fi
  done
  exit 0
fi

for t in "${TARGETS[@]}"; do
  dest="$t/$SKILL_NAME"
  mkdir -p "$dest"
  # SKILL.md is what the agent loads; REFERENCE.md is what it reads on demand, so both travel.
  ref="$(dirname "$SRC")/REFERENCE.md"
  if [ -f "$dest/SKILL.md" ] && cmp -s "$SRC" "$dest/SKILL.md" \
     && { [ ! -f "$ref" ] || cmp -s "$ref" "$dest/REFERENCE.md"; }; then
    echo "unchanged  $dest (version ${SRC_V:-unknown})"
  else
    prev="$(version_of "$dest/SKILL.md")"
    cp "$SRC" "$dest/SKILL.md"
    [ -f "$ref" ] && cp "$ref" "$dest/REFERENCE.md"
    if [ -n "$prev" ]; then echo "updated    $dest ($prev -> ${SRC_V:-unknown})"
    else echo "installed  $dest (version ${SRC_V:-unknown})"; fi
  fi
done

echo
echo "The skill is available as $SKILL_NAME. It documents how to check for a newer version."

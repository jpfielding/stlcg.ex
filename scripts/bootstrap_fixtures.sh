#!/usr/bin/env bash
# Bootstrap the Python oracle environment and regenerate fixtures from
# the pinned upstream StanfordASL/stlcg commit.
#
# Outputs: fixtures/*.json (deterministic; commit the results).
# Idempotent: safe to re-run.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
CACHE="$ROOT/.cache/stlcg"
PINNED_SHA="abd16c92108f1b57a72d66c58492c949b6c5a8ea"

say() { printf "\033[1;36m==>\033[0m %s\n" "$*"; }

# 1. venv ---------------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
    say "Creating Python venv at $VENV (using $(python3 --version))"
    python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 2. Python deps --------------------------------------------------------
say "Installing pinned Python deps"
pip install --quiet --upgrade pip
pip install --quiet -r "$ROOT/fixtures/requirements.txt"

# 3. Clone upstream at pinned SHA --------------------------------------
if [[ ! -d "$CACHE/.git" ]]; then
    say "Cloning upstream stlcg into $CACHE"
    mkdir -p "$(dirname "$CACHE")"
    git clone --quiet https://github.com/StanfordASL/stlcg "$CACHE"
fi

say "Checking out pinned SHA $PINNED_SHA"
git -C "$CACHE" fetch --quiet origin "$PINNED_SHA" 2>/dev/null || true
git -C "$CACHE" checkout --quiet "$PINNED_SHA"

# 4. Smoke import ------------------------------------------------------
say "Smoke import"
PYTHONPATH="$CACHE/src" python -c "import stlcg; stlcg.LessThan(lhs='x', val=1.0); print('  ok: stlcg module imports and constructs LessThan')"

# 5. Generate fixtures -------------------------------------------------
say "Generating fixtures"
PYTHONPATH="$CACHE/src" python "$ROOT/fixtures/gen_fixtures.py" --out "$ROOT/fixtures"

say "Done. Generated fixtures in $ROOT/fixtures/*.json"

#!/usr/bin/env bash
# Fresh rented GPU box -> working studio, in one line:
#
#   curl -fsSL https://raw.githubusercontent.com/adhyanmahajan26-boop/ultron-engine/main/linux/bootstrap.sh | HF_TOKEN=hf_xxx bash
#
# Clones (or updates) the engine into $ULTRON_HOME/engine and runs linux/setup.sh with any arguments you pass:
#   curl -fsSL .../bootstrap.sh | HF_TOKEN=hf_xxx bash -s -- --components ltx,asr,latentsync
# Env: ULTRON_HOME (default /workspace/ultron if /workspace exists, else ~/ultron-work), ULTRON_BRANCH (default main),
#      ULTRON_REPO (default this repo). Everything else: linux/config.example.env.
set -euo pipefail
REPO_URL="${ULTRON_REPO:-https://github.com/adhyanmahajan26-boop/ultron-engine.git}"
BRANCH="${ULTRON_BRANCH:-main}"
if [ -z "${ULTRON_HOME:-}" ]; then
  if [ -d /workspace ] && [ -w /workspace ]; then ULTRON_HOME=/workspace/ultron; else ULTRON_HOME="$HOME/ultron-work"; fi
fi
export ULTRON_HOME
mkdir -p "$ULTRON_HOME"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
if ! command -v git >/dev/null; then
  $SUDO apt-get update -qq && $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git curl ca-certificates
fi
DIR="$ULTRON_HOME/engine"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch -q origin "$BRANCH" && git -C "$DIR" checkout -q "$BRANCH" && git -C "$DIR" pull -q --ff-only origin "$BRANCH" \
    || echo "note: could not fast-forward $DIR (local changes?); using it as is"
else
  git clone -q --branch "$BRANCH" "$REPO_URL" "$DIR"
fi
ln -sf "$DIR/linux/ultron-linux" /usr/local/bin/ultron-linux 2>/dev/null || $SUDO ln -sf "$DIR/linux/ultron-linux" /usr/local/bin/ultron-linux 2>/dev/null || true
exec bash "$DIR/linux/setup.sh" "$@"

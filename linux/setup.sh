#!/usr/bin/env bash
# Ultron Engine on a rented Linux + NVIDIA GPU: ONE command installs everything, downloads the weights, starts the
# model server and the Claude connector, and prints the connector URL. Safe to re-run: finished steps are skipped,
# so on a box with a persistent volume the daily start takes a minute instead of an hour.
#
#   HF_TOKEN=hf_xxx bash linux/setup.sh                 # profile "voice" (default): lip-sync + voice + ASR, ~40 GB
#   HF_TOKEN=hf_xxx bash linux/setup.sh --profile full  # + LTX-2.5 video, Qwen-Image, ACE-Step music, HyperFrames
#   bash linux/setup.sh --components latentsync,asr      # exactly these components
#   bash linux/setup.sh --no-start                       # install only
#   bash linux/setup.sh --dry-run                        # print the plan, change nothing
#
# Settings come from the environment or from $ULTRON_HOME/config.env (see linux/config.example.env).
# Docs: docs/LINUX.md
set -Eeuo pipefail

ENGINE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINUX="$ENGINE/linux"
ALL_COMPONENTS="ltx,qwen_image,qwen_image_turbo,music,fish,chatterbox,qwen3tts,asr,latentsync,hyperframes"
# profiles: voice (default, day one: lip-sync + audio, ~40 GB of weights) | full (adds video, images, music, ~210 GB)
PROFILE_VOICE="latentsync,fish,chatterbox,asr,qwen3tts"
PROFILE_FULL="ltx,qwen_image,music,fish,chatterbox,qwen3tts,asr,latentsync,hyperframes"

# ── settings ────────────────────────────────────────────────────────────────
DRY=0; START=1; COMPONENTS=""; WARM=""; PROFILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --components) COMPONENTS="$2"; shift 2;;
    --profile) PROFILE="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --no-start) START=0; shift;;
    --warm) WARM="$2"; shift 2;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) echo "unknown option $1"; exit 2;;
  esac
done

if [ -z "${ULTRON_HOME:-}" ]; then
  # RunPod / Vast / Lambda persistent volumes usually mount at /workspace: keep weights there so they survive
  if [ -d /workspace ] && [ -w /workspace ]; then ULTRON_HOME=/workspace/ultron; else ULTRON_HOME="$HOME/ultron-work"; fi
fi
export ULTRON_HOME
mkdir -p "$ULTRON_HOME"/{logs,state,secrets,run,venvs,inputs,outputs,cache}
[ -f "$ULTRON_HOME/config.env" ] && set -a && . "$ULTRON_HOME/config.env" && set +a
PROFILE="${PROFILE:-${ULTRON_PROFILE:-voice}}"
case "$PROFILE" in voice) PCOMP="$PROFILE_VOICE";; full) PCOMP="$PROFILE_FULL";; *) echo "unknown profile '$PROFILE' (voice|full)"; exit 2;; esac
COMPONENTS="${COMPONENTS:-${ULTRON_COMPONENTS:-$PCOMP}}"
for c in ${COMPONENTS//,/ }; do [[ ",$ALL_COMPONENTS," == *",$c,"* ]] || { echo "unknown component '$c' (known: $ALL_COMPONENTS)"; exit 2; }; done
LOG="$ULTRON_HOME/logs/setup-$(date +%Y%m%d-%H%M%S).log"
export HF_HOME="${HF_HOME:-$ULTRON_HOME/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ULTRON_HOME/cache/uv}"
export UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR="$ULTRON_HOME/cache/python"
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$ULTRON_HOME/bin:$PATH"
AVATAR="$ULTRON_HOME/avatar"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

has() { [[ ",$COMPONENTS," == *",$1,"* ]]; }
say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
warn() { printf '\033[1;33m   WARNING: %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*"; exit 1; }
run() { if [ $DRY = 1 ]; then echo "   [dry-run] $*"; else "$@"; fi; }
in_dir() { local d=$1; shift; if [ $DRY = 1 ]; then echo "   [dry-run] (in $d) $*"; else (cd "$d" && "$@"); fi; }
# step NAME KEY: true if the step still has to run (KEY changes when its inputs change, e.g. a requirements file)
step_needed() { [ "$(cat "$ULTRON_HOME/state/$1" 2>/dev/null)" != "$2" ]; }
step_done() { [ $DRY = 1 ] || echo "$2" > "$ULTRON_HOME/state/$1"; }
trap 'echo; echo "setup failed at line $LINENO (log: $LOG)"; exit 1' ERR
[ $DRY = 1 ] || exec > >(tee -a "$LOG") 2>&1

say "Ultron Engine Linux setup   home=$ULTRON_HOME   profile=$PROFILE   components=$COMPONENTS"

# ── 0. machine checks ───────────────────────────────────────────────────────
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this needs an NVIDIA GPU with the driver installed (rent a CUDA image)."
GPU="$(nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader | head -1)"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
RAM_GB=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
FREE_GB=$(df -BG --output=avail "$ULTRON_HOME" | tail -1 | tr -dc 0-9)
note "GPU: $GPU   RAM: ${RAM_GB} GB   free disk at $ULTRON_HOME: ${FREE_GB} GB"
NEED=40  # venvs + tools
has ltx && NEED=$((NEED+75)); has qwen_image && NEED=$((NEED+35)); has qwen_image_turbo && NEED=$((NEED+35))
has music && NEED=$((NEED+30)); has fish && NEED=$((NEED+12)); has chatterbox && NEED=$((NEED+3))
has qwen3tts && NEED=$((NEED+5)); has asr && NEED=$((NEED+2)); has latentsync && NEED=$((NEED+8))
note "approximate disk needed for these components on a fresh box: ${NEED} GB (less when weights are already cached)"
[ "${FREE_GB:-0}" -lt 20 ] && die "less than 20 GB free at $ULTRON_HOME"
[ "${FREE_GB:-0}" -lt "$NEED" ] && warn "free disk (${FREE_GB} GB) may be too small for a first download of ${NEED} GB"
[ "$VRAM_MB" -lt 23000 ] && warn "GPU has $((VRAM_MB/1024)) GB VRAM; LTX-2.5 and Fish S2 Pro need about 24 GB+ (they will offload and be slow)"
[ "$RAM_GB" -lt 60 ] && has ltx && warn "LTX-2.5 on a 32 GB GPU streams weights from system RAM; ${RAM_GB} GB RAM may not be enough (64 GB+ recommended)"
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HF_HOME/token" ]; then
  has fish && warn "HF_TOKEN not set: if fishaudio/s2-pro asks you to accept its licence on huggingface.co, the download needs a token"
  has ltx && warn "HF_TOKEN not set: Lightricks/LTX-2.5-Diffusers is gated (accept the licence on huggingface.co, then set HF_TOKEN)"
fi
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN

# ── 1. system packages ─────────────────────────────────────────────────────
APT_PKGS="git git-lfs curl ca-certificates jq unzip rsync aria2 build-essential pkg-config python3 python3-venv ffmpeg \
sox libsox-dev libsndfile1 portaudio19-dev libgl1 libglib2.0-0 fonts-noto-core fonts-noto-color-emoji psmisc procps"
# headless Chrome for HyperFrames renders
APT_CHROME="libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libxshmfence1"
KEY="$(echo "$APT_PKGS $APT_CHROME" | sha1sum | cut -c1-12)"
if step_needed apt "$KEY"; then
  say "1/8 system packages"
  if command -v apt-get >/dev/null; then
    run $SUDO apt-get update -qq
    run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $APT_PKGS
    run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $APT_CHROME || true
    run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libasound2t64 2>/dev/null || run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libasound2 || true
  else
    warn "no apt-get: install git, ffmpeg, sox, libsndfile, portaudio and build tools yourself"
  fi
  step_done apt "$KEY"
else note "1/8 system packages: already installed"; fi

if ! command -v uv >/dev/null; then
  say "installing uv"
  run sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
fi
if has hyperframes && ! command -v node >/dev/null; then
  say "installing Node.js 22 (HyperFrames)"
  run sh -c "curl -fsSL https://deb.nodesource.com/setup_22.x | $SUDO bash - && $SUDO apt-get install -y -qq nodejs"
fi
if ! command -v cloudflared >/dev/null; then
  say "installing cloudflared (HTTPS tunnel for the connector)"
  ARCH=$(dpkg --print-architecture 2>/dev/null || echo amd64)
  run curl -fsSL -o /tmp/cloudflared.deb "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$ARCH.deb"
  run $SUDO dpkg -i /tmp/cloudflared.deb
fi

# ── 2. optional: pull cached weights from your own bucket (fast daily restore) ──
if [ -n "${ULTRON_WEIGHTS_MIRROR:-}" ]; then
  say "2/8 restoring weights from mirror $ULTRON_WEIGHTS_MIRROR"
  command -v rclone >/dev/null || run sh -c "curl -fsSL https://rclone.org/install.sh | $SUDO bash"
  run rclone copy "$ULTRON_WEIGHTS_MIRROR/hf" "$HF_HOME" --transfers 32 --checkers 32 --multi-thread-streams 8 --fast-list -P || warn "mirror restore failed; falling back to Hugging Face"
  for d in fish-speech/checkpoints acestep/ACE-Step-1.5/checkpoints latentsync/checkpoints; do
    run rclone copy "$ULTRON_WEIGHTS_MIRROR/$d" "$ULTRON_HOME/$d" --transfers 16 -P 2>/dev/null || true
  done
else note "2/8 no ULTRON_WEIGHTS_MIRROR set: weights come from Hugging Face (cached in $HF_HOME)"; fi

# ── 3. Python environments (uv; torch build chosen for this driver by --torch-backend=auto) ──
mkvenv() {  # mkvenv NAME PYVER REQFILE [extra uv pip args...]
  local name=$1 pyver=$2 req=$3; shift 3
  local key f; key="$( (cat "$req"; for f in "$@"; do [ -f "$f" ] && cat "$f"; done; echo "$pyver $*") | sha1sum | cut -c1-12)"
  if step_needed "venv-$name" "$key" || [ ! -x "$ULTRON_HOME/venvs/$name/bin/python" ]; then
    note "venv $name (python $pyver) <- $(basename "$req")"
    run uv venv -q --allow-existing --python "$pyver" "$ULTRON_HOME/venvs/$name"
    run uv pip install -q --python "$ULTRON_HOME/venvs/$name/bin/python" -r "$req" "$@"
    step_done "venv-$name" "$key"
  else note "venv $name: up to date"; fi
}
say "3/8 Python environments"
mkvenv avatar 3.12 "$LINUX/requirements/avatar.txt"
if has ltx || has qwen_image || has qwen_image_turbo; then mkvenv gen 3.12 "$LINUX/requirements/gen.txt"; fi
has qwen3tts && mkvenv qwentts 3.12 "$LINUX/requirements/qwentts.txt"
has chatterbox && mkvenv chatterbox 3.11 "$LINUX/requirements/chatterbox.txt" --override "$LINUX/requirements/chatterbox-overrides.txt"

clone_at() {  # clone_at URL DIR [REF]
  local url=$1 dir=$2 ref=${3:-}
  if [ ! -d "$dir/.git" ]; then run git clone -q "$url" "$dir"; fi
  if [ -n "$ref" ]; then run git -C "$dir" fetch -q origin "$ref" 2>/dev/null || true; run git -C "$dir" checkout -q "$ref"; fi
}
if has fish; then
  FISH_DIR="$ULTRON_HOME/fish-speech"
  clone_at https://github.com/fishaudio/fish-speech.git "$FISH_DIR" "${ULTRON_FISH_REF:-}"
  KEY="$(git -C "$FISH_DIR" rev-parse HEAD 2>/dev/null || echo new)"
  if step_needed venv-fish "$KEY" || [ ! -x "$FISH_DIR/.venv/bin/python" ]; then
    note "fish-speech venv (uv sync --extra cu128/cu129)"
    # fish-speech ships cu126 / cu128 / cu129 extras; Blackwell (RTX 50xx) needs 12.8+
    DRV_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
    CU=cu128; awk "BEGIN{exit !(${DRV_CUDA:-0} >= 12.9)}" && CU=cu129
    in_dir "$FISH_DIR" uv sync -q --python 3.12 --extra "$CU"
    step_done venv-fish "$KEY"
  else note "fish-speech venv: up to date"; fi
fi
if has music; then
  ACE_DIR="$ULTRON_HOME/acestep/ACE-Step-1.5"
  mkdir -p "$ULTRON_HOME/acestep"
  clone_at https://github.com/ace-step/ACE-Step-1.5.git "$ACE_DIR" "${ULTRON_ACESTEP_REF:-}"
  KEY="$(git -C "$ACE_DIR" rev-parse HEAD 2>/dev/null || echo new)"
  if step_needed venv-acestep "$KEY" || [ ! -x "$ACE_DIR/.venv/bin/python" ]; then
    note "ACE-Step venv (uv sync)"
    in_dir "$ACE_DIR" uv sync -q
    step_done venv-acestep "$KEY"
  else note "ACE-Step venv: up to date"; fi
fi
if has latentsync; then
  LS_DIR="$ULTRON_HOME/latentsync"
  LS_REF="${ULTRON_LATENTSYNC_REF:-a229c3948406bc2cf6eaf4873e662e70c6a04746}"
  if [ -x "$ENGINE/scripts/setup_latentsync_official.sh" ]; then
    # owned by the lip-sync work (src/lipsync/): checkout + venv + weights, idempotent
    note "LatentSync via scripts/setup_latentsync_official.sh"
    run env ULTRON_LATENTSYNC_DIR="$LS_DIR" ULTRON_WORK="$ULTRON_HOME" "$ENGINE/scripts/setup_latentsync_official.sh" --device cuda --versions "${ULTRON_LATENTSYNC_VERSIONS:-1.6}" --yes-download
  else
    clone_at https://github.com/bytedance/LatentSync.git "$LS_DIR" "$LS_REF"
    if step_needed venv-latentsync "$LS_REF" || [ ! -x "$LS_DIR/.venv/bin/python" ]; then
      note "LatentSync venv (upstream requirements, torch overridden for this GPU)"
      printf 'torch>=2.7\ntorchvision>=0.22\ntorchaudio>=2.7\n' > "$ULTRON_HOME/state/latentsync-overrides.txt"
      run uv venv -q --allow-existing --python 3.10 "$LS_DIR/.venv"
      # upstream pins torch 2.5.1+cu121 (no Blackwell kernels) and an extra index; let uv choose the CUDA build
      [ $DRY = 1 ] || grep -v -e '^--extra-index-url' -e '^torch' -e '^xformers' "$LS_DIR/requirements.txt" > "$ULTRON_HOME/state/latentsync-req.txt"
      run uv pip install -q --python "$LS_DIR/.venv/bin/python" torch torchvision torchaudio -r "$ULTRON_HOME/state/latentsync-req.txt" --override "$ULTRON_HOME/state/latentsync-overrides.txt"
      step_done venv-latentsync "$LS_REF"
    else note "LatentSync venv: up to date"; fi
  fi
fi
if has hyperframes && command -v npx >/dev/null; then
  if step_needed hyperframes v1; then
    note "prefetching the HyperFrames CLI"
    run npx -y hyperframes --version >/dev/null 2>&1 || warn "could not prefetch hyperframes; it is fetched on first render"
    step_done hyperframes v1
  fi
fi

# ── 4. model weights (Hugging Face cache; resumable; skipped when complete) ──
say "4/8 model weights"
PYA="$ULTRON_HOME/venvs/avatar/bin/python"
hfget() {  # hfget STATEKEY REPO [--local-dir D] [--include ...] [--exclude ...]
  local key=$1; shift
  if step_needed "w-$key" "$*"; then
    note "downloading $1"
    run "$ULTRON_HOME/venvs/avatar/bin/hf" download "$@" >/dev/null
    step_done "w-$key" "$*"
  else note "weights $1: present"; fi
}
has ltx && hfget ltx Lightricks/LTX-2.5-Diffusers --exclude "transformer_full/*" --exclude "diffusion_decoder/*"
has qwen_image && hfget qwen_image Qwen/Qwen-Image-2.1
has qwen_image_turbo && hfget qwen_image_turbo Qwen/Qwen-Image-2.1-Turbo
has qwen3tts && hfget qwen3tts Qwen/Qwen3-TTS-12Hz-1.7B-Base
has chatterbox && hfget chatterbox ResembleAI/chatterbox
has fish && hfget fish fishaudio/s2-pro --local-dir "$ULTRON_HOME/fish-speech/checkpoints/s2-pro"
if has music; then
  hfget ace_base ACE-Step/Ace-Step1.5 --local-dir "$ACE_DIR/checkpoints"
  hfget ace_xl ACE-Step/acestep-v15-xl-turbo --local-dir "$ACE_DIR/checkpoints/acestep-v15-xl-turbo"
fi
if has asr; then
  mkdir -p "$AVATAR/models"
  hfget asr mobiuslabsgmbh/faster-whisper-large-v3-turbo --local-dir "$AVATAR/models/whisper-turbo-ct2"
fi
if has latentsync && [ ! -x "$ENGINE/scripts/setup_latentsync_official.sh" ]; then
  hfget latentsync ByteDance/LatentSync-1.6 --local-dir "$ULTRON_HOME/latentsync/checkpoints"
fi
if [ -n "${ULTRON_WEIGHTS_MIRROR:-}" ] && [ "${ULTRON_WEIGHTS_MIRROR_PUSH:-0}" = 1 ]; then
  note "pushing weights to mirror (first day only; set ULTRON_WEIGHTS_MIRROR_PUSH=0 afterwards)"
  run rclone copy "$HF_HOME" "$ULTRON_WEIGHTS_MIRROR/hf" --transfers 32 --exclude "token" -P
  for d in fish-speech/checkpoints acestep/ACE-Step-1.5/checkpoints latentsync/checkpoints; do
    [ -d "$ULTRON_HOME/$d" ] && run rclone copy "$ULTRON_HOME/$d" "$ULTRON_WEIGHTS_MIRROR/$d" --transfers 16 -P
  done
fi

# ── 5. avatar folder: same layout the Mac studio uses (scripts/, models/, inputs/) ──
say "5/8 studio layout"
run mkdir -p "$AVATAR/scripts" "$AVATAR/inputs" "$AVATAR/models"
run ln -sfn "$ULTRON_HOME/venvs/avatar" "$AVATAR/.venv"
for s in asr.py voice_clone.py voiceconvert.py; do run ln -sf "$LINUX/avatar/scripts/$s" "$AVATAR/scripts/$s"; done
if [ -f "$ENGINE/src/lipsync/avatar_shim.py" ]; then
  run ln -sf "$ENGINE/src/lipsync/avatar_shim.py" "$AVATAR/scripts/lipsync.py"
else
  run ln -sf "$LINUX/avatar/scripts/lipsync_fallback.py" "$AVATAR/scripts/lipsync.py"
fi
# personal assets (voice reference, talking-head footage) from your bucket, if configured
if [ -n "${ULTRON_ASSETS_REMOTE:-}" ]; then
  command -v rclone >/dev/null && run rclone copy "$ULTRON_ASSETS_REMOTE" "$AVATAR/inputs" -P || warn "assets sync failed"
fi
[ -f "$AVATAR/inputs/my_voice_ref.wav" ] || note "no voice reference yet: upload one as $AVATAR/inputs/my_voice_ref.wav (+ my_voice_ref.txt transcript)"

# presets for this machine: everything the Mac does with H3 goes to LTX-2.5 here
PRESETS="$ULTRON_HOME/presets.json"
TURBO=qwen-image-2.1-turbo; has qwen_image_turbo || TURBO=qwen-image-2.1
cat > "$PRESETS" <<EOF
{"video": {"fast": "ltx-2.5", "quality": "ltx-2.5", "ltx": "ltx-2.5"},
 "image": {"quality": "qwen-image-2.1", "fast": "$TURBO"},
 "edit": "qwen-image-2.1", "music": "ace-step-1.5-xl-turbo", "tts": "qwen3-tts-1.7b"}
EOF

# connector secret: kept on the (persistent) volume so the connector URL does not change between days
TOKF="$ULTRON_HOME/secrets/mcp_token"
if [ -n "${ULTRON_MCP_TOKEN:-}" ]; then echo "$ULTRON_MCP_TOKEN" > "$TOKF"; fi
[ -s "$TOKF" ] || (umask 077; python3 -c "import secrets; print(secrets.token_urlsafe(30))" > "$TOKF")
chmod 600 "$TOKF"

# environment file every service and shell sources
TORCHLIB="$( (ls -d "$ULTRON_HOME"/venvs/avatar/lib/python3*/site-packages/nvidia/*/lib 2>/dev/null || true) | tr '\n' ':')"
cat > "$ULTRON_HOME/ultron-linux.env" <<EOF
# written by linux/setup.sh
export ULTRON_HOME='$ULTRON_HOME'
export ULTRON_ENGINE_DIR='$ENGINE'
export ULTRON_AVATAR='$AVATAR'
export ULTRON_ROOT='$ENGINE/workspace'
export HF_HOME='$HF_HOME'
export HF_XET_HIGH_PERFORMANCE=1
export CONTENT_STUDIO_OUT='$ULTRON_HOME/outputs'
export CONTENT_STUDIO_PRESETS='$PRESETS'
export CONTENT_STUDIO_NODE_BIN='$(dirname "$(command -v node 2>/dev/null || echo /usr/bin/node)")'
export ULTRON_ASR_SCRIPT='$LINUX/avatar/scripts/asr.py'
export ULTRON_ASR_MODEL='$AVATAR/models/whisper-turbo-ct2'
export ULTRON_FISH_DIR='$ULTRON_HOME/fish-speech'
export ULTRON_CHATTERBOX_PY='$ULTRON_HOME/venvs/chatterbox/bin/python'
export ULTRON_LATENTSYNC_DIR='$ULTRON_HOME/latentsync'
export ULTRON_LATENTSYNC_PY='$ULTRON_HOME/latentsync/.venv/bin/python'
export ULTRON_LIPSYNC_BACKEND=official
export MLX_SERVE_URL='http://127.0.0.1:${CUDA_SERVE_PORT:-11234}'
export CUDA_SERVE_PORT='${CUDA_SERVE_PORT:-11234}'
export ULTRON_MCP_PORT='${ULTRON_MCP_PORT:-8765}'
export LD_LIBRARY_PATH="${TORCHLIB}\${LD_LIBRARY_PATH:-}"
export PATH="$ENGINE/runtime/tools:$LINUX:\$PATH"
EOF
[ -n "${HF_TOKEN:-}" ] && [ $DRY = 0 ] && "$PYA" -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)" >/dev/null 2>&1 || true

# ── 6. render the engine's runtime/ (tools, pipelines) for this machine ─────
say "6/8 engine runtime"
if [ ! -f "$ENGINE/ultron.config.json" ] && [ $DRY = 0 ]; then
  cat > "$ENGINE/ultron.config.json" <<EOF
{"mode": "linux-cuda", "work_dir": "$ULTRON_HOME", "components": {}, "cloud": {}}
EOF
fi
run env ULTRON_AVATAR="$AVATAR" python3 "$ENGINE/ultron.py" render

# ── 7. start services ──────────────────────────────────────────────────────
if [ $START = 1 ] && [ $DRY = 0 ]; then
  say "7/8 starting cuda-serve, the studio connector and the HTTPS tunnel"
  "$LINUX/ultron-linux" up ${WARM:+--warm "$WARM"}
  say "8/8 health check"
  "$LINUX/ultron-linux" doctor || warn "doctor found problems (see above)"
  "$LINUX/ultron-linux" url
else
  note "services not started (--no-start / --dry-run). Start later with: $LINUX/ultron-linux up"
fi
say "done in ${SECONDS}s   log: $LOG"

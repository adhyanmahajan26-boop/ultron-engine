#!/usr/bin/env bash
# Install the OFFICIAL ByteDance LatentSync (PyTorch) for src/lipsync, idempotently.
#
#   scripts/setup_latentsync_official.sh [--device cuda|mps|cpu] [--versions 1.6[,1.5]] [--yes-download] [--dir PATH] [--skip-venv]
#
#   * clones github.com/bytedance/LatentSync at the tested commit into $ULTRON_LATENTSYNC_DIR
#     (default $ULTRON_HOME/latentsync, ULTRON_HOME default ~/ultron-work)
#   * makes its own venv (.venv) with upstream's pinned stack; on Apple Silicon swaps the two
#     CUDA-only wheels (decord -> eva-decord, onnxruntime-gpu -> onnxruntime) and uses MPS torch;
#     on Blackwell GPUs (compute capability >= 12, e.g. RTX 5090) upstream's torch 2.5.1+cu121 has
#     no kernels, so torch>=2.7 from a CUDA 12.8 build replaces it (--skip-venv: venv is made elsewhere)
#   * weights are downloaded ONLY with --yes-download (sizes are printed first):
#       1.6 -> checkpoints/latentsync_unet.pt + checkpoints/whisper/tiny.pt   (upstream layout)
#       1.5 -> checkpoints/1.5/latentsync_unet.pt + checkpoints/1.5/whisper/tiny.pt
#     plus the SD VAE (stabilityai/sd-vae-ft-mse) into the Hugging Face cache
#   * on a Mac that already has the MLX port, reuses its insightface models (no re-download)
#   * writes $ULTRON_LATENTSYNC_DIR/ultron-latentsync.env with the exports src/lipsync reads
set -euo pipefail

COMMIT=a229c3948406bc2cf6eaf4873e662e70c6a04746
REPO_URL=https://github.com/bytedance/LatentSync.git
DEVICE=""; VERSIONS="1.6"; YES=0; SKIP_VENV=0
ULTRON_HOME="${ULTRON_HOME:-$HOME/ultron-work}"
LS_DIR="${ULTRON_LATENTSYNC_DIR:-$ULTRON_HOME/latentsync}"
ENGINE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --versions) VERSIONS="$2"; shift 2 ;;
    --yes-download) YES=1; shift ;;
    --dir) LS_DIR="$2"; shift 2 ;;
    --skip-venv) SKIP_VENV=1; shift ;;
    -h|--help) sed -n 2,20p "$0"; exit 0 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

if [ -z "$DEVICE" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then DEVICE=cuda
  elif [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then DEVICE=mps
  else DEVICE=cpu; fi
fi
echo "== LatentSync official setup: device=$DEVICE versions=$VERSIONS dir=$LS_DIR"

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg is required (apt install ffmpeg / brew install ffmpeg)"; exit 1; }

# ---------------------------------------------------------------- 1. upstream checkout at the tested commit
if [ ! -d "$LS_DIR/.git" ]; then
  mkdir -p "$(dirname "$LS_DIR")"
  GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "$REPO_URL" "$LS_DIR"
fi
if [ "$(git -C "$LS_DIR" rev-parse HEAD)" != "$COMMIT" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git -C "$LS_DIR" fetch --depth 1 origin "$COMMIT" 2>/dev/null || GIT_LFS_SKIP_SMUDGE=1 git -C "$LS_DIR" fetch origin
  GIT_LFS_SKIP_SMUDGE=1 git -C "$LS_DIR" checkout -q "$COMMIT"
fi
echo "   upstream at $(git -C "$LS_DIR" rev-parse --short HEAD)"

# ---------------------------------------------------------------- 2. venv (upstream wants python 3.10)
PY="${ULTRON_LATENTSYNC_PY:-$LS_DIR/.venv/bin/python}"
if [ "$SKIP_VENV" = 1 ]; then
  [ -x "$PY" ] || { echo "--skip-venv but no python at $PY (set ULTRON_LATENTSYNC_PY)"; exit 1; }
  echo "   using existing venv python $PY"
fi
if [ "$SKIP_VENV" = 0 ] && [ ! -x "$PY" ]; then
  if command -v uv >/dev/null 2>&1; then uv venv -p 3.10 "$LS_DIR/.venv"
  elif command -v python3.10 >/dev/null 2>&1; then python3.10 -m venv "$LS_DIR/.venv"
  else echo "   [warn] python3.10 not found, using $(python3 --version)"; python3 -m venv "$LS_DIR/.venv"; fi
fi
# uv wants the target after the subcommand: `uv pip install --python PY ...`
pip_install() {
  if command -v uv >/dev/null 2>&1; then uv pip install --python "$PY" "$@"
  else "$PY" -m pip install "$@"; fi
}
command -v uv >/dev/null 2>&1 || "$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip >/dev/null || true

CAP=0
if [ "$DEVICE" = cuda ]; then
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || echo 0)
fi
STAMP="$LS_DIR/.venv/.ultron-deps-$DEVICE-cc$CAP-$COMMIT"
if [ "$SKIP_VENV" = 0 ] && [ ! -f "$STAMP" ]; then
  REQ="$LS_DIR/.venv/requirements.ultron.txt"
  if [ "$DEVICE" = cuda ] && [ "${CAP:-0}" -ge 12 ]; then
    # Blackwell (sm_120): torch 2.5.1+cu121 has no kernels -> torch>=2.7 built for CUDA 12.8
    echo "   GPU compute capability $CAP: using torch>=2.7 (CUDA 12.8) instead of upstream 2.5.1+cu121"
    grep -v -E '^(--extra-index-url|torch==|torchvision==|torchaudio==|xformers)' "$LS_DIR/requirements.txt" > "$REQ"
    if command -v uv >/dev/null 2>&1; then
      uv pip install --python "$PY" --torch-backend=auto "torch>=2.7" "torchvision>=0.22" "torchaudio>=2.7"
    else
      "$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7" "torchvision>=0.22" "torchaudio>=2.7"
    fi
    printf 'torch>=2.7\ntorchvision>=0.22\ntorchaudio>=2.7\n' > "$LS_DIR/.venv/overrides.ultron.txt"
  elif [ "$DEVICE" = cuda ]; then
    cp "$LS_DIR/requirements.txt" "$REQ"
  else
    # Apple Silicon / CPU: same pins minus the CUDA-only wheels
    grep -v -E '^(--extra-index-url|decord==|onnxruntime-gpu==)' "$LS_DIR/requirements.txt" > "$REQ"
    echo "eva-decord==0.6.1" >> "$REQ"
    echo "onnxruntime==1.21.0" >> "$REQ"
  fi
  pip_install -q --upgrade pip wheel setuptools || true
  if [ -f "$LS_DIR/.venv/overrides.ultron.txt" ] && command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" -r "$REQ" --overrides "$LS_DIR/.venv/overrides.ultron.txt"
  elif [ -f "$LS_DIR/.venv/overrides.ultron.txt" ]; then
    "$PY" -m pip install -r "$REQ" -c "$LS_DIR/.venv/overrides.ultron.txt"
  else
    pip_install -r "$REQ"
  fi
  touch "$STAMP"
fi
# scipy 1.15.x wheels for py3.10 (pulled in by scikit-image/librosa) ship a _spropack dylib that newer
# macOS dyld refuses ("__DATA/__thread_bss has a zero-fill section type, but offset field is not zero"),
# which breaks `import diffusers`. 1.14.1 loads; checked outside the stamp so existing venvs get fixed too.
if [ "$SKIP_VENV" = 0 ] && [ "$(uname -s)" = Darwin ] && ! "$PY" -c "import scipy.sparse.linalg" >/dev/null 2>&1; then
  echo "   scipy $("$PY" -c 'import scipy;print(scipy.__version__)' 2>/dev/null) fails to load on this macOS; pinning scipy==1.14.1"
  pip_install "scipy==1.14.1"
fi
# librosa (used by the QA in src/lipsync/qa.py) imports pkg_resources, which setuptools 81+ removed.
if [ "$SKIP_VENV" = 0 ] && ! "$PY" -c "import pkg_resources" >/dev/null 2>&1; then
  echo "   setuptools has no pkg_resources (librosa needs it); pinning setuptools<81"
  pip_install "setuptools<81"
fi
"$PY" - <<'EOF'
import torch
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"   torch {torch.__version__} sees {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
EOF

# ---------------------------------------------------------------- 3. weights (consent required)
HFCLI="$(dirname "$PY")/huggingface-cli"
size_of() {  # repo file -> human size from the HF API (no download)
  "$PY" - "$1" "$2" <<'EOF' 2>/dev/null || echo "?"
import sys
from huggingface_hub import HfApi
info = HfApi().model_info(sys.argv[1], files_metadata=True)
s = sum((f.size or 0) for f in info.siblings if f.rfilename == sys.argv[2])
print(f"{s/1e9:.2f} GB")
EOF
}
vae_cached() {  # true if diffusers can load the VAE offline
  "$PY" -c "from huggingface_hub import try_to_load_from_cache as t; import sys; sys.exit(0 if isinstance(t('stabilityai/sd-vae-ft-mse','diffusion_pytorch_model.safetensors'), str) else 1)" 2>/dev/null
}
need=()
IFS=',' read -ra VLIST <<< "$VERSIONS"
for v in "${VLIST[@]}"; do
  if [ "$v" = 1.6 ]; then dst="$LS_DIR/checkpoints"; else dst="$LS_DIR/checkpoints/$v"; fi
  for f in latentsync_unet.pt whisper/tiny.pt; do
    if [ ! -s "$dst/$f" ]; then need+=("$v|$f|$dst"); fi
  done
done
if [ ${#need[@]} -gt 0 ]; then
  echo "   weights to download:"
  for n in "${need[@]}"; do IFS='|' read -r v f dst <<< "$n"; echo "     ByteDance/LatentSync-$v  $f  ($(size_of "ByteDance/LatentSync-$v" "$f"))  -> $dst/$f"; done
  if vae_cached; then echo "     stabilityai/sd-vae-ft-mse already in the Hugging Face cache"
  else echo "     stabilityai/sd-vae-ft-mse (~0.33 GB) -> Hugging Face cache"; fi
  if [ "$YES" = 1 ]; then
    for n in "${need[@]}"; do
      IFS='|' read -r v f dst <<< "$n"
      HF_HUB_DISABLE_XET=1 "$HFCLI" download "ByteDance/LatentSync-$v" "$f" --local-dir "$dst"
    done
    "$PY" -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')" >/dev/null
  else
    echo "   NOT downloading (re-run with --yes-download once approved)"
  fi
else
  echo "   weights present for: $VERSIONS"
fi

# ---------------------------------------------------------------- 4. insightface models: reuse the MLX port's copy on the Mac
AUX="$LS_DIR/checkpoints/auxiliary"
MLX_AUX="${ULTRON_AVATAR:-$ULTRON_HOME/avatar}/latentsync-mlx/checkpoints/auxiliary"
if [ ! -e "$AUX/models/buffalo_l" ] && [ -d "$MLX_AUX/models/buffalo_l" ]; then
  mkdir -p "$AUX/models"; ln -s "$MLX_AUX/models/buffalo_l" "$AUX/models/buffalo_l"
  echo "   linked insightface buffalo_l from the MLX port"
fi

# ---------------------------------------------------------------- 5. env file
ENVF="$LS_DIR/ultron-latentsync.env"
cat > "$ENVF" <<EOF
export ULTRON_ENGINE_DIR='$ENGINE_DIR'
export ULTRON_LATENTSYNC_DIR='$LS_DIR'
export ULTRON_LATENTSYNC_PY='$PY'
export PYTORCH_ENABLE_MPS_FALLBACK=1
EOF
if [ "$DEVICE" = cuda ]; then echo "export ULTRON_LIPSYNC_BACKEND=official" >> "$ENVF"; fi
echo "== done. source $ENVF"
echo "   smoke test: \"$PY\" \"$ENGINE_DIR/src/lipsync/official_runner.py\" --video $LS_DIR/assets/demo1_video.mp4 --audio $LS_DIR/assets/demo1_audio.wav --out /tmp/ls_demo.mp4"

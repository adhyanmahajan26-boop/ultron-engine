"""Path and device resolution shared by every lip-sync backend (standard library only).

Environment variables (all optional; the Linux installer and ultron.env set them):
  ULTRON_ENGINE_DIR       repo root (default: found relative to this file)
  ULTRON_HOME             work root on Linux boxes (default: ~/ultron-work)
  ULTRON_AVATAR           local-avatar folder holding the MLX port + its venv (default: $ULTRON_HOME/avatar)
  ULTRON_LATENTSYNC_DIR   checkout of bytedance/LatentSync (default: $ULTRON_HOME/latentsync)
  ULTRON_LATENTSYNC_PY    python of the venv that has LatentSync's pinned torch stack
                          (default: $ULTRON_LATENTSYNC_DIR/.venv/bin/python)
  ULTRON_LIPSYNC_BACKEND  auto | official | mlx | cloud   (default: auto)
"""
from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

VERSIONS = {
    # version: (unet config inside the upstream checkout, resolution, upstream default guidance)
    "1.6": ("configs/unet/stage2_512.yaml", 512, 1.5),
    "1.5": ("configs/unet/stage2.yaml", 256, 1.5),
}


def _p(env: str, default: Path) -> Path:
    v = os.environ.get(env)
    return Path(v).expanduser() if v else default


def engine_dir() -> Path:
    return _p("ULTRON_ENGINE_DIR", Path(__file__).resolve().parents[2])


def home() -> Path:
    return _p("ULTRON_HOME", Path.home() / "ultron-work")


def avatar_dir() -> Path:
    return _p("ULTRON_AVATAR", home() / "avatar")


def latentsync_dir() -> Path:
    return _p("ULTRON_LATENTSYNC_DIR", home() / "latentsync")


def latentsync_python() -> Path:
    return _p("ULTRON_LATENTSYNC_PY", latentsync_dir() / ".venv" / "bin" / "python")


def checkpoint(version: str, root: Path | None = None) -> Path:
    """UNet weights for a version. Accepts checkpoints/<ver>/latentsync_unet.pt or, for 1.6 only,
    the upstream default checkpoints/latentsync_unet.pt (what `huggingface-cli download
    ByteDance/LatentSync-1.6` produces)."""
    root = root or latentsync_dir()
    per_version = root / "checkpoints" / version / "latentsync_unet.pt"
    if per_version.exists():
        return per_version
    flat = root / "checkpoints" / "latentsync_unet.pt"
    if version == "1.6" and flat.exists():
        return flat
    return per_version  # caller reports it missing


def whisper_checkpoint(version: str, root: Path | None = None) -> Path:
    root = root or latentsync_dir()
    for c in (root / "checkpoints" / version / "whisper" / "tiny.pt", root / "checkpoints" / "whisper" / "tiny.pt"):
        if c.exists():
            return c
    return root / "checkpoints" / "whisper" / "tiny.pt"


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def default_backend() -> str:
    """auto: NVIDIA -> official (CUDA); Apple Silicon -> official (MPS) when its venv exists, else mlx."""
    b = os.environ.get("ULTRON_LIPSYNC_BACKEND", "auto")
    if b != "auto":
        return b
    if has_nvidia():
        return "official"
    if is_apple_silicon():
        return "official" if latentsync_python().exists() and checkpoint("1.6").exists() else "mlx"
    return "official"


def ffmpeg() -> str:
    for c in (shutil.which("ffmpeg"), str(avatar_dir() / "bin" / "ffmpeg"), str(Path.home() / ".local/bin/ffmpeg")):
        if c and Path(c).exists():
            return c
    return "ffmpeg"

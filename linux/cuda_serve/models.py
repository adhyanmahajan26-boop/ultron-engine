"""Model registry for cuda-serve: which worker script, which venv and which env vars serve each model id,
plus the Mac (mlx-serve) ids that alias to them so existing scripts, skills and the studio server work unchanged.

Paths come from env vars that linux/setup.sh writes to $ULTRON_HOME/ultron-linux.env.
"""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.environ.get("ULTRON_HOME", Path.home() / "ultron-work"))
VENVS = HOME / "venvs"
WORKERS = Path(__file__).resolve().parent / "workers"


def _py(venv: str) -> str:
    return str(VENVS / venv / "bin" / "python")


MODELS: dict[str, dict] = {
    "ltx-2.5": {
        "capabilities": ["video"], "worker": "ltx.py", "python": _py("gen"),
        "env": {"ULTRON_LTX_REPO": "Lightricks/LTX-2.5-Diffusers"}, "hf": ["Lightricks/LTX-2.5-Diffusers"],
        "aliases": ["Lightricks/LTX-2.5", "Lightricks/LTX-2.5-Diffusers", "ltx", "ltx-2.5-distilled"],
    },
    "qwen-image-2.1": {
        "capabilities": ["image"], "worker": "qwen_image.py", "python": _py("gen"),
        "env": {"ULTRON_QWEN_IMAGE_REPO": "Qwen/Qwen-Image-2.1"}, "hf": ["Qwen/Qwen-Image-2.1"],
        "aliases": ["Qwen/Qwen-Image-2.1", "ddalcu/Qwen-Image-2.1-MLX-Serve-8bit",
                    # Qwen-Image 2.1 edits with up to 10 references, so the Mac edit ids map here too
                    "mlx-community/qwen-image-edit-2511-8bit"],
    },
    "qwen-image-2.1-turbo": {
        "capabilities": ["image"], "worker": "qwen_image.py", "python": _py("gen"),
        "env": {"ULTRON_QWEN_IMAGE_REPO": "Qwen/Qwen-Image-2.1-Turbo"}, "hf": ["Qwen/Qwen-Image-2.1-Turbo"],
        "aliases": ["Qwen/Qwen-Image-2.1-Turbo", "ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit"],
        "fallback": "qwen-image-2.1",  # used when the turbo weights were not downloaded
    },
    "ace-step-1.5-xl-turbo": {
        "capabilities": ["music"], "worker": "acestep.py", "python": str(HOME / "acestep/ACE-Step-1.5/.venv/bin/python"),
        "env": {"ACESTEP_DIR": str(HOME / "acestep/ACE-Step-1.5"),
                "ACESTEP_CHECKPOINTS_DIR": str(HOME / "acestep/ACE-Step-1.5/checkpoints")},
        "local_dirs": [str(HOME / "acestep/ACE-Step-1.5/checkpoints/acestep-v15-xl-turbo")],
        "aliases": ["ACE-Step/acestep-v15-xl-turbo", "ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit"],
    },
    "qwen3-tts-1.7b": {
        "capabilities": ["audio"], "worker": "qwen3_tts.py", "python": _py("qwentts"),
        "env": {}, "hf": ["Qwen/Qwen3-TTS-12Hz-1.7B-Base"],
        "aliases": ["Qwen/Qwen3-TTS-12Hz-1.7B-Base", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"],
    },
}

# Mac-only or too big for one 32 GB GPU: answer with a clear error instead of a confusing 404.
UNSUPPORTED = {
    "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit":
        "MiniMax-H3 (33B transformer + 32B Qwen3-VL text encoder, ~62 GB + 48 GB in bf16) does not fit one 32 GB GPU. "
        "Use quality='ltx' (LTX-2.5) on this machine, or rent an 80 GB+ / multi-GPU box (see docs/LINUX.md).",
    "ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit": "Hunyuan3D is not part of the Linux setup yet.",
}


# Offline tests (linux/tests/test_offline.py): every model is served by a fake worker that writes tiny real files.
if os.environ.get("CUDA_SERVE_TEST_WORKER"):
    import sys as _sys
    for _m in MODELS.values():
        _m["worker"], _m["python"] = os.environ["CUDA_SERVE_TEST_WORKER"], _sys.executable


def resolve(model_id: str) -> str | None:
    if model_id in MODELS:
        return model_id
    for k, m in MODELS.items():
        if model_id in m["aliases"]:
            return k
    return None


def hf_cache_dir(repo: str) -> Path:
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    return hub / ("models--" + repo.replace("/", "--"))


def bytes_on_disk(key: str) -> int:
    m = MODELS[key]
    tot = 0
    dirs = [hf_cache_dir(r) / "blobs" for r in m.get("hf", [])] + [Path(d) for d in m.get("local_dirs", [])]
    for d in dirs:
        if d.exists():
            tot += sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and not f.is_symlink())
    return tot


def installed(key: str) -> bool:
    m = MODELS[key]
    return Path(m["python"]).exists() and bytes_on_disk(key) > 0

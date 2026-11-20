"""Adapter that exposes the unofficial MLX LatentSync port to parity.py's teacher forcing.

The port lives outside this repo ($ULTRON_AVATAR/latentsync-mlx, package `latentsync_mlx`), so its
API is wired here on the Mac. Run `python src/lipsync/mlx_adapter.py inspect` in the avatar venv to
print the port's modules/classes/signatures, then fill the four methods below.

Contract (numpy in, numpy out, OFFICIAL layouts; parity.py permutes NHWC automatically for compare):
  audio_chunks(wav_path, start, n) -> (n, 50, 384)   whisper-tiny features for frames start..start+n
  vae_encode(x)                   -> (f, 4, h/8, w/8) latent MEAN * 0.18215, x is (f, 3, H, W) in [-1, 1]
  unet(x, t, audio, dump=None)    -> same shape as x[:, :4]; x is (b, 13, f, h, w) = [noisy, mask, masked, ref]
                                     audio is (b*f, 50, 384); call dump(name, array) for conv_in,
                                     down_blocks.i, mid_block, up_blocks.i, conv_out if you can
  vae_decode(z)                   -> (f, 3, H, W) in [-1, 1], z is (1, 4, f, h, w) scaled latents
"""
from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import sys
from pathlib import Path

import numpy as np

PORT = Path(os.environ.get("ULTRON_LATENTSYNC_MLX", Path(os.environ.get("ULTRON_AVATAR", Path.home() / "ultron-work/avatar")) / "latentsync-mlx"))
sys.path.insert(0, str(PORT))


class Adapter:
    def __init__(self, version: str = os.environ.get("LS_VERSION", "1.6")):
        self.version = version
        # TODO(on the Mac): build the port's UNet / VAE / whisper exactly as scripts/lipsync.py does
        # for this version, e.g.  from latentsync_mlx.pipeline import ...; self.unet = ...
        raise NotImplementedError("wire Adapter to latentsync_mlx (run: python src/lipsync/mlx_adapter.py inspect)")

    def audio_chunks(self, wav_path: str, start: int, n: int) -> np.ndarray:
        raise NotImplementedError

    def vae_encode(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def unet(self, x: np.ndarray, t: float, audio: np.ndarray, dump=None) -> np.ndarray:
        raise NotImplementedError

    def vae_decode(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def inspect_port(pkg_name: str = "latentsync_mlx"):
    pkg = importlib.import_module(pkg_name)
    print(f"{pkg_name} at {Path(pkg.__file__).parent}")
    mods = [pkg_name] + [f"{pkg_name}.{m.name}" for m in pkgutil.walk_packages(pkg.__path__)]
    for mn in mods:
        try:
            m = importlib.import_module(mn)
        except Exception as e:  # noqa: BLE001
            print(f"  {mn}: import failed {e}")
            continue
        for name, obj in vars(m).items():
            if getattr(obj, "__module__", None) != mn or name.startswith("_"):
                continue
            if inspect.isclass(obj) or inspect.isfunction(obj):
                try:
                    sig = str(inspect.signature(obj))
                except (TypeError, ValueError):
                    sig = "(...)"
                print(f"  {mn}.{name}{sig}")
    for hint in ("ref_pv", "LS_REF_MODE", "load_weights", "strict", "attn2", "encoder_hidden_states", "randn", "guidance"):
        for f in Path(pkg.__file__).parent.rglob("*.py"):
            for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                if hint in line:
                    print(f"  [{hint}] {f.name}:{i}: {line.strip()[:120]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        inspect_port(sys.argv[2] if len(sys.argv) > 2 else "latentsync_mlx")
    else:
        print(__doc__)

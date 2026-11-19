#!/usr/bin/env python
"""Fallback lip-sync entry for Linux, used only until src/lipsync/avatar_shim.py (the lip-sync backend work) is in
this checkout; setup.sh links that shim to $ULTRON_AVATAR/scripts/lipsync.py when present.
Same flags the studio server passes on the Mac:  lipsync.py --video V --audio A --out O [--resolution R] [--steps N]
Runs official ByteDance LatentSync 1.6 (scripts/inference.py) in its own venv.
"""
import argparse
import os
import pathlib
import subprocess
import sys

HOME = pathlib.Path(os.environ.get("ULTRON_HOME", pathlib.Path.home() / "ultron-work"))
LS = pathlib.Path(os.environ.get("ULTRON_LATENTSYNC_DIR", HOME / "latentsync"))
PY = os.environ.get("ULTRON_LATENTSYNC_PY", str(LS / ".venv/bin/python"))

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--audio", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--resolution", type=int, default=512)  # 1.6 is trained at 512; the Mac 1.5 port used 256
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--guidance", type=float, default=1.5)
ap.add_argument("--seed", type=int, default=1247)
a = ap.parse_args()
cfg = "configs/unet/stage2_512.yaml" if a.resolution >= 512 else "configs/unet/stage2.yaml"
ckpt = LS / "checkpoints/latentsync_unet.pt"
if not ckpt.exists():
    sys.exit(f"LatentSync weights missing: {ckpt} (run linux/setup.sh with component latentsync)")
cmd = [PY, "-m", "scripts.inference", "--unet_config_path", cfg, "--inference_ckpt_path", str(ckpt),
       "--inference_steps", str(a.steps or 20), "--guidance_scale", str(a.guidance),
       "--video_path", str(pathlib.Path(a.video).resolve()), "--audio_path", str(pathlib.Path(a.audio).resolve()),
       "--video_out_path", str(pathlib.Path(a.out).resolve()), "--seed", str(a.seed), "--enable_deepcache"]
print("+", " ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd, cwd=LS))

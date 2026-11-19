"""ACE-Step 1.5 XL Turbo worker (music with vocals). Runs in the ACE-Step venv ($ULTRON_HOME/acestep/.venv), with
the ACE-Step-1.5 checkout as the working directory, using its documented Python API
(acestep.handler.AceStepHandler, acestep.llm_inference.LLMHandler, acestep.inference.generate_music).

env: ACESTEP_DIR (checkout), ACESTEP_CHECKPOINTS_DIR, ULTRON_ACE_DIT (default acestep-v15-xl-turbo),
     ULTRON_ACE_LM (default acestep-5Hz-lm-1.7B; 4B fits on 32 GB but is slower), ULTRON_ACE_LM_BACKEND (default pt)

params (same fields mlx-serve's /v1/audio/music-generations takes): prompt, lyrics, instrumental, duration_seconds,
        bpm, vocal_language, seed, out (wav path)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workerlib import serve  # noqa: E402

ACE_DIR = Path(os.environ.get("ACESTEP_DIR", Path.home() / "ultron-work/acestep/ACE-Step-1.5"))
CKPT = os.environ.get("ACESTEP_CHECKPOINTS_DIR", str(ACE_DIR / "checkpoints"))
DIT = os.environ.get("ULTRON_ACE_DIT", "acestep-v15-xl-turbo")
LM = os.environ.get("ULTRON_ACE_LM", "acestep-5Hz-lm-1.7B")
LM_BACKEND = os.environ.get("ULTRON_ACE_LM_BACKEND", "pt")


class Ace:
    def load(self):
        sys.path.insert(0, str(ACE_DIR))
        os.chdir(ACE_DIR)
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        t = time.time()
        self.dit = AceStepHandler()
        self.dit.initialize_service(project_root=str(ACE_DIR), config_path=DIT, device="cuda")
        self.llm = LLMHandler()
        self.llm.initialize(checkpoint_dir=CKPT, lm_model_path=LM, backend=LM_BACKEND, device="cuda")
        print(f"[ace] loaded {DIT} + {LM} ({LM_BACKEND}) in {time.time() - t:.0f}s", file=sys.stderr)

    def run(self, p, ctx):
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        instrumental = bool(p.get("instrumental")) or not p.get("lyrics")
        gp = dict(caption=p["prompt"], lyrics="[Instrumental]" if instrumental else p["lyrics"],
                  instrumental=instrumental, duration=float(max(10, min(600, p.get("duration_seconds", 60)))),
                  inference_steps=int(p.get("steps") or 8), shift=3.0)
        if p.get("bpm"):
            gp["bpm"] = int(p["bpm"])
        if p.get("vocal_language"):
            gp["vocal_language"] = p["vocal_language"]
        seed = p.get("seed")
        if seed is not None:
            gp["seed"] = int(seed)
        cfg = dict(batch_size=1, audio_format="wav", use_random_seed=seed is None)
        if seed is not None:
            cfg["seeds"] = [int(seed)]
        ctx.progress(message="generating music")
        t = time.time()
        with tempfile.TemporaryDirectory() as td:
            res = generate_music(self.dit, self.llm, GenerationParams(**gp), GenerationConfig(**cfg), save_dir=td)
            if not getattr(res, "success", False):
                raise RuntimeError(getattr(res, "error", None) or "ACE-Step generation failed")
            src = Path(res.audios[0]["path"] if isinstance(res.audios[0], dict) else res.audios[0].path)
            dst = Path(p["out"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix.lower() == ".wav":
                shutil.copy(src, dst)
            else:  # keep the engine's contract: a 48 kHz stereo WAV
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", "48000", "-ac", "2", str(dst)], check=True)
        return {"path": str(dst), "seconds": round(time.time() - t, 1), "model": DIT}


if __name__ == "__main__":
    serve(Ace())

"""Qwen-Image 2.1 worker (CUDA, diffusers QwenImage21Pipeline). Runs in the `gen` venv.

One model does both jobs: text-to-image (keyframes) and instruction edits with up to 10 reference images
(identity-keeping composites). ULTRON_QWEN_IMAGE_REPO picks the checkpoint:
  Qwen/Qwen-Image-2.1        quality (40 steps)   -> the engine's "quality" preset
  Qwen/Qwen-Image-2.1-Turbo  fast (8-step schedule saved in the repo) -> the engine's "fast" preset

params: prompt, negative_prompt, size "WxH", steps, seed, true_cfg_scale, images [paths] (edit mode), out (png path)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workerlib import offload_mode, seed_generator, serve, vram_gb  # noqa: E402

REPO = os.environ.get("ULTRON_QWEN_IMAGE_REPO", "Qwen/Qwen-Image-2.1")


def parse_size(s):
    if not s:
        return None, None
    w, _, h = str(s).lower().partition("x")
    return (int(w) // 16) * 16, (int(h) // 16) * 16


class QwenImage:
    def load(self):
        import torch
        from diffusers import QwenImage21Pipeline

        t = time.time()
        pipe = QwenImage21Pipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)
        mode = offload_mode()
        if mode == "none":
            pipe.to("cuda")
        elif mode == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()
        self.pipe, self.turbo = pipe, "turbo" in REPO.lower()
        print(f"[qwen-image] loaded {REPO} offload={mode} vram={vram_gb():.0f}GB in {time.time() - t:.0f}s", file=sys.stderr)

    def run(self, p, ctx):
        from PIL import Image

        w, h = parse_size(p.get("size"))
        images = [Image.open(x).convert("RGBA") for x in p.get("images") or []]
        steps = p.get("steps")
        kw = dict(prompt=p["prompt"], generator=seed_generator(p.get("seed")))
        if w and h:
            kw.update(width=w, height=h)
        if images:
            kw["image"] = images if len(images) > 1 else images[0]
        if steps:
            kw["num_inference_steps"] = int(steps)
        elif not self.turbo:
            kw["num_inference_steps"] = 40
        if p.get("negative_prompt"):
            kw["negative_prompt"] = p["negative_prompt"]
        if p.get("true_cfg_scale") is not None:
            kw["true_cfg_scale"] = float(p["true_cfg_scale"])
        total = kw.get("num_inference_steps") or 8

        def on_step(pipe, i, t, cb):
            ctx.progress(i + 1, total, f"step {i + 1}/{total}")
            return cb

        kw["callback_on_step_end"] = on_step
        t = time.time()
        img = self.pipe(**kw).images[0]
        dst = Path(p["out"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = img.convert("RGB") if img.mode == "RGBA" and not p.get("keep_alpha") else img
        img.save(dst)
        return {"path": str(dst), "size": f"{img.width}x{img.height}", "seconds": round(time.time() - t, 1), "model": REPO}


if __name__ == "__main__":
    serve(QwenImage())

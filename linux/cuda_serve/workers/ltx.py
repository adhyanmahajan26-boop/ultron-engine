"""LTX-2.5 worker (CUDA, diffusers). Runs in the `gen` venv.

Image-to-video with native audio from a first frame (and optional last frame), or text-to-video with no image.
The engine's Mac route used the distilled model at 1024x576, 24 fps; this keeps the same defaults.

32 GB GPUs (RTX 5090): the 22B transformer is stored in fp8 on the GPU and computed in bf16 (layerwise casting),
and the other components (Gemma 4 text encoder, VAEs) move to the GPU only while they run (model CPU offload).
That needs about 70 GB of system RAM. Override with ULTRON_OFFLOAD=none|model|group and ULTRON_LTX_FP8=0|1.

params: prompt, negative_prompt, width, height, num_frames (8k+1, e.g. 97 = 4 s, 241 = 10 s), frame_rate, seed,
        first_frame_image (path), last_frame_image (path), image_strength, steps, distilled (default true), out (mp4 path)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workerlib import offload_mode, seed_generator, serve, vram_gb  # noqa: E402

REPO = os.environ.get("ULTRON_LTX_REPO", "Lightricks/LTX-2.5-Diffusers")


def frames_8k1(n: int) -> int:
    """LTX needs num_frames = 8k+1. Round to the nearest valid count (241 for a 10 s clip at 24 fps)."""
    n = max(9, int(n))
    return ((n - 1 + 4) // 8) * 8 + 1


def mult32(x: int) -> int:
    return max(32, int(round(int(x) / 32)) * 32)


class LTX:
    def load(self):
        import torch
        from diffusers import LTX2ConditionPipeline

        t = time.time()
        self.torch = torch
        # the optional Gemma 4 prompt enhancer is not used (the engine writes full prompts); skip loading it
        try:
            pipe = LTX2ConditionPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16, prompt_enhancer=None)
        except (TypeError, ValueError):
            pipe = LTX2ConditionPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)
        mode = offload_mode()
        fp8 = os.environ.get("ULTRON_LTX_FP8", "auto")
        if fp8 == "1" or (fp8 == "auto" and vram_gb() < 70):
            pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        if mode == "none":
            pipe.to("cuda")
        elif mode == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        self.pipe = pipe
        print(f"[ltx] loaded {REPO} offload={mode} fp8={fp8} vram={vram_gb():.0f}GB in {time.time() - t:.0f}s", file=sys.stderr)

    def run(self, p, ctx):
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES
        from diffusers.utils import encode_video, load_image

        width, height = mult32(p.get("width", 1024)), mult32(p.get("height", 576))
        num_frames = frames_8k1(p.get("num_frames", 97))
        fps = float(p.get("frame_rate", 24))
        distilled = p.get("distilled", True)
        conds = []
        strength = float(p.get("image_strength", 1.0))
        if p.get("first_frame_image"):
            conds.append(LTX2VideoCondition(frames=load_image(p["first_frame_image"]).convert("RGB"), index=0, strength=strength))
        if p.get("last_frame_image"):
            conds.append(LTX2VideoCondition(frames=load_image(p["last_frame_image"]).convert("RGB"), index=-1, strength=strength))
        total = len(DISTILLED_SIGMA_VALUES) if distilled else int(p.get("steps") or 30)

        def on_step(pipe, i, t, kw):
            ctx.progress(i + 1, total, f"denoising {i + 1}/{total}")
            return kw

        kw = dict(conditions=conds or None, prompt=p["prompt"],
                  negative_prompt=p.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT,
                  width=width, height=height, num_frames=num_frames, frame_rate=fps,
                  generator=seed_generator(p.get("seed")), output_type="np", return_dict=False,
                  callback_on_step_end=on_step)
        if distilled:  # fixed 8-step schedule, no CFG (the transformer/ folder of LTX-2.5-Diffusers is the distilled one)
            kw.update(sigmas=DISTILLED_SIGMA_VALUES, guidance_scale=1.0, audio_guidance_scale=1.0, stg_scale=0.0,
                      audio_stg_scale=0.0, modality_scale=1.0, audio_modality_scale=1.0)
        else:
            kw.update(num_inference_steps=total)
        t = time.time()
        out = self.pipe(**kw)
        video, audio = (out[0], out[1]) if isinstance(out, tuple) and len(out) > 1 else (out[0] if isinstance(out, tuple) else out, None)
        ctx.progress(total, total, "encoding mp4")
        dst = Path(p["out"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        a = audio[0].float().cpu() if audio is not None else None
        sr = getattr(getattr(self.pipe, "vocoder", None), "config", None)
        sr = getattr(sr, "output_sampling_rate", 24000) if sr is not None else 24000
        encode_video(video[0], fps=fps, output_path=str(dst), audio=a, audio_sample_rate=sr if a is not None else None)
        return {"path": str(dst), "format": "mp4", "frames": num_frames, "fps": fps, "width": width, "height": height,
                "seconds": round(num_frames / fps, 2), "render_seconds": round(time.time() - t, 1), "model": REPO}


if __name__ == "__main__":
    serve(LTX())

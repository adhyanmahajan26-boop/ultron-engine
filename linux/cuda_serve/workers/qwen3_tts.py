"""Qwen3-TTS 1.7B worker (plain speech + voice clone). Runs in the `qwentts` venv (qwen-tts pins its own transformers).

params (mlx-serve /v1/audio/speech fields): input, ref_audio (path), ref_text, language, speed, out (wav path)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workerlib import serve  # noqa: E402

REPO = os.environ.get("ULTRON_QWEN3_TTS_REPO", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")


class TTS:
    def load(self):
        import torch
        from qwen_tts import Qwen3TTSModel

        t = time.time()
        self.m = Qwen3TTSModel.from_pretrained(REPO, device_map="cuda", dtype=torch.bfloat16)
        print(f"[qwen3-tts] loaded {REPO} in {time.time() - t:.0f}s", file=sys.stderr)

    def run(self, p, ctx):
        import soundfile as sf

        t = time.time()
        kw = {"text": p["input"], "language": p.get("language") or "Auto"}
        if not p.get("ref_audio") and os.environ.get("ULTRON_DEFAULT_VOICE_REF"):
            p["ref_audio"] = os.environ["ULTRON_DEFAULT_VOICE_REF"]
        if not p.get("ref_audio"):
            raise ValueError("Qwen3-TTS Base clones a voice: pass ref_audio (voice='mine'), or set ULTRON_DEFAULT_VOICE_REF")
        if p.get("ref_audio"):
            kw["ref_audio"] = p["ref_audio"]
            if p.get("ref_text"):
                kw["ref_text"] = p["ref_text"]
            else:
                kw["x_vector_only_mode"] = True  # no transcript: clone timbre from the speaker vector only
        wavs, sr = self.m.generate_voice_clone(**kw)
        dst = Path(p["out"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        y = wavs[0]
        if p.get("speed") and abs(float(p["speed"]) - 1) > 1e-3:
            import librosa
            y = librosa.effects.time_stretch(y.astype("float32"), rate=float(p["speed"]))
        sf.write(dst, y, sr)
        return {"path": str(dst), "seconds": round(time.time() - t, 1), "model": REPO}


if __name__ == "__main__":
    serve(TTS())

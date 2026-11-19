#!/usr/bin/env python
"""Linux speech recognition for Content Studio (faster-whisper large-v3-turbo on CUDA). Same contract as the Mac
src/studio/asr.py (mlx-audio):  asr.py AUDIO [--language en] [--model DIR]
Prints one line `ASR_JSON:{...}` with {"text": str, "words": [{"word","start","end"}]}.
"""
import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("ULTRON_AVATAR", pathlib.Path.home() / "ultron-work/avatar"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--language", default="en")
    ap.add_argument("--model", default=os.environ.get("ULTRON_ASR_MODEL", str(ROOT / "models/whisper-turbo-ct2")))
    a = ap.parse_args()

    from faster_whisper import WhisperModel

    model_id = a.model if pathlib.Path(a.model).exists() else "large-v3-turbo"
    try:
        m = WhisperModel(model_id, device="cuda", compute_type="float16")
    except Exception as e:  # no GPU / CUDA libs missing: still work, slower
        print(f"asr: CUDA unavailable ({e}); using CPU int8", file=sys.stderr)
        m = WhisperModel(model_id, device="cpu", compute_type="int8")
    lang = None if a.language in ("", "auto") else a.language
    segs, _info = m.transcribe(str(a.audio), language=lang, word_timestamps=True, vad_filter=False, beam_size=5)
    text, words = [], []
    for s in segs:
        text.append(s.text.strip())
        for w in s.words or []:
            words.append({"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)})
    print("ASR_JSON:" + json.dumps({"text": " ".join(t for t in text if t), "words": words}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

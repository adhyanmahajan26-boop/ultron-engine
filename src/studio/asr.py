#!/usr/bin/env python
"""Local speech recognition (Whisper-turbo via mlx-audio) for Content Studio.

Run with the local-avatar venv:
    asr.py AUDIO [--language en]

Prints one line `ASR_JSON:{...}` with {"text": str, "words": [{"word","start","end"}]}.
"""
import argparse
import os
import json
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("ULTRON_AVATAR", pathlib.Path.home() / "ultron-work/avatar"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--language", default="en")
    ap.add_argument("--model", default=str(ROOT / "models/whisper-turbo"))
    a = ap.parse_args()

    from mlx_audio.stt.utils import load

    model = load(a.model)
    r = model.generate(str(a.audio), language=a.language, word_timestamps=True)
    text = (getattr(r, "text", None) or "").strip()
    words = []
    for s in getattr(r, "segments", None) or []:
        ws = getattr(s, "words", None) if not isinstance(s, dict) else s.get("words")
        for w in ws or []:
            g = (lambda k: w.get(k) if isinstance(w, dict) else getattr(w, k, None))
            words.append({"word": (g("word") or "").strip(), "start": g("start"), "end": g("end")})
    print("ASR_JSON:" + json.dumps({"text": text, "words": words}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Batch word-timestamp ASR (local Whisper-turbo via mlx-audio) -> one json per audio.
usage: asr_words.py [--lang en] out_dir a.wav b.wav ...    (writes out_dir/<stem>.json = {"text":..., "words":[{word,start,end}]})
Same format as content-studio transcribe job results (result.words), usable by voicefit2.py via a .json path.
"""
import sys, os, json
from mlx_audio.stt.utils import load
args = sys.argv[1:]; lang = "en"
if args and args[0] == "--lang": lang = args[1]; args = args[2:]
out_dir, files = args[0], args[1:]
os.makedirs(out_dir, exist_ok=True)
m = load("@AVATAR@/models/whisper-turbo")
for f in files:
    r = m.generate(f, language=lang, word_timestamps=True)
    words = []
    for s in getattr(r, "segments", None) or []:
        for w in (s.get("words") if isinstance(s, dict) else getattr(s, "words", None)) or []:
            g = (lambda k: w[k] if isinstance(w, dict) else getattr(w, k))
            words.append({"word": str(g("word")).strip(), "start": round(float(g("start")), 3), "end": round(float(g("end")), 3)})
    p = os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + ".json")
    json.dump({"text": (getattr(r, "text", "") or "").strip(), "words": words}, open(p, "w"), ensure_ascii=False)
    print(os.path.basename(f), len(words), "words")

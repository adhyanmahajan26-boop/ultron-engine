#!/usr/bin/env python
"""Linux voice conversion with Chatterbox (PyTorch, CUDA). Same contract as src/tools/voiceconvert.py on the Mac:
    voiceconvert.py source.wav reference.wav out.wav
Keeps the SOURCE words, rhythm and timing (e.g. LTX's native speech, which is perfectly lip-synced) and swaps the
voice to the reference speaker. Prints voice similarity to the reference before and after, and duration drift.
Runs in the chatterbox venv; when started with another python it re-executes itself there.
"""
import os
import pathlib
import sys

HOME = pathlib.Path(os.environ.get("ULTRON_HOME", pathlib.Path.home() / "ultron-work"))
PY = os.environ.get("ULTRON_CHATTERBOX_PY", str(HOME / "venvs/chatterbox/bin/python"))
if pathlib.Path(PY).exists() and pathlib.Path(sys.executable).resolve() != pathlib.Path(PY).resolve() and not os.environ.get("_ULTRON_VC_REEXEC"):
    os.environ["_ULTRON_VC_REEXEC"] = "1"
    os.execv(PY, [PY, __file__, *sys.argv[1:]])

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from chatterbox.vc import ChatterboxVC  # noqa: E402

src, ref, out = sys.argv[1:4]
dev = "cuda" if torch.cuda.is_available() else "cpu"
vc = ChatterboxVC.from_pretrained(dev)
wav = vc.generate(src, target_voice_path=ref).squeeze(0).numpy()
sf.write(out, wav, vc.sr)


def emb(ve, path):
    w, _ = librosa.load(path, sr=16000)
    e = ve.embeds_from_wavs([w], sample_rate=16000)
    e = np.asarray(e).mean(axis=0).ravel()
    return e / (np.linalg.norm(e) + 1e-9)


try:
    from chatterbox.models.voice_encoder import VoiceEncoder
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(hf_hub_download("ResembleAI/chatterbox", "ve.safetensors")))
    ve.to(dev).eval()
    er, es, eo = emb(ve, ref), emb(ve, src), emb(ve, out)
    d_src = librosa.get_duration(path=src)
    print(f"source {d_src:.2f}s -> converted {len(wav) / vc.sr:.2f}s | voice similarity to reference: "
          f"source {float(er @ es):.2f} -> converted {float(er @ eo):.2f}")
except Exception as e:  # the similarity line is informational
    print(f"converted -> {out} ({len(wav) / vc.sr:.2f}s); similarity check skipped: {e}")

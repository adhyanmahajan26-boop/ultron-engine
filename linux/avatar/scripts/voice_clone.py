#!/usr/bin/env python
"""Linux Fish Audio S2 Pro voice clone (official fish-speech, CUDA). Same flags the studio server passes on the Mac:
    voice_clone.py --ref-audio REF.wav --text "..." --out OUT.wav [--ref-text "..."] [--speed 1.1]
Runs fish-speech's documented three steps in its own venv ($ULTRON_FISH_DIR/.venv):
  1. fish_speech/models/dac/inference.py -i REF --checkpoint-path checkpoints/s2-pro/codec.pth   -> fake.npy
  2. fish_speech/models/text2semantic/inference.py --text T --prompt-text R --prompt-tokens fake.npy -> codes_0.npy
  3. fish_speech/models/dac/inference.py -i codes_0.npy --checkpoint-path .../codec.pth            -> fake.wav
S2 Pro needs at least 24 GB of GPU memory. Licence: Fish Audio Research License (non-commercial).
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HOME = pathlib.Path(os.environ.get("ULTRON_HOME", pathlib.Path.home() / "ultron-work"))
FISH = pathlib.Path(os.environ.get("ULTRON_FISH_DIR", HOME / "fish-speech"))
PY = os.environ.get("ULTRON_FISH_PY", str(FISH / ".venv/bin/python"))
CKPT = FISH / "checkpoints/s2-pro"


def run(args, cwd):
    print("+", " ".join(map(str, args))[:300], file=sys.stderr, flush=True)
    subprocess.run([PY, *map(str, args)], cwd=cwd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if not (CKPT / "codec.pth").exists():
        sys.exit(f"Fish S2 Pro weights missing at {CKPT}; run linux/setup.sh (component fish)")
    ref = pathlib.Path(a.ref_audio).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        td = pathlib.Path(td)
        # fish-speech scripts write fake.npy / codes_0.npy / fake.wav into the working directory and resolve their
        # default checkpoint paths relative to it, so link the checkout's checkpoints/ into the temp dir.
        (td / "checkpoints").symlink_to(FISH / "checkpoints")
        run([FISH / "fish_speech/models/dac/inference.py", "-i", ref, "--checkpoint-path", CKPT / "codec.pth"], td)
        t2s = [FISH / "fish_speech/models/text2semantic/inference.py", "--text", a.text,
               "--prompt-tokens", td / "fake.npy"]
        if a.ref_text:
            t2s += ["--prompt-text", a.ref_text]
        if a.seed is not None:
            t2s += ["--seed", str(a.seed)]
        run(t2s, td)
        codes = sorted(td.glob("codes_*.npy"))
        if not codes:
            sys.exit("fish-speech produced no codes_*.npy")
        (td / "fake.wav").unlink(missing_ok=True)
        run([FISH / "fish_speech/models/dac/inference.py", "-i", codes[0], "--checkpoint-path", CKPT / "codec.pth"], td)
        wav = td / "fake.wav"
        if not wav.exists():
            sys.exit("fish-speech produced no fake.wav")
        if a.speed and abs(a.speed - 1) > 1e-3:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-filter:a", f"atempo={a.speed}", str(out)], check=True)
        else:
            shutil.copy(wav, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

---
name: reel-engine
description: Generate talking-avatar videos and reels locally on this Mac - clone the user's voice, drive their face from their own footage, write scripts, fetch b-roll, add captions and plaque cards, and assemble a finished MP4. Use when the user wants a reel, a talking-head video, an avatar video, voice cloning, a video of themselves saying something, or asks about the local video engine / Engine Studio.
---

# Reel Engine

Local video pipeline at `@AVATAR@`.
Everything runs offline. Nothing is uploaded.

## Run it

```bash
cd @AVATAR@
./start-studio.sh          # UI at http://127.0.0.1:8189
```

The UI has three tabs: **Reel Engine** (topic → finished reel), **Voice**, **Avatar**.

## Command line

```bash
V=.venv/bin/python

# 1. topic -> scene breakdown
$V scripts/write_script.py --topic "..." --scenes 5 --out out/script.json

# 2. clone the voice for one line
$V scripts/voice_clone.py --ref-audio inputs/my_voice_ref.wav \
   --ref-text "$(cat inputs/my_voice_ref.txt)" --text "..." --out out/say.wav

# 3. drive the face from real footage
$V scripts/lipsync.py --video inputs/source_clip.mp4 --audio out/say.wav --out out/talk.mp4

# 4. whole pipeline, topic to MP4
$V scripts/make_content.py --topic "..." --scenes 5 --footage inputs/source_clip.mp4

# 5. turn generated scenes into a real edit (b-roll, captions, transitions)
$V scripts/edit_reel.py --work outputs/reel --out outputs/reel/edited.mp4
```

Always run `make_content.py` before `edit_reel.py` — the editor consumes what
the pipeline produces.

## Rules that matter

**Never use an image-to-video avatar model here.** (LongCat was tried and deleted.) It does not preserve the
user's identity — it generates a different person. Lip-sync on their real
footage is the only approach that keeps their face, because it repaints the
mouth region and leaves everything else untouched.

**Voice choice.** Fish S2 Pro matches the user's Hinglish accent best and is
the default; it is research-licensed, which is fine for personal and community
use. Qwen3-TTS is Apache-2.0 if commercial use ever matters. Chatterbox is MIT
but does not hold the accent — the user rejected it for that.

**`ref_text` must match the reference audio phonetically.** The user speaks
Hinglish, so the transcript is Devanagari, not an English translation. This is
the single biggest lever on voice quality. It lives in
`inputs/my_voice_ref.txt`.

**Keep lip-sync at 256px.** 512px costs ~18 GB more and looks *worse* —
upstream has known mask-edge artifacts around the mouth.

**Footage must be at least as long as the longest scene.** Output length
follows the audio. Expressions come from the source footage, so pick an
animated stretch — a still take gives a talking statue.

**Preserve audio sample rate through assembly.** LatentSync resamples to 16 kHz
for its Whisper pass and muxes that back. Always re-attach the voice model's
original 44.1 kHz audio, or narration sounds muffled.

## Memory

| Stage | Needs |
|---|---|
| Script (Ornith 9B) | ~9 GB |
| Voice (Fish) | 20 GB |
| Voice (Qwen3 / Chatterbox) | 18 GB / 5 GB |
| Lip-sync 256px | 10 GB |
| Lip-sync 512px | 28 GB |

Stages run sequentially, so the requirement is the largest stage, not the sum.
MLX grows to fill free RAM; `--mlx-limit N` caps it with no quality change.
**`ps` and RSS under-report MLX badly** — a process holding 75 GB showed as
0.07 GB. Use `top -l 1 -o mem -n 15 -s 0 -stats pid,command,mem,cmprs`.

Stop `ui/server.py` when finished; it holds weights resident (it now auto-drops
them after 10 idle minutes).

## Environment traps

- **`HF_HUB_DISABLE_XET=1` is required.** With Xet enabled, downloads stall to
  0 B/s on this connection; plain HTTP sustains ~12 MB/s. Fetch one repo at a
  time. Never SIGSTOP a download — the sockets die and it spins without
  resuming.
- **The Xcode licence is not accepted**, so `git`, `clang` and
  `/usr/bin/python3` all refuse to run. Use tarballs, prebuilt wheels and
  `uv`. Fix needs their password: `sudo xcodebuild -license accept`.
- **ffmpeg** is `bin/ffmpeg`, symlinked from the imageio-ffmpeg wheel. There is
  no Homebrew on this machine.

See `references/broll.md` before touching asset fetching, and
`references/troubleshooting.md` when something breaks.

---
name: reel
description: Make an Instagram/YouTube reel locally on the Mac Studio — talking-head reels of the user (Fish Audio voice clone + LatentSync lip-sync on their real footage), AI-video reels (MiniMax-H3 clips), or motion-graphics reels (HyperFrames), with ACE-Step music and captions. Use when the user types /reel or asks for a reel, short, TikTok or vertical video. Asks reel type, voice, music and length as clickable options first.
---

# /reel — local reels

Tools: `content-studio` MCP → `make_reel`, `generate_speech`, `lipsync`,
`generate_video`, `generate_image`, `generate_music`, `mux_audio`,
`render_hyperframes`, `free_memory`, `job_status`, `studio_status`.
Pipeline details and pitfalls: the `reel-engine` skill.

## 1. Ask — one question call with options

`AskUserQuestion` (Claude Code) / `question` (OpenCode); otherwise a numbered list.
Skip anything already answered; always get the **topic** (ask in text if missing).

| Question | Options |
|---|---|
| Reel type | **Me talking (lip-synced)** — my face + cloned voice · **AI video clips** — MiniMax-H3 shots · **Motion graphics** — HyperFrames text/graphics · **Mix** — me talking + AI b-roll |
| Voice | **My voice — Fish S2 Pro** (best Hinglish, slower) · **My voice — Qwen3-TTS** (faster) · **No voiceover, music only** |
| Music | **ACE-Step background music** · **No music** |
| Length | **15 s** · **30 s** · **60 s** |

Scenes ≈ length / 7 s (15 s → 2, 30 s → 4, 60 s → 8).

## 2. Build by type

**Me talking** (fastest path, fully automatic):
`make_reel(topic, scenes, engine="fish"|"qwen3", edit=true, captions=true)`.
It writes the script, clones the voice, lip-syncs the user's footage
(`inputs/source_clip.mp4`, must be ≥ the longest scene), adds plaque cards,
b-roll, captions and cross-dissolves. Takes many minutes — poll
`job_status(job_id, wait_seconds=40)` and report progress lines.
Then if music was chosen: `generate_music(..., instrumental=true, duration_seconds=length)`
and `mux_audio(final.mp4, music.wav, audio_volume=0.15, keep_original_audio=true)`.

**AI video clips**:
1. Write a script: hook line + one shot per scene (subject, action, camera move).
2. `free_memory()`, then `generate_video` per shot (fast preset, 512×896, ~97 frames).
3. Voiceover if chosen: `generate_speech(full_script, engine)`.
4. Music if chosen: `generate_music(instrumental=true, duration_seconds=length)`.
5. Assemble in HyperFrames (1080×1920): clips in order, voiceover + ducked music,
   word-timed captions, hook title in the first 2 s. `render_hyperframes(project_dir)`.

**Motion graphics**: author a HyperFrames composition with the HyperFrames skills
(`/hyperframes` → `hyperframes-core`, `hyperframes-animation`, `hyperframes-registry`),
1080×1920, with `generate_image` stills as needed, voiceover/music from the tools
above. Preview, then `render_hyperframes`.

**Mix**: run "Me talking" with `edit=false`, generate 2–4 AI b-roll clips, and cut
them together in HyperFrames with captions and music.

## 3. Memory

Stages run one at a time. Fish ≈ 20 GB, LatentSync ≈ 10 GB, MiniMax 4-bit ≈ 41 GB,
8-bit ≈ 70 GB (96 GB total). Call `free_memory()` between heavy stages.

## 4. Deliver

Give the final MP4 path (under `~/ai/content-studio/outputs/`), its length, and the
script used. Offer: a different hook, re-voice, swap music, or a 15 s cut.

## Long jobs

Tool calls return within ~45 s (MCP clients time out at 60 s). If a result says
`status: "running"`, keep calling `job_status(job_id)` — each call waits up to 40 s —
and tell the user the progress line between calls. Jobs keep running even if the
chat or MCP server restarts; `list_jobs()` finds them again.

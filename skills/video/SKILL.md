---
name: video
description: Generate short AI video clips with sound locally on the Mac Studio (MiniMax-H3 via mlx-serve) through the content-studio MCP. Use when the user types /video or asks for an AI-generated clip, b-roll, animated shot or image-to-video made locally. Asks quality, shape, length and starting point as clickable options first.
---

# /video — local AI video clips

Tools: `content-studio` MCP → `generate_video`, `generate_image`, `mux_audio`,
`free_memory`, `job_status`.

## 1. Ask — one question call with options

Use `AskUserQuestion` (Claude Code) / `question` (OpenCode); otherwise a numbered list.
Skip what the user already said.

| Question | Options |
|---|---|
| Model | **Fast — MiniMax-H3 4-bit** (41 GB, `quality:"fast"`) · **Best — MiniMax-H3 8-bit** (70 GB, `quality:"quality"`) |
| Shape | **Vertical reel 512×896** · **Landscape 896×512** · **Square 640×640** |
| Length | **~2 s (49 frames)** · **~4 s (97 frames)** · **~8 s (193 frames)** |
| Start from | **Text only** · **Generate a still first, then animate it** · **My own image** (ask for the path) |

Clips come with their own generated soundtrack. The model rounds frame counts up
(49 → 56 frames ≈ 2.3 s at 24 fps). A small 512×896 clip takes ~2 min on the fast model.

Vertical frames work well for people and street scenes, but wide horizon shots
(ocean, landscapes) sometimes come out sideways in 512×896 — use landscape for those,
or check a frame before using the clip.

## 2. Prepare

- "Generate a still first": run `generate_image` (quality preset, same aspect) and
  pass its path as `first_frame_image`.
- Before the 8-bit model, call `free_memory()` — it needs ~70 GB.
- Write a shot-style prompt: subject + action, camera move (dolly, orbit, handheld,
  slow push-in), lighting, mood. One shot per clip.

## 3. Generate

`generate_video(prompt, quality, width, height, num_frames, first_frame_image)` →
poll `job_status(job_id, wait_seconds=40)` until done.

For a multi-shot piece, generate each shot as its own clip, then hand off to `/reel`
(HyperFrames edit) to cut them together with music and captions.

## 4. Deliver

Give the MP4 path(s) (under `~/ai/content-studio/outputs/video/`), duration and size.
Offer: another take (new seed), add music (`/song` then `mux_audio`), or build a reel.

## Long jobs

Tool calls return within ~45 s (MCP clients time out at 60 s). If a result says
`status: "running"`, keep calling `job_status(job_id)` — each call waits up to 40 s —
and tell the user the progress line between calls. Jobs keep running even if the
chat or MCP server restarts; `list_jobs()` finds them again.

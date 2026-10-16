---
name: image
description: Generate or edit images locally on the Mac Studio (Qwen-Image 2.1, Mage-Flow) through the content-studio MCP. Use when the user types /image or asks for a picture, photo, poster, thumbnail, product shot or image edit made locally. Asks the user for quality, aspect ratio, count and style as clickable options before generating.
---

# /image — local image generation

Tools come from the `content-studio` MCP server (`generate_image`, `edit_image`,
`job_status`, `studio_status`). Everything runs on this Mac; nothing is uploaded.

## 1. Ask what they want — as options, in ONE question call

Use the ask-question tool (`AskUserQuestion` in Claude Code, `question` in OpenCode).
If no such tool exists, print the same choices as a numbered list and wait.
Skip any question the user's message already answered.

| Question | Options |
|---|---|
| Quality | **Best — Qwen-Image 2.1** (~1–3 min, `quality:"quality"`) · **Fast — Mage-Flow Turbo** (~5–10 s, `quality:"fast"`) |
| Shape | **Square 1024×1024** · **Reel / story 9:16 → 768×1344** · **Landscape 16:9 → 1344×768** · **Portrait 4:5 → 1024×1280** |
| How many | **1** · **2** · **4** (different seeds) |
| Style | **Photoreal** · **Illustration** · **3D render** · **Cinematic** |

If they attached or named an existing image and want it changed, that is an
**edit**: skip Quality/Style and call `edit_image(prompt, image_paths=[...])`.

## 2. Write the prompt

Expand the request into one rich prompt: subject, setting, lighting, lens/medium,
mood, colour palette, plus the chosen style. Keep any text that must appear in the
image short and in quotes.

## 3. Generate

Call `generate_image(prompt, quality, size, count)`. If it returns
`status:"running"`, call `job_status(job_id, wait_seconds=40)` until done.

## 4. Deliver

Show the file paths (they are under `~/ai/content-studio/outputs/images/`). If you
can display or send files, show the image. Offer one-line follow-ups: variations
(same prompt, new seed), an edit, or turning it into a video with `/video`.

If a tool errors with a model that is not downloaded, say which one and fall back
to the other preset.

## Long jobs

Tool calls return within ~45 s (MCP clients time out at 60 s). If a result says
`status: "running"`, keep calling `job_status(job_id)` — each call waits up to 40 s —
and tell the user the progress line between calls. Jobs keep running even if the
chat or MCP server restarts; `list_jobs()` finds them again.

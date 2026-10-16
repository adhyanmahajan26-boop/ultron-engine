# Content Studio

Local MCP server for content generation on the Mac Studio. Nothing leaves the machine.

| Tool | Backend |
|---|---|
| `generate_image`, `edit_image` | mlx-serve: Qwen-Image 2.1 (quality), Mage-Flow Turbo (fast / edit) |
| `generate_video` | mlx-serve: MiniMax-H3 4-bit (fast) / 8-bit (quality), with soundtrack |
| `generate_music` | mlx-serve: ACE-Step 1.5 XL Turbo |
| `generate_speech` | Fish S2 Pro voice clone (local-avatar) or Qwen3-TTS (mlx-serve). Output is auto-checked with Whisper (`asr_text`, `word_error_rate`, `ok`); a reference with no transcript is transcribed first |
| `transcribe` | Whisper-turbo (local-avatar `models/whisper-turbo`): text + word timestamps for any audio/video |
| `lipsync` | LatentSync 1.5 (local-avatar) |
| `make_reel` | local-avatar `make_content.py` + `edit_reel.py` |
| `render_hyperframes` | `npx hyperframes render` |
| `mux_audio` | ffmpeg |
| `studio_status`, `job_status`, `list_jobs`, `cancel_job`, `free_memory` | housekeeping |

Heavy jobs (image, video, music, speech, lipsync, reel, render, transcribe) run **one at a time**: extra jobs wait
in a queue (progress says `queued`). Raise `CONTENT_STUDIO_MAX_HEAVY` only on a machine with headroom. Starting many
at once crashed the Mac on 2026-10-03.

Generators run as background jobs; each call waits `wait_seconds` and otherwise
returns a `job_id` for `job_status`. Outputs land in `outputs/<kind>/`, and every
finished job is appended to `outputs/jobs.jsonl`.

Needs: MLX Core (mlx-serve on 127.0.0.1:11234) running, `uv`.

Commands (Claude Code skills in `~/.claude/skills/`): `/image`, `/video`, `/song`, `/reel`.
Each asks its options first, then calls these tools.

Test without a client:

```bash
uv run --script test_client.py studio_status '{}'
uv run --script test_client.py generate_image '{"prompt":"a fox","quality":"fast"}'
```

Changes 2026-10-03/04: heavy-job queue, `transcribe` tool, automatic voice verification, render quality options fixed
(`draft|looks|standard|high|delivery`; `final` was invalid). Backup of the old server: `server.py.bak-2026-10-03`.
The MCP client must reconnect to pick up the new `transcribe` tool; job workers use the new code immediately.

---
name: song
description: Make a song, jingle, beat or background music locally with ACE-Step 1.5 (the "ACE" music model) through the content-studio MCP, optionally turned into a lyric video or music video with HyperFrames. Use when the user types /song or asks for music, a track, a jingle, background music or a lyric video. Asks type, genre, language and length as clickable options first.
---

# /song — local music (ACE-Step 1.5) + optional video

Tools: `content-studio` MCP → `generate_music`, `generate_image`, `generate_video`,
`mux_audio`, `render_hyperframes`, `job_status`.

## 1. Ask — one question call with options

`AskUserQuestion` (Claude Code) / `question` (OpenCode); otherwise a numbered list.
Skip anything already answered.

| Question | Options |
|---|---|
| What | **Full song with vocals** · **Instrumental / background music** · **Song + lyric video (HyperFrames)** · **Song + AI music video** |
| Genre | **Bollywood / Hindi pop** · **Lo-fi / chill** · **Hip-hop / rap** · **Cinematic / epic** (Other = their own) |
| Language | **Hindi** · **English** · **Hinglish** · **Punjabi** (vocals only) |
| Length | **30 s jingle / reel** · **1 min** · **2 min** · **Full ~3 min** |

## 2. Lyrics

For vocals, write lyrics yourself unless the user gave them: tag sections
`[verse]`, `[chorus]`, `[bridge]`, keep lines short and singable, match the length
(30 s ≈ one verse + chorus). Write Hindi/Punjabi lyrics in Devanagari/Gurmukhi or
romanised as the user prefers. Show the lyrics in one short block, then generate —
don't wait for approval unless the user asked to review first.

## 3. Generate the music

`generate_music(prompt, lyrics | instrumental=true, duration_seconds, vocal_language, bpm?)`
- `prompt` = genre + mood + instruments + vocal type + tempo, e.g.
  "romantic hindi pop, soft male vocals, acoustic guitar, tabla, 90 bpm".
- `vocal_language`: "hi", "en", "pa" (Hinglish → "hi").
Poll `job_status(job_id, wait_seconds=40)` until done.

## 4. Video (only if chosen)

- **Lyric video**: author a HyperFrames composition with the HyperFrames skills
  (`/hyperframes` → `hyperframes-core`, `hyperframes-animation`), 1080×1920 for reels,
  lyrics timed to the track (use `npx hyperframes beats`/transcribe for timing),
  the WAV as the audio track, optionally a `generate_image` background. Preview,
  then `render_hyperframes(project_dir)`.
- **AI music video**: plan 3–6 shots matching the song's sections, generate each with
  `generate_video` (fast preset, vertical), cut them together in HyperFrames over
  the track, render with `render_hyperframes`. For a single clip, `mux_audio` is enough.

## 5. Deliver

Give the WAV/MP4 paths (under `~/ai/content-studio/outputs/`), the lyrics used, and
offer: another take (new seed), a different genre, or a 15 s cut for reels.

## Long jobs

Tool calls return within ~45 s (MCP clients time out at 60 s). If a result says
`status: "running"`, keep calling `job_status(job_id)` — each call waits up to 40 s —
and tell the user the progress line between calls. Jobs keep running even if the
chat or MCP server restarts; `list_jobs()` finds them again.

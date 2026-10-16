# Instructions for Claude Code (and other AI assistants) opening this repo

This is **Ultron Engine**, a local AI video studio (images, video, music, voice, lip-sync, editing). Models are large and nothing may be downloaded without the user's consent.

## First time in this repo? (no `ultron.config.json` yet) -> run the onboarding

1. Run `python3 ultron.py doctor` and `python3 ultron.py models`. Read the hardware line (`python3 ultron.py setup` prints it, then Ctrl-C if you only want it) and `docs/HARDWARE.md`.
2. **Ask the user with AskUserQuestion** (do not decide for them):
   - **Where should generation run?** `local` (all models on this machine, Apple Silicon only), `cloud` (rented GPU endpoints), or `mixed` (choose per component, for example video in the cloud and voice/ASR local).
   - If `mixed`: ask per component group (video, image, music, voice, lipsync). Use the sizes and peak RAM from the catalog in your question text.
   - If any `cloud`: ask for the endpoint URL and the **name** of the environment variable that holds the API key. Never ask for, type, store or echo the key itself. Point them at `docs/CLOUD.md`.
   - Ask which working folder to use for models (default `~/ultron-work`).
3. For every local model the user wants: tell them **filename/repo, source URL and real size** (the wizard prints the size from the Hugging Face API) and **get an explicit yes before downloading**. Then run, for example:
   `python3 ultron.py setup --mode mixed --local video_ltx,image_qwen --cloud "video_h3=https://their-endpoint|THEIR_KEY_ENV" --download video_ltx`
   (`--download` lists only what the user approved. Anything not listed is skipped.)
4. Offer to install the skills into `~/.claude/skills` (`--install-skills`); say they will not overwrite existing skills.
5. Finish with `python3 ultron.py doctor` and point the user to `docs/QUICKSTART.md`.

## Rules that protect the machine (learned the hard way, see docs/MEMORY_AND_QUEUE.md)
- **One heavy job at a time** (LTX, Qwen-edit, LatentSync, HyperFrames render, studio models). Run `runtime/tools/heavy_guard.sh &&` before each heavy job. Starting several at once crashed the Mac.
- Never download models, audio or video the user has not approved. Never pull copyrighted audio from the internet.
- Do not publish generated fan content of copyrighted characters; keep it private.
- Before using any model prompt style, read `skills/studio-ltx-video/SKILL.md` (action prompting rules).

## Where things are
`ultron.py` CLI | `ultron/catalog.json` model catalog | `src/tools` QA + audio tools | `src/studio` MCP server | `src/pipelines` example edit builders | `skills/` Claude Code skills | `docs/` everything else.

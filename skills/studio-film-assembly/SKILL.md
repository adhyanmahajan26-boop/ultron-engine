---
name: studio-film-assembly
description: Assemble a continuous, polished showcase film from generated studio clips in HyperFrames (persistent stage, crossfades, editor UI with animated avatar, infinity/galaxy endings, loudness-correct render) and verify every requested element actually appears. Use for any multi-scene flex/promo film, "show the engine" video, or when cuts look ugly/blank, requested elements don't show, or audio is quiet.
---

# Film assembly (HyperFrames) — what the user demanded

## Process (do not skip)
1. **Write down every element the user asks for** (e.g. "HyperFrames editor", "animated avatar riding", "galaxy ending", "remove pauses") and tick each against the **rendered frames** (1 fps contact sheets), not the code. In v1 the editor/avatar "didn't show at all" because I built a tiny text card; the user marked it failed.
2. **No gaps, no hard flash cuts.** One persistent `#stage` background that never goes blank; a docked monitor that moves/scales between layouts; crossfade talking windows (≈90 ms overlap, slices start in silence); whip/push-in/flash only for scene changes. Don't fade windows in/out of an empty background.
3. **Continuity over cleverness.** No awkward pauses (see studio-voice-clone), face-locked windows (see studio-face-lipsync-qa), no lips moving over silence, trim windows to speech.
4. **Check the render yourself before sending**: `ffmpeg fps=1 → tile` sheets; look at all of them. Check overlap/z-order (e.g. avatar covering the face), text legibility, loudness.
5. Send a **phone copy < 30 MB** (SendUserFile limit): `scripts/finish.sh VERSION` renders, loudness-normalises to −14 LUFS (v1 at −19.8 was too quiet), and makes the 720p copy. The full file stays in the project folder.

## HyperFrames gotchas hit
- `npx hyperframes lint` must be 0 errors; `check` before render. `fromTo` on `#flash` etc. needs `immediateRender:false` (+`overwrite:"auto"` where tweens overlap). Don't animate `letterSpacing`/`strokeDashoffset` is fine; opacity-only reveals need `opacity:1` in the destination vars.
- Page-script variables must be injected from Python (a `GAL_DUR is not defined` page error renders **all black**; snapshots reveal it).
- Duration lookups: use `os.path.splitext`, not `[:-4]` (`.webm`).
- `mix-blend-mode` disables the fast capture path (OK). Keep one root `index.html` (extra root files error).
- Preview with `npx hyperframes snapshot --at t1,t2,…` after each structural change.
- macOS shell: `sed -i` needs `''`; zsh won't word-split variables (`${=VAR}`); no `timeout`; `sleep N` loops are blocked → use `run_in_background` + a one-shot waiter (`scripts/wait_jobs.sh`) or a Monitor. Never poll.

## Pieces that worked (see scripts/)
- `build_template.py` — full composition builder (persistent stage, editor with bin/inspector/4-lane timeline, avatar riding the playhead on a glowing board, crossfaded monitor windows, beat card, finale). `patch_city_infinity.py` — avatar surfing into NYC + speaking, and the **infinity ending** (SVG ∞ path draw, galaxy infinite-zoom tunnel of 3 looping scaled/rotated layers, stars along a lemniscate, title).
- `avatar_key.py in.mp4 out.webm` — matte (hyperframes remove-background) + green despill + tight crop → RGBA webm for an animated avatar generated on a solid-green background.
- Animated avatar: edit_image the user's still into a semi-real Pixar-style 3D avatar, full-body on a flat green background (surfing a hoverboard), H3 animate, key it. Edit-image can't composite avatar into a scene reliably (room bleeds in) → composite the keyed avatar over a real/aerial plate in HyperFrames.
- Infinity/galaxy ending: user wants the galaxy as its own closing shot (not mixed into the speaker), moving stars, a transition *into* infinity, title "ULTRON ENGINE V3".

## Memory
User standards are also saved in memory: `feedback_video_editing_standards.md`.

## v4 (LTX) rebuild lessons (2026-10-12)
- **Rebuild by data, not by hand**: `film/patch4.py` patches `build3.py` -> `build4.py` (swap `assets/v2/a2` -> `assets/v4/a4`, add shots, add overlays). Same asset *names* as v3 so the timeline logic is untouched; copy v2 assets into v4 first so unfinished clips still resolve while testing.
- **Realistic Iron Man**: LTX copies the first frame, so use PHOTOREAL stills. Qwen-Image 2.1 `generate_image` for the suit/sky/airliner plate; **Qwen-Image-Edit 2511 (mflux, ~5-7 min)** for composites with his face + room + suit (Mage-Flow 2-ref edit garbled arms/faces and ignored 'no airplane'). Keep ONE canonical suit for the whole film (user: 'the suit changed - keep the real Iron Man suit').
- Transformations: LTX `--image A 0 1.0 --image B 96 1.0` (start+end frame): suit assembling on him in his room worked first try; reverse it for suit-off.
- **Talking windows**: LTX clip per line with the exact words in the prompt + calm wording; `ironman4/make_assets.py` fits the existing narration (voicefit2, phrase-level) and trims the ~0.5 s dead lead, upscales to 1080p.
- **Overlays added**: boot text (1.2 s), achievement card with REAL measured numbers (stats4.json), V4 title. Card tiles: nowrap, 60 px big numbers.
- Launch long jobs with Bash `run_in_background` (visible in the task list), not detached nohup.

---
name: studio-face-lipsync-qa
description: Generate talking shots of the user (or their animated avatar) with the local studio so the face stays accurate and lip-sync is clean. Use whenever making H3 clips of the user, running LatentSync lip-sync, checking identity/mouth quality, or fixing jiggling mouths, closed-mouth smirks, face drift, or cut-to-cut face jumps.
---

# Face fidelity + lip-sync QA (local studio)

## Generation rules (MiniMax-H3 via `generate_video`)
- **Every shot starts from a real still of the user** (extract frames from their footage; use as `first_frame_image`). Real photos are identity references only — **never trim real footage into the final cut** (user rule; also never reuse another AI's clips).
- Face holds ~2 s. ArcFace similarity: real-vs-real ≈ 0.90; fresh H3 window starts 0.95 and decays; accept sustained ≥ ~0.65; a 4 s clip fell to 0.4. So make **short (≈49-frame, 1024×576, quality=8-bit) windows** and cut between them; re-seed instead of using drifting ones.
- Measure with `scripts/face_sim.py clip.mp4 STEP ref1.png ref2.png …` (insightface buffalo_l). Looking is not enough.
- Prompts for talking windows: say "**speaks clearly, mouth moving with every word, not smirking, talking from the first frame**". Smirk/closed-lip/finger-pose sources → LatentSync leaves a static mouth (rejected W4b, W6b). A window that *grins broadly* drops similarity.
- Never ask H3 to invent the face under a mask/helmet. For "mask opens": generate the suit *assembling* onto a real still, then play it **reversed** (and slow it).
- Keep ONE canonical suit design (separately generated clips invent different suits). Replace backgrounds with an edit_image prompt that says "keep man + armour identical, replace ONLY background"; check similarity ≥ 0.9.
- Person-matting tools (`hyperframes remove-background`, u2net) **drop armour/props** → don't use them on suits. They are fine on plain hoodie/cartoon avatars.

## Lip-sync (LatentSync)
- Use **1.6 at 512 px** (`scripts/lipsync.py --version 1.6`): clearly sharper than 1.5, no mask artifacts seen, ~3 min per 2 s of video (≈4× slower than 1.5). One heavy job at a time; chain windows in a queue script that waits for the GPU to be idle.
- Slice the audio per window from tightened speech; clip must be ≥ audio length; output length follows audio.
- Windows with no speech: feed silent audio of the same length.
- **Mouth jiggle fix** (user-approved): `scripts/smooth_mouth.py in.mp4 audio.wav out.mp4 0.6` (1-2-1 temporal blend only inside a feathered mouth/jaw box).
- Review each result: ArcFace + 8-frame mouth crop strip; **send each approved window to the user as it lands**; reject windows whose mouth barely moves and regenerate the *source* clip, not just the sync.
- LatentSync works on a cartoon/animated avatar face too (face detector finds it); good for "the animated clone speaks".

## LatentSync reality check (learned the hard way — read before promising any lip-sync)
- **The MLX LatentSync 1.6 (512px) barely follows audio**: with a silent-audio control its mouth curve matched the speech version (corr .96) and the source footage (.97). Earlier "good" 1.6 windows were mostly the H3 clip's own mouth. v1.5 (256px) follows modestly.
- **Cause: reference leak.** `latentsync_mlx/pipeline.py` conditions on the unmasked SAME frame (`ref_pv`), so the model copies the source mouth. Opt-in patch (backup `.bak-before-refpatch`): env `LS_REF_MODE=fixed LS_REF_IDX=<closed-mouth frame>` or `LS_REF_MODE=shift LS_REF_SHIFT=n`. It raised voice-following on a clean test (1.6: -0.16 -> +0.52; 1.5: +0.52 -> +0.67). A closed-mouth/neutral source is best.
- **Always run**: (1) silent-audio control, (2) `tools/audio_influence.py` (delta vs source mouth in speech vs silence, corr with voice), (3) `tools/sync_metric.py`, (4) a **full-face frame contact sheet** — mouth-crop strips and metrics alone hid a pink "duck-pout" on a real bearded face (v1.5, guidance 4-6, fixed or shifted ref). Metric-tuned guidance 6 looked cartoonish.
- **RULE (proven, controlled test): the BASE must be a closed-mouth, frontal, steady face** — a clean still + slow zoom (ffmpeg zoompan) works on a real bearded face too (his own still: voice-follow .85, ArcFace .88-.94, natural lips). The pink 'duck-pout' came from MOVING real footage (head tilt, looking down, gestures, source mouth already talking), NOT from the face, skin tone or lighting. For motion, generate closed-mouth idle H3 clips as the base.
- Photoreal faces on moving footage are the hard case here. Prefer a stylised/animated avatar still (closed mouth, front-facing, face >=400px) and let LatentSync drive it; a still can be turned into a long uncut base with a slow zoom.

## One long uncut take (user asked for "one clip, no pauses, LatentSync only")
- LatentSync follows audio of any length: build ONE continuous master audio (phrases chained with ~80 ms breaths, tightened, warm EQ, loudnorm -14) and run it ONCE over a continuous source. A 30 s run takes ~57 min on this Mac (≈1.9 min per second at 1.6/512px).
- Source options: a continuous stretch of the user's real footage (scan 1 fps with insightface: all frames detected, |yaw|<.22, steady face width/centre, high motion; pick best 31 s) — only way to get an uncut take. Generated H3 clips are ≤~4.5 s, so a generated base needs chaining from last frames (check drift).
- **Default guidance 1.5 under-articulates a long take (lips nearly closed). Use `--guidance 3.0`** (tested on a 2.4 s excerpt: clearly wider shapes/teeth; slight pressed-lip smear on "m"). Always test guidance on a short excerpt first and compare mouth crops (crop around landmark 52-71 centre, 10 frames at 120 ms).
- Verify sync objectively: lip-opening vs audio RMS cross-correlation, best lag should be 0 ±1 frame (but low correlation does not prove articulation — look at crops).
- Never run LatentSync and an H3 job at the same time (memory).

## Cuts that don't look ugly
- Face centre wanders ±55 px between windows → run `scripts/facelock.py apply OUTDIR clips…` (scales ≥1.14× cover and aligns face centre/width) before editing.
- Cut at silent gap midpoints; overlap incoming clip ~90 ms (crossfade).
- Hand/arm gestures: footage may not contain the needed gesture (e.g. arms spread) → generate it from a real still.

## Audio-driven avatar models tried (don't repeat blindly)
- **LongCat-Video-Avatar 1.5 q8 (MLX port, xocialize/longcat-avatar-mlx)** on M3 Ultra 96 GB: 37 s/frame (57 min for 3.1 s @480x832), runner = one ~93-frame window from image+audio only (no video input, no continuation), and the output was a *different generic man* (ArcFace ~0 vs reference, sync corr -0.17). Not usable. InfiniteTalk/Wan2.2-S2V: no Mac build found. Stay with LatentSync v1.5 on a steady still + `tools/liveliness.py` for motion/blinks.

- LatentSync **v1.6 checkpoint was deleted** (disk cleanup; it lost every A/B). Only v1.5 (`--version 1.5`, 256px) is installed.

- If `generate_video`/`edit_image` return `ConnectError: All connection attempts failed`, mlx-serve died. Restart: `nohup "/Applications/MLX Core.app/Contents/MacOS/mlx-serve" serve --host 127.0.0.1 --port 11234 &` (not on PATH), check `curl 127.0.0.1:11234/health`.

- **Prefer LTX-2.5 now** (see skill `studio-ltx-video`): ~4x faster than H3, sharper, native speech+lips in en/hi. H3 + LatentSync stays as fallback.

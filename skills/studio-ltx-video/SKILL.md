---
name: studio-ltx-video
description: Generate talking / expressive video of the user (or anyone) locally with LTX-2.5 (MLX) on this Mac - image-to-video with native speech and lip-sync, English/Hindi/Hinglish by putting the words in the prompt, up to 10 s in one pass. Use when making talking clips, when H3 is too slow/soft, or when lips must match the words. Replaces the H3 + LatentSync workflow as the first choice.
---

# LTX-2.5 on this Mac (measured 2026-10-12)

## What it is / why it won
- Runner: **xocialize/ltx-2-mlx** (`$ULTRON_LTX/ltx-2-mlx-xoc`, `uv sync --all-extras`, needs mlx-lm >= 0.31.2). NOT dgrauet/ltx-2-mlx: that one wants its own pack format (`text_encoder.safetensors`) and fails on mlx-community's Gemma-4 folder ("gemma4_unified not supported" on old mlx-lm).
- Weights (HF cache, ~41 GB): `mlx-community/ltx-2.5-mlx-ditq8` (int8 DiT, file `transformer-distilled.safetensors`) + `mlx-community/ltx-2.5-mlx-q8` (everything else incl. `gemma4-12b-ltx-v1/`). Combine with a **symlink folder** `$ULTRON_LTX/pack` (all q8 files + the DiT symlink). Do not download the 100 GB base repo.
- Measured on M3 Ultra 96 GB, 1024x576: **97 frames (4.04 s) = ~112 s total** (stage1 8 steps ~35 s, stage2 3 steps ~55 s, + decode). H3 for the same length = ~7 min clean run. Sharper than H3, real expressions, mouth + speech generated together.

## Command (image-to-video with speech)
```
cd $ULTRON_LTX/ltx-2-mlx-xoc
uv run ltx-2-mlx generate --distilled --model @LTX@/pack \
  --prompt "A man ... looks into the camera and says in clear English: '<EXACT WORDS>'. His lips move precisely with each word, natural expressions, small head movements, static camera." \
  --image still_1024x576.png --height 576 --width 1024 --frames 97 --frame-rate 24 --seed 7 -o out.mp4
```
- Frames must be 8k+1 (97 = 4 s, 241 = 10 s). Do NOT pass `--gemma` (it would trigger a Gemma-3 download / wrong loader).
- **It speaks exactly the quoted words** - verified with the studio `transcribe` tool (language en / hi): English perfect, Hindi in Devanagari recognised, Hinglish ok. Speech pace ~2.4 words/s English, ~2.7 Hindi: write ~24-27 words for 10 s.
- The voice is LTX's own generic voice, NOT the user's clone.

## Putting the user's cloned voice on it
- `a2v` (audio-driven) does NOT work with these packs: it needs the dev transformer + LTX-2.3 distilled LoRA. Don't retry.
- Option A: LatentSync v1.5 g3 on the LTX clip (25 fps copy) with the clone audio: voice-follow +0.37 (vs raw LTX -0.06), face stays lively. Option B: Fish clone of the same words, then time-warp phrases to LTX's Whisper word times (keeps lips matched). Keep both versions, say which scored better (tools/sync_metric.py).

## Gotchas
- mlx-serve (studio) and LTX are both heavy: never run together; 241 frames uses the most memory (system used ~65 GB measured).
- First-frame still: use a closed-mouth frontal still without hands (clone10b/s48_1024.png). A mic in the still is carried into the clip.
- Identity: ArcFace vs the still 0.68-0.79 (lower than still+LatentSync .89-.94, similar to H3). Check it looks like him in frames, not only the metric.
- Audio is stereo 48 kHz AAC in the mp4; extract with ffmpeg -vn -ac 1 -ar 16000 for Whisper/sync metrics.

- **Prompt tone matters a lot**: 'expressive / natural expressions / eyebrow raises / small head movements' made LTX clips *overexpressive* (user feedback). For talking-head default to: 'speaks calmly in a steady, relaxed, composed manner ... subtle restrained facial expressions, head almost still, minimal gestures, calm even delivery'.

- **Voice fit: use `tools/voicefit2.py` (phrase-level, cuts only in pauses), NOT `voicefit.py`** (word-level cut inside words -> audible breaks: 11-17 abrupt jumps). Always run the break check (abrupt dB jumps inside speech) before sending audio.

- **Keyframes**: `--image PATH FRAME_IDX STRENGTH` repeated, FRAME_IDX must be NUMERIC (96 = last frame of 97), `last` is rejected. Start+end images make LTX interpolate transformations (helmet opening, suit assembling).
- **Realism**: LTX copies the look of the first frame. CG/ornate armour stills -> cartoonish video. Make a photoreal first frame (Qwen-Image 2.1 'photorealistic cinematic film still, real aerial camera, matte metallic suit, film grain') then edit_image 2-ref (suit image + his face) for the helmet-open frame.

- **a2v (audio-driven) works with the distilled weights, no download**: `ltx-2-mlx a2v --low-ram --model pack --audio clean.wav --image still.png --frames 97 --stage1-steps 12 --cfg-scale 1.0` (stage 2 swaps to the pre-fused distilled file). ~4.5-5 min per 4-5 s. Output audio == input audio. Mouth LEADS the audio by ~0.29 s -> advance the audio 0.29 s (trim silence at its start). Test1 prompt that worked: 'A man in a black hoodie sits in a warm lamp-lit room and speaks to the camera in a calm, composed, steady manner, his lips moving exactly with the voice. Subtle restrained expression, head almost still, natural blinks, static camera.'
- **Never put negations about objects/hands in LTX prompts** ('hands not visible' made a bottle appear and the head toss).

## a2v on real footage of the user (C0656 test, 2026-11-19)
- Still: a frontal, mouth-closed frame from the user's own footage, cropped to 576x1024 (vertical). Skip frames with hands even if they score higher on face metrics.
- Audio: one continuous phrase cut at a real pause, mono, loudnorm -14 LUFS; keep the full-rate copy for the final mux.
- Command: `ltx-2-mlx a2v --low-ram --model pack --audio clean.wav --image still.png --height 1024 --width 576 --frames 121 --frame-rate 24 --stage1-steps 12 --cfg-scale 1.0` (a2v needs `--frame-rate 24`). 121 frames = 5 s took ~5.5 min (stage1 3:38, stage2 1:19), peak ~45 GB system.
- Prompt: the calm talking-head prompt, but describe the still's real scene (clothes, background) instead of a generic room.
- **Audio offset: measure it per clip with SyncNet** (scan -0.12 to +0.36 s). On this clip SyncNet aligned at +0.08 s; the old fixed 0.29 s advance was worse. The mouth-vs-loudness metric over-reads because LTX opens the mouth on the breath before a word.
- **Seeds vary a lot**: seed 7 SyncNet 3.9, ArcFace 0.86 vs the still (calm, sharp); seed 42 SyncNet 1.7, over-expressive "O" mouths and raised brows. Run 2+ seeds and reject any take with SyncNet < 3.
- **Best lip sync from LTX = words in the prompt, not a2v** (same still, same phrase, 2026-11-19): `generate` with the exact transcript quoted, then Chatterbox voice conversion to the user (tools/voiceconvert.py, ref voice/ref_C.wav) keeps LTX's timing: SyncNet 6.55 / offset 0, ArcFace 0.85, voice similarity 0.70 -> 0.85, 135 s per 5 s take, peak ~64 GB (no low-RAM mode). a2v on the same phrase: SyncNet 3.9. Trade-off: the delivery (pace, intonation) is LTX's, and it can drop a leading filler word ("and"). Advance the audio by the SyncNet-measured offset (+0.08 s here); 0.29 s was 5-6 frames late.
- **Teeth** (user's top complaint, 2026-11-20): at 576x1024 with a mid shot the mouth is ~65 px wide, so LTX draws one white band, opens too wide and flickers. Fix that worked: a tighter still (face ~60% of frame width, cropped from full-res footage) plus the prompt "speaks with small, relaxed mouth movements, lips parting only slightly, upper teeth tips just visible, natural off-white teeth" -> separate teeth with gaps, SyncNet 8.44, ArcFace 0.85. A GFPGAN 1.4 mouth-only restore sharpens but can't fix shape and makes lips plastic; skip it on top of the tight render. Voice conversion: use an ~8 s clip of the user's own voice as reference (a full 10 s ref turned "data" into "beta"; ref_C scored 0.67 vs 0.83 similarity). If VC mangles word onsets on a line, use the Fish clone for that line.
- Compared with LatentSync 1.5 + mouthblend on the same footage (SyncNet 8.1-9.1): LTX a2v moves the whole head naturally but syncs far less tightly. Use the LatentSync blend for lip-sync on existing footage; use LTX a2v when there is no usable source video or the whole head should move.

## Memory rules (user complained twice about memory pressure)
- ONE heavy job at a time: LTX, mflux Qwen-Edit (2-ref @1024x576 = ~65 GB incl. compressed!), LatentSync, HyperFrames render, mlx-serve models. Run `tools/heavy_guard.sh && <job>` before every heavy job; it refuses if another is running, if the studio server holds models (call free_memory), or if used memory > 55 GB.
- Stopping a LatentSync/other job: `pkill -f scripts/lipsync.py` is NOT enough, its worker `scripts/inference.py` survives (was holding 23 GB). Kill both by name, then check `top -l 1 -o mem -n 5 -stats pid,command,mem,cmprs`.
- Check real usage with `top -o mem` (includes compressed), not `ps` RSS.

## Hindi (user decision 2026-10-12)
- The Hindi sample made with **LTX native Hindi speech + Chatterbox voice conversion to his voice (tools/voiceconvert.py, ref voice/ref_C.wav) + audio advanced 0.29 s** was rated **10/10 by the user**. NO LatentSync. The mouth-opening-vs-envelope sync metric read -0.27 on it and was WRONG for Hindi: for Hindi, sync is advisory only; trust the user's eye. Gate still enforces words, face, calm, no props, clean audio (tools/clip_qa.py).

## Action-scene prompting (lab 2026-10-16, priest stills, same seed, old vs new prompt)
Old style = effect list ("body splits, ash and blood spraying, camera orbiting"). New style won in 3/3 shots (clear cause -> effect, fewer chaotic frames).
1. Don't re-describe the still; say what CHANGES. Present tense, ONE flowing paragraph, 60-110 words.
2. ONE dominant event + at most 2 chained beats per 4 s (use "as / then / while"): wind-up -> impact -> reaction. 3+ beats = scene drift.
3. Block space: "priest on the left ... vampire on the right", foreground/background. Name who contacts whom.
4. Describe the physical result of the hit (head dissolves into ash, body staggers then buckles, sparks fly toward camera) and secondary motion (coat snaps, rain streaks, shell falls).
5. Camera: locked or slow handheld push-in. NEVER "orbit/drifts around/circles": F10 with "drifts left around the priest" jumped to a different room and a different man.
6. Avoid chaotic physics / crowds / many subjects; one hero + one opponent. No negations ("no X" creates X).
7. Sources: LTX prompt guides (rundiffusion, ltxworkflow): subject -> action -> camera -> mood, 4-8 sentences.

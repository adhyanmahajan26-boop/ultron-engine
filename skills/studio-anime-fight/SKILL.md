---
name: studio-anime-fight
description: Make a fan-made anime fight edit cut to music (hero = the user, villain/minions, signature moves, sad-to-fight arc) with the local studio (Qwen-Image-Edit + LTX-2.5 + ACE-Step + ffmpeg). Use when the user asks for an AMV-style fight scene; read the lessons first, v1 was judged 'trash' on 2026-10-13.
---

# Anime fight edit - what we learned (v1 = user: "scrap try")

## Pipeline that worked mechanically (3.5 h total, ~2.5 h generation)
- Music: ACE-Step via `generate_music`. Sad part (piano/strings 68 bpm, 24 s) + fight part (150 bpm electro-rock) joined by a 0.12 s silence at exactly 20.0 s. User rejected: ending-style song (129 bpm), phonk ('too strong'). Tempo/hit-density measured with librosa but only the user's ears decide.
- Stills: Qwen-Image 2.1 (`generate_image`) for villain/minion designs from a text description; Qwen-Image-Edit 2511 (`mflux-generate-qwen-edit`, ~5-8 min) for the anime hero from the user's still (H0), then every other hero still edited from H0 so the face stays; 2-ref edits for hero+villain composites. Hero eyes (R Sharingan red / L Rinnegan purple) came out right only in close-ups; wide action stills drew BLUE glowing eyes.
- Video: LTX-2.5 i2v from each still, 97f (sad) / 73f (fight), 2 takes each, `--distilled`, prompt ends with '2D anime, cel-shaded, hand-drawn look, keep the anime art style, no realism'. ~85-110 s per take.
- Edit: `fight/build_fight.py` (ffmpeg only): opening slowed to fill 20 s with 0.25 s dissolves, fight = hard cuts on the beat grid (librosa), white 'fade in' flashes on impact cuts, SFX via adelay/amix, loudnorm -14.
- SFX: user's real teleport sound (eagle cry from his mp3). First cleaning chains DULLED it (measured centroid 3321 -> 3175 Hz); fix = light denoise + presence/air boost + exciter (centroid 5182 Hz). Check brightness numerically after any 'clean'.

## Why the user called it trash (all four ticked): fight feels fake, can't see the hero, music/sync, looks too AI
- LTX gives *effects shots* (explosions, lightning, flames), not choreography: no real hero-vs-villain exchange, no cause and effect between shots.
- Hero identity drifts between shots (face in fireball shot, outfit in teleport-behind shot); wide shots tiny/chaotic; minion mob shots unreadable.
- Beat cuts landed within 0.03 s of the grid but on soft beats: AI-composed tracks have weak/jittery beats; the drop idea was good but cuts were not on hits.
- Morphing/warping in 3 s anime clips.

## Do differently next time
1. Storyboard 1 clear action per shot with continuity (A attacks -> B reacts -> counter) BEFORE generating; use close/medium shots of ONE hero design; avoid mobs and 6-subject wide shots.
2. Keep first/last-frame pairs (`--image A 0 1.0 --image B 96 1.0`) so an attack visibly connects with the hit shot.
3. Generate eyes/emblems at close-up only; for wide shots add the glow in the edit instead.
4. Build the cut around real percussive hits: onset-detect the track and put cuts on the strongest onsets, not on a tempo grid.
5. Ask the user to see the hero + villain stills AND a 10 s animatic first; they are away-prone, so review via contact sheets.
6. Longer takes of fewer shots beat many 1.2 s fragments.
- Copyright note given to user: Sharingan/Rinnegan/Momoshiki-style designs are fan content; OK for a private/presentation clip, grey area if published. Never download copyrighted audio/images; use files the user supplies.

---
name: studio-voice-clone
description: Make a realistic, accent-correct cloned voice of the user with Fish Audio in the local content-studio MCP, with no awkward pauses. Use for any narration/voiceover in the user's own voice (reels, films, avatar videos), or when the voice sounds flat, wrong accent, mispronounced, or has gaps.
---

# Studio voice clone (Fish Audio, Hinglish speaker)

Hard-won rules from a 20-hour film build. Follow in order.

## 1. Reference + transcript (the #1 cause of "wrong accent")
- Cut 3 candidate 18 s references from the user's real footage audio (clean, speech-dense, low noise). Score by long-term-average-spectrum distance to the whole footage; keep the best 3 (`ref_A/B/C.wav`).
- **Never trust Fish's auto-transcribe on Hindi/Hinglish references** — it garbles them (e.g. "9 Oh o pa I 2010…") and the clone drifts to a neutral accent. Run `mcp__content-studio__transcribe` with `language="hi"` on each reference and pass the result as `ref_text` explicitly.
- Always `generate_speech(engine="fish", ref_audio=<ref>, ref_text=<hindi transcript>)`. Write the English lines in **normal English spelling** (the user chose this; phonetic Devanagari was tested but not needed).
- Let the user A/B the references blind (send one mp3: current, A, B, C). The user picked ref_C.

## 2. Never pitch-shift
Pitch-shifting to "sound deeper" sounds inhuman (user: "not even human"). Only EQ warmth is acceptable: `equalizer=f=100:g=5, f=220:g=3, f=3200:g=-2.5, highshelf=f=6500:g=-4, acompressor, loudnorm=-18`.

## 3. Remove awkward pauses BEFORE lip-sync
Fish leaves 0.5–1.6 s gaps mid-sentence and sometimes stretches a word ("waaatch"). Run `scripts/tighten.py in.wav out.wav 0.20 0.10` on the **raw Fish wav** (threshold = p95 − 16 dB; do it on raw, not loudness-normalised audio, or the floor hides the gaps). It shrinks silences > 0.2 s to 0.1 s and prints the new phrase spans — cut lip-sync windows at those gap midpoints. If a line is still drawn out (energy envelope smeared, not silent), regenerate it with different punctuation/speed (1.05–1.1) instead of editing.
- Apply the warm EQ **after** tightening.
- Verify each line with the tool's ASR (`word_error_rate` 0). Brand words may need respelling ("Hyper Frames").

## 4. Checklist
raw → ASR ok → tighten → warm EQ → slice by phrase → (lip-sync) ; slices must start/end in silence so crossfades are invisible.

## 5. Cleanliness (user: 'audio breaks and background noise') - measured 2026-10-12
- The old chain (tighten -> warm EQ with acompressor -> loudnorm) **over-compressed** the voice: soft sounds only -8.8 dB below speech (raw Fish: -18 dB) and +6.5 dB low rumble -> reads as noise/pumping. 
- Use: raw Fish -> tighten.py -> `highpass=f=90, equalizer f=160 g=+1.5` -> peak-normalise -3 dB. **No compressor, no per-line loudnorm.** Loudnorm once, on the final film master only.
- Never cut a line mid-sentence: end clips at a phrase gap and add a 40-60 ms fade-out. Check with abrupt-jump count + 'quiet-frame noise' (target <= -15 dB rel. speech) before sending.
- Feed the SAME clean audio to LTX a2v (mouth follows it); LTX mouth leads the audio by ~0.29 s -> advance audio by 0.29 s.

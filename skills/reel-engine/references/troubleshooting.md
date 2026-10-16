# Troubleshooting

## Voice sounds muffled or dull
Audio was downsampled. LatentSync resamples to 16 kHz for its Whisper pass and
muxes that copy back. Take the *video* from LatentSync but re-attach the voice
model's original 44.1 kHz wav at assembly. Verify by comparing energy above
8 kHz against the source wav — a correct mux matches it.

## Voice does not sound like the user
Check `ref_text` matches the reference audio phonetically. The user speaks
Hinglish; the transcript must be Devanagari, not an English translation.
Chatterbox ignores `ref_text` entirely, which is why it loses the accent.

## The face is a different person
An image-to-video model was used. Switch to lip-sync on real footage. This is
not tunable — guidance changes did not fix it and made it worse.

## Captions overlap on one frame
`between(t, a, b)` is inclusive at both ends, so adjacent captions both draw on
the boundary frame. Subtract ~0.05s from each end, and enforce a
non-overlapping sequence before building the filter chain.

## Every b-roll scene comes back empty
Almost always `page_size > 20` against Openverse (HTTP 401). See `broll.md`.

## Downloads stall at 0 B/s
Xet. Set `HF_HUB_DISABLE_XET=1`. Do not SIGSTOP a running download to free
bandwidth — sockets die and the process spins without resuming. Kill and
re-run; `hf` resumes from `.incomplete` files.

## A model appears to use far more RAM than its weights
MLX grows into free RAM and caches. Cap with `--mlx-limit N`; output is
unchanged. Note the cap only bounds MLX — torch, InsightFace and frame buffers
sit outside it, so the floor is higher than the cap.

## `git`, `clang` or `/usr/bin/python3` refuse to run
The Xcode licence was never accepted. Work around with tarballs and prebuilt
wheels, or have the user run `sudo xcodebuild -license accept`.

## LatentSync import errors on Apple Silicon
Already patched in this install, but if rebuilding:
- `decord` has no arm64 wheel — `shims/decord/` provides a real `AudioReader`
  on librosa (`read_audio` needs it) and a raising stub for `VideoReader`.
- `cuda_to_int` rejects non-CUDA devices — return `-1` (InsightFace CPU ctx).
- `mask.png` resolves relative to CWD — run with CWD=`latentsync-mlx` and
  symlink `latentsync -> upstream/latentsync`.
- Whisper shells out to a literal `ffmpeg` — put `bin/ffmpeg` on PATH.
- Upstream has no `setup.py`; it goes on PYTHONPATH.
`scripts/lipsync.py` handles all of this.

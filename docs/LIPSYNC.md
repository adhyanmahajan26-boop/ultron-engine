# Lip-sync (LatentSync) backends

Goal: a real-footage clone whose mouth follows the cloned voice as well as HeyGen does, in one uncut take of
any length. Everything lives in `src/lipsync/`.

| file | job |
|---|---|
| `backends.py` | one `Job`, three backends: `official` (ByteDance PyTorch), `mlx` (unofficial port in `$ULTRON_AVATAR`), `cloud`. Standard library only. |
| `official_runner.py` | upstream inference (pinned commit `a229c39`) made portable: CUDA, Apple MPS, CPU. Reference-frame modes, original-rate audio, per-chunk audio sensitivity, tensor dumps. |
| `avatar_shim.py` | drop-in for `$ULTRON_AVATAR/scripts/lipsync.py` so the studio server's call works unchanged on Linux. Re-execs under `$ULTRON_LATENTSYNC_PY`. |
| `refmodes.py` | which frame the model sees as its unmasked reference (below). |
| `qa.py` | silent-audio control, voice-follow, voice-driven delta, ArcFace identity, SyncNet confidence, contact sheets, PASS/FAIL. |
| `parity.py` + `mlx_adapter.py` | find where the MLX port diverges from official: weights audit, teacher-forced stage compare. |
| `clone.py` | the whole uncut-clone workflow: base check, voice, settings sweep, full render, silent control, QA report. |
| `run.py` | one job from the command line, optional `--silent-control --qa`. |

Install the official code on any machine: `scripts/setup_latentsync_official.sh [--device cuda|mps] [--versions 1.6,1.5] [--yes-download]`
(prints sizes, downloads only with `--yes-download`; Blackwell GPUs get torch>=2.7 automatically). Then `source $ULTRON_LATENTSYNC_DIR/ultron-latentsync.env`.

## Why the MLX 1.6 run ignored the voice, and what this changes

1. **Reference leak.** Upstream *training* (`latentsync/data/unet_dataset.py`) takes the reference from a different,
   non-overlapping 16-frame window of the same video. Upstream *inference* passes the very frame being redrawn
   (`ref_pixel_values` = target). If the source mouth is already moving, the model can copy it.
   `--ref-mode window` restores the training setup: each chunk gets a whole other 16-frame window, the one with the most
   closed mouth. `fixed` (one closed-mouth frame) and `shift` match the old MLX env patch (`LS_REF_MODE`).
   The paste-back always uses the target frame (upstream uses the ref, harmless only in `same`).
2. **Port bugs are now measurable.** Upstream loads the UNet with `strict=False`, so a port that drops the audio
   cross-attention (`attn2`) still runs and silently ignores audio. `official_runner` refuses such a checkpoint, and
   `parity.py weights` finds missing/mismatched tensors in the MLX weights by value, whatever the names or layout.
   Every official run logs `audio_sensitivity` = ‖ε(voice) − ε(silence)‖ / ‖ε(silence)‖ per chunk; near 0 means deaf.
3. **Same noise for all frames.** Upstream draws one noise frame and repeats it across the clip (less flicker). The
   parity dump saves it so the port can be checked against it.

## Studio plan (Mac, one heavy model at a time)

```sh
scripts/setup_latentsync_official.sh --device mps --versions 1.6 --yes-download   # after Addy approves the download
source ~/ultron-work/latentsync/ultron-latentsync.env
# A. official reference on the old calibration clip + its silent control
python src/lipsync/run.py --backend official --video cal.mp4 --audio cal.wav --out ls/off16.mp4 --silent-control --qa
# B. weights audit of the MLX 1.6 port
python src/lipsync/parity.py weights ~/ultron-work/latentsync/checkpoints/latentsync_unet.pt ~/ultron-work/avatar/latentsync-mlx/<1.6 weights>
# C. deterministic dump, then wire mlx_adapter.py (python src/lipsync/mlx_adapter.py inspect) and compare
python src/lipsync/official_runner.py --video cal2s.mp4 --audio cal2s.wav --out ls/par.mp4 --parity --dump par/official
python src/lipsync/parity.py teacher par/official par/mlx && python src/lipsync/parity.py compare par/official par/mlx
# D. the clone
python src/lipsync/clone.py --clip addy.mp4 --voice addy_fish.wav --out-dir work/clone --sweep quick
```

Keep whichever backend matches official quality (MLX only if parity passes and it is faster). Judge every result on
the full-face contact sheet, not only the numbers: metric-tuned settings drew a cartoon "duck pout" before.

## Quality rules carried over (skills/studio-face-lipsync-qa)

- Best source: steady, frontal, face 400+ px wide, mouth closed or calm. Moving/talking sources need `--ref-mode window`.
- Voice must not be longer than the footage (upstream ping-pongs the video otherwise, which reads as a cut).
- Guidance: upstream 1.5 under-articulates long takes; the sweep tries 1.5 / 2.0 / 2.5 (full sweep adds 3.0).
- Always the silent-audio control. PASS needs voice corr ≥ 0.35 at |lag| ≤ 80 ms, speech-vs-silent mouth corr ≤ 0.8,
  ArcFace ≥ 0.80. Starting thresholds from the Oct/Nov runs; tune with Addy's eye.

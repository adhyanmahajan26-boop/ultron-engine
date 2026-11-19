# Linux + NVIDIA GPU (rented box, one command)

Run the whole studio on a rented Linux GPU (built for an RTX 5090, 32 GB) and use it from the Claude app through the
same **content-studio** connector as on the Mac. Boxes that are deleted every day are the normal case: the setup is
one command, safe to re-run, and skips everything that is already there.

## The one command

On a fresh Ubuntu 22.04/24.04 box with the NVIDIA driver (any "PyTorch/CUDA" image on RunPod, Vast, Lambda, etc.):

```bash
curl -fsSL https://raw.githubusercontent.com/adhyanmahajan26-boop/ultron-engine/main/linux/bootstrap.sh | HF_TOKEN=hf_xxx bash
```

**Profiles.** The default profile is `voice`: lip-sync and audio only (LatentSync 1.6, Fish S2 Pro, Chatterbox, Whisper,
Qwen3-TTS), about 30 GB of weights. `--profile full` adds LTX-2.5 video, Qwen-Image 2.1, ACE-Step music and HyperFrames
(about 210 GB on the first day). Re-running with `--profile full` later only adds what is missing:

```bash
curl -fsSL https://raw.githubusercontent.com/adhyanmahajan26-boop/ultron-engine/main/linux/bootstrap.sh | HF_TOKEN=hf_xxx bash -s -- --profile full
```

It clones this repo into `$ULTRON_HOME/engine`, then `linux/setup.sh`:

1. installs system packages (ffmpeg, sox, fonts incl. Devanagari, headless-Chrome libs), `uv`, Node 22, `cloudflared`;
2. builds one Python environment per model family with `uv` (the CUDA build of torch is picked for the driver, so
   Blackwell cards get a CUDA 12.8+ build);
3. downloads the weights into a Hugging Face cache on `$ULTRON_HOME` (resumable);
4. lays out the same `avatar/` folder the Mac uses, writes `ultron-linux.env`, renders `runtime/`;
5. starts **cuda-serve** (`:11234`), the **connector** (`:8765`) and an **HTTPS tunnel**, runs `doctor`, and prints:

```
  Claude connector URL (Settings > Connectors > Add custom connector):
      https://<random>.trycloudflare.com/<secret>/mcp
```

Paste that URL into Claude (Settings > Connectors > Add custom connector). The secret in the path is the password:
keep the URL private.

`HF_TOKEN`: a Hugging Face read token, after accepting the licence of `fishaudio/s2-pro` (and, for the full profile,
`Lightricks/LTX-2.5` (gated) and `Qwen/Qwen-Image-2.1`) on huggingface.co.

Options: `--components ltx,asr,latentsync` (install a subset), `--no-start`, `--dry-run` (print the plan),
`--warm ltx-2.5` (load a model right after start). All settings: [linux/config.example.env](../linux/config.example.env)
(copy it to `$ULTRON_HOME/config.env`).

## Daily routine

| Box keeps a disk (RunPod network volume, Vast volume, mounted at `/workspace`) | Box has no persistent disk |
|---|---|
| `$ULTRON_HOME` defaults to `/workspace/ultron`: venvs, weights, connector secret survive. Next day run the same one-liner: it only checks and starts (about a minute). The connector URL stays the same if you use a fixed tunnel. | Set `ULTRON_WEIGHTS_MIRROR` to an rclone bucket (Cloudflare R2 has free egress). Day 1 with `ULTRON_WEIGHTS_MIRROR_PUSH=1` uploads the weights after the HF download; every later day restores them from the bucket in parallel, which is usually much faster than Hugging Face. |

Before deleting a box: `ultron-linux sync-outputs` (copies `outputs/` and `inputs/` to `ULTRON_OUTPUTS_REMOTE`), or
download results with the connector's `download_link` tool.

Fixed connector address: the default quick tunnel gets a new `trycloudflare.com` address on every start (paste the
new URL into Claude each day). For a permanent one, create a Cloudflare tunnel to `http://localhost:8765` and set
`ULTRON_TUNNEL=cloudflare-named`, `CLOUDFLARE_TUNNEL_TOKEN`, `ULTRON_PUBLIC_URL`. On RunPod you can instead expose
port 8765 and set `ULTRON_TUNNEL=runpod`.

## Commands

```
ultron-linux up | down | restart | status | url | doctor | logs [cuda|mcp|tunnel|worker-<model>]
ultron-linux warm ltx-2.5            # load a model now
ultron-linux selftest                # voice profile: Qwen3-TTS speech, Whisper ASR, Fish clone, Chatterbox conversion
ultron-linux selftest all            # + image, 4 s video, music (full profile)
ultron-linux selftest video10        # the 10-second one-take check (241 frames, 1024x576)
ultron-linux sync-outputs | update
```

## Architecture

```
Claude app ──HTTPS──> cloudflared ──> linux/mcp_remote.py :8765  (token gate, /files, /upload)
                                         └─ src/studio/server.py  (the SAME content-studio tools as the Mac)
                                               ├─ HTTP ──> linux/cuda_serve/server.py :11234  (mlx-serve compatible API)
                                               │             └─ ONE worker process at a time, each in its own venv:
                                               │                 ltx.py (LTX-2.5) · qwen_image.py (Qwen-Image 2.1)
                                               │                 acestep.py (ACE-Step 1.5 XL) · qwen3_tts.py
                                               └─ $ULTRON_AVATAR/scripts (same contract as the Mac):
                                                     asr.py (faster-whisper) · voice_clone.py (Fish S2 Pro)
                                                     voiceconvert.py (Chatterbox) · lipsync.py (official LatentSync 1.6)
```

Why this shape: on the Mac every tool, skill and project script talks to mlx-serve on `127.0.0.1:11234`. cuda-serve
answers the same routes (and accepts the Mac model ids as aliases), so the studio server, skills and scripts such as
`gen_img.py` / `gen_music.py` run unchanged. The one-heavy-model rule is enforced by construction: cuda-serve keeps
only one worker process alive and ends it before starting another, which returns VRAM completely.

Studio changes that keep the Mac path identical: `src/studio/server.py` reads optional `CONTENT_STUDIO_PRESETS`,
`ULTRON_ASR_SCRIPT`, `ULTRON_ASR_MODEL`, `CONTENT_STUDIO_NODE_BIN` (unset on the Mac), accepts a finished MP4 path from
the video endpoint besides mlx-serve's raw frames, reports GPU/RAM on Linux, and `generate_video` gained an optional
`last_frame_image`. `ultron.py render` copies `linux/tools/*` (Linux `heavy_guard.sh`) over the Mac tools on Linux.

Connector extras on Linux: `fetch_input(url)` (pull a clip/photo onto the box, Google Drive links work),
`download_link(path)`, `list_inputs()` (also shows the upload URL), `machine_status()`.

## Models on a 32 GB GPU

| Studio role | Mac (MLX) | Linux (CUDA) | Fits 32 GB? | Download |
|---|---|---|---|---|
| video, native speech | LTX-2.5 MLX q8 | `Lightricks/LTX-2.5-Diffusers` (distilled, diffusers) | yes: transformer stored fp8, other parts offloaded; needs ~64 GB+ system RAM | ~72 GB |
| video, best action | MiniMax-H3 8-bit | not installed | **no**: 33B transformer + 32B text encoder (~62 + 48 GB bf16). Needs an 80 GB+ or multi-GPU box | |
| keyframes + edits | Qwen-Image 2.1 / Mage-Flow / Qwen-Edit 2511 | `Qwen/Qwen-Image-2.1` (edits with up to 10 references), optional `-Turbo` for "fast" | yes, with model offload | ~35 GB each |
| music | ACE-Step 1.5 XL Turbo | `ACE-Step/acestep-v15-xl-turbo` + LM 1.7B, official repo | yes (20 GB+ without offload) | ~30 GB |
| voice clone | Fish S2 Pro | `fishaudio/s2-pro`, official fish-speech | yes (needs 24 GB) | ~11 GB |
| voice conversion | Chatterbox MLX | `ResembleAI/chatterbox` (PyTorch) | yes | ~3 GB |
| TTS | Qwen3-TTS 1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | yes | ~4 GB |
| ASR | Whisper large-v3-turbo (mlx) | faster-whisper large-v3-turbo | yes | ~1.6 GB |
| lip-sync | LatentSync 1.5 MLX port | official ByteDance LatentSync 1.6 (owned by `src/lipsync`) | yes | ~7 GB |
| edit/render | HyperFrames, ffmpeg | same | n/a | |

Memory policy (`ULTRON_OFFLOAD=auto`): 70 GB+ VRAM keeps everything on the GPU; 20-70 GB uses model CPU offload
(LTX's transformer stored in fp8); smaller cards use sequential offload (slow). Disk for everything: about 210 GB on
the first day.

Not on Linux yet: `make_reel` (it drives the Mac's local-avatar `make_content.py` / `edit_reel.py`, which are not in
this repo), Hunyuan3D, H3 on a single 32 GB card.

## 10-second one-take clips

LTX needs `num_frames = 8k+1`: at 24 fps, 10 s is **241 frames** (`generate_video(num_frames=241, width=1024,
height=576, first_frame_image=...)`; `last_frame_image` pins the end). That is one generation, no cuts.
`ultron-linux selftest video10` measures time and peak memory for it on the actual box.

## Status: what is tested

- Tested here without a GPU: `linux/tests/test_offline.py` runs cuda-serve with fake workers and the connector over
  streamable HTTP with a real MCP client (14 checks: Mac id aliases, worker switching, SSE progress, H3 refusal,
  unload, token gate, generate_image / generate_video jobs incl. 241 frames + first/last frame, download links,
  upload, machine status). `setup.sh --dry-run` and `ultron-linux up/doctor/down` were run against stubs.
- **Not yet run on a real GPU**: model loading and generation in each worker, the CUDA wheel choice, download sizes,
  speeds and VRAM peaks. Run `ultron-linux doctor` and `ultron-linux selftest all` on the first box and fix what fails.

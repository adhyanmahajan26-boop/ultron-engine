# Models

Run `python3 ultron.py models` for the live table. Sizes are from our disks; `setup` asks the Hugging Face API for the real size before you confirm. Nothing downloads without a yes.

| key | what | source | notes |
|---|---|---|---|
| video_ltx | LTX-2.5 image-to-video + native speech | `mlx-community/ltx-2.5-mlx-ditq8` (transformer only) + `mlx-community/ltx-2.5-mlx-q8` (without transformers); runner github.com/xocialize/ltx-2-mlx | about 43 GB; peak RAM 52 GB (4 s), 85 GB (10 s); about 2 min per 4 s |
| video_h3 | MiniMax-H3 8-bit | `ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit` via mlx-serve | 70 GB; best action choreography in our tests; 13-20 min per 4 s |
| image_qwen | Qwen-Image 2.1 8-bit | `ddalcu/Qwen-Image-2.1-MLX-Serve-8bit` | 17.6 GB, about 22 GB resident |
| image_fast_edit | Mage-Flow Edit Turbo | `ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit` | 9.7 GB, about 20 s per edit; weaker for two-reference composites |
| image_edit_qwen2511 | Qwen-Image-Edit 2511 8-bit via `mflux==0.22.0` | `mlx-community/qwen-image-edit-2511-8bit` | 5-8 min per edit, 39 GB peak; best at keeping identity |
| music_ace | ACE-Step 1.5 XL Turbo | `ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit` | 5.4 GB |
| voice_fish | Fish Audio S2 Pro bf16 | `mlx-community/fish-audio-s2-pro-bf16` | research licence |
| voice_chatterbox | Chatterbox multilingual v3 + S3 tokenizer | `mlx-community/chatterbox-multilingual-v3`, `mlx-community/S3TokenizerV2` | MIT; used to turn LTX's native speech into a cloned voice |
| tts_qwen3 | Qwen3-TTS 1.7B | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | |
| asr_whisper | Whisper large-v3-turbo | `mlx-community/whisper-large-v3-turbo` | |
| lipsync_latentsync | LatentSync 1.5 | `ByteDance/LatentSync-1.5` weights + an MLX port | the MLX port is not in this repo; optional |
| model3d_hunyuan | Hunyuan3D 2.1 | `ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit` | optional |

Models we tried and dropped (so you do not repeat it): LongCat (quality and speed), Wan 2.2 and InfiniteTalk (replaced by LTX), a 4-bit H3 (lower quality), LatentSync 1.6 (we kept 1.5). LTX audio-to-video mode hallucinated props in 4 of 5 tests, so we use native speech plus voice conversion.

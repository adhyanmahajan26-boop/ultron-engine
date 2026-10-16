# Quick start

1. **Prerequisites** (`python3 ultron.py doctor` tells you what is missing): `ffmpeg`, `uv`, `node` (for HyperFrames), Python 3.11+, git. For the studio server (image, music, H3) also [mlx-serve](https://github.com/ddalcu/mlx-serve) (menu-bar app MLX Core).
2. `python3 ultron.py setup`. It detects your hardware, asks local / cloud / mixed, writes `ultron.config.json`, renders `runtime/` (all scripts with your paths filled in) and `ultron.env`, then asks per model whether to download. `source ultron.env` afterwards.
3. **First video (LTX, about 2 min):**
   ```bash
   source ultron.env
   runtime/tools/heavy_guard.sh && cd "$ULTRON_LTX/ltx-2-mlx-xoc" && uv run ltx-2-mlx generate --distilled --model "$ULTRON_LTX/pack" \
     --prompt "The man on the left raises a sword and swings it down as sparks fly toward the camera. Locked camera." \
     --image /path/to/your_1024x576.png --height 576 --width 1024 --frames 97 --frame-rate 24 --seed 1 -o out.mp4
   ```
   Frames must be 8k+1 (73 = 3 s, 97 = 4 s, 241 = 10 s). Read [PROMPTING.md](PROMPTING.md) first: prompt style decides whether the action reads.
4. **Make stills** with Qwen-Image (via the studio server tool `generate_image`), one clear action per shot.
5. **Cut it**: `src/pipelines/build_priest.py` is a complete example (cuts on the music's strongest hits, white flashes, SFX, loudnorm). Copy it and change the shot list.
6. **QA**: `runtime/tools/clip_qa.py`, `face_sim.py`, `sync_metric.py` (see [ARCHITECTURE.md](ARCHITECTURE.md)).

Always run one heavy job at a time: [MEMORY_AND_QUEUE.md](MEMORY_AND_QUEUE.md).

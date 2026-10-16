# Memory and the one-heavy-job rule

On 2026-10-03 starting several heavy jobs at once crashed the Mac, and a LatentSync run next to a Qwen edit pushed use to 86 GB. Since then:
1. **One heavy job at a time**: LTX, Qwen-edit, LatentSync, HyperFrames render, any studio model. The studio server queues extra jobs (`CONTENT_STUDIO_MAX_HEAVY=1`).
2. Before a heavy job call `free_memory` (unloads studio models) and run `runtime/tools/heavy_guard.sh && <job>`.
3. Long queues are chained scripts whose lines each wait for the guard (`while ! heavy_guard.sh; do sleep 15; done; <job>`) and append a status line when done.
4. Kill LatentSync by name (`scripts/lipsync.py` and `scripts/inference.py`).
5. If mlx-serve dies (`ConnectError`), restart it: `mlx-serve serve --host 127.0.0.1 --port 11234`.

# Cloud GPU setup (experimental)

`python3 ultron.py setup` lets each component run `local`, `cloud` or `skip` (mode `mixed` = choose per component). For cloud components you give an **endpoint URL** and the **name** of the environment variable that holds the API key. The key itself is never stored or committed.

## Protocol (what `cloud/backends.py` speaks)
```
POST {url}/run          {"role": "video.ltx", "params": {"prompt": "...", "image_url": "...", "frames": 97, "seed": 1}}   -> {"job_id": "abc"}
GET  {url}/jobs/abc     -> {"status": "queued|running|done|error", "result_url": "https://...", "error": null}
Authorization: Bearer $YOUR_KEY_ENV
```
Roles used in the catalog: `video.ltx`, `video.h3`, `image.qwen`, `image.edit`, `image.fastedit`, `music.ace`, `voice.fish`, `voice.convert`, `voice.tts`, `asr.whisper`, `lipsync.latentsync`, `3d.hunyuan`.

## What is NOT included
`cloud/worker_template.py` implements the protocol with no model code. You must plug in CUDA builds of the models (for LTX-2.5 the upstream Lightricks release) on your provider (RunPod, Modal, Lambda, your own box). We have not run this end to end on any provider. Treat it as a design to build on, and cloud speed, cost and quality as unmeasured.
Call from Python: `python3 cloud/backends.py video_ltx '{"prompt": "...", "frames": 97}'`.

"""cuda-serve: an mlx-serve compatible HTTP API on Linux + NVIDIA.

The Mac studio talks to mlx-serve on 127.0.0.1:11234 (studio MCP server, skills, project scripts such as gen_img.py
and gen_music.py, heavy_guard.sh). cuda-serve answers the same routes on the same port with CUDA builds of the models,
so all of that runs on a rented GPU without changes:

  GET  /v1/models                      ids, capabilities, bytes_on_disk, bytes_resident, state
  POST /v1/unload-model                {"model": id}  (or no body: unload whatever is loaded)
  POST /v1/images/generations          {"model","prompt","size","steps","seed"}            -> {"data":[{"b64_json"}]}
  POST /v1/images/edits                multipart: model, prompt, size, image[] (1-10)       -> {"data":[{"b64_json"}]}
  POST /v1/audio/music-generations     {"model","prompt","lyrics","instrumental","duration_seconds","bpm","vocal_language","seed"} -> WAV
  POST /v1/audio/speech                {"model","input","ref_audio"(base64),"ref_text","speed"} -> WAV
  POST /v1/video/generations           {"model","prompt","width","height","num_frames","seed","first_frame_image"(b64),
                                        "last_frame_image"(b64),"frame_rate","stream"}
        stream=true -> SSE: {"type":"progress",...} ... {"type":"complete","format":"mp4","path":...}
        (mlx-serve sends raw rgb8 frames; a 10 s 1024x576 clip would be ~570 MB of base64, so cuda-serve hands back
         the path of the finished MP4 instead. src/studio/server.py accepts both.)

Each model runs in its own worker process and venv (see models.py, workers/). At most ONE worker is alive: asking
for another model ends the old process first, so VRAM is returned in full. Requests are served one at a time.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CUDA_SERVE_OUT", M.HOME / "cuda-serve" / "out"))
LOGS = Path(os.environ.get("ULTRON_LOGS", M.HOME / "logs"))
IDLE_UNLOAD_S = float(os.environ.get("CUDA_SERVE_IDLE_UNLOAD", 0))  # 0 = keep the last model loaded
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


class Worker:
    def __init__(self, key: str, proc: asyncio.subprocess.Process, log):
        self.key, self.proc, self.log, self.seq = key, proc, log, 0
        self.started = time.time()

    async def call(self, op: str, params: dict | None, on_progress=None) -> dict:
        self.seq += 1
        rid = f"r{self.seq}"
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write((json.dumps({"id": rid, "op": op, "params": params or {}}) + "\n").encode())
        await self.proc.stdin.drain()
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                rc = await self.proc.wait()
                raise RuntimeError(f"{self.key} worker exited ({rc}); see {LOGS / f'worker-{self.key}.log'}")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") != rid:
                continue
            if msg["type"] == "progress":
                if on_progress:
                    await on_progress(msg)
            elif msg["type"] == "result":
                return msg["result"]
            else:
                raise RuntimeError(msg.get("error") or "worker error")


class Manager:
    """Keeps at most one model worker alive and serves one request at a time."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.cur: Worker | None = None
        self.last_used = time.time()

    async def stop(self) -> str | None:
        w, self.cur = self.cur, None
        if not w:
            return None
        try:
            if w.proc.stdin:
                w.proc.stdin.close()
            await asyncio.wait_for(w.proc.wait(), 15)
        except Exception:
            w.proc.kill()
            await w.proc.wait()
        w.log.close()
        return w.key

    async def ensure(self, key: str) -> Worker:
        if self.cur and self.cur.key == key and self.cur.proc.returncode is None:
            return self.cur
        await self.stop()
        m = M.MODELS[key]
        if not Path(m["python"]).exists():
            raise HTTPException(503, f"{key}: its environment is not installed ({m['python']}). Run linux/setup.sh.")
        env = {**os.environ, **m.get("env", {}), "PYTHONPATH": f"{HERE}:{os.environ.get('PYTHONPATH', '')}",
               "PYTHONUNBUFFERED": "1"}
        log = open(LOGS / f"worker-{key}.log", "a")
        log.write(f"\n==== start {time.ctime()} ====\n")
        log.flush()
        proc = await asyncio.create_subprocess_exec(m["python"], str(M.WORKERS / m["worker"]), env=env,  # abs path wins
                                                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                    stderr=log, limit=2**26)
        self.cur = Worker(key, proc, log)
        await self.cur.call("load", None)
        return self.cur

    async def run(self, model_id: str, params: dict, on_progress=None) -> dict:
        key = pick(model_id)
        async with self.lock:
            w = await self.ensure(key)
            try:
                return await w.call("run", params, on_progress)
            finally:
                self.last_used = time.time()


def pick(model_id: str | None) -> str:
    if model_id in M.UNSUPPORTED:
        raise HTTPException(400, M.UNSUPPORTED[model_id])
    key = M.resolve(model_id or "")
    if not key:
        raise HTTPException(404, f"unknown model {model_id!r}; known: {sorted(M.MODELS)}")
    fb = M.MODELS[key].get("fallback")
    if fb and not M.installed(key) and M.installed(fb):
        return fb
    return key


mgr = Manager()
app = FastAPI(title="cuda-serve (mlx-serve compatible)")


@app.on_event("startup")
async def _idle_reaper():
    if IDLE_UNLOAD_S <= 0:
        return

    async def loop():
        while True:
            await asyncio.sleep(30)
            if mgr.cur and not mgr.lock.locked() and time.time() - mgr.last_used > IDLE_UNLOAD_S:
                async with mgr.lock:
                    await mgr.stop()
    asyncio.create_task(loop())


def _new_out(ext: str) -> Path:
    return OUT / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.{ext}"


def _b64_to_file(b64: str, suffix: str) -> str:
    if "," in b64[:100] and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=OUT)
    f.write(base64.b64decode(b64))
    f.close()
    return f.name


@app.exception_handler(RuntimeError)
async def _rt(_, e: RuntimeError):
    return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {"ok": True, "loaded": mgr.cur.key if mgr.cur else None, "busy": mgr.lock.locked()}


@app.get("/v1/models")
async def list_models():
    data = []
    for k, m in M.MODELS.items():
        disk = M.bytes_on_disk(k)
        loaded = bool(mgr.cur and mgr.cur.key == k and mgr.cur.proc.returncode is None)
        data.append({"id": k, "object": "model", "capabilities": m["capabilities"], "aliases": m["aliases"],
                     "bytes_on_disk": disk, "bytes_resident": disk if loaded else 0, "loaded": loaded,
                     "state": "loaded" if loaded else "unloaded", "installed": M.installed(k)})
    return {"object": "list", "data": data}


@app.post("/v1/unload-model")
async def unload(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}
    want = M.resolve(body.get("model") or "") if body.get("model") else None
    async with mgr.lock:
        if mgr.cur and (want is None or mgr.cur.key == want):
            return {"unloaded": await mgr.stop()}
    return {"unloaded": None}


@app.post("/v1/load-model")
async def load(body: dict):
    """Warm a model now (e.g. right after boot) so the first real request does not pay the load time."""
    key = pick(body.get("model"))
    t = time.time()
    async with mgr.lock:
        await mgr.ensure(key)
    return {"loaded": key, "seconds": round(time.time() - t, 1)}


@app.post("/v1/images/generations")
async def images(body: dict):
    p = {"prompt": body["prompt"], "size": body.get("size"), "steps": body.get("steps"), "seed": body.get("seed"),
         "negative_prompt": body.get("negative_prompt"), "out": str(_new_out("png"))}
    r = await mgr.run(body.get("model") or "qwen-image-2.1", p)
    return {"created": int(time.time()), "data": [{"b64_json": base64.b64encode(Path(r["path"]).read_bytes()).decode(),
                                                    "path": r["path"]}]}


@app.post("/v1/images/edits")
async def image_edits(req: Request):
    form = await req.form()
    refs = []
    for key in ("image[]", "image"):
        for f in form.getlist(key):
            if hasattr(f, "read"):
                dst = _new_out(Path(f.filename or "ref.png").suffix.lstrip(".") or "png")
                dst.write_bytes(await f.read())
                refs.append(str(dst))
    if not refs:
        raise HTTPException(400, "no image[] given")
    p = {"prompt": form.get("prompt"), "size": form.get("size"), "images": refs[:10], "seed": form.get("seed"),
         "out": str(_new_out("png"))}
    if form.get("steps"):
        p["steps"] = int(form.get("steps"))
    r = await mgr.run(form.get("model") or "qwen-image-2.1", p)
    return {"created": int(time.time()), "data": [{"b64_json": base64.b64encode(Path(r["path"]).read_bytes()).decode(),
                                                    "path": r["path"]}]}


@app.post("/v1/audio/music-generations")
async def music(body: dict):
    p = {k: body.get(k) for k in ("prompt", "lyrics", "instrumental", "duration_seconds", "bpm", "vocal_language", "seed", "steps")}
    p["out"] = str(_new_out("wav"))
    r = await mgr.run(body.get("model") or "ace-step-1.5-xl-turbo", p)
    return Response(Path(r["path"]).read_bytes(), media_type="audio/wav", headers={"X-Output-Path": r["path"]})


@app.post("/v1/audio/speech")
async def speech(body: dict):
    p = {"input": body["input"], "ref_text": body.get("ref_text"), "speed": body.get("speed"),
         "language": body.get("language"), "out": str(_new_out("wav"))}
    if body.get("ref_audio"):
        p["ref_audio"] = _b64_to_file(body["ref_audio"], ".wav") if len(body["ref_audio"]) > 1024 else body["ref_audio"]
    r = await mgr.run(body.get("model") or "qwen3-tts-1.7b", p)
    return Response(Path(r["path"]).read_bytes(), media_type="audio/wav", headers={"X-Output-Path": r["path"]})


def _video_params(body: dict) -> dict:
    p: dict[str, Any] = {"prompt": body["prompt"], "width": body.get("width", 1024), "height": body.get("height", 576),
                         "num_frames": body.get("num_frames", 97), "frame_rate": body.get("frame_rate", 24),
                         "seed": body.get("seed"), "steps": body.get("steps"), "negative_prompt": body.get("negative_prompt"),
                         "image_strength": body.get("image_strength", 1.0), "out": str(_new_out("mp4"))}
    if body.get("distilled") is not None:
        p["distilled"] = bool(body["distilled"])
    for k in ("first_frame_image", "last_frame_image"):
        v = body.get(k)
        if v:
            p[k] = v if (len(v) < 1024 and Path(v).is_file()) else _b64_to_file(v, ".png")
    return p


@app.post("/v1/video/generations")
async def video(body: dict):
    model = body.get("model") or "ltx-2.5"
    pick(model)  # fail fast (400/404) before opening a stream
    p = _video_params(body)
    if not body.get("stream"):
        r = await mgr.run(model, p)
        return {"type": "complete", **r}

    q: asyncio.Queue = asyncio.Queue()

    async def prog(msg):
        await q.put({"type": "progress", **{k: v for k, v in msg.items() if k not in ("id", "type")}})

    async def job():
        try:
            r = await mgr.run(model, p, prog)
            await q.put({"type": "complete", **r})
        except Exception as e:
            await q.put({"type": "error", "message": getattr(e, "detail", None) or str(e)})
        await q.put(None)

    async def gen():
        task = asyncio.create_task(job())
        if mgr.lock.locked():
            yield f"data: {json.dumps({'type': 'progress', 'message': 'queued behind another job'})}\n\n"
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), 15)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if ev is None:
                break
            yield f"data: {json.dumps(ev)}\n\n"
        await task

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("CUDA_SERVE_HOST", "127.0.0.1"), port=int(os.environ.get("CUDA_SERVE_PORT", 11234)))

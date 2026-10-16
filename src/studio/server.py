# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp>=1.2,<2", "httpx>=0.27"]
# ///
"""Content Studio MCP: local image / video / music / voice / lip-sync / reel generation.

Everything runs on this Mac:
  mlx-serve (127.0.0.1:11234)  image, image edit, video, music (ACE-Step), TTS (Qwen3-TTS)
  local-avatar pipeline         Fish S2 Pro voice clone, LatentSync lip-sync, full reels
  HyperFrames CLI (npx)         render HTML compositions to MP4

Generation is slow, so every generator runs as a detached worker process
(`server.py --job <id>`) whose state lives in outputs/jobs/<id>.json. Jobs therefore
survive the MCP server being restarted or disconnected by the client. A tool waits
at most MAX_WAIT seconds (default 45, under the common 60 s MCP request timeout)
and otherwise returns a job id to check with `job_status`.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP

HOME = Path.home()
STUDIO = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CONTENT_STUDIO_OUT", STUDIO / "outputs"))
MLX = os.environ.get("MLX_SERVE_URL", "http://127.0.0.1:11234")
AVATAR = Path(os.environ.get("ULTRON_AVATAR", HOME / "ultron-work/avatar"))
AVATAR_PY = AVATAR / ".venv/bin/python"
VOICE_REF = AVATAR / "inputs/my_voice_ref.wav"
VOICE_REF_TEXT = AVATAR / "inputs/my_voice_ref.txt"
DEFAULT_FOOTAGE = AVATAR / "inputs/source_clip.mp4"
NODE_BIN = HOME / "Library/pnpm"

# Quality presets -> mlx-serve model ids. Anything else passed as `model` is used verbatim.
IMAGE_MODELS = {
    "quality": "ddalcu/Qwen-Image-2.1-MLX-Serve-8bit",
    "fast": "ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit",  # Krea 2 Turbo once downloaded
}
EDIT_MODEL = "ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit"
VIDEO_MODELS = {
    "fast": "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit",  # 4-bit removed to free disk
    "quality": "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit",
}
MUSIC_MODEL = "ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit"
TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
ASR_MODEL = AVATAR / "models/whisper-turbo"      # local Whisper-turbo (mlx-audio)
ASR_SCRIPT = STUDIO / "asr.py"

mcp = FastMCP("content-studio")

# ── jobs ────────────────────────────────────────────────────────────────────

JOBS_DIR = OUT / "jobs"
JOBS_FILE = OUT / "jobs.jsonl"
MAX_WAIT = float(os.environ.get("CONTENT_STUDIO_MAX_WAIT", 45))
WORKERS: dict[str, Any] = {}

# Heavy jobs (model inference, renders) run one at a time by default. Starting many at once
# exhausted memory and crashed the Mac on 2026-10-03; extra jobs now wait in a queue instead.
HEAVY_KINDS = {"image", "image_edit", "video", "music", "speech", "lipsync", "reel", "hyperframes", "asr"}
MAX_HEAVY = max(1, int(os.environ.get("CONTENT_STUDIO_MAX_HEAVY", 1)))
LOCK_DIR = OUT / "locks"


def worker(kind: str):
    """Register the function that does the work for a job kind: async fn(params, job)."""
    def deco(fn):
        WORKERS[kind] = fn
        return fn
    return deco


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (s[:n].rstrip("-") or "untitled")


def _out_path(kind: str, label: str, ext: str) -> Path:
    d = OUT / kind
    d.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return d / f"{stamp}_{_slug(label)}.{ext}"


class Job(dict):
    """Job state that writes itself to disk on every change (progress throttled)."""

    def __init__(self, path: Path, data: dict):
        super().__init__(data)
        self._path = path
        self._last = 0.0

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "progress" and time.monotonic() - self._last < 0.5:
            return
        self.save()

    def save(self):
        self._last = time.monotonic()
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self))
        tmp.replace(self._path)


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{re.sub(r'[^a-f0-9]', '', job_id)}.json"


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_job(job_id: str) -> dict | None:
    p = _job_path(job_id)
    if not p.exists():
        return None
    try:
        job = json.loads(p.read_text())
    except json.JSONDecodeError:
        time.sleep(0.05)
        job = json.loads(p.read_text())
    if job.get("status") == "running" and not _alive(job.get("pid")) and job.get("pid"):
        job["status"] = "error"
        job["error"] = "worker process died unexpectedly; see " + str(p.with_suffix(".log"))
    return job


def _public(job: dict[str, Any]) -> dict[str, Any]:
    j = {k: v for k, v in job.items() if k not in ("params", "pid")}
    if job.get("status") == "running":
        started = dt.datetime.fromisoformat(job["started"])
        j["elapsed_seconds"] = int((dt.datetime.now() - started).total_seconds())
        j["note"] = f"Still running. Call job_status(job_id='{job['job_id']}', wait_seconds=40) to check."
    return j


async def _wait(job_id: str, wait_seconds: float) -> dict:
    deadline = time.monotonic() + max(0.0, min(wait_seconds, MAX_WAIT))
    while True:
        job = _read_job(job_id) or {"job_id": job_id, "status": "error", "error": "job file missing"}
        if job.get("status") != "running" or time.monotonic() >= deadline:
            return _public(job)
        await asyncio.sleep(0.5)


async def _start(kind: str, summary: str, params: dict, wait_seconds: float) -> dict[str, Any]:
    """Spawn a detached worker for this job and wait briefly for it."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:8]
    path = _job_path(job_id)
    job = Job(path, {"job_id": job_id, "kind": kind, "summary": summary, "status": "running",
                     "started": _now(), "progress": "starting", "params": params})
    job.save()
    log = path.with_suffix(".log").open("w")
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--job", job_id],
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, cwd=str(STUDIO))
    log.close()  # the worker records its own pid; writing it here would race its first update
    return await _wait(job_id, wait_seconds)


def _acquire_heavy(job: "Job"):
    """Block until one of MAX_HEAVY slots is free; return the held lock file (released on exit)."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(MAX_HEAVY):
            f = (LOCK_DIR / f"heavy{i}.lock").open("w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except BlockingIOError:
                f.close()
        job["progress"] = "queued: waiting for the running heavy job to finish"
        time.sleep(2)


def _heavy_slots_busy() -> int:
    busy = 0
    for i in range(MAX_HEAVY):
        lf = LOCK_DIR / f"heavy{i}.lock"
        if not lf.exists():
            continue
        with lf.open("w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            except BlockingIOError:
                busy += 1
    return busy


def _run_job(job_id: str) -> None:
    """Worker-process entry point: run one job to completion, recording state on disk."""
    path = _job_path(job_id)
    job = Job(path, json.loads(path.read_text()))
    job["pid"] = os.getpid()

    def on_term(*_):
        child = job.get("child_pgid")
        if child:
            try:
                os.killpg(child, signal.SIGTERM)
            except ProcessLookupError:
                pass
        job["status"] = "cancelled"
        job["finished"] = _now()
        sys.exit(1)

    signal.signal(signal.SIGTERM, on_term)
    lock = _acquire_heavy(job) if job["kind"] in HEAVY_KINDS else None
    job["progress"] = "starting"
    try:
        job["result"] = asyncio.run(WORKERS[job["kind"]](job["params"], job))
        job["status"] = "done"
    except Exception as e:  # surface the real error to the caller
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
    job["finished"] = _now()
    job["progress"] = None
    job.save()
    if lock:
        lock.close()
    with JOBS_FILE.open("a") as f:
        f.write(json.dumps({k: v for k, v in job.items() if k != "params"}) + "\n")


# ── helpers ─────────────────────────────────────────────────────────────────

def _client(timeout: float = 3600) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=MLX, timeout=httpx.Timeout(timeout, connect=10))


async def _post_bytes(path: str, body: dict, out: Path) -> Path:
    async with _client() as c:
        r = await c.post(path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"mlx-serve {path} HTTP {r.status_code}: {r.text[:500]}")
        out.write_bytes(r.content)
    return out


async def _post_sse(path: str, body: dict, job: dict) -> dict:
    """POST with stream:true and return the final `complete` event."""
    body = {**body, "stream": True}
    async with _client() as c:
        async with c.stream("POST", path, json=body) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"mlx-serve {path} HTTP {r.status_code}: {(await r.aread())[:500]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                t = ev.get("type")
                if t == "progress":
                    job["progress"] = {k: v for k, v in ev.items() if k not in ("type", "preview")}
                elif t == "error":
                    raise RuntimeError(ev.get("message") or ev.get("error") or json.dumps(ev)[:500])
                elif t == "complete":
                    return ev
    raise RuntimeError("stream ended without a complete event")


async def _run(cmd: list[str], job: dict, cwd: Path | None = None, env: dict | None = None) -> str:
    """Run a subprocess, keep the last log lines on the job as progress."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, **(env or {})}, start_new_session=True,
    )
    job["child_pgid"] = proc.pid
    tail: list[str] = []
    assert proc.stdout
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        for part in line.split("\r"):
            if part.strip():
                tail.append(part.strip())
        tail = tail[-30:]
        job["progress"] = tail[-1][:300] if tail else None
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} {cmd[1] if len(cmd) > 1 else ''} exited {rc}:\n" + "\n".join(tail[-15:]))
    return "\n".join(tail)


def _resolve(preset_or_id: str | None, presets: dict[str, str], default: str) -> str:
    key = preset_or_id or default
    return presets.get(key, key)


def _b64_file(p: str | Path) -> str:
    return base64.b64encode(Path(p).expanduser().read_bytes()).decode()


def _require_file(p: str | Path, what: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def _norm_words(t: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", t.lower().replace("-", " ")).split()


def _wer(ref: str, hyp: str) -> float:
    """Word error rate of hyp against ref (0 = identical)."""
    r, h = _norm_words(ref), _norm_words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return round(d[len(h)] / len(r), 3)


async def _asr(audio: str | Path, job: dict, language: str = "en") -> dict:
    """Transcribe a local audio file with Whisper-turbo; returns {"text", "words"}."""
    if not ASR_MODEL.exists():
        raise FileNotFoundError(f"speech-recognition model missing: {ASR_MODEL}")
    saved = job.get("progress")
    out = await _run([str(AVATAR_PY), str(ASR_SCRIPT), str(audio), "--language", language], job, cwd=AVATAR)
    line = next((l for l in reversed(out.splitlines()) if l.startswith("ASR_JSON:")), None)
    if not line:
        raise RuntimeError("speech recognition produced no result:\n" + out[-500:])
    job["progress"] = saved
    return json.loads(line[len("ASR_JSON:"):])


# ── status & jobs ───────────────────────────────────────────────────────────

@mcp.tool()
async def studio_status() -> dict:
    """Show what this machine can generate right now: models per category (with
    whether each is downloaded), the voice-clone / lip-sync pipeline, HyperFrames,
    and free memory. Call this first when choosing options to offer the user."""
    status: dict[str, Any] = {"outputs_dir": str(OUT)}
    try:
        async with _client(10) as c:
            models = (await c.get("/v1/models")).json()["data"]
        cats: dict[str, list] = {}
        for m in models:
            caps = m.get("capabilities") or []
            downloaded = bool(m.get("bytes_on_disk"))
            entry = {
                "id": m["id"], "downloaded": downloaded,
                "size_gb": round((m.get("bytes_on_disk") or 0) / 1e9, 1),
                "state": m.get("state"),
            }
            for cap in caps:
                if cap in ("image", "video", "music", "audio", "3d", "chat"):
                    cats.setdefault(cap, []).append(entry)
        status["mlx_serve"] = {"up": True, "models": cats}
    except Exception as e:
        status["mlx_serve"] = {"up": False, "error": str(e),
                               "fix": "Open the MLX Core app so mlx-serve listens on 127.0.0.1:11234."}
    status["presets"] = {
        "image": IMAGE_MODELS, "image_edit": EDIT_MODEL, "video": VIDEO_MODELS,
        "music": MUSIC_MODEL, "tts": TTS_MODEL,
    }
    status["voice_clone_fish"] = {
        "available": AVATAR_PY.exists() and VOICE_REF.exists(),
        "voice_reference": str(VOICE_REF),
    }
    status["lipsync_latentsync"] = {
        "available": AVATAR_PY.exists(), "default_footage": str(DEFAULT_FOOTAGE),
        "footage_exists": DEFAULT_FOOTAGE.exists(),
    }
    status["hyperframes"] = {"node": shutil.which("node", path=f"{NODE_BIN}:{os.environ.get('PATH','')}")}
    status["speech_recognition"] = {"available": ASR_MODEL.exists() and AVATAR_PY.exists(),
                                    "model": str(ASR_MODEL), "tool": "transcribe",
                                    "note": "generate_speech auto-verifies its output with this"}
    status["heavy_jobs"] = {"max_at_once": MAX_HEAVY, "slots_busy": _heavy_slots_busy(),
                            "rule": "extra jobs queue; do not submit more than one heavy job at a time"}
    try:
        vm = (await asyncio.create_subprocess_exec("vm_stat", stdout=asyncio.subprocess.PIPE))
        out = (await vm.communicate())[0].decode()
        page = 16384
        free = sum(int(re.search(rf'{k}:\s+(\d+)', out).group(1)) for k in ("Pages free", "Pages inactive", "Pages purgeable"))
        status["approx_free_memory_gb"] = round(free * page / 1e9, 1)
    except Exception:
        pass
    status["running_jobs"] = [_public(j) for j in _all_jobs(50) if j.get("status") == "running"]
    return status


@mcp.tool()
async def job_status(job_id: str, wait_seconds: float = 40) -> dict:
    """Check a generation job, waiting up to `wait_seconds` (max 45) for it to finish.
    Jobs keep running even if this MCP server restarts."""
    if not _read_job(job_id):
        return {"error": f"no job {job_id}"}
    return await _wait(job_id, wait_seconds)


def _all_jobs(limit: int = 20) -> list[dict]:
    files = sorted(JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    return [j for j in (_read_job(p.stem) for p in files) if j]


@mcp.tool()
async def list_jobs(limit: int = 20) -> list[dict]:
    """List recent generation jobs (running and finished), newest first."""
    return [_public(j) for j in _all_jobs(limit)]


@mcp.tool()
async def cancel_job(job_id: str) -> dict:
    """Cancel a running job (stops its worker and any model subprocess)."""
    job = _read_job(job_id)
    if not job:
        return {"error": f"no job {job_id}"}
    if job.get("status") != "running":
        return {"job_id": job_id, "status": job.get("status")}
    try:
        os.kill(job["pid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    return await _wait(job_id, 5)


@mcp.tool()
async def free_memory(model: str | None = None) -> dict:
    """Unload mlx-serve models to free RAM (one model id, or all loaded ones).
    Useful before a big video job."""
    async with _client(60) as c:
        models = (await c.get("/v1/models")).json()["data"]
        targets = [model] if model else [m["id"] for m in models if m.get("loaded") or m.get("state") not in (None, "unloaded")]
        done = []
        for t in targets:
            r = await c.post("/v1/unload-model", json={"model": t})
            done.append({"model": t, "http": r.status_code})
    return {"unloaded": done}


# ── generators ──────────────────────────────────────────────────────────────
# Each tool validates its inputs and starts a job; the matching @worker runs it
# in a detached process.

DEFAULT_WAIT = 40


@mcp.tool()
async def generate_image(
    prompt: str,
    quality: str = "quality",
    size: str = "1024x1024",
    count: int = 1,
    steps: int | None = None,
    seed: int | None = None,
    wait_seconds: float = DEFAULT_WAIT,
) -> dict:
    """Generate image(s) locally.
    quality: "quality" (Qwen-Image 2.1, best, ~1-3 min) | "fast" (Mage-Flow Turbo, ~5-10 s) | any mlx-serve image model id.
    size: "WxH", e.g. 1024x1024 square, 768x1344 reel/story, 1344x768 landscape, 1024x1280 portrait.
    count: number of images (different seeds). Returns PNG paths, or a job id if still running."""
    p = {"prompt": prompt, "model": _resolve(quality, IMAGE_MODELS, "quality"), "size": size,
         "count": max(1, min(count, 8)), "steps": steps, "seed": seed}
    return await _start("image", prompt[:80], p, wait_seconds)


@worker("image")
async def _image(p, job):
    paths = []
    for i in range(p["count"]):
        body: dict[str, Any] = {"model": p["model"], "prompt": p["prompt"], "size": p["size"]}
        if p["steps"]:
            body["steps"] = p["steps"]
        if p["seed"] is not None:
            body["seed"] = p["seed"] + i
        job["progress"] = f"image {i + 1}/{p['count']}"
        async with _client() as c:
            r = await c.post("/v1/images/generations", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
        out = _out_path("images", p["prompt"], "png")
        if i:
            out = out.with_name(out.stem + f"-{i + 1}.png")
        out.write_bytes(base64.b64decode(r.json()["data"][0]["b64_json"]))
        paths.append(str(out))
    return {"files": paths, "model": p["model"], "size": p["size"]}


@mcp.tool()
async def edit_image(
    prompt: str,
    image_paths: list[str],
    size: str | None = None,
    model: str | None = None,
    wait_seconds: float = DEFAULT_WAIT,
) -> dict:
    """Edit an image with an instruction (Mage-Flow Edit), e.g. "make it night time",
    "put this product on a marble table". One or more reference images. Size defaults
    to the reference's own size."""
    refs = [str(_require_file(x, "image")) for x in image_paths]
    p = {"prompt": prompt, "refs": refs, "size": size, "model": model or EDIT_MODEL}
    return await _start("image_edit", prompt[:80], p, wait_seconds)


@worker("image_edit")
async def _image_edit(p, job):
    files = [("model", (None, p["model"])), ("prompt", (None, p["prompt"]))]
    if p["size"]:
        files.append(("size", (None, p["size"])))
    for ref in map(Path, p["refs"]):
        files.append(("image[]", (ref.name, ref.read_bytes(), "image/png")))
    async with _client() as c:
        r = await c.post("/v1/images/edits", files=files)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    out = _out_path("images", "edit-" + p["prompt"], "png")
    out.write_bytes(base64.b64decode(r.json()["data"][0]["b64_json"]))
    return {"files": [str(out)], "model": p["model"]}


@mcp.tool()
async def generate_music(
    prompt: str,
    lyrics: str | None = None,
    instrumental: bool = False,
    duration_seconds: int = 60,
    bpm: int | None = None,
    vocal_language: str | None = None,
    seed: int | None = None,
    wait_seconds: float = DEFAULT_WAIT,
) -> dict:
    """Generate a song or instrumental with ACE-Step 1.5 XL (48 kHz stereo WAV).
    prompt: style / genre / mood / instruments, e.g. "upbeat hindi pop, female vocals, dhol, 120bpm".
    lyrics: song lyrics with optional [verse]/[chorus] tags. Omit and set instrumental=true for no vocals.
    duration_seconds: 10-600. vocal_language: e.g. "en", "hi"."""
    body: dict[str, Any] = {"model": MUSIC_MODEL, "prompt": prompt,
                            "duration_seconds": max(10, min(600, duration_seconds))}
    if instrumental:
        body["instrumental"] = True
    elif lyrics:
        body["lyrics"] = lyrics
    for k, v in (("bpm", bpm), ("vocal_language", vocal_language), ("seed", seed)):
        if v is not None:
            body[k] = v
    return await _start("music", prompt[:80], {"body": body}, wait_seconds)


@worker("music")
async def _music(p, job):
    out = _out_path("music", p["body"]["prompt"], "wav")
    await _post_bytes("/v1/audio/music-generations", p["body"], out)
    return {"files": [str(out)], "model": MUSIC_MODEL, "duration_seconds": p["body"]["duration_seconds"]}


@mcp.tool()
async def generate_speech(
    text: str,
    engine: Literal["fish", "qwen3"] = "fish",
    voice: Literal["mine", "default"] = "mine",
    ref_audio: str | None = None,
    ref_text: str | None = None,
    speed: float | None = None,
    verify: bool = True,
    wait_seconds: float = DEFAULT_WAIT,
) -> dict:
    """Speak text as a WAV voiceover. By default the result is transcribed back with local
    Whisper and compared to `text` (result.asr_text / word_error_rate / ok); a custom ref_audio
    with no ref_text is transcribed first so the clone gets its exact transcript.
    engine "fish": Fish Audio S2 Pro clone, best for the user's Hinglish accent (~15 s, ~20 GB).
    engine "qwen3": Qwen3-TTS via mlx-serve (faster, Apache-2.0).
    voice "mine" clones the user's own voice from the saved reference clip; "default" uses
    the engine's stock voice (qwen3 only). ref_audio/ref_text override the reference clip."""
    ref = Path(ref_audio).expanduser() if ref_audio else VOICE_REF
    rtext = ref_text if ref_text is not None else (
        VOICE_REF_TEXT.read_text().strip() if VOICE_REF_TEXT.exists() and not ref_audio else None)
    clone = engine == "fish" or voice == "mine" or bool(ref_audio)
    if clone:
        _require_file(ref, "voice reference")
    p = {"text": text, "engine": engine, "clone": clone, "ref": str(ref), "ref_text": rtext, "speed": speed,
         "verify": verify}
    return await _start("speech", text[:80], p, wait_seconds)


@worker("speech")
async def _speech(p, job):
    out = _out_path("speech", p["text"], "wav")
    if p["engine"] == "fish" and not p["ref_text"]:
        job["progress"] = "transcribing the reference clip"
        p["ref_text"] = (await _asr(p["ref"], job))["text"] or None
    if p["engine"] == "fish":
        cmd = [str(AVATAR_PY), "scripts/voice_clone.py", "--ref-audio", p["ref"], "--text", p["text"], "--out", str(out)]
        if p["ref_text"]:
            cmd += ["--ref-text", p["ref_text"]]
        if p["speed"]:
            cmd += ["--speed", str(p["speed"])]
        await _run(cmd, job, cwd=AVATAR, env={"HF_HUB_DISABLE_XET": "1"})
    else:
        body: dict[str, Any] = {"model": TTS_MODEL, "input": p["text"]}
        if p["clone"]:
            body["ref_audio"] = _b64_file(p["ref"])
        if p["speed"]:
            body["speed"] = p["speed"]
        await _post_bytes("/v1/audio/speech", body, out)
    res: dict[str, Any] = {"files": [str(out)], "engine": p["engine"], "ref_text_used": p.get("ref_text")}
    if p.get("verify", True):
        job["progress"] = "checking the voice line with speech recognition"
        heard = (await _asr(out, job))["text"]
        wer = _wer(p["text"], heard)
        res.update(asr_text=heard, word_error_rate=wer, ok=wer <= 0.25)
    return res


@mcp.tool()
async def transcribe(audio_path: str, language: str = "en", wait_seconds: float = DEFAULT_WAIT) -> dict:
    """Transcribe an audio or video file with local Whisper-turbo. Returns text plus word timestamps.
    Use it to check voiceovers, get exact reference transcripts for voice cloning, and time captions."""
    f = _require_file(audio_path, "audio")
    return await _start("asr", f.name, {"audio": str(f), "language": language}, wait_seconds)


@worker("asr")
async def _asr_job(p, job):
    return await _asr(p["audio"], job, p["language"])


@mcp.tool()
async def lipsync(
    audio_path: str,
    video_path: str | None = None,
    steps: int | None = None,
    wait_seconds: float = DEFAULT_WAIT,
) -> dict:
    """Make the person in real footage speak new audio (LatentSync 1.5, 256px, ~2-3 min).
    video_path defaults to the user's saved talking-head clip and must be at least as
    long as the audio. Output length follows the audio."""
    audio = _require_file(audio_path, "audio")
    video = _require_file(video_path or DEFAULT_FOOTAGE, "footage")
    p = {"audio": str(audio), "video": str(video), "steps": steps}
    return await _start("lipsync", audio.name, p, wait_seconds)


@worker("lipsync")
async def _lipsync(p, job):
    audio = Path(p["audio"])
    out = _out_path("video", "lipsync-" + audio.stem, "mp4")
    raw = out.with_name(out.stem + ".raw.mp4")
    cmd = [str(AVATAR_PY), "scripts/lipsync.py", "--video", p["video"], "--audio", str(audio),
           "--out", str(raw), "--resolution", "256"]
    if p["steps"]:
        cmd += ["--steps", str(p["steps"])]
    await _run(cmd, job, cwd=AVATAR, env={"HF_HUB_DISABLE_XET": "1"})
    # LatentSync muxes back its 16 kHz Whisper resample, which sounds muffled.
    # Re-attach the original full-rate audio.
    job["progress"] = "re-attaching original audio"
    await _run([_ffmpeg(), "-y", "-i", str(raw), "-i", str(audio), "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], job)
    raw.unlink(missing_ok=True)
    return {"files": [str(out)]}


@mcp.tool()
async def generate_video(
    prompt: str,
    quality: str = "fast",
    width: int = 768,
    height: int = 512,
    num_frames: int = 97,
    steps: int | None = None,
    seed: int | None = None,
    first_frame_image: str | None = None,
    turbo: bool = True,
    wait_seconds: float = DEFAULT_WAIT,
) -> dict:
    """Generate a short video clip with sound (MiniMax-H3) from text, optionally starting
    from an image. quality: "fast" (4-bit, 41 GB) | "quality" (8-bit, 70 GB) | model id.
    Vertical reel: width 512, height 896. num_frames ~24 per second. ~2+ min per clip."""
    body: dict[str, Any] = {"model": _resolve(quality, VIDEO_MODELS, "fast"), "prompt": prompt,
                            "width": width, "height": height, "num_frames": num_frames, "turbo": turbo}
    if steps:
        body["steps"] = steps
    if seed is not None:
        body["seed"] = seed
    ffi = str(_require_file(first_frame_image, "first frame image")) if first_frame_image else None
    return await _start("video", prompt[:80], {"body": body, "first_frame_image": ffi}, wait_seconds)


@worker("video")
async def _video(p, job):
    body = dict(p["body"])
    if p["first_frame_image"]:
        body["first_frame_image"] = _b64_file(p["first_frame_image"])
    ev = await _post_sse("/v1/video/generations", body, job)
    return await _save_video(ev, body["prompt"], job)


def _ffmpeg() -> str:
    return shutil.which("ffmpeg", path=f"{HOME / '.local/bin'}:{AVATAR / 'bin'}:{os.environ.get('PATH', '')}") or "ffmpeg"


async def _save_video(ev: dict, label: str, job: dict) -> dict:
    """mlx-serve's complete event carries raw rgb8 frames + s16le PCM (both base64).
    Encode them into an H.264/AAC MP4."""
    out = _out_path("video", label, "mp4")
    if ev.get("format") != "rgb8" or not isinstance(ev.get("data"), str):
        raise RuntimeError(f"unrecognised video response: { {k: v for k, v in ev.items() if not isinstance(v, str) or len(v) < 100} }")
    w, h, fps = ev["width"], ev["height"], ev.get("fps", 24)
    tmp = out.with_suffix(".tmp")
    tmp.mkdir(exist_ok=True)
    try:
        frames = tmp / "frames.rgb"
        frames.write_bytes(base64.b64decode(ev["data"]))
        cmd = [_ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
               "-i", str(frames)]
        if isinstance(ev.get("audio_data"), str) and ev["audio_data"]:
            pcm = tmp / "audio.pcm"
            pcm.write_bytes(base64.b64decode(ev["audio_data"]))
            cmd += ["-f", "s16le", "-ar", str(ev.get("audio_sample_rate", 32000)),
                    "-ac", str(ev.get("audio_channels", 2)), "-i", str(pcm),
                    "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-movflags", "+faststart", str(out)]
        job["progress"] = "encoding mp4"
        await _run(cmd, job)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"files": [str(out)], "frames": ev.get("frames"), "fps": fps,
            "seconds": round(ev.get("frames", 0) / fps, 2), "size": f"{w}x{h}"}


@mcp.tool()
async def make_reel(
    topic: str | None = None,
    script_file: str | None = None,
    scenes: int = 4,
    footage: str | None = None,
    engine: Literal["fish", "qwen3"] = "fish",
    edit: bool = True,
    captions: bool = True,
    broll: bool = True,
    wait_seconds: float = 5,
) -> dict:
    """Full talking-head reel of the user, topic in -> MP4 out (local-avatar pipeline):
    Ornith script -> cloned voice (Fish/Qwen3) -> LatentSync on the user's real footage ->
    plaque cards + b-roll -> assembled MP4, then optionally edited with captions, b-roll
    cut-aways and cross-dissolves. Takes ~3 min per scene; returns a job id."""
    if not topic and not script_file:
        raise ValueError("give a topic or a script_file")
    work_dir = _out_path("reels", topic or Path(script_file).stem, "d").with_suffix("")
    p = {"topic": topic, "script_file": str(Path(script_file).expanduser()) if script_file else None,
         "scenes": scenes, "footage": str(_require_file(footage or DEFAULT_FOOTAGE, "footage")),
         "engine": engine, "edit": edit, "captions": captions, "broll": broll, "work_dir": str(work_dir)}
    return await _start("reel", topic or script_file, p, wait_seconds)


@worker("reel")
async def _reel(p, job):
    work_dir = Path(p["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    raw = work_dir / "raw.mp4"
    cmd = [str(AVATAR_PY), "scripts/make_content.py", "--scenes", str(p["scenes"]),
           "--engine", p["engine"], "--work", str(work_dir), "--out", str(raw), "--footage", p["footage"]]
    cmd += ["--topic", p["topic"]] if p["topic"] else ["--script-file", p["script_file"]]
    if not p["broll"]:
        cmd.append("--no-assets")
    job["progress"] = "stage 1/2: generating scenes"
    await _run(cmd, job, cwd=AVATAR, env={"HF_HUB_DISABLE_XET": "1"})
    files = [str(raw)]
    if p["edit"]:
        job["progress"] = "stage 2/2: editing"
        final = work_dir / "final.mp4"
        cmd = [str(AVATAR_PY), "scripts/edit_reel.py", "--work", str(work_dir), "--out", str(final)]
        if not p["captions"]:
            cmd.append("--no-captions")
        await _run(cmd, job, cwd=AVATAR)
        files.insert(0, str(final))
    return {"files": files, "work_dir": str(work_dir)}


@mcp.tool()
async def render_hyperframes(
    project_dir: str,
    quality: Literal["draft", "looks", "standard", "high", "delivery"] = "looks",
    output_name: str | None = None,
    wait_seconds: float = DEFAULT_WAIT,
) -> dict:
    """Render a HyperFrames composition project (a folder with index.html) to MP4.
    Author the composition first with the HyperFrames skills, then call this."""
    proj = Path(project_dir).expanduser()
    if not (proj / "index.html").exists():
        raise FileNotFoundError(f"no index.html in {proj}")
    p = {"project": str(proj), "quality": quality, "output_name": output_name or proj.name}
    return await _start("hyperframes", proj.name, p, wait_seconds)


@worker("hyperframes")
async def _hyperframes(p, job):
    out = _out_path("video", p["output_name"], "mp4")
    env = {"PATH": f"{NODE_BIN}:{HOME / '.local/bin'}:{os.environ.get('PATH', '')}"}
    await _run(["npx", "-y", "hyperframes", "render", "--quality", p["quality"], "--output", str(out)],
               job, cwd=Path(p["project"]), env=env)
    if not out.exists():
        raise RuntimeError("render finished but no output file was written")
    return {"files": [str(out)]}


@mcp.tool()
async def mux_audio(video_path: str, audio_path: str, audio_volume: float = 1.0,
                    keep_original_audio: bool = False, wait_seconds: float = DEFAULT_WAIT) -> dict:
    """Lay an audio track (music / voiceover) under a video with ffmpeg. Output is
    trimmed to the video length. keep_original_audio mixes it with the video's own sound."""
    p = {"video": str(_require_file(video_path, "video")), "audio": str(_require_file(audio_path, "audio")),
         "volume": audio_volume, "keep": keep_original_audio}
    return await _start("mux", Path(video_path).name, p, wait_seconds)


@worker("mux")
async def _mux(p, job):
    v = Path(p["video"])
    out = _out_path("video", v.stem + "-mixed", "mp4")
    if p["keep"]:
        filt = f"[1:a]volume={p['volume']}[m];[0:a][m]amix=inputs=2:duration=first[a]"
    else:
        filt = f"[1:a]volume={p['volume']}[a]"
    await _run([_ffmpeg(), "-y", "-i", str(v), "-i", p["audio"], "-filter_complex", filt,
                "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", str(out)], job)
    return {"files": [str(out)]}


@mcp.tool()
async def preserve_face(reference_path: str, target_path: str, video: bool = False, wait_seconds: float = DEFAULT_WAIT) -> dict:
    """CPU VFX compositing: preserve the user's actual photographed face on a generated
    body/stage. Image paths for a still, or two matching-timing videos when video=True.
    Returns a composite and measured similarity. Does not synthesize a new face.
    """
    p = {"reference": str(_require_file(reference_path, "reference")),
         "target": str(_require_file(target_path, "target")), "video": video}
    return await _start("identity_composite", Path(target_path).stem, p, wait_seconds)

@worker("identity_composite")
async def _identity_composite(p, job):
    out = _out_path("video" if p["video"] else "images", "identity-" + Path(p["target"]).stem, "mp4" if p["video"] else "png")
    cmd = [str(AVATAR_PY), str(Path(__file__).with_name("identity_composite.py")), "--reference", p["reference"], "--target", p["target"], "--output", str(out)]
    if p["video"]: cmd += ["--video"]
    await _run(cmd, job)
    return {"files": [str(out)], "method": "actual source-face pixels, CPU VFX alignment", "report": json.loads(out.with_suffix(".identity.json").read_text())}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--job":
        _run_job(sys.argv[2])
    else:
        mcp.run()

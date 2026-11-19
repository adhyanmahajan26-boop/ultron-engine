"""Small real generations against cuda-serve, to prove a fresh GPU box works end to end.
    ultron-linux selftest [image|video|video10|music|speech|asr|all]
Writes files to $ULTRON_HOME/outputs/selftest and prints time per step. Standard library only.
`video10` is the 10-second one-take check: 241 frames at 24 fps, 1024x576, from a still made by the image step.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--what", default="all")
ap.add_argument("--out", required=True)
ap.add_argument("--port", default="11234")
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
HOME = Path(os.environ.get("ULTRON_HOME", Path.home() / "ultron-work"))
results = []


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


def step(name, fn):
    t = time.time()
    try:
        info = fn()
        results.append((name, "OK", time.time() - t, info))
    except Exception as e:
        results.append((name, "FAIL", time.time() - t, str(e)[:400]))
    print(f"{results[-1][1]:4s} {name:8s} {results[-1][2]:7.1f}s  {results[-1][3]}", flush=True)


def ffprobe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,nb_frames",
                        "-of", "json", str(p)], capture_output=True, text=True)
    return json.loads(r.stdout or "{}")


STILL = OUT / "still.png"


def image():
    raw, _ = post("/v1/images/generations", {"model": "qwen-image-2.1", "size": "1024x576", "seed": 7,
                                              "prompt": "A cinematic photo of a young man sitting in a dim Mumbai apartment at night, "
                                                        "warm lamp light, looking at the camera, 35mm film"})
    STILL.write_bytes(base64.b64decode(json.loads(raw)["data"][0]["b64_json"]))
    return str(STILL)


def video(frames=97, name="video.mp4"):
    def go():
        body = {"model": "ltx-2.5", "width": 1024, "height": 576, "num_frames": frames, "seed": 7,
                "prompt": "The man looks into the camera and says calmly: \"This is the Ultron Engine, running on a rented GPU.\" "
                          "Static camera, natural head movement, soft lamp light, room tone."}
        if STILL.exists():
            body["first_frame_image"] = base64.b64encode(STILL.read_bytes()).decode()
        raw, _ = post("/v1/video/generations", body)
        r = json.loads(raw)
        dst = OUT / name
        os.replace(r["path"], dst)
        pr = ffprobe(dst)
        return f"{dst} {pr.get('format', {}).get('duration')}s render {r.get('render_seconds')}s"
    return go


def music():
    raw, _ = post("/v1/audio/music-generations", {"model": "ace-step-1.5-xl-turbo", "prompt": "dark phonk, heavy 808, cowbell, 140 bpm",
                                                   "instrumental": True, "duration_seconds": 20, "seed": 7})
    (OUT / "music.wav").write_bytes(raw)
    return f"{OUT / 'music.wav'} {len(raw)} bytes"


def speech():
    ref = HOME / "avatar/inputs/my_voice_ref.wav"
    body = {"model": "qwen3-tts-1.7b", "input": "Testing the cloned voice on the rented GPU."}
    if ref.exists():
        body["ref_audio"] = base64.b64encode(ref.read_bytes()).decode()
    raw, _ = post("/v1/audio/speech", body)
    (OUT / "speech.wav").write_bytes(raw)
    return str(OUT / "speech.wav")


def asr():
    src = OUT / "speech.wav"
    if not src.exists():
        return "skipped (no speech.wav)"
    py = HOME / "venvs/avatar/bin/python"
    r = subprocess.run([str(py), os.environ.get("ULTRON_ASR_SCRIPT", ""), str(src)], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("ASR_JSON:")]
    if not line:
        raise RuntimeError(r.stderr[-400:])
    return json.loads(line[0][9:])["text"]


what = a.what
plan = {"image": [("image", image)], "video": [("video", video())], "video10": [("video10", video(241, "video10s.mp4"))],
        "music": [("music", music)], "speech": [("speech", speech)], "asr": [("asr", asr)]}
if what == "all":
    order = ["image", "video", "music", "speech", "asr"]
else:
    order = what.split(",")
for w in order:
    for n, f in plan[w]:
        step(n, f)
post("/v1/unload-model", {})
print(json.dumps([{"step": n, "status": s, "seconds": round(t, 1), "info": i} for n, s, t, i in results], indent=1))
sys.exit(0 if all(s == "OK" for _, s, _, _ in results) else 1)

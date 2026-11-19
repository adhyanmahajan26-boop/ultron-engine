"""Offline end-to-end test of the Linux stack without a GPU: cuda-serve (with fake workers) + the studio connector
over streamable HTTP, driven by a real MCP client, exactly as the Claude app would.

    python linux/tests/test_offline.py        (needs: fastapi uvicorn python-multipart mcp httpx, ffmpeg)
"""
import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

LINUX = Path(__file__).resolve().parents[1]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait(url, t=30):
    for _ in range(t * 10):
        try:
            if httpx.get(url, timeout=1).status_code < 500:
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"{url} did not come up")


async def mcp_calls(url):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = {t.name for t in (await s.list_tools()).tools}

            async def call(name, args):
                res = await s.call_tool(name, args)
                assert not res.isError, f"{name}: {res.content}"
                return json.loads(res.content[0].text)
            return tools, call, s


async def main():
    home = Path(tempfile.mkdtemp(prefix="ultron-test-"))
    cport, mport = free_port(), free_port()
    token = "t" * 20 + "0123456789abcdef"
    presets = home / "presets.json"
    presets.write_text(json.dumps({"video": {"fast": "ltx-2.5", "quality": "ltx-2.5"},
                                   "image": {"quality": "qwen-image-2.1", "fast": "qwen-image-2.1-turbo"},
                                   "edit": "qwen-image-2.1", "music": "ace-step-1.5-xl-turbo", "tts": "qwen3-tts-1.7b"}))
    env = {**os.environ, "ULTRON_HOME": str(home), "CUDA_SERVE_PORT": str(cport), "ULTRON_MCP_PORT": str(mport),
           "CUDA_SERVE_TEST_WORKER": str(LINUX / "tests/fake_worker.py"), "ULTRON_MCP_TOKEN": token,
           "MLX_SERVE_URL": f"http://127.0.0.1:{cport}", "CONTENT_STUDIO_PRESETS": str(presets),
           "ULTRON_AVATAR": str(home / "avatar"), "CONTENT_STUDIO_OUT": str(home / "outputs")}
    procs = [subprocess.Popen([sys.executable, str(LINUX / "cuda_serve/server.py")], env=env),
             subprocess.Popen([sys.executable, str(LINUX / "mcp_remote.py")], env=env)]
    ok = []
    try:
        wait(f"http://127.0.0.1:{cport}/health")
        wait(f"http://127.0.0.1:{mport}/healthz")
        C = f"http://127.0.0.1:{cport}"

        # mlx-serve compatible routes
        m = httpx.get(C + "/v1/models").json()["data"]
        assert {x["id"] for x in m} >= {"ltx-2.5", "qwen-image-2.1"}, m
        r = httpx.post(C + "/v1/images/generations", json={"model": "ddalcu/Qwen-Image-2.1-MLX-Serve-8bit", "prompt": "x", "size": "64x32"}, timeout=60)
        assert r.status_code == 200 and base64.b64decode(r.json()["data"][0]["b64_json"])[:4] == b"\x89PNG", r.text
        ok.append("image via Mac model id alias")
        loaded = [x for x in httpx.get(C + "/v1/models").json()["data"] if x["state"] == "loaded"]
        assert [x["id"] for x in loaded] == ["qwen-image-2.1"], loaded
        r = httpx.post(C + "/v1/images/edits", files=[("model", (None, "qwen-image-2.1")), ("prompt", (None, "night")),
                                                      ("image[]", ("a.png", b"\x89PNG fake", "image/png"))], timeout=60)
        assert r.status_code == 200, r.text
        ok.append("image edit multipart")
        r = httpx.post(C + "/v1/audio/music-generations", json={"model": "ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit", "prompt": "phonk"}, timeout=60)
        assert r.content[:4] == b"RIFF", r.text[:200]
        ok.append("music (worker switch)")
        r = httpx.post(C + "/v1/video/generations", json={"model": "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit", "prompt": "x"})
        assert r.status_code == 400 and "does not fit" in r.text, r.text
        ok.append("H3 refused with a clear message")
        events = []
        with httpx.stream("POST", C + "/v1/video/generations", json={"model": "ltx-2.5", "prompt": "x", "width": 64, "height": 64,
                                                                    "num_frames": 25, "stream": True}, timeout=60) as s:
            for line in s.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:]))
        assert events[-1]["type"] == "complete" and Path(events[-1]["path"]).exists(), events
        assert any(e["type"] == "progress" for e in events)
        ok.append("video SSE progress + mp4 path")
        r = httpx.post(C + "/v1/unload-model", json={})
        assert r.json()["unloaded"] == "ltx-2.5", r.text
        ok.append("unload")

        # connector: auth gate
        M = f"http://127.0.0.1:{mport}"
        assert httpx.post(M + "/mcp", json={}).status_code == 401
        assert httpx.get(M + "/wrongtoken/files/x").status_code == 401
        ok.append("token gate")

        # connector: a real MCP client over streamable HTTP, token in the path (how Claude connects)
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(f"{M}/{token}/mcp") as (rd, wr, _):
            async with ClientSession(rd, wr) as s:
                await s.initialize()
                tools = {t.name for t in (await s.list_tools()).tools}
                need = {"generate_image", "generate_video", "lipsync", "transcribe", "studio_status", "fetch_input",
                        "download_link", "list_inputs", "machine_status"}
                assert need <= tools, need - tools
                ok.append(f"MCP over HTTP: {len(tools)} tools")

                async def call(name, args):
                    res = await s.call_tool(name, args)
                    assert not res.isError, f"{name}: {res.content}"
                    return json.loads(res.content[0].text)

                st = await call("studio_status", {})
                assert st["mlx_serve"]["up"], st
                assert st["presets"]["video"]["fast"] == "ltx-2.5", st["presets"]
                ok.append("studio_status sees cuda-serve + Linux presets")
                img = await call("generate_image", {"prompt": "a fox", "size": "64x64", "wait_seconds": 40})
                assert img.get("status") == "done", img
                still = img["result"]["files"][0]
                ok.append("generate_image job")
                vid = await call("generate_video", {"prompt": "a fox runs", "width": 64, "height": 64, "num_frames": 241,
                                                    "first_frame_image": still, "last_frame_image": still, "wait_seconds": 40})
                assert vid.get("status") == "done", vid
                f = Path(vid["result"]["files"][0])
                assert f.exists() and f.stat().st_size > 1000, vid
                assert vid["result"]["frames"] == 241, vid
                ok.append("generate_video job (241 frames = 10 s, first+last frame) -> mp4 moved into outputs")
                link = await call("download_link", {"path": str(f)})
                body = httpx.get(link["url"]).content
                assert body == f.read_bytes()
                ok.append("download_link serves the file")
                up = httpx.put(f"{M}/{token}/upload?name=clip.mp4", content=b"12345").json()
                inputs = await call("list_inputs", {})
                assert any(x["path"] == up["path"] for x in inputs["files"]), inputs
                ok.append("upload + list_inputs")
                ms = await call("machine_status", {})
                assert isinstance(ms["models"], list), ms
                ok.append("machine_status")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(10)
    print("\n".join("PASS " + x for x in ok))
    print(f"{len(ok)} checks passed")


if __name__ == "__main__":
    asyncio.run(main())

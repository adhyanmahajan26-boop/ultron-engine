"""Serve the Content Studio MCP connector (src/studio/server.py, unchanged tools) over HTTPS-ready streamable HTTP,
so the Claude app can use a rented Linux GPU as a custom connector.

  https://<public-host>/<ULTRON_MCP_TOKEN>/mcp          the MCP endpoint (paste this into Claude > Connectors)
  https://<public-host>/<token>/files/<path>            download an output (paths under $ULTRON_HOME)
  POST https://<public-host>/<token>/upload?name=x.mp4  upload an input (raw body), saved to $ULTRON_HOME/inputs
  /healthz                                              no auth

The token in the path is the secret (Claude custom connectors accept a URL; a long random path is the simplest
auth that works with them). `Authorization: Bearer <token>` is accepted too, for Claude Code and curl.

Extra tools added on top of the studio's own: fetch_input (download a clip/image from a URL onto this machine),
download_link (public link for an output file), list_inputs, machine_status.
"""
from __future__ import annotations

import asyncio
import hmac
import importlib.util
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("ULTRON_HOME", Path.home() / "ultron-work"))
INPUTS = HOME / "inputs"
INPUTS.mkdir(parents=True, exist_ok=True)
TOKEN = os.environ.get("ULTRON_MCP_TOKEN", "")
if len(TOKEN) < 16:
    sys.exit("ULTRON_MCP_TOKEN must be set (>= 16 chars); linux/setup.sh generates one in $ULTRON_HOME/secrets/mcp_token")
os.environ.setdefault("CONTENT_STUDIO_OUT", str(HOME / "outputs"))

spec = importlib.util.spec_from_file_location("content_studio_server", REPO / "src/studio/server.py")
studio = importlib.util.module_from_spec(spec)
sys.modules["content_studio_server"] = studio
spec.loader.exec_module(studio)
mcp = studio.mcp
OUT = Path(studio.OUT)


def public_url() -> str:
    u = os.environ.get("ULTRON_PUBLIC_URL") or ""
    f = HOME / "run/public_url"
    if not u and f.exists():
        u = f.read_text().strip()
    return u.rstrip("/") or f"http://127.0.0.1:{os.environ.get('ULTRON_MCP_PORT', 8765)}"


def _link(p: Path) -> str:
    return f"{public_url()}/{TOKEN}/files/{urllib.parse.quote(str(p.resolve().relative_to(HOME.resolve())))}"


def _allowed(p: Path) -> bool:
    try:
        p.resolve().relative_to(HOME.resolve())
        return True
    except ValueError:
        return False


@mcp.tool()
async def fetch_input(url: str, filename: str | None = None) -> dict:
    """Download a file (your clip, a reference photo, a voice sample) from a URL onto the GPU machine, into
    $ULTRON_HOME/inputs. Use the returned `path` with the other tools. Google Drive share links are converted to
    direct downloads. For big files you can also upload with:  curl -T clip.mp4 "<upload_url>?name=clip.mp4" """
    m = re.search(r"drive\.google\.com/(?:file/d/|open\?id=)([\w-]+)", url)
    if m:
        url = f"https://drive.usercontent.google.com/download?id={m.group(1)}&export=download&confirm=t"
    name = filename or Path(urllib.parse.urlparse(url).path).name or f"input-{int(time.time())}"
    name = re.sub(r"[^\w.\- ]", "_", name)[:120]
    dst = INPUTS / name

    def go():
        req = urllib.request.Request(url, headers={"User-Agent": "ultron-engine/1.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    await asyncio.to_thread(go)
    return {"path": str(dst), "bytes": dst.stat().st_size}


@mcp.tool()
async def download_link(path: str) -> dict:
    """Get an HTTPS link to download a file produced on the GPU machine (any path under $ULTRON_HOME, e.g. a
    result from generate_video / lipsync). The link contains the connector secret: share it only with yourself."""
    p = Path(path).expanduser()
    if not p.is_file() or not _allowed(p):
        return {"error": f"not a file under {HOME}: {p}"}
    return {"url": _link(p), "bytes": p.stat().st_size}


@mcp.tool()
async def list_inputs() -> dict:
    """List files uploaded or fetched onto the GPU machine ($ULTRON_HOME/inputs), newest first, and the upload URL."""
    files = sorted(INPUTS.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"upload_url": f"{public_url()}/{TOKEN}/upload",
            "files": [{"path": str(p), "bytes": p.stat().st_size} for p in files if p.is_file()][:100]}


@mcp.tool()
async def machine_status() -> dict:
    """GPU, VRAM, RAM and disk of the rented Linux machine, plus which models are installed (cuda-serve)."""
    r: dict = {"public_url": public_url()}
    try:
        r["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                   "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        r["gpu"] = f"nvidia-smi failed: {e}"
    mem = dict(l.split(":", 1) for l in open("/proc/meminfo") if ":" in l)
    r["ram_gb"] = {"total": int(mem["MemTotal"].split()[0]) // 2**20, "available": int(mem["MemAvailable"].split()[0]) // 2**20}
    du = shutil.disk_usage(HOME)
    r["disk_gb"] = {"free": du.free // 2**30, "total": du.total // 2**30}
    try:
        with urllib.request.urlopen(f"{studio.MLX}/v1/models", timeout=5) as f:
            r["models"] = [{k: m.get(k) for k in ("id", "installed", "state", "capabilities")} for m in json.load(f)["data"]]
    except Exception as e:
        r["models"] = f"cuda-serve not reachable: {e}"
    return r


# ── ASGI: token gate + files + upload in front of the MCP app ───────────────

from mcp.server.transport_security import TransportSecuritySettings  # noqa: E402

# The tunnel's public hostname is not known in advance and the path token is the auth, so the MCP SDK's
# localhost-only Host check would reject every request coming through the tunnel.
mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
mcp.settings.streamable_http_path = "/mcp"
inner = mcp.streamable_http_app()


async def _send_simple(send, status: int, body: bytes, ctype: bytes = b"text/plain", extra=()):
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", ctype), (b"content-length", str(len(body)).encode()), *extra]})
    await send({"type": "http.response.body", "body": body})


async def _serve_file(send, p: Path):
    if not p.is_file() or not _allowed(p):
        return await _send_simple(send, 404, b"not found")
    ctype = (mimetypes.guess_type(p.name)[0] or "application/octet-stream").encode()
    size = p.stat().st_size
    await send({"type": "http.response.start", "status": 200, "headers": [
        (b"content-type", ctype), (b"content-length", str(size).encode()),
        (b"content-disposition", f'inline; filename="{p.name}"'.encode())]})
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
    await send({"type": "http.response.body", "body": b""})


async def _upload(receive, send, query: str):
    q = urllib.parse.parse_qs(query)
    name = re.sub(r"[^\w.\- ]", "_", (q.get("name") or [f"upload-{int(time.time())}"])[0])[:120]
    dst = INPUTS / name
    with open(dst, "wb") as f:
        while True:
            msg = await receive()
            f.write(msg.get("body", b""))
            if not msg.get("more_body"):
                break
    await _send_simple(send, 200, json.dumps({"path": str(dst), "bytes": dst.stat().st_size}).encode(), b"application/json")


def _tok_ok(t: str) -> bool:
    return hmac.compare_digest(t.encode(), TOKEN.encode())


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        return await inner(scope, receive, send)
    if scope["type"] != "http":
        return
    path = scope.get("path", "")
    if path == "/healthz":
        return await _send_simple(send, 200, b"ok")
    parts = path.lstrip("/").split("/", 1)
    rest = "/" + (parts[1] if len(parts) > 1 else "")
    if parts and _tok_ok(parts[0]):
        if rest.startswith("/files/"):
            return await _serve_file(send, HOME / urllib.parse.unquote(rest[len("/files/"):]))
        if rest.rstrip("/") == "/upload" and scope["method"] in ("POST", "PUT"):
            return await _upload(receive, send, scope.get("query_string", b"").decode())
        return await inner({**scope, "path": rest, "raw_path": rest.encode()}, receive, send)
    auth = dict(scope.get("headers") or []).get(b"authorization", b"").decode()
    if auth.startswith("Bearer ") and _tok_ok(auth[7:]) and path.startswith("/mcp"):
        return await inner(scope, receive, send)
    return await _send_simple(send, 401, b"unauthorized")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("ULTRON_MCP_HOST", "127.0.0.1"), port=int(os.environ.get("ULTRON_MCP_PORT", 8765)),
                proxy_headers=True, forwarded_allow_ips="*")

"""Tiny line protocol between cuda-serve and a model worker process. Standard library only, so every
model venv can import it (cuda-serve puts this folder on PYTHONPATH).

cuda-serve -> worker (stdin, one JSON object per line):
    {"id": "r1", "op": "load"}                      load weights now (warm-up), reply {"type": "result"}
    {"id": "r2", "op": "run", "params": {...}}      run one request
worker -> cuda-serve (stdout, one JSON object per line, nothing else may be printed to stdout):
    {"id": "r2", "type": "progress", "step": 3, "total": 8, "message": "..."}
    {"id": "r2", "type": "result", "result": {...}}  result is JSON; binary outputs are written to files
    {"id": "r2", "type": "error", "error": "..."}
Logs go to stderr (cuda-serve writes them to $ULTRON_HOME/logs/worker-<model>.log).

A worker keeps its model resident between requests. cuda-serve keeps at most ONE worker alive at a time
(the one-heavy-model rule), so switching models frees VRAM completely by ending the old process.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

_OUT = None


def _emit(obj: dict) -> None:
    _OUT.write(json.dumps(obj) + "\n")
    _OUT.flush()


class Ctx:
    """Passed to the handler so it can report progress for the current request."""

    def __init__(self, rid: str):
        self.rid = rid

    def progress(self, step: int | None = None, total: int | None = None, message: str | None = None) -> None:
        _emit({"id": self.rid, "type": "progress", "step": step, "total": total, "message": message})


def serve(handler) -> None:
    """handler: object with .load() and .run(params, ctx) -> dict."""
    global _OUT
    # Libraries love printing to stdout; keep the protocol channel clean by moving fd 1 to stderr.
    _OUT = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    loaded = False
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = msg.get("id", "?")
        try:
            if not loaded:
                handler.load()
                loaded = True
            if msg.get("op") == "load":
                _emit({"id": rid, "type": "result", "result": {"loaded": True}})
                continue
            res = handler.run(msg.get("params") or {}, Ctx(rid))
            _emit({"id": rid, "type": "result", "result": res})
        except Exception as e:  # report and keep serving
            traceback.print_exc()
            _emit({"id": rid, "type": "error", "error": f"{type(e).__name__}: {e}"})


def vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        pass
    return 0.0


def offload_mode() -> str:
    """ULTRON_OFFLOAD=none|model|group overrides; otherwise pick from VRAM."""
    m = os.environ.get("ULTRON_OFFLOAD", "auto")
    if m != "auto":
        return m
    v = vram_gb()
    return "none" if v >= 70 else "model" if v >= 20 else "group"


def seed_generator(seed):
    import torch
    g = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
    return g.manual_seed(int(seed)) if seed is not None else g.manual_seed(int.from_bytes(os.urandom(4), "little"))

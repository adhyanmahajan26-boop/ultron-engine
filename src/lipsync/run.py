"""Run one lip-sync job on any backend, optionally with the silent-audio control and QA.

  python src/lipsync/run.py --video base.mp4 --audio voice.wav --out out.mp4 \
      [--backend auto|official|mlx|cloud] [--version 1.6] [--guidance 2.0] [--ref-mode window] \
      [--silent-control] [--qa]

--silent-control renders the same job on silent audio (out_silent.mp4); --qa writes out.qa.json and
out.qa.jpg (source / lip-synced / silent rows). One heavy job at a time: on the Mac the jobs run
one after another and each waits for runtime/tools/heavy_guard.sh when it exists.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lipsync import config, refmodes  # noqa: E402
from lipsync.backends import Job, run_job  # noqa: E402


def heavy_guard(max_wait_s: int = 3600):
    g = config.engine_dir() / "runtime" / "tools" / "heavy_guard.sh"
    if not config.is_apple_silicon() or not g.exists():  # the guard is a macOS (zsh + vm_stat) script
        return
    t0 = time.time()
    while subprocess.run(["sh", str(g)], capture_output=True).returncode != 0:
        if time.time() - t0 > max_wait_s:
            raise SystemExit("heavy_guard: another heavy job is still running after an hour")
        print("waiting for the GPU (heavy_guard) ...", flush=True)
        time.sleep(30)


def job_from_args(a, out=None, audio=None) -> Job:
    return Job(video=a.video, audio=audio or a.audio, out=out or a.out, version=a.version, guidance=a.guidance,
               steps=a.steps, seed=a.seed, ref_mode=a.ref_mode, ref_idx=a.ref_idx, ref_shift=a.ref_shift,
               backend=a.backend, device=a.device, dtype=a.dtype, deepcache=a.deepcache)


def add_args(ap):
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--version", default="1.6", choices=sorted(config.VERSIONS))
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1247)
    ap.add_argument("--ref-mode", default="same", choices=refmodes.MODES)
    ap.add_argument("--ref-idx", default=None)
    ap.add_argument("--ref-shift", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--deepcache", action=argparse.BooleanOptionalAction, default=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    ap.add_argument("--silent-control", action="store_true")
    ap.add_argument("--qa", action="store_true")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    if a.ref_idx not in (None, "auto"):
        a.ref_idx = int(a.ref_idx)
    heavy_guard()
    res = run_job(job_from_args(a), log=a.log)
    print(json.dumps({"out": res["out"], "backend": res["backend"],
                      "audio_sensitivity": (res["report"] or {}).get("audio_sensitivity_mean")}))
    silent_out = None
    if a.silent_control:
        from lipsync import qa

        sw = qa.silent_wav(a.audio, str(Path(a.out).with_suffix("")) + "_silence.wav")
        silent_out = str(Path(a.out).with_suffix("")) + "_silent.mp4"
        heavy_guard()
        r2 = run_job(job_from_args(a, out=silent_out, audio=sw), log=a.log)
        print(json.dumps({"silent_out": silent_out, "audio_sensitivity": (r2["report"] or {}).get("audio_sensitivity_mean")}))
    if a.qa:
        from lipsync import qa

        m = qa.evaluate(a.video, a.audio, a.out, silent_out)
        base = str(Path(a.out).with_suffix(""))
        Path(base + ".qa.json").write_text(json.dumps(m, indent=1))
        rows = [("source", a.video), ("lip-synced", a.out)] + ([("silent control", silent_out)] if silent_out else [])
        qa.contact_sheet(rows, base + ".qa.jpg")
        print(json.dumps({"verdict": m["verdict"], "why": m["why"]}, indent=1))


if __name__ == "__main__":
    main()

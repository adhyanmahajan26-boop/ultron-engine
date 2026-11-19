"""One lip-sync job, three backends (standard library only, importable by the studio server).

    from lipsync.backends import Job, run_job
    run_job(Job(video="in.mp4", audio="voice.wav", out="out.mp4", version="1.6", guidance=2.0))

Backends
  official  ByteDance PyTorch code in the LatentSync venv (CUDA on Linux, MPS on Apple Silicon)
  mlx       the unofficial MLX port: $ULTRON_AVATAR/scripts/lipsync.py (Mac only)
  cloud     cloud/backends.py protocol, adapter role "lipsync.latentsync"
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lipsync import config  # noqa: E402

BACKENDS = ("official", "mlx", "cloud")


@dataclasses.dataclass
class Job:
    video: str
    audio: str
    out: str
    version: str = "1.6"
    guidance: Optional[float] = None  # None = backend default (upstream 1.5)
    steps: int = 20
    seed: int = 1247
    ref_mode: str = "same"            # same | fixed | shift | window (see refmodes.py)
    ref_idx: Optional[int | str] = None
    ref_shift: Optional[int] = None
    backend: str = "auto"
    device: str = "auto"              # official only: auto | cuda | mps | cpu
    dtype: str = "auto"               # official only
    deepcache: bool = True            # official only (upstream default on)
    dump: Optional[str] = None        # official only: tensor dump dir for parity.py
    parity: bool = False              # official only: deterministic run
    report: Optional[str] = None

    def resolved_backend(self) -> str:
        b = config.default_backend() if self.backend == "auto" else self.backend
        if b not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS} or auto, got {b!r}")
        return b


def _ref_env(job: Job) -> dict:
    env = {"LS_REF_MODE": job.ref_mode}
    if job.ref_idx is not None:
        env["LS_REF_IDX"] = str(job.ref_idx)
    if job.ref_shift is not None:
        env["LS_REF_SHIFT"] = str(job.ref_shift)
    return env


def command(job: Job) -> tuple[list[str], dict, Optional[str]]:
    """(argv, extra env, cwd) for a local backend."""
    b = job.resolved_backend()
    if b == "official":
        runner = Path(__file__).resolve().with_name("official_runner.py")
        py = config.latentsync_python()
        cmd = [str(py if py.exists() else sys.executable), str(runner), "--video", job.video, "--audio", job.audio,
               "--out", job.out, "--version", job.version, "--steps", str(job.steps), "--seed", str(job.seed),
               "--ref-mode", job.ref_mode, "--device", job.device, "--dtype", job.dtype]
        if job.guidance is not None:
            cmd += ["--guidance", str(job.guidance)]
        if job.ref_idx is not None:
            cmd += ["--ref-idx", str(job.ref_idx)]
        if job.ref_shift is not None:
            cmd += ["--ref-shift", str(job.ref_shift)]
        if not job.deepcache:
            cmd += ["--no-deepcache"]
        if job.dump:
            cmd += ["--dump", job.dump]
        if job.parity:
            cmd += ["--parity"]
        if job.report:
            cmd += ["--report", job.report]
        return cmd, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}, None
    if b == "mlx":
        av = config.avatar_dir()
        env = _ref_env(job)
        if job.ref_mode == "window":  # the MLX patch has no window mode; nearest equivalent
            env.update({"LS_REF_MODE": "shift", "LS_REF_SHIFT": str(job.ref_shift or 16)})
        if job.ref_mode == "fixed" and job.ref_idx in (None, "auto"):
            env["LS_REF_IDX"] = "0"
        if job.ref_mode == "same":
            env.pop("LS_REF_MODE")
        cmd = [str(av / ".venv" / "bin" / "python"), "scripts/lipsync.py", "--version", job.version,
               "--video", job.video, "--audio", job.audio, "--out", job.out]
        if job.guidance is not None:
            cmd += ["--guidance", str(job.guidance)]
        if job.steps:
            cmd += ["--steps", str(job.steps)]
        return cmd, {**env, "HF_HUB_DISABLE_XET": "1"}, str(av)
    raise ValueError("cloud has no local command")


def remux_original_audio(video: str, audio: str, out: str) -> None:
    """Swap LatentSync's 16 kHz mux for the original full-rate voice, trimmed to the video."""
    tmp = str(Path(out).with_suffix(".remux.mp4"))
    subprocess.run([config.ffmpeg(), "-loglevel", "error", "-y", "-nostdin", "-i", video, "-i", audio,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", tmp],
                   check=True)
    os.replace(tmp, out)


def run_job(job: Job, log=None) -> dict:
    """Run synchronously. Returns {"out":..., "backend":..., "report": {...} | None}."""
    b = job.resolved_backend()
    Path(job.out).parent.mkdir(parents=True, exist_ok=True)
    if b == "cloud":
        sys.path.insert(0, str(config.engine_dir()))
        from cloud import backends as cloud  # noqa: WPS433 (needs ultron.config.json)

        url = cloud.call("lipsync_latentsync", {k: v for k, v in dataclasses.asdict(job).items()
                                                 if k not in ("backend", "dump", "report")})
        return {"out": url, "backend": b, "report": None}
    job.report = job.report or job.out + ".report.json"
    cmd, env, cwd = command(job)
    full_env = {**os.environ, **env}
    src = str(Path(__file__).resolve().parents[1])
    full_env["PYTHONPATH"] = src + (os.pathsep + full_env["PYTHONPATH"] if full_env.get("PYTHONPATH") else "")
    with (open(log, "a") if log else open(os.devnull, "w")) as lf:
        p = subprocess.run(cmd, env=full_env, cwd=cwd, stdout=lf if log else None, stderr=subprocess.STDOUT if log else None)
    if p.returncode != 0:
        raise RuntimeError(f"{b} lipsync failed ({p.returncode}): {' '.join(cmd)}" + (f" - see {log}" if log else ""))
    if b == "mlx":
        remux_original_audio(job.out, job.audio, job.out)
    rep = None
    if Path(job.report).exists():
        rep = json.loads(Path(job.report).read_text())
    elif b == "mlx":
        rep = {"backend": "mlx", "out": job.out, "version": job.version, "guidance": job.guidance, "ref_mode": job.ref_mode}
        Path(job.report).write_text(json.dumps(rep, indent=1))
    return {"out": job.out, "backend": b, "report": rep}

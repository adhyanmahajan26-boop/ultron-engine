#!/usr/bin/env python3
"""Drop-in replacement for $ULTRON_AVATAR/scripts/lipsync.py that runs the OFFICIAL LatentSync.

The studio server (src/studio/server.py) calls
    $ULTRON_AVATAR/.venv/bin/python scripts/lipsync.py --video V --audio A --out O --resolution 256 [--steps N]
On Linux the installer symlinks this file there, so the call stays identical on every platform.
This shim re-executes itself under $ULTRON_LATENTSYNC_PY (the venv with LatentSync's pinned
torch stack) when that differs from the current interpreter, then hands off to official_runner.

Flags (superset of the MLX script): --video --audio --out [--version 1.6|1.5] [--resolution 512|256]
[--guidance G] [--steps N] [--seed S] (--resolution is accepted and ignored: the version fixes it); env LS_REF_MODE / LS_REF_IDX / LS_REF_SHIFT as in the MLX patch.
If the requested version's weights are missing it falls back to the other one with a warning.
"""
import argparse
import os
import sys
from pathlib import Path


def _engine_src() -> Path:
    env = os.environ.get("ULTRON_ENGINE_DIR")
    if env:
        return Path(env).expanduser() / "src"
    return Path(os.path.realpath(__file__)).parents[1]  # follows the symlink back into the repo


def main():
    src = _engine_src()
    target = os.environ.get("ULTRON_LATENTSYNC_PY")
    if target and Path(target).exists() and os.path.realpath(target) != os.path.realpath(sys.executable) \
            and not os.environ.get("_ULTRON_SHIM_REEXEC"):
        os.environ["_ULTRON_SHIM_REEXEC"] = "1"
        os.execv(target, [target, os.path.realpath(__file__)] + sys.argv[1:])
    sys.path.insert(0, str(src))
    from lipsync import config, official_runner

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default=None)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1247)
    a, rest = ap.parse_known_args()
    # Official 1.6 (512 px) is the quality target, so it is the default even though the studio server
    # still passes --resolution 256 (a leftover from the MLX 1.5 days). Ask for 1.5 with --version 1.5.
    version = a.version or "1.6"
    if not config.checkpoint(version).exists():
        other = "1.5" if version == "1.6" else "1.6"
        if config.checkpoint(other).exists():
            print(f"[shim] LatentSync {version} weights not installed, using {other}", file=sys.stderr)
            version = other
    argv = ["--video", a.video, "--audio", a.audio, "--out", a.out, "--version", version,
            "--steps", str(a.steps), "--seed", str(a.seed)] + rest
    if a.guidance is not None:
        argv += ["--guidance", str(a.guidance)]
    official_runner.main(argv)


if __name__ == "__main__":
    main()

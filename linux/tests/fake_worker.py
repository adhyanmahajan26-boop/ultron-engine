"""Stand-in model worker for offline tests: same protocol as the real workers, writes small real files with ffmpeg."""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuda_serve"))
from workerlib import serve  # noqa: E402


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


class Fake:
    def load(self):
        print("[fake] load", os.environ.get("ULTRON_QWEN_IMAGE_REPO") or os.environ.get("ULTRON_LTX_REPO"), file=sys.stderr)

    def run(self, p, ctx):
        out = Path(p["out"])
        print("this stray stdout line must not break the protocol")
        for i in range(3):
            ctx.progress(i + 1, 3, f"fake step {i + 1}")
        if out.suffix == ".png":
            w, _, h = (p.get("size") or "64x64").partition("x")
            ff("-f", "lavfi", "-i", f"color=c=red:s={w}x{h}", "-frames:v", "1", str(out))
            return {"path": str(out), "size": f"{w}x{h}", "refs": len(p.get("images") or [])}
        if out.suffix == ".wav":
            ff("-f", "lavfi", "-i", "sine=f=440:d=1", "-ar", "48000", "-ac", "2", str(out))
            return {"path": str(out)}
        if out.suffix == ".mp4":
            n = int(p.get("num_frames", 25))
            ff("-f", "lavfi", "-i", f"testsrc=s={p.get('width', 64)}x{p.get('height', 64)}:r=24", "-f", "lavfi",
               "-i", "sine=f=220", "-frames:v", str(n), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out))
            return {"path": str(out), "format": "mp4", "frames": n, "fps": 24, "width": p.get("width"),
                    "height": p.get("height"), "seconds": round(n / 24, 2), "first": bool(p.get("first_frame_image")),
                    "last": bool(p.get("last_frame_image"))}
        raise ValueError("unknown output " + str(out))


serve(Fake())

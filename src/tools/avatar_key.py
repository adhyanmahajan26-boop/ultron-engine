"""Matte an animated-avatar H3 clip (person matte), despill green, crop to the union bbox, write RGBA webm 520x560 (contain).
usage: avatar_key.py in.mp4 out.webm
"""
import sys, os, subprocess, json, numpy as np
FF = "@FFMPEG@"
NPX = "@PNPM@/npx"
inp, out = sys.argv[1], sys.argv[2]
tmp = out + ".raw.webm"
env = dict(os.environ, PATH="@PNPM@:" + os.environ["PATH"])
subprocess.run([NPX, "hyperframes", "remove-background", inp, "-o", tmp, "--quality", "best"], check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# union bbox of alpha (at quarter res)
info = subprocess.run([FF, "-hide_banner", "-c:v", "libvpx-vp9", "-i", tmp], capture_output=True, text=True).stderr
import re
m = re.search(r"(\d{3,4})x(\d{3,4})", info.split("Video:")[1]); W, H = int(m.group(1)), int(m.group(2))
w4, h4 = W // 4, H // 4
raw = subprocess.run([FF, "-loglevel", "error", "-c:v", "libvpx-vp9", "-i", tmp, "-vf", f"scale={w4}:{h4},format=rgba", "-f", "rawvideo", "-"], capture_output=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, h4, w4, 4)[..., 3]
mask = (fr > 40).any(axis=0)
ys, xs = np.where(mask)
x0, x1, y0, y1 = xs.min() * 4, (xs.max() + 1) * 4, ys.min() * 4, (ys.max() + 1) * 4
pad = 18; x0, y0 = max(0, x0 - pad), max(0, y0 - pad); x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
cw, ch = x1 - x0, y1 - y0
vf = (f"crop={cw}:{ch}:{x0}:{y0},format=rgba,despill=type=green:mix=0.6:expand=0.2,"
      f"scale=520:560:force_original_aspect_ratio=decrease:flags=lanczos,pad=520:560:(ow-iw)/2:oh-ih:color=0x00000000,format=yuva420p")
subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-c:v", "libvpx-vp9", "-i", tmp, "-vf", vf, "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "22", "-auto-alt-ref", "0", out], check=True)
os.remove(tmp)
print("avatar keyed:", out, "bbox", (x0, y0, x1, y1))

"""Temporal smoothing around the mouth only (kills per-frame texture shimmer / jiggle from lip-sync).
usage: smooth_mouth.py in.mp4 audio.wav out.mp4 [strength 0..1, default .5]
out = in everywhere except a feathered mouth/jaw box, where each frame is blended with its neighbours.
"""
import sys, os, subprocess, numpy as np, cv2
os.environ["OMP_NUM_THREADS"] = "2"
from insightface.app import FaceAnalysis
FF = "@FFMPEG@"
inp, wav, out = sys.argv[1], sys.argv[2], sys.argv[3]
k = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
root = "@AVATAR@/latentsync-mlx/checkpoints/auxiliary"
fa = FaceAnalysis(name="buffalo_l", root=root, allowed_modules=["detection", "landmark_2d_106"],
                  providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
cap = cv2.VideoCapture(inp); fps = cap.get(cv2.CAP_PROP_FPS) or 25
fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(f)
n = len(fr); H, W = fr[0].shape[:2]
geo = {}
for i in range(0, n, 3):
    fc = fa.get(fr[i])
    if fc:
        f0 = max(fc, key=lambda q: (q.bbox[2]-q.bbox[0])*(q.bbox[3]-q.bbox[1])); p = f0.landmark_2d_106
        mc = p[52:72].mean(0); fw = float(f0.bbox[2]-f0.bbox[0]); geo[i] = (mc[0], mc[1], fw)
ks = sorted(geo)
def at(i):
    if not ks: return (W/2, H/2, 200)
    j = np.searchsorted(ks, i); a = ks[max(0, j-1)]; b = ks[min(len(ks)-1, j)]
    t = 0 if a == b else (i-a)/(b-a); return tuple(geo[a][m]*(1-t)+geo[b][m]*t for m in range(3))
ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
res = []
for i in range(n):
    cx, cy, fw = at(i)
    d = np.sqrt(((xs-cx)/(0.62*fw))**2 + ((ys-(cy+0.08*fw))/(0.50*fw))**2)
    m = np.clip((1.15-d)/0.5, 0, 1)[..., None]*k
    prv = fr[max(0, i-1)].astype(np.float32); cur = fr[i].astype(np.float32); nxt = fr[min(n-1, i+1)].astype(np.float32)
    sm = 0.25*prv + 0.5*cur + 0.25*nxt
    res.append((cur*(1-m) + sm*m).clip(0, 255).astype(np.uint8))
p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                      "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out],
                     stdin=subprocess.PIPE)
for f in res: p.stdin.write(f.tobytes())
p.stdin.close(); p.wait(); print("wrote", out, n, "frames")

"""Face-lock: normalise face position/size across clips so cuts between windows are near-invisible.
usage: facelock.py measure clipA.mp4 clipB.mp4 ...        -> prints face centre/width per clip (source px)
       facelock.py apply  OUTDIR clipA.mp4 ...            -> writes 1920x1080 locked clips (same name) using the median face as target
"""
import sys, os, subprocess, json, numpy as np, cv2
os.environ["OMP_NUM_THREADS"] = "2"
from insightface.app import FaceAnalysis
FF = "@FFMPEG@"
root = "@AVATAR@/latentsync-mlx/checkpoints/auxiliary"
fa = FaceAnalysis(name="buffalo_l", root=root, allowed_modules=["detection"], providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1, det_size=(640, 640))

def measure(path, n=5):
    cap = cv2.VideoCapture(path); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = cap.get(cv2.CAP_PROP_FRAME_WIDTH); H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    rows = []
    for i in np.linspace(0, max(0, N - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        if not ok: continue
        fc = fa.get(f)
        if not fc: continue
        b = max(fc, key=lambda q: (q.bbox[2]-q.bbox[0])*(q.bbox[3]-q.bbox[1])).bbox
        rows.append(((b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0]))
    if not rows: return None
    m = np.median(np.array(rows), axis=0); return dict(cx=float(m[0]), cy=float(m[1]), fw=float(m[2]), W=W, H=H)

mode = sys.argv[1]
if mode == "measure":
    for p in sys.argv[2:]:
        print(os.path.basename(p), json.dumps({k: round(v, 1) for k, v in (measure(p) or {}).items()}))
elif mode == "apply":
    out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    ms = {p: measure(p) for p in sys.argv[3:]}
    ok = [m for m in ms.values() if m]
    tcx = np.median([m["cx"] / m["W"] for m in ok]); tcy = np.median([m["cy"] / m["H"] for m in ok]); tfw = np.median([m["fw"] / m["W"] for m in ok])
    print("target (fractions): cx %.3f cy %.3f fw %.3f" % (tcx, tcy, tfw))
    for p, m in ms.items():
        name = os.path.basename(p)
        if not m: print("no face", name); continue
        # scale so that face width fraction == tfw in the 1920 frame; never below cover (1920/W)
        k = max((tfw * 1920) / m["fw"] * 1.0, max(1920 / m["W"], 1080 / m["H"]) * 1.14)
        sw, sh = m["W"] * k, m["H"] * k
        # crop origin so the face centre lands on target centre, clamped inside
        x = np.clip(m["cx"] * k - tcx * 1920, 0, sw - 1920); y = np.clip(m["cy"] * k - tcy * 1080, 0, sh - 1080)
        vf = f"scale={int(round(sw))}:{int(round(sh))}:flags=lanczos,crop=1920:1080:{int(round(x))}:{int(round(y))}"
        subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", p, "-vf", vf + ",setsar=1", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", f"{out}/{name}"], check=True)
        print(name, "k=%.2f" % k, "face now at", round(x), round(y))

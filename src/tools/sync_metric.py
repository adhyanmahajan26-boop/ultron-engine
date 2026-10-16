"""Objective lip-sync check. usage: sync_metric.py audio.wav label1=clip1.mp4 [label2=clip2.mp4 ...] [--t0 SEC --dur SEC]
For each clip: inner-lip opening (insightface 106 landmarks 64-71, normalised by face height) vs the voice RMS envelope.
Reports corr@lag0, best lag (ms), opening during speech vs silence, and opening std (dynamic range). Writes crops sheet sync_crops.png.
Good sync: corr >= ~0.5 at lag ~0, speech opening clearly > silence opening. Still crops alone are NOT evidence.
"""
import sys, os, cv2, numpy as np, soundfile as sf, librosa
os.environ["OMP_NUM_THREADS"] = "3"
from insightface.app import FaceAnalysis
root = "@AVATAR@/latentsync-mlx/checkpoints/auxiliary"
fa = FaceAnalysis(name="buffalo_l", root=root, allowed_modules=["detection", "landmark_2d_106"], providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
argv = sys.argv[1:]; t0 = 0.0; dur = None
if "--t0" in argv: i = argv.index("--t0"); t0 = float(argv[i + 1]); del argv[i:i + 2]
if "--dur" in argv: i = argv.index("--dur"); dur = float(argv[i + 1]); del argv[i:i + 2]
aud = argv[0]; clips = [a.split("=", 1) for a in argv[1:]]
y, sr = sf.read(aud); y = y if y.ndim == 1 else y.mean(1)
def opening(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS); op = []; crops = []
    n0 = int(t0 * fps); cap.set(cv2.CAP_PROP_POS_FRAMES, n0); k = 0
    while True:
        if dur and k >= int(dur * fps): break
        ok, f = cap.read()
        if not ok: break
        s = f if f.shape[1] <= 1100 else cv2.resize(f, (960, 540)); sc = f.shape[1] / s.shape[1]
        fc = fa.get(s)
        if fc:
            b = max(fc, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1])); p = b.landmark_2d_106
            inn = p[64:72]; op.append((inn[:, 1].max() - inn[:, 1].min()) / (b.bbox[3] - b.bbox[1]))
            if k % 3 == 0:
                m = p[52:72]; cx = int(m[:, 0].mean() * sc); cy = int(m[:, 1].mean() * sc); w = int((b.bbox[2] - b.bbox[0]) * sc * 0.75)
                crops.append(cv2.resize(f[max(0, cy - w // 2):cy + w // 2, max(0, cx - w // 2):cx + w // 2], (120, 80)))
        else: op.append(op[-1] if op else 0)
        k += 1
    return np.array(op), fps, crops
rows = []
for label, path in clips:
    op, fps, crops = opening(path)
    a0 = int(t0 * sr); seg = y[a0:a0 + int(len(op) / fps * sr)] if True else y
    r = librosa.feature.rms(y=seg, frame_length=int(sr / fps * 2), hop_length=int(sr / fps))[0]; n = min(len(op), len(r)); op = op[:n]; r = r[:n]
    a = (r - r.mean()) / (r.std() + 1e-9); o = (op - op.mean()) / (op.std() + 1e-9)
    res = {lag: np.corrcoef(a[max(0, lag):n + min(0, lag)], o[max(0, -lag):n + min(0, -lag)])[0, 1] for lag in range(-18, 19)}
    best = max(res.items(), key=lambda t: t[1]); sil = r < np.percentile(r, 25)
    print(f"{label:22s} n={n:3d} corr@0={res[0]:+.2f} best lag {best[0]*1000/fps:+4.0f}ms corr {best[1]:+.2f} | open speech {op[~sil].mean():.3f} silence {op[sil].mean():.3f} ratio {op[~sil].mean()/max(op[sil].mean(),1e-6):.2f} | std {op.std():.4f}")
    if crops: rows.append(np.hstack(crops[:12]))
if rows:
    W = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite("sync_crops.png", np.vstack(rows)); print("crops: sync_crops.png (rows in the order given)")

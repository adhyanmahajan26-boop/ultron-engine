"""Add head/body motion + blinks to a lip-synced steady-base clip WITHOUT touching the mouth region's sync.
usage: liveliness.py lipsynced.mp4 matte.png still.png blinksrc.mp4 audio.wav out.mp4 [blink_times comma list]
 - person matte (RGBA png of the still) is warped (roll/shift/breath) over an inpainted wall plate, then a gentle push-in
 - blinks: eye band from blinksrc frames (H3 idle clip that started from the same still), ECC-aligned, feathered, colour matched
"""
import sys, cv2, numpy as np, subprocess, os
ls, matte, still, bsrc, audio, out = sys.argv[1:7]
btimes = [float(x) for x in (sys.argv[7].split(",") if len(sys.argv) > 7 else "1.9,4.3".split(","))]
FF = "@FFMPEG@"

def frames(p):
    cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS); L = []
    while True:
        ok, f = cap.read()
        if not ok: break
        L.append(f)
    return L, fps

PROC = (bsrc == "procedural")
F, fps = frames(ls); B, bfps = ([], 30) if PROC else frames(bsrc)
H, W = F[0].shape[:2]
rgba = cv2.imread(matte, cv2.IMREAD_UNCHANGED); alpha = rgba[..., 3].astype(np.float32) / 255
S = cv2.imread(still)
# background plate: inpaint the person away
hole = cv2.dilate((alpha > 0.1).astype(np.uint8) * 255, np.ones((25, 25), np.uint8))
plate = cv2.inpaint(S, hole, 9, cv2.INPAINT_TELEA); plate = cv2.GaussianBlur(plate, (0, 0), 3)

if PROC:
    import os as _os
    _os.environ["OMP_NUM_THREADS"] = "3"
    from insightface.app import FaceAnalysis
    _fa = FaceAnalysis(name="buffalo_l", root="@AVATAR@/latentsync-mlx/checkpoints/auxiliary", allowed_modules=["detection", "landmark_2d_106"], providers=["CPUExecutionProvider"]); _fa.prepare(ctx_id=-1, det_size=(640, 640))
    _f = max(_fa.get(S), key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1])); _fw = _f.bbox[2] - _f.bbox[0]; _lm = _f.landmark_2d_106
    EYES = []
    for c in _f.kps[:2]:
        d = np.hypot(_lm[:, 0] - c[0], _lm[:, 1] - c[1]); pts = _lm[d < 0.075 * _fw]
        EYES.append((pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()))
    # consistent boxes from the eye centres (landmark clusters can clip one eye)
    _hw = max(e[1] - e[0] for e in EYES) / 2 + 3; _hh = max(e[3] - e[2] for e in EYES) / 2 + 1
    EYES = [(c[0] - _hw, c[0] + _hw, c[1] - _hh, c[1] + _hh) for c in _f.kps[:2]]
    print("eyes", [tuple(round(float(v)) for v in e) for e in EYES], file=sys.stderr)
    def blink_frame(g, b):
        out = g.copy()
        for (x0, x1, y0, y1) in EYES:
            pad = int((x1 - x0) * 0.35); X0, X1 = int(x0) - pad, int(x1) + pad
            ytop = int(y0 - (y1 - y0) * 1.6); ylid = int(y0); ylow = int(y1)
            ynew = ylid + b * (ylow - ylid) * 1.0
            ys = np.arange(ytop, ylow + 3, dtype=np.float32); src = ys.copy()
            up = ys <= ynew
            src[up] = ytop + (ys[up] - ytop) * (ylid - ytop) / max(1e-3, (ynew - ytop))
            lid = (ys > ynew) & (ys <= ylow + 2)
            src[lid] = ylid - 1 - np.minimum(ylow + 2 - ys[lid], 2) * 0.5  # lid skin just above the lash line
            xs = np.arange(X0, X1, dtype=np.float32)
            mx_, my_ = np.meshgrid(xs, src)
            patch = cv2.remap(g, mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            w = np.zeros((len(ys), len(xs)), np.float32); w[:] = 1.0
            fx = np.hanning(len(xs)) ** 0.35; fy = np.clip(np.minimum(ys - ytop, ylow + 3 - ys) / 6.0, 0, 1)
            w = (w * fx[None, :] * fy[:, None])[..., None]
            roi = out[ytop:ylow + 3, X0:X1].astype(np.float32)
            out[ytop:ylow + 3, X0:X1] = (roi * (1 - w) + patch.astype(np.float32) * w).astype(np.uint8)
        return out
    BL = [0.3, 0.75, 1.0, 0.8, 0.4]
    blink = BL; nb = len(BL); env = None
else:
    # eye band
    ex0, ex1, ey0, ey1 = 372, 664, 150, 245
    band = np.zeros((H, W), np.float32); cv2.ellipse(band, ((ex0 + ex1) // 2, (ey0 + ey1) // 2), ((ex1 - ex0) // 2, (ey1 - ey0) // 2), 0, 0, 360, 1, -1)
    band = cv2.GaussianBlur(band, (0, 0), 9)
    Sg = cv2.cvtColor(S, cv2.COLOR_BGR2GRAY)
    def reg(bf):
        g = cv2.cvtColor(bf, cv2.COLOR_BGR2GRAY); M = np.eye(2, 3, dtype=np.float32)
        try:
            _, M = cv2.findTransformECC(Sg, g, M, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
        except cv2.error: pass
        return cv2.warpAffine(bf, M, (W, H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    BR = [reg(b) for b in B]
    lo, hi = 36, 44  # quick blink in c1 (eye-band brightness peak at frames 38-42)
    blink = BR[lo:hi + 1]; print(f"blink frames {lo}-{hi} ({len(blink)})", file=sys.stderr)
    nb = len(blink); env = np.hanning(nb + 2)[1:-1]

pivot = (W / 2, 520.0)
outf = []
for i, f in enumerate(F):
    t = i / fps
    roll = 1.5 * np.sin(2 * np.pi * t / 3.3) + 0.5 * np.sin(2 * np.pi * t / 1.37 + 0.8)
    dx = 7 * np.sin(2 * np.pi * t / 4.4 + 1.0) + 2 * np.sin(2 * np.pi * t / 1.7)
    dy = 5 * np.sin(2 * np.pi * t / 2.1) + 2 * np.sin(2 * np.pi * t / 0.9 + 2)
    sc = 1 + 0.010 * np.sin(2 * np.pi * t / 3.9)
    g = f.copy()
    for bt in btimes:
        k = int(round((t - bt) * fps))
        if 0 <= k < nb:
            if PROC: g = blink_frame(g, blink[k])
            else:
                w = band[..., None]
                bsl = blink[k].astype(np.float32); m = (np.abs(w) > 0.02)[..., 0]
                dm = (g.astype(np.float32)[m].mean(0) - bsl[m].mean(0)); bsl = np.clip(bsl + dm * 0.8, 0, 255)
                g = (g * (1 - w) + bsl * w).astype(np.uint8)
    M = cv2.getRotationMatrix2D(pivot, roll, sc); M[0, 2] += dx; M[1, 2] += dy
    wp = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    wa = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    comp = (wp.astype(np.float32) * wa + plate.astype(np.float32) * (1 - wa))
    # slow camera push-in on everything
    z = 1.0 + 0.045 * (i / max(1, len(F) - 1))
    Mz = cv2.getRotationMatrix2D((W / 2, H * 0.45), 0, z)
    comp = cv2.warpAffine(comp, Mz, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    outf.append(np.clip(comp, 0, 255).astype(np.uint8))

tmp = out + ".silent.mp4"
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
for f in outf: vw.write(f)
vw.release()
subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", tmp, "-i", audio, "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)
os.remove(tmp); print("wrote", out)

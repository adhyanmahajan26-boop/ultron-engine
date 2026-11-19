"""Mouth-only blend: keep LatentSync's lips, take everything else (beard, moustache, skin) from the source.

LatentSync redraws the whole lower half of the face through the VAE, which smears beard texture, and on
rounded vowels ("o") it can overshoot into a pink pout that bleeds into the beard. This post-pass:
  1. finds the lips in both the source and the lip-synced frame (insightface 106-point landmarks, 52:72),
     smoothed over time so the mask doesn't flicker,
  2. builds a mask from the union of both lip outlines, grown by --margin (fraction of mouth width) and feathered,
  3. pulls the synced mouth's colour (LAB a/b only, so teeth and the dark mouth stay as generated) towards
     the source's colour in the same region (--chroma 0..1), which takes the pink out,
  4. composites: synced inside the mask, source outside. Audio is copied from the synced clip.

    python src/lipsync/mouthblend.py --src base.mp4 --synced take.mp4 --out take_blend.mp4 [--sheet cmp.jpg]
Needs numpy, opencv-python, insightface, onnxruntime (all in the LatentSync venv). Nothing is downloaded.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lipsync import config  # noqa: E402

MOUTH = slice(52, 72)


def read_frames(path: str):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return frames, fps


def landmarks(frames, app) -> list:
    """106-pt landmarks of the largest face per frame; misses reuse the previous frame."""
    out, last = [], None
    for f in frames:
        faces = app.get(f)
        if faces:
            b = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            last = np.asarray(b.landmark_2d_106, dtype=np.float32)
        out.append(last)
    first = next((x for x in out if x is not None), None)
    if first is None:
        raise SystemExit("no face found")
    return [first if x is None else x for x in out]


def smooth(lmks: list, radius: int = 1) -> list:
    a = np.stack(lmks)
    o = np.empty_like(a)
    for i in range(len(a)):
        o[i] = a[max(0, i - radius): i + radius + 1].mean(0)
    return list(o)


def mouth_mask(shape, pts_a, pts_b, margin: float, feather: float) -> np.ndarray:
    """Float mask in [0,1]: hull of both lip outlines, grown by margin*mouth width, Gaussian-feathered."""
    import cv2
    h, w = shape[:2]
    pts = np.concatenate([pts_a[MOUTH], pts_b[MOUTH]]).astype(np.float32)
    width = float(np.ptp(pts[:, 0])) or 1.0
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(pts.astype(np.int32)), 255)
    grow = max(1, int(round(margin * width)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    sigma = max(0.5, feather * width)
    m = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), sigma)
    return np.clip(m, 0.0, 1.0)


def match_chroma(gen, src, mask, strength: float):
    """Move gen's LAB a/b mean and std inside the mask towards src's. L (brightness) is untouched."""
    import cv2
    if strength <= 0:
        return gen
    sel = mask > 0.5
    if sel.sum() < 20:
        return gen
    g = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in (1, 2):
        gm, gs = g[..., c][sel].mean(), g[..., c][sel].std() + 1e-6
        sm, ss = s[..., c][sel].mean(), s[..., c][sel].std() + 1e-6
        t = (g[..., c] - gm) * (ss / gs) + sm
        g[..., c] = g[..., c] + strength * (t - g[..., c]) * (mask if mask.ndim == 2 else mask[..., 0])
    return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def blend(src_frames, gen_frames, src_l, gen_l, margin=0.22, feather=0.10, chroma=0.7):
    out = []
    for s, g, ls, lg in zip(src_frames, gen_frames, src_l, gen_l):
        if s.shape != g.shape:
            import cv2
            s = cv2.resize(s, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
        m = mouth_mask(g.shape, lg, ls, margin, feather)
        gc = match_chroma(g, s, m, chroma)
        m3 = m[..., None]
        out.append((gc.astype(np.float32) * m3 + s.astype(np.float32) * (1 - m3)).round().astype(np.uint8))
    return out


def write_video(frames, fps, audio_from: str, out: str, crf: int = 14):
    h, w = frames[0].shape[:2]
    cmd = [config.ffmpeg(), "-loglevel", "error", "-y", "-nostdin", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-i", audio_from, "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy",
           "-shortest", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg failed writing " + out)


def sheet(src_frames, gen_frames, out_frames, gen_l, idx, path):
    """Mouth crops: rows source / synced / blended at the given frame indices."""
    import cv2
    rows = [[], [], []]
    for i in idx:
        if i >= len(out_frames):
            continue
        p = gen_l[i][MOUTH]
        cx, cy = p.mean(0)
        r = int(max(np.ptp(p[:, 0]), 40) * 1.1)
        y0, y1, x0, x1 = int(cy - r), int(cy + r), int(cx - r), int(cx + r)
        for row, fr in zip(rows, (src_frames[i], gen_frames[i], out_frames[i])):
            fr = cv2.copyMakeBorder(fr, r, r, r, r, cv2.BORDER_REPLICATE)
            row.append(cv2.resize(fr[y0 + r:y1 + r, x0 + r:x1 + r], (200, 200)))
    if rows[0]:
        cv2.imwrite(path, np.vstack([np.hstack(r) for r in rows]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source footage (same frames the lip-sync used, 25 fps)")
    ap.add_argument("--synced", required=True, help="LatentSync output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=0.22, help="grow the lip mask by this x mouth width")
    ap.add_argument("--feather", type=float, default=0.10, help="Gaussian feather, x mouth width")
    ap.add_argument("--chroma", type=float, default=0.7, help="0 = keep generated colour, 1 = full match to source")
    ap.add_argument("--sheet", help="write a mouth-crop comparison jpg")
    ap.add_argument("--sheet-frames", default="", help="comma list of frame indices for --sheet (default: 8 evenly)")
    a = ap.parse_args(argv)

    from lipsync.qa import face_app
    src, _ = read_frames(a.src)
    gen, fps = read_frames(a.synced)
    n = min(len(src), len(gen))
    if abs(len(src) - len(gen)) > 1:
        print(f"[warn] frame counts differ: src {len(src)} vs synced {len(gen)}; using {n}", file=sys.stderr)
    src, gen = src[:n], gen[:n]
    app = face_app()
    ls, lg = smooth(landmarks(src, app)), smooth(landmarks(gen, app))
    out = blend(src, gen, ls, lg, a.margin, a.feather, a.chroma)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_video(out, fps, a.synced, a.out)
    if a.sheet:
        idx = [int(x) for x in a.sheet_frames.split(",") if x.strip()] or list(np.linspace(0, n - 1, 8).astype(int))
        sheet(src, gen, out, lg, idx, a.sheet)
    print(f"wrote {a.out} ({n} frames, margin={a.margin} feather={a.feather} chroma={a.chroma})")


if __name__ == "__main__":
    main()

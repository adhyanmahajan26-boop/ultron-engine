"""Lip-sync QA in one place: voice-follow, silent-audio control, identity, contact sheets, verdict.

Everything the old notes say to always do (skills/studio-face-lipsync-qa):
  1. silent-audio control  - same job, same seed, silent audio. If the mouth moves the same way,
                             the model is copying the source mouth, not listening.
  2. voice-driven delta    - (output mouth - source mouth) vs voice loudness (tools/audio_influence.py)
  3. sync                  - output mouth vs voice loudness, best lag (tools/sync_metric.py)
  4. identity              - ArcFace similarity output vs source, same frame
  5. full-face contact sheet (metrics alone hid the pink "duck-pout")
  + SyncNet confidence (LSE-C) from upstream eval/, when checkpoints/auxiliary/syncnet_v2.model exists

Usage
  python src/lipsync/qa.py --src base.mp4 --audio voice.wav --clip out.mp4 [--silent out_silent.mp4] \
      --json qa.json --sheet qa.jpg
Needs: numpy opencv-python insightface onnxruntime librosa soundfile (all in the LatentSync and avatar venvs).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lipsync import config, refmodes  # noqa: E402

FPS = 25
# Starting thresholds, from the October/November runs (good windows: voice corr .40-.85, identity .86-.94;
# the broken MLX 1.6 run: speech-vs-silent mouth corr .96). Tune with Addy's eye, not instead of it.
THRESH = {"voice_corr_min": 0.35, "silent_same_max": 0.80, "identity_min": 0.80, "silent_open_ratio_max": 0.75}

_FA = None


def face_app():
    global _FA
    if _FA is None:
        os.environ.setdefault("OMP_NUM_THREADS", "3")
        from insightface.app import FaceAnalysis

        roots = [os.environ.get("ULTRON_INSIGHTFACE_ROOT"), str(config.latentsync_dir() / "checkpoints" / "auxiliary"),
                 str(config.avatar_dir() / "latentsync-mlx" / "checkpoints" / "auxiliary")]
        root = next((r for r in roots if r and (Path(r) / "models" / "buffalo_l").exists()), roots[1])
        _FA = FaceAnalysis(name="buffalo_l", root=root, allowed_modules=["detection", "landmark_2d_106", "recognition"],
                           providers=["CPUExecutionProvider"])
        _FA.prepare(ctx_id=-1, det_size=(640, 640))
    return _FA


def read_frames(path: str, step: int = 1, max_w: int = 1100):
    import cv2

    cap = cv2.VideoCapture(path)
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % step == 0:
            yield k, f
        k += 1
    cap.release()


def analyse(path: str, with_embed_every: int = 5) -> dict:
    """Per-frame mouth opening (normalised), mouth box, and an ArcFace embedding every N frames."""
    import cv2

    fa = face_app()
    opn, boxes, emb = [], [], {}
    for k, f in read_frames(path):
        s = f if f.shape[1] <= 1100 else cv2.resize(f, (960, int(960 * f.shape[0] / f.shape[1])))
        sc = f.shape[1] / s.shape[1]
        faces = fa.get(s)
        if not faces:
            opn.append(opn[-1] if opn else 0.0)
            boxes.append(boxes[-1] if boxes else None)
            continue
        b = max(faces, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]))
        p = b.landmark_2d_106
        opn.append(refmodes.mouth_openness(p, b.bbox[3] - b.bbox[1]))
        m = p[52:72]
        w = (b.bbox[2] - b.bbox[0]) * 0.75
        boxes.append((m[:, 0].mean() * sc, m[:, 1].mean() * sc, w * sc, b.bbox * sc))
        if with_embed_every and k % with_embed_every == 0 and getattr(b, "normed_embedding", None) is not None:
            emb[k] = b.normed_embedding
    return {"open": np.array(opn), "boxes": boxes, "emb": emb}


def voice_rms(audio: str, n: int) -> np.ndarray:
    import librosa
    import soundfile as sf

    y, sr = sf.read(audio)
    y = y if y.ndim == 1 else y.mean(1)
    r = librosa.feature.rms(y=y, frame_length=int(sr / FPS * 2), hop_length=int(sr / FPS))[0]
    return np.pad(r, (0, max(0, n - len(r))))[:n]


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def _corr(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    return float(np.corrcoef(_z(a[:n]), _z(b[:n]))[0, 1])


def lag_corr(a, b, max_lag=6):
    """best (lag_frames, corr) with b shifted against a; positive lag = mouth after voice."""
    n = min(len(a), len(b))
    a, b = _z(a[:n]), _z(b[:n])
    res = {}
    for L in range(-max_lag, max_lag + 1):
        x, y = (a[:n - L], b[L:]) if L >= 0 else (a[-L:], b[:n + L])
        if len(x) > 3:
            res[L] = float(np.corrcoef(x, y)[0, 1])
    best = max(res.items(), key=lambda t: t[1])
    return best[0], best[1], res.get(0, float("nan"))


def identity(src: dict, out: dict) -> dict:
    common = sorted(set(src["emb"]) & set(out["emb"]))
    sims = [float(np.dot(src["emb"][k], out["emb"][k])) for k in common]
    if not sims:
        return {"mean": None, "p10": None}
    return {"mean": round(float(np.mean(sims)), 3), "p10": round(float(np.percentile(sims, 10)), 3)}


def syncnet_conf(video: str) -> dict | None:
    """Upstream SyncNet confidence (higher = better sync; upstream reports ~8-9 for real talking video)."""
    ls = config.latentsync_dir()
    if not (ls / "checkpoints" / "auxiliary" / "syncnet_v2.model").exists():
        return None
    py = config.latentsync_python()
    with tempfile.TemporaryDirectory() as td:
        try:
            p = subprocess.run([str(py), "-m", "eval.eval_sync_conf", "--video_path", str(Path(video).resolve()),
                                "--temp_dir", str(Path(td) / "t")], cwd=ls, capture_output=True, text=True, timeout=900)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        finally:
            subprocess.run(["rm", "-rf", str(ls / "detect_results")])
    m = re.search(r"SyncNet confidence:\s*([\d.]+)", p.stdout)
    o = re.search(r"AV offset:\s*(-?\d+)", p.stdout)
    if not m:
        return {"error": (p.stderr or p.stdout)[-400:]}
    return {"confidence": float(m.group(1)), "av_offset": int(o.group(1)) if o else None}


def evaluate(src: str, audio: str, clip: str, silent: str | None = None, speech_mask_db: float = -35.0,
             run_syncnet: bool = True) -> dict:
    S = analyse(src)
    C = analyse(clip)
    n = min(len(S["open"]), len(C["open"]))
    r = voice_rms(audio, n)
    db = 20 * np.log10(r + 1e-9)
    speech = db > (db.max() + speech_mask_db * 0.5)  # loud frames relative to peak
    o = C["open"][:n]
    delta = o - S["open"][:n]
    lag, best, c0 = lag_corr(r, o)
    res = {
        "frames": int(n),
        "voice_corr_lag0": round(c0, 3), "voice_corr_best": round(best, 3), "best_lag_ms": int(lag * 1000 / FPS),
        "delta_voice_corr": round(_corr(delta, r), 3),
        "delta_in_speech": round(float(delta[speech].mean()) if speech.any() else 0.0, 4),
        "delta_outside": round(float(delta[~speech].mean()) if (~speech).any() else 0.0, 4),
        "open_speech_vs_quiet": round(float(o[speech].mean() / max(o[~speech].mean(), 1e-6)), 2) if speech.any() and (~speech).any() else None,
        "mouth_std": round(float(o.std()), 4),
        "source_follow_corr": round(_corr(o, S["open"][:n]), 3),
        "identity": identity(S, C),
    }
    if silent:
        Q = analyse(silent, with_embed_every=0)
        m = min(n, len(Q["open"]))
        res["silent"] = {
            "speech_vs_silent_corr": round(_corr(o[:m], Q["open"][:m]), 3),
            "silent_open_ratio": round(float(Q["open"][:m].std() / max(o[:m].std(), 1e-9)), 3),
            "silent_voice_corr": round(_corr(Q["open"][:m], r[:m]), 3),
        }
    if run_syncnet:
        res["syncnet"] = syncnet_conf(clip)
    res["verdict"], res["why"] = verdict(res)
    return res


def verdict(m: dict) -> tuple[str, list[str]]:
    why = []
    if not (m["voice_corr_best"] >= THRESH["voice_corr_min"]):
        why.append(f"mouth barely follows the voice (corr {m['voice_corr_best']} < {THRESH['voice_corr_min']})")
    if abs(m["best_lag_ms"]) > 80:
        why.append(f"sync offset {m['best_lag_ms']} ms")
    s = m.get("silent")
    if s:
        if s["speech_vs_silent_corr"] > THRESH["silent_same_max"]:
            why.append(f"silent-audio control moves the same way (corr {s['speech_vs_silent_corr']}): copying the source mouth")
        if s["silent_open_ratio"] > THRESH["silent_open_ratio_max"]:
            why.append(f"mouth still moves a lot on silence (ratio {s['silent_open_ratio']})")
    else:
        why.append("no silent-audio control run")
    idm = m["identity"]["mean"]
    if idm is not None and idm < THRESH["identity_min"]:
        why.append(f"face drifts from the source (ArcFace {idm} < {THRESH['identity_min']})")
    hard = [w for w in why if not w.startswith("no silent")]
    return ("PASS" if not hard else "FAIL"), why


def contact_sheet(rows: list[tuple[str, str]], out: str, every: int = 25, width: int = 240, crops: bool = True,
                  boxes_from: str | None = None) -> str:
    """rows of (label, video): full-face frames every `every` frames, plus a mouth-crop strip per row."""
    import cv2

    ref = analyse(boxes_from or rows[0][1], with_embed_every=0)["boxes"] if crops else None
    lines = []
    for label, path in rows:
        frames = [f for _, f in read_frames(path, step=every)]
        if not frames:
            continue
        tiles = []
        for i, f in enumerate(frames):
            k = i * every
            if ref and k < len(ref) and ref[k]:
                cx, cy, w, bb = ref[k]
                x1, y1, x2, y2 = [int(v) for v in bb]
                pad = int((x2 - x1) * 0.25)
                f = f[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            h = int(width * f.shape[0] / max(f.shape[1], 1))
            tiles.append(cv2.resize(f, (width, h)))
        H = max(t.shape[0] for t in tiles)
        tiles = [np.pad(t, ((0, H - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
        row = np.hstack(tiles)
        bar = np.full((28, row.shape[1], 3), 20, np.uint8)
        cv2.putText(bar, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        lines.append(np.vstack([bar, row]))
    W = max(r.shape[1] for r in lines)
    sheet = np.vstack([np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in lines])
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def silent_wav(like: str, out: str) -> str:
    import soundfile as sf

    info = sf.info(like)
    sf.write(out, np.zeros((int(info.frames), info.channels), dtype="float32"), info.samplerate)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the source (base) video the clip was lip-synced on")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--silent", help="the same job run on silent audio")
    ap.add_argument("--json")
    ap.add_argument("--sheet")
    ap.add_argument("--no-syncnet", action="store_true")
    a = ap.parse_args()
    res = evaluate(a.src, a.audio, a.clip, a.silent, run_syncnet=not a.no_syncnet)
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    if a.sheet:
        rows = [("source", a.src), ("lip-synced", a.clip)] + ([("silent control", a.silent)] if a.silent else [])
        contact_sheet(rows, a.sheet)


if __name__ == "__main__":
    main()

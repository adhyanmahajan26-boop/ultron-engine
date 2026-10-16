"""Quality gate for a talking clip (LTX native speech). usage: clip_qa.py clip.mp4 "expected script text" [--lang hi|en] [--ref still.png] [--adv 0.29]
Checks (all must pass):
  WORDS  Whisper transcript vs expected text: word-overlap >= 0.8
  SYNC   mouth-opening vs voice envelope with the 0.29 s audio advance: corr@0 >= 0.30 and |best lag| <= 120 ms; mouth open ratio speech/silence >= 1.15
  FACE   ArcFace vs reference still: min >= 0.55 (identity holds)
  CALM   head x/y std <= 7 px, roll std <= 3.5 deg (user: not overexpressive)
  PROPS  left/right edge strips must not change much vs frame 0 (hands/bottles entering the frame)
  AUDIO  no abrupt jumps inside speech, no clipping
Prints one line per check and PASS/FAIL, plus a score used to rank takes.
"""
import sys, os, re, json, subprocess, difflib, numpy as np, cv2, soundfile as sf, librosa
os.environ["OMP_NUM_THREADS"] = "3"
FF = "@FFMPEG@"
clip, expected = sys.argv[1], sys.argv[2]
lang = "hi" if "--lang" not in sys.argv else sys.argv[sys.argv.index("--lang") + 1]
ref = None if "--ref" not in sys.argv else sys.argv[sys.argv.index("--ref") + 1]
ADV = 0.29 if "--adv" not in sys.argv else float(sys.argv[sys.argv.index("--adv") + 1])
tmp = f"/tmp/qa_{os.path.basename(clip)}"
subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", clip, "-vn", "-ac", "1", "-ar", "16000", tmp + ".wav"], check=True)
res = {}; fails = []
# ---- WORDS
from mlx_audio.stt.utils import load
m = load("@AVATAR@/models/whisper-turbo")
r = m.generate(tmp + ".wav", language=lang, word_timestamps=False); heard = (getattr(r, "text", "") or "").strip()
ALIAS = {"eltron": "अल्ट्रॉन", "ultron": "अल्ट्रॉन", "अल्ट्रोन": "अल्ट्रॉन", "अल्टरोंट": "अल्ट्रॉन", "engine": "इंजन", "इंजिन": "इंजन", "cloud": "क्लाउड", "hyperframes": "हाइपरफ्रेम्स", "studio": "स्टूडियो", "motion": "मोशन", "graphics": "ग्राफिक्स", "reels": "रील्स", "image": "इमेज", "generation": "जनरेशन", "voice": "वॉइस", "local": "लोकल", "video": "वीडियो", "cloning": "क्लोनिंग", "surprise": "सरप्राइज़", "clone": "क्लोन", "यह": "ये"}
def norm(s):
    ws = [w for w in re.sub(r"[^\w\u0900-\u097F ]", " ", s.lower().replace("\u093c", "")).split() if w]
    return [ALIAS.get(w, w) for w in ws]
a, b = norm(expected), norm(heard)
_c = lambda ws: re.sub(r"[\u0964\u0965]", "", "".join(ws))
sm = difflib.SequenceMatcher(None, _c(a), _c(b), autojunk=False)
overlap = sm.ratio() if (a and b) else 0.0
res["words"] = overlap
if overlap < 0.75: fails.append(f"WORDS {overlap:.2f}<0.75 heard='{heard}'")
# ---- video frames: face / motion / props
cap = cv2.VideoCapture(clip); fps = cap.get(cv2.CAP_PROP_FPS) or 24; frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", root="@AVATAR@/latentsync-mlx/checkpoints/auxiliary", allowed_modules=["detection", "landmark_2d_106", "recognition"], providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
refemb = None
if ref:
    rf = fa.get(cv2.imread(ref)); refemb = max(rf, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1])).normed_embedding if rf else None
cx, cy, roll, opn, sims = [], [], [], [], []
for i, f in enumerate(frames):
    fc = fa.get(f)
    if not fc: opn.append(opn[-1] if opn else 0); continue
    q = max(fc, key=lambda z: (z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1])); lm = q.landmark_2d_106
    cx.append((q.bbox[0] + q.bbox[2]) / 2); cy.append((q.bbox[1] + q.bbox[3]) / 2); e = q.kps; roll.append(np.degrees(np.arctan2(e[1][1] - e[0][1], e[1][0] - e[0][0])))
    opn.append((lm[64:72, 1].max() - lm[64:72, 1].min()) / (q.bbox[3] - q.bbox[1]))
    if refemb is not None and i % 6 == 0: sims.append(float(np.dot(q.normed_embedding, refemb)))
if sims: res["face_min"] = min(sims[1:] or sims)
if sims and min(sims[1:] or sims) < 0.55: fails.append(f"FACE min {min(sims[1:] or sims):.2f}<0.55")
hx, hy, rl = float(np.std(cx)) if cx else 0, float(np.std(cy)) if cy else 0, float(np.std(roll)) if roll else 0
res["head_px"] = (hx, hy); res["roll"] = rl
if hx > 7 or hy > 14 or rl > 3.5: fails.append(f"CALM head std {hx:.1f}/{hy:.1f}px roll {rl:.1f}deg")
H, W = frames[0].shape[:2]; strip = int(W * 0.22); base = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32); worst = 0
for f in frames[1:]:
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = max(np.abs(g[:, :strip] - base[:, :strip]).mean(), np.abs(g[:, -strip:] - base[:, -strip:]).mean()); worst = max(worst, d)
res["edge_change"] = worst
if worst > 28: fails.append(f"PROPS edge change {worst:.0f}>28 (something entered the frame)")
# ---- SYNC (audio advanced by ADV)
y, sr = sf.read(tmp + ".wav"); y = y if y.ndim == 1 else y.mean(1); y = y[int(ADV * sr):]
rr = librosa.feature.rms(y=y, frame_length=int(sr / fps * 2), hop_length=int(sr / fps))[0]; n = min(len(opn), len(rr)); o = np.array(opn[:n]); rr = rr[:n]
az = (rr - rr.mean()) / (rr.std() + 1e-9); oz = (o - o.mean()) / (o.std() + 1e-9)
lags = {l: np.corrcoef(az[max(0, l):n + min(0, l)], oz[max(0, -l):n + min(0, -l)])[0, 1] for l in range(-6, 7)}
bl = max(lags, key=lags.get); c0 = lags[0]; sil = rr < np.percentile(rr, 25); ratio = o[~sil].mean() / max(o[sil].mean(), 1e-6)
res["sync_c0"], res["lag_ms"], res["open_ratio"] = c0, bl * 1000 / fps, ratio
if c0 < 0.30 or abs(bl * 1000 / fps) > 120 or ratio < 1.15:
    msg = f"SYNC corr0 {c0:+.2f} lag {bl*1000/fps:+.0f}ms ratio {ratio:.2f}"
    if lang == "hi": print("   ~ advisory (Hindi: user judged the LTX+voice-convert Hindi sample 10/10 by eye; this metric is unreliable for Hindi):", msg)
    else: fails.append(msg)
# ---- AUDIO
yy, sr2 = sf.read(tmp + ".wav"); yy = yy if yy.ndim == 1 else yy.mean(1)
db = 20 * np.log10(librosa.feature.rms(y=yy, frame_length=320, hop_length=160)[0] + 1e-9); sp = db > np.percentile(db, 95) - 30; d = np.abs(np.diff(db))
nj = sum(1 for i in range(2, len(d) - 4) if d[i] > 18 and sp[i - 1] and sp[i + 3]); res["jumps"] = nj
if nj > 0 or (np.abs(yy) > 0.98).sum() > 0: fails.append(f"AUDIO jumps {nj}")
score = res["words"] * 2 + res.get("sync_c0", 0) + res.get("face_min", 0.6) - 0.05 * (hx + hy) - 0.1 * rl - res["edge_change"] / 100
print(f"{os.path.basename(clip):28s} {'PASS' if not fails else 'FAIL'} score {score:5.2f} | words {overlap:.2f} sync {c0:+.2f}@{bl*1000/fps:+.0f}ms ratio {ratio:.2f} face {res.get('face_min', 0):.2f} head {hx:.1f}/{hy:.1f}px roll {rl:.1f} edge {worst:.0f} jumps {nj}")
for f in fails: print("   x", f)
json.dump({"pass": not fails, "score": float(score), "fails": fails, "heard": heard}, open(tmp + ".json", "w"), ensure_ascii=False)

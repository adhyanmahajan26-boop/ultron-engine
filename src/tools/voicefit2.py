"""Phrase-level voice fit (replaces voicefit.py word-level mode, which cut inside words and caused audible breaks).
usage: voicefit2.py ltx_asr_job clone_asr_job clone.wav out.wav [total_dur]
 - words matched in order (difflib); phrase boundaries = LTX pauses >= 0.22 s
 - the clone is cut ONLY at the quietest point inside the matching pause (never inside a word)
 - each phrase is time-stretched as a whole (ffmpeg atempo, pitch-preserving, clamped 0.8-1.25) so its speech starts at the LTX phrase start
 - extra/less time is absorbed by the pauses between phrases. Prints per-phrase tempo and a break check.
"""
import sys, json, re, os, subprocess, tempfile, difflib
import numpy as np, soundfile as sf, librosa
J = "@STUDIO@/outputs/jobs"
FF = "@FFMPEG@"
ltx_id, cl_id, wav, out = sys.argv[1:5]
total = float(sys.argv[5]) if len(sys.argv) > 5 else None
def words(j):
    if str(j).endswith(".json"): return json.load(open(j))["words"]   # asr_words.py output
    return json.load(open(f"{J}/{j}.json"))["result"]["words"]
def norm(w):
    w = re.sub(r"[^\wऀ-ॿ]", "", w.lower()).replace("़", "")
    for a, b in (("यह", "ये"), ("यही", "यहीं"), ("चहरा", "चेहरा"), ("अल्टरोंट", "ultron")): w = w.replace(a, b)
    return w
L, C = words(ltx_id), words(cl_id)
sm = difflib.SequenceMatcher(None, [norm(w["word"]) for w in L], [norm(w["word"]) for w in C], autojunk=False)
pairs = [(L[a + k], C[b + k]) for a, b, n in sm.get_matching_blocks() for k in range(n)]
print(f"matched {len(pairs)} of {len(L)} ltx / {len(C)} clone words", file=sys.stderr)
y, sr = sf.read(wav); y = (y if y.ndim == 1 else y.mean(1)).astype(np.float32)
dur_c = len(y) / sr; dur_l = total or max(w["end"] for w in L) + 0.2
rms = librosa.feature.rms(y=y, frame_length=int(sr * 0.01), hop_length=int(sr * 0.005))[0]
def quiet_point(t0, t1):
    a, b = int(t0 / 0.005), max(int(t0 / 0.005) + 1, int(t1 / 0.005)); return (a + int(np.argmin(rms[a:b]))) * 0.005 if b > a else (t0 + t1) / 2
# phrase splits at LTX pauses
starts = [0]
for k in range(1, len(pairs)):
    if pairs[k][0]["start"] - pairs[k - 1][0]["end"] >= 0.22: starts.append(k)
starts.append(len(pairs))
phr = []
for a, b in zip(starts[:-1], starts[1:]): phr.append(pairs[a:b])
print(f"{len(phr)} phrases", file=sys.stderr)
cuts = [0.0]
for i in range(1, len(phr)):
    cuts.append(quiet_point(phr[i - 1][-1][1]["end"], phr[i][0][1]["start"]))
cuts.append(dur_c)
buf = np.zeros(int(dur_l * sr) + sr, dtype=np.float32); ratios = []
for i, ph in enumerate(phr):
    c_first, c_last = ph[0][1]["start"], ph[-1][1]["end"]; l_first, l_last = ph[0][0]["start"], ph[-1][0]["end"]
    tempo = (c_last - c_first) / max(0.05, (l_last - l_first)); t_used = min(max(tempo, 0.72), 1.25); ratios.append((round(tempo, 2), round(t_used, 2)))
    seg = y[int(cuts[i] * sr):int(cuts[i + 1] * sr)]
    ti, to = tempfile.mktemp(suffix=".wav"), tempfile.mktemp(suffix=".wav"); sf.write(ti, seg, sr)
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", ti, "-af", f"atempo={t_used:.4f}", to], check=True)
    s2, _ = sf.read(to); s2 = (s2 if s2.ndim == 1 else s2.mean(1)).astype(np.float32); os.remove(ti); os.remove(to)
    f = min(len(s2) // 4, int(0.015 * sr)); s2[:f] *= np.linspace(0, 1, f); s2[len(s2) - f:] *= np.linspace(1, 0, f)
    p0 = int((l_first - (c_first - cuts[i]) / t_used) * sr)
    if p0 < 0: s2 = s2[-p0:]; p0 = 0
    e = min(len(buf), p0 + len(s2)); buf[p0:e] += s2[:e - p0]
sf.write(out, buf[:int(dur_l * sr)], sr)
print("phrase tempo (raw, used):", ratios, file=sys.stderr)
rr = librosa.feature.rms(y=buf[:int(dur_l * sr)], frame_length=int(sr * 0.01), hop_length=int(sr * 0.005))[0] + 1e-6; db = 20 * np.log10(rr)
sp = db > np.percentile(db, 90) - 35; d = np.abs(np.diff(db)); nj = sum(1 for i in range(2, len(d) - 6) if d[i] > 18 and sp[i - 2] and sp[i + 4])
print("abrupt jumps in speech:", nj, "| wrote", out, f"{dur_l:.2f}s")

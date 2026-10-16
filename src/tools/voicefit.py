"""Fit a cloned-voice reading to the word timing of another speech track (e.g. LTX's own generated speech) so lips stay matched.
usage: voicefit.py ltx_asr_job_id clone_asr_job_id clone.wav out.wav [total_dur_sec] [group=3]
 - both *_asr_job_id are content-studio `transcribe` job ids (word timestamps)
 - words are matched in order (difflib); anchors every `group` matched words; each chunk of the clone is time-stretched
   (ffmpeg atempo, pitch-preserving) to the matching chunk length and placed at the target time. No pitch shifting.
Prints stretch ratios; ratios outside 0.75-1.35 sound strained.
"""
import sys, json, re, os, subprocess, tempfile, difflib
import numpy as np, soundfile as sf
J = "@STUDIO@/outputs/jobs"
FF = "@FFMPEG@"
ltx_id, cl_id, wav, out = sys.argv[1:5]
total = float(sys.argv[5]) if len(sys.argv) > 5 else None
group = int(sys.argv[6]) if len(sys.argv) > 6 else 3

def words(jid):
    return json.load(open(f"{J}/{jid}.json"))["result"]["words"]
def norm(w):
    w = re.sub(r"[^\wऀ-ॿ]", "", w.lower()); w = w.replace("़", "")
    for a, b in (("यह", "ये"), ("यही", "यहीं"), ("चहरा", "चेहरा"), ("आवाज़", "आवाज")): w = w.replace(a, b)
    return w
L, C = words(ltx_id), words(cl_id)
sm = difflib.SequenceMatcher(None, [norm(w["word"]) for w in L], [norm(w["word"]) for w in C], autojunk=False)
pairs = []  # matched (ltx word, clone word)
for a, b, n in sm.get_matching_blocks():
    for k in range(n): pairs.append((L[a + k], C[b + k]))
print(f"matched {len(pairs)} of {len(L)} ltx / {len(C)} clone words", file=sys.stderr)
y, sr = sf.read(wav); y = y if y.ndim == 1 else y.mean(1)
dur_c = len(y) / sr; dur_l = total or max(w["end"] for w in L) + 0.2
# word-level placement: each matched word (plus any unmatched words/short gap that follow it) starts where LTX's word starts,
# stretched gently (0.8-1.25x); leftover timing lands in the gaps between words.
out_y = np.zeros(int(dur_l * sr) + sr // 2, dtype=np.float32)
ratios = []
for i, (lw, cw) in enumerate(pairs):
    nxt = pairs[i + 1] if i + 1 < len(pairs) else None
    c0 = max(0.0, cw["start"] - 0.03)
    c1 = (nxt[1]["start"] - 0.04) if (nxt and nxt[1]["start"] - cw["end"] < 0.45) else cw["end"] + 0.05
    l0 = lw["start"]
    l1 = (nxt[0]["start"] - 0.02) if (nxt and nxt[0]["start"] - lw["end"] < 0.45) else lw["end"]
    c1 = min(c1, dur_c)
    if c1 - c0 < 0.04 or l1 - l0 < 0.04: continue
    seg = y[int(c0 * sr):int(c1 * sr)]
    tempo = (c1 - c0) / (l1 - l0); t_used = min(max(tempo, 0.8), 1.25); ratios.append(round(tempo, 2))
    tmp_in = tempfile.mktemp(suffix=".wav"); tmp_out = tempfile.mktemp(suffix=".wav"); sf.write(tmp_in, seg, sr)
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", tmp_in, "-af", f"atempo={t_used:.4f}", tmp_out], check=True)
    s2, _ = sf.read(tmp_out); s2 = s2 if s2.ndim == 1 else s2.mean(1); s2 = s2.astype(np.float32)
    f = min(len(s2) // 4, int(0.008 * sr))
    if f: s2[:f] *= np.linspace(0, 1, f); s2[len(s2) - f:] *= np.linspace(1, 0, f)
    p0 = int(l0 * sr); end = min(len(out_y), p0 + len(s2)); out_y[p0:end] += s2[:end - p0]
    os.remove(tmp_in); os.remove(tmp_out)
sf.write(out, out_y[:int(dur_l * sr)], sr); print("tempo ratios per chunk (>1 = clone sped up):", ratios, file=sys.stderr); print("wrote", out, f"{dur_l:.2f}s")

"""Remove awkward pauses: shrink every internal silence longer than MIN_GAP to KEEP (keeps breaths natural),
trim head/tail silence, crossfade the joins. usage: tighten.py in.wav out.wav [min_gap=0.22] [keep=0.10]
Prints the kept speech spans (new timeline) so lip-sync windows can be cut on word boundaries.
"""
import sys, numpy as np, soundfile as sf, librosa
inp, out = sys.argv[1], sys.argv[2]
MIN_GAP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.22
KEEP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
PAD = 0.05
y, sr = sf.read(inp); y = y if y.ndim == 1 else y.mean(1)
hop = 160
rms = librosa.feature.rms(y=y, frame_length=480, hop_length=hop)[0]
db = 20 * np.log10(rms + 1e-9)
thr = np.percentile(db, 95) - 16                    # relative threshold: speech vs noise floor
speech = db > thr
# fill very short dips (stops/plosives) < 90 ms so words don't get chopped
min_dip = int(0.09 * sr / hop)
i = 0
while i < len(speech):
    if not speech[i]:
        j = i
        while j < len(speech) and not speech[j]: j += 1
        if (j - i) < min_dip and i > 0 and j < len(speech): speech[i:j] = True
        i = j
    else: i += 1
# spans of speech
spans = []; i = 0
while i < len(speech):
    if speech[i]:
        j = i
        while j < len(speech) and speech[j]: j += 1
        spans.append([i * hop / sr, j * hop / sr]); i = j
    else: i += 1
# merge spans separated by gaps shorter than MIN_GAP
merged = [spans[0]]
for s in spans[1:]:
    if s[0] - merged[-1][1] < MIN_GAP: merged[-1][1] = s[1]
    else: merged.append(s)
pieces = []; cursor = 0.0; report = []
for k, (a, b) in enumerate(merged):
    a2, b2 = max(0, a - PAD), min(len(y) / sr, b + PAD)
    seg = y[int(a2 * sr):int(b2 * sr)].copy()
    fade = int(0.012 * sr); seg[:fade] *= np.linspace(0, 1, fade); seg[-fade:] *= np.linspace(1, 0, fade)
    if k > 0: pieces.append(np.zeros(int(KEEP * sr)))
    start_t = sum(len(p) for p in pieces) / sr
    pieces.append(seg); report.append((round(start_t, 3), round(start_t + len(seg) / sr, 3), round(a2, 3), round(b2, 3)))
res = np.concatenate(pieces)
sf.write(out, res, sr)
print(f"{inp.split('/')[-1]}: {len(y)/sr:.2f}s -> {len(res)/sr:.2f}s  ({len(merged)} phrases)  new spans:", report)

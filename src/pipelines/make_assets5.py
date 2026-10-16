"""v5 assets: for every picked LTX Hindi take -> voice-convert LTX's speech to the user's voice (Chatterbox S3Gen, ref voice/ref_C.wav),
advance audio 0.29 s (LTX mouth leads its own audio), trim dead lead/tail, upscale video to 1080p.
Writes film/assets/v5/<name>.mp4 (silent) and film/assets/a5/<name>.wav (clean, peak -3 dB, 40 ms fade-out, no compressor).
usage: make_assets5.py   (reads v5/picks_final.json)
"""
import sys, os, json, subprocess, numpy as np, soundfile as sf, librosa, mlx.core as mx
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio
from mlx_audio.tts.models.chatterbox.s3tokenizer import log_mel_spectrogram
from mlx_audio.tts.models.chatterbox.chatterbox import drop_invalid_tokens, SPEECH_VOCAB_SIZE
P = "@ULTRON_ROOT@"; FF = "@FFMPEG@"
V5 = f"{P}/v5"; OUTV = f"{P}/film/assets/v5"; OUTA = f"{P}/film/assets/a5"; W = f"{V5}/work"; ADV = 0.29
for d in (OUTV, OUTA, W): os.makedirs(d, exist_ok=True)
picks = json.load(open(f"{V5}/picks_final.json"))
if len(sys.argv) > 1: picks = {k: v for k, v in picks.items() if k in sys.argv[1:]}
m = load_model("@AVATAR@/models/chatterbox-mlx")
conds = m.prepare_conditionals(f"{P}/voice/ref_C.wav", 24000)
def run(*a): subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", *a], check=True)
report = []
for name, take in picks.items():
    src = f"{V5}/shots/{take}.mp4"
    if not os.path.exists(src): print("MISSING", name, take, flush=True); continue
    wav = f"{W}/{name}_ltx.wav"; run("-i", src, "-vn", "-ac", "1", "-ar", "24000", wav)
    w16 = load_audio(wav, sample_rate=16000)
    if w16.ndim == 2: w16 = w16.squeeze(0)
    mel = mx.expand_dims(log_mel_spectrogram(w16), 0); toks, _ = m._s3_tokenizer(mel, mx.array([mel.shape[2]]))
    t = np.asarray(drop_invalid_tokens(toks[0])).astype(np.int64); t = mx.array(t[t < SPEECH_VOCAB_SIZE])
    out = m.s3gen(speech_tokens=mx.expand_dims(t, 0), ref_dict=conds.gen, finalize=True)
    if out.ndim == 2: out = out.squeeze(0)
    mx.eval(out); conv = np.asarray(out, dtype=np.float32); sf.write(f"{W}/{name}_conv.wav", conv, 24000)
    # speech span from LTX's own audio (converted audio has identical timing)
    y, sr = sf.read(wav); y = y if y.ndim == 1 else y.mean(1)
    r = librosa.feature.rms(y=y, frame_length=int(sr * .03), hop_length=int(sr * .01))[0]; act = np.where(r > r.max() * 0.08)[0]
    ts, te = act[0] * 0.01, (act[-1] + 3) * 0.01
    ss = max(0.0, ts - ADV - 0.10); d = max(1.0, te - ss - ADV + 0.35)
    ca = conv / (np.abs(conv).max() + 1e-9) * 0.708                      # peak -3 dB, no compression
    sf.write(f"{W}/{name}_convn.wav", ca, 24000)
    run("-ss", f"{ss + ADV:.3f}", "-t", f"{d:.3f}", "-i", f"{W}/{name}_convn.wav", "-af", f"apad=whole_dur={d:.3f},afade=t=out:st={d-0.06:.3f}:d=0.05", "-ar", "48000", "-ac", "1", f"{OUTA}/{name}.wav")
    run("-ss", f"{ss:.3f}", "-t", f"{d:.3f}", "-i", src, "-an", "-vf", "scale=1920:1080:flags=lanczos,unsharp=5:5:0.5:5:5:0.0,fps=24", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", f"{OUTV}/{name}.mp4")
    report.append((name, round(ts, 2), round(te, 2), round(d, 2))); print(f"{name:5s} {take:18s} speech {ts:.2f}-{te:.2f}s -> clip {d:.2f}s", flush=True)
if len(sys.argv) == 1: json.dump(report, open(f"{V5}/assets5_report.json", "w"))
print("done", len(report))

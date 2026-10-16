"""Voice conversion with the local MLX Chatterbox (S3 tokenizer -> S3Gen with the target speaker's conditioning).
Keeps the SOURCE words, rhythm and timing; swaps the voice to the reference speaker.
usage: voiceconvert.py source.wav reference.wav out.wav
 - source: e.g. LTX's own generated speech (perfect lip-sync + pronunciation), reference: the user's Fish/real voice sample (>= 6 s clean)
Prints speaker-embedding cosine similarity (voice encoder) of source->ref and converted->ref, and duration drift.
"""
import sys, numpy as np, mlx.core as mx, soundfile as sf
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio, resample_audio
from mlx_audio.tts.models.chatterbox.s3tokenizer import log_mel_spectrogram
from mlx_audio.tts.models.chatterbox.chatterbox import drop_invalid_tokens, SPEECH_VOCAB_SIZE
src, ref, out = sys.argv[1:4]
m = load_model("@AVATAR@/models/chatterbox-mlx")
conds = m.prepare_conditionals(ref, 24000)
w16 = load_audio(src, sample_rate=16000)
if w16.ndim == 2: w16 = w16.squeeze(0)
mel = mx.expand_dims(log_mel_spectrogram(w16), 0)
toks, tlen = m._s3_tokenizer(mel, mx.array([mel.shape[2]]))
toks = toks[0]
toks = drop_invalid_tokens(toks)
_t = np.asarray(toks).astype(np.int64); toks = mx.array(_t[_t < SPEECH_VOCAB_SIZE])
wav = m.s3gen(speech_tokens=mx.expand_dims(toks, 0), ref_dict=conds.gen, finalize=True)
if wav.ndim == 2: wav = wav.squeeze(0)
mx.eval(wav)
y = np.asarray(wav, dtype=np.float32); sf.write(out, y, 24000)
def emb(p):
    w = load_audio(p, sample_rate=16000)
    e = m.ve.embeds_from_wavs([w], sample_rate=16000); e = np.asarray(mx.mean(e, axis=0)).ravel(); return e / (np.linalg.norm(e) + 1e-9)
er, es, eo = emb(ref), emb(src), emb(out)
print(f"source {len(w16)/16000:.2f}s -> converted {len(y)/24000:.2f}s | voice similarity to reference: source {float(er @ es):.2f} -> converted {float(er @ eo):.2f}")

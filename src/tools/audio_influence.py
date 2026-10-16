"""audio_influence.py src.mp4 audio.wav speech_t0 speech_t1 label=clip.mp4 ...
Does the clip's mouth follow the VOICE (not just the source video's own mouth)? delta = clip opening - src opening.
Reports mean delta inside speech / outside speech and corr(delta, voice RMS). Good: delta>0 in speech, ~0 outside, corr high."""
import sys, os, cv2, numpy as np, soundfile as sf, librosa
os.environ["OMP_NUM_THREADS"]="3"
from insightface.app import FaceAnalysis
fa=FaceAnalysis(name="buffalo_l",root="@AVATAR@/latentsync-mlx/checkpoints/auxiliary",allowed_modules=["detection","landmark_2d_106"],providers=["CPUExecutionProvider"]);fa.prepare(ctx_id=-1,det_size=(640,640))
def curve(path):
    cap=cv2.VideoCapture(path); op=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        s=f if f.shape[1]<=1100 else cv2.resize(f,(960,540)); fc=fa.get(s)
        if fc:
            b=max(fc,key=lambda q:(q.bbox[2]-q.bbox[0])*(q.bbox[3]-q.bbox[1])); p=b.landmark_2d_106; inn=p[64:72]
            op.append((inn[:,1].max()-inn[:,1].min())/(b.bbox[3]-b.bbox[1]))
        else: op.append(op[-1] if op else 0)
    return np.array(op)
src, wav, s0, s1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]); clips=[a.split("=",1) for a in sys.argv[5:]]
cs=curve(src); y,sr=sf.read(wav); y=y if y.ndim==1 else y.mean(1)
r=librosa.feature.rms(y=y,frame_length=int(sr/25*2),hop_length=int(sr/25))[0]
for lab,p in clips:
    c=curve(p); n=min(len(c),len(cs),len(r)); d=c[:n]-cs[:n]; a=r[:n]; sp=(np.arange(n)/25>=s0)&(np.arange(n)/25<=s1)
    dz=(d-d.mean())/(d.std()+1e-9); az=(a-a.mean())/(a.std()+1e-9)
    print(f"{lab:18s} delta in speech {d[sp].mean():+.4f} | outside {d[~sp].mean():+.4f} | corr(delta,voice) {np.corrcoef(dz,az)[0,1]:+.2f} | opening std {c[:n].std():.4f}")

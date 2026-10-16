"""ArcFace cosine similarity of every Nth frame of a video vs reference stills. usage: face_sim.py video step ref1 [ref2..]"""
import sys, os, subprocess, numpy as np, cv2
os.environ["OMP_NUM_THREADS"]="2"
from insightface.app import FaceAnalysis
root="@AVATAR@/latentsync-mlx/checkpoints/auxiliary"
fa=FaceAnalysis(name="buffalo_l",root=root,allowed_modules=["detection","recognition"],providers=["CPUExecutionProvider"]);fa.prepare(ctx_id=-1,det_size=(640,640))
def emb(im):
    f=fa.get(im)
    if not f: return None
    f=max(f,key=lambda q:(q.bbox[2]-q.bbox[0])*(q.bbox[3]-q.bbox[1])); e=f.normed_embedding; return e
vid,step=sys.argv[1],int(sys.argv[2]); refs=[emb(cv2.imread(r)) for r in sys.argv[3:]]; refs=[r for r in refs if r is not None]
cap=cv2.VideoCapture(vid); n=0; out=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    if n%step==0:
        e=emb(fr); out.append((n, None if e is None else round(float(max(e@r for r in refs)),2)))
    n+=1
print(os.path.basename(vid), out)

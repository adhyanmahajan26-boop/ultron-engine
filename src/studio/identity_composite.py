"""CPU VFX: preserve actual photographed facial pixels on a generated body plate."""
import argparse,json,subprocess,os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2'
import cv2,numpy as np,onnxruntime as ort
from insightface.app import FaceAnalysis
ap=argparse.ArgumentParser();ap.add_argument('--reference',required=True);ap.add_argument('--target',required=True);ap.add_argument('--output',required=True);ap.add_argument('--video',action='store_true');a=ap.parse_args()
root='@AVATAR@/latentsync-mlx/checkpoints/auxiliary'
assert (Path(root)/'models/buffalo_l/det_10g.onnx').exists()
opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
analyser=FaceAnalysis(name='buffalo_l',root=root,allowed_modules=['detection','recognition','landmark_2d_106'],providers=['CPUExecutionProvider'],sess_options=opt);analyser.prepare(ctx_id=-1,det_size=(640,640))
seg=ort.InferenceSession('@HOME@/.cache/hyperframes/background-removal/models/u2net_human_seg.onnx',sess_options=opt,providers=['CPUExecutionProvider'])
def getface(im):
 faces=analyser.get(im)
 if not faces:return None
 return max(faces,key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
def headmask(im,face,include_hair=True):
 h,w=im.shape[:2];mask=np.zeros((h,w),np.float32)
 hull=cv2.convexHull(face.landmark_2d_106.astype(np.int32));cv2.fillConvexPoly(mask,hull,1)
 x0,y0,x1,y1=face.bbox;fw=x1-x0;fh=y1-y0
 if not include_hair:
  mask=cv2.dilate(mask,np.ones((5,5),np.uint8));return cv2.GaussianBlur(mask,(0,0),max(.8,fw*.009))
 rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);inp=cv2.resize(rgb,(320,320)).astype(np.float32)/255;inp=((inp-[.485,.456,.406])/[.229,.224,.225]).transpose(2,0,1)[None].astype(np.float32)
 pred=seg.run(None,{seg.get_inputs()[0].name:inp})[0][0,0];pred=(pred-pred.min())/(pred.max()-pred.min()+1e-8);human=np.clip((cv2.resize(pred,(w,h))-.2)/.6,0,1)
 # Preserve hair above the eye line; chair and microphone stay outside this region.
 hair=np.zeros((h,w),np.float32);lx=max(0,int(x0-.35*fw));rx=min(w,int(x1+.35*fw));ty=max(0,int(y0-.65*fh));by=min(h,int(y0+.35*fh));hair[ty:by,lx:rx]=human[ty:by,lx:rx];mask=np.maximum(mask,hair)
 mask=cv2.dilate(mask,np.ones((5,5),np.uint8));mask=cv2.GaussianBlur(mask,(0,0),max(.8,fw*.009));return mask
source=cv2.imread(a.reference) if not a.video else None
sourcecap=cv2.VideoCapture(a.reference) if a.video else None
if a.video:
 ok,source=sourcecap.read();assert ok
sourceface=getface(source);assert sourceface is not None,'No reference face'
mask=headmask(source,sourceface,not a.video);reference_embedding=sourceface.normed_embedding
last_target_face=None;metrics=[]
def compose(target,src,srcface,smask):
 global last_target_face
 face=getface(target)
 if face is None:
  if a.video:return target
  raise RuntimeError('Target face not detected')
 else:last_target_face=face
 transform,_=cv2.estimateAffinePartial2D(srcface.kps,face.kps,method=cv2.LMEDS)
 if transform is None:raise RuntimeError('Face alignment failed')
 h,w=target.shape[:2];warped=cv2.warpAffine(src,transform,(w,h),flags=cv2.INTER_CUBIC);alpha=cv2.warpAffine(smask,transform,(w,h),flags=cv2.INTER_LINEAR);alpha=np.clip(alpha,0,1)
 out=np.clip(target*(1-alpha[...,None])+warped*alpha[...,None],0,255).astype(np.uint8)
 return out
if not a.video:
 target=cv2.imread(a.target);assert target is not None
 out=compose(target,source,sourceface,mask);cv2.imwrite(a.output,out);face=getface(out);score=float(np.dot(reference_embedding,face.normed_embedding)) if face is not None else None
 report={'output':a.output,'cosine_to_source':score,'method':'actual camera-face pixels + tracked alignment; no facial generation'};Path(a.output).with_suffix('.identity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
else:
 targetcap=cv2.VideoCapture(a.target);fps=targetcap.get(cv2.CAP_PROP_FPS);w=int(targetcap.get(3));h=int(targetcap.get(4));count=int(targetcap.get(7));sfps=sourcecap.get(cv2.CAP_PROP_FPS)
 F='@FFMPEG@';enc=subprocess.Popen([F,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p',a.output],stdin=subprocess.PIPE)
 for i in range(count):
  ok,target=targetcap.read()
  if not ok:break
  sourcecap.set(cv2.CAP_PROP_POS_MSEC,i/fps*1000);ok,newsource=sourcecap.read()
  if ok:
   newface=getface(newsource)
   if newface is not None:source=newsource;sourceface=newface;mask=headmask(source,sourceface,False)
  out=compose(target,source,sourceface,mask);enc.stdin.write(out.tobytes())
  if i%25==0:
   face=getface(out);metrics.append({'frame':i,'cosine_to_source':float(np.dot(reference_embedding,face.normed_embedding)) if face is not None else None});print(f'frame {i}/{count}',flush=True)
 enc.stdin.close();rc=enc.wait();assert rc==0;Path(a.output).with_suffix('.identity.json').write_text(json.dumps(metrics,indent=2));print(json.dumps({'output':a.output,'frames':i+1,'identity_samples':metrics}))

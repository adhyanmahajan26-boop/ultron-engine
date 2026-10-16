"""Priest (LTX, beat-prompted) 44 s trailer cut to songA_phonk excerpt 9.0-53.0 s; cuts on the track's strongest hits, white flashes, SFX, loudnorm."""
import subprocess, sys
P="@ULTRON_ROOT@"; D=f"{P}/priest"; FF="@FFMPEG@"; SFX=f"{P}/film/assets/a4"
EL=sys.argv[1] if len(sys.argv)>1 else "elder_s711"
UP="scale=1920:1080:flags=lanczos,unsharp=5:5:0.4:5:5:0.0,fps=24,format=yuv420p,setsar=1"
T0=9.0; TOT=44.0
S=[("priest_s711",0,4.04,0.0,4.93,False),("F1_s701",0,4.04,4.93,8.37,True),("S06_s701",0,2.4,8.37,10.08,True),("F4_s711",0,3.0,10.08,12.64,True),
   ("F3_s711",0,3.0,12.64,15.2,False),("F5_s711",0,3.0,15.2,17.78,False),("S09_s711",0,3.5,17.78,21.2,True),("S07_s711",0,2.6,21.2,23.8,False),
   ("S08_s712",0,2.6,23.8,26.35,False),("F8_s711",0,3.6,26.35,29.9,True),("S12_s711",0,2.0,29.9,31.5,False),("F10_s712",0,3.4,31.5,34.93,True),
   ("S11_s711",0,2.6,34.93,37.5,True),(EL,0,4.04,37.5,TOT,False)]
ops,ins=[],[]
for i,(n,s0,sl,t0,t1,fl) in enumerate(S):
    dur=t1-t0; sp=dur/sl; ins+=["-i",f"{D}/shots/{n}.mp4"]
    slow="minterpolate=fps=24:mi_mode=blend," if sp>1.25 else ""
    flash=",fade=t=in:st=0:d=0.1:color=white" if fl else ""
    ops.append(f"[{i}:v]trim={s0}:{s0+sl},setpts=(PTS-STARTPTS)*{sp:.4f},{slow}{UP}{flash},trim=0:{dur:.3f}[v{i}]")
ops.append("".join(f"[v{i}]" for i in range(len(S)))+f"concat=n={len(S)}:v=1:a=0,fade=t=in:st=0:d=0.8,fade=t=out:st={TOT-1.5}:d=1.5[vout]")
sfx=[("boom.wav",4.93,1.0),("boom.wav",8.37,0.8),("thud.wav",10.08,0.8),("thud.wav",12.64,0.7),("whoosh.wav",17.6,0.5),("boom.wav",17.78,0.9),("thud.wav",21.2,0.6),("boom.wav",26.35,0.9),("thud.wav",31.5,0.8),("boom.wav",34.93,1.0),("whoosh_long.wav",36.9,0.5)]
ai=len(S); ins+=["-ss",str(T0),"-t",str(TOT),"-i",f"{D}/audio/songA_phonk.wav"]
af=[f"[{ai}:a]aresample=48000,volume=0.9,afade=t=in:st=0:d=0.8,afade=t=out:st={TOT-2}:d=2[mus]"]; mix=["[mus]"]
for j,(f,at,v) in enumerate(sfx):
    ins+=["-i",f"{SFX}/{f}"]; ms=int(at*1000)
    af.append(f"[{ai+1+j}:a]aresample=48000,volume={v},adelay={ms}|{ms}[s{j}]"); mix.append(f"[s{j}]")
af.append("".join(mix)+f"amix=inputs={len(mix)}:normalize=0,alimiter=limit=0.95[aout]")
subprocess.run([FF,"-hide_banner","-loglevel","error","-y"]+ins+["-filter_complex",";".join(ops+af),"-map","[vout]","-map","[aout]","-t",str(TOT),"-c:v","libx264","-crf","16","-pix_fmt","yuv420p","-c:a","aac","-b:a","224k",f"{D}/review/priest_raw.mp4"],check=True)
subprocess.run([FF,"-hide_banner","-loglevel","error","-y","-i",f"{D}/review/priest_raw.mp4","-c:v","copy","-af","loudnorm=I=-15:TP=-1.5:LRA=11","-ar","48000","-c:a","aac","-b:a","224k",f"{D}/review/priest_ltx_v1.mp4"],check=True)
subprocess.run([FF,"-hide_banner","-loglevel","error","-y","-i",f"{D}/review/priest_ltx_v1.mp4","-vf","scale=1280:-2","-c:v","libx264","-crf","24","-pix_fmt","yuv420p","-c:a","aac","-b:a","160k","-movflags","+faststart",f"{D}/review/priest_ltx_v1_phone.mp4"],check=True)
print("built")

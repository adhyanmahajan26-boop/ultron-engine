"""groupreel.py out.mp4 name1 name2 ...  -> concatenates ls3s windows with their audio (720p)"""
import sys, subprocess
FF="@FFMPEG@"; R="@ULTRON_ROOT@"
out=sys.argv[1]; names=sys.argv[2:]; args=[]; fc=[]
for i,n in enumerate(names):
    args+=["-i",f"{R}/ls3s/{n}.mp4"]
    fc.append(f"[{i}:v]scale=960:540,fps=25,format=yuv420p[v{i}];[{i}:a]aresample=44100,aformat=channel_layouts=mono[a{i}]")
fc.append("".join(f"[v{i}][a{i}]" for i in range(len(names)))+f"concat=n={len(names)}:v=1:a=1[v][a]")
subprocess.run([FF,"-hide_banner","-loglevel","error","-y",*args,"-filter_complex",";".join(fc),"-map","[v]","-map","[a]","-c:v","libx264","-crf","21","-c:a","aac",out],check=True)

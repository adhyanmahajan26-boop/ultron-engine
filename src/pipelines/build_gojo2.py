"""Gojo vs Sukuna: 48 s cut. Intro 0-16.51 s slow (S01-S03), fight cut on the track's strongest percussive hits, white flashes + SFX, loudnorm -14."""
import subprocess
P = "@ULTRON_ROOT@"; D = f"{P}/gojo"; FF = "@FFMPEG@"
SFX = f"{P}/film/assets/a4"
UP = "scale=1920:1080:flags=lanczos,unsharp=5:5:0.4:5:5:0.0,fps=24,format=yuv420p,setsar=1"
# (take, source start, source seconds used, timeline start, timeline end, flash-in)
H = [16.51, 23.13, 24.57, 26.43, 29.23, 32.65, 36.52, 39.0, 41.21, 42.84, 45.5, 48.0]
S = [("S01_s801", 0, 4.04, 0.0, 6.5, False), ("S02_s801", 0, 3.04, 6.5, 11.5, False), ("S03_s801", 0, 3.04, 11.5, 16.51, False),
     ("S04_h901", 0, 3.04, 16.51, 19.55, True), ("S05_s801", 0, 3.04, 19.55, 23.13, False), ("S06_h901", 0.9, 2.1, 23.13, 24.57, True),
     ("S07_h901", 0, 2.6, 24.57, 26.43, False), ("S08_h901", 0, 2.9, 26.43, 29.23, True), ("S09_h901", 0, 3.5, 29.23, 32.65, True),
     ("S10_s801", 0, 3.87, 32.65, 36.52, False), ("S11_s801", 0, 2.48, 36.52, 39.0, False), ("S12_h901", 1.5, 2.2, 39.0, 41.21, True),
     ("S13_s802", 0, 1.63, 41.21, 42.84, True), ("S14_s802", 0, 2.66, 42.84, 45.5, True), ("S15_s802", 0, 2.5, 45.5, 48.0, False)]
ops, ins = [], []
for i, (n, s0, sl, t0, t1, fl) in enumerate(S):
    dur = t1 - t0; sp = dur / sl; ins += ["-i", f"{D}/shots/{n}.mp4"]
    slow = "minterpolate=fps=24:mi_mode=blend," if sp > 1.25 else ""
    flash = ",fade=t=in:st=0:d=0.12:color=white" if fl else ""
    ops.append(f"[{i}:v]trim={s0}:{s0 + sl},setpts=(PTS-STARTPTS)*{sp:.4f},{slow}{UP}{flash},trim=0:{dur:.3f}[v{i}]")
ops.append("".join(f"[v{i}]" for i in range(len(S))) + f"concat=n={len(S)}:v=1:a=0[cat]")
# mid-clip impact flashes on strong hits + fade in/out
mid = [24.57, 25.18, 25.81, 34.26, 37.13, 37.95]
dbx = ",".join(f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.55:t=fill:enable='between(t,{t:.2f},{t + 0.07:.2f})'" for t in mid)
ops.append(f"[cat]{dbx},fade=t=in:st=0:d=0.8,fade=t=out:st=46.6:d=1.4[vout]")
# audio
sfx = [("boom.wav", 16.51, 0.8), ("whoosh.wav", 19.4, 0.5), ("boom.wav", 23.13, 0.8), ("thud.wav", 24.57, 0.7), ("thud.wav", 25.81, 0.7), ("boom.wav", 26.43, 0.9),
       ("boom.wav", 29.23, 1.0), ("whoosh.wav", 32.4, 0.6), ("thud.wav", 34.26, 0.8), ("thud.wav", 37.13, 0.8), ("whoosh_long.wav", 38.7, 0.6), ("boom.wav", 41.21, 1.0), ("whoosh.wav", 42.6, 0.5)]
ai = len(S); ins += ["-i", f"{D}/audio/track.wav"]; af = [f"[{ai}:a]aresample=48000,volume=0.9[mus]"]; mix = ["[mus]"]
for j, (f, at, v) in enumerate(sfx):
    ins += ["-i", f"{SFX}/{f}"]; ms = int(at * 1000)
    af.append(f"[{ai + 1 + j}:a]aresample=48000,volume={v},adelay={ms}|{ms}[s{j}]"); mix.append(f"[s{j}]")
af.append("".join(mix) + f"amix=inputs={len(mix)}:normalize=0,alimiter=limit=0.95[aout]")
subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y"] + ins + ["-filter_complex", ";".join(ops + af), "-map", "[vout]", "-map", "[aout]", "-t", "48.0",
               "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "224k", f"{D}/review/gojo_v2_raw.mp4"], check=True)
subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", f"{D}/review/gojo_v2_raw.mp4", "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "aac", "-b:a", "224k", f"{D}/review/gojo_v2.mp4"], check=True)
subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", f"{D}/review/gojo_v2.mp4", "-vf", "scale=1280:-2", "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", f"{D}/review/gojo_v2_phone.mp4"], check=True)
print("built")

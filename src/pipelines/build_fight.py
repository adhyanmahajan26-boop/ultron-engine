"""Assemble the 45 s anime fight: 0-20 s slow sad opening (5 shots, slowed + dissolves), 20-45 s fight cut on the beat, white impact flashes,
teleport eagle SFX at the swap moments, impact hits, music = fight_mix.wav. Output review/fight_v1.mp4 (+ phone copy)."""
import subprocess, json, numpy as np, soundfile as sf, librosa, os, sys
D = "@ULTRON_ROOT@/fight"; FF = "@FFMPEG@"
SFXD = "@ULTRON_ROOT@/film/assets/a4"
UP = "scale=1920:1080:flags=lanczos,unsharp=5:5:0.4:5:5:0.0,fps=24,format=yuv420p,setsar=1"
# ---------------- beat grid of the fight half
y, sr = sf.read(f"{D}/audio/fight_mix.wav"); m = (y.mean(1) if y.ndim == 2 else y).astype(np.float32)
seg = m[int(20.0 * sr):]; tempo, bt = librosa.beat.beat_track(y=seg, sr=sr, units="time", start_bpm=150, tightness=120)
tempo = float(np.atleast_1d(tempo)[0]); per = 60.0 / tempo if 100 < tempo < 220 else 0.4
if len(bt) > 8: per = float(np.median(np.diff(bt)))
print(f"fight tempo {tempo:.0f} bpm, beat period {per:.3f}s, first beats after 20 s:", [round(float(b), 2) for b in bt[:5]])
grid = lambda k: 20.0 + k * per
# ---------------- opening: 5 shots, each slowed a touch so they fill exactly 20 s with 0.25 s dissolves
opening = [("O1", 601), ("O2", 601), ("O3", 602), ("O4", 601), ("O5", 602)]
XD = 0.25; L = (20.0 + 4 * XD) / 5.0
ops = []
for i, (n, sd) in enumerate(opening):
    ops.append(f"[{i}:v]trim=0:4.04,setpts=(PTS-STARTPTS)*{L / 4.04:.5f},{UP}[o{i}]")
# ---------------- fight: (clip, start-in-source, source seconds used, beats, flash?)
fight = [("F1_s601", 0.0, 3.0, 6, False), ("F2_s601", 0.0, 1.2, 3, True), ("F3_s601", 0.0, 2.6, 6, False), ("F4_s601", 0.0, 3.0, 6, False), ("F5_s602", 0.0, 3.0, 6, True),
         ("F6_s601", 0.0, 1.6, 4, False), ("F7_s602", 0.0, 3.0, 6, False), ("F8_s601", 0.0, 3.0, 4, False), ("F9b_s612", 0.0, 3.0, 6, True), ("F10_s602", 0.0, 3.0, 12, True)]
tot_beats = sum(f[3] for f in fight); scale = 25.0 / (tot_beats * per)
durs = [f[3] * per * scale for f in fight]; print("fight durations", [round(d, 2) for d in durs], "sum", round(sum(durs), 2))
fi = []; t = 20.0; starts = []
for k, ((n, s0, sl, bts, fl), dur) in enumerate(zip(fight, durs)):
    idx = 5 + k; sp = dur / sl
    flash = ",fade=t=in:st=0:d=0.14:color=white" if fl else ""
    fi.append(f"[{idx}:v]trim={s0}:{s0 + sl},setpts=(PTS-STARTPTS)*{sp:.4f},{UP}{flash}[f{k}]"); starts.append(t); t += dur
ops += fi
# chain: opening dissolves then hard cuts for the fight
chain = "[o0]"; run = L
for i in range(1, 5):
    chain_out = f"[x{i}]"; ops.append(f"{chain}[o{i}]xfade=transition=fade:duration={XD}:offset={run - XD:.3f}{chain_out}"); chain = chain_out; run += L - XD
ops.append(f"{chain}trim=0:20.0,setpts=PTS-STARTPTS[open]")
ops.append("".join(f"[f{k}]" for k in range(len(fight))) + f"concat=n={len(fight)}:v=1:a=0[fightv]")
ops.append("[open][fightv]concat=n=2:v=1:a=0,fade=t=out:st=43.8:d=1.2[vout]")
inputs = []
for n, sd in opening: inputs += ["-i", f"{D}/shots/{n}_s{sd}.mp4"]
for n, *_ in fight: inputs += ["-i", f"{D}/shots/{n}.mp4"]
# ---------------- audio: music + sfx
sfx = [("SFX_teleport_FINAL.wav", starts[1] - 0.05, 1.0), ("SFX_teleport_FINAL.wav", starts[8] - 0.05, 1.0), ("SFX_teleport_FINAL.wav", starts[5] - 0.05, 0.7),
       ("boom.wav", starts[0] + 0.02, 0.7), ("boom.wav", starts[4], 0.8), ("thud.wav", starts[7], 0.6), ("boom.wav", starts[9], 1.0), ("whoosh.wav", starts[2] - 0.05, 0.6), ("whoosh.wav", starts[3] - 0.05, 0.6)]
ai = len(opening) + len(fight); afl = [f"[{ai}:a]aresample=48000,volume=0.9[mus]"]; mix = ["[mus]"]
for j, (f, at, vol) in enumerate(sfx):
    p = f"{D}/audio/{f}" if f.startswith("SFX") else f"{SFXD}/{f}"; inputs += ["-i", p]
    afl.append(f"[{ai + 1 + j}:a]aresample=48000,volume={vol},adelay={int(max(0, at) * 1000)}|{int(max(0, at) * 1000)}[s{j}]"); mix.append(f"[s{j}]")
afl.append("".join(mix) + f"amix=inputs={len(mix)}:normalize=0,alimiter=limit=0.95[aout]")
inputs.insert(2 * 0, "-i"); inputs.pop(2 * 0)   # no-op safeguard
# music input must sit at index ai: add after videos
vin = inputs[: 2 * ai]; ain = [("-i", f"{D}/audio/fight_mix.wav")]
rest = inputs[2 * ai:]
cmd = [FF, "-hide_banner", "-loglevel", "error", "-y"] + vin + ["-i", f"{D}/audio/fight_mix.wav"] + rest + ["-filter_complex", ";".join(ops + afl), "-map", "[vout]", "-map", "[aout]", "-t", "45.0",
       "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "224k", f"{D}/review/fight_raw.mp4"]
subprocess.run(cmd, check=True)
subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", f"{D}/review/fight_raw.mp4", "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "aac", "-b:a", "224k", f"{D}/review/fight_v1.mp4"], check=True)
subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", f"{D}/review/fight_v1.mp4", "-vf", "scale=1280:-2", "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", f"{D}/review/fight_v1_phone.mp4"], check=True)
print("built", [(f[0], round(s, 2)) for f, s in zip(fight, starts)])

"""Shorten awkward mid-line pauses (> MAXGAP) in a lip-synced clip: the paused stretch of VIDEO is sped up (smooth, no cut) and the AUDIO silence trimmed to TARGET.
usage: trim_pauses.py name [name ...]     reads/writes film/assets/v5/<name>.mp4 + film/assets/a5/<name>.wav (originals kept in v5/pre_trim/)
"""
import sys, os, shutil, subprocess, numpy as np, soundfile as sf, librosa
P = "@ULTRON_ROOT@"; FF = "@FFMPEG@"
MAXGAP, TARGET, PAD = 0.30, 0.20, 0.04
os.makedirs(f"{P}/v5/pre_trim", exist_ok=True)
for n in sys.argv[1:]:
    va, aa = f"{P}/film/assets/v5/{n}.mp4", f"{P}/film/assets/a5/{n}.wav"
    if not os.path.exists(f"{P}/v5/pre_trim/{n}.mp4"): shutil.copy(va, f"{P}/v5/pre_trim/{n}.mp4"); shutil.copy(aa, f"{P}/v5/pre_trim/{n}.wav")
    va0, aa0 = f"{P}/v5/pre_trim/{n}.mp4", f"{P}/v5/pre_trim/{n}.wav"
    y, sr = sf.read(aa0); y = y if y.ndim == 1 else y.mean(1)
    r = librosa.feature.rms(y=y, frame_length=int(sr * .03), hop_length=int(sr * .01))[0]; sp = r > r.max() * 0.06; idx = np.where(sp)[0]
    a, b = idx[0], idx[-1]; gaps = []; i = a
    while i < b:
        if not sp[i]:
            j = i
            while j < b and not sp[j]: j += 1
            if (j - i) * 0.01 > MAXGAP: gaps.append((i * 0.01 + PAD, j * 0.01 - PAD))
            i = j
        else: i += 1
    if not gaps: print(n, "no long pauses"); continue
    dur = len(y) / sr; cuts = [0.0]; segs = []                       # (start, end, kind)
    for g0, g1 in gaps: segs.append((cuts[-1], g0, "s")); segs.append((g0, g1, "g")); cuts.append(g1)
    segs.append((cuts[-1], dur, "s"))
    vf, af, vl, al = [], [], [], []
    for k, (s0, s1, kind) in enumerate(segs):
        if kind == "s":
            vf.append(f"[0:v]trim={s0:.3f}:{s1:.3f},setpts=PTS-STARTPTS[v{k}]"); af.append(f"[1:a]atrim={s0:.3f}:{s1:.3f},asetpts=PTS-STARTPTS[a{k}]")
        else:
            f = (s1 - s0) / TARGET
            vf.append(f"[0:v]trim={s0:.3f}:{s1:.3f},setpts=(PTS-STARTPTS)/{f:.4f}[v{k}]"); af.append(f"[1:a]atrim={s0:.3f}:{s0 + TARGET:.3f},asetpts=PTS-STARTPTS[a{k}]")
        vl.append(f"[v{k}]"); al.append(f"[a{k}]")
    fv = ";".join(vf) + f";{''.join(vl)}concat=n={len(segs)}:v=1:a=0,fps=24[vo]"
    fa = ";".join(x.replace("[1:a]", "[0:a]") for x in af) + f";{''.join(al)}concat=n={len(segs)}:v=0:a=1[ao]"
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", va0, "-filter_complex", fv, "-map", "[vo]", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", va], check=True)
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", aa0, "-filter_complex", fa, "-map", "[ao]", "-ar", "48000", "-ac", "1", aa], check=True)
    ny = sf.info(aa).duration
    print(f"{n}: {len(gaps)} pause(s) {[round(g1-g0+2*PAD,2) for g0,g1 in gaps]} -> {TARGET:.2f}s each; clip {dur:.2f}s -> {ny:.2f}s")

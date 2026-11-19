"""Uncut clone: Addy's real clip + cloned voice -> ONE continuous lip-synced take, any length, no cuts.

  python src/lipsync/clone.py --clip addy.mp4 (--voice voice.wav | --text "…") --out-dir work/clone \
      [--start 0] [--seconds N] [--backend auto] [--version 1.6] [--sweep quick|full|none]

Length is not limited: the whole clip (or --start/--seconds of it) is one take. Quality first:
the sweep runs on a short excerpt only to choose settings, the final render is the full take.

Steps (each writes into --out-dir, re-running skips finished steps):
  1 base     trim to one continuous stretch, 25 fps, keep full resolution; check it (face size,
             frontal share, head steadiness, is the mouth already talking?) and pick ref settings
  2 voice    use --voice, or make it with Fish (avatar scripts/voice_clone.py, Addy's reference);
             loudness-normalise; pad with silence to the clip length so the take ends with a
             closed mouth; refuse a voice longer than the clip (that would need loops = cuts)
  3 sweep    on the 2.4 s with the most speech: guidance x ref-mode grid, scored by QA, plus a
             contact sheet to judge by eye (metric-tuned settings looked cartoonish before)
  4 render   full take with the best settings + the silent-audio control (same settings)
  5 qa       verdict, metrics, full-face contact sheet, REPORT.md
  (6 polish) optional --smooth K: src/tools/smooth_mouth.py temporal blend inside the mouth box only
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lipsync import config, qa  # noqa: E402
from lipsync.backends import Job, run_job  # noqa: E402
from lipsync.run import heavy_guard  # noqa: E402

FF = config.ffmpeg()
SWEEPS = {
    "none": [],
    "quick": [(1.5, "same"), (2.0, "window"), (2.5, "window")],
    "full": [(g, r) for g in (1.5, 2.0, 2.5, 3.0) for r in ("same", "window", "fixed")],
}


def sh(cmd):
    print("  $", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def probe_duration(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout.strip()
    return float(out or 0)


# ----------------------------------------------------------------------------- 1 base
def make_base(clip: Path, start: float, seconds: float, d: Path) -> Path:
    base = d / "base.mp4"
    if not base.exists():
        sh([FF, "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", clip, "-t", f"{seconds:.3f}", "-r", "25",
            "-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p", "-an", base])
    return base


def check_base(base: Path) -> dict:
    A = qa.analyse(str(base), with_embed_every=0)
    boxes = [b for b in A["boxes"] if b]
    n = len(A["boxes"])
    widths = np.array([b[3][2] - b[3][0] for b in boxes]) if boxes else np.array([0])
    cx = np.array([b[0] for b in boxes]) if boxes else np.array([0])
    cy = np.array([b[1] for b in boxes]) if boxes else np.array([0])
    o = A["open"]
    talking = float(np.mean(o > np.percentile(o, 10) + 0.02))
    rep = {"frames": n, "face_found": round(len(boxes) / max(n, 1), 3), "face_width_px_median": int(np.median(widths)),
           "head_move_px_std": round(float(np.hypot(cx.std(), cy.std())), 1),
           "mouth_open_mean": round(float(o.mean()), 4), "mouth_open_std": round(float(o.std()), 4),
           "share_frames_mouth_open": round(talking, 2), "warnings": []}
    w = rep["warnings"]
    if rep["face_found"] < 0.98:
        w.append("face lost in some frames - LatentSync stops on a frame with no face")
    if rep["face_width_px_median"] < 300:
        w.append(f"face is small ({rep['face_width_px_median']} px wide); 1.6 works on a 512 crop, aim for 400+ px")
    if rep["head_move_px_std"] > 40:
        w.append("a lot of head movement - expect more mouth artefacts; steadier footage syncs better")
    if rep["mouth_open_std"] > 0.03 or talking > 0.4:
        w.append("the source mouth is already moving/talking - using a non-overlapping reference window "
                 "(ref-mode window) so the model does not copy the old mouth")
    # suggested ref mode: closed/quiet source -> upstream 'same' is fine, talking source -> 'window'
    rep["suggested_ref_mode"] = "window" if (rep["mouth_open_std"] > 0.02 or talking > 0.3) else "same"
    return rep


# ----------------------------------------------------------------------------- 2 voice
def make_voice(a, base_len: float, d: Path) -> Path:
    raw = d / "voice_raw.wav"
    if a.voice:
        sh([FF, "-loglevel", "error", "-y", "-i", a.voice, "-ac", "1", "-ar", "44100", raw])
    elif a.text:
        av = config.avatar_dir()
        ref = a.voice_ref or str(av / "inputs" / "my_voice_ref.wav")
        cmd = [av / ".venv" / "bin" / "python", "scripts/voice_clone.py", "--ref-audio", ref, "--text", a.text,
               "--out", raw]
        if a.speed:
            cmd += ["--speed", str(a.speed)]
        print("  $", " ".join(map(str, cmd)))
        subprocess.run([str(c) for c in cmd], check=True, cwd=av)
    else:
        raise SystemExit("give --voice WAV or --text")
    dur = probe_duration(raw)
    if dur > base_len + 0.05:
        raise SystemExit(f"voice is {dur:.2f}s but the clip is {base_len:.2f}s. An uncut take needs footage at least "
                         f"as long as the voice: shorten the line (tools/tighten.py) or give a longer clip.")
    voice = d / "voice.wav"
    # loudness-normalise, then pad with silence to the exact clip length (mouth closes after the line)
    sh([FF, "-loglevel", "error", "-y", "-i", raw, "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,apad,atrim=0:{base_len:.3f}",
        "-ar", "44100", "-ac", "1", voice])
    return voice


def densest_window(voice: Path, seconds: float = 2.4) -> float:
    import librosa
    import soundfile as sf

    y, sr = sf.read(str(voice))
    y = y if y.ndim == 1 else y.mean(1)
    r = librosa.feature.rms(y=y, frame_length=2048, hop_length=sr // 25)[0]
    k = int(seconds * 25)
    if len(r) <= k:
        return 0.0
    s = np.convolve(r, np.ones(k), "valid")
    return float(np.argmax(s) / 25.0)


def cut(src: Path, t0: float, dur: float, dst: Path, audio=False):
    if dst.exists():
        return dst
    args = [FF, "-loglevel", "error", "-y", "-ss", f"{t0:.3f}", "-i", src, "-t", f"{dur:.3f}"]
    args += (["-ac", "1", "-ar", "44100"] if audio else ["-r", "25", "-c:v", "libx264", "-crf", "12", "-an"])
    sh(args + [dst])
    return dst


# ----------------------------------------------------------------------------- 3 sweep + 4 render
def job(a, video, audio, out, g, ref, **kw) -> Job:
    return Job(video=str(video), audio=str(audio), out=str(out), version=a.version, guidance=g, steps=a.steps,
               seed=a.seed, ref_mode=ref, ref_idx=("auto" if ref == "fixed" else None), backend=a.backend, **kw)


def score(m: dict) -> float:
    """higher is better; identity is a hard floor, and an over-moving mouth vs the source is penalised
    (that is what the cartoon 'duck pout' looked like in numbers)."""
    idm = m["identity"]["mean"] or 0
    if idm < qa.THRESH["identity_min"]:
        return -1 + idm
    s = m["voice_corr_best"] + 0.5 * m["delta_voice_corr"] + 0.5 * (idm - 0.8)
    if abs(m["best_lag_ms"]) > 80:
        s -= 0.3
    return round(s, 3)


def sweep(a, base, voice, d: Path, report: dict):
    grid = SWEEPS[a.sweep]
    if not grid:
        return a.guidance or 2.0, a.ref_mode or report["base"]["suggested_ref_mode"]
    t0 = densest_window(voice)
    sd = d / "sweep"
    sd.mkdir(exist_ok=True)
    vb, vv = cut(base, t0, 2.4, sd / "ex_base.mp4"), cut(voice, t0, 2.4, sd / "ex_voice.wav", audio=True)
    rows, results = [("source", str(vb))], []
    for g, ref in grid:
        out = sd / f"ex_g{g}_{ref}.mp4"
        if not out.exists():
            heavy_guard()
            run_job(job(a, vb, vv, out, g, ref), log=str(d / "lipsync.log"))
        m = qa.evaluate(str(vb), str(vv), str(out), run_syncnet=False)
        results.append({"guidance": g, "ref_mode": ref, "score": score(m), **{k: m[k] for k in (
            "voice_corr_best", "best_lag_ms", "delta_voice_corr", "mouth_std", "source_follow_corr")},
            "identity": m["identity"]["mean"]})
        rows.append((f"g{g} {ref}  score {score(m)}", str(out)))
        print(json.dumps(results[-1]), flush=True)
    qa.contact_sheet(rows, str(d / "sweep_sheet.jpg"), every=6, width=200)
    best = max(results, key=lambda r: r["score"])
    report["sweep"] = {"excerpt_start_s": round(t0, 2), "results": results, "picked": best}
    return best["guidance"], best["ref_mode"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--voice")
    ap.add_argument("--text")
    ap.add_argument("--voice-ref", default=None)
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=None, help="default: the whole clip from --start")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--version", default="1.6")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1247)
    ap.add_argument("--guidance", type=float, default=None, help="skip the sweep's guidance choice")
    ap.add_argument("--ref-mode", default=None, help="skip the sweep's ref choice")
    ap.add_argument("--sweep", default="quick", choices=sorted(SWEEPS))
    ap.add_argument("--smooth", type=float, default=0.0, help="mouth-only temporal smoothing 0..1 (0 = off)")
    a = ap.parse_args()
    d = Path(a.out_dir).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    report = {"clip": a.clip, "backend": a.backend, "version": a.version}

    clip_len = probe_duration(Path(a.clip))
    seconds = clip_len - a.start if a.seconds is None else min(a.seconds, clip_len - a.start)
    base = make_base(Path(a.clip), a.start, seconds, d)
    report["base"] = check_base(base)
    print(json.dumps(report["base"], indent=1), flush=True)
    voice = make_voice(a, probe_duration(base), d)

    if a.guidance is not None and a.ref_mode:
        g, ref = a.guidance, a.ref_mode
    else:
        g, ref = sweep(a, base, voice, d, report)
        g = a.guidance if a.guidance is not None else g
        ref = a.ref_mode or ref
    report["final_settings"] = {"guidance": g, "ref_mode": ref, "steps": a.steps, "seed": a.seed}

    take = d / "take.mp4"
    if not take.exists():
        heavy_guard()
        r = run_job(job(a, base, voice, take, g, ref), log=str(d / "lipsync.log"))
        report["render"] = r["report"]
    sil_wav = Path(qa.silent_wav(str(voice), str(d / "silence.wav")))
    sil = d / "take_silent.mp4"
    if not sil.exists():
        heavy_guard()
        r = run_job(job(a, base, sil_wav, sil, g, ref), log=str(d / "lipsync.log"))
        report["silent_render"] = r["report"]

    final = take
    if a.smooth > 0:
        sm = config.engine_dir() / "runtime" / "tools" / "smooth_mouth.py"
        if sm.exists():
            final = d / "take_smooth.mp4"
            sh([sys.executable, sm, take, voice, final, a.smooth])
        else:
            print("[warn] runtime/tools/smooth_mouth.py not rendered (run ultron.py setup); skipping --smooth")

    m = qa.evaluate(str(base), str(voice), str(final), str(sil))
    report["qa"] = m
    qa.contact_sheet([("source", str(base)), ("clone", str(final)), ("silent-audio control", str(sil))],
                     str(d / "qa_sheet.jpg"), every=25, width=220)
    deliver = d / "clone.mp4"
    shutil.copy(final, deliver)
    (d / "report.json").write_text(json.dumps(report, indent=1, default=str))
    s = m.get("silent", {})
    md = [f"# Uncut clone ({probe_duration(base):.1f} s): {m['verdict']}", "",
          f"- settings: LatentSync {a.version}, guidance {g}, ref-mode {ref}, steps {a.steps}, backend {a.backend}",
          f"- voice follow: corr {m['voice_corr_best']} at {m['best_lag_ms']} ms; voice-driven delta corr {m['delta_voice_corr']}",
          f"- silent-audio control: speech-vs-silent mouth corr {s.get('speech_vs_silent_corr')} (low is good), "
          f"silent movement ratio {s.get('silent_open_ratio')}",
          f"- identity (ArcFace vs source): {m['identity']}",
          f"- SyncNet: {m.get('syncnet')}",
          "", "Problems:" if m["why"] else "No automatic problems found. Still look at qa_sheet.jpg before sending.",
          *[f"- {w}" for w in m["why"]], "",
          "Files: clone.mp4, take_silent.mp4, qa_sheet.jpg, sweep_sheet.jpg, report.json"]
    (d / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

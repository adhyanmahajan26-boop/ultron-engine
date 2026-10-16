#!/usr/bin/env python3
"""Ultron Engine CLI. Standard library only, so it runs straight after `git clone`.

  python3 ultron.py setup      interactive: local / cloud GPU / mixed, per component, asks before EVERY download
  python3 ultron.py doctor     check tools, config and downloaded models
  python3 ultron.py models     list the model catalog with sizes and licences
  python3 ultron.py render     (re)generate runtime/ from src/ using ultron.config.json
  python3 ultron.py download KEY [KEY ...]   download specific components (asks to confirm)

Non-interactive (what an AI assistant should use after asking the user):
  python3 ultron.py setup --mode mixed --local video_ltx,image_qwen --cloud "video_h3=https://my-gpu/endpoint|MY_KEY_ENV" --download video_ltx
Nothing is ever downloaded unless the user said yes (interactively) or listed it in --download.
"""
import argparse, fnmatch, json, os, platform, shutil, subprocess, sys, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
CAT = json.loads((REPO / "ultron" / "catalog.json").read_text())
CFG_PATH = REPO / "ultron.config.json"


# ----------------------------------------------------------------- small helpers
def ask(q, choices=None, default=None):
    hint = f" [{'/'.join(choices)}]" if choices else ""
    d = f" (default {default})" if default else ""
    while True:
        a = input(f"{q}{hint}{d}: ").strip().lower() or (default or "")
        if not choices or a in choices:
            return a
        print("  please answer one of:", ", ".join(choices))


def yes(q, default="n"):
    return ask(q, ["y", "n"], default) == "y"


def hardware():
    h = {"os": platform.system(), "machine": platform.machine(), "ram_gb": None, "nvidia": None}
    try:
        if h["os"] == "Darwin":
            h["ram_gb"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) // 2**30
        elif h["os"] == "Linux":
            h["ram_gb"] = int([l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]) // 2**20
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            h["nvidia"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
        except Exception:
            pass
    h["apple_silicon"] = h["os"] == "Darwin" and h["machine"] == "arm64"
    return h


def hf_size_gb(repo, include=None, exclude=None):
    """Real size from the Hugging Face API (no download)."""
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=15) as r:
            files = json.load(r)["siblings"]
        tot = 0
        for f in files:
            n = f["rfilename"]
            if include and not any(fnmatch.fnmatch(n, p) for p in include):
                continue
            if exclude and any(fnmatch.fnmatch(n, p) for p in exclude):
                continue
            tot += f.get("size") or 0
        return round(tot / 1e9, 1)
    except Exception:
        return None


def load_cfg():
    return json.loads(CFG_PATH.read_text()) if CFG_PATH.exists() else None


# ----------------------------------------------------------------- paths + templating
def paths(cfg):
    work = Path(cfg["work_dir"]).expanduser()
    p = {"ULTRON_ROOT": str(REPO / "workspace"), "AVATAR": str(work / "avatar"), "LTX": str(work / "ltx"),
         "TOOLVENVS": str(work / "tools-venvs"), "STUDIO": str(REPO / "runtime" / "studio"),
         "MLX_SERVE_HOME": str(Path.home() / ".mlx-serve"), "PNPM": str(Path.home() / "Library" / "pnpm"),
         "HOME": str(Path.home()), "FFMPEG": shutil.which("ffmpeg") or "ffmpeg"}
    p["PYTHON"] = str(Path(p["AVATAR"]) / ".venv" / "bin" / "python")
    return p


def subst(text, p):
    for k, v in sorted(p.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(f"@{k}@", v)
    return text


def render(cfg):
    p = paths(cfg)
    out = REPO / "runtime"
    if out.exists():
        shutil.rmtree(out)
    n = 0
    for sub in ("tools", "pipelines", "studio"):
        for f in (REPO / "src" / sub).rglob("*"):
            if f.is_file():
                dst = out / sub / f.relative_to(REPO / "src" / sub)
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dst.write_text(subst(f.read_text(), p))
                except UnicodeDecodeError:
                    shutil.copy(f, dst)
                dst.chmod(f.stat().st_mode)
                n += 1
    Path(p["ULTRON_ROOT"]).mkdir(parents=True, exist_ok=True)
    for d in ("logs", "stills", "shots", "audio", "review"):
        (Path(p["ULTRON_ROOT"]) / d).mkdir(exist_ok=True)
    env = [f"export ULTRON_ROOT='{p['ULTRON_ROOT']}'", f"export ULTRON_AVATAR='{p['AVATAR']}'", f"export ULTRON_LTX='{p['LTX']}'",
           f"export ULTRON_TOOLVENVS='{p['TOOLVENVS']}'", f"export PATH=\"{REPO / 'runtime' / 'tools'}:$PATH\""]
    (REPO / "ultron.env").write_text("\n".join(env) + "\n")
    print(f"rendered {n} files into runtime/ and wrote ultron.env  (source it: `source ultron.env`)")


# ----------------------------------------------------------------- downloads (always confirmed)
def run(cmd, **kw):
    print("  $", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True, **kw)


def ensure_hf():
    try:
        import huggingface_hub  # noqa
    except ImportError:
        print("  huggingface_hub is needed for downloads.")
        if not yes("  install it with pip now?", "y"):
            sys.exit("cannot download without huggingface_hub")
        run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])


def download(key, cfg, assume_yes=False):
    comp = CAT["components"][key]["local"]
    work = Path(cfg["work_dir"]).expanduser()
    ensure_hf()
    from huggingface_hub import snapshot_download
    print(f"\n== {CAT['components'][key]['title']}")
    total = 0
    for h in comp.get("hf", []):
        s = hf_size_gb(h["repo"], h.get("include"), h.get("exclude"))
        total += s or 0
        print(f"   {h['repo']}   {('%.1f GB' % s) if s else 'size unknown'}   from https://huggingface.co/{h['repo']}")
    print(f"   licence: {comp.get('license')}   approx total: {total or comp.get('size_gb_approx') or '?'} GB   into {work / 'models'}")
    if not (assume_yes or yes("   Download these files now?", "n")):
        print("   skipped.")
        return False
    for h in comp.get("hf", []):
        dest = work / "models" / h["repo"].replace("/", "__")
        snapshot_download(h["repo"], local_dir=str(dest), allow_patterns=h.get("include"), ignore_patterns=h.get("exclude"))
    if key == "video_ltx":
        setup_ltx(cfg)
    if comp.get("pip"):
        v = work / "tools-venvs" / "mflux"
        if not v.exists():
            run([sys.executable, "-m", "venv", str(v)])
        run([str(v / "bin" / "pip"), "install", "-q", comp["pip"]])
    if comp.get("manual"):
        print("   NOTE:", comp["notes"])
    return True


def setup_ltx(cfg):
    """Assemble the LTX 'pack' folder (symlinks) exactly as our engine expects, then fetch the runner."""
    work = Path(cfg["work_dir"]).expanduser()
    q8 = work / "models" / "mlx-community__ltx-2.5-mlx-q8"
    dit = work / "models" / "mlx-community__ltx-2.5-mlx-ditq8"
    pack = work / "ltx" / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    for f in q8.iterdir():
        t = pack / f.name
        if not t.exists():
            t.symlink_to(f)
    for name in ("transformer-distilled.safetensors", "transformer-dev.safetensors"):
        t = pack / name
        if t.exists() or t.is_symlink():
            t.unlink()
        t.symlink_to(dit / "transformer-distilled.safetensors")
    runner = work / "ltx" / "ltx-2-mlx-xoc"
    if not runner.exists() and yes("   Clone the LTX runner (github.com/xocialize/ltx-2-mlx) and run `uv sync --all-extras`?", "y"):
        run(["git", "clone", "https://github.com/xocialize/ltx-2-mlx", str(runner)])
        run(["git", "-C", str(runner), "checkout", CAT["components"]["video_ltx"]["local"]["runner_commit"]])
        run(["uv", "sync", "--all-extras"], cwd=str(runner))
    print(f"   LTX pack ready at {pack}")


def make_avatar_venv(cfg):
    av = Path(paths(cfg)["AVATAR"])
    if (av / ".venv").exists():
        return
    if yes(f"Create the Python environment for voice/ASR/QA tools at {av / '.venv'} (pip: numpy soundfile librosa mlx mlx-audio huggingface_hub)?", "y"):
        av.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "-m", "venv", str(av / ".venv")])
        run([str(av / ".venv" / "bin" / "pip"), "install", "-q", "-r", str(REPO / "requirements.txt")])


# ----------------------------------------------------------------- setup wizard
def cmd_setup(a):
    h = hardware()
    print("\n=== Ultron Engine setup ===")
    print(f"Detected: {h['os']} {h['machine']}, RAM {h['ram_gb']} GB, NVIDIA GPU: {h['nvidia'] or 'none'}")
    if not h["apple_silicon"]:
        print("NOTE: the local models in this engine are MLX builds and run on Apple Silicon only.\n"
              "      On this machine 'local' would not work for them; 'cloud' (or a Mac for the local part) is recommended.")
    elif (h["ram_gb"] or 0) < 64:
        print(f"NOTE: {h['ram_gb']} GB RAM. We measured LTX at 52 GB peak (4 s clip) and 85 GB (10 s). Smaller Macs: use clips of 3-4 s, or send video to cloud ('mixed').")
    mode = a.mode or ask("\nWhere should generation run?  local = everything on this machine | cloud = rented GPU endpoints | mixed = choose per component",
                         ["local", "cloud", "mixed"], "local" if h["apple_silicon"] else "cloud")
    work = a.work or input(f"Folder for models and environments [default {Path.home() / 'ultron-work'}]: ").strip() or str(Path.home() / "ultron-work")
    cfg = {"mode": mode, "work_dir": work, "components": {}, "cloud": {}}
    local_keys = set(filter(None, (a.local or "").split(",")))
    cloud_map = dict(x.split("=", 1) for x in (a.cloud or "").split(",") if "=" in x)
    for key, c in CAT["components"].items():
        if a.local is not None or a.cloud is not None:        # non-interactive
            where = "cloud" if key in cloud_map else "local" if key in local_keys else "skip"
        elif mode == "local":
            where = "local" if yes(f"\nUse {c['title']}?  (~{c['local'].get('size_gb_approx') or '?'} GB, {c['local'].get('license')})", "y" if c["role"] != "3d" else "n") else "skip"
        elif mode == "cloud":
            where = "cloud" if yes(f"\nUse {c['title']} via a cloud GPU endpoint?", "y" if c["role"] != "3d" else "n") else "skip"
        else:
            where = ask(f"\n{c['title']}  (local ~{c['local'].get('size_gb_approx') or '?'} GB)", ["local", "cloud", "skip"], "skip")
        cfg["components"][key] = where
        if where == "cloud":
            if key in cloud_map:
                url, _, env = cloud_map[key].partition("|")
            else:
                url = input("   Endpoint URL of your cloud worker (see docs/CLOUD.md for the protocol): ").strip()
                env = input("   NAME of the environment variable that holds its API key (the key itself is never stored): ").strip()
            cfg["cloud"][c["role"] + "." + key] = {"url": url, "api_key_env": env, "adapter_role": c["cloud"]["adapter_role"]}
    CFG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"\nwrote {CFG_PATH.name}")
    render(cfg)
    chosen = [k for k, v in cfg["components"].items() if v == "local"]
    if any(CAT["components"][k]["role"] in ("voice", "asr", "lipsync") for k in chosen):
        make_avatar_venv(cfg)
    to_dl = set(filter(None, (a.download or "").split(",")))
    for k in chosen:
        if a.local is not None and k not in to_dl:
            print(f"(not downloading {k}: not in --download)")
            continue
        download(k, cfg, assume_yes=k in to_dl)
    install_skills(cfg, a.install_skills)
    print("\nDone. Next: `python3 ultron.py doctor`, then read docs/QUICKSTART.md")


def install_skills(cfg, flag):
    dst = Path.home() / ".claude" / "skills"
    if not (flag or (sys.stdin.isatty() and yes(f"\nInstall the Claude Code skills (studio-ltx-video, studio-anime-fight, studio-voice-clone, ...) into {dst}?", "y"))):
        return
    p = paths(load_cfg() or cfg)
    for s in (REPO / "skills").iterdir():
        t = dst / s.name
        if t.exists():
            print(f"  skill {s.name} already exists, left untouched")
            continue
        for f in s.rglob("*"):
            if f.is_file():
                o = t / f.relative_to(s)
                o.parent.mkdir(parents=True, exist_ok=True)
                o.write_text(subst(f.read_text(), p))
        print(f"  installed skill {s.name}")


# ----------------------------------------------------------------- doctor / models
def cmd_doctor(a):
    cfg = load_cfg()
    print("tools:")
    for k, t in CAT["tools"].items():
        print(f"  {'OK ' if shutil.which(t['check']) else 'MISSING'} {k:10s} {'' if shutil.which(t['check']) else t['install']}")
    if not cfg:
        print("no ultron.config.json yet: run `python3 ultron.py setup`")
        return
    work = Path(cfg["work_dir"]).expanduser()
    print(f"mode: {cfg['mode']}   work dir: {work}")
    for k, where in cfg["components"].items():
        if where == "local":
            ok = all((work / "models" / h["repo"].replace("/", "__")).exists() for h in CAT["components"][k]["local"].get("hf", []))
            print(f"  {'OK ' if ok else 'NOT DOWNLOADED'} local  {k}")
        elif where == "cloud":
            c = [v for kk, v in cfg["cloud"].items() if kk.endswith("." + k)][0]
            print(f"  cloud  {k}: {c['url']}  key env {c['api_key_env']} {'(set)' if os.environ.get(c['api_key_env']) else '(NOT set in this shell)'}")


def cmd_models(a):
    for k, c in CAT["components"].items():
        l = c["local"]
        print(f"{k:22s} {c['title']}\n{'':22s} ~{l.get('size_gb_approx') or '?'} GB | peak RAM {l.get('ram_gb_peak') or '-'} | {l.get('license')} | {l.get('speed', '')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("setup")
    s.add_argument("--mode", choices=["local", "cloud", "mixed"])
    s.add_argument("--work")
    s.add_argument("--local", help="comma list of component keys to run locally (non-interactive)")
    s.add_argument("--cloud", help="comma list KEY=URL|API_KEY_ENV_NAME")
    s.add_argument("--download", help="comma list of local components the user approved for download")
    s.add_argument("--install-skills", action="store_true")
    sp.add_parser("doctor"), sp.add_parser("models"), sp.add_parser("render")
    d = sp.add_parser("download")
    d.add_argument("keys", nargs="+")
    a = ap.parse_args()
    if a.cmd == "setup":
        cmd_setup(a)
    elif a.cmd == "doctor":
        cmd_doctor(a)
    elif a.cmd == "models":
        cmd_models(a)
    elif a.cmd == "render":
        render(load_cfg() or sys.exit("run setup first"))
    elif a.cmd == "download":
        cfg = load_cfg() or sys.exit("run setup first")
        for k in a.keys:
            download(k, cfg)


if __name__ == "__main__":
    main()

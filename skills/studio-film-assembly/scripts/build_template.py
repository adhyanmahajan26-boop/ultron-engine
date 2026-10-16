"""v2: continuous film. Persistent stage (never blank), crossfaded windows, real HyperFrames editor w/ avatar riding the timeline.
Reads durations from film/assets/v2/*.mp4 and film/assets/a2/*.wav ; writes film/index.html
"""
import subprocess, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
FFP = "@FFMPEG@"

def dur(path):
    out = subprocess.run([FFP, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    t = out.split("Duration: ")[1].split(",")[0].split(":")
    return round(float(t[0]) * 3600 + float(t[1]) * 60 + float(t[2]), 3)

D = {os.path.splitext(n)[0]: dur(f"{HERE}/assets/v2/{n}") for n in os.listdir(f"{HERE}/assets/v2") if n.endswith((".mp4", ".webm"))}
A = {os.path.splitext(n)[0]: dur(f"{HERE}/assets/a2/{n}") for n in os.listdir(f"{HERE}/assets/a2") if n.endswith(".wav")}
XF = 0.09                                   # crossfade overlap between consecutive talking windows (slices start in silence)

# ------------------------------------------------------------------ timeline
T = {}; t = 0.15
def put(name, d=None, gap=0.0):
    global t
    T[name] = round(t, 3); t += (d if d is not None else D[name]) + gap
put("W1a"); put("W1b")
T["webcam_adjust"] = round(t - 0.28, 3)          # whip in during W1b's last frames; its own length covers L2
t = T["webcam_adjust"] + D["webcam_adjust"]
T["L2"] = round(T["webcam_adjust"] + 0.28 + 0.10, 3)
T["webcam_lean"] = round(t - 0.14, 3); t = T["webcam_lean"] + D["webcam_lean"]
T["L3"] = round(T["webcam_lean"] + 0.28, 3)
T["enter"] = round(t - 0.9, 3)                    # push into the laptop screen starts here
T["stage"] = round(t - 0.35, 3)                   # stage appears under the push-in
t = T["stage"] + 0.35 + 0.0
for n in ["W4a", "W4b", "W4c"]: put(n)
EDIT_IN = t
for n in ["S1", "S2", "S3", "S4", "S5"]: put(n)
EDIT_OUT = t
for n in ["W6a", "W6b"]: put(n)
T["assemble"] = round(t, 3); t += D["assemble"]
T["flight_b"] = round(t, 3); t += D["flight_b"]
T["flight_a"] = round(t, 3); t += D["flight_a"]
T["speak_nyc"] = round(t + 0.06, 3); t += 0.06 + D["speak_nyc"]
T["suit_off"] = round(t, 3); t += D["suit_off"]
T["enter2"] = round(t - 0.7, 3)
T["stage2"] = round(t - 0.3, 3)
EDIT2_IN = t
for n in ["W7a", "W7b"]: put(n)
EDIT2_OUT = t
put("W8")
T["beat"] = round(t, 3); t += 2.0
put("W9a"); put("W9b")
GAL = round(t, 2); GAL_DUR = 6.6
T["title"] = GAL + 0.9
END = round(GAL + GAL_DUR + 0.5, 2)
print("timeline:", json.dumps({k: round(v, 2) for k, v in T.items()}), "END", END)

# ------------------------------------------------------------------ markup helpers
def vid(name, start, d=None, style="", extra="", src=None, rate=None):
    d = d if d is not None else D[name]
    src = src or f"assets/v2/{name}.mp4"
    return (f'<video id="v_{name}" src="{src}" muted playsinline preload="auto" data-start="{start:.3f}" data-duration="{d:.3f}" {extra} style="{style}"></video>')
aid = [0]
def aud(name, start, vol=1.0, d=None, ms=0, src=None):
    aid[0] += 1
    d = d if d is not None else A[name]
    src = src or f"assets/a2/{name}.wav"
    m = f' data-media-start="{ms:.3f}"' if ms else ""
    return (f'<audio id="a{aid[0]}_{name}" src="{src}" data-start="{start:.3f}" data-duration="{d:.3f}" data-volume="{vol}"{m}></audio>')

FULL = "position:absolute;left:0;top:0;width:1920px;height:1080px;object-fit:cover;"
MONV = "position:absolute;left:0;top:0;width:1400px;height:754px;object-fit:cover;"
audio = []
# ---- voice: every slice sits at its window start (slices start in silence at the gap midpoint)
for n in ["W1a", "W1b", "W4a", "W4b", "W4c", "S1", "S2", "S3", "S4", "S5", "W6a", "W6b", "W7a", "W7b", "W8", "W9a", "W9b"]:
    audio.append(aud(n, T[n], 1.0))
audio.append(aud("L2", T["L2"], 1.0)); audio.append(aud("L3", T["L3"], 1.0))
audio.append(aud("speak_nyc", T["speak_nyc"], 1.0))
# ---- music (levels: bed under talk, hot on the action + finale)
def mus(a, b, vol, first=False):
    audio.append(aud("phonk", a, vol, d=b - a, ms=a - T["stage"]))
mus(T["stage"], T["assemble"], 0.15)
mus(T["assemble"], T["suit_off"], 0.34)
mus(T["suit_off"], T["W8"] + D["W8"], 0.15)
mus(T["W8"] + D["W8"], END - 0.2, 0.52)
# ---- sfx
def sf(n, s, v=0.9, d=None): audio.append(aud(n, s, v, d=d))
sf("whoosh", T["webcam_adjust"], 0.7); sf("whoosh", T["webcam_lean"], 0.4)
sf("riser", T["enter"] - 0.2, 0.8, d=1.2); sf("boom", T["stage"] + 0.2, 0.9, d=1.6)
for k in ["W4b", "W4c"]: sf("whoosh", T[k] - 0.12, 0.8, d=0.6)
sf("whoosh", EDIT_IN - 0.15, 0.8, d=0.8); sf("thud", EDIT_IN + 0.35, 0.8)
for k in ["S2", "S3", "S4", "S5"]: sf("click", T[k] - 0.02, 0.7)
sf("riser", T["assemble"] - 1.0, 0.8, d=1.05); sf("boom", T["assemble"], 0.9, d=1.8)
sf("whoosh_long", T["flight_b"] - 0.2, 0.8); sf("whoosh_long", T["flight_a"] - 0.2, 0.7)
sf("click", T["speak_nyc"], 1.0); sf("whoosh", T["speak_nyc"] - 0.05, 0.9)
sf("whoosh_long", T["suit_off"] - 0.1, 0.6)
sf("riser", T["enter2"], 0.8, d=0.9); sf("boom", T["stage2"] + 0.25, 0.8, d=1.4); sf("thud", EDIT2_IN + 0.3, 0.7)
sf("riser", T["beat"] - 0.8, 0.8, d=0.85); sf("boom", T["beat"], 1.0, d=2.0)
sf("riser", GAL - 0.9, 0.8, d=0.95); sf("boom", GAL, 1.0, d=2.2); sf("shimmer", GAL + 0.1, 0.9)

# ------------------------------------------------------------------ pieces
rooms = "\n".join([
    vid("W1a", T["W1a"], D["W1a"] + XF, FULL + "z-index:11;"), vid("W1b", T["W1b"] - XF, D["W1b"] + XF + 0.28, FULL + "z-index:12;"),
    vid("webcam_adjust", T["webcam_adjust"], style=FULL + "z-index:13;"), vid("webcam_lean", T["webcam_lean"], style=FULL + "z-index:14;")])

def monwin(name, extra_dur=0.0, start_shift=0.0):
    return vid(name, T[name] - start_shift, D[name] + extra_dur + start_shift, MONV)
mon_vids = "\n".join([monwin("W4a"), monwin("W4b", 0, XF), monwin("W4c", 0, XF),
                      monwin("S1"), monwin("S2", 0, XF), monwin("S3", 0, XF), monwin("S4", 0, XF), monwin("S5", 0, XF),
                      monwin("W6a"), monwin("W6b", 0, XF), monwin("W7a"), monwin("W7b", 0, XF), monwin("W8")])

# avatar (keyed alpha webm), three segments per ride so 8s is covered
def av_seg(i, name, start, d, ms=0.0):
    m = f'data-media-start="{ms:.3f}"' if ms else ""
    return (f'<video id="av{i}" src="assets/v2/{name}.webm" muted playsinline preload="auto" data-start="{start:.3f}" data-duration="{d:.3f}" {m} '
            f'style="position:absolute;left:0;top:0;width:520px;height:560px;object-fit:contain;"></video>')
ride1 = EDIT_OUT - EDIT_IN
ride2 = EDIT2_OUT - EDIT2_IN
seg = D.get("av_surf", 3.0)
av1 = []; tt = EDIT_IN + 0.15; i = 0
names = ["av_surf", "av_carve", "av_surf", "av_carve"]
while tt < EDIT_OUT - 0.05:
    n = names[i % len(names)]; d = min(D.get(n, 3.0), EDIT_OUT - tt)
    av1.append(av_seg(i, n, tt, d)); tt += d; i += 1
av2 = []; tt = EDIT2_IN + 0.15
while tt < EDIT2_OUT - 0.05:
    n = names[i % len(names)]; d = min(D.get(n, 3.0), EDIT2_OUT - tt)
    av2.append(av_seg(i, n, tt, d)); tt += d; i += 1

block_defs = [("castle", "#ff6b8b"), ("cyber", "#b48bff"), ("perfume", "#ffb454"), ("galaxy", "#36d8ff"), ("nyc", "#6dffb0"), ("avatar", "#ff6b8b")]
lanes_html = ""
lane_rows = [("V2", 0), ("V1", 1), ("A1", 2), ("FX", 3)]
import random
rnd = random.Random(7)
for li, (lab, row) in enumerate(lane_rows):
    x = 0; blocks = ""
    for k in range(18):
        w = rnd.choice([220, 280, 340, 190, 400]); col = ["#36d8ff", "#ffb454", "#b48bff", "#6dffb0", "#ff6b8b"][(k + li * 2) % 5]
        thumb = ""
        if lab == "V1": thumb = f"background-image:linear-gradient(90deg,rgba(0,0,0,.0),rgba(0,0,0,.35)),url(assets/img/{['castle','perfume','cyber','galaxy','nyc','avatar'][k%6]}.jpg);background-size:cover;"
        blocks += f'<div class="blk" id="bk{li}_{k}" style="left:{x}px;width:{w}px;{thumb}--c:{col}"><span>{lab.lower()}_{k:02d}</span></div>'
        x += w + 10
    lanes_html += f'<div class="lane l{li}"><b>{lab}</b><div class="strip">{blocks}</div></div>'

bin_imgs = "".join(f'<img src="assets/img/{n}.jpg" id="bi{i}">' for i, n in enumerate(["castle", "perfume", "cyber", "galaxy", "nyc", "avatar"]))
insp = '''<div class="row"><span>SCALE</span><div class="sl"><i id="sl0"></i></div></div><div class="row"><span>OPACITY</span><div class="sl"><i id="sl1"></i></div></div>
<div class="row"><span>EASE</span><div class="sl"><i id="sl2"></i></div></div><div class="row"><span>VOICE</span><em>CLONE ✓</em></div><div class="row"><span>MODEL</span><em>H3 · 8-BIT</em></div><div class="row"><span>RENDER</span><em>1080p · 30</em></div>'''

stage = f'''
<div id="stage">
  <div id="bg"></div>
  <div id="bin" class="panel"><h3>MEDIA</h3><div class="grid">{bin_imgs}</div></div>
  <div id="insp" class="panel"><h3>INSPECTOR</h3>{insp}</div>
  <div id="mon" class="win">
    <div class="bar"><i></i><i></i><i></i><span id="mt0">ENGINE &nbsp;›&nbsp; CLONE &nbsp;›&nbsp; avatar_01</span><span id="mt1" style="position:absolute;left:84px;opacity:0">VIDEO GENERATION</span><span id="mt2" style="position:absolute;left:84px;opacity:0">MUSIC · ACE-STEP</span><b id="monChip">VOICE: CLONED</b></div>
    <div class="vwrap" id="monv" style="top:36px;height:754px;">{mon_vids}<div id="wipe"></div></div>
    <div class="chips"><em id="sc0">OUTFIT · BLACK</em><em id="sc1">OUTFIT · COBALT</em><em id="sc2">OUTFIT · CRIMSON</em></div>
    <div class="prompt" id="prompt"><span>prompt</span><p>“a man in an armored suit, falling through the screen…”</p><button id="genBtn">GENERATE ▶</button></div>
  </div>
  <div id="tl"><div class="ruler"></div><div class="lanes" id="lanes">{lanes_html}</div><div class="head"><s></s></div><div class="tlbl">HYPERFRAMES · TIMELINE</div></div>
  <div id="avwrap1" class="avw"><div class="board"></div><div class="avin">{"".join(av1)}</div></div>
  <div id="avwrap2" class="avw"><div class="board"></div><div class="avin">{"".join(av2)}</div></div>
</div>'''

falls = "\n".join([
    vid("assemble", T["assemble"], style=FULL + "z-index:20;"), vid("flight_b", T["flight_b"], style=FULL + "z-index:21;"),
    vid("flight_a", T["flight_a"], style=FULL + "z-index:22;"), vid("speak_nyc", T["speak_nyc"], style=FULL + "z-index:23;"),
    vid("suit_off", T["suit_off"], style=FULL + "z-index:24;")])

star_rng = [((i * 137.508) % 360, 0.35 + ((i * 53) % 100) / 100.0, 2.6 + ((i * 29) % 15) / 10.0, ((i * 41) % 100) / 100.0 * 4.6) for i in range(90)]
stars_html = "".join(f'<i class="st" data-a="{a:.2f}" data-d="{d:.2f}" data-t="{tt:.2f}" data-o="{o:.2f}"></i>' for a, d, tt, o in star_rng)
finale = "\n".join([
    vid("W9a", T["W9a"], D["W9a"] + XF, FULL + "z-index:30;"), vid("W9b", T["W9b"] - XF, D["W9b"] + XF, FULL + "z-index:31;"),
    f'<video id="v_galaxy" src="assets/v2/galaxy.mp4" muted playsinline preload="auto" data-start="{GAL:.3f}" data-duration="{GAL_DUR:.3f}" data-playback-rate="{D["galaxy"]/GAL_DUR:.4f}" style="{FULL}z-index:32;"></video>'])

beat = f'<div id="beat"><h1>EVEN THE SONGS</h1><div class="bars">{"".join("<u></u>" for _ in range(48))}</div></div>'

html = f'''<!doctype html>
<html lang="en"><head><meta charset="UTF-8" /><meta name="viewport" content="width=1920, height=1080" /><title>Ultron Engine</title>
<script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
<style>
  :root {{ --ink:#e9edf5; --dim:#8b93a7; --cy:#36d8ff; --am:#ffb454; --rd:#ff3b5c; }}
  html, body {{ margin:0; background:#000; }}
  #root {{ position:relative; width:100%; height:100%; overflow:hidden; background:#000; font-family:"Helvetica Neue",Arial,sans-serif; color:var(--ink); }}
  #stage {{ position:absolute; inset:0; z-index:3; opacity:0; background:#05070d; }}
  #bg {{ position:absolute; inset:0; background:radial-gradient(1300px 900px at 50% 38%, #101a36 0%, #070a16 62%, #04050a 100%); }}
  #bg:after {{ content:""; position:absolute; inset:0; opacity:.55; background-image:linear-gradient(rgba(54,216,255,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(54,216,255,.07) 1px,transparent 1px); background-size:64px 64px; }}
  .panel {{ position:absolute; top:40px; width:360px; height:640px; box-sizing:border-box; padding:22px; background:rgba(10,14,26,.94); border:1px solid rgba(255,255,255,.14); border-radius:16px; opacity:0; }}
  #bin {{ left:30px; }} #insp {{ left:1530px; }}
  .panel h3 {{ margin:0 0 16px; font-size:15px; letter-spacing:.22em; color:var(--cy); font-weight:600; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }} .grid img {{ width:100%; height:112px; object-fit:cover; border-radius:8px; border:1px solid rgba(255,255,255,.12); }}
  .row {{ display:flex; align-items:center; justify-content:space-between; margin:20px 0; font-size:15px; letter-spacing:.14em; color:var(--dim); }}
  .row em {{ font-style:normal; color:var(--ink); }} .sl {{ width:170px; height:8px; background:#1b2236; border-radius:4px; position:relative; }} .sl i {{ position:absolute; left:0; top:0; height:8px; width:30%; border-radius:4px; background:linear-gradient(90deg,var(--cy),var(--am)); }}
  .win {{ position:absolute; left:0; top:0; width:1400px; height:790px; transform-origin:0 0; background:#0b0e18; border:1px solid rgba(255,255,255,.14); border-radius:16px; overflow:hidden; box-shadow:0 40px 120px rgba(0,0,0,.65),0 0 0 1px rgba(54,216,255,.08); opacity:0; z-index:6; }}
  .bar {{ position:absolute; left:0; top:0; right:0; height:36px; background:linear-gradient(#171c2c,#101421); display:flex; align-items:center; gap:8px; padding:0 14px; font-size:13px; letter-spacing:.14em; color:var(--dim); z-index:3; }}
  .bar i {{ width:11px; height:11px; border-radius:50%; background:#ff5f57; display:inline-block; }} .bar i:nth-child(2) {{ background:#febc2e; }} .bar i:nth-child(3) {{ background:#28c840; margin-right:10px; }}
  .bar b {{ margin-left:auto; color:var(--cy); font-weight:600; }}
  .vwrap {{ position:absolute; left:0; right:0; overflow:hidden; background:#000; }}
  #wipe {{ position:absolute; inset:0; z-index:9; background:#2a63ff; transform:translateX(-101%); pointer-events:none; }}
  .chips {{ position:absolute; left:18px; bottom:18px; z-index:4; }} .chips em {{ position:absolute; left:0; bottom:0; font-style:normal; font-size:15px; letter-spacing:.16em; padding:8px 14px; border-radius:20px; background:rgba(5,8,16,.78); border:1px solid rgba(54,216,255,.45); color:var(--cy); opacity:0; white-space:nowrap; }}
  .prompt {{ position:absolute; left:0; right:0; bottom:0; height:0; background:#0a0d18; border-top:1px solid rgba(255,255,255,.08); overflow:hidden; z-index:5; box-sizing:border-box; padding:0 26px; }}
  .prompt span {{ display:block; margin-top:18px; font-size:13px; letter-spacing:.2em; color:var(--dim); }} .prompt p {{ margin:8px 0 0; font-size:26px; }}
  #genBtn {{ position:absolute; right:26px; top:30px; padding:16px 30px; font-size:20px; letter-spacing:.14em; border-radius:12px; border:1px solid var(--cy); background:#0d2a38; color:var(--cy); font-family:inherit; }}
  #tl {{ position:absolute; left:30px; top:700px; width:1860px; height:340px; overflow:hidden; background:rgba(9,12,22,.95); border:1px solid rgba(255,255,255,.14); border-radius:16px; opacity:0; z-index:5; }}
  #tl .ruler {{ position:absolute; left:100px; right:0; top:0; height:44px; border-bottom:1px solid rgba(255,255,255,.12); background-image:repeating-linear-gradient(90deg,rgba(255,255,255,.35) 0 1px,transparent 1px 40px); }}
  .tlbl {{ position:absolute; left:20px; top:12px; font-size:13px; letter-spacing:.2em; color:var(--cy); }}
  .lanes {{ position:absolute; left:0; top:50px; right:0; }}
  .lane {{ position:relative; height:66px; margin-bottom:8px; overflow:hidden; }} .lane b {{ position:absolute; left:0; top:0; width:100px; height:66px; z-index:2; background:#0b0f1c; color:var(--dim); font-size:14px; letter-spacing:.14em; display:flex; align-items:center; justify-content:center; }}
  .strip {{ position:absolute; left:100px; top:0; width:6000px; height:66px; }}
  .blk {{ position:absolute; top:4px; height:58px; border-radius:8px; background:linear-gradient(180deg,var(--c),color-mix(in srgb, var(--c) 55%, #000)); box-sizing:border-box; padding:6px 10px; color:#05101a; font-size:13px; font-weight:700; letter-spacing:.1em; opacity:.92; }}
  #tl .head {{ position:absolute; top:0; bottom:0; left:760px; width:3px; background:#fff; box-shadow:0 0 16px #fff; z-index:4; }} #tl .head s {{ position:absolute; left:-9px; top:0; width:0; height:0; border-left:10px solid transparent; border-right:10px solid transparent; border-top:16px solid #fff; }}
  .avw {{ position:absolute; left:0; top:0; width:520px; height:600px; z-index:8; opacity:0; pointer-events:none; }}
  .avin {{ position:absolute; left:0; top:0; width:520px; height:560px; }} .board {{ position:absolute; left:110px; top:536px; width:300px; height:26px; border-radius:13px; background:linear-gradient(90deg,#1b7fb3,#36d8ff,#1b7fb3); box-shadow:0 0 30px 6px rgba(54,216,255,.75),0 0 80px 14px rgba(54,216,255,.35); z-index:0; }}
  .avin {{ z-index:1; }}
  #beat {{ position:absolute; inset:0; z-index:28; opacity:0; background:#04050a; display:flex; flex-direction:column; align-items:center; justify-content:center; }}
  #beat h1 {{ margin:0; font-size:190px; letter-spacing:.06em; font-weight:800; color:#fff; text-shadow:0 0 60px rgba(255,59,92,.65); white-space:nowrap; }}
  #beat .bars {{ position:absolute; left:80px; right:80px; bottom:90px; height:260px; display:flex; align-items:flex-end; gap:10px; }}
  #beat .bars u {{ flex:1; background:linear-gradient(#ff3b5c,#ffb454); border-radius:4px 4px 0 0; height:100%; transform-origin:bottom; transform:scaleY(.1); }}
  #stars {{ position:absolute; inset:0; z-index:35; pointer-events:none; opacity:0; overflow:hidden; }}
  #stars .st {{ position:absolute; left:50%; top:50%; width:6px; height:6px; margin:-3px 0 0 -3px; border-radius:50%; background:#fff; box-shadow:0 0 10px 2px rgba(180,210,255,.9); opacity:0; }}
  #flash {{ position:absolute; inset:0; z-index:50; opacity:0; background:#fff; pointer-events:none; }}
  #speed {{ position:absolute; inset:0; z-index:25; opacity:0; pointer-events:none; background:repeating-linear-gradient(90deg,transparent 0 26px,rgba(255,255,255,.16) 26px 28px,transparent 28px 74px); mix-blend-mode:screen; }}
  #title {{ position:absolute; inset:0; z-index:40; display:flex; flex-direction:column; align-items:center; justify-content:center; opacity:0; pointer-events:none; background:radial-gradient(ellipse at center,rgba(0,0,0,.45) 0%,transparent 62%); }}
  #title h1 {{ margin:0; font-size:108px; letter-spacing:.24em; font-weight:700; padding-left:.24em; white-space:nowrap; text-shadow:0 0 50px rgba(120,170,255,.55); }} #title h1 span {{ color:#7fb6ff; }}
  #title p {{ margin:18px 0 0; font-size:30px; letter-spacing:.5em; color:#fff; padding-left:.5em; text-shadow:0 0 18px rgba(0,0,0,.9); }}
  #vig {{ position:absolute; inset:0; z-index:60; pointer-events:none; background:radial-gradient(ellipse at center,transparent 58%,rgba(0,0,0,.5) 100%); }}
  #black {{ position:absolute; inset:0; z-index:70; background:#000; pointer-events:none; }}
</style></head>
<body>
<div id="root" data-composition-id="main" data-start="0" data-width="1920" data-height="1080" data-duration="{END}">
  {stage}
  <div id="rooms">{rooms}</div>
  <div id="falls">{falls}</div>
  {beat}
  <div id="fin">{finale}</div>
  <div id="stars">{stars_html}</div>
  <div id="speed"></div><div id="flash"></div>
  <div id="title"><h1>ULTRON ENGINE <span>V3</span></h1><p>UNLIMITED POSSIBILITIES</p></div>
  <div id="vig"></div><div id="black"></div>
  {chr(10).join(audio)}
</div>
<script>
  const T = {json.dumps({k: round(v, 3) for k, v in T.items()})};
  const D = {json.dumps(D)};
  const END = {END}, GAL = {GAL}, GAL_DUR = {GAL_DUR}, XF = {XF}, EDIT_IN = {round(EDIT_IN,3)}, EDIT_OUT = {round(EDIT_OUT,3)}, EDIT2_IN = {round(EDIT2_IN,3)}, EDIT2_OUT = {round(EDIT2_OUT,3)};
  const tl = gsap.timeline({{ paused: true }});
  const $ = (s) => document.querySelector(s);
  // monitor layouts (transform of #mon): LARGE centered, EDIT docked top-centre, SONG medium
  const LARGE = {{ x: 260, y: 128, scale: 1 }}, EDIT = {{ x: 456, y: 30, scale: 0.72 }}, SONG = {{ x: 460, y: 190, scale: 1 }};
  tl.set("#mon", {{ ...LARGE, opacity: 0, transformOrigin: "0 0" }}, 0);
  tl.to("#black", {{ opacity: 0, duration: 0.45, ease: "power2.out" }}, 0);
  tl.to("#black", {{ opacity: 1, duration: 0.9, ease: "power2.in" }}, END - 1.0);

  // ---------- opening: talk -> whip behind -> lean in -> push into the screen
  const xfade = (id, t0, d = XF) => tl.fromTo(id, {{ opacity: 0 }}, {{ opacity: 1, duration: d, ease: "none", immediateRender: false }}, t0);
  tl.fromTo("#v_W1a", {{ scale: 1.06 }}, {{ scale: 1.0, duration: 0.9, ease: "power3.out" }}, T.W1a);
  xfade("#v_W1b", T.W1b - XF);
  tl.fromTo("#v_W1b", {{ scale: 1.0 }}, {{ scale: 1.05, duration: D.W1b + XF, ease: "none", immediateRender: false }}, T.W1b - XF);
  // whip: webcam_adjust slides in from the right with blur while W1b slides out left
  tl.fromTo("#v_webcam_adjust", {{ x: 1920, filter: "blur(18px)" }}, {{ x: 0, filter: "blur(0px)", duration: 0.28, ease: "power3.out", immediateRender: false }}, T.webcam_adjust);
  tl.fromTo("#v_W1b", {{ x: 0 }}, {{ x: -900, duration: 0.28, ease: "power3.in", immediateRender: false }}, T.webcam_adjust);
  tl.fromTo("#v_webcam_adjust", {{ scale: 1.0 }}, {{ scale: 1.07, duration: D.webcam_adjust, ease: "none", immediateRender: false }}, T.webcam_adjust + 0.28);
  xfade("#v_webcam_lean", T.webcam_lean, 0.14);
  tl.fromTo("#v_webcam_lean", {{ scale: 1.0 }}, {{ scale: 1.05, duration: 1.0, ease: "none", immediateRender: false }}, T.webcam_lean);
  // push into the laptop screen (white glow floods), stage appears under it
  tl.set("#v_webcam_lean", {{ transformOrigin: "66% 54%" }}, T.enter);
  tl.fromTo("#v_webcam_lean", {{ scale: 1.05 }}, {{ scale: 3.4, duration: T.stage + 0.7 - T.enter, ease: "power2.in", immediateRender: false }}, T.enter);
  tl.fromTo("#flash", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.45, ease: "power2.in", immediateRender: false }}, T.stage - 0.2);
  tl.set("#stage", {{ opacity: 1 }}, T.stage);
  tl.to("#flash", {{ opacity: 0, duration: 0.7, ease: "power2.out" }}, T.stage + 0.25);
  tl.set(["#v_webcam_lean", "#v_webcam_adjust"], {{ opacity: 0 }}, T.stage + 0.3);

  // ---------- clone scene (LARGE monitor), colour wipes inside the monitor
  tl.fromTo("#mon", {{ opacity: 0, scale: 0.9 }}, {{ opacity: 1, scale: 1, duration: 0.5, ease: "back.out(1.3)", immediateRender: false }}, T.stage + 0.2);
  const mv = (n, t0, d = XF) => tl.fromTo("#v_" + n, {{ opacity: 0 }}, {{ opacity: 1, duration: d, ease: "none", immediateRender: false }}, t0);
  ["W4b", "W4c", "S2", "S3", "S4", "S5", "W6b", "W7b"].forEach((n) => mv(n, T[n] - XF));
  const chips = ["#sc0", "#sc1", "#sc2"], cut = [T.W4a, T.W4b, T.W4c];
  chips.forEach((c, i) => {{ tl.to(c, {{ opacity: 1, duration: 0.15 }}, cut[i] + 0.1); tl.to(c, {{ opacity: 0, duration: 0.1 }}, (i < 2 ? cut[i + 1] : EDIT_IN) - 0.05); }});
  [["#2a63ff", T.W4b], ["#e0173b", T.W4c]].forEach(([col, t0]) => {{
    tl.set("#wipe", {{ backgroundColor: col, x: "-101%" }}, t0 - 0.3);
    tl.to("#wipe", {{ x: "0%", duration: 0.2, ease: "power3.in" }}, t0 - 0.3);
    tl.to("#wipe", {{ x: "101%", duration: 0.3, ease: "power3.out" }}, t0 - 0.08);
  }});
  // ---------- EDIT layout: dock the monitor, panels + timeline slide in, avatar rides the timeline
  const toEdit = (t0, d = 0.7) => {{
    tl.to("#mon", {{ ...EDIT, duration: d, ease: "power3.inOut" }}, t0);
    tl.fromTo("#bin", {{ x: -420, opacity: 0 }}, {{ x: 0, opacity: 1, duration: d, ease: "power3.out", immediateRender: false }}, t0 + 0.05);
    tl.fromTo("#insp", {{ x: 420, opacity: 0 }}, {{ x: 0, opacity: 1, duration: d, ease: "power3.out", immediateRender: false }}, t0 + 0.05);
    tl.fromTo("#tl", {{ y: 420, opacity: 0 }}, {{ y: 0, opacity: 1, duration: d, ease: "power3.out", immediateRender: false }}, t0 + 0.1);
  }};
  const fromEdit = (t0, d = 0.6) => {{
    tl.to("#bin", {{ x: -420, opacity: 0, duration: d, ease: "power3.in" }}, t0);
    tl.to("#insp", {{ x: 420, opacity: 0, duration: d, ease: "power3.in" }}, t0);
    tl.to("#tl", {{ y: 420, opacity: 0, duration: d, ease: "power3.in" }}, t0);
  }};
  toEdit(EDIT_IN - 0.5);
  document.querySelectorAll(".lane").forEach((l, i) => tl.fromTo(l.querySelector(".strip"), {{ x: 0 }}, {{ x: -2400 - i * 260, duration: EDIT_OUT - EDIT_IN + (EDIT2_OUT - EDIT2_IN) + 12, ease: "none", immediateRender: false }}, EDIT_IN));
  ["#sl0", "#sl1", "#sl2"].forEach((s, i) => tl.to(s, {{ width: (45 + i * 17) + "%", duration: 2.5, ease: "sine.inOut", yoyo: true, repeat: 3 }}, EDIT_IN + i * 0.2));
  [0, 1, 2, 3, 4, 5].forEach((i) => tl.fromTo("#bi" + i, {{ scale: 0.8, opacity: 0 }}, {{ scale: 1, opacity: 1, duration: 0.35, ease: "back.out(1.6)", immediateRender: false }}, EDIT_IN + 0.15 + i * 0.12));
  // avatar rides: jumps in, surfs left->right along the timeline, hops off at the end
  const ride = (wrap, t0, t1, x0, x1) => {{
    const base = 700 + 52 - 560 + 26;           // feet on the ruler line
    tl.set(wrap, {{ x: x0, y: base - 520, opacity: 0, scale: 0.66, transformOrigin: "50% 100%" }}, t0 - 0.01);
    tl.to(wrap, {{ opacity: 1, duration: 0.15 }}, t0);
    tl.to(wrap, {{ y: base, duration: 0.55, ease: "bounce.out" }}, t0);
    tl.to(wrap, {{ x: x1, duration: t1 - t0 - 0.5, ease: "sine.inOut" }}, t0 + 0.2);
    tl.to(wrap, {{ y: base - 14, duration: 0.55, ease: "sine.inOut", yoyo: true, repeat: Math.floor((t1 - t0) / 1.1) }}, t0 + 0.7);
    tl.to(wrap, {{ y: base - 700, opacity: 0, duration: 0.45, ease: "power2.in" }}, t1 - 0.4);
  }};
  ride("#avwrap1", EDIT_IN + 0.2, EDIT_OUT - 0.1, 130, 1250);
  // ---------- LARGE again for "infinite possibilities / watch this"
  fromEdit(EDIT_OUT - 0.55);
  tl.to("#mon", {{ ...LARGE, duration: 0.6, ease: "power3.inOut" }}, EDIT_OUT - 0.55);
  tl.to("#prompt", {{ height: 134, duration: 0.4, ease: "power3.out" }}, T.W6b + 0.15);
  tl.to("#mt0", {{ opacity: 0, duration: 0.15 }}, T.W6a); tl.to("#mt1", {{ opacity: 1, duration: 0.15 }}, T.W6a);
  tl.to("#genBtn", {{ scale: 1.14, backgroundColor: "#36d8ff", color: "#021016", duration: 0.12, yoyo: true, repeat: 1 }}, T.W6b + D.W6b - 0.5);
  // fall into the screen: monitor zooms into the face, blue flash, armour assembles as he falls
  tl.to("#mon", {{ scale: 2.6, x: -300, y: -330, duration: 0.55, ease: "power3.in" }}, T.assemble - 0.55);
  tl.fromTo("#flash", {{ opacity: 0, backgroundColor: "#36d8ff" }}, {{ opacity: 0.95, duration: 0.3, immediateRender: false }}, T.assemble - 0.3);
  tl.to("#flash", {{ opacity: 0, duration: 0.5 }}, T.assemble + 0.02);
  tl.set("#flash", {{ backgroundColor: "#ffffff" }}, T.assemble + 0.6);
  tl.set("#stage", {{ opacity: 0 }}, T.assemble + 0.05);
  tl.fromTo("#v_assemble", {{ scale: 1, y: 0, rotation: 0 }}, {{ scale: 1.5, y: 120, rotation: -5, duration: D.assemble, ease: "power2.in", immediateRender: false }}, T.assemble);
  tl.to("#speed", {{ opacity: 0.8, duration: 0.3 }}, T.assemble + 0.2);
  tl.fromTo("#speed", {{ backgroundPositionY: 0 }}, {{ backgroundPositionY: 900, duration: D.assemble, ease: "none", immediateRender: false }}, T.assemble);
  tl.to("#speed", {{ opacity: 0, duration: 0.25 }}, T.flight_b + 0.5);
  tl.fromTo("#v_flight_b", {{ scale: 1.15 }}, {{ scale: 1, duration: 1.0, ease: "power3.out", immediateRender: false }}, T.flight_b);
  tl.fromTo("#v_flight_a", {{ scale: 1.1 }}, {{ scale: 1, duration: 0.8, ease: "power3.out", immediateRender: false }}, T.flight_a);
  tl.set("#flash", {{ backgroundColor: "#ffd9a0" }}, T.speak_nyc - 0.3);
  tl.fromTo("#flash", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.12, ease: "power2.in", immediateRender: false }}, T.speak_nyc - 0.12);
  tl.to("#flash", {{ opacity: 0, duration: 0.45, ease: "power2.out", overwrite: "auto" }}, T.speak_nyc + 0.05);
  tl.set("#flash", {{ backgroundColor: "#ffffff" }}, T.speak_nyc + 0.7);
  tl.fromTo("#v_speak_nyc", {{ scale: 1.08 }}, {{ scale: 1, duration: 0.9, ease: "power3.out", immediateRender: false }}, T.speak_nyc);
  tl.fromTo("#v_suit_off", {{ scale: 1.1 }}, {{ scale: 1, duration: 1.2, ease: "power3.out", immediateRender: false }}, T.suit_off);
  // ---------- back into the engine for "from video generation to cloning": push in + flash, editor again, avatar rides again
  tl.set("#v_suit_off", {{ transformOrigin: "50% 62%" }}, T.enter2);
  tl.fromTo("#v_suit_off", {{ scale: 1 }}, {{ scale: 2.6, duration: T.stage2 + 0.5 - T.enter2, ease: "power2.in", immediateRender: false }}, T.enter2);
  tl.fromTo("#flash", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4, ease: "power2.in", immediateRender: false, overwrite: "auto" }}, T.stage2 - 0.15);
  tl.set("#stage", {{ opacity: 1 }}, T.stage2);
  tl.set("#mon", {{ ...EDIT, opacity: 1 }}, T.stage2);
  tl.set("#prompt", {{ height: 0 }}, T.stage2);
  tl.set(["#bin", "#insp", "#tl"], {{ x: 0, y: 0, opacity: 1 }}, T.stage2);
  tl.to("#flash", {{ opacity: 0, duration: 0.6, ease: "power2.out" }}, T.stage2 + 0.2);
  tl.set("#falls", {{ opacity: 0 }}, T.stage2 + 0.3);
  ride("#avwrap2", EDIT2_IN + 0.1, EDIT2_OUT - 0.05, 160, 1150);
  // slow camera drift on the whole editor stage so it never feels static
  tl.fromTo("#stage", {{ scale: 1.0, transformOrigin: "50% 50%" }}, {{ scale: 1.025, duration: EDIT_OUT - EDIT_IN, ease: "none", immediateRender: false }}, EDIT_IN);
  tl.fromTo("#stage", {{ scale: 1.0 }}, {{ scale: 1.025, duration: EDIT2_OUT - EDIT2_IN, ease: "none", immediateRender: false }}, EDIT2_IN);
  // song: dock to centre, then the beat card
  tl.to(["#bin", "#insp"], {{ opacity: 0, duration: 0.3 }}, T.W8 - 0.3);
  tl.to("#tl", {{ opacity: 0, y: 160, duration: 0.3 }}, T.W8 - 0.3);
  tl.to("#mon", {{ ...SONG, duration: 0.45, ease: "power3.inOut" }}, T.W8 - 0.35);
  tl.to("#mt1", {{ opacity: 0, duration: 0.15 }}, T.W8 - 0.1); tl.to("#mt0", {{ opacity: 0, duration: 0.01 }}, T.W8 - 0.1); tl.to("#mt2", {{ opacity: 1, duration: 0.15 }}, T.W8);
  tl.fromTo("#beat", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.1, immediateRender: false }}, T.beat);
  tl.fromTo("#beat h1", {{ scale: 0.85 }}, {{ scale: 1.0, duration: 0.4286, ease: "power3.out", repeat: 4 }}, T.beat);
  const bars = document.querySelectorAll("#beat .bars u"), bh = [0.45, 0.9, 0.6, 1, 0.5, 0.8, 0.35, 0.95, 0.7, 0.4, 0.85, 0.55];
  bars.forEach((u, i) => tl.to(u, {{ scaleY: bh[i % 12], duration: 0.21, yoyo: true, repeat: 9, ease: "power2.out" }}, T.beat + (i % 6) * 0.03));
  tl.set("#stage", {{ opacity: 0 }}, T.W9a);
  tl.to("#beat", {{ opacity: 0, duration: 0.2 }}, T.W9a - 0.05);
  // ---------- finale
  tl.fromTo("#v_W9a", {{ scale: 1.06 }}, {{ scale: 1, duration: 0.8, ease: "power3.out", immediateRender: false }}, T.W9a);
  xfade("#v_W9b", T.W9b - XF);
  tl.fromTo("#v_W9b", {{ scale: 1 }}, {{ scale: 1.08, duration: D.W9b + XF, ease: "power1.inOut", immediateRender: false }}, T.W9b - XF);
  tl.fromTo("#flash", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.16, ease: "power2.in", immediateRender: false }}, GAL - 0.16);
  tl.to("#flash", {{ opacity: 0, duration: 0.7, ease: "power2.out" }}, GAL + 0.02);
  tl.fromTo("#v_galaxy", {{ scale: 1.0 }}, {{ scale: 1.08, duration: GAL_DUR, ease: "power1.inOut", immediateRender: false }}, GAL);
  tl.to("#stars", {{ opacity: 1, duration: 0.4 }}, GAL + 0.1);
  document.querySelectorAll("#stars .st").forEach((el) => {{
    const a = parseFloat(el.dataset.a) * Math.PI / 180, d = 900 + parseFloat(el.dataset.d) * 500, tt = parseFloat(el.dataset.t), o = parseFloat(el.dataset.o);
    tl.fromTo(el, {{ x: Math.cos(a) * 40, y: Math.sin(a) * 40, opacity: 0, scale: 0.2 }}, {{ x: Math.cos(a) * d, y: Math.sin(a) * d * 0.62, opacity: 0.95, scale: 1.9, duration: tt, ease: "power2.in", immediateRender: false }}, GAL + 0.15 + o);
    tl.to(el, {{ opacity: 0, duration: 0.3 }}, GAL + 0.15 + o + tt - 0.3);
  }});
  tl.fromTo("#title", {{ opacity: 0 }}, {{ opacity: 1, duration: 1.0, ease: "power2.out", immediateRender: false }}, T.title);
  tl.fromTo("#title h1", {{ scale: 1.14 }}, {{ scale: 1.0, duration: 2.4, ease: "power3.out", immediateRender: false }}, T.title);
  window.__timelines["main"] = tl;
</script></body></html>'''
open(f"{HERE}/index.html", "w").write(html)
print("wrote index.html", len(html), "bytes; END", END)

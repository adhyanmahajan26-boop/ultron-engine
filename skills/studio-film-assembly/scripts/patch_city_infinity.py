"""build2.py -> build3.py : adds (1) animated avatar entering NYC and saying 'Even this is possible', (2) infinity-symbol ending."""
import re
s = open("build2.py").read()

# ---------------------------------------------------------------- timeline: city scene between speak_nyc and suit_off
s = s.replace('T["speak_nyc"] = round(t + 0.06, 3); t += 0.06 + D["speak_nyc"]\n',
'''T["speak_nyc"] = round(t + 0.06, 3); t += 0.06 + D["speak_nyc"]
T["city"] = round(t, 3)
CITY_FLY = 2.1
T["av_talk"] = round(T["city"] + CITY_FLY, 3)
t = T["av_talk"] + D["av_talk_cut"] + 0.5
CITY_D = round(t - T["city"], 3)
''')
# finale timing: keep, but longer ending for the infinity tunnel
s = s.replace("GAL = round(t, 2); GAL_DUR = 6.6", "GAL = round(t, 2); GAL_DUR = 7.4")
s = s.replace('T["title"] = GAL + 0.9', 'T["title"] = GAL + 1.3')
s = s.replace("XF = 0.09 ", "XF = 0.09 ")

# ---------------------------------------------------------------- audio / sfx
s = s.replace('audio.append(aud("speak_nyc", T["speak_nyc"], 1.0))',
'''audio.append(aud("speak_nyc", T["speak_nyc"], 1.0))
audio.append(aud("av_talk", T["av_talk"] + 0.02, 1.0))''')
s = s.replace('sf("whoosh_long", T["suit_off"] - 0.1, 0.6)',
'''sf("whoosh_long", T["city"] - 0.1, 0.8); sf("riser", T["city"] + 0.9, 0.7, d=1.0); sf("boom", T["av_talk"] - 0.1, 0.9, d=1.4); sf("click", T["av_talk"] + 0.02, 0.8)
sf("whoosh_long", T["suit_off"] - 0.1, 0.6)''')
s = s.replace('sf("riser", GAL - 0.9, 0.8, d=0.95); sf("boom", GAL, 1.0, d=2.2); sf("shimmer", GAL + 0.1, 0.9)',
'''sf("riser", T["W9b"] + 1.2, 0.8, d=1.3); sf("shimmer", T["W9b"] + 1.4, 0.9)
sf("riser", GAL - 0.9, 0.8, d=0.95); sf("boom", GAL, 1.0, d=2.2); sf("shimmer", GAL + 0.3, 0.9)''')

# ---------------------------------------------------------------- html: city layers inside #falls
s = s.replace('''    vid("suit_off", T["suit_off"], style=FULL + "z-index:24;")])''',
'''    vid("suit_off", T["suit_off"], style=FULL + "z-index:24;")])
city_html = (f'<div id="city">'
    + vid("nyc_city", T["city"], CITY_D, FULL + "z-index:25;")
    + f'<div id="cav1" class="cav"><video id="v_cav1" src="assets/v2/av_surf.webm" muted playsinline preload="auto" data-start="{T["city"]:.3f}" data-duration="{CITY_FLY + 0.1:.3f}" style="position:absolute;left:0;top:0;width:520px;height:560px;object-fit:contain;"></video><div class="board"></div></div>'
    + f'<video id="v_cav2" src="assets/v2/av_talk_cut.webm" muted playsinline preload="auto" data-start="{T["av_talk"]:.3f}" data-duration="{D["av_talk_cut"]:.3f}" style="{FULL}z-index:27;-webkit-mask-image:radial-gradient(ellipse 44% 58% at 50% 38%,#000 60%,transparent 100%);mask-image:radial-gradient(ellipse 44% 58% at 50% 38%,#000 60%,transparent 100%);"></video>'
    + '</div>')
falls = falls + city_html''')

# ---------------------------------------------------------------- finale html (replace galaxy video + stars)
a = s.index("star_rng = [")
b = s.index("beat = f'")
new_fin = '''import math
def lem(n, a=440, cx=960, cy=420):
    out = []
    for k in range(n):
        th = 2 * math.pi * k / n; d = 1 + math.sin(th) ** 2
        out.append((cx + a * math.cos(th) / d, cy + a * math.sin(th) * math.cos(th) / d))
    return out
INF_D = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in lem(260)) + " Z"
PART = [[round(x, 1), round(y, 1)] for x, y in lem(60)]
finale = "\\n".join([
    vid("W9a", T["W9a"], D["W9a"] + XF, FULL + "z-index:30;"), vid("W9b", T["W9b"] - XF, D["W9b"] + XF, FULL + "z-index:31;")])
zoom_html = '<div id="zoom">' + "".join(f'<img src="assets/img/galaxy_big_{i}.jpg" id="zl{i}">' for i in range(3)) + '</div>'
inf_html = (f'<svg id="inf" viewBox="0 0 1920 1080" width="1920" height="1080"><defs>'
    '<linearGradient id="gr" gradientUnits="userSpaceOnUse" x1="520" y1="0" x2="1400" y2="0"><stop offset="0" stop-color="#36d8ff"/><stop offset="0.5" stop-color="#b48bff"/><stop offset="1" stop-color="#ffb454"/></linearGradient>'
    '<filter id="gl" x="-20%" y="-40%" width="140%" height="180%"><feGaussianBlur stdDeviation="16"/></filter></defs>'
    f'<path id="infg" d="{INF_D}" pathLength="1" stroke="url(#gr)" stroke-width="34" fill="none" stroke-linecap="round" filter="url(#gl)" stroke-dasharray="1" stroke-dashoffset="1" opacity="0.95"/>'
    f'<path id="infp" d="{INF_D}" pathLength="1" stroke="url(#gr)" stroke-width="13" fill="none" stroke-linecap="round" stroke-dasharray="1" stroke-dashoffset="1"/></svg>'
    + "".join(f'<i class="pt" id="pt{i}"></i>' for i in range(30)))

'''
s = s[:a] + new_fin + s[b:]
s = s.replace('  <div id="fin">{finale}</div>\n  <div id="stars">{stars_html}</div>', '  <div id="fin">{finale}</div>\n  <div id="zoomwrap">{zoom_html}</div>\n  <div id="infwrap">{inf_html}</div>')

# ---------------------------------------------------------------- css
s = s.replace("  #stars {{ position:absolute; inset:0; z-index:35; pointer-events:none; opacity:0; overflow:hidden; }}\n", "  #zoomwrap {{ position:absolute; inset:0; z-index:32; opacity:0; background:#000; overflow:hidden; }}\n  #zoom img {{ position:absolute; left:50%; top:50%; width:1920px; height:1080px; margin:-540px 0 0 -960px; mix-blend-mode:screen; opacity:0; filter:contrast(1.7) brightness(.72) saturate(1.3); -webkit-mask-image:radial-gradient(circle at 50% 50%,#000 16%,transparent 66%); mask-image:radial-gradient(circle at 50% 50%,#000 16%,transparent 66%); }}\n  #infwrap {{ position:absolute; inset:0; z-index:36; pointer-events:none; }}\n  #inf {{ position:absolute; left:0; top:0; opacity:0; transform-origin:960px 420px; }}\n  .pt {{ position:absolute; left:0; top:0; width:12px; height:12px; margin:-6px 0 0 -6px; border-radius:50%; background:#fff; box-shadow:0 0 14px 4px rgba(190,215,255,.95); opacity:0; }}\n  .cav {{ position:absolute; left:0; top:0; width:520px; height:600px; z-index:26; opacity:0; }}\n  #city {{ position:absolute; inset:0; }}\n")
s = re.sub(r"  #stars \.st \{\{.*?\}\}\n", "", s)
s = s.replace("justify-content:center; opacity:0; pointer-events:none; background:radial-gradient(ellipse at center,rgba(0,0,0,.45) 0%,transparent 62%);",
              "justify-content:flex-end; padding-bottom:150px; box-sizing:border-box; opacity:0; pointer-events:none; background:linear-gradient(transparent 52%,rgba(0,0,0,.6) 100%);")
s = s.replace("#title h1 {{ margin:0; font-size:108px;", "#title h1 {{ margin:0; font-size:92px;")

# ---------------------------------------------------------------- js: city + finale
s = s.replace('  tl.fromTo("#v_suit_off", {{ scale: 1.1 }}, {{ scale: 1, duration: 1.2, ease: "power3.out", immediateRender: false }}, T.suit_off);',
'''  // ---- city: the animated avatar surfs in over New York, passes the camera, lands and says "Even this is possible."
  tl.fromTo("#v_nyc_city", {{ scale: 1.0 }}, {{ scale: 1.14, duration: CITY_D, ease: "none", immediateRender: false }}, T.city);
  tl.fromTo("#cav1", {{ x: 1050, y: 330, scale: 0.14, opacity: 0, transformOrigin: "50% 50%" }}, {{ x: 330, y: 190, scale: 1.55, opacity: 1, duration: 1.85, ease: "power2.in", immediateRender: false }}, T.city);
  tl.to("#cav1", {{ x: -420, y: 40, scale: 2.2, opacity: 0, duration: 0.3, ease: "power2.in" }}, T.city + 1.85);
  tl.fromTo("#v_cav2", {{ scale: 1.1, y: 40 }}, {{ scale: 1.0, y: 0, duration: 0.7, ease: "power3.out", immediateRender: false }}, T.av_talk);
  tl.fromTo("#flash", {{ opacity: 0, backgroundColor: "#ffd9a0" }}, {{ opacity: 0.8, duration: 0.1, immediateRender: false, overwrite: "auto" }}, T.av_talk - 0.12);
  tl.to("#flash", {{ opacity: 0, duration: 0.4, overwrite: "auto" }}, T.av_talk - 0.02);
  tl.set("#flash", {{ backgroundColor: "#ffffff" }}, T.av_talk + 0.5);
  tl.fromTo("#v_suit_off", {{ scale: 1.1 }}, {{ scale: 1, duration: 1.2, ease: "power3.out", immediateRender: false }}, T.suit_off);''')

a = s.index("  // ---------- finale\n")
b = s.index('  tl.fromTo("#title", {{ opacity: 0 }}')
js_fin = '''  // ---------- finale: arms open -> the infinity symbol draws itself between them -> flash -> an infinite galaxy tunnel behind the symbol -> title
  tl.fromTo("#v_W9a", {{ scale: 1.06 }}, {{ scale: 1, duration: 0.8, ease: "power3.out", immediateRender: false }}, T.W9a);
  xfade("#v_W9b", T.W9b - XF);
  tl.fromTo("#v_W9b", {{ scale: 1 }}, {{ scale: 1.08, duration: D.W9b + XF, ease: "power1.inOut", immediateRender: false }}, T.W9b - XF);
  const INF_T0 = T.W9b + 1.3, INF_DRAW = 1.5;
  tl.set("#inf", {{ opacity: 1, scale: 0.38, y: -350, transformOrigin: "960px 420px" }}, INF_T0 - 0.02);
  tl.to("#inf", {{ scale: 1.0, y: 0, duration: 0.55, ease: "power3.inOut" }}, GAL - 0.3);
  tl.fromTo(["#infg", "#infp"], {{ strokeDashoffset: 1 }}, {{ strokeDashoffset: 0, duration: INF_DRAW, ease: "power2.inOut", immediateRender: false }}, INF_T0);
  tl.fromTo("#flash", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.16, ease: "power2.in", immediateRender: false, overwrite: "auto" }}, GAL - 0.16);
  tl.to("#flash", {{ opacity: 0, duration: 0.7, ease: "power2.out", overwrite: "auto" }}, GAL + 0.02);
  tl.set("#zoomwrap", {{ opacity: 1 }}, GAL);
  // infinite zoom: three galaxy layers, each a galaxy inside the previous one, looping forever
  const R = 3.1, STEP = 0.25;
  document.querySelectorAll("#zoom img").forEach((img, i) => {{
    for (let n = 0; n * STEP < GAL_DUR + 0.3; n++) {{
      const eA = (i + n * STEP) % 3, eB = eA + STEP;
      const f = (e) => ({{ scale: 0.26 * Math.pow(R, e), opacity: Math.min(e / 0.5, 1) * Math.min((3 - e) / 0.75, 1) * 0.9, rotation: e * 18 }});
      const A = f(eA), B = f(Math.min(eB, 3));
      tl.fromTo(img, A, {{ ...B, duration: STEP, ease: "none", immediateRender: false }}, GAL + n * STEP);
    }}
  }});
  tl.fromTo("#zoom", {{ rotation: 0 }}, {{ rotation: 34, duration: GAL_DUR, ease: "none", immediateRender: false }}, GAL);
  tl.fromTo("#infwrap", {{ scale: 1 }}, {{ scale: 1.03, duration: GAL_DUR, ease: "power1.inOut", immediateRender: false }}, GAL);
  tl.to("#inf", {{ scale: 1.05, duration: 1.2, yoyo: true, repeat: 5, ease: "sine.inOut", overwrite: "auto" }}, GAL + 0.4);
  // stars stream along the infinity path
  const loops = Math.ceil((GAL_DUR + 0.4) / 3.0);
  for (let k = 0; k < 30; k++) {{
    const off = Math.floor(k * PT.length / 30);
    const pts = PT.map((_, j) => PT[(j + off) % PT.length]);
    tl.set("#pt" + k, {{ x: pts[0][0], y: pts[0][1] }}, GAL);
    tl.fromTo("#pt" + k, {{ opacity: 0 }}, {{ opacity: 0.4 + (k % 4) * 0.15, duration: 0.3, immediateRender: false }}, GAL + 0.3 + k * 0.05);
    tl.to("#pt" + k, {{ keyframes: pts.map((p) => ({{ x: p[0], y: p[1], duration: 3.0 / pts.length, ease: "none" }})), repeat: loops - 1 }}, GAL + 0.2);
  }}
'''
s = s[:a] + js_fin + s[b:]
s = s.replace('const END = {END}, GAL = {GAL}', 'const PT = {json.dumps(PART)};\n  const CITY_D = {CITY_D};\n  const END = {END}, GAL = {GAL}')
s = s.replace('tl.fromTo("#title h1", {{ scale: 1.14 }}', 'tl.fromTo("#title h1", {{ scale: 1.1 }}')
open("build3.py", "w").write(s)
print("build3.py written")

// stage.js — the 3D viewport. Everything drawn here comes from the real model
// in core.js; this file only turns its numbers into shapes, light and motion.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { toyLoss } from "./core.js";

const POS = new THREE.Color(0x4cc9f0); // positive numbers: cyan
const NEG = new THREE.Color(0xff8a4c); // negative numbers: orange
const IDLE = new THREE.Color(0x26324a); // a node holding (almost) nothing
const RIM = new THREE.Color(0x8193b6);
const BG = new THREE.Color(0x070a10);
const HOT = new THREE.Color(0xeaf6ff); // probability
const GRAD = new THREE.Color(0xb49cff); // the error signal flowing backward during training
const FLOOR_Y = -3.6;
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const COARSE = matchMedia("(pointer: coarse)").matches;

const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const showTok = (t) => (t === " " ? "␣" : t.replace(/^ /, "␣"));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const sgn = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const num = (v, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
const cls = (v) => (v >= 0 ? "pos" : "neg");

// colour for a signed value: idle slate → cyan/orange as |v| grows
function valColor(v, scale, out) {
  const a = Math.pow(Math.min(Math.abs(v) / scale, 1), 0.6);
  out.copy(IDLE).lerp(v >= 0 ? POS : NEG, a);
  if (a > 0.85) out.multiplyScalar(1 + (a - 0.85) * 3); // only the strongest reach the glow threshold
  return out;
}

// ───────────────────────────── HTML labels pinned to 3D points ─────────────────────────────
class Labels {
  constructor(host) {
    this.layer = document.createElement("div");
    this.layer.className = "stage-labels";
    host.appendChild(this.layer);
    this.items = new Set();
    this.v = new THREE.Vector3();
  }
  // ax: 0 = label starts at the point, 0.5 = centred, 1 = label ends at the point
  add(mode, html, cls = "", pos = new THREE.Vector3(), { ax = 0.5, clamp = false, tag = "div" } = {}) {
    const el = document.createElement(tag);
    el.className = "lbl " + cls;
    if (tag === "button") el.type = "button";
    el.innerHTML = html;
    this.layer.appendChild(el);
    const item = { el, mode, pos, show: true, html, ax, clamp };
    this.items.add(item);
    return item;
  }
  set(item, html) { if (item.html !== html) { item.html = html; item.el.innerHTML = html; item.w = undefined; } }
  update(mode, camera, w, h) {
    for (const it of this.items) {
      if (!it.show || it.mode !== mode) { if (it.el.style.display !== "none") it.el.style.display = "none"; continue; }
      this.v.copy(it.pos).project(camera);
      if (this.v.z > 1) { it.el.style.display = "none"; continue; }
      const px = (this.v.x * 0.5 + 0.5) * w, py = (-this.v.y * 0.5 + 0.5) * h;
      if (!it.clamp && (px < 0 || px > w || py < 0 || py > h)) { it.el.style.display = "none"; continue; }
      if (it.el.style.display === "none") { it.el.style.display = ""; it.w = undefined; }
      if (it.w === undefined) { it.w = it.el.offsetWidth; it.h = it.el.offsetHeight; }
      let x = px - it.ax * it.w, y = py - it.h / 2;
      if (it.clamp) { x = Math.max(8, Math.min(w - it.w - 8, x)); y = Math.max(8, Math.min(h - it.h - 8, y)); }
      it.el.style.transform = `translate(${x.toFixed(1)}px,${y.toFixed(1)}px)`;
    }
  }
}

// ───────────────────────────── Thick lines (one draw call per width) ─────────────────────────────
class FatLines {
  constructor(cap, width, opacity = 1) {
    this.cap = cap;
    this.width = width;
    this.geo = new LineSegmentsGeometry();
    this.pos = new Float32Array(cap * 6);
    this.col = new Float32Array(cap * 6);
    this.geo.setPositions(this.pos);
    this.geo.setColors(this.col);
    this.mat = new LineMaterial({ linewidth: width, vertexColors: true, worldUnits: false, transparent: true, opacity, depthWrite: false });
    this.baseOpacity = opacity;
    this.mesh = new LineSegments2(this.geo, this.mat);
    this.mesh.frustumCulled = false;
    this.segs = [];
    this.geo.instanceCount = 0;
  }
  // segs: [{a, b, col, layer}]; grow(layer) says how far (0–1) each layer's lines are drawn
  setSegs(segs) { this.segs = segs.slice(0, this.cap); }
  draw(grow) {
    const p = this.pos, c = this.col;
    let n = 0;
    for (const s of this.segs) {
      const g = grow(s.layer);
      if (g <= 0.001) continue;
      const o = n * 6;
      p[o] = s.a.x; p[o + 1] = s.a.y; p[o + 2] = s.a.z;
      p[o + 3] = s.a.x + (s.b.x - s.a.x) * g; p[o + 4] = s.a.y + (s.b.y - s.a.y) * g; p[o + 5] = s.a.z + (s.b.z - s.a.z) * g;
      c[o] = c[o + 3] = s.col.r; c[o + 1] = c[o + 4] = s.col.g; c[o + 2] = c[o + 5] = s.col.b;
      n++;
    }
    this.geo.instanceCount = n;
    this.geo.attributes.instanceStart.data.needsUpdate = true;
    this.geo.attributes.instanceColorStart.data.needsUpdate = true;
  }
}

// ───────────────────────────── Nodes: flat fill + light rim ─────────────────────────────
function makeNodes(positions, radius, segs) {
  const geo = new THREE.SphereGeometry(1, segs[0], segs[1]);
  const fill = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ fog: false, toneMapped: false }), positions.length);
  const rim = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ side: THREE.BackSide, fog: false, toneMapped: false }), positions.length);
  for (let i = 0; i < positions.length; i++) { fill.setColorAt(i, IDLE); rim.setColorAt(i, RIM); }
  const m = new THREE.Matrix4();
  const nodes = {
    fill, rim, pos: positions, r: radius, count: positions.length,
    set(i, s, col, rimCol) {
      const r = radius * s;
      m.makeScale(r, r, r).setPosition(positions[i]); fill.setMatrixAt(i, m);
      m.makeScale(r * 1.22, r * 1.22, r * 1.22).setPosition(positions[i]); rim.setMatrixAt(i, m);
      fill.setColorAt(i, col);
      if (rimCol) rim.setColorAt(i, rimCol);
    },
    commit() {
      fill.instanceMatrix.needsUpdate = rim.instanceMatrix.needsUpdate = true;
      fill.instanceColor.needsUpdate = rim.instanceColor.needsUpdate = true;
    },
  };
  return nodes;
}

// ───────────────────────────── The network ─────────────────────────────
// Forward-pass stages, matching the five steps of Chapter 5.
export const NET_STAGES = ["Tokens", "Embeddings", "Hidden layer", "Scores", "Probabilities"];

class NetworkView {
  constructor(stage, cfg) {
    const { V, D, N, H, vocab, outOrder, categoryOf, catHex } = cfg;
    Object.assign(this, { stage, V, D, N, H, I: N * D, vocab, categoryOf });
    this.group = new THREE.Group();
    // Layers face the default camera: every node in a layer is visible side by side.
    const U = new THREE.Vector3(0.74, 0, -0.67).normalize();
    this.U = U;
    this.inPos = []; this.hidPos = []; this.outPos = new Array(V);
    for (let n = 0; n < N; n++) for (let d = 0; d < D; d++) this.inPos.push(new THREE.Vector3(-7, ((N - 1) / 2 - n) * 1.25, 0).addScaledVector(U, (d - (D - 1) / 2) * 0.46));
    const hc = 6, hr = Math.ceil(H / hc);
    for (let j = 0; j < H; j++) this.hidPos.push(new THREE.Vector3(0, ((hr - 1) / 2 - Math.floor(j / hc)) * 0.66, 0).addScaledVector(U, ((j % hc) - (hc - 1) / 2) * 0.66));
    const oc = Math.ceil(Math.sqrt(V)), orr = Math.ceil(V / oc);
    (outOrder || [...Array(V).keys()]).forEach((k, slot) => {
      this.outPos[k] = new THREE.Vector3(7, ((orr - 1) / 2 - Math.floor(slot / oc)) * 0.42, 0).addScaledVector(U, ((slot % oc) - (oc - 1) / 2) * 0.42);
    });
    this.allPos = [...this.inPos, ...this.hidPos, ...this.outPos];

    const segs = stage.narrow ? [10, 8] : [16, 12];
    this.nIn = makeNodes(this.inPos, 0.17, segs);
    this.nHid = makeNodes(this.hidPos, 0.21, segs);
    this.nOut = makeNodes(this.outPos, 0.135, segs);
    this.outRim = [];
    for (let k = 0; k < V; k++) {
      const c = new THREE.Color(catHex[categoryOf(vocab[k])] || 0x58657a).lerp(RIM, 0.45);
      this.outRim.push(c);
      this.nOut.rim.setColorAt(k, c);
    }
    for (const n of [this.nIn, this.nHid, this.nOut]) this.group.add(n.rim, n.fill);

    // Faint 1-pixel context: every input→hidden weight plus the strongest hidden→output ones.
    const ctxN = this.I * H + 1400;
    const cg = new THREE.BufferGeometry();
    this.ctxPos = new Float32Array(ctxN * 6);
    cg.setAttribute("position", new THREE.BufferAttribute(this.ctxPos, 3));
    let q = 0;
    for (let j = 0; j < H; j++) for (let i = 0; i < this.I; i++) {
      const a = this.inPos[i], b = this.hidPos[j];
      this.ctxPos.set([a.x, a.y, a.z, b.x, b.y, b.z], q); q += 6;
    }
    this.ctxW2Start = q;
    this.ctxMat = new THREE.LineBasicMaterial({ color: 0x8796b8, transparent: true, opacity: stage.narrow ? 0.04 : 0.065, depthWrite: false, fog: false });
    this.context = new THREE.LineSegments(cg, this.ctxMat);
    this.context.frustumCulled = false;
    this.group.add(this.context);

    const w = stage.narrow ? [3, 1.75, 1.2] : [4, 2.25, 1.25];
    this.B = [new FatLines(600, w[2], 0.85), new FatLines(600, w[1]), new FatLines(600, w[0])]; // thin first, thick drawn on top
    this.focusLines = new FatLines(260, stage.narrow ? 2.5 : 3);
    for (const b of [...this.B, this.focusLines]) this.group.add(b.mesh);

    // Glowing signals that travel along the strongest input × weight paths.
    // Each spark is a head plus a short fading tail; brightness follows the real product size.
    this.SPARK_MAX = 320;
    this.stPos = new Float32Array(this.SPARK_MAX * 6);
    this.stCol = new Float32Array(this.SPARK_MAX * 6);
    this.streakGeo = new LineSegmentsGeometry();
    this.streakGeo.setPositions(this.stPos);
    this.streakGeo.setColors(this.stCol);
    this.streakGeo.instanceCount = 0;
    this.streakMat = new LineMaterial({ linewidth: stage.narrow ? 1.6 : 2, vertexColors: true, worldUnits: false, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false });
    this.streaks = new LineSegments2(this.streakGeo, this.streakMat);
    this.streaks.frustumCulled = false;
    this.streaks.renderOrder = 3;
    this.group.add(this.streaks);
    this.sparks = [];
    this.flow = [];
    this.gradFlow = [];
    this.training = false;
    this.nextWave = 0;
    // a halo that marks the model's top guess
    this.halo = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.36, 48), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, toneMapped: false, side: THREE.DoubleSide, depthWrite: false, fog: false }));
    this.halo.visible = false;
    this.group.add(this.halo);

    // labels
    const L = stage.labels;
    // ── The way in: text → tokenizer → token tiles with IDs → one embedding row each
    this.intake = new THREE.Group();
    this.group.add(this.intake);
    this.tiles = [];
    this.tileKey = "";
    this.sentenceAt = new THREE.Vector3(-7, this.inPos[0].y + 1.5, 0).addScaledVector(U, -0.9);
    this.textLabel = L.add("net", "", "intake", this.sentenceAt.clone().add(new THREE.Vector3(0, 0.75, 0)));
    this.slotLabels = [];
    for (let n = 0; n < N; n++) {
      const lab = L.add("net", `<i aria-hidden="true">▾</i>`, "slot", this.inPos[n * D].clone(), { ax: 1, tag: "button" });
      lab.el.addEventListener("click", (e) => { e.stopPropagation(); this.onSlot?.(n, lab.el); });
      this.slotLabels.push(lab);
    }
    const leftOf = (n, d) => this.inPos[n * D].clone().addScaledVector(U, -d);
    this.intakeFit = [leftOf(0, 2.6), leftOf(1, 2.6), leftOf(2, 2.6), this.sentenceAt.clone().addScaledVector(U, -2.2), this.sentenceAt.clone().addScaledVector(U, 2.2), this.sentenceAt.clone().add(new THREE.Vector3(0, 1.0, 0))];
    this.allPos = [...this.inPos, ...this.hidPos, ...this.outPos, ...this.intakeFit];
    this.inZoom = [...this.inPos, ...this.intakeFit];
    const below = (arr, dy) => { const c = arr.reduce((s, p) => s.add(p), new THREE.Vector3()).multiplyScalar(1 / arr.length); c.y = Math.min(...arr.map((p) => p.y)) - dy; return c; };
    this.layerLabels = [
      L.add("net", `Embeddings<small>${N} tokens × ${D} numbers</small>`, "layer", below(this.inPos, 0.75), { tag: "button" }),
      L.add("net", `Hidden layer<small>${H} neurons</small>`, "layer", below(this.hidPos, 0.75), { tag: "button" }),
      L.add("net", `Output<small>${V} token scores</small>`, "layer", below(this.outPos.filter(Boolean), 0.7), { tag: "button" }),
    ];
    this.layerLabels.forEach((l, i) => l.el.addEventListener("click", (e) => { e.stopPropagation(); stage.zoomLayer(i); }));
    this.rankLabels = Array.from({ length: 5 }, (_, r) => L.add("net", String(r + 1), "rank", new THREE.Vector3()));

    this.model = null;
    this.f = null;          // the forward pass being shown
    this.pick = null;       // token id that was sampled, if any
    this.view = "signal";
    this.P = 4; this.S = 4; this.speed = 1.3;
    this.focusSel = null; this.pinned = false;
    this.topW1 = null; this.topW2 = null; this.topAt = -1;
    this.dirty = true;
    this.ver = 0;
    this.c = new THREE.Color(); this.c2 = new THREE.Color();
  }

  setModel(model) {
    if (this.model === model) return;
    this.model = model;
    this.f = null;
    this.topAt = -1;
    this.refreshWeights(true);
  }

  // weight-view lines and context lines from the current weights
  refreshWeights(force) {
    const m = this.model;
    if (!m) return;
    const now = performance.now();
    if (force || now - this.topAt > 1000 || !this.topW1) {
      this.topAt = now;
      const n1 = this.stage.narrow ? 120 : 240, n2 = this.stage.narrow ? 160 : 320;
      this.topW1 = topAbs(m.W1, n1);
      this.topW2 = topAbs(m.W2, Math.max(n2, 1400));
      // refresh the faint context set too
      const H = this.H;
      let q = this.ctxW2Start;
      for (const t of this.topW2.slice(0, 1400)) {
        const k = Math.floor(t / H), j = t % H, a = this.hidPos[j], b = this.outPos[k];
        this.ctxPos[q] = a.x; this.ctxPos[q + 1] = a.y; this.ctxPos[q + 2] = a.z; this.ctxPos[q + 3] = b.x; this.ctxPos[q + 4] = b.y; this.ctxPos[q + 5] = b.z; q += 6;
      }
      this.context.geometry.attributes.position.needsUpdate = true;
    }
    this.dirty = true;
  }

  // ── what the thick lines show
  buildSegs() {
    const m = this.model, f = this.f, H = this.H, I = this.I, narrow = this.stage.narrow;
    const segs = [];
    const seg = (a, b, v, rel, layer) => segs.push({ a, b, v, rel, layer, col: new THREE.Color().copy(v >= 0 ? POS : NEG).multiplyScalar(0.3 + 0.7 * rel) });
    let mode = this.view === "signal" && f ? "signal" : "weights";
    if (this.focusSel) mode = "focus";
    if (mode === "weights") {
      const n2 = narrow ? 160 : 320;
      let m1 = 0.9, m2 = 0.7; // a trained model's typical largest weights; an untrained model's output weights look faint, as they are
      for (const t of this.topW1) m1 = Math.max(m1, Math.abs(m.W1[t]));
      for (const t of this.topW2.slice(0, n2)) m2 = Math.max(m2, Math.abs(m.W2[t]));
      for (const t of this.topW1) seg(this.inPos[t % I], this.hidPos[Math.floor(t / I)], m.W1[t], Math.abs(m.W1[t]) / m1, 1);
      for (const t of this.topW2.slice(0, n2)) seg(this.hidPos[t % H], this.outPos[Math.floor(t / H)], m.W2[t], Math.abs(m.W2[t]) / m2, 2);
      this.scaleNote = `Each line is one weight. Showing the strongest ${this.topW1.length + n2} of ${(I * H + H * this.V).toLocaleString("en-US")}.`;
    } else if (mode === "signal") {
      const c1 = [];
      for (let j = 0; j < H; j++) for (let i = 0; i < I; i++) c1.push([f.x[i] * m.W1[j * I + i], i, j]);
      c1.sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]));
      const k1 = narrow ? 100 : 200;
      const m1 = Math.max(Math.abs(c1[0][0]), 0.15);
      for (const [v, i, j] of c1.slice(0, k1)) seg(this.inPos[i], this.hidPos[j], v, Math.abs(v) / m1, 1);
      const top = this.top();
      const c2 = [];
      top.forEach((k, r) => {
        const arr = [];
        for (let j = 0; j < H; j++) arr.push([f.h[j] * m.W2[k * H + j], j, k]);
        arr.sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]));
        c2.push(...arr.slice(0, r === 0 ? (narrow ? 48 : 48) : narrow ? 8 : 16));
      });
      const m2 = Math.max(...c2.map((x) => Math.abs(x[0])), 0.1);
      for (const [v, j, k] of c2) seg(this.hidPos[j], this.outPos[k], v, Math.abs(v) / m2, 2);
      this.scaleNote = `Each line is input × weight for this prediction. Brightest = ${Math.max(Math.abs(c1[0][0]), m2).toFixed(2)}.`;
    }
    // focus set (drawn separately, on top)
    const fsegs = [];
    if (mode === "focus") {
      const s = this.focusSel, push = (a, b, v, mx, layer) => fsegs.push({ a, b, v, layer, col: new THREE.Color().copy(v >= 0 ? POS : NEG).multiplyScalar(0.35 + 0.65 * Math.abs(v) / mx) });
      const xs = f ? f.x : null, hs = f ? f.h : null;
      if (s.layer === "hid") {
        const j = s.index, inc = [], out = [];
        for (let i = 0; i < I; i++) inc.push([xs ? xs[i] * m.W1[j * I + i] : m.W1[j * I + i], i]);
        for (let k = 0; k < this.V; k++) out.push([hs ? hs[j] * m.W2[k * H + j] : m.W2[k * H + j], k]);
        out.sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]));
        const mi = Math.max(...inc.map((x) => Math.abs(x[0])), 1e-6), mo = Math.max(Math.abs(out[0][0]), 1e-6);
        for (const [v, i] of inc) push(this.inPos[i], this.hidPos[j], v, mi, 1);
        for (const [v, k] of out.slice(0, 24)) push(this.hidPos[j], this.outPos[k], v, mo, 2);
      } else if (s.layer === "out") {
        const k = s.index, inc = [];
        for (let j = 0; j < H; j++) inc.push([hs ? hs[j] * m.W2[k * H + j] : m.W2[k * H + j], j]);
        const mi = Math.max(...inc.map((x) => Math.abs(x[0])), 1e-6);
        for (const [v, j] of inc) push(this.hidPos[j], this.outPos[k], v, mi, 2);
      } else {
        const i = s.index, out = [];
        for (let j = 0; j < H; j++) out.push([xs ? xs[i] * m.W1[j * I + i] : m.W1[j * I + i], j]);
        const mo = Math.max(...out.map((x) => Math.abs(x[0])), 1e-6);
        for (const [v, j] of out) push(this.inPos[i], this.hidPos[j], v, mo, 1);
      }
      fsegs.sort((a, b) => Math.abs(a.v) - Math.abs(b.v));
      this.scaleNote = "Highlighted: this neuron's connections, as input × weight.";
      const r = { in: new Set(), hid: new Set(), out: new Set() };
      r[s.layer].add(s.index);
      for (const g of fsegs) for (const [arr, layer] of [[this.inPos, "in"], [this.hidPos, "hid"], [this.outPos, "out"]]) {
        const ia = arr.indexOf(g.a), ib = arr.indexOf(g.b);
        if (ia >= 0) r[layer].add(ia);
        if (ib >= 0) r[layer].add(ib);
      }
      this.related = r;
    } else this.related = null;
    this.computeFlow();
    segs.sort((a, b) => a.rel - b.rel);
    this.B[2].setSegs(segs.filter((s) => s.rel >= 0.5));
    this.B[1].setSegs(segs.filter((s) => s.rel >= 0.2 && s.rel < 0.5));
    this.B[0].setSegs(segs.filter((s) => s.rel < 0.2));
    this.focusLines.setSegs(fsegs);
    this.mode = mode;
    this.dirty = false;
    this.ver++;
    this.stage.onScale?.(this.scaleNote);
  }

  // the paths that carry the most signal for the current prediction (real x·w and h·w products)
  computeFlow() {
    const m = this.model, f = this.f, H = this.H, I = this.I;
    this.flow = [];
    if (!f) return;
    const n1 = this.stage.narrow ? 40 : 80;
    const c1 = [];
    for (let j = 0; j < H; j++) for (let i = 0; i < I; i++) c1.push([f.x[i] * m.W1[j * I + i], i, j]);
    c1.sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]));
    const m1 = Math.max(Math.abs(c1[0][0]), 0.15);
    for (const [v, i, j] of c1.slice(0, n1)) this.flow.push({ a: this.inPos[i], b: this.hidPos[j], v, rel: Math.abs(v) / m1, layer: 1 });
    const top = this.top(), c2 = [];
    top.slice(0, 3).forEach((k, r) => {
      const arr = [];
      for (let j = 0; j < H; j++) arr.push([f.h[j] * m.W2[k * H + j], j, k]);
      arr.sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]));
      c2.push(...arr.slice(0, r === 0 ? (this.stage.narrow ? 24 : 40) : 8));
    });
    const m2 = Math.max(...c2.map((x) => Math.abs(x[0])), 0.1);
    for (const [v, j, k] of c2) this.flow.push({ a: this.hidPos[j], b: this.outPos[k], v, rel: Math.abs(v) / m2, layer: 2 });
  }

  top() {
    const p = this.f ? this.f.p : null;
    return p ? [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, 5) : [];
  }

  // Show a forward pass. from = first stage to animate from (0 = replay everything).
  show(f, { from = 0, speed = 1.3, pick = null } = {}) {
    this.f = f;
    this.pick = pick;
    this.P = REDUCED ? 4 : Math.min(this.P, from);
    this.S = 4;
    this.speed = speed;
    this.labelSlots();
    this.dirty = true;
    this.ver++;
    this.updateReadout();
  }
  // New numbers, same picture: used while training so nothing restarts.
  live(f) {
    this.f = f;
    this.pick = null;
    this.labelSlots();
    this.dirty = true;
    this.updateReadout();
  }
  // During training: the latest step's real gradients. The streaks run backward (output → hidden → input)
  // along the weights that got the biggest nudge, which is what backpropagation does.
  setTraining(on) { this.training = on; if (!on) { this.gradFlow = []; this.sparks.length = 0; } }
  trainStep(model) {
    if (!this.training) return;
    const H = this.H, I = this.I, n2 = this.stage.narrow ? 30 : 60, n1 = this.stage.narrow ? 30 : 60;
    const g2 = topAbs(model.gW2, n2), g1 = topAbs(model.gW1, n1);
    const m2 = Math.abs(model.gW2[g2[0]]) || 1e-9, m1 = Math.abs(model.gW1[g1[0]]) || 1e-9;
    this.gradFlow = [
      ...g2.map((t) => ({ a: this.outPos[Math.floor(t / H)], b: this.hidPos[t % H], rel: Math.abs(model.gW2[t]) / m2, layer: 2, grad: true })),
      ...g1.map((t) => ({ a: this.hidPos[Math.floor(t / I)], b: this.inPos[t % I], rel: Math.abs(model.gW1[t]) / m1, layer: 1, grad: true })),
    ];
  }
  setStage(k, speed = 2.6) { this.S = Math.max(0, Math.min(4, k)); this.speed = speed; if (REDUCED) this.P = this.S; this.stage.onStage?.(this.S); }
  setView(v) { this.view = v; this.dirty = true; }
  focus(sel, pinned = false) {
    const same = sel && this.focusSel && sel.layer === this.focusSel.layer && sel.index === this.focusSel.index;
    if (same && pinned === this.pinned) return;
    this.focusSel = sel; this.pinned = !!sel && pinned;
    this.dirty = true;
  }
  labelSlots() {
    if (!this.f) return;
    const ctx = this.f.ctx, key = ctx.join(",");
    this.slotLabels.forEach((l, n) => l.el.setAttribute("aria-label", `Word ${n + 1}: ${showTok(this.vocab[ctx[n]])}. Change it`));
    const text = ctx.map((id) => this.vocab[id]).filter((t) => !t.startsWith("[")).join("").trim();
    this.stage.labels.set(this.textLabel, `<span class="k">model reads</span> “${esc(text || "(start of a sentence)")}”<span class="k2">tokenizer → ${ctx.map((id) => `${esc(showTok(this.vocab[id]))} <b>#${id}</b>`).join(" · ")}</span>`);
    if (key === this.tileKey) return;
    this.tileKey = key;
    for (const t of this.tiles) { this.intake.remove(t.mesh); t.mesh.geometry.dispose(); t.mesh.material.dispose(); t.face.material.map.dispose(); t.face.material.dispose(); }
    this.tiles = ctx.map((id, n) => makeTokenTile(showTok(this.vocab[id]), id, this.vocab[id].startsWith("[")));
    // start: side by side as one piece of text, above the input layer
    const total = this.tiles.reduce((s, t) => s + t.w, 0);
    let x = -total / 2;
    this.tiles.forEach((t, n) => {
      t.n = n;
      t.joined = x + t.w / 2; x += t.w;
      t.home = this.inPos[n * this.D].clone().addScaledVector(this.U, -(0.42 + t.w / 2));
      t.mesh.userData.slot = n;
      this.intake.add(t.mesh);
      this.slotLabels[n].pos.copy(t.home).addScaledVector(this.U, -(t.w / 2 + 0.12));
    });
  }

  updateReadout() {
    const el = this.stage.readout;
    if (!el) return;
    const f = this.f;
    if (!f) { el.hidden = true; return; }
    const top = this.top();
    const ctx = f.ctx.map((id) => esc(showTok(this.vocab[id]))).join(" ");
    if (this.stage.narrow) {
      el.innerHTML = `<span class="ro-k">next →</span> ` + top.slice(0, 3).map((k) => `<b>${esc(showTok(this.vocab[k]))}</b> ${(f.p[k] * 100).toFixed(0)}%`).join(" · ");
    } else {
      const p0 = f.p[top[0]];
      el.innerHTML = `<div class="ro-h">Next token after <span>${ctx}</span></div>` + top.map((k, r) =>
        `<div class="ro-r${k === this.pick ? " pick" : ""}"><span class="ro-n">${r + 1}</span><span class="ro-t">${esc(showTok(this.vocab[k]))}</span><span class="ro-b"><i style="width:${((f.p[k] / Math.max(p0, 1e-9)) * 100).toFixed(1)}%"></i></span><span class="ro-p">${(f.p[k] * 100).toFixed(1)}%</span></div>`).join("") +
        (this.pick !== null && !top.includes(this.pick) ? `<div class="ro-r pick"><span class="ro-n">·</span><span class="ro-t">${esc(showTok(this.vocab[this.pick]))}</span><span class="ro-b"></span><span class="ro-p">${(f.p[this.pick] * 100).toFixed(1)}%</span></div>` : "");
    }
  }

  // Which node is under the pointer? Screen-space search, so small nodes are still easy to hit.
  pickAt(px, py, w, h, touch) {
    const cam = this.stage.camera, v = new THREE.Vector3();
    let best = null, bd = Infinity;
    const test = (arr, layer, r) => arr.forEach((p, index) => {
      v.copy(p).project(cam);
      const x = (v.x * 0.5 + 0.5) * w, y = (-v.y * 0.5 + 0.5) * h;
      const d = Math.hypot(x - px, y - py);
      const rad = touch ? 24 : Math.max(12, r);
      if (d < rad && d < bd) { bd = d; best = { layer, index, x, y }; }
    });
    const scale = h / (2 * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)) * cam.position.distanceTo(this.stage.controls.target));
    test(this.inPos, "in", 0.17 * scale * 1.4);
    test(this.hidPos, "hid", 0.21 * scale * 1.4);
    test(this.outPos, "out", 0.135 * scale * 1.6);
    return best;
  }

  // The tooltip: real numbers for the node under the pointer.
  tipHtml(sel) {
    const m = this.model, f = this.f, H = this.H, I = this.I, D = this.D, V = this.V;
    const tokOf = (i) => esc(showTok(this.vocab[f ? f.ctx[Math.floor(i / D)] : 0]));
    const rows = (arr) => arr.map((r) => `<div class="tip-r">${r}</div>`).join("");
    if (sel.layer === "hid") {
      const j = sel.index;
      let html = `<div class="tip-h">Hidden neuron #${j + 1}</div>`;
      if (!f) return html + `<div class="tip-m">Run a prediction to see its value.</div>`;
      const inc = [];
      for (let i = 0; i < I; i++) inc.push([f.x[i] * m.W1[j * I + i], i]);
      inc.sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]));
      const out = [];
      for (let k = 0; k < V; k++) out.push([f.h[j] * m.W2[k * H + j], k]);
      out.sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]));
      html += `<div class="tip-m">z = ${num(f.z1[j])} → tanh(z) = <b class="${cls(f.h[j])}">${num(f.h[j], 3)}</b></div>`;
      html += `<div class="tip-s">Biggest inputs (x × w)</div>` + rows(inc.slice(0, 3).map(([v, i]) => `${tokOf(i)}·${(i % D) + 1} &nbsp;${num(f.x[i])} × ${num(m.W1[j * I + i])} = <b class="${cls(v)}">${sgn(v)}</b>`));
      html += `<div class="tip-s">Pushes on the next token</div>` + rows(out.slice(0, 3).map(([v, k]) => `${esc(showTok(this.vocab[k]))} <b class="${cls(v)}">${sgn(v)}</b>`));
      return html;
    }
    if (sel.layer === "out") {
      const k = sel.index;
      let html = `<div class="tip-h">Output: ${esc(showTok(this.vocab[k]))}</div>`;
      if (!f) return html + `<div class="tip-m">Run a prediction to see its score.</div>`;
      const inc = [];
      for (let j = 0; j < H; j++) inc.push([f.h[j] * m.W2[k * H + j], j]);
      inc.sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]));
      html += `<div class="tip-m">score ${num(f.logits[k])} → p = <b>${(f.p[k] * 100).toFixed(2)}%</b></div>`;
      html += `<div class="tip-s">Biggest pushes (h × w)</div>` + rows(inc.slice(0, 3).map(([v, j]) => `from #${j + 1} &nbsp;${num(f.h[j])} × ${num(m.W2[k * H + j])} = <b class="${cls(v)}">${sgn(v)}</b>`));
      html += `<div class="tip-r tip-m">bias b₂ = ${num(m.b2[k], 3)}</div>`;
      return html;
    }
    const i = sel.index;
    let html = `<div class="tip-h">Input ${tokOf(i)}, number ${(i % D) + 1} of ${D}</div>`;
    if (!f) return html;
    const out = [];
    for (let j = 0; j < H; j++) out.push([f.x[i] * m.W1[j * I + i], j]);
    out.sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]));
    html += `<div class="tip-m">value x${i + 1} = <b class="${cls(f.x[i])}">${num(f.x[i], 3)}</b> (row ${f.ctx[Math.floor(i / D)]} of the embedding table)</div>`;
    html += `<div class="tip-s">Strongest effects (x × w)</div>` + rows(out.slice(0, 3).map(([v, j]) => `→ neuron #${j + 1} <b class="${cls(v)}">${sgn(v)}</b>`));
    return html;
  }

  update(t, dt) {
    if (!this.model) return;
    if (this.dirty) this.buildSegs();
    // progress through the five stages
    if (this.P !== this.S) {
      const step = dt * this.speed;
      this.P = this.P < this.S ? Math.min(this.S, this.P + step) : Math.max(this.S, this.P - step * 2);
    }
    const P = this.f ? this.P : 0;
    const tk = clamp01(P / 0.55), e1 = clamp01((P - 0.5) / 0.5), g1 = clamp01((P - 1) / 0.6), e2 = clamp01((P - 1.6) / 0.4), g2 = clamp01((P - 2) / 0.6), e3 = clamp01((P - 2.6) / 0.4), e4 = clamp01(P - 3);
    const growing = this.mode === "signal";
    const grow = (layer) => (growing ? ease(layer === 1 ? g1 : g2) : 1);
    const lineKey = `${this.ver}|${growing ? g1.toFixed(3) + g2.toFixed(3) : "w"}`;
    if (lineKey !== this.lineKey) { this.lineKey = lineKey; for (const b of this.B) b.draw(grow); this.focusLines.draw(() => 1); }
    const focusing = !!this.focusSel;
    for (const b of this.B) b.mat.opacity = focusing ? 0.12 : b.baseOpacity * (this.training && !REDUCED ? 0.75 : 1);
    this.ctxMat.opacity = (focusing ? 0.02 : this.stage.narrow ? 0.04 : 0.065) * (this.stage.quality > 1 ? 0 : 1);

    // a short flash as each layer switches on
    const bell = (e) => (REDUCED ? 0 : Math.sin(Math.PI * e));
    const pop1 = bell(e1), pop2 = bell(e2), pop4 = bell(e4);
    this.updateSparks(t, g1, g2, focusing);
    this.updateTiles(t, tk, focusing);
    // node colours (only recomputed when the picture changes)
    const f = this.f, c = this.c, c2 = this.c2, sel = this.focusSel;
    const nodeKey = `${this.ver}|${P.toFixed(3)}|${sel ? sel.layer + sel.index : ""}`;
    if (nodeKey === this.nodeKey) { this.updateLabelsOnly(t, e4, focusing, tk); return; }
    this.nodeKey = nodeKey;
    const related = sel ? this.related : null;
    const dim = (col, layer, i) => (related && !(related[layer] && related[layer].has(i)) ? col.lerp(BG, 0.6) : col);
    const sx = f ? (f._sx ??= Math.max(...Array.from(f.x, Math.abs), 0.3)) : 1, sh = f ? (f._sh ??= Math.max(...Array.from(f.h, Math.abs), 0.3)) : 1;
    for (let i = 0; i < this.I; i++) {
      c.copy(IDLE);
      const ei = clamp01(e1 * 1.7 - ((i % this.D) / this.D) * 0.7); // numbers are read out of the table left to right
      if (f && ei > 0) c.lerp(valColor(f.x[i], sx, c2), ei).multiplyScalar(1 + 0.4 * bell(ei));
      const isSel = sel && sel.layer === "in" && sel.index === i;
      this.nIn.set(i, (isSel ? 1.4 : 1) * (1 + 0.3 * pop1), dim(c, "in", i), isSel ? HOT : RIM);
    }
    for (let j = 0; j < this.H; j++) {
      c.copy(IDLE);
      let s = 1;
      if (f && e2 > 0) { c.lerp(valColor(f.h[j], sh, c2), e2).multiplyScalar(1 + 0.45 * pop2 * Math.abs(f.h[j])); s = (1 + (0.75 + 0.45 * Math.abs(f.h[j]) - 1) * e2) * (1 + 0.15 * pop2 * Math.abs(f.h[j])); }
      const isSel = sel && sel.layer === "hid" && sel.index === j;
      this.nHid.set(j, isSel ? 1.4 : s, dim(c, "hid", j), isSel ? HOT : RIM);
    }
    let lmax = -Infinity, lmean = 0;
    if (f) { for (const l of f.logits) { lmax = Math.max(lmax, l); lmean += l / this.V; } }
    for (let k = 0; k < this.V; k++) {
      c.copy(IDLE);
      let s = 1;
      if (f && e3 > 0) {
        const lz = clamp01((f.logits[k] - lmean) / Math.max(lmax - lmean, 1e-6));
        c2.copy(IDLE).lerp(HOT, Math.pow(lz, 1.4) * 0.65);
        if (e4 > 0) {
          const pc = new THREE.Color().copy(IDLE).lerp(HOT, Math.min(1, Math.sqrt(f.p[k] / 0.5)));
          c2.lerp(pc, e4);
          s = 1 + (0.9 + 1.4 * Math.sqrt(f.p[k]) - 1) * e4;
          if (f.p[k] > 0.05) { c2.multiplyScalar(1 + 0.35 * pop4); s *= 1 + 0.15 * pop4; }
        }
        c.lerp(c2, e3);
      }
      const isSel = sel && sel.layer === "out" && sel.index === k;
      this.nOut.set(k, isSel ? Math.max(1.5, s) : s, dim(c, "out", k), isSel ? HOT : this.outRim[k]);
    }
    for (const n of [this.nIn, this.nHid, this.nOut]) n.commit();

    this.updateLabelsOnly(t, e4, focusing, tk);
  }

  updateLabelsOnly(t, e4, focusing, tk) {
    const f = this.f;
    // labels
    const top = this.top();
    this.slotLabels.forEach((l) => (l.show = !!f && tk >= 1));
    this.textLabel.show = !!f;
    this.rankLabels.forEach((l, r) => {
      l.show = !!f && e4 > 0.5 && r < top.length && !(this.stage.narrow && r > 2);
      if (l.show) l.pos.copy(this.outPos[top[r]]).add(new THREE.Vector3(0, 0.3, 0));
    });
    const ro = this.stage.readout;
    if (ro) ro.hidden = !(f && e4 > 0.3);
    // halo on the winner
    this.halo.visible = !!f && e4 > 0.4 && top.length > 0 && !focusing;
    if (this.halo.visible) {
      this.halo.position.copy(this.outPos[top[0]]);
      this.halo.quaternion.copy(this.stage.camera.quaternion);
      const ph = REDUCED ? 0 : (t * 1.4) % 1;
      this.halo.scale.setScalar(1.05 * (1 + ph * 0.6));
      this.halo.material.opacity = (REDUCED ? 0.5 : 0.55 * (1 - ph)) * e4;
      this.halo.material.color.copy(HOT);
    }
  }

  // the tokenizer at work: the text sits as one piece, cracks into tokens, and each token drops into its row
  updateTiles(t, tk, focusing) {
    this.intake.visible = !!this.f;
    const U = this.U, v = new THREE.Vector3();
    for (const tile of this.tiles) {
      const split = ease(clamp01(tk / 0.35)), fly = ease(clamp01((tk - 0.35) / 0.65));
      const gap = 0.04 + 0.42 * split;
      const off = tile.joined + (tile.n - 1) * gap;
      v.copy(this.sentenceAt).addScaledVector(U, off);
      v.y += Math.sin(Math.PI * fly) * 0.35;
      tile.mesh.position.copy(v).lerp(tile.home, fly);
      // face the camera's side of the layers
      tile.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), U);
      const glow = REDUCED ? 0 : Math.sin(Math.PI * clamp01(tk / 0.35)) * 0.6;
      tile.mesh.material.emissiveIntensity = 0.25 + glow * 0.5;
      tile.mesh.material.opacity = focusing ? 0.35 : 1;
      tile.face.material.opacity = focusing ? 0.35 : 1;
    }
  }

  updateSparks(t, g1, g2, focusing) {
    const pos = this.stPos, col = this.stCol;
    const done = (n) => {
      this.streakGeo.instanceCount = n;
      this.streakGeo.attributes.instanceStart.data.needsUpdate = true;
      this.streakGeo.attributes.instanceColorStart.data.needsUpdate = true;
    };
    if (REDUCED || !this.f || focusing) { this.sparks.length = 0; return done(0); }
    // training: every 1.2 s, the error from the latest step flows backward, output → hidden first, then hidden → input
    if (this.training && t > this.nextWave && this.gradFlow.length) {
      this.nextWave = t + 1.2;
      for (const fl of this.gradFlow) {
        const delay = (fl.layer === 2 ? 0 : 0.5) + Math.random() * 0.15;
        this.sparks.push({ fl, t0: t + delay, dur: 0.6 });
      }
      if (this.sparks.length > this.SPARK_MAX) this.sparks.splice(0, this.sparks.length - this.SPARK_MAX);
    }
    let n = 0;
    // one slim streak: a bright head fading to nothing at the tail
    const put = (head, len, fade, fl) => {
      if (n >= this.SPARK_MAX) return;
      const tail = Math.max(0, head - len), o = n * 6;
      pos[o] = fl.a.x + (fl.b.x - fl.a.x) * tail; pos[o + 1] = fl.a.y + (fl.b.y - fl.a.y) * tail; pos[o + 2] = fl.a.z + (fl.b.z - fl.a.z) * tail;
      pos[o + 3] = fl.a.x + (fl.b.x - fl.a.x) * head; pos[o + 4] = fl.a.y + (fl.b.y - fl.a.y) * head; pos[o + 5] = fl.a.z + (fl.b.z - fl.a.z) * head;
      const base = fl.grad ? GRAD : fl.v >= 0 ? POS : NEG, br = (0.35 + 0.75 * fl.rel) * fade, wh = 0.12 * fade * fl.rel;
      col[o] = col[o + 1] = col[o + 2] = 0;
      col[o + 3] = base.r * br + wh; col[o + 4] = base.g * br + wh; col[o + 5] = base.b * br + wh;
      n++;
    };
    // while a layer is being revealed, a streak rides the tip of each growing line
    if (!this.training) {
      if (g1 > 0.02 && g1 < 0.98) for (const fl of this.flow) if (fl.layer === 1) put(ease(g1), 0.16, 0.9, fl);
      if (g2 > 0.02 && g2 < 0.98) for (const fl of this.flow) if (fl.layer === 2) put(ease(g2), 0.16, 0.9, fl);
    }
    this.sparks = this.sparks.filter((sp) => t < sp.t0 + sp.dur);
    for (const sp of this.sparks) {
      if (t < sp.t0) continue;
      const k = clamp01((t - sp.t0) / sp.dur), head = ease(k);
      put(head, 0.18, Math.min(1, k * 6) * Math.min(1, (1 - k) * 5), sp.fl);
    }
    done(n);
  }


}

function makeTokenTile(label, id, special) {
  const w = Math.max(1.0, 0.32 * label.length + 0.5), h = 0.7;
  const hue = (id * 0.61803398875) % 1;
  const color = special ? new THREE.Color(0x2a3242) : new THREE.Color().setHSL(hue, 0.55, 0.16);
  const mesh = new THREE.Mesh(new RoundedBoxGeometry(w, h, 0.18, 3, 0.07), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.25, roughness: 0.6, metalness: 0.0, envMapIntensity: 0.25, transparent: true, fog: false }));
  const cv = document.createElement("canvas"), ppu = 200;
  cv.width = Math.ceil(w * ppu); cv.height = Math.ceil(h * ppu);
  const x = cv.getContext("2d");
  x.textAlign = "center"; x.textBaseline = "middle";
  x.fillStyle = "rgba(240,246,255,0.96)";
  x.font = `600 ${Math.round(ppu * 0.3)}px "IBM Plex Mono", ui-monospace, monospace`;
  x.fillText(label, cv.width / 2, cv.height * 0.4);
  x.fillStyle = "rgba(240,246,255,0.7)";
  x.font = `500 ${Math.round(ppu * 0.16)}px "IBM Plex Mono", ui-monospace, monospace`;
  x.fillText("ID " + id, cv.width / 2, cv.height * 0.78);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const face = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex, transparent: true, toneMapped: false, depthWrite: false, fog: false }));
  face.position.z = 0.095;
  const holder = new THREE.Group();
  face.userData.tileFace = true;
  mesh.add(face);
  holder.add(mesh);
  holder.material = mesh.material;
  holder.geometry = mesh.geometry;
  return { mesh: holder, body: mesh, face, w };
}

function topAbs(arr, k) {
  const idx = Array.from(arr.keys());
  idx.sort((a, b) => Math.abs(arr[b]) - Math.abs(arr[a]));
  return idx.slice(0, k);
}

// ───────────────────────────── Token tiles (tokenizer chapter) ─────────────────────────────
class TilesView {
  constructor(stage) {
    this.stage = stage;
    this.group = new THREE.Group();
    this.tiles = new Map();
    this.dying = [];
    this.bounds = 6;
    this.flash = null;
  }
  makeTile(tok) {
    const label = showTok(tok.text);
    const w = Math.max(0.95, 0.36 * label.length + 0.5);
    const hue = (tok.id * 0.61803398875) % 1;
    const color = new THREE.Color().setHSL(hue, 0.62, 0.22);
    const geo = new RoundedBoxGeometry(w, 1.1, 0.42, 4, 0.14);
    const mat = new THREE.MeshPhysicalMaterial({ color, roughness: 0.35, metalness: 0.1, clearcoat: 0.8, clearcoatRoughness: 0.12, envMapIntensity: 0.55 });
    const mesh = new THREE.Mesh(geo, mat);
    const cv = document.createElement("canvas");
    const ppu = 160;
    cv.width = Math.ceil(w * ppu); cv.height = Math.ceil(1.1 * ppu);
    const x = cv.getContext("2d");
    x.fillStyle = "rgba(235,242,255,0.92)";
    x.textAlign = "center"; x.textBaseline = "middle";
    x.font = `600 ${Math.round(ppu * 0.36)}px "IBM Plex Mono", ui-monospace, monospace`;
    x.fillText(label, cv.width / 2, cv.height * 0.42);
    x.fillStyle = "rgba(235,242,255,0.6)";
    x.font = `500 ${Math.round(ppu * 0.17)}px "IBM Plex Mono", ui-monospace, monospace`;
    x.fillText("#" + tok.id, cv.width / 2, cv.height * 0.8);
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    const face = new THREE.Mesh(new THREE.PlaneGeometry(w, 1.1), new THREE.MeshBasicMaterial({ map: tex, transparent: true, toneMapped: false, depthWrite: false }));
    face.position.z = 0.215;
    mesh.add(face);
    mesh.userData.tok = tok;
    const tile = { mesh, w, tok, target: new THREE.Vector3(), scale: 0.01, targetScale: 1, phase: Math.random() * 6, glow: 0 };
    this.group.add(mesh);
    return tile;
  }
  set(tokens, justMerged = null) {
    const gap = 0.2, maxW = 12.5;
    const widths = tokens.map((t) => Math.max(0.95, 0.36 * showTok(t.text).length + 0.5));
    const rows = [[]];
    let rw = 0;
    tokens.forEach((t, i) => {
      if (rw + widths[i] > maxW && rows[rows.length - 1].length) { rows.push([]); rw = 0; }
      rows[rows.length - 1].push(i);
      rw += widths[i] + gap;
    });
    const placed = new Map();
    rows.forEach((r, ri) => {
      const total = r.reduce((s, i) => s + widths[i], 0) + gap * (r.length - 1);
      let x = -total / 2;
      for (const i of r) { placed.set(i, new THREE.Vector3(x + widths[i] / 2, 0.6 + ((rows.length - 1) / 2 - ri) * 1.55, 0)); x += widths[i] + gap; }
    });
    this.bounds = Math.max(4, Math.min(maxW, Math.max(...rows.map((r) => r.reduce((s, i) => s + widths[i] + gap, 0)))) / 2 + 1, rows.length * 0.9);
    const next = new Map();
    tokens.forEach((t, i) => {
      const key = t.start + ":" + t.end + ":" + t.id;
      let tile = this.tiles.get(key);
      if (!tile) {
        tile = this.makeTile(t);
        const parts = [...this.tiles.values()].filter((o) => o.tok.start >= t.start && o.tok.end <= t.end);
        if (parts.length) tile.mesh.position.copy(parts.reduce((s, o) => s.add(o.mesh.position), new THREE.Vector3()).multiplyScalar(1 / parts.length));
        else tile.mesh.position.copy(placed.get(i)).add(new THREE.Vector3(0, 2.5, 0));
        if (justMerged !== null && t.id === justMerged) tile.glow = 1;
      }
      tile.target.copy(placed.get(i));
      tile.targetScale = 1;
      next.set(key, tile);
    });
    for (const [key, tile] of this.tiles) {
      if (next.has(key)) continue;
      const host = [...next.values()].find((n) => n.tok.start <= tile.tok.start && n.tok.end >= tile.tok.end);
      tile.target.copy(host ? host.target : tile.mesh.position);
      tile.targetScale = 0;
      this.dying.push(tile);
    }
    this.tiles = next;
  }
  pickAt(ray) {
    const hit = ray.intersectObjects([...this.tiles.values()].map((t) => t.mesh), false)[0];
    return hit ? hit.object.userData.tok : null;
  }
  update(t, dt) {
    const k = 1 - Math.pow(0.0015, dt);
    const step = (tile) => {
      tile.mesh.position.lerp(tile.target, k);
      tile.scale += (tile.targetScale - tile.scale) * k;
      const s = Math.max(tile.scale, 0.001) * (1 + tile.glow * 0.12);
      tile.mesh.scale.set(s, s, s);
      tile.mesh.material.emissive?.setScalar(tile.glow * 0.35);
      tile.glow = Math.max(0, tile.glow - dt * 1.2);
      if (!REDUCED) tile.mesh.rotation.x = Math.sin(t * 0.8 + tile.phase) * 0.05;
    };
    for (const tile of this.tiles.values()) step(tile);
    this.dying = this.dying.filter((tile) => {
      step(tile);
      if (tile.scale < 0.02) {
        this.group.remove(tile.mesh);
        tile.mesh.geometry.dispose();
        tile.mesh.children[0].material.map.dispose();
        return false;
      }
      return true;
    });
  }
}

// ───────────────────────────── Embedding map ─────────────────────────────
class EmbedView {
  constructor(stage, ids, vocab, categoryOf, catHex) {
    Object.assign(this, { stage, ids, vocab });
    this.group = new THREE.Group();
    const geo = new THREE.SphereGeometry(0.16, 28, 18);
    this.nodes = ids.map((id) => {
      const cat = categoryOf(vocab[id]);
      const col = catHex[cat];
      const mat = new THREE.MeshPhysicalMaterial({ color: col, roughness: 0.2, metalness: 0.05, clearcoat: 1, clearcoatRoughness: 0.08, emissive: col, emissiveIntensity: 0.35 });
      const mesh = new THREE.Mesh(geo, mat);
      this.group.add(mesh);
      const label = stage.labels.add("embed", esc(vocab[id].trim() || "␣"), "emb", new THREE.Vector3());
      label.el.style.setProperty("--c", "#" + new THREE.Color(col).getHexString());
      return { id, mesh, label, target: new THREE.Vector3(), cat };
    });
    const lg = new THREE.BufferGeometry();
    lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(ids.length * 6), 3));
    this.drops = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x3a4a66, transparent: true, opacity: 0.4 }));
    this.drops.frustumCulled = false;
    this.group.add(this.drops);
    const ng = new THREE.BufferGeometry();
    ng.setAttribute("position", new THREE.BufferAttribute(new Float32Array(5 * 6), 3));
    this.nbLines = new THREE.LineSegments(ng, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.75, toneMapped: false }));
    this.nbLines.frustumCulled = false;
    this.group.add(this.nbLines);
    this.ring = new THREE.Mesh(new THREE.TorusGeometry(0.32, 0.022, 12, 64), new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false }));
    this.group.add(this.ring);
    this.selected = null;
    this.neighbors = [];
    this.hovered = null;
  }
  setPoints(pts) {
    // scale so most points fill the view; a few outliers are clamped to the edge
    const r = pts.map((p) => Math.hypot(p[0], p[1], p[2])).sort((a, b) => a - b);
    const s = 3.6 / (r[Math.floor(r.length * 0.9)] || 1e-6);
    const c = (v) => Math.max(-5.2, Math.min(5.2, v * s));
    pts.forEach((p, i) => this.nodes[i].target.set(c(p[0]), c(p[1]) * 0.7 + 0.4, c(p[2])));
  }
  select(id, neighbors = []) { this.selected = id; this.neighbors = neighbors; }
  pickAt(px, py, w, h, touch) {
    const v = new THREE.Vector3();
    let best = null, bd = touch ? 26 : 16;
    for (const n of this.nodes) {
      v.copy(n.mesh.position).project(this.stage.camera);
      const d = Math.hypot((v.x * 0.5 + 0.5) * w - px, (-v.y * 0.5 + 0.5) * h - py);
      if (d < bd) { bd = d; best = n.id; }
    }
    return best;
  }
  update(t, dt) {
    const k = 1 - Math.pow(0.02, dt);
    const pa = this.drops.geometry.attributes.position.array;
    const nb = new Set(this.neighbors);
    this.nodes.forEach((n, i) => {
      n.mesh.position.lerp(n.target, k);
      n.label.pos.copy(n.mesh.position).add(new THREE.Vector3(0, 0.34, 0));
      const isSel = n.id === this.selected;
      n.label.show = isSel || nb.has(n.id) || n.id === this.hovered || this.stage.allLabels;
      n.label.el.classList.toggle("sel", isSel);
      n.mesh.scale.setScalar(isSel ? 1.5 : nb.has(n.id) ? 1.2 : 1);
      pa.set([n.mesh.position.x, n.mesh.position.y, n.mesh.position.z, n.mesh.position.x, FLOOR_Y, n.mesh.position.z], i * 6);
    });
    this.drops.geometry.attributes.position.needsUpdate = true;
    const sel = this.nodes.find((n) => n.id === this.selected);
    this.ring.visible = !!sel;
    const la = this.nbLines.geometry.attributes.position.array;
    la.fill(0);
    if (sel) {
      this.ring.position.copy(sel.mesh.position);
      this.ring.quaternion.copy(this.stage.camera.quaternion);
      this.ring.scale.setScalar(1 + (REDUCED ? 0 : 0.08 * Math.sin(t * 4)));
      this.neighbors.slice(0, 5).forEach((id, i) => {
        const o = this.nodes.find((n) => n.id === id);
        if (o) la.set([sel.mesh.position.x, sel.mesh.position.y, sel.mesh.position.z, o.mesh.position.x, o.mesh.position.y, o.mesh.position.z], i * 6);
      });
    }
    this.nbLines.geometry.attributes.position.needsUpdate = true;
  }
}

// ───────────────────────────── Single neuron ─────────────────────────────
class NeuronView {
  constructor(stage) {
    this.stage = stage;
    this.group = new THREE.Group();
    this.inPos = [new THREE.Vector3(-4.2, 2.1, 0), new THREE.Vector3(-4.2, 0.2, 0), new THREE.Vector3(-4.2, -1.7, 0)];
    this.nPos = new THREE.Vector3(0.6, 0.2, 0);
    this.outPos = new THREE.Vector3(4.6, 0.2, 0);
    this.bPos = new THREE.Vector3(0.6, -2.4, 0);
    const glass = () => new THREE.MeshPhysicalMaterial({ color: 0xaac4ee, roughness: 0.05, clearcoat: 1, transparent: true, opacity: 0.18, depthWrite: false, envMapIntensity: 1.2 });
    const core = () => new THREE.MeshBasicMaterial({ color: 0x223, toneMapped: false });
    const orb = (p, r) => {
      const g = new THREE.Group();
      const s = new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32), glass());
      const c = new THREE.Mesh(new THREE.SphereGeometry(r * 0.78, 32, 20), core());
      s.renderOrder = 2;
      g.add(c, s); g.position.copy(p);
      this.group.add(g);
      return c;
    };
    this.inCores = this.inPos.map((p) => orb(p, 0.45));
    this.nCore = orb(this.nPos, 0.85);
    this.bCore = orb(this.bPos, 0.3);
    this.outCore = orb(this.outPos, 0.5);
    this.links = [...this.inPos, this.bPos].map((p) => this.tube(p, this.nPos));
    this.outLink = this.tube(this.nPos, this.outPos);
    const L = stage.labels;
    this.inLabels = this.inPos.map((p) => L.add("neuron", "", "nlbl", p.clone().add(new THREE.Vector3(0, 0.8, 0))));
    this.wLabels = this.inPos.map((p) => L.add("neuron", "", "wlbl", p.clone().lerp(this.nPos, 0.42).add(new THREE.Vector3(0, -0.34, 0))));
    this.bLabel = L.add("neuron", "", "wlbl", this.bPos.clone().add(new THREE.Vector3(0, -0.6, 0)));
    this.nLabel = L.add("neuron", "", "nlbl big", this.nPos.clone().add(new THREE.Vector3(0, 1.5, 0)));
    this.oLabel = L.add("neuron", "", "nlbl", this.outPos.clone().add(new THREE.Vector3(0, 0.9, 0)));
    this.state = null;
    this.lastEmit = 0;
    this.sparks = [];
    const sg = new THREE.BufferGeometry();
    this.sp = new Float32Array(120 * 3); this.sc = new Float32Array(120 * 3);
    sg.setAttribute("position", new THREE.BufferAttribute(this.sp, 3));
    sg.setAttribute("color", new THREE.BufferAttribute(this.sc, 3));
    this.points = new THREE.Points(sg, new THREE.PointsMaterial({ size: 0.28, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false, map: dotTexture() }));
    this.points.frustumCulled = false;
    this.group.add(this.points);
  }
  tube(a, b) {
    const len = b.clone().sub(a).length();
    const geo = new THREE.CylinderGeometry(1, 1, len, 20, 1, true);
    geo.translate(0, len / 2, 0);
    geo.rotateX(Math.PI / 2);
    const mat = new THREE.MeshPhysicalMaterial({ color: 0x4cc9f0, emissive: 0x4cc9f0, emissiveIntensity: 0.6, roughness: 0.3, transparent: true, opacity: 0.9 });
    const m = new THREE.Mesh(geo, mat);
    m.position.copy(a);
    m.lookAt(b);
    this.group.add(m);
    return m;
  }
  set(s) {
    this.state = s;
    const c = new THREE.Color();
    s.x.forEach((x, i) => {
      this.inCores[i].material.color.copy(valColor(x, 1, c));
      const w = s.w[i], r = 0.025 + 0.07 * Math.min(Math.abs(w), 2.5);
      this.links[i].scale.set(r, r, 1);
      const lc = w >= 0 ? POS : NEG;
      this.links[i].material.color.copy(lc); this.links[i].material.emissive.copy(lc);
      this.stage.labels.set(this.inLabels[i], `${s.names[i]} <b>${s.x[i].toFixed(2)}</b>`);
      this.stage.labels.set(this.wLabels[i], `w${"₁₂₃"[i]} = ${num(w)}`);
    });
    const rb = 0.025 + 0.07 * Math.min(Math.abs(s.b), 2.5);
    this.links[3].scale.set(rb, rb, 1);
    const bc = s.b >= 0 ? POS : NEG;
    this.links[3].material.color.copy(bc); this.links[3].material.emissive.copy(bc);
    this.bCore.material.color.copy(valColor(s.b, 2, c));
    this.stage.labels.set(this.bLabel, `bias b = ${num(s.b)}`);
    this.nCore.material.color.copy(valColor(s.a, 1, c));
    this.outCore.material.color.copy(valColor(s.a, 1, c));
    this.outLink.material.color.copy(s.a >= 0 ? POS : NEG); this.outLink.material.emissive.copy(s.a >= 0 ? POS : NEG);
    this.outLink.scale.set(0.03 + 0.06 * Math.abs(s.a), 0.03 + 0.06 * Math.abs(s.a), 1);
    this.stage.labels.set(this.nLabel, `z = ${num(s.z)} → tanh(z) = <b>${num(s.a)}</b>`);
    this.stage.labels.set(this.oLabel, s.verdict);
  }
  update(t, dt) {
    const s = this.state;
    if (s && !REDUCED && t - this.lastEmit > 0.18) {
      this.lastEmit = t;
      const c = new THREE.Color();
      s.x.forEach((x, i) => { const v = x * s.w[i]; if (Math.abs(v) > 0.03) this.sparks.push({ a: this.inPos[i], b: this.nPos, t0: t, d: 0.9, c: valColor(v, 1, c).clone() }); });
      if (Math.abs(s.a) > 0.05) this.sparks.push({ a: this.nPos, b: this.outPos, t0: t, d: 0.8, c: valColor(s.a, 1, c).clone() });
    }
    this.sparks = this.sparks.filter((p) => t < p.t0 + p.d).slice(-120);
    this.sc.fill(0);
    this.sparks.forEach((p, i) => {
      const k = ease(clamp01((t - p.t0) / p.d));
      this.sp[i * 3] = p.a.x + (p.b.x - p.a.x) * k; this.sp[i * 3 + 1] = p.a.y + (p.b.y - p.a.y) * k; this.sp[i * 3 + 2] = p.a.z + (p.b.z - p.a.z) * k;
      const f = Math.sin(Math.PI * k) * 1.6;
      this.sc[i * 3] = p.c.r * f; this.sc[i * 3 + 1] = p.c.g * f; this.sc[i * 3 + 2] = p.c.b * f;
    });
    this.points.geometry.attributes.position.needsUpdate = true;
    this.points.geometry.attributes.color.needsUpdate = true;
  }
}

function dotTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const x = c.getContext("2d");
  const g = x.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.3, "rgba(255,255,255,0.5)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  x.fillStyle = g;
  x.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

// ───────────────────────────── Loss landscape (two-weight toy) ─────────────────────────────
class LandscapeView {
  constructor(stage) {
    this.stage = stage;
    this.group = new THREE.Group();
    this.range = 4; this.size = 10; this.hScale = 1.15; this.base = -3.2;
    const seg = 140;
    const geo = new THREE.PlaneGeometry(this.size, this.size, seg, seg);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position, colors = new Float32Array(pos.count * 3);
    const lo = new THREE.Color(0x0b6e8a), mid = new THREE.Color(0x5b4bd6), hi = new THREE.Color(0xff8a4c), c = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const [w, b] = this.toWB(pos.getX(i), pos.getZ(i));
      const L = toyLoss(w, b);
      pos.setY(i, this.base + L * this.hScale);
      const k = Math.min(L / 2.9, 1);
      if (k < 0.5) c.copy(lo).lerp(mid, k * 2); else c.copy(mid).lerp(hi, (k - 0.5) * 2);
      colors.set([c.r, c.g, c.b], i * 3);
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    this.surface = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.42, metalness: 0.15, side: THREE.DoubleSide, envMapIntensity: 0.9 }));
    this.group.add(this.surface);
    const lines = [];
    const step = this.size / 20;
    for (let a = -this.size / 2; a <= this.size / 2 + 1e-6; a += step) {
      for (let bb = -this.size / 2; bb < this.size / 2 - 1e-6; bb += step / 4) {
        for (const [x1, z1, x2, z2] of [[a, bb, a, bb + step / 4], [bb, a, bb + step / 4, a]]) lines.push(x1, this.heightAt(x1, z1) + 0.012, z1, x2, this.heightAt(x2, z2) + 0.012, z2);
      }
    }
    const lg = new THREE.BufferGeometry();
    lg.setAttribute("position", new THREE.Float32BufferAttribute(lines, 3));
    this.group.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.12 })));
    this.ball = new THREE.Mesh(new THREE.SphereGeometry(0.2, 48, 32), new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.08, envMapIntensity: 2.2 }));
    this.group.add(this.ball);
    this.trailMax = 500;
    const tg = new THREE.BufferGeometry();
    tg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.trailMax * 3), 3));
    this.trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: 0xfff1a8, toneMapped: false }));
    this.trail.frustumCulled = false;
    this.group.add(this.trail);
    const star = new THREE.Mesh(new THREE.OctahedronGeometry(0.12), new THREE.MeshBasicMaterial({ color: 0x7ee787, toneMapped: false }));
    const [sx, sz] = this.toXZ(0.94, -0.14);
    star.position.set(sx, this.heightAt(sx, sz) + 0.05, sz);
    this.group.add(star);
    const L = stage.labels;
    L.add("land", "lowest point", "wlbl", star.position.clone().add(new THREE.Vector3(0, 0.45, 0)));
    L.add("land", "weight w →", "axis", new THREE.Vector3(0, this.base - 0.15, this.size / 2 + 0.7));
    L.add("land", "bias b →", "axis", new THREE.Vector3(this.size / 2 + 0.9, this.base - 0.15, 0));
    L.add("land", "↑ loss", "axis", new THREE.Vector3(-this.size / 2 - 0.4, this.base + 2.2, -this.size / 2));
    this.ballLabel = L.add("land", "", "nlbl", new THREE.Vector3());
    this.target = new THREE.Vector3();
    this.setTrail([[-2.6, 2.8]]);
  }
  toXZ(w, b) { return [(w / this.range) * (this.size / 2), (-b / this.range) * (this.size / 2)]; }
  toWB(x, z) { return [(x / (this.size / 2)) * this.range, (-z / (this.size / 2)) * this.range]; }
  heightAt(x, z) { const [w, b] = this.toWB(x, z); return this.base + toyLoss(w, b) * this.hScale; }
  setTrail(path) {
    const a = this.trail.geometry.attributes.position.array;
    const n = Math.min(path.length, this.trailMax), start = path.length - n;
    for (let i = 0; i < n; i++) { const [w, b] = path[start + i]; const [x, z] = this.toXZ(w, b); a.set([x, this.heightAt(x, z) + 0.06, z], i * 3); }
    this.trail.geometry.setDrawRange(0, n);
    this.trail.geometry.attributes.position.needsUpdate = true;
    const [w, b] = path[path.length - 1];
    const [x, z] = this.toXZ(w, b);
    this.target.set(x, this.heightAt(x, z) + 0.2, z);
    this.stage.labels.set(this.ballLabel, `w ${num(w)} · b ${num(b)} · loss <b>${toyLoss(w, b).toFixed(3)}</b>`);
  }
  pick(ray) {
    const hit = ray.intersectObject(this.surface)[0];
    return hit ? this.toWB(hit.point.x, hit.point.z).map((v) => Math.max(-this.range, Math.min(this.range, v))) : null;
  }
  update(t, dt) {
    this.ball.position.lerp(this.target, 1 - Math.pow(0.0005, dt));
    this.ballLabel.pos.copy(this.ball.position).add(new THREE.Vector3(0, 0.55, 0));
  }
}

// ───────────────────────────── The stage ─────────────────────────────
export async function createStage(host, cfg) {
  const narrowMQ = matchMedia("(max-width: 900px)");
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  let dpr = Math.min(window.devicePixelRatio || 1, narrowMQ.matches ? 1.5 : 1.75);
  renderer.setPixelRatio(dpr);
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.domElement.tabIndex = 0;
  host.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = BG.clone();
  const fog = new THREE.FogExp2(0x070a10, 0.026);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;

  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 200);
  camera.position.set(8, 3, 16);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enableZoom = false;
  controls.enablePan = false;
  controls.rotateSpeed = 0.55;

  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(6, 10, 8);
  const rim = new THREE.DirectionalLight(0x6fa8ff, 1.1);
  rim.position.set(-8, 4, -10);
  scene.add(key, rim, new THREE.AmbientLight(0x30405a, 0.6));

  const floor = new THREE.Mesh(new THREE.CircleGeometry(40, 96), new THREE.MeshStandardMaterial({ color: 0x070a12, roughness: 0.6, metalness: 0.2, envMapIntensity: 0.1 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = FLOOR_Y;
  scene.add(floor);
  const grid = new THREE.PolarGridHelper(24, 24, 16, 96, 0x1a2a44, 0x111a2c);
  grid.position.y = FLOOR_Y + 0.005;
  grid.material.transparent = true; grid.material.opacity = 0.5;
  scene.add(grid);
  const dustN = 600, dpos = new Float32Array(dustN * 3);
  for (let i = 0; i < dustN; i++) dpos.set([(Math.random() - 0.5) * 50, FLOOR_Y + Math.random() * 16, (Math.random() - 0.5) * 50], i * 3);
  const dg = new THREE.BufferGeometry();
  dg.setAttribute("position", new THREE.BufferAttribute(dpos, 3));
  const dust = new THREE.Points(dg, new THREE.PointsMaterial({ color: 0x5f7ba8, size: 0.045, transparent: true, opacity: 0.45, depthWrite: false }));
  scene.add(dust);

  const samples = dpr < 1.5 ? 4 : 0;
  const rt = new THREE.WebGLRenderTarget(2, 2, { type: THREE.HalfFloatType, samples });
  const composer = new EffectComposer(renderer, rt);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.35, 0.3, 0.92);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  const stage = { renderer, scene, camera, controls, time: 0, mode: "net", narrow: narrowMQ.matches, quality: 0, allLabels: false, zoomK: 1 };
  stage.labels = new Labels(host);
  stage.readout = host.querySelector("#readout");

  const net = new NetworkView(stage, cfg);
  const tiles = new TilesView(stage);
  const embed = new EmbedView(stage, cfg.embedIds, cfg.vocab, cfg.categoryOf, cfg.catHex);
  const neuron = new NeuronView(stage);
  const land = new LandscapeView(stage);
  const views = { net, tiles, embed, neuron, land };
  for (const v of Object.values(views)) { v.group.visible = false; scene.add(v.group); }
  Object.assign(stage, views);

  // camera framings
  const FRAMES = {
    hero: { view: "net", dir: [0.58, 0.22, 1], fit: "net" },
    net: { view: "net", dir: [0.55, 0.22, 1], fit: "net" },
    tiles: { view: "tiles", dir: [0.12, 0.28, 1], r: 6, target: [0, 0.4, 0] },
    embed: { view: "embed", dir: [0.8, 0.45, 1], r: 6.2, target: [0, 0.2, 0], spin: true },
    neuron: { view: "neuron", dir: [0.06, 0.12, 1], r: 6.0, target: [0.2, 0, 0] },
    land: { view: "land", dir: [0.75, 0.8, 1], r: 6.6, target: [0, -2.5, 0], spin: true },
  };
  // where the network may sit, in CSS px kept free at each edge for the overlays
  const netPads = () => {
    const W = host.clientWidth;
    return stage.narrow ? { t: 58, b: 60, l: 72, r: 14 } : { t: 84, b: 84, l: 100, r: 28 };
  };
  let frame = FRAMES.hero, camFrom = null, camT = 1, userTouched = false, spin = true, layerZoom = null;
  const v3 = new THREE.Vector3();
  const goal = () => {
    const d = new THREE.Vector3(...frame.dir).normalize();
    if (frame.fit === "net") {
      const pts = layerZoom !== null ? [net.inZoom, net.hidPos, net.outPos][layerZoom] : net.allPos;
      const box = new THREE.Box3().setFromPoints(pts);
      const target = box.getCenter(new THREE.Vector3());
      const W = host.clientWidth || 1, Hh = host.clientHeight || 1, p = netPads();
      camera.setViewOffset(W, Hh, (p.r - p.l) / 2, (p.b - p.t) / 2, W, Hh);
      camera.updateProjectionMatrix();
      const x0 = -1 + (2 * p.l) / W, x1 = 1 - (2 * p.r) / W, y0 = -1 + (2 * p.b) / Hh, y1 = 1 - (2 * p.t) / Hh;
      // binary search for the closest distance where every node fits the free area
      let lo = 2, hi = 90;
      const cam = camera.clone();
      for (let it = 0; it < 26; it++) {
        const dist = (lo + hi) / 2;
        cam.position.copy(target).addScaledVector(d, dist);
        cam.lookAt(target);
        cam.updateMatrixWorld();
        let ok = true;
        for (const q of pts) { v3.copy(q).project(cam); if (v3.x < x0 || v3.x > x1 || v3.y < y0 || v3.y > y1) { ok = false; break; } }
        ok ? (hi = dist) : (lo = dist);
      }
      return { pos: target.clone().addScaledVector(d, hi * stage.zoomK), target };
    }
    camera.clearViewOffset();
    camera.updateProjectionMatrix();
    const v = THREE.MathUtils.degToRad(camera.fov) / 2, hh = Math.atan(Math.tan(v) * camera.aspect);
    const r = frame.view === "tiles" ? Math.max(frame.r, tiles.bounds) : frame.r;
    const target = new THREE.Vector3(...frame.target);
    return { pos: target.clone().addScaledVector(d, (r / Math.sin(Math.min(v, hh))) * stage.zoomK), target };
  };
  let goalCache = null;
  const tweenTo = () => { goalCache = goal(); camFrom = { pos: camera.position.clone(), target: controls.target.clone() }; camT = REDUCED ? 1 : 0; if (REDUCED) { camera.position.copy(goalCache.pos); controls.target.copy(goalCache.target); } };
  controls.addEventListener("start", () => { userTouched = true; camT = 1; stage.onUserMove?.(); });

  const applyLimits = () => {
    if (frame.view === "net") {
      const d = new THREE.Vector3(...frame.dir).normalize();
      const polar = Math.acos(d.y), az = Math.atan2(d.x, d.z);
      controls.minPolarAngle = stage.narrow ? polar : 0.95; controls.maxPolarAngle = stage.narrow ? polar : 1.5;
      controls.minAzimuthAngle = az - 0.95; controls.maxAzimuthAngle = az + 0.95;
    } else {
      controls.minPolarAngle = 0.15; controls.maxPolarAngle = frame.view === "land" ? 1.45 : 1.62;
      controls.minAzimuthAngle = -Infinity; controls.maxAzimuthAngle = Infinity;
    }
  };

  stage.setMode = (name) => {
    const f = FRAMES[name];
    if (!f) return;
    const changed = f !== frame;
    frame = f;
    stage.mode = f.view;
    layerZoom = null;
    stage.zoomK = 1;
    for (const [k, v] of Object.entries(views)) v.group.visible = k === f.view;
    const isNet = f.view === "net";
    renderer.toneMapping = isNet ? THREE.NoToneMapping : THREE.ACESFilmicToneMapping;
    scene.fog = isNet ? null : fog;
    dust.visible = !isNet;
    floor.visible = false;
    bloom.strength = isNet ? 0.5 : 0.65; bloom.radius = isNet ? 0.35 : 0.5; bloom.threshold = isNet ? 0.85 : 0.86;
    if (changed) { userTouched = false; net.focus(null); stage.onFocus?.(null); if (tip) tip.hidden = true; }
    if (stage.readout && !isNet) stage.readout.hidden = true;
    spin = !!(f.spin || f.sway) && !REDUCED;
    controls.autoRotate = !!f.spin && spin;
    controls.autoRotateSpeed = 0.35;
    applyLimits();
    tweenTo();
    host.dataset.mode = name;
  };
  stage.resetView = () => { userTouched = false; layerZoom = null; stage.zoomK = 1; controls.autoRotate = !!frame.spin && spin; tweenTo(); };
  stage.zoom = (dir) => { stage.zoomK = Math.max(0.55, Math.min(1.6, stage.zoomK * (dir > 0 ? 0.8 : 1.25))); userTouched = false; tweenTo(); };
  stage.zoomLayer = (i) => { if (frame.view !== "net") return; layerZoom = layerZoom === i ? null : i; stage.zoomK = 1; userTouched = false; tweenTo(); stage.onZoomLayer?.(layerZoom); };
  stage.setSpin = (on) => { spin = on && !REDUCED; controls.autoRotate = !!frame.spin && spin; if (on) userTouched = false; };
  stage.getSpin = () => spin;
  stage.setLabels = (on) => { stage.labels.layer.hidden = !on; };
  stage.setAllLabels = (on) => { stage.allLabels = on; };

  // ── pointer: hover and click on whatever is under the pointer
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  const tip = host.querySelector("#stageTip");
  let down = null, hoverRaf = 0;
  const local = (e) => { const r = renderer.domElement.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top, r.width, r.height]; };
  const placeTip = (sel) => {
    if (!tip) return;
    if (!sel) { tip.hidden = true; return; }
    tip.innerHTML = net.tipHtml(sel) + (net.pinned ? `<button type="button" class="tip-x" aria-label="Close">×</button>` : `<div class="tip-f">${COARSE ? "Tap" : "Click"} to pin</div>`);
    tip.hidden = false;
    const W = host.clientWidth, Hh = host.clientHeight;
    v3.copy(sel.layer === "in" ? net.inPos[sel.index] : sel.layer === "hid" ? net.hidPos[sel.index] : net.outPos[sel.index]).project(camera);
    const x = (v3.x * 0.5 + 0.5) * W, y = (-v3.y * 0.5 + 0.5) * Hh;
    if (stage.narrow) { tip.style.transform = ""; return; }
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let tx = x + 20, ty = y - th / 2;
    if (tx + tw > W - 10) tx = x - 20 - tw;
    ty = Math.max(64, Math.min(Hh - th - 10, ty));
    tip.style.transform = `translate(${tx.toFixed(0)}px,${ty.toFixed(0)}px)`;
  };
  stage.placeTip = placeTip;
  renderer.domElement.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
  renderer.domElement.addEventListener("pointermove", (e) => {
    if (e.pointerType !== "mouse" || (down && e.buttons)) return;
    cancelAnimationFrame(hoverRaf);
    hoverRaf = requestAnimationFrame(() => {
      const [x, y, w, h] = local(e);
      let hit = false;
      if (stage.mode === "net" && net.model) {
        const sel = net.pickAt(x, y, w, h, false);
        hit = !!sel;
        if (!sel && net.tiles.length) { ndc.set((x / w) * 2 - 1, -(y / h) * 2 + 1); ray.setFromCamera(ndc, camera); hit = !!ray.intersectObjects(net.tiles.map((t) => t.body), true)[0]; }
        if (!net.pinned) { net.focus(sel); placeTip(sel); stage.onFocus?.(sel); }
      } else if (stage.mode === "embed") {
        const id = embed.pickAt(x, y, w, h, false);
        embed.hovered = id; hit = id !== null;
      } else if (stage.mode === "tiles") {
        ndc.set((x / w) * 2 - 1, -(y / h) * 2 + 1); ray.setFromCamera(ndc, camera);
        hit = !!tiles.pickAt(ray);
      } else if (stage.mode === "land") hit = true;
      renderer.domElement.style.cursor = hit ? "pointer" : "";
    });
  });
  renderer.domElement.addEventListener("pointerleave", () => {
    if (stage.mode === "net" && !net.pinned) { net.focus(null); placeTip(null); stage.onFocus?.(null); }
    embed.hovered = null;
  });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6) { down = null; return; }
    down = null;
    const [x, y, w, h] = local(e);
    const touch = e.pointerType !== "mouse";
    ndc.set((x / w) * 2 - 1, -(y / h) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    if (stage.mode === "land") { const wb = land.pick(ray); if (wb) stage.onLandPick?.(wb); }
    else if (stage.mode === "net" && net.model) {
      const th = net.tiles.length ? ray.intersectObjects(net.tiles.map((t) => t.body), true)[0] : null;
      if (th) { let o = th.object; while (o && o.userData.slot === undefined) o = o.parent; if (o) { net.onSlot?.(o.userData.slot); return; } }
      const sel = net.pickAt(x, y, w, h, touch);
      if (sel) { net.focus(sel, true); placeTip(sel); stage.onFocus?.(sel); stage.onPick?.(sel); }
      else if (net.pinned || net.focusSel) { net.focus(null); placeTip(null); stage.onFocus?.(null); }
    } else if (stage.mode === "embed") {
      const id = embed.pickAt(x, y, w, h, touch);
      if (id !== null) stage.onEmbedPick?.(id);
    } else if (stage.mode === "tiles") {
      const tok = tiles.pickAt(ray);
      stage.onTilePick?.(tok, x, y);
    }
  });
  tip?.addEventListener("click", (e) => { if (e.target.closest(".tip-x")) { net.focus(null); placeTip(null); stage.onFocus?.(null); } });

  // keyboard: arrows orbit, Esc clears a pinned neuron
  const sph = new THREE.Spherical();
  renderer.domElement.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { net.focus(null); placeTip(null); stage.onFocus?.(null); return; }
    const k = { ArrowLeft: [-0.21, 0], ArrowRight: [0.21, 0], ArrowUp: [0, -0.14], ArrowDown: [0, 0.14] }[e.key];
    if (!k) return;
    e.preventDefault();
    userTouched = true; controls.autoRotate = false; camT = 1;
    sph.setFromVector3(camera.position.clone().sub(controls.target));
    sph.theta = Math.max(controls.minAzimuthAngle, Math.min(controls.maxAzimuthAngle, sph.theta + k[0]));
    sph.phi = Math.max(Math.max(controls.minPolarAngle, 0.15), Math.min(Math.min(controls.maxPolarAngle, 1.5), sph.phi + k[1]));
    camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(sph));
  });

  let W = 1, Hh = 1;
  const resize = () => {
    W = host.clientWidth || 1; Hh = host.clientHeight || 1;
    stage.narrow = W < 560 || narrowMQ.matches;
    renderer.setSize(W, Hh, false);
    renderer.domElement.style.width = W + "px"; renderer.domElement.style.height = Hh + "px";
    composer.setPixelRatio(dpr);
    composer.setSize(W, Hh);
    bloom.resolution.set(W / 2, Hh / 2);
    camera.aspect = W / Hh;
    camera.updateProjectionMatrix();
    for (const b of [...net.B, net.focusLines]) b.mat.resolution.set(W, Hh);
    net.streakMat.resolution.set(W, Hh);
    applyLimits();
    if (!userTouched) { goalCache = goal(); if (camT >= 1) { camera.position.copy(goalCache.pos); controls.target.copy(goalCache.target); } }
    net.updateReadout();
  };
  new ResizeObserver(resize).observe(host);
  resize();
  stage.setMode("hero");
  camera.position.copy(goalCache.pos); controls.target.copy(goalCache.target); camT = 1;

  // ── frame loop, with a simple quality governor for slow devices
  const clock = new THREE.Clock();
  let visible = true, ema = 16, slowFor = 0;
  new IntersectionObserver((es) => (visible = es[0].isIntersecting)).observe(host);
  stage.hooks = [];
  if ((navigator.hardwareConcurrency || 8) <= 4) stage.quality = 1;
  const loop = () => {
    requestAnimationFrame(loop);
    const raw = clock.getDelta();
    if (!visible || document.hidden) return;
    const dt = Math.min(raw, 0.05);
    stage.time += Math.min(raw, 0.25);
    const t = stage.time;
    ema = ema * 0.95 + raw * 1000 * 0.05;
    slowFor = ema > 28 ? slowFor + raw : 0;
    if (slowFor > 2 && stage.quality < 2) {
      stage.quality++;
      slowFor = 0;
      if (stage.quality === 2 && dpr > 1) { dpr = Math.max(1, dpr - 0.25); renderer.setPixelRatio(dpr); resize(); }
    }
    bloom.enabled = stage.quality === 0;
    if (camT < 1 && camFrom) {
      camT = Math.min(1, camT + Math.min(raw, 0.25) / 1.2);
      const k = ease(camT);
      camera.position.lerpVectors(camFrom.pos, goalCache.pos, k);
      controls.target.lerpVectors(camFrom.target, goalCache.target, k);
    } else if (frame.sway && spin && !userTouched && goalCache) {
      // a slow sideways sway instead of a full spin, so left-to-right stays left-to-right
      const a = Math.sin((t / 16) * Math.PI * 2) * 0.18;
      const off = goalCache.pos.clone().sub(goalCache.target).applyAxisAngle(new THREE.Vector3(0, 1, 0), a);
      camera.position.copy(goalCache.target).add(off);
    }
    controls.update();
    if (dust.visible && !REDUCED) dust.rotation.y += dt * 0.01;
    for (const h of stage.hooks) h(t, dt);
    views[frame.view].update(t, dt);
    composer.render();
    stage.labels.update(frame.view, camera, W, Hh);
    if (net.pinned && net.focusSel && frame.view === "net" && !stage.narrow) placeTip(net.focusSel);
  };
  loop();
  return stage;
}

// app.js — the chapters, the training engine and every interactive panel.
import * as C from "./core.js";

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 2) => (Object.is(n, -0) ? 0 : n).toFixed(d).replace("-", "−");
const int = (n) => Math.round(n).toLocaleString("en-US");
const show = (t) => (t === " " ? "␣" : t.replace(/^ /, "␣"));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const frame = () => new Promise((r) => requestAnimationFrame(r));
const ICON_PLAY = `<svg class="i f" viewBox="0 0 16 16"><path d="M4.5 2.8v10.4L13 8z"/></svg>`;
const ICON_PAUSE = `<svg class="i f" viewBox="0 0 16 16"><rect x="3.5" y="2.5" width="3.2" height="11" rx="1"/><rect x="9.3" y="2.5" width="3.2" height="11" rx="1"/></svg>`;

// ───────────────────────────── Data, tokenizer, models ─────────────────────────────
const N = 3, D = 10, H = 48;
const bpe = new C.BPE().train(C.TRAIN_SENTENCES);
const V = bpe.vocab.length;
const TR = C.makeExamples(C.TRAIN_SENTENCES, bpe, N);
const VA = C.makeExamples(C.VAL_SENTENCES, bpe, N);
const START = bpe.id.get(C.START);
const DOT = bpe.id.get(".");
const learner = new C.Model(V, { D, N, H }).init(1);
const ref = new C.Model(V, { D, N, H }).init(7);
const REF_STEPS = 1000;
let refReady = false, refStep = 0;

const ctxOf = (text) => [...Array(N).fill(START), ...bpe.ids(text)].slice(-N);
const ctxLabel = (ctx) => ctx.map((id) => show(bpe.vocab[id])).join(" ");
const TRAIN_WORDS = new Set(C.SENTENCES.flatMap((s) => s.split(" ")));

const CATS = {
  animals: "cat dog bird cow fox mouse", people: "boy girl",
  food: "fish meat bread cake apples seeds grass cheese milk water juice",
  places: "house park garden tree nest barn forest field",
  actions: "eats drinks sees likes chases sleeps plays sits walks runs gives is",
  describing: "big small happy red fast slow soft blue", little: "the in to . [start]",
};
const CAT_HEX = { animals: "#ffb454", people: "#ff7eb6", food: "#7ee787", places: "#6cb6ff", actions: "#c297ff", describing: "#f2d45c", little: "#9aa7bd", pieces: "#58657a" };
const CAT_NAME = { animals: "animals", people: "people", food: "food & drink", places: "places", actions: "action words", describing: "describing words", little: "small words", pieces: "word pieces" };
const catMap = new Map();
for (const [c, ws] of Object.entries(CATS)) for (const w of ws.split(" ")) catMap.set(w, c);
const categoryOf = (tok) => catMap.get(tok.trim()) || "pieces";
const catOrder = Object.keys(CAT_HEX);
const byCat = (a, b) => catOrder.indexOf(categoryOf(bpe.vocab[a])) - catOrder.indexOf(categoryOf(bpe.vocab[b])) || a - b;
const usedIds = [...new Set(TR.flatMap((e) => [...e.ctx, e.y]))];
const embedIds = [...usedIds].sort(byCat);
const outOrder = [...embedIds, ...[...Array(V).keys()].filter((k) => !usedIds.includes(k))];
const wordIds = embedIds.filter((id) => categoryOf(bpe.vocab[id]) !== "pieces" && id !== START);
const groupIds = wordIds.filter((id) => categoryOf(bpe.vocab[id]) !== "little");
const GROUP_GOAL = Math.ceil(groupIds.length * 0.75);

// ───────────────────────────── Tasks (missions) ─────────────────────────────
const MISSIONS = [
  { id: "guess3", ch: 1, text: "Guess the next word in all three sentences" },
  { id: "write", ch: 1, text: "Watch the model write a sentence to the full stop" },
  { id: "replay", ch: 2, text: `Replay all ${bpe.merges.length} merges` },
  { id: "unseen", ch: 2, text: "Type a word the model never saw" },
  { id: "groups", ch: 3, text: `Train until ${GROUP_GOAL} of ${groupIds.length} words sit next to a word from their own group`, live: () => `${groupScore().hits} / ${groupIds.length}` },
  { id: "stay", ch: 4, text: "Teach the neuron to say “stay home”, using only learning steps" },
  { id: "undecided", ch: 4, text: "Find inputs where the neuron can't decide" },
  { id: "swap", ch: 5, text: "Change one word and change the model's top guess" },
  { id: "inspect", ch: 5, text: "Inspect a hidden neuron (in the 3D view or the strip)" },
  { id: "ball", ch: 6, text: "Roll the ball to the bottom in under 100 steps" },
  { id: "val130", ch: 6, text: "Train until the held-out loss is below 1.30", live: () => (isNaN(engine.lastVal) ? "" : `now ${fmt(engine.lastVal)}`) },
  { id: "early", ch: 6, text: "Let early stopping end a run" },
  { id: "newsent", ch: 7, text: "Get a model to write a sentence that is not in its training data" },
  { id: "own600", ch: 7, text: "Write with your own model after 600+ training steps" },
];
const done = new Set();
try { for (const id of JSON.parse(localStorage.getItem("aifs:missions:v1") || "[]")) done.add(id); } catch (e) { /* storage unavailable: progress lives in memory only */ }
function saveDone() { try { localStorage.setItem("aifs:missions:v1", JSON.stringify([...done])); } catch (e) { /* ignore */ } }
function markDone(id) {
  if (done.has(id)) return;
  done.add(id);
  saveDone();

}
function renderTasks() {}
function updateLiveTasks() {}
function announce(msg) { const el = $("srLive"); el.textContent = ""; setTimeout(() => (el.textContent = msg), 30); }

// ───────────────────────────── Training engine: bounded runs, early stopping, best checkpoint ─────────────────────────────
const PATIENCE = 300, MIN_DELTA = 0.002, MAX_TOTAL = 10000, STEPS_PER_EPOCH = TR.length / 32;
const engine = {
  phase: "ready", run: null, budget: 2000, earlyStop: true, lr: 0.3, speed: 3, batch: 32,
  rng: C.makeRng(123), history: [], ghost: [], marks: [], samples: [], seen: 0,
  best: { step: 0, val: Infinity, seen: 0 }, bestModel: new C.Model(V, { D, N, H }), endModel: new C.Model(V, { D, N, H }),
  lastRun: null, lastTrain: NaN, lastVal: NaN, rateLog: [], subs: new Set(),
  get epoch() { return this.seen / TR.length; },
  measureVal() { this.lastVal = learner.meanLoss(VA); return this.lastVal; },
  measureTrain() { this.lastTrain = learner.meanLoss(TR); return this.lastTrain; },
  record(withTrain) {
    const row = { step: learner.step, val: this.measureVal() };
    if (withTrain) row.train = this.measureTrain();
    const last = this.history[this.history.length - 1];
    if (last && last.step === row.step) this.history[this.history.length - 1] = { ...last, ...row };
    else this.history.push(row);
    if (this.lastVal < this.best.val - MIN_DELTA) this.saveBest();
  },
  saveBest() { this.bestModel.copyFrom(learner); this.best = { step: learner.step, val: this.lastVal, seen: this.seen }; },
  reset() {
    learner.init(1);
    this.rng = C.makeRng(123);
    Object.assign(this, { phase: "ready", run: null, history: [], ghost: [], marks: [], samples: [], seen: 0, lastRun: null, best: { step: 0, val: Infinity, seen: 0 } });
    this.record(true);
    this.emit("reset");
  },
  start({ steps = this.budget, source = "ch6", earlyStop = this.earlyStop } = {}) {
    if (this.phase === "paused" && this.run) { this.phase = "running"; this.emit("run"); return; }
    if (learner.step >= MAX_TOTAL) { this.emit("capped"); return; }
    this.ghost = [];
    this.run = { start: learner.step, end: Math.min(learner.step + steps, MAX_TOTAL), source, earlyStop, patienceFrom: learner.step };
    this.marks.push({ step: learner.step, kind: "start" });
    this.lastRun = null;
    this.rateLog = [];
    this.phase = "running";
    this.emit("run");
  },
  pause() { if (this.phase === "running") { this.phase = "paused"; this.emit("run"); } },
  toggle(opts) { if (this.phase === "running") this.pause(); else this.start(opts); },
  train(n) {
    for (let i = 0; i < n && this.phase === "running"; i++) this.oneStep();
  },
  oneStep() {
    learner.trainStep(TR, this.batch, this.lr, this.rng);
    this.seen += this.batch;
    if (learner.step % 10 === 0) this.record(learner.step % 50 === 0);
    if (learner.step % 100 === 0) this.samples.unshift({ step: learner.step, text: writeSentence(learner, 0.8, this.rng) });
    const r = this.run;
    if (!r) return;
    if (r.earlyStop && learner.step - Math.max(this.best.step, r.patienceFrom) >= PATIENCE) return this.finish("early");
    if (learner.step >= r.end) return this.finish(learner.step >= MAX_TOTAL ? "cap" : "budget");
  },
  finish(reason) {
    if (this.history[this.history.length - 1]?.step !== learner.step) this.record(true); else this.measureTrain();
    this.endModel.copyFrom(learner);
    const r = this.run;
    this.lastRun = { reason, start: r.start, end: r.end, stoppedAt: learner.step, seenAtStop: this.seen, best: { ...this.best }, bestTrain: this.bestModel.meanLoss(TR), finalVal: this.lastVal, finalTrain: this.lastTrain, restored: false, source: r.source };
    this.run = null;
    this.phase = "ready";
    if (reason === "early") this.restoreBest(true);
    this.emit("done");
  },
  restoreBest(quiet) {
    const b = this.best;
    this.ghost = this.history.filter((p) => p.step > b.step);
    this.history = this.history.filter((p) => p.step <= b.step);
    learner.copyFrom(this.bestModel);
    this.seen = b.seen;
    this.measureVal(); this.measureTrain();
    const last = this.history[this.history.length - 1];
    if (last) last.train = this.lastTrain;
    if (this.lastRun) this.lastRun.restored = true;
    if (!quiet) this.emit("restore");
  },
  keepFinal() {
    learner.copyFrom(this.endModel);
    this.history = this.history.concat(this.ghost).sort((a, b) => a.step - b.step);
    this.ghost = [];
    this.seen = this.lastRun.seenAtStop;
    this.measureVal(); this.measureTrain();
    this.lastRun.restored = false;
    this.emit("restore");
  },
  setLr(v) {
    if (this.run && v !== this.lr) { this.marks.push({ step: learner.step, kind: "lr", from: this.lr, to: v }); this.run.patienceFrom = learner.step; }
    this.lr = v;
  },
  stepOnce() {
    if (this.phase === "running") this.pause();
    this.oneStep();
    this.record(true);
    this.emit("step");
  },
  rate() {
    const L = this.rateLog;
    if (L.length < 2) return 0;
    const a = L[0], b = L[L.length - 1];
    return (b.step - a.step) / Math.max((b.t - a.t) / 1000, 1e-3);
  },
  on(fn) { this.subs.add(fn); },
  emit(kind) { for (const f of this.subs) f(kind); },
};

function writeSentence(model, temp, rng, maxLen = 12) {
  let ctx = Array(N).fill(START), out = "";
  for (let i = 0; i < maxLen; i++) {
    const { id } = model.sample(ctx, temp, rng);
    out += bpe.vocab[id] === C.START ? " [start]" : bpe.vocab[id];
    if (id === DOT) break;
    ctx = [...ctx.slice(1), id];
  }
  return out.trim();
}

let lastEmit = 0;
function tick(now) {
  requestAnimationFrame(tick);
  if (engine.phase === "running") {
    engine.train(Math.min(engine.speed, 40));
    engine.rateLog.push({ t: now, step: learner.step });
    while (engine.rateLog.length > 2 && now - engine.rateLog[0].t > 2000) engine.rateLog.shift();
    if (now - lastEmit > 90) { lastEmit = now; engine.emit("step"); }
  }
}

// undo-able reset (no confirm() dialogs in this frame)
let undoSnap = null, undoTimer = 0;
function resetWithUndo(hostEl) {
  if (learner.step > 0) {
    undoSnap = { model: learner.clone(), best: { ...engine.best }, bestModel: engine.bestModel.clone(), history: engine.history.slice(), ghost: engine.ghost.slice(), marks: engine.marks.slice(), samples: engine.samples.slice(), seen: engine.seen, lastRun: engine.lastRun, lastTrain: engine.lastTrain, lastVal: engine.lastVal };
  }
  engine.reset();
  if (!undoSnap) return;
  clearTimeout(undoTimer);
  document.querySelectorAll(".undo").forEach((u) => (u.hidden = true));
  hostEl.hidden = false;
  hostEl.innerHTML = `<span>Model reset to step 0.</span><button class="btn sm" type="button">Undo</button>`;
  hostEl.querySelector("button").onclick = () => {
    const s = undoSnap;
    if (!s) return;
    learner.copyFrom(s.model);
    engine.bestModel.copyFrom(s.bestModel);
    Object.assign(engine, { best: s.best, history: s.history, ghost: s.ghost, marks: s.marks, samples: s.samples, seen: s.seen, lastRun: s.lastRun, lastTrain: s.lastTrain, lastVal: s.lastVal });
    undoSnap = null;
    hostEl.hidden = true;
    engine.emit("restore");
  };
  undoTimer = setTimeout(() => { hostEl.hidden = true; undoSnap = null; }, 10000);
}

// ───────────────────────────── 3D stage (separate module; the page still works without WebGL) ─────────────────────────────
let stage = null;
const SCENES = {
  hero: { what: "The pre-trained copy", hint: "Drag to turn. Hover a neuron to see its numbers. Click a word to change it." },
  net: { what: "Your model", hint: "Drag to turn. Hover or tap a neuron to see its numbers. Click a word to change it." },
  tiles: { what: "Your text, as tokens", hint: "Each tile is one token with its ID. Click a tile to see how it was built." },
  embed: { what: "Every word's embedding", hint: "Drag to turn. Click a dot to see its closest words." },
  neuron: { what: "One neuron", hint: "Line thickness shows the size of each weight." },
  land: { what: "Loss landscape of a two-weight neuron", hint: "Click the surface to drop the ball. Drag to turn." },
};
const hintSeen = new Set();

async function loadStage() {
  try {
    const mod = await import("./stage.js");
    const hex = Object.fromEntries(Object.entries(CAT_HEX).map(([k, v]) => [k, parseInt(v.slice(1), 16)]));
    try { await Promise.race([document.fonts.load('600 58px "IBM Plex Mono"'), wait(1500)]); } catch (e) { /* fallback font is fine */ }
    stage = await mod.createStage($("stage"), { V, D, N, H, vocab: bpe.vocab, embedIds, outOrder, categoryOf, catHex: hex });
    $("stageFallback").remove();
    stage.net.setModel(ref);
    stage.onLandPick = (wb) => land.reset(wb, true);
    stage.onFocus = (sel) => { if (sel?.layer === "hid") { fwd.highlight(sel.index); } };
    stage.onPick = (sel) => onNetPick(sel);
    stage.net.onSlot = (n, el) => openSlotPop(n);
    stage.onScale = (note) => { if (stage.mode === "net") setLegend(); };
    stage.onEmbedPick = (id) => { embSel = id; renderEmb(); };
    stage.onTilePick = (tok, x, y) => tileInfo(tok, x, y);
    stage.onUserMove = () => hideHint();
    stage.onZoomLayer = () => {};
    stage.tiles.set(bpe.encode($("tokInput").value, +$("tokMerges").value).slice(0, 40));
    updateEmbed(true);
    neuron.render();
    land.render();
    stage.hooks.push(stageTick);
    activate(currentBeat, true);
  } catch (err) {
    console.error(err);
    $("stageFallback").textContent = "The 3D view could not start. It needs WebGL and access to cdn.jsdelivr.net. Every panel in the lesson still works.";
  }
}

function hideHint() { const s = $("stage").dataset.mode; if (s) hintSeen.add(s); $("stageHint").classList.add("gone"); }

function setLegend() {
  const L = $("stLegend"), mode = activeScene;
  if (mode === "hero" || mode === "net") {
    const tr = stage?.net.training;
    L.innerHTML = (tr ? `<span class="gradkey"><i></i>error flowing back</span>` : "") + `<span class="bar"></span><span class="ticks"><span>negative</span><span>positive</span></span>`;
  } else if (mode === "embed") {
    L.innerHTML = `<span class="cats">${catOrder.filter((c) => c !== "little").map((c) => `<span><i style="background:${CAT_HEX[c]}"></i>${CAT_NAME[c]}</span>`).join("")}</span>`;
  } else if (mode === "neuron") {
    L.innerHTML = `<span class="bar"></span><span class="ticks"><span>−</span><span>0</span><span>+</span></span>`;
  } else L.innerHTML = "";
}

function stageMeta(text) { $("stageMeta").textContent = text; }

// which model a beat shows
function beatModel(beat) {
  const m = beat?.dataset.model;
  return m === "ref" ? ref : m === "fwd" ? fwd.model() : m === "gen" ? gen.model() : learner;
}

let heroNext = 0, liveNext = 0, weightsNext = 0, lastWeightsStep = -1, embedNext = 0, autoHero = true;
const HERO_CTX = TR.filter((e) => (e.ctx[0] !== START || e.ctx[1] !== START) && [...e.ctx, e.y].every((id) => id === START || categoryOf(bpe.vocab[id]) !== "pieces"));
let heroI = 0;
function stageTick(t) {
  const net = stage.net;
  if (activeScene === "hero" && refReady && autoHero && t > heroNext && !genBusy && !net.pinned && $("stagePop").hidden) {
    heroNext = t + (REDUCED ? 8 : 6.5);
    const e = HERO_CTX[(heroI = (heroI * 7 + 13) % HERO_CTX.length)];
    net.show(ref.forward(e.ctx), { from: 0, speed: 1.3 });
    stageMeta(`pre-trained copy · reading “${ctxLabel(e.ctx)}”`);
  }
  if (activeScene === "net" && net.model === learner) {
    if (t > weightsNext && learner.step !== lastWeightsStep) { weightsNext = t + 0.3; lastWeightsStep = learner.step; net.refreshWeights(); }
    const trainingView = engine.phase === "running" && !genBusy;
    if (net.training !== trainingView) { net.setTraining(trainingView); setLegend(); }
    if (trainingView && t > liveNext) {
      liveNext = t + 0.25;
      net.live(learner.forward(currentBeat?.dataset.fwd ? fwd.ctx : probeCtx()));
      net.trainStep(learner);
    }
  }
  else if (net.training) { net.setTraining(false); setLegend(); }
  if (activeScene === "embed" && t > embedNext) { embedNext = t + 0.2; updateEmbed(); }
  updateDockStage();
}

// ───────────────────────────── Scroll: which beat is being read? ─────────────────────────────
let currentBeat = null, activeScene = "hero";
const navLinks = [...document.querySelectorAll("#nav a")];
const CH_NAMES = ["What is AI", "Tokens", "Embeddings", "One neuron", "The network", "Training", "Generating"];
navLinks.forEach((a, i) => (a.dataset.name = CH_NAMES[i]));
function activate(beat, force) {
  if (!beat || (beat === currentBeat && !force)) return;
  currentBeat = beat;
  const scene = beat.dataset.scene;
  activeScene = scene;
  const ch = beat.closest(".chapter");
  navLinks.forEach((a) => { const on = !!ch && a.getAttribute("href") === "#" + ch.id; a.classList.toggle("on", on); if (on) a.setAttribute("aria-current", "step"); else a.removeAttribute("aria-current"); });
  $("navName").textContent = ch ? CH_NAMES[+ch.dataset.ch - 1] : "Start";
  const model = beatModel(beat);
  const isNet = scene === "hero" || scene === "net";
  $("stageWhat").textContent = isNet ? (model === ref ? "The pre-trained copy" : "Your model") : SCENES[scene].what;
  $("stageHint").textContent = SCENES[scene].hint;
  $("stageHint").classList.toggle("gone", hintSeen.has(scene));
  $("dkStep").hidden = !isNet;
  updateTrainDock();
  if (!stage) return;
  stage.setMode(scene);
  setLegend();
  closeSlotPop();
  if (isNet) {
    const net = stage.net;
    if (net.model !== model) net.setModel(model);
    // pick a sensible default for what the lines show
    setView(beat.dataset.train ? "weights" : "signal");
    if (beat.dataset.fwd || beat.dataset.model === "fwd") fwd.toStage(false);
    else if (beat.dataset.model === "gen" && gen.lastF) net.show(gen.lastF.f, { from: 4, pick: gen.lastF.id });
    else if (model === learner) net.show(learner.forward(probeCtx()), { from: 0 });
    else if (scene === "hero") heroNext = 0;
    else net.show(model.forward(gen.ctx || probeCtx()), { from: 0 });
    metaForNet();
  } else if (scene === "tiles") stageMeta($("tokStats").textContent + " · " + $("tokMergeNow").textContent.split(":")[0]);
  else if (scene === "embed") { updateEmbed(true); stageMeta(`your model · step ${int(learner.step)} · 10 numbers per word, drawn in 3D`); }
  else if (scene === "neuron") neuron.render();
  else if (scene === "land") land.render();
}
function metaForNet() {
  if (!stage) return;
  if (stage.net.model === ref) { if (refReady) stageMeta(`pre-trained copy · ${int(REF_STEPS)} steps`); }
  else stageMeta(`your model · step ${int(learner.step)}${isNaN(engine.lastVal) ? "" : ` · held-out loss ${fmt(engine.lastVal)}`}`);
}
function setupScroll() {
  const beats = [...document.querySelectorAll(".beat[data-scene]")];
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) activate(e.target);
  }, { rootMargin: matchMedia("(max-width: 900px)").matches ? "-62% 0px -30% 0px" : "-40% 0px -50% 0px" });
  beats.forEach((b) => io.observe(b));
  currentBeat = beats[0];
}

// ───────────────────────────── Stage dock and tools ─────────────────────────────
function setView(v) {
  stage?.net.setView(v);
}
const SAY = [
  "The tokenizer splits the text into tokens and gives each one an ID number.",
  "Each token pulls its row of 10 numbers out of the embedding table.",
  "48 hidden neurons each multiply those 30 numbers by their weights and add them up.",
  "Every word the model knows gets a score from the hidden neurons.",
  "Scores turn into chances. The brightest output is the model's guess.",
];
let lastSay = "";
function updateSay() {
  let html = "";
  if (stage && (activeScene === "hero" || activeScene === "net")) {
    const net = stage.net;
    if (net.training) html = `<b>training</b>Violet light is the error flowing backward. Each weight it passes gets nudged to make the next guess better.`;
    else if (net.f) { const k = Math.min(4, Math.max(0, Math.floor(net.P + 0.001))); html = SAY[k]; }
  }
  if (html !== lastSay) { lastSay = html; $("stSay").innerHTML = html; }
}
function updateDockStage() {
  updateSay();
  const netScene = activeScene === "hero" || activeScene === "net";
  $("dkStep").hidden = !netScene || !!stage?.net.training;
  if (!stage) return;
  const S = Math.min(4, Math.max(0, Math.floor(stage.net.P + 0.001))), names = ["Tokens", "Embeddings", "Hidden", "Scores", "Probabilities"];
  const html = `Step ${S + 1} of 5`;
  if ($("dkLabel").innerHTML !== html) $("dkLabel").innerHTML = html;
  $("dkPrev").disabled = !stage.net.f || S <= 0;
  $("dkNext").disabled = !stage.net.f || S >= 4;
}
function dockStep(d) {
  if (!stage?.net.f) return;
  autoPause();
  const k = Math.max(0, Math.min(4, Math.round(stage.net.S) + d));
  stage.net.setStage(k);
  if (currentBeat?.dataset.fwd) { fwd.step = k; fwd.render(); }
}
$("dkPrev").addEventListener("click", () => dockStep(-1));
$("dkNext").addEventListener("click", () => dockStep(1));
$("dkRun").addEventListener("click", () => {
  if (!stage) return;
  if (currentBeat?.dataset.fwd || currentBeat?.dataset.model === "fwd") return fwd.play();
  const f = stage.net.f || stage.net.model.forward(probeCtx());
  stage.net.show(f, { from: 0, speed: 1.3 });
  if (activeScene === "hero") heroNext = stage.time + 8;
});
function autoPause() { if (activeScene === "hero" && autoHero) { autoHero = false; setTimeout(() => (autoHero = true), 20000); } }
$("tReset").addEventListener("click", () => stage?.resetView());
$("dkTrainBtn").addEventListener("click", () => engine.toggle({ source: "dock" }));
$("dkStepBtn").addEventListener("click", () => engine.stepOnce());

// slot popover: change one of the three words the network reads
let popSlot = -1;
function openSlotPop(n) {
  if (!stage?.net.f) return;
  popSlot = n;
  const cur = stage.net.f.ctx[n];
  const groups = catOrder.filter((c) => c !== "pieces").map((c) => {
    const ids = wordIds.filter((id) => categoryOf(bpe.vocab[id]) === c);
    if (c === "little") ids.unshift(START);
    return ids.length ? `<div class="pop-grid">${ids.map((id) => `<button type="button" class="w" style="--c:${CAT_HEX[c]}" data-id="${id}" aria-current="${id === cur}">${esc(show(bpe.vocab[id]))}</button>`).join("")}</div>` : "";
  }).join("");
  const pop = $("stagePop");
  pop.innerHTML = `<h4>Word ${n + 1} of 3: pick a replacement</h4>${groups}`;
  pop.hidden = false;
  autoPause();
  pop.querySelector('[aria-current="true"]')?.focus() || pop.querySelector("button")?.focus();
}
function closeSlotPop(refocus) { if ($("stagePop").hidden) return; $("stagePop").hidden = true; if (refocus && stage) stage.net.slotLabels[popSlot]?.el.focus(); popSlot = -1; }
$("stagePop").addEventListener("click", (e) => {
  const b = e.target.closest("button.w");
  if (!b || popSlot < 0) return;
  const ctx = stage.net.f.ctx.slice();
  ctx[popSlot] = +b.dataset.id;
  closeSlotPop(true);
  useContext(ctx);
});
$("stagePop").addEventListener("keydown", (e) => {
  if (e.key === "Escape") { e.preventDefault(); closeSlotPop(true); return; }
  const btns = [...$("stagePop").querySelectorAll("button.w")], i = btns.indexOf(document.activeElement);
  const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
  if (d && i >= 0) { e.preventDefault(); btns[(i + d + btns.length) % btns.length].focus(); }
});
document.addEventListener("pointerdown", (e) => { if (!$("stagePop").hidden && !e.target.closest("#stagePop") && !e.target.closest(".lbl.slot")) closeSlotPop(); });
function useContext(ctx) {
  if (currentBeat?.dataset.fwd || currentBeat?.dataset.model === "fwd") { fwd.setCtx(ctx); return; }
  const m = stage.net.model;
  stage.net.show(m.forward(ctx), { from: 1, speed: 1.6 });
  if (activeScene === "hero") stageMeta(`pre-trained copy · reading “${ctxLabel(ctx)}”`);
}

function onNetPick(sel) {
  if (sel.layer === "hid") markDone("inspect");
  if (!(currentBeat?.dataset.fwd || currentBeat?.dataset.model === "fwd")) return;
  if (sel.layer === "hid") { fwd.j = sel.index; fwd.step = 2; fwd.render(); }
  else if (sel.layer === "out") { fwd.extra = sel.index; fwd.step = 4; fwd.render(); }
  else { fwd.step = 1; fwd.render(); }
}

// tokenizer tile info
let tileTimer = 0;
function tileInfo(tok, x, y) {
  const tip = $("stageTip");
  clearTimeout(tileTimer);
  if (!tok) { tip.hidden = true; return; }
  const mi = tok.id - (1 + C.BASE_CHARS.length);
  const body = mi >= 0
    ? `<div class="tip-h">${esc(show(tok.text))} · token #${tok.id}</div><div class="tip-m">= ${esc(show(bpe.merges[mi].a))} + ${esc(show(bpe.merges[mi].b))}</div><div class="tip-m">merge ${mi + 1} of ${bpe.merges.length}, seen ${bpe.merges[mi].count} times in training</div>`
    : `<div class="tip-h">${esc(show(tok.text))} · token #${tok.id}</div><div class="tip-m">a single character from the starting alphabet</div>`;
  tip.innerHTML = body;
  tip.hidden = false;
  const W = $("stage").clientWidth;
  tip.style.transform = `translate(${Math.min(W - 262, x + 14)}px,${Math.max(64, y - 30)}px)`;
  tileTimer = setTimeout(() => (tip.hidden = true), 5000);
}

// ───────────────────────────── Shared UI pieces ─────────────────────────────
function bars(el, rows) {
  el.innerHTML = rows.map((r) => `<div class="bar${r.hit ? " hit" : ""}"><span>${esc(r.label)}</span><div class="track"><div class="fill" style="width:${Math.max(0.5, r.p * 100).toFixed(1)}%"></div></div><span>${(r.p * 100).toFixed(1)}%</span></div>`).join("");
}
function setupCanvas(cv) {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const x = cv.getContext("2d");
  x.setTransform(dpr, 0, 0, dpr, 0, 0);
  x.clearRect(0, 0, w, h);
  return [x, w, h];
}
function axes(x, w, h, pad, xr, yr, xt, yt, xfmt) {
  const X = (v) => pad.l + ((v - xr[0]) / (xr[1] - xr[0])) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - ((v - yr[0]) / (yr[1] - yr[0])) * (h - pad.t - pad.b);
  x.strokeStyle = css("--line"); x.lineWidth = 1; x.fillStyle = css("--muted");
  x.font = `11px ${css("--mono")}`;
  x.textAlign = "right"; x.textBaseline = "middle";
  for (const v of yt) { x.beginPath(); x.moveTo(pad.l, Y(v)); x.lineTo(w - pad.r, Y(v)); x.stroke(); x.fillText(fmt(v, Number.isInteger(v) ? 0 : 1), pad.l - 6, Y(v)); }
  x.textAlign = "center"; x.textBaseline = "top";
  for (const v of xt) x.fillText(xfmt ? xfmt(v) : Math.abs(v) >= 1000 ? int(v) : fmt(v, Number.isInteger(v) ? 0 : 1), X(v), h - pad.b + 6);
  return [X, Y];
}
function niceTicks(max, n = 4) {
  const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}
function seg(el, attr, onPick) {
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button[role=radio]");
    if (!b || el.getAttribute("aria-disabled") === "true") return;
    el.querySelectorAll("button[role=radio]").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    onPick(b.dataset[attr], b);
  });
  el.addEventListener("keydown", (e) => {
    const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!d) return;
    const btns = [...el.querySelectorAll("button[role=radio]")];
    const i = btns.findIndex((b) => b.getAttribute("aria-checked") === "true");
    const nb = btns[(i + d + btns.length) % btns.length];
    e.preventDefault();
    nb.focus(); nb.click();
  });
  const sync = () => el.querySelectorAll("button[role=radio]").forEach((b) => b.tabIndex = b.getAttribute("aria-checked") === "true" ? 0 : -1);
  new MutationObserver(sync).observe(el, { attributes: true, subtree: true, attributeFilter: ["aria-checked"] });
  sync();
}
function segSet(el, attr, val) { el.querySelectorAll("button[role=radio]").forEach((b) => b.setAttribute("aria-checked", String(b.dataset[attr] === String(val)))); }
// slider fill: unsigned fills from the left in ink; signed (data-signed) fills from zero in cyan/orange
function fillRange(el) {
  const lo = +el.min, hi = +el.max, v = +el.value, p = ((v - lo) / (hi - lo)) * 100;
  if (el.dataset.signed !== undefined) {
    const z = ((0 - lo) / (hi - lo)) * 100, a = Math.min(z, p), b = Math.max(z, p), c = v >= 0 ? "var(--pos)" : "var(--neg)";
    el.style.setProperty("--fill", `linear-gradient(90deg, var(--well) ${a}%, ${c} ${a}%, ${c} ${b}%, var(--well) ${b}%), linear-gradient(90deg, transparent calc(${z}% - 1px), var(--control) calc(${z}% - 1px), var(--control) calc(${z}% + 1px), transparent calc(${z}% + 1px))`);
  } else el.style.setProperty("--fill", `linear-gradient(90deg, var(--ink) ${p}%, var(--well) ${p}%)`);
}
document.addEventListener("input", (e) => { if (e.target.matches?.('input[type="range"]')) fillRange(e.target); });
const fillAll = () => document.querySelectorAll('input[type="range"]').forEach(fillRange);

// predict-then-check
function predict(host, { q, options, onPick }) {
  host.innerHTML = `<p class="q">Predict first: ${esc(q)}</p><div class="chips">${options.map((o) => `<button class="chip" type="button" aria-pressed="false" data-v="${o.v}">${esc(o.label)}</button>`).join("")}</div><p class="verdict" hidden></p>`;
  const state = { pick: null, host };
  host.querySelector(".chips").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    state.pick = b.dataset.v;
    host.querySelectorAll(".chips button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    onPick?.(state.pick);
  });
  state.verdict = (right, text) => { const v = host.querySelector(".verdict"); v.hidden = false; v.innerHTML = text; host.classList.toggle("right", right); };
  return state;
}

// ───────────────────────────── Opening: guess the next word ─────────────────────────────
const GUESSES = [
  { text: "the cow eats", opts: ["grass", "fish", "bread", "cake"] },
  { text: "the dog chases the", opts: ["ball", "seeds", "milk", "barn"] },
  { text: "the girl drinks", opts: ["juice", "grass", "meat", "nest"] },
];
let gi = 0, guessAnswered = new Set(), guessMatches = 0;
function renderGuess() {
  const g = GUESSES[gi];
  $("guessCount").textContent = `${gi + 1} of ${GUESSES.length}`;
  $("guessQ").innerHTML = `${esc(g.text)} <span class="blank">?</span>`;
  const opts = [...g.opts].sort((a, b) => ((a.charCodeAt(1) * 7 + gi) % 5) - ((b.charCodeAt(1) * 7 + gi) % 5));
  $("guessOpts").innerHTML = opts.map((o) => `<button class="chip" type="button" aria-pressed="false" data-w="${o}"${refReady ? "" : ' aria-disabled="true"'}>${o}</button>`).join("");
  $("guessReveal").hidden = true;
}
$("guessOpts").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (!refReady) { $("guessWait").hidden = false; return; }
  const g = GUESSES[gi], pick = " " + b.dataset.w;
  $("guessOpts").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  $("guessQ").querySelector(".blank").textContent = b.dataset.w;
  const ctx = ctxOf(g.text), f = ref.forward(ctx);
  const top = C.topK(f.p, 5);
  const pid = bpe.id.get(pick);
  if (pid !== undefined && !top.includes(pid)) top.push(pid);
  const t0 = top[0];
  bars($("guessBars"), top.map((k) => ({ label: show(bpe.vocab[k]) + (k === pid ? " ←you" : ""), p: f.p[k], hit: k === pid })));
  const first = !guessAnswered.has(gi);
  guessAnswered.add(gi);
  if (pid === t0) { if (first) guessMatches++; $("guessVerdict").innerHTML = `Same as the model's top guess (${(f.p[t0] * 100).toFixed(1)}%).`; }
  else $("guessVerdict").innerHTML = `The model gives “${esc(b.dataset.w)}” ${(f.p[pid] * 100).toFixed(1)}%. Its top guess is “${esc(bpe.vocab[t0].trim())}” (${(f.p[t0] * 100).toFixed(1)}%).`;
  $("guessScore").textContent = `You matched the model ${guessMatches} of ${guessAnswered.size}.`;
  $("guessReveal").hidden = false;
  $("guessNext").textContent = guessAnswered.size === GUESSES.length && gi === GUESSES.length - 1 ? "Start again" : "Next sentence";
  if (guessAnswered.size === GUESSES.length) markDone("guess3");
  if (stage && activeScene === "hero") { autoPause(); stage.net.show(f, { from: 0, speed: 1.6, pick: pid }); stageMeta(`pre-trained copy · reading “${ctxLabel(ctx)}”`); }
});
$("guessNext").addEventListener("click", () => { gi = (gi + 1) % GUESSES.length; renderGuess(); });

// writing, one token at a time (Chapter 1 and 7)
let genBusy = false;
async function animateWrite({ model, ctx, outEl, barsEl, temp, rng, maxLen = 12, single = false, onToken }) {
  genBusy = true;
  for (let i = 0; i < (single ? 1 : maxLen); i++) {
    const { id, f, q } = model.sample(ctx, temp, rng);
    const top = C.topK(q, 6);
    if (!top.includes(id)) top.push(id);
    bars(barsEl, top.map((k) => ({ label: show(bpe.vocab[k]), p: q[k], hit: k === id })));
    outEl.querySelectorAll(".new").forEach((n) => n.classList.remove("new"));
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "t new";
    btn.textContent = show(bpe.vocab[id]);
    btn.setAttribute("aria-pressed", "false");
    btn.title = "See the other choices it had here";
    btn._step = { f, q, id, top };
    outEl.appendChild(btn);
    onToken?.(btn);
    if (stage && (activeScene === "hero" || activeScene === "net")) {
      if (stage.net.model !== model) stage.net.setModel(model);
      stage.net.show({ ...f, p: q }, { from: 0, speed: REDUCED ? 99 : 5, pick: id });
      heroNext = stage.time + 4;
    }
    ctx = [...ctx.slice(1), id];
    if (id === DOT) break;
    if (!single) await wait(REDUCED ? 300 : 1100);
  }
  genBusy = false;
  return ctx;
}
function revisit(outEl, barsEl, btn) {
  const s = btn._step;
  outEl.querySelectorAll("button.t").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
  bars(barsEl, s.top.map((k) => ({ label: show(bpe.vocab[k]), p: s.q[k], hit: k === s.id })));
  if (stage && (activeScene === "hero" || activeScene === "net")) { autoPause(); stage.net.show({ ...s.f, p: s.q }, { from: 3, speed: 3, pick: s.id }); }
}
const ch1Rng = C.makeRng((Date.now() & 0xffff) + 1);
$("ch1Gen").addEventListener("click", async () => {
  if (genBusy || !refReady) return;
  $("ch1Gen").disabled = true; $("ch1Gen").textContent = "Writing…";
  $("ch1Out").innerHTML = "";
  autoPause();
  const ctx = await animateWrite({ model: ref, ctx: Array(N).fill(START), outEl: $("ch1Out"), barsEl: $("ch1Bars"), temp: 0.8, rng: ch1Rng });
  if (ctx[N - 1] === DOT) markDone("write");
  $("ch1Gen").disabled = false; $("ch1Gen").textContent = "Write another";
  $("ch1Note").textContent = "Click any word it wrote to see the other choices it had at that point.";
});
$("ch1Out").addEventListener("click", (e) => { const b = e.target.closest("button.t"); if (b && !genBusy) revisit($("ch1Out"), $("ch1Bars"), b); });

// ───────────────────────────── Chapter 2: tokenizer ─────────────────────────────
const MERGES = bpe.merges.length;
$("tokMerges").max = MERGES;
$("tokMerges").value = MERGES;
$("mergeCount").textContent = MERGES;
document.querySelectorAll(".vs").forEach((e) => (e.textContent = V));
$("mergeRows").innerHTML = bpe.merges.map((m, i) => `<tr id="mr${i}"><td class="n">${i + 1}</td><td class="mono">${esc(show(m.a))} + ${esc(show(m.b))}</td><td class="mono">${esc(show(m.a + m.b))}</td><td class="n">${m.count}</td></tr>`).join("");
let lastLimit = MERGES;
function renderTokens() {
  const raw = $("tokInput").value.slice(0, 120);
  const limit = +$("tokMerges").value;
  $("tokMergesOut").textContent = limit;
  const toks = bpe.encode(raw, limit);
  const chars = C.pretokenize(raw).join("").length;
  $("tokStats").textContent = `${chars} characters → ${toks.length} tokens`;
  $("tokChips").innerHTML = toks.map((t) => `<span class="chip tok" style="--h:${Math.round(((t.id * 0.61803398875) % 1) * 360)}">${esc(show(t.text))}<small>${t.id}</small></span>`).join("") || `<span class="note">Type some letters above.</span>`;
  const m = bpe.merges[limit - 1];
  $("tokMergeNow").textContent = limit === 0 ? "No merges yet: every character is its own token." : `Merge ${limit} of ${MERGES}: ${show(m.a)} + ${show(m.b)} → ${show(m.a + m.b)} (seen ${m.count} times)`;
  document.querySelectorAll("#mergeRows tr.now").forEach((r) => r.classList.remove("now"));
  if (limit > 0) { const row = $("mr" + (limit - 1)); row.classList.add("now"); if (row.closest("details").open) { const w = $("mergeWrap"); w.scrollTop = row.offsetTop - w.clientHeight / 2; } }
  const merged = limit === lastLimit + 1 && limit > 0 ? bpe.id.get(m.a + m.b) : null;
  lastLimit = limit;
  if (stage) stage.tiles.set(toks.slice(0, 40), merged);
  if (activeScene === "tiles") stageMeta(`${chars} characters → ${toks.length} tokens · ${limit} of ${MERGES} merges`);
  for (const w of C.normalize(raw).split(/[ .]+/)) if (w.length >= 2 && !TRAIN_WORDS.has(w) && raw !== $("tokInput").defaultValue) { markDone("unseen"); break; }
}
$("tokInput").addEventListener("input", renderTokens);
$("tokMerges").addEventListener("input", () => { stopReplay(); renderTokens(); });
const stepMerge = (d) => { stopReplay(); const s = $("tokMerges"); s.value = Math.max(0, Math.min(MERGES, +s.value + d)); fillRange(s); renderTokens(); };
$("tokPrev").addEventListener("click", () => stepMerge(-1));
$("tokNext").addEventListener("click", () => { stepMerge(1); if (+$("tokMerges").value === MERGES) markDone("replay"); });
let tokTimer = null;
function stopReplay() { if (!tokTimer) return; clearInterval(tokTimer); tokTimer = null; $("tokPlay").innerHTML = `${ICON_PLAY}Replay from 0`; }
$("tokPlay").addEventListener("click", () => {
  if (tokTimer) return stopReplay();
  const s = $("tokMerges");
  if (+s.value >= MERGES) s.value = 0;
  fillRange(s); renderTokens();
  $("tokPlay").innerHTML = `${ICON_PAUSE}Pause`;
  tokTimer = setInterval(() => {
    s.value = +s.value + 1; fillRange(s); renderTokens();
    if (+s.value >= MERGES) { stopReplay(); markDone("replay"); }
  }, REDUCED ? 15 : 110);
});

// ───────────────────────────── Chapter 3: embeddings ─────────────────────────────
let embSel = bpe.id.get(" cat");
$("embChips").innerHTML = wordIds.map((id) => `<button type="button" class="chip cat" style="--c:${CAT_HEX[categoryOf(bpe.vocab[id])]}" aria-pressed="false" data-id="${id}">${esc(show(bpe.vocab[id]))}</button>`).join("");
$("embChips").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; embSel = +b.dataset.id; renderEmb(); });
function rowOf(m, id) { return m.E.subarray(id * D, id * D + D); }
function cosine(a, b) { let s = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return s / Math.sqrt(x * y + 1e-12); }
function nearest(id, pool, k = 5) { const r = rowOf(learner, id); return pool.filter((o) => o !== id).map((o) => [cosine(r, rowOf(learner, o)), o]).sort((a, b) => b[0] - a[0]).slice(0, k); }
function groupScore() {
  let hits = 0;
  for (const id of groupIds) { const [[, nb]] = nearest(id, groupIds, 1); if (categoryOf(bpe.vocab[nb]) === categoryOf(bpe.vocab[id])) hits++; }
  return { hits };
}
function cellStyle(v, scale) {
  const a = Math.min(Math.abs(v) / scale, 1) * 0.6;
  return `background: color-mix(in srgb, var(${v >= 0 ? "--pos" : "--neg"}) ${(a * 100).toFixed(0)}%, var(--raised))`;
}
function renderEmb() {
  $("embChips").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.id === embSel)));
  const row = rowOf(learner, embSel);
  $("embVec").innerHTML = Array.from(row, (v) => `<div style="${cellStyle(v, 1.5)}" title="${v}">${fmt(v)}</div>`).join("");
  $("embStep").textContent = `row ${embSel} · step ${int(learner.step)}`;
  const nb = nearest(embSel, wordIds);
  $("embNear").textContent = `Closest to ${show(bpe.vocab[embSel])} right now: ` + nb.map(([c, o]) => `${show(bpe.vocab[o])} ${c.toFixed(2)}`).join(" · ");
  if (stage) stage.embed.select(embSel, nb.map((x) => x[1]));
  const gs = groupScore();
  if (gs.hits >= GROUP_GOAL) markDone("groups");
  updateLiveTasks();
}
let pcaAxes = null;
function updateEmbed(force) {
  if (!stage || (!force && activeScene !== "embed")) return;
  const vecs = embedIds.map((id) => Array.from(rowOf(learner, id)));
  const { pts, axes: ax } = C.pca3(vecs, pcaAxes);
  pcaAxes = ax;
  stage.embed.setPoints(pts);
  stage.embed.select(embSel, nearest(embSel, wordIds).map((x) => x[1]));
}
$("embTrain").addEventListener("click", () => {
  if (engine.phase === "running") return engine.pause();
  if (engine.phase === "paused") return engine.start();
  engine.start({ steps: 800, source: "ch3" });
});
$("embReset").addEventListener("click", () => resetWithUndo($("embProg")));
function checkEmbPredict() {}

// ───────────────────────────── Chapter 4: one neuron ─────────────────────────────
const neuron = {
  names: ["sunshine", "free time", "rain"],
  x0: [0.8, 0.6, 0.3], w0: [1.2, 0.9, -1.5], b0: -0.2,
  x: null, w: null, b: 0, target: 1, hist: [], showAll: false, busy: false,
  reset() { this.x = [...this.x0]; this.w = [...this.w0]; this.b = this.b0; this.hist = []; this.sync(); this.render(); $("nHist").innerHTML = ""; },
  calc() { const z = this.x.reduce((s, x, i) => s + x * this.w[i], this.b); return { z, a: Math.tanh(z) }; },
  sliders() {
    const rows = [];
    this.names.forEach((n, i) => rows.push(["x" + i, `${n} (x${"₁₂₃"[i]})`, 0, 1, 0.01, false]));
    this.names.forEach((n, i) => rows.push(["w" + i, `weight w${"₁₂₃"[i]}`, -3, 3, 0.05, true]));
    rows.push(["b", "bias b", -3, 3, 0.05, true]);
    $("nSliders").innerHTML = rows.map(([id, lab, lo, hi, st, signed]) => `<div class="fader"><label for="n_${id}">${lab}</label><output id="n_${id}_o" for="n_${id}"></output><input type="range" id="n_${id}" min="${lo}" max="${hi}" step="${st}"${signed ? " data-signed" : ""}></div>`).join("");
    $("nSliders").addEventListener("input", (e) => {
      const id = e.target.id.slice(2), v = +e.target.value;
      if (id === "b") this.b = v; else this[id[0]][+id[1]] = v;
      this.render();
      if (Math.abs(this.calc().a) < 0.2) markDone("undecided");
    });
  },
  sync() {
    this.x.forEach((v, i) => ($("n_x" + i).value = v));
    this.w.forEach((v, i) => ($("n_w" + i).value = v));
    $("n_b").value = this.b;
    $("nSliders").querySelectorAll("input").forEach(fillRange);
  },
  verdict(a) { return a > 0.2 ? "go to the park" : a < -0.2 ? "stay home" : "can't decide"; },
  render() {
    const { z, a } = this.calc();
    this.x.forEach((v, i) => ($("n_x" + i + "_o").textContent = fmt(v)));
    this.w.forEach((v, i) => ($("n_w" + i + "_o").textContent = fmt(v)));
    $("n_b_o").textContent = fmt(this.b);
    const p = (v) => (v < 0 ? `(${fmt(v)})` : fmt(v));
    const terms = this.x.map((x, i) => `${fmt(x)}×${p(this.w[i])}`).join(" + ");
    $("nEq").innerHTML = `z = ${terms} + ${p(this.b)} = <b>${fmt(z, 3)}</b><br>a = tanh(z) = <b class="${a >= 0 ? "pos" : "neg"}">${fmt(a, 3)}</b> → ${this.verdict(a)}`;
    this.drawTanh(z, a);
    this.renderGrad();
    if (stage) stage.neuron.set({ x: this.x, w: this.w, b: this.b, z, a, names: this.names, verdict: `a = <b>${fmt(a)}</b> · ${this.verdict(a)}` });
    if (activeScene === "neuron") stageMeta(`a = ${fmt(a)} → ${this.verdict(a)}`);
  },
  grads() {
    const { a } = this.calc(), t = this.target;
    const dA = 2 * (a - t), dZ = dA * (1 - a * a);
    return { a, loss: (a - t) ** 2, dA, slope: 1 - a * a, dZ, gw: this.x.map((x) => dZ * x), gb: dZ };
  },
  renderGrad() {
    const g = this.grads(), lr = +$("nLr").value, t = this.target, sub = "₁₂₃";
    const tt = t > 0 ? "1" : "(−1)";
    const line = (i) => `∂loss/∂w${sub[i]} = ${fmt(g.dA, 3)} × ${fmt(g.slope, 3)} × ${fmt(this.x[i])} = <b>${fmt(g.gw[i], 4)}</b> → new w${sub[i]} = ${fmt(this.w[i])} − ${fmt(lr)} × (${fmt(g.gw[i], 4)}) = <b>${fmt(this.w[i] - lr * g.gw[i], 3)}</b>`;
    $("nGrad").innerHTML =
      `loss = (a − ${tt})² = (${fmt(g.a, 3)} − ${tt})² = <b>${fmt(g.loss, 4)}</b><br>` +
      `∂loss/∂a = 2(a − ${tt}) = ${fmt(g.dA, 3)} · tanh slope (1 − a²) = ${fmt(g.slope, 3)}<br>` +
      (this.showAll ? [0, 1, 2].map(line).join("<br>") + `<br>∂loss/∂b = <b>${fmt(g.gb, 4)}</b> → new b = <b>${fmt(this.b - lr * g.gb, 3)}</b>` : line(0)) +
      `<br><button type="button" class="btn link" id="nAll" aria-expanded="${this.showAll}">${this.showAll ? "Show fewer" : "Show all 4 gradients"}</button>`;
    $("nAll").onclick = () => { this.showAll = !this.showAll; this.renderGrad(); };
  },
  async learn(n) {
    if (this.busy) return;
    this.busy = true;
    const lr = +$("nLr").value;
    for (let k = 0; k < n; k++) {
      const g = this.grads();
      this.w = this.w.map((w, i) => Math.max(-3, Math.min(3, w - lr * g.gw[i])));
      this.b = Math.max(-3, Math.min(3, this.b - lr * g.gb));
      const after = this.grads();
      this.hist.unshift(after.loss);
      this.sync();
      this.render();
      $("nHist").innerHTML = this.hist.slice(0, 30).map((l, i) => `<div><span>step ${this.hist.length - i}</span>loss ${fmt(l, 5)}</div>`).join("");
      if (this.target < 0 && after.loss < 0.01) markDone("stay");
      if (n > 1 && !REDUCED) await wait(120);
    }
    this.busy = false;
  },
  drawTanh(z, a) {
    const [x, w, h] = setupCanvas($("tanhChart"));
    const [X, Y] = axes(x, w, h, { l: 34, r: 12, t: 10, b: 22 }, [-4, 4], [-1.1, 1.1], [-4, -2, 0, 2, 4], [-1, 0, 1]);
    x.strokeStyle = css("--ink"); x.lineWidth = 2; x.beginPath();
    for (let v = -4; v <= 4.001; v += 0.05) { const px = X(v), py = Y(Math.tanh(v)); v === -4 ? x.moveTo(px, py) : x.lineTo(px, py); }
    x.stroke();
    x.fillStyle = a >= 0 ? css("--pos") : css("--neg");
    x.beginPath(); x.arc(X(Math.max(-4, Math.min(4, z))), Y(a), 6, 0, 7); x.fill();
    x.fillStyle = css("--muted"); x.textAlign = "left"; x.textBaseline = "top";
    x.fillText("a = tanh(z)", X(-3.9), Y(1.08));
  },
};
neuron.sliders();
$("nReset").addEventListener("click", () => neuron.reset());
$("nStep").addEventListener("click", () => neuron.learn(1));
$("nStep10").addEventListener("click", () => neuron.learn(10));
$("nLr").addEventListener("input", () => { $("nLrOut").textContent = fmt(+$("nLr").value); neuron.renderGrad(); });
seg($("nTarget"), "t", (t) => { neuron.target = +t; neuron.hist = []; $("nHist").innerHTML = ""; neuron.renderGrad(); });

// ───────────────────────────── Chapter 5: the network ─────────────────────────────
const P = [
  ["Embedding table", "E", `${V} × ${D}`, V * D],
  ["Hidden weights", "W₁", `${H} × ${N * D}`, H * N * D],
  ["Hidden biases", "b₁", `${H}`, H],
  ["Output weights", "W₂", `${V} × ${H}`, V * H],
  ["Output biases", "b₂", `${V}`, V],
];
$("paramRows").innerHTML = P.map(([n, s, sh, c]) => `<tr><td>${n} <span class="mono">${s}</span></td><td class="mono">${sh}</td><td class="n">${int(c)}</td></tr>`).join("") + `<tr><td><b>Total</b></td><td></td><td class="n"><b>${int(learner.paramCount)}</b></td></tr>`;
$("factParams").textContent = int(learner.paramCount);
$("pcount").textContent = int(learner.paramCount);
$("cmpParams").textContent = int(learner.paramCount);
$("cmpVocab").textContent = V;
document.querySelectorAll("button.term").forEach((b) => b.addEventListener("click", () => stage?.zoomLayer(+b.dataset.layer)));

const FWD_PRESETS = ["the cow eats", "the cat eats", "the dog chases the", "the girl gives the bird", "the bird sits in the", "the fox drinks", "the boy plays in"];
$("fwdEx").insertAdjacentHTML("beforeend", FWD_PRESETS.slice(0, 5).map((t, i) => `<button type="button" class="btn sm" data-i="${i}">${esc(t)}</button>`).join(""));
const fwd = {
  who: "ref", whoChosen: false, ctx: ctxOf("the cow eats"), step: 4, j: null, extra: null, f: null, prevTop: null, openSlot: -1, playTimer: 0, firesCache: null,
  names: ["Tokens", "Embed", "Hidden", "Scores", "Probs"],
  model() { return this.who === "ref" ? ref : learner; },
  setCtx(ctx) { this.ctx = ctx; syncFwdText(); this.run(true, true); },
  run(animate = true, userChange = false) {
    const m = this.model();
    this.f = m.forward(this.ctx);
    const top = C.topK(this.f.p, 1)[0];
    if (userChange && this.prevTop !== null && top !== this.prevTop && m === ref) markDone("swap");
    if (userChange && this.prevTop !== null && top !== this.prevTop && m === learner && learner.step >= 300) markDone("swap");
    this.prevTop = top;
    let best = 0;
    for (let j = 0; j < H; j++) if (Math.abs(this.f.h[j]) > Math.abs(this.f.h[best])) best = j;
    if (this.j === null || animate) this.j = best;
    this.extra = null;
    this.renderSlots();
    if (animate) this.play(); else { this.toStage(false); this.render(); }
  },
  toStage(animate) {
    if (!stage || !(currentBeat?.dataset.fwd || currentBeat?.dataset.model === "fwd")) return;
    if (stage.net.model !== this.model()) stage.net.setModel(this.model());
    if (!this.f) this.f = this.model().forward(this.ctx);
    stage.net.show(this.f, { from: animate ? 0 : 4 });
    stage.net.setStage(this.step, animate ? 1.6 : 99);
  },
  play() {
    clearInterval(this.playTimer);
    this.step = 0;
    this.render();
    if (stage && (currentBeat?.dataset.fwd || currentBeat?.dataset.model === "fwd")) {
      if (stage.net.model !== this.model()) stage.net.setModel(this.model());
      stage.net.show(this.f, { from: 0 });
      stage.net.setStage(0, 3);
    }
    if (REDUCED) { this.step = 4; stage?.net.setStage(4); this.render(); return; }
    this.playTimer = setInterval(() => {
      this.step++;
      stage?.net.setStage(this.step, 1.6);
      this.render();
      if (this.step >= 4) clearInterval(this.playTimer);
    }, 900);
  },
  renderSlots() {
    $("fwdSlots").innerHTML = this.ctx.map((id, n) => `<button type="button" class="slot" data-n="${n}" aria-expanded="${this.openSlot === n}" aria-label="Word ${n + 1}: ${esc(show(bpe.vocab[id]))}. Change it">${esc(show(bpe.vocab[id]))}<i>▾</i></button>`).join("") + `<span class="arrow">→ ?</span>`;
    const p = $("fwdPicker");
    if (this.openSlot < 0) { p.hidden = true; return; }
    const cur = this.ctx[this.openSlot];
    p.hidden = false;
    p.innerHTML = [START, ...wordIds].map((id) => `<button type="button" class="chip cat" style="--c:${CAT_HEX[categoryOf(bpe.vocab[id])]}" aria-pressed="${id === cur}" data-id="${id}">${esc(show(bpe.vocab[id]))}</button>`).join("");
  },
  highlight(j) {
    const s = $("hStrip");
    if (!s || this.step !== 2) return;
    s.querySelectorAll("button").forEach((b) => b.style.outline = +b.dataset.j === j ? `2px solid var(--pos)` : "");
  },
  fires(j) {
    // which training contexts make neuron j fire most (and least)? computed over all 594 examples
    const m = this.model(), key = m === ref ? "ref" : "l" + learner.step;
    if (!this.firesCache || this.firesCache.key !== key) {
      const seen = new Map();
      for (const e of TR) { const k = e.ctx.join(); if (!seen.has(k)) seen.set(k, { ctx: e.ctx, h: m.forward(e.ctx).h }); }
      this.firesCache = { key, rows: [...seen.values()] };
    }
    const rows = this.firesCache.rows.map((r) => ({ ctx: r.ctx, v: r.h[j] })).sort((a, b) => b.v - a.v);
    return { hi: rows.slice(0, 4), lo: rows.slice(-4).reverse() };
  },
  render() {
    const f = this.f;
    if (!f) return;
    if (this.j === null) { let best = 0; for (let j = 0; j < H; j++) if (Math.abs(f.h[j]) > Math.abs(f.h[best])) best = j; this.j = best; }
    $("fwdSteps").innerHTML = this.names.map((n, i) => `<button type="button" role="tab" id="fwdTab${i}" aria-controls="fwdBody" aria-selected="${i === this.step}" tabindex="${i === this.step ? 0 : -1}" class="${i < this.step ? "done" : ""}" data-i="${i}"><span>${i + 1}</span>${n}</button>`).join("");
    $("fwdBody").setAttribute("aria-labelledby", "fwdTab" + this.step);
    $("fwdBack").disabled = this.step === 0;
    $("fwdNextB").disabled = this.step === 4;
    const m = this.model(), ctx = f.ctx, b = $("fwdBody");
    const whoTxt = m === ref ? "pre-trained copy" : `your model at step ${int(learner.step)}`;
    $("fwdNote").textContent = m === learner && learner.step < 300 ? `Untrained (step ${int(learner.step)}): its numbers are all close to 0, so the network stays dark and every token gets about ${(100 / V).toFixed(1)}%. Train it in Chapter 6 and come back.` : `Live numbers from the ${whoTxt}.`;
    if (this.step === 0) {
      b.innerHTML = `<p>The tokenizer from Chapter 2 breaks the text into tokens and looks up each one's ID. In the 3D view the text cracks into tiles and each tile drops onto its row. From here on, the model only sees numbers.</p><div class="chips">${ctx.map((id) => `<span class="chip tok" style="--h:${Math.round(((id * 0.61803398875) % 1) * 360)}">${esc(show(bpe.vocab[id]))}<small>ID ${id}</small></span>`).join("")}</div><p class="note">Sentences start with the special <span class="mono">[start]</span> token, so there are always 3 tokens to read.</p>`;
    } else if (this.step === 1) {
      b.innerHTML = `<p>Each ID picks its row of 10 numbers from the embedding table. The three rows sit side by side: <b>30 input numbers</b>, x₁ to x₃₀.</p>` +
        ctx.map((id, n) => `<div class="note mono">${esc(show(bpe.vocab[id]))} → row ${id}</div><div class="well" style="padding:6px"><div class="vec">${Array.from({ length: D }, (_, d) => { const v = f.x[n * D + d]; return `<div style="${cellStyle(v, 1.5)}">${fmt(v)}</div>`; }).join("")}</div></div>`).join("");
    } else if (this.step === 2) {
      const j = this.j, I = N * D;
      let sum = 0;
      const terms = [];
      for (let i = 0; i < I; i++) { const w = m.W1[j * I + i], p = f.x[i] * w; sum += p; terms.push(`<div>x${i + 1}×w = ${fmt(f.x[i])}×${fmt(w)} = <span class="${p >= 0 ? "pos" : "neg"}">${fmt(p, 3)}</span></div>`); }
      const scale = Math.max(...Array.from(f.h, Math.abs), 0.3);
      const fi = this.fires(j);
      const chip = (r) => `<span class="chip"><span>${esc(ctxLabel(r.ctx))}</span><small>${fmt(r.v, 2)}</small></span>`;
      b.innerHTML = `<p>All 48 hidden neurons work at once, each like the neuron in Chapter 4. Click a square, or a neuron in the 3D view. Showing neuron <b>#${j + 1}</b>.</p>
        <div class="strip" id="hStrip" role="group" aria-label="Hidden neurons">${Array.from(f.h, (v, k) => `<button type="button" aria-pressed="${k === j}" data-j="${k}" aria-label="Neuron ${k + 1}: ${fmt(v, 2)}" title="neuron ${k + 1}: ${fmt(v, 3)}" style="${cellStyle(v, scale)}"></button>`).join("")}</div>
        <div class="well terms">${terms.join("")}</div>
        <div class="well eq">sum of the 30 products = ${fmt(sum, 4)}<br>+ bias b = ${fmt(m.b1[j], 4)} → z = ${fmt(f.z1[j], 4)}<br>h = tanh(z) = <b class="${f.h[j] >= 0 ? "pos" : "neg"}">${fmt(f.h[j], 4)}</b></div>
        <p class="note">Neuron #${j + 1} fires most for: </p><div class="chips">${fi.hi.map(chip).join("")}</div>
        <p class="note">…and most against:</p><div class="chips">${fi.lo.map(chip).join("")}</div>`;
      $("hStrip").addEventListener("click", (e) => { const d = e.target.closest("[data-j]"); if (d) { this.j = +d.dataset.j; markDone("inspect"); this.render(); stage?.net.focus({ layer: "hid", index: this.j }, true); stage?.placeTip({ layer: "hid", index: this.j }); } });
      $("hStrip").addEventListener("keydown", (e) => {
        const d = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 24, ArrowUp: -24 }[e.key];
        if (!d) return;
        e.preventDefault();
        this.j = (this.j + d + H) % H; this.render(); $("hStrip").querySelector(`[data-j="${this.j}"]`).focus();
      });
    } else if (this.step === 3) {
      const top = C.topK(f.p, 5), k0 = top[0];
      const terms = [];
      for (let j = 0; j < H; j++) { const w = m.W2[k0 * H + j], p = f.h[j] * w; terms.push(`<div>h${j + 1}×w = ${fmt(f.h[j])}×${fmt(w)} = <span class="${p >= 0 ? "pos" : "neg"}">${fmt(p, 3)}</span></div>`); }
      b.innerHTML = `<p>Each of the ${V} output neurons multiplies the 48 hidden values by its own 48 weights and adds them up. The result is a <b>score</b> (also called a logit). Here is the full sum for the top-scoring token, <span class="mono">${esc(show(bpe.vocab[k0]))}</span>:</p>
        <div class="well terms">${terms.join("")}</div>
        <div class="well eq">score(${esc(show(bpe.vocab[k0]))}) = sum + bias ${fmt(m.b2[k0], 3)} = <b>${fmt(f.logits[k0], 3)}</b></div>
        <div class="tbl-wrap"><table><thead><tr><th>Token</th><th class="n">Score</th></tr></thead><tbody>${top.map((k) => `<tr><td class="mono">${esc(show(bpe.vocab[k]))}</td><td class="n">${fmt(f.logits[k], 3)}</td></tr>`).join("")}</tbody></table></div>`;
    } else {
      const top = C.topK(f.p, 5);
      if (this.extra !== null && !top.includes(this.extra)) top.push(this.extra);
      let S = 0;
      for (const l of f.logits) S += Math.exp(l);
      const ex = (v) => { const e = Math.exp(v); return e >= 1000 ? int(e) : e.toPrecision(3); };
      const Sx = S >= 1000 ? int(S) : S.toPrecision(4);
      b.innerHTML = `<p><b>Softmax</b> turns scores into probabilities: raise <i>e</i> (2.718…) to the power of each score so everything is positive, then divide by the total so the results add up to 1.</p>
        <div class="tbl-wrap"><table><thead><tr><th>Token</th><th class="n">Score</th><th class="n math">e<sup>score</sup></th><th class="n math">÷ ${Sx}</th></tr></thead><tbody>
        ${top.map((k) => `<tr class="${k === this.extra ? "now" : ""}"><td class="mono">${esc(show(bpe.vocab[k]))}</td><td class="n">${fmt(f.logits[k], 3)}</td><td class="n">${ex(f.logits[k])}</td><td class="n"><b>${(f.p[k] * 100).toFixed(2)}%</b></td></tr>`).join("")}
        </tbody></table></div>
        <p class="note">${Sx} is the sum of e<sup>score</sup> over all ${V} tokens. The other tokens share the remaining ${((1 - top.reduce((s, k) => s + f.p[k], 0)) * 100).toFixed(2)}%.</p>`;
    }
  },
};
$("fwdSteps").addEventListener("click", (e) => {
  const t = e.target.closest("button");
  if (!t) return;
  clearInterval(fwd.playTimer);
  fwd.step = +t.dataset.i;
  fwd.render();
  stage?.net.setStage(fwd.step);
});
$("fwdSteps").addEventListener("keydown", (e) => {
  const k = { ArrowRight: 1, ArrowLeft: -1, Home: -9, End: 9 }[e.key];
  if (!k) return;
  e.preventDefault();
  clearInterval(fwd.playTimer);
  fwd.step = Math.max(0, Math.min(4, fwd.step + k));
  fwd.render();
  stage?.net.setStage(fwd.step);
  $("fwdTab" + fwd.step).focus();
});
$("fwdBack").addEventListener("click", () => { clearInterval(fwd.playTimer); fwd.step = Math.max(0, fwd.step - 1); fwd.render(); stage?.net.setStage(fwd.step); });
$("fwdNextB").addEventListener("click", () => { clearInterval(fwd.playTimer); fwd.step = Math.min(4, fwd.step + 1); fwd.render(); stage?.net.setStage(fwd.step); });
$("fwdRun").addEventListener("click", () => fwd.play());
$("fwdEx").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; fwd.ctx = ctxOf(FWD_PRESETS[+b.dataset.i]); fwd.openSlot = -1; syncFwdText(); fwd.run(true, true); });
$("fwdSlots").addEventListener("click", (e) => { const b = e.target.closest("button.slot"); if (!b) return; const n = +b.dataset.n; fwd.openSlot = fwd.openSlot === n ? -1 : n; fwd.renderSlots(); $("fwdPicker").querySelector('[aria-pressed="true"]')?.focus(); });
$("fwdPicker").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  const ctx = fwd.ctx.slice();
  ctx[fwd.openSlot] = +b.dataset.id;
  fwd.openSlot = -1;
  fwd.setCtx(ctx);
});
// type your own text: the tokenizer breaks it up, the model reads the last 3 tokens
function renderFwdTokens() {
  const toks = bpe.encode($("fwdText").value);
  $("fwdTokens").innerHTML = toks.length
    ? toks.map((t, i) => `<span class="chip tok${i < toks.length - N ? " gone" : ""}" style="--h:${Math.round(((t.id * 0.61803398875) % 1) * 360)}" title="${i < toks.length - N ? "Out of the model's 3-token view" : "The model reads this"}">${esc(show(t.text))}<small>${t.id}</small></span>`).join("")
    : `<span class="note">Empty: the model sees three [start] tokens and guesses the first word.</span>`;
}
let fwdTextTimer = 0;
$("fwdText").addEventListener("input", () => {
  renderFwdTokens();
  clearTimeout(fwdTextTimer);
  fwdTextTimer = setTimeout(() => {
    fwd.ctx = [...Array(N).fill(START), ...bpe.ids($("fwdText").value)].slice(-N);
    fwd.openSlot = -1;
    fwd.run(true, true);
  }, 350);
});
function syncFwdText() {
  $("fwdText").value = fwd.ctx.map((id) => bpe.vocab[id]).filter((t) => !t.startsWith("[")).join("").trim();
  renderFwdTokens();
}
seg($("fwdWho"), "who", (who) => { fwd.who = who; fwd.whoChosen = true; fwd.firesCache = null; fwd.run(false); if (currentBeat?.dataset.model === "fwd") activate(currentBeat, true); });

// ───────────────────────────── Chapter 6: loss calculator ─────────────────────────────
function renderLossCalc() {
  const lp = +$("lossP").value, p = Math.pow(10, lp), L = -Math.log(p);
  $("lossPOut").textContent = p >= 0.1 ? p.toFixed(2) : p >= 0.01 ? p.toFixed(3) : p.toFixed(4);
  $("lossRead").textContent = `p = ${$("lossPOut").textContent} → loss = −ln(p) = ${L.toFixed(3)}`;
  const [x, w, h] = setupCanvas($("lossCurve"));
  const [X, Y] = axes(x, w, h, { l: 30, r: 12, t: 10, b: 22 }, [-3, 0], [0, 7], [-3, -2, -1, 0], [0, 2, 4, 6], (v) => String(Math.pow(10, v)));
  x.strokeStyle = css("--ink"); x.lineWidth = 2; x.beginPath();
  for (let v = -3; v <= 0.0001; v += 0.02) { const px = X(v), py = Y(-Math.log(Math.pow(10, v))); v === -3 ? x.moveTo(px, py) : x.lineTo(px, py); }
  x.stroke();
  const mark = (pp, label, alignRight) => {
    const v = Math.log10(pp);
    x.fillStyle = css("--muted"); x.beginPath(); x.arc(X(v), Y(-Math.log(pp)), 3, 0, 7); x.fill();
    x.textAlign = alignRight ? "right" : "left"; x.textBaseline = "bottom"; x.font = `11px ${css("--sans")}`;
    x.fillText(label, X(v) + (alignRight ? -6 : 6), Y(-Math.log(pp)) - 4);
  };
  mark(1 / V, `even guess over ${V} tokens: ${Math.log(V).toFixed(2)}`, false);
  mark(0.9, "90% sure and right: 0.105", true);
  x.fillStyle = css("--pos"); x.beginPath(); x.arc(X(lp), Y(L), 6, 0, 7); x.fill();
}
$("lossP").addEventListener("input", renderLossCalc);

// ───────────────────────────── Chapter 6: landscape ─────────────────────────────
const land = {
  path: [[-2.6, 2.8]], timer: null, runStartLen: 1,
  reset(wb, fromPick) { this.stop(); this.path = [wb || [-2.6, 2.8]]; this.runStartLen = 1; $("landResult").textContent = fromPick ? "Ball placed. Press Roll downhill." : ""; this.render(); },
  step() {
    const [w, b] = this.path[this.path.length - 1];
    const [gw, gb] = C.toyGrad(w, b), lr = +$("landLr").value;
    this.path.push([Math.max(-4, Math.min(4, w - lr * gw)), Math.max(-4, Math.min(4, b - lr * gb))]);
    this.render();
    return Math.hypot(gw, gb);
  },
  run() {
    if (this.timer) return this.stop();
    $("landRun").innerHTML = `${ICON_PAUSE}Pause`;
    let n = 0;
    const steps0 = this.path.length - 1;
    this.timer = setInterval(() => {
      const g = this.step();
      n++;
      const [w, b] = this.path[this.path.length - 1], L = C.toyLoss(w, b);
      if (L < 0.01) {
        const used = this.path.length - 1;
        $("landResult").textContent = `Reached the bottom (loss below 0.01) after ${used} steps.`;
        if (used < 100) markDone("ball");
        this.stop();
      } else if (n >= 400 || g < 1e-5) { $("landResult").textContent = `Stopped after ${this.path.length - 1} steps with loss ${L.toFixed(3)}.`; this.stop(); }
    }, REDUCED ? 15 : 55);
  },
  stop() { clearInterval(this.timer); this.timer = null; $("landRun").innerHTML = `${ICON_PLAY}Roll downhill`; },
  render() {
    const [w, b] = this.path[this.path.length - 1];
    const [gw, gb] = C.toyGrad(w, b);
    $("landNote").textContent = `step ${this.path.length - 1} · w = ${fmt(w, 3)} · b = ${fmt(b, 3)} · loss = ${fmt(C.toyLoss(w, b), 4)} · gradient = (${fmt(gw, 3)}, ${fmt(gb, 3)})`;
    if (stage) stage.land.setTrail(this.path);
    if (activeScene === "land") stageMeta(`step ${this.path.length - 1} · loss ${fmt(C.toyLoss(w, b), 4)}`);
    const [x, W, Hh] = setupCanvas($("toyChart"));
    const [X, Y] = axes(x, W, Hh, { l: 34, r: 12, t: 10, b: 22 }, [-2.6, 2.6], [-1.15, 1.15], [-2, -1, 0, 1, 2], [-1, 0, 1]);
    x.strokeStyle = css("--pos"); x.lineWidth = 2.2; x.beginPath();
    for (let v = -2.6; v <= 2.601; v += 0.04) { const px = X(v), py = Y(Math.tanh(w * v + b)); v === -2.6 ? x.moveTo(px, py) : x.lineTo(px, py); }
    x.stroke();
    x.fillStyle = css("--ink");
    for (const p of C.TOY) { x.beginPath(); x.arc(X(p.x), Y(p.y), 4.5, 0, 7); x.fill(); }
    x.fillStyle = css("--muted"); x.textAlign = "right"; x.textBaseline = "bottom";
    x.fillText(`curve: tanh(${fmt(w)}·x + ${fmt(b)})`, X(2.55), Y(-0.75));
  },
};
$("landRun").addEventListener("click", () => land.run());
$("landStep").addEventListener("click", () => { land.stop(); land.step(); });
$("landReset").addEventListener("click", () => land.reset());
$("landLr").addEventListener("input", () => ($("landLrOut").textContent = fmt(+$("landLr").value)));
$("landBig").addEventListener("click", () => { $("landLr").value = 3; $("landLrOut").textContent = "3.00"; fillRange($("landLr")); land.reset(); land.run(); });
$("landFlat").addEventListener("click", () => { $("landLr").value = 0.6; $("landLrOut").textContent = "0.60"; fillRange($("landLr")); land.reset([3, -3]); land.run(); });

// ───────────────────────────── Chapter 6: training the real model ─────────────────────────────
const PROBES = ["the girl drinks", "the cat eats", "the cow eats", "the dog chases the", "the bird sits in the"];
$("trProbe").innerHTML = PROBES.map((t, i) => `<option value="${i}">${esc(t)} → ?</option>`).join("");
const probeCtx = () => ctxOf(PROBES[+$("trProbe").value || 0]);
$("trProbe").addEventListener("change", () => { renderProbe(); if (stage && activeScene === "net" && stage.net.model === learner) stage.net.show(learner.forward(probeCtx()), { from: 0 }); });
function renderProbe() {
  const f = learner.forward(probeCtx()), top = C.topK(f.p, 3);
  bars($("trProbeBars"), top.map((k) => ({ label: show(bpe.vocab[k]), p: f.p[k] })));
}
seg($("trBudget"), "b", (b) => { engine.budget = +b; onEngine("budget"); });
$("trEarly").addEventListener("change", () => { engine.earlyStop = $("trEarly").checked; if (engine.run) engine.run.earlyStop = engine.earlyStop; drawLoss(); });
seg($("trSpeed"), "s", (s) => (engine.speed = +s));
$("trRun").addEventListener("click", () => engine.toggle());
$("hudTrain").addEventListener("click", () => engine.toggle({ source: "hud" }));
$("trStep").addEventListener("click", () => engine.stepOnce());
$("trReset").addEventListener("click", () => resetWithUndo($("trUndo")));
$("trLr").addEventListener("input", () => { engine.setLr(+$("trLr").value); $("trLrOut").textContent = fmt(engine.lr); oneStep.render(); });

function checkTrPredict() {}

function runLabel() {
  const r = engine.run, s = learner.step;
  if (engine.phase === "running") return `Pause · ${int(s - r.start)} / ${int(r.end - r.start)}`;
  if (engine.phase === "paused") return `Resume · ${int(s - r.start)} / ${int(r.end - r.start)}`;
  if (s >= MAX_TOTAL) return `Reached ${int(MAX_TOTAL)} steps`;
  return s === 0 ? `Train ${int(engine.budget)} steps` : `Train ${int(Math.min(engine.budget, MAX_TOTAL - s))} more`;
}
function runFrac() { const r = engine.run; return r ? (learner.step - r.start) / Math.max(1, r.end - r.start) : engine.lastRun ? 1 : 0; }
function updateTrainDock() {
  const show = !!currentBeat?.dataset.train || engine.phase !== "ready";
  $("dkTrain").hidden = !show;
  if (!show) return;
  const r = engine.run;
  $("dkTrainBtn").innerHTML = engine.phase === "running" ? `${ICON_PAUSE}Pause` : `${ICON_PLAY}${engine.phase === "paused" ? "Resume" : learner.step ? "Train more" : "Train"}`;
  $("dkCount").textContent = r ? `${int(learner.step - r.start)} / ${int(r.end - r.start)}` : `step ${int(learner.step)}`;
  $("dkBar").style.width = (runFrac() * 100).toFixed(1) + "%";
  $("dkMeter").classList.toggle("done", !r && !!engine.lastRun);
}
let toastTimer = 0, trPanelVisible = false;
new IntersectionObserver((es) => (trPanelVisible = es[0].isIntersecting)).observe($("trPanel"));
function toast(html) {
  const t = $("toast");
  t.innerHTML = `<span>${html}</span><button type="button" aria-label="Close">×</button>`;
  t.hidden = false;
  t.querySelector("button").onclick = () => (t.hidden = true);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 7000);
}
function renderSummary() {
  const el = $("trSummary"), r = engine.lastRun;
  if (!r) { el.hidden = true; return; }
  el.hidden = false;
  const ep = (seen) => (seen / TR.length).toFixed(1);
  let html = "", btns = "";
  if (r.reason === "early") {
    html = r.restored
      ? `<p><b>Stopped early</b> at step ${int(r.stoppedAt)} (epoch ${ep(r.seenAtStop)}). The held-out loss was lowest at step ${int(r.best.step)} (${fmt(r.best.val, 3)}) and did not improve for the next ${PATIENCE} steps, so your model was rolled back to step ${int(r.best.step)}.</p>`
      : `<p>You kept the step-${int(r.stoppedAt)} weights. Held-out loss there is ${fmt(engine.lastVal, 3)}; its best was ${fmt(r.best.val, 3)} at step ${int(r.best.step)}.</p>`;
    btns = r.restored ? `<button class="btn sm" type="button" data-a="keep">Keep step ${int(r.stoppedAt)} instead</button>` : `<button class="btn sm" type="button" data-a="best">Roll back to step ${int(r.best.step)}</button>`;
    btns += `<button class="btn sm" type="button" data-a="more">Train 1,000 more without early stopping</button>`;
  } else if (r.reason === "budget") {
    html = `<p><b>Used the full run</b> of ${int(r.end - r.start)} steps (now epoch ${ep(r.seenAtStop)}). Held-out loss is ${fmt(r.finalVal, 3)}; its best was ${fmt(r.best.val, 3)} at step ${int(r.best.step)}.</p>`;
    if (r.best.step < r.stoppedAt && !r.restored) btns += `<button class="btn sm" type="button" data-a="best">Roll back to step ${int(r.best.step)}</button>`;
    if (r.restored) btns += `<button class="btn sm" type="button" data-a="keep">Keep step ${int(r.stoppedAt)} instead</button>`;
  } else {
    html = `<p>Your model has trained for ${int(MAX_TOTAL)} steps (epoch ${ep(r.seenAtStop)}). Training loss has flattened near ${fmt(r.finalTrain)}, so more steps would only memorise. Reset to try other settings.</p>`;
  }
  el.innerHTML = html + (btns ? `<div class="row">${btns}</div>` : "");
}
$("trSummary").addEventListener("click", (e) => {
  const a = e.target.closest("button")?.dataset.a;
  if (a === "keep") engine.keepFinal();
  else if (a === "best") engine.restoreBest();
  else if (a === "more") { $("trEarly").checked = false; engine.earlyStop = false; segSet($("trBudget"), "b", 1000); engine.budget = 1000; engine.start({ steps: 1000, earlyStop: false }); }
});

function drawLoss() {
  const [x, w, h] = setupCanvas($("lossChart"));
  const hist = engine.history, ghost = engine.ghost, r = engine.run, lr = engine.lastRun;
  const last = hist[hist.length - 1];
  const end = r ? r.end : lr ? lr.stoppedAt : Math.min(learner.step + engine.budget, MAX_TOTAL);
  const maxStep = Math.max(500, end, last?.step || 0, ...ghost.map((p) => p.step));
  const xt = niceTicks(maxStep, 4);
  const xmax = xt[xt.length - 1];
  const [X, Y] = axes(x, w, h, { l: 30, r: 14, t: 14, b: 24 }, [0, xmax], [0, 5.5], xt, [0, 1, 2, 3, 4, 5]);
  const ink = css("--ink"), s2 = css("--s2"), muted = css("--muted");
  // run end
  if (r || !lr) {
    x.strokeStyle = muted; x.setLineDash([2, 3]); x.beginPath(); x.moveTo(X(end), Y(0)); x.lineTo(X(end), Y(5.5)); x.stroke(); x.setLineDash([]);
    x.fillStyle = muted; x.font = `11px ${css("--sans")}`; x.textAlign = X(end) > w - 90 ? "right" : "left"; x.textBaseline = "top";
    x.fillText(`run ends · ${int(end)}`, X(end) + (x.textAlign === "right" ? -4 : 4), Y(5.4));
  }
  // patience band
  if (r && r.earlyStop && isFinite(engine.best.val)) {
    const a = Math.max(engine.best.step, r.patienceFrom), b = Math.min(a + PATIENCE, xmax);
    x.fillStyle = s2; x.globalAlpha = 0.12; x.fillRect(X(a), Y(5.5), X(b) - X(a), Y(0) - Y(5.5)); x.globalAlpha = 1;
    if (X(b) - X(a) > 80) { x.fillStyle = s2; x.font = `10.5px ${css("--sans")}`; x.textAlign = "center"; x.textBaseline = "top"; x.fillText(`patience ${PATIENCE}`, (X(a) + X(b)) / 2, Y(5.2)); }
  }
  const line = (pts, key, color, dash, alpha = 1) => {
    const p = pts.filter((q) => q[key] !== undefined);
    if (p.length < 1) return;
    x.globalAlpha = alpha; x.strokeStyle = color; x.lineWidth = 2; x.setLineDash(dash || []);
    x.beginPath(); p.forEach((q, i) => (i ? x.lineTo(X(q.step), Y(q[key])) : x.moveTo(X(q.step), Y(q[key])))); x.stroke();
    x.setLineDash([]); x.globalAlpha = 1;
  };
  if (ghost.length) { const g0 = hist[hist.length - 1]; line([g0, ...ghost], "val", s2, [4, 3], 0.4); line([g0, ...ghost], "train", ink, [4, 3], 0.35); }
  line(hist, "train", ink);
  line(hist, "val", s2, [6, 4]);
  // learning-rate changes
  for (const m of engine.marks.filter((q) => q.kind === "lr")) {
    x.strokeStyle = muted; x.setLineDash([1, 3]); x.beginPath(); x.moveTo(X(m.step), Y(0)); x.lineTo(X(m.step), Y(5.5)); x.stroke(); x.setLineDash([]);
    x.fillStyle = muted; x.font = `10px ${css("--mono")}`; x.textAlign = "left"; x.textBaseline = "bottom"; x.fillText(`lr ${fmt(m.from)}→${fmt(m.to)}`, X(m.step) + 3, Y(0.15));
  }
  // best checkpoint
  const best = engine.best;
  if (isFinite(best.val) && hist.length > 3) {
    x.strokeStyle = s2; x.lineWidth = 2; x.fillStyle = css("--paper");
    x.beginPath(); x.arc(X(best.step), Y(best.val), 5, 0, 7); x.fill(); x.stroke();
    if (last && last.val > best.val + 0.05 && last.step > best.step + 200) {
      x.fillStyle = s2; x.font = `11.5px ${css("--sans")}`; x.textAlign = "left"; x.textBaseline = "bottom";
      x.fillText("held-out loss rising: overfitting", Math.min(X(best.step) + 8, w - 200), Y(best.val) - 18);
    }
  }
}

function renderProgress() {
  const r = engine.run, lr = engine.lastRun, el = $("trProg");
  const frac = runFrac();
  el.querySelector("i").style.width = (frac * 100).toFixed(1) + "%";
  el.classList.toggle("done", !r && !!lr);
  el.setAttribute("aria-valuemax", r ? r.end - r.start : engine.budget);
  el.setAttribute("aria-valuenow", r ? learner.step - r.start : 0);
  let text;
  if (r) {
    const rate = engine.rate(), left = rate > 0 ? Math.ceil((r.end - learner.step) / rate) : null;
    text = `step ${int(learner.step - r.start)} / ${int(r.end - r.start)} · epoch ${engine.epoch.toFixed(1)}${engine.phase === "running" && left !== null ? ` · about ${left} s left` : engine.phase === "paused" ? " · paused" : ""}`;
  } else if (lr) text = lr.reason === "early" ? `Stopped early at step ${int(lr.stoppedAt)}${lr.restored ? ` · rolled back to ${int(lr.best.step)}` : ""}` : lr.reason === "budget" ? `Finished ${int(lr.end - lr.start)} steps` : `Reached ${int(MAX_TOTAL)} steps`;
  else text = `Ready: ${int(Math.min(engine.budget, MAX_TOTAL - learner.step))} steps (about ${Math.round(Math.min(engine.budget, MAX_TOTAL - learner.step) / STEPS_PER_EPOCH)} epochs)`;
  $("trProgText").textContent = text;
  el.setAttribute("aria-valuetext", text);
  // header progress
  const hp = $("hudProg");
  hp.querySelector("i").style.width = (r ? frac * 100 : 0).toFixed(1) + "%";
}

let lastFwdRefresh = 0, prevPhase = "ready";
function onEngine(kind) {
  const s = learner.step, r = engine.run;
  $("hudStep").textContent = r ? `${int(s)} / ${int(r.end)}` : int(s);
  $("hudTrain").innerHTML = engine.phase === "running" ? "Pause" : engine.phase === "paused" ? "Resume" : s >= MAX_TOTAL ? "Max" : s ? "Train more" : "Train";
  $("hudTrain").disabled = s >= MAX_TOTAL && engine.phase === "ready";
  const runBtn = $("trRun");
  runBtn.innerHTML = `${engine.phase === "running" ? ICON_PAUSE : ICON_PLAY}${runLabel()}<span class="bar"></span>`;
  runBtn.style.setProperty("--p", (runFrac() * 100).toFixed(1) + "%");
  runBtn.disabled = s >= MAX_TOTAL && engine.phase === "ready";
  $("trBudget").setAttribute("aria-disabled", String(engine.phase !== "ready"));
  $("trBudget").title = engine.phase !== "ready" ? "Applies to the next run" : "";
  // chapter 3 button mirrors the same run
  const eb = $("embTrain");
  eb.innerHTML = (engine.phase === "running" ? "Pause" : engine.phase === "paused" ? "Resume" : s ? "Train 800 more" : "Train 800 steps and watch") + `<span class="bar"></span>`;
  eb.style.setProperty("--p", (r ? runFrac() * 100 : 0).toFixed(1) + "%");
  if (!$("embProg").querySelector("button")) $("embProg").textContent = r ? (r.source === "ch3" ? `step ${int(s - r.start)} / ${int(r.end - r.start)}` : `Training from ${r.source === "hud" || r.source === "dock" ? "the top bar" : "Chapter 6"}: step ${int(s - r.start)} / ${int(r.end - r.start)}`) : s ? `Your model is at step ${int(s)}.` : "";
  $("kStep").textContent = int(s);
  $("kEpoch").textContent = engine.epoch.toFixed(1);
  $("kTrain").textContent = fmt(engine.lastTrain, 3);
  $("kVal").firstChild.textContent = fmt(engine.lastVal, 3);
  $("kBest").textContent = isFinite(engine.best.val) && s > 0 ? `best ${fmt(engine.best.val, 3)} · step ${int(engine.best.step)}` : "";
  $("trSamples").innerHTML = engine.samples.length ? engine.samples.slice(0, 40).map((x) => `<div><span>step ${int(x.step)}</span>${esc(x.text)}</div>`).join("") : `<div class="note">Start training to see samples.</div>`;
  $("osApply").disabled = engine.phase === "running";
  $("osWhy").textContent = engine.phase === "running" ? "Pause training to apply a single step." : "";
  $("fwdWhoStep").textContent = `step ${int(s)}`;
  $("genWhoStep").textContent = `step ${int(s)}`;
  drawLoss();
  renderProgress();
  updateTrainDock();
  renderEmb();
  renderProbe();
  if ($("oneStep").open) oneStep.render();
  if (!isNaN(engine.lastVal) && engine.lastVal < 1.3 && s > 0) markDone("val130");
  if (activeScene === "embed") { updateEmbed(); stageMeta(`your model · step ${int(s)} · 10 numbers per word, drawn in 3D`); }
  if (activeScene === "net" && stage?.net.model === learner) metaForNet();
  const now = performance.now();
  if (fwd.who === "learner" && now - lastFwdRefresh > 500 && (kind === "done" || kind === "reset" || kind === "restore" || kind === "step")) { lastFwdRefresh = now; fwd.f = learner.forward(fwd.ctx); fwd.render(); }
  if (kind === "reset" || kind === "restore") { pcaAxes = null; if (stage) { stage.net.refreshWeights(true); if (stage.net.model === learner && activeScene === "net") stage.net.show(learner.forward(currentBeat?.dataset.fwd ? fwd.ctx : probeCtx()), { from: 4 }); } renderSummary(); }
  if (kind === "run") { renderSummary(); if (engine.phase === "running") { $("trSummary").hidden = true; announce(`Training started: ${int(r.end - r.start)} steps.`); } }
  if (kind === "done") {
    const lr = engine.lastRun;
    renderSummary();
    if (lr.reason === "early") markDone("early");
    const msg = lr.reason === "early" ? `Stopped early at step ${int(lr.stoppedAt)} and rolled back to step ${int(lr.best.step)} (best held-out loss ${fmt(lr.best.val, 3)}).` : lr.reason === "budget" ? `Finished ${int(lr.end - lr.start)} steps. Held-out loss ${fmt(lr.finalVal, 3)}.` : `Reached ${int(MAX_TOTAL)} steps.`;
    announce(msg);
    if (!trPanelVisible) toast(`${esc(msg)} <a href="#trPanel">See the chart</a>`);
    $("hudProg").classList.add("done");
    setTimeout(() => $("hudProg").classList.remove("done"), 1500);
    if (lr.source === "ch3" || learner.step >= 800) checkEmbPredict();
    checkTrPredict();
    if (stage && activeScene === "net" && stage.net.model === learner) stage.net.show(learner.forward(currentBeat?.dataset.fwd ? fwd.ctx : probeCtx()), { from: 0 });
  }
  if (prevPhase === "running" && engine.phase === "paused" && stage && activeScene === "net" && stage.net.model === learner) stage.net.show(learner.forward(probeCtx()), { from: 0 });
  prevPhase = engine.phase;
  gen.warn();
  updateLiveTasks();
}
engine.on(onEngine);

// one real training step, worked through with numbers
const OS_EX = [["the cat eats", " fish"], ["the cow eats", " grass"], ["the bird eats", " seeds"], ["the dog chases the", " ball"], ["the girl drinks", " juice"], ["the cow sleeps in the", " barn"], ["the fox is", " red"]]
  .map(([t, y]) => ({ ctx: ctxOf(t), y: bpe.id.get(y) })).filter((e) => e.y !== undefined);
$("osEx").innerHTML = OS_EX.map((e, i) => `<option value="${i}">${esc(e.ctx.map((id) => bpe.vocab[id]).filter((t) => !t.startsWith("[")).join("").trim())} → ${esc(bpe.vocab[e.y].trim())}</option>`).join("");
const oneStep = {
  clone: null,
  render() {
    const e = OS_EX[+$("osEx").value], lr = engine.lr;
    const f = learner.forward(e.ctx), y = e.y, py = f.p[y], L = -Math.log(py);
    let j = 0;
    for (let k = 0; k < H; k++) if (Math.abs(f.h[k]) > Math.abs(f.h[j])) j = k;
    const dly = py - 1, w2 = learner.W2[y * H + j], g2 = dly * f.h[j];
    let dh = 0;
    for (let k = 0; k < V; k++) dh += (f.p[k] - (k === y ? 1 : 0)) * learner.W2[k * H + j];
    const slope = 1 - f.h[j] * f.h[j], dz = dh * slope, I = N * D;
    let i = 0;
    for (let q = 0; q < I; q++) if (Math.abs(f.x[q]) > Math.abs(f.x[i])) i = q;
    const w1 = learner.W1[j * I + i], g1 = dz * f.x[i];
    this.clone = learner.clone();
    this.clone.zeroGrad();
    this.clone.accumulate(e.ctx, y, 1);
    this.clone.applyGrad(lr);
    const pa = this.clone.forward(e.ctx).p[y];
    const others = C.topK(f.p, 4).filter((k) => k !== y).slice(0, 3);
    const yt = esc(show(bpe.vocab[y]));
    $("osBody").innerHTML = `
      <p><b>1. Forward.</b> Your model (step ${int(learner.step)}) gives the right answer <span class="mono">${yt}</span> a probability of <b>${(py * 100).toFixed(2)}%</b>. Loss = −ln(${py.toPrecision(4)}) = <b>${fmt(L, 4)}</b>.</p>
      <p><b>2. Gradient at the output.</b> ∂L/∂score = p − (1 for the right token, otherwise 0):</p>
      <div class="well eq">${yt}: ${py.toFixed(4)} − 1 = <b class="neg">${fmt(dly, 4)}</b> (negative, so this score goes up)<br>${others.map((k) => `${esc(show(bpe.vocab[k]))}: ${f.p[k].toFixed(4)} − 0 = <b class="pos">${f.p[k].toFixed(4)}</b>`).join("<br>")}${others.length ? " (positive, so these go down)" : ""}</div>
      <p><b>3. One output weight.</b> The weight from hidden neuron #${j + 1} (h = ${fmt(f.h[j], 4)}) to <span class="mono">${yt}</span>:</p>
      <div class="well eq">∂L/∂w = (∂L/∂score) × h = ${fmt(dly, 4)} × ${fmt(f.h[j], 4)} = <b>${fmt(g2, 5)}</b><br>new w = ${fmt(w2, 5)} − ${fmt(lr)} × (${fmt(g2, 5)}) = <b>${fmt(w2 - lr * g2, 5)}</b></div>
      <p><b>4. One layer back (the chain rule).</b> The weight from input x${i + 1} (= ${fmt(f.x[i], 4)}) into neuron #${j + 1}:</p>
      <div class="well eq">∂L/∂h = sum over all ${V} tokens of (∂L/∂score × weight) = ${fmt(dh, 5)}<br>× tanh slope (1 − h²) = ${fmt(slope, 4)} → ∂L/∂z = ${fmt(dz, 5)}<br>× x${i + 1} = ${fmt(f.x[i], 4)} → ∂L/∂w = <b>${fmt(g1, 6)}</b><br>new w = ${fmt(w1, 5)} − ${fmt(lr)} × (${fmt(g1, 6)}) = <b>${fmt(w1 - lr * g1, 5)}</b></div>
      <p><b>5. Result.</b> After all ${int(learner.paramCount)} weights take this step together, the probability of <span class="mono">${yt}</span> goes from ${(py * 100).toFixed(2)}% to <b class="good">${(pa * 100).toFixed(2)}%</b>, and the loss falls from ${fmt(L, 4)} to <b>${fmt(-Math.log(pa), 4)}</b>.</p>
      <p class="note">Real training averages this over a batch of 32 examples before stepping, so no single sentence pulls too hard.</p>`;
  },
};
$("osEx").addEventListener("change", () => oneStep.render());
$("oneStep").addEventListener("toggle", () => $("oneStep").open && oneStep.render());
$("osApply").addEventListener("click", () => {
  if (!oneStep.clone || engine.phase === "running") return;
  if (engine.phase === "paused") engine.pause();
  const steps = learner.step + 1;
  learner.copyFrom(oneStep.clone);
  learner.step = steps;
  engine.seen += 1;
  engine.record(true);
  engine.emit("step");
  stage?.net.refreshWeights(true);
});

// ───────────────────────────── Chapter 7: generation ─────────────────────────────
const genRng = C.makeRng(((Date.now() >> 3) & 0xffff) + 7);
const gen = {
  ctx: null, done: false, lastF: null, lastStep: null, chosen: false,
  model() { return this.who() === "ref" ? ref : learner; },
  who() { return $("genWho").querySelector('[aria-checked="true"]').dataset.who; },
  reset() {
    this.ctx = [...Array(N).fill(START), ...bpe.ids($("genPrompt").value)].slice(-N);
    $("genOut").innerHTML = bpe.encode($("genPrompt").value).map((t) => `<span class="t prompt">${esc(show(t.text))}</span>`).join("");
    $("genBars").innerHTML = "";
    $("genNew").textContent = "";
    this.done = false; this.lastStep = null;
    this.window();
  },
  window() {
    const toks = [...$("genOut").querySelectorAll(".t")];
    toks.forEach((t, i) => { t.classList.toggle("seen", i >= toks.length - N); t.classList.toggle("gone", i < toks.length - N); });
    const seen = this.ctx.map((id) => show(bpe.vocab[id])).join(" ");
    $("genWindow").textContent = `What the model sees (its 3-token window): ${seen}`;
  },
  warn() {
    const m = this.model();
    $("genWarn").textContent = m === learner && learner.step < 300 ? `Your model has trained for ${int(learner.step)} steps, so expect nonsense. Train it in Chapter 6, or switch to the pre-trained copy.` : "";
  },
  tempWord() { const T = +$("genTemp").value; return T < 0.5 ? "safe and repetitive" : T <= 1.2 ? "balanced" : "wild"; },
  async go(single) {
    if (genBusy) return;
    if (this.done) this.reset();
    $("genGo").disabled = $("genOne").disabled = true;
    $("genGo").textContent = "Writing…";
    const m = this.model();
    this.ctx = await animateWrite({
      model: m, ctx: this.ctx, outEl: $("genOut"), barsEl: $("genBars"), temp: +$("genTemp").value, rng: genRng, single,
      onToken: (btn) => { this.lastStep = btn._step; this.lastF = { f: btn._step.f, id: btn._step.id }; this.ctx = [...this.ctx.slice(1), btn._step.id]; this.window(); },
    });
    $("genGo").disabled = $("genOne").disabled = false;
    $("genGo").textContent = "Write the rest";
    if (this.ctx[N - 1] === DOT || $("genOut").children.length > 24) {
      this.done = true;
      const text = C.normalize([...$("genOut").querySelectorAll(".t")].map((t) => t.textContent.replace(/␣/g, " ")).join("")).replace(/\s*\.$/, "").trim();
      const isNew = !C.SENTENCES.includes(text);
      $("genNew").innerHTML = isNew ? `<span class="good">New sentence:</span> “${esc(text)}.” is not one of the 104 it learned from.` : `“${esc(text)}.” is in the training data.`;
      if (isNew && this.ctx[N - 1] === DOT) markDone("newsent");
      if (m === learner && learner.step >= 600 && this.ctx[N - 1] === DOT) markDone("own600");
    }
  },
};
$("genPrompt").addEventListener("input", () => gen.reset());
seg($("genWho"), "who", () => { gen.chosen = true; gen.reset(); gen.warn(); if (currentBeat?.dataset.model === "gen") activate(currentBeat, true); });
$("genTemp").addEventListener("input", () => {
  $("genTempOut").textContent = fmt(+$("genTemp").value);
  $("genTempWord").textContent = gen.tempWord();
  const s = gen.lastStep;
  if (!s || genBusy) return;
  // reshape the last step's odds at the new temperature, without sampling again
  const q = C.softmaxT(s.f.logits, +$("genTemp").value), top = C.topK(q, 6);
  bars($("genBars"), top.map((k) => ({ label: show(bpe.vocab[k]), p: q[k], hit: k === s.id })));
  if (stage && activeScene === "net") stage.net.live({ ...s.f, p: q });
});
$("genGo").addEventListener("click", () => gen.go(false));
$("genOne").addEventListener("click", () => gen.go(true));
document.querySelectorAll("[data-try]").forEach((b) => b.addEventListener("click", () => { $("genPrompt").value = b.dataset.try; gen.reset(); gen.go(true); }));
$("genOut").addEventListener("click", (e) => { const b = e.target.closest("button.t"); if (b && !genBusy) revisit($("genOut"), $("genBars"), b); });

// dataset listing + progress reset
$("dataset").innerHTML = C.SENTENCES.map((s) => `<div class="${C.HELD_OUT.has(s) ? "held" : ""}">${esc(s)}.</div>`).join("");


// ───────────────────────────── Boot ─────────────────────────────
async function trainReference() {
  // The "pre-trained copy" is trained right here when the page opens, with
  // the same code and data you use in Chapter 6. You can watch it happen.
  const rng = C.makeRng(99);
  for (let s = 0; s < REF_STEPS; s += 50) {
    for (let k = 0; k < 50; k++) ref.trainStep(TR, 32, 0.5, rng);
    refStep = s + 50;
    if (refStep % 100 === 0 || refStep === REF_STEPS) {
      if (activeScene === "hero" || stage?.net.model === ref) stageMeta(`training the pre-trained copy here · step ${int(refStep)} / ${int(REF_STEPS)} · loss ${fmt(ref.meanLoss(VA))}`);
      if (stage) stage.net.refreshWeights(true);
    }
    await (REDUCED ? wait(0) : frame());
  }
  refReady = true;
  $("ch1Gen").disabled = false;
  $("ch1Gen").textContent = "Write a sentence";
  $("guessWait").hidden = true;
  $("guessOpts").querySelectorAll("button").forEach((b) => b.removeAttribute("aria-disabled"));
  if (stage) { stage.net.refreshWeights(true); heroNext = 0; }
  if (activeScene === "hero") stageMeta(`pre-trained copy · ${int(REF_STEPS)} steps`);
  if (fwd.who === "ref") { fwd.f = ref.forward(fwd.ctx); fwd.render(); if (currentBeat?.dataset.fwd || currentBeat?.dataset.model === "fwd") fwd.toStage(false); }
}

renderTasks();
renderGuess();
renderTokens();
neuron.reset();
renderLossCalc();
land.render();
engine.record(true);
fwd.f = ref.forward(fwd.ctx);
fwd.renderSlots();
fwd.render();
renderFwdTokens();
gen.reset();
renderProbe();
onEngine("init");
$("genTempWord").textContent = gen.tempWord();
fillAll();
setupScroll();
requestAnimationFrame(tick);
Promise.race([loadStage(), wait(2500)]).then(() => trainReference());
const redraw = () => { drawLoss(); neuron.render(); land.render(); renderLossCalc(); };
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
let rz = null;
addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(redraw, 150); });
addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("toast").hidden) $("toast").hidden = true; });

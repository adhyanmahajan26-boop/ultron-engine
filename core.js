// core.js — everything that actually "does AI" on this page.
// No libraries: the tokenizer, the network, backpropagation and the optimizer
// are all written out below in plain JavaScript so a reader can check them.

// ───────────────────────────── Random numbers ─────────────────────────────
// A seeded generator, so "Reset" gives the same starting weights every time.
export function makeRng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  next.gauss = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0, v = 0;
    while (u === 0) u = next();
    v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  next.int = (n) => Math.floor(next() * n);
  return next;
}

// ───────────────────────────── The dataset ─────────────────────────────
// 104 short sentences written for this page. Small enough to read in full,
// big enough that the patterns (who eats what, who sleeps where) can be learned.
export const SENTENCES = [
  "the girl sees the cow", "the boy sleeps in the house", "the dog is happy", "the boy likes the bird",
  "the big cow eats grass", "the mouse is fast", "the fox is red", "the girl sees the bird",
  "the cat runs to the house", "the boy drinks milk", "the dog plays in the park", "the girl gives the bird seeds",
  "the girl plays in the park", "the boy likes the dog", "the cat is small", "the boy plays in the park",
  "the small cat sleeps in the house", "the girl likes the dog", "the cow is slow", "the big cow sleeps in the barn",
  "the boy plays in the garden", "the cow runs to the barn", "the small bird eats seeds", "the bird sleeps in the tree",
  "the mouse runs to the house", "the girl eats apples", "the boy eats cake", "the dog chases the cat",
  "the bird drinks water", "the dog is big", "the red fox eats meat", "the girl gives the fox meat",
  "the boy gives the fox meat", "the cow sleeps in the barn", "the cat chases the bird", "the dog runs to the house",
  "the girl chases the ball", "the dog chases the ball", "the girl plays in the garden", "the fox is fast",
  "the small cat eats fish", "the dog plays in the garden", "the cow drinks water", "the fox chases the bird",
  "the boy sees the cow", "the boy gives the dog bread", "the girl sleeps in the house", "the cat chases the mouse",
  "the cat sleeps in the house", "the boy sees the fox", "the fox chases the mouse", "the fox eats meat",
  "the bird eats seeds", "the dog sleeps in the house", "the girl sees the fox", "the cow eats grass",
  "the cat drinks milk", "the girl eats cake", "the happy dog eats bread", "the mouse is small",
  "the cat plays in the garden", "the cat eats fish", "the fox sleeps in the forest", "the boy drinks juice",
  "the boy likes the cat", "the cat is soft", "the girl gives the dog bread", "the boy gives the cat fish",
  "the girl sees the cat", "the boy gives the bird seeds", "the bird is small", "the girl gives the cow grass",
  "the red fox sleeps in the forest", "the small bird sleeps in the nest", "the girl gives the cat fish", "the girl drinks juice",
  "the girl likes the bird", "the mouse eats cheese", "the boy sees the cat", "the boy eats bread",
  "the girl chases the cat", "the happy dog sleeps in the house", "the dog eats bread", "the fox walks in the forest",
  "the girl sees the dog", "the boy sees the dog", "the bird sits in the tree", "the dog drinks water",
  "the dog eats meat", "the boy chases the ball", "the bird is blue", "the boy sees the bird",
  "the fox drinks water", "the bird sits in the garden", "the girl drinks milk", "the cow is big",
  "the boy chases the dog", "the fox runs to the forest", "the bird sleeps in the nest", "the bird runs to the nest",
  "the girl likes the cat", "the cow walks in the field", "the boy gives the cow grass", "the cat plays in the house",
];

// Held back from training. The model never learns from these; we only use
// them to check whether it learned general patterns or just memorised.
export const HELD_OUT = new Set([
  "the girl gives the cow grass", "the boy sees the fox", "the dog drinks water", "the girl likes the dog",
  "the cat chases the mouse", "the happy dog sleeps in the house", "the bird sleeps in the tree",
  "the boy plays in the garden", "the fox eats meat", "the girl drinks milk",
]);
export const TRAIN_SENTENCES = SENTENCES.filter((s) => !HELD_OUT.has(s));
export const VAL_SENTENCES = SENTENCES.filter((s) => HELD_OUT.has(s));

// ───────────────────────────── Tokenizer (BPE) ─────────────────────────────
// Byte-Pair Encoding, the same family of algorithm GPT models use.
// Start from single characters, then repeatedly glue together the pair of
// neighbouring pieces that appears most often in the training text.
export const START = "[start]";
export const BASE_CHARS = [" ", ".", ..."abcdefghijklmnopqrstuvwxyz"];

export function normalize(text) {
  return text.toLowerCase().replace(/[^a-z. ]+/g, " ").replace(/\s+/g, " ").trim();
}

// Split text into words before BPE runs. Like GPT-2, the space before a word
// belongs to that word (" cat"), and punctuation stands alone.
export function pretokenize(text) {
  const t = " " + normalize(text);
  return t.match(/ ?[a-z]+|\./g) || [];
}

export class BPE {
  constructor() {
    this.merges = []; // [{a, b, count}] in the order they were learned
    this.vocab = [START, ...BASE_CHARS];
    this.id = new Map(this.vocab.map((t, i) => [t, i]));
    this.rank = new Map();
  }

  train(sentences, maxMerges = 400) {
    const counts = new Map();
    for (const s of sentences) for (const w of pretokenize(s)) counts.set(w, (counts.get(w) || 0) + 1);
    const words = [...counts.entries()].map(([w, c]) => ({ syms: [...w], count: c }));
    for (let m = 0; m < maxMerges; m++) {
      const pairs = new Map();
      for (const { syms, count } of words) {
        for (let i = 0; i < syms.length - 1; i++) {
          const k = syms[i] + "\u0000" + syms[i + 1];
          pairs.set(k, (pairs.get(k) || 0) + count);
        }
      }
      let best = null, bestCount = 1;
      for (const [k, c] of pairs) {
        if (c > bestCount || (c === bestCount && best !== null && k < best)) { best = k; bestCount = c; }
      }
      if (best === null || bestCount < 2) break; // nothing appears twice: stop
      const [a, b] = best.split("\u0000");
      const tok = a + b;
      this.merges.push({ a, b, count: bestCount });
      this.rank.set(best, this.merges.length - 1);
      this.id.set(tok, this.vocab.length);
      this.vocab.push(tok);
      for (const w of words) w.syms = mergePair(w.syms, a, b);
    }
    return this;
  }

  // Tokenize one pre-token using only the first `limit` merges.
  encodeWord(word, limit = Infinity) {
    let syms = [...word];
    for (;;) {
      let bestRank = Infinity, bi = -1;
      for (let i = 0; i < syms.length - 1; i++) {
        const r = this.rank.get(syms[i] + "\u0000" + syms[i + 1]);
        if (r !== undefined && r < limit && r < bestRank) { bestRank = r; bi = i; }
      }
      if (bi < 0) return syms;
      const m = this.merges[bestRank];
      syms = mergePair(syms, m.a, m.b);
    }
  }

  // Returns [{text, id, start, end}] — start/end are character offsets into
  // the joined pre-tokens, used to animate pieces sliding together.
  encode(text, limit = Infinity) {
    const out = [];
    let pos = 0;
    for (const w of pretokenize(text)) {
      for (const t of this.encodeWord(w, limit)) {
        out.push({ text: t, id: this.id.get(t), start: pos, end: pos + t.length });
        pos += t.length;
      }
    }
    return out;
  }

  ids(text) { return this.encode(text).map((t) => t.id); }
}

function mergePair(syms, a, b) {
  const out = [];
  for (let i = 0; i < syms.length; i++) {
    if (i < syms.length - 1 && syms[i] === a && syms[i + 1] === b) { out.push(a + b); i++; }
    else out.push(syms[i]);
  }
  return out;
}

// Turn sentences into (context → next token) training examples.
// Each sentence is padded with [start] tokens and ends with ".".
export function makeExamples(sentences, bpe, N) {
  const ex = [];
  const startId = bpe.id.get(START);
  for (const s of sentences) {
    const ids = bpe.ids(s + ".");
    const seq = [...Array(N).fill(startId), ...ids];
    for (let i = N; i < seq.length; i++) ex.push({ ctx: seq.slice(i - N, i), y: seq[i] });
  }
  return ex;
}

// ───────────────────────────── The neural network ─────────────────────────────
// A small "multi-layer perceptron" language model (the design from Bengio et
// al., 2003 — the ancestor of today's large models):
//   1. look up a vector of D numbers (an "embedding") for each of the N context tokens
//   2. glue them into one list x of N·D numbers
//   3. hidden layer:  h = tanh(W1·x + b1)            (H neurons)
//   4. output layer:  logits = W2·h + b2             (one score per token in the vocabulary)
//   5. softmax turns the scores into probabilities that add up to 1
export class Model {
  constructor(V, { D = 10, N = 3, H = 48 } = {}) {
    Object.assign(this, { V, D, N, H, I: N * D });
    this.E = new Float64Array(V * D);
    this.W1 = new Float64Array(H * this.I);
    this.b1 = new Float64Array(H);
    this.W2 = new Float64Array(V * H);
    this.b2 = new Float64Array(V);
    this.gE = new Float64Array(this.E.length);
    this.gW1 = new Float64Array(this.W1.length);
    this.gb1 = new Float64Array(H);
    this.gW2 = new Float64Array(this.W2.length);
    this.gb2 = new Float64Array(V);
    this.step = 0;
  }

  get paramCount() { return this.E.length + this.W1.length + this.b1.length + this.W2.length + this.b2.length; }

  init(seed) {
    const rng = makeRng(seed);
    // embeddings start tiny, so what training writes into them stands out
    for (let i = 0; i < this.E.length; i++) this.E[i] = rng.gauss() * 0.03;
    const s1 = (5 / 3) / Math.sqrt(this.I); // keeps tanh out of its flat ends at the start
    for (let i = 0; i < this.W1.length; i++) this.W1[i] = rng.gauss() * s1;
    this.b1.fill(0);
    const s2 = 0.05 / Math.sqrt(this.H); // tiny: the untrained model guesses almost evenly
    for (let i = 0; i < this.W2.length; i++) this.W2[i] = rng.gauss() * s2;
    this.b2.fill(0);
    this.step = 0;
    return this;
  }

  copyFrom(m) {
    for (const k of ["E", "W1", "b1", "W2", "b2"]) this[k].set(m[k]);
    this.step = m.step;
    return this;
  }

  clone() { return new Model(this.V, this).copyFrom(this); }

  forward(ctx) {
    const { D, H, I, V } = this;
    const x = new Float64Array(I);
    for (let n = 0; n < ctx.length; n++) for (let d = 0; d < D; d++) x[n * D + d] = this.E[ctx[n] * D + d];
    const z1 = new Float64Array(H), h = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      let s = this.b1[j];
      const row = j * I;
      for (let i = 0; i < I; i++) s += this.W1[row + i] * x[i];
      z1[j] = s;
      h[j] = Math.tanh(s);
    }
    const logits = new Float64Array(V);
    let max = -Infinity;
    for (let k = 0; k < V; k++) {
      let s = this.b2[k];
      const row = k * H;
      for (let j = 0; j < H; j++) s += this.W2[row + j] * h[j];
      logits[k] = s;
      if (s > max) max = s;
    }
    const p = new Float64Array(V);
    let sum = 0;
    for (let k = 0; k < V; k++) { p[k] = Math.exp(logits[k] - max); sum += p[k]; }
    for (let k = 0; k < V; k++) p[k] /= sum;
    return { ctx, x, z1, h, logits, p };
  }

  // Backpropagation for one example, adding into the gradient buffers.
  // Returns the loss  −log p(correct token).
  accumulate(ctx, y, scale = 1) {
    const { D, H, I, V } = this;
    const f = this.forward(ctx);
    const loss = -Math.log(Math.max(f.p[y], 1e-12));
    // ∂loss/∂logit_k = p_k − (1 if k is the right answer else 0)
    const dlog = new Float64Array(V);
    for (let k = 0; k < V; k++) dlog[k] = (f.p[k] - (k === y ? 1 : 0)) * scale;
    const dh = new Float64Array(H);
    for (let k = 0; k < V; k++) {
      const g = dlog[k];
      if (g === 0) continue;
      const row = k * H;
      this.gb2[k] += g;
      for (let j = 0; j < H; j++) {
        this.gW2[row + j] += g * f.h[j];
        dh[j] += g * this.W2[row + j];
      }
    }
    const dx = new Float64Array(I);
    for (let j = 0; j < H; j++) {
      const dz = dh[j] * (1 - f.h[j] * f.h[j]); // tanh'(z) = 1 − tanh(z)²
      this.gb1[j] += dz;
      const row = j * I;
      for (let i = 0; i < I; i++) {
        this.gW1[row + i] += dz * f.x[i];
        dx[i] += dz * this.W1[row + i];
      }
    }
    for (let n = 0; n < ctx.length; n++) for (let d = 0; d < D; d++) this.gE[ctx[n] * D + d] += dx[n * D + d];
    return loss;
  }

  zeroGrad() { for (const g of [this.gE, this.gW1, this.gb1, this.gW2, this.gb2]) g.fill(0); }

  // Plain gradient descent: every weight moves a small step downhill.
  //   w ← w − learningRate × ∂loss/∂w
  applyGrad(lr) {
    const pairs = [[this.E, this.gE], [this.W1, this.gW1], [this.b1, this.gb1], [this.W2, this.gW2], [this.b2, this.gb2]];
    for (const [w, g] of pairs) for (let i = 0; i < w.length; i++) w[i] -= lr * g[i];
  }

  trainStep(examples, batchSize, lr, rng) {
    this.zeroGrad();
    let loss = 0;
    const scale = 1 / batchSize;
    for (let b = 0; b < batchSize; b++) {
      const e = examples[rng.int(examples.length)];
      loss += this.accumulate(e.ctx, e.y, scale);
    }
    this.applyGrad(lr);
    this.step++;
    return loss / batchSize;
  }

  meanLoss(examples) {
    let s = 0;
    for (const e of examples) s -= Math.log(Math.max(this.forward(e.ctx).p[e.y], 1e-12));
    return s / examples.length;
  }

  // Pick the next token. Temperature < 1 makes safe choices; > 1 makes wild ones.
  sample(ctx, temperature, rng) {
    const f = this.forward(ctx);
    const q = softmaxT(f.logits, temperature);
    let r = rng(), k = 0;
    for (; k < q.length - 1; k++) { r -= q[k]; if (r <= 0) break; }
    return { id: k, f, q };
  }
}

export function softmaxT(logits, T) {
  const t = Math.max(T, 1e-3);
  let max = -Infinity;
  for (const l of logits) if (l > max) max = l;
  const q = new Float64Array(logits.length);
  let s = 0;
  for (let k = 0; k < logits.length; k++) { q[k] = Math.exp((logits[k] - max) / t); s += q[k]; }
  for (let k = 0; k < q.length; k++) q[k] /= s;
  return q;
}

export function topK(p, k) {
  return [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, k);
}

// ───────────────────────────── PCA (for the 3D embedding map) ─────────────────────────────
// Embeddings have D numbers each; we can only draw 3. PCA finds the three
// directions along which the vectors differ the most, and we draw those.
export function pca3(vectors, prevAxes) {
  const n = vectors.length, D = vectors[0].length;
  const mean = new Float64Array(D);
  for (const v of vectors) for (let d = 0; d < D; d++) mean[d] += v[d] / n;
  const C = Array.from({ length: D }, () => new Float64Array(D));
  for (const v of vectors) for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) C[a][b] += (v[a] - mean[a]) * (v[b] - mean[b]) / n;
  const axes = [];
  for (let c = 0; c < 3; c++) {
    let u = new Float64Array(D).map((_, i) => (prevAxes ? prevAxes[c][i] : 0) + Math.cos(i * 1.7 + c * 2.3));
    for (let it = 0; it < 80; it++) {
      const w = new Float64Array(D);
      for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) w[a] += C[a][b] * u[b];
      for (const prev of axes) { const dp = dot(w, prev); for (let i = 0; i < D; i++) w[i] -= dp * prev[i]; }
      const nrm = Math.hypot(...w) || 1;
      u = w.map((x) => x / nrm);
    }
    if (prevAxes && dot(u, prevAxes[c]) < 0) u = u.map((x) => -x); // stop the picture from flipping
    axes.push(u);
  }
  const pts = vectors.map((v) => axes.map((ax) => { let s = 0; for (let d = 0; d < D; d++) s += (v[d] - mean[d]) * ax[d]; return s; }));
  return { pts, axes };
}

function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

// ───────────────────────────── Toy problem for the loss landscape ─────────────────────────────
// One neuron with just two weights (w and b), so its loss can be drawn as a
// real 3D surface:  prediction = tanh(w·x + b),  loss = average of (prediction − target)²
export const TOY = [
  { x: -2.0, y: -0.95 }, { x: -1.2, y: -0.85 }, { x: -0.5, y: -0.55 },
  { x: 0.2, y: 0.05 }, { x: 0.9, y: 0.6 }, { x: 1.6, y: 0.88 }, { x: 2.3, y: 0.97 },
];

export function toyLoss(w, b) {
  let s = 0;
  for (const { x, y } of TOY) { const a = Math.tanh(w * x + b); s += (a - y) ** 2; }
  return s / TOY.length;
}

export function toyGrad(w, b) {
  let gw = 0, gb = 0;
  for (const { x, y } of TOY) {
    const a = Math.tanh(w * x + b);
    const g = 2 * (a - y) * (1 - a * a) / TOY.length;
    gw += g * x; gb += g;
  }
  return [gw, gb];
}

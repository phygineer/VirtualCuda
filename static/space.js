// 3D views under the inspector, built from the same captured trace (nothing illustrative):
//   brain    x = layer, y = neuron index / that layer's MLP width, z = token. The 16 strongest
//            MLP neurons of every layer light up as each token goes through; older tokens fade.
//   journey  each token's hidden state, embedding → every layer's output, drawn as a line in the
//            top-3 PCA space of the unit-normalized residual stream.
// Everything drawn comes from one small JSON (the same shape is exported, and hidden inside
// exported PNGs), so a plot can be reopened later without the model.
//
// Uses the page's globals from sim.html: meta, steps, vec, residOut, norm, cur, ui, saveUi,
// showTip, hideTip, showStepEnd, streamDone, $.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const FORMAT = "virtualcuda.journey", VERSION = 1, PNG_KEY = "virtualcuda";
const SPACE_NOTE = "top-3 principal components of the residual stream, each vector scaled to length 1 first";

let live = null;          // built from the running trace (same shape as an exported file)
let imported = null;      // loaded from a .json / .png
let cursor = null;        // {step, layersDone, embedDone}: what has been computed so far
let mode = ui.spMode === "journey" ? "journey" : "brain";
let pcaBasis = null, pcaFor = 0, pcaTimer = null;
const data = () => imported || live;

// ---------------------------------------------------------------- data
function tokenOf(s) {
  const t = s.input_tokens[s.input_tokens.length - 1];
  return { step: 0, position: s.position, input: { id: t.id, text: t.text },
           chosen: { id: s.chosen.id, text: s.chosen.text, prob: s.chosen.prob } };
}

function liveData() {
  return {
    format: FORMAT, version: VERSION,
    model: { id: meta.model, architecture: meta.architecture, layers: meta.layers, hidden_size: meta.hidden_size,
             intermediate_size: meta.intermediate_size, mlp_n: steps[0].layers.map((l) => l.mlp_n) },
    prompt: meta.template_text,
    generated: steps.map((s) => s.chosen.text).join(""),
    tokens: steps.map((s, i) => ({ ...tokenOf(s), step: i })),
    brain: {
      axes: { x: "layer / (layers − 1)", y: "neuron index / MLP width of that layer", z: "token / (tokens − 1)" },
      top_k: steps[0].layers[0].mlp_top.length,
      sparks: steps.map((s) => s.layers.map((l) => l.mlp_top.flat())),  // [neuron, activation, …] per layer
    },
    journey: live?.journey ?? null,
  };
}

// Top-3 principal components by subspace iteration (warm-started from the last basis, so axes
// don't flip while tokens stream in).
function pca3(X, N, D, init) {
  let Q = init ? init.map((q) => q.slice()) : [0, 1, 2].map(() => Float32Array.from({ length: D }, () => Math.random() - 0.5));
  const Y = new Float64Array(N * 3);
  const project = () => {
    for (let i = 0; i < N; i++) {
      const o = i * D;
      let a = 0, b = 0, c = 0;
      for (let d = 0; d < D; d++) { const x = X[o + d]; a += x * Q[0][d]; b += x * Q[1][d]; c += x * Q[2][d]; }
      Y[3 * i] = a; Y[3 * i + 1] = b; Y[3 * i + 2] = c;
    }
  };
  for (let it = 0, n = init ? 10 : 40; it < n; it++) {
    project();
    const Z = [0, 1, 2].map(() => new Float64Array(D));
    for (let i = 0; i < N; i++) {
      const o = i * D, a = Y[3 * i], b = Y[3 * i + 1], c = Y[3 * i + 2];
      for (let d = 0; d < D; d++) { const x = X[o + d]; Z[0][d] += x * a; Z[1][d] += x * b; Z[2][d] += x * c; }
    }
    for (let c = 0; c < 3; c++) {  // Gram-Schmidt
      for (let p = 0; p < c; p++) {
        let dot = 0;
        for (let d = 0; d < D; d++) dot += Z[c][d] * Z[p][d];
        for (let d = 0; d < D; d++) Z[c][d] -= dot * Z[p][d];
      }
      let n2 = 0;
      for (let d = 0; d < D; d++) n2 += Z[c][d] ** 2;
      const k = 1 / (Math.sqrt(n2) || 1);
      for (let d = 0; d < D; d++) Z[c][d] *= k;
    }
    Q = Z.map((z) => Float32Array.from(z));
  }
  for (let c = 0; c < 3; c++) {  // stable signs: agree with the previous basis, else positive skew
    let s = 0;
    if (init) for (let d = 0; d < D; d++) s += Q[c][d] * init[c][d];
    else { project(); for (let i = 0; i < N; i++) s += Y[3 * i + c] ** 3; }
    if (s < 0) for (let d = 0; d < D; d++) Q[c][d] = -Q[c][d];
  }
  project();
  let total = 0;
  for (let i = 0; i < N * D; i++) total += X[i] * X[i];
  const explained = [0, 1, 2].map((c) => { let e = 0; for (let i = 0; i < N; i++) e += Y[3 * i + c] ** 2; return e / (total || 1); });
  return { Y, Q, explained };
}

function computeJourney() {
  const L = meta.layers, n = steps.length, S = L + 1, D = meta.hidden_size;
  const X = new Float32Array(n * S * D), norms = [], mean = new Float64Array(D);
  for (let s = 0; s < n; s++) {
    const row = [];
    for (let k = 0; k < S; k++) {
      const v = k === 0 ? vec(s, 0, "resid_in") : residOut(s, k - 1), nr = norm(v) || 1, o = (s * S + k) * D;
      row.push(+nr.toFixed(2));
      for (let d = 0; d < D; d++) { X[o + d] = v[d] / nr; mean[d] += v[d] / nr; }
    }
    norms.push(row);
  }
  for (let d = 0; d < D; d++) mean[d] /= n * S;
  for (let i = 0; i < n * S; i++) for (let d = 0; d < D; d++) X[i * D + d] -= mean[d];
  const { Y, Q, explained } = pca3(X, n * S, D, pcaBasis?.length === 3 && pcaBasis[0].length === D ? pcaBasis : null);
  pcaBasis = Q;
  let m = 0;
  for (const y of Y) m = Math.max(m, Math.abs(y));
  const r = (x) => +(x / (m || 1)).toFixed(4);
  return {
    space: SPACE_NOTE,
    stops: ["embed", ...Array.from({ length: L }, (_, i) => `L${i}`)],
    explained: explained.map((e) => +e.toFixed(4)),
    norms,
    points: Array.from({ length: n }, (_, s) => Array.from({ length: S }, (_, k) => {
      const i = s * S + k;
      return [r(Y[3 * i]), r(Y[3 * i + 1]), r(Y[3 * i + 2])];
    })),
  };
}

function journeyNow() {
  if (!live || imported) return;
  clearTimeout(pcaTimer); pcaTimer = null;
  if (pcaFor === steps.length && live.journey) return;
  live.journey = computeJourney();
  pcaFor = steps.length;
  buildJourney(live);
  applyCursor();
}

// Only while the journey is on screen; throttled while tokens stream in.
function scheduleJourney() {
  if (imported || !live || mode !== "journey" || ui.fold.sp || pcaFor === steps.length || pcaTimer) return;
  pcaTimer = setTimeout(() => {
    pcaTimer = null;
    if (!live || imported) return;
    live.journey = computeJourney();
    pcaFor = steps.length;
    buildJourney(live);
    applyCursor();
    scheduleJourney();
  }, live.journey && !streamDone ? 1500 : 30);
}

// ---------------------------------------------------------------- scene
const view = $("spView"), canvas = view.querySelector("canvas");
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
} catch {
  $("spEmpty").textContent = "WebGL isn't available in this browser, so the 3D views can't be drawn.";
}
const scene = new THREE.Scene();
scene.background = new THREE.Color("#0d0f12");
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 50);
const HOME = new THREE.Vector3(2.9, 1.7, 3.4);
camera.position.copy(HOME);
const controls = renderer && new OrbitControls(camera, canvas);
if (controls) { controls.enableDamping = true; controls.dampingFactor = 0.08; controls.autoRotateSpeed = 0.8; }

// ---------------------------------------------------------------- style: palette + trails
// Remembered per browser (ui.spStyle) and saved in exports, so an imported file looks the same.
const PRESETS = {
  ember:   { pos: "#f97316", neg: "#3b82f6", p0: "#5eead4", p1: "#a78bfa", p2: "#f97316", bg: "#0d0f12" },
  aurora:  { pos: "#4ade80", neg: "#f472b6", p0: "#60a5fa", p1: "#4ade80", p2: "#facc15", bg: "#0a0f1c" },
  magma:   { pos: "#fbbf24", neg: "#a855f7", p0: "#7c3aed", p1: "#e11d48", p2: "#fbbf24", bg: "#0b0710" },
  viridis: { pos: "#fde725", neg: "#3b82c4", p0: "#3b528b", p1: "#21918c", p2: "#fde725", bg: "#0d0f12" },
  ice:     { pos: "#e0f2fe", neg: "#0ea5e9", p0: "#1e40af", p1: "#38bdf8", p2: "#f0f9ff", bg: "#05080f" },
  mono:    { pos: "#f5f5f5", neg: "#8a8a8a", p0: "#4b4b4b", p1: "#a3a3a3", p2: "#ffffff", bg: "#000000" },
};
const DEFAULT_STYLE = { preset: "ember", ...PRESETS.ember, brainBy: "sign", journeyBy: "layer",
                        trail: "fade", trailN: 3, fadeLen: 2.5, lines: true, lineAlpha: 0.8, size: 1, stats: true };
const REBUILD = new Set(["pos", "neg", "p0", "p1", "p2", "brainBy", "journeyBy", "size"]);
const cleanStyle = (s) => {
  const o = { ...DEFAULT_STYLE };
  for (const k in DEFAULT_STYLE) if (s && typeof s[k] === typeof DEFAULT_STYLE[k]) o[k] = s[k];
  return o;
};
let style = cleanStyle(ui.spStyle);
const WHITE = new THREE.Color("#ffffff");
let POS, NEG, PATH;
function applyPalette() {
  POS = new THREE.Color(style.pos); NEG = new THREE.Color(style.neg);
  PATH = [style.p0, style.p1, style.p2].map((c) => new THREE.Color(c));
  scene.background.set(style.bg);
}
applyPalette();
const pathColor = (t) => (t < 0.5 ? PATH[0].clone().lerp(PATH[1], t * 2) : PATH[1].clone().lerp(PATH[2], t * 2 - 1));
// One hue per token (violet-red → ... → violet), lightness l.
const tokColor = (s, n, l) => new THREE.Color().setHSL((n > 1 ? s / (n - 1) : 0) * 0.8, 0.85, l);
// Opacity of a token `age` steps behind the current one (0 hides it).
function pastAlpha(age) {
  if (style.trail === "all") return 0.75;
  if (style.trail === "current") return 0;
  if (style.trail === "last") return age <= style.trailN ? 0.25 + 0.55 * (1 - (age - 1) / style.trailN) : 0;
  return 0.1 + 0.6 * Math.exp(-age / style.fadeLen);
}

const sparkMaterial = () => new THREE.ShaderMaterial({
  uniforms: { uTime: { value: 0 }, uScale: { value: 400 } },
  vertexShader: `
    attribute vec3 aColor; attribute float aSize, aAlpha, aBorn, aCur;
    uniform float uTime, uScale;
    varying vec3 vColor; varying float vAlpha;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      float flash = exp(-max(uTime - aBorn, 0.0) * 3.5);
      float twinkle = aCur * 0.22 * sin(uTime * 7.0 + position.y * 40.0 + position.x * 13.0);
      gl_PointSize = aSize * (1.0 + 1.2 * flash + twinkle) * uScale / -mv.z;
      gl_Position = projectionMatrix * mv;
      vColor = mix(aColor, vec3(1.0), flash * 0.35);
      vAlpha = aAlpha * (1.0 + 0.5 * flash);
    }`,
  fragmentShader: `
    varying vec3 vColor; varying float vAlpha;
    void main() {
      float d = length(gl_PointCoord - 0.5) * 2.0;
      if (d > 1.0) discard;
      float glow = pow(1.0 - d, 2.0) + smoothstep(0.35, 0.0, d) * 0.8;
      gl_FragColor = vec4(vColor * glow * vAlpha, 1.0);
    }`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});

// A point cloud whose per-point size / alpha / flash are rewritten as the cursor moves.
function sparkCloud(n) {
  const g = new THREE.BufferGeometry();
  const attr = (name, k) => { const a = new THREE.BufferAttribute(new Float32Array(n * k), k); g.setAttribute(name, a); return a.array; };
  const c = { pos: attr("position", 3), color: attr("aColor", 3), size: attr("aSize", 1), alpha: attr("aAlpha", 1),
              born: attr("aBorn", 1).fill(-100), cur: attr("aCur", 1), base: new Float32Array(n), shown: new Uint8Array(n) };
  c.points = new THREE.Points(g, sparkMaterial());
  c.points.frustumCulled = false;
  c.dirty = () => ["aSize", "aAlpha", "aBorn", "aCur"].forEach((a) => (g.attributes[a].needsUpdate = true));
  return c;
}

function label(text, { color = "#8a919c", h = 0.075, bold = false } = {}) {
  const c = document.createElement("canvas"), ctx = c.getContext("2d"), font = `${bold ? "600 " : ""}44px ui-monospace, Menlo, monospace`;
  ctx.font = font;
  c.width = Math.ceil(ctx.measureText(text).width) + 16; c.height = 60;
  ctx.font = font; ctx.fillStyle = color; ctx.textBaseline = "middle";
  ctx.fillText(text, 8, 30);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, depthTest: false }));
  s.scale.set((h * c.width) / c.height, h, 1);
  return s;
}
const at = (obj, x, y, z) => (obj.position.set(x, y, z), obj);

function frame(group) {
  const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)),
    new THREE.LineBasicMaterial({ color: "#2f3540" }));
  group.add(box);
}

function clear(group) {
  group.traverse((o) => { o.geometry?.dispose(); o.material?.map?.dispose(); o.material?.dispose(); });
  group.clear();
}

const brain = { group: new THREE.Group() }, journey = { group: new THREE.Group() };
scene.add(brain.group, journey.group);

// ---------------------------------------------------------------- brain
const brainX = (L, i) => (L > 1 ? (i / (L - 1)) * 2 - 1 : 0);
const tokZ = (n, s) => (n > 1 ? (s / (n - 1)) * 2 - 1 : 0);

function buildBrain(D) {
  clear(brain.group);
  const L = D.model.layers, n = D.tokens.length, sp = D.brain.sparks;
  const count = sp.reduce((t, s) => t + s.reduce((u, l) => u + l.length / 2, 0), 0);
  const maxA = new Float32Array(L);
  sp.forEach((s) => s.forEach((l, i) => { for (let j = 1; j < l.length; j += 2) maxA[i] = Math.max(maxA[i], Math.abs(l[j])); }));
  const c = sparkCloud(count);
  const info = { step: new Int32Array(count), layer: new Int32Array(count), neuron: new Int32Array(count), act: new Float32Array(count) };
  let p = 0;
  const first = sp.map((s) => s.map(() => 0));   // index of each (token, layer)'s strongest spark
  sp.forEach((s, si) => s.forEach((l, li) => {
    first[si][li] = p;
    for (let j = 0; j < l.length; j += 2, p++) {
      const nrn = l[j], a = l[j + 1], t = Math.abs(a) / (maxA[li] || 1);
      c.pos.set([brainX(L, li), (nrn / D.model.mlp_n[li]) * 2 - 1, tokZ(n, si)], 3 * p);
      const col = (style.brainBy === "layer" ? pathColor(L > 1 ? li / (L - 1) : 0) : style.brainBy === "token" ? tokColor(si, n, 0.55)
        : a >= 0 ? POS.clone() : NEG.clone()).lerp(WHITE, 0.45 * t * t);
      c.color.set([col.r, col.g, col.b], 3 * p);
      c.base[p] = (0.022 + 0.085 * Math.sqrt(t)) * style.size;
      info.step[p] = si; info.layer[p] = li; info.neuron[p] = nrn; info.act[p] = a;
    }
  }));
  Object.assign(brain, { cloud: c, info, L, n, D });
  frame(brain.group);
  // floor rails, one per layer
  const rails = [];
  for (let i = 0; i < L; i++) rails.push(brainX(L, i), -1, -1, brainX(L, i), -1, 1);
  const rg = new THREE.BufferGeometry();
  rg.setAttribute("position", new THREE.Float32BufferAttribute(rails, 3));
  brain.group.add(new THREE.LineSegments(rg, new THREE.LineBasicMaterial({ color: "#1d2229" })));
  // the "now" plane sweeping through tokens
  brain.now = new THREE.Mesh(new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: style.p0, transparent: true, opacity: 0.045, depthWrite: false, side: THREE.DoubleSide }));
  brain.group.add(brain.now);
  // signal paths: one line per token through each layer's strongest neuron, in that spark's color
  brain.paths = sp.map((s, si) => {
    const pos = [], col = [];
    s.forEach((l, li) => {
      pos.push(brainX(L, li), l.length ? (l[0] / D.model.mlp_n[li]) * 2 - 1 : 0, tokZ(n, si));
      const k = Math.min(first[si][li], p - 1);
      const cc = new THREE.Color(c.color[3 * k], c.color[3 * k + 1], c.color[3 * k + 2]).lerp(WHITE, 0.4);
      col.push(cc.r, cc.g, cc.b);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    line.frustumCulled = false;
    brain.group.add(line);
    return line;
  });
  brain.group.add(c.points);
  // axes
  brain.group.add(
    at(label("L0"), -1, -1.12, 1.12), at(label(`L${L - 1}`), 1, -1.12, 1.12), at(label("layer →", { color: "#e6e8eb" }), 0, -1.26, 1.18),
    at(label("0"), -1.12, -1, 1.06), at(label("1"), -1.12, 1, 1.06), at(label("neuron / width", { color: "#e6e8eb" }), -1.3, 0.1, 1.08),
    at(label("token 1"), 1.2, -1.08, -1), at(label(`token ${n}`), 1.2, -1.08, 1), at(label("token →", { color: "#e6e8eb" }), 1.32, -1.2, 0));
  brain.tokLabel = null; brain.tokLabelStep = -1;
}

// ---------------------------------------------------------------- journey
function buildJourney(D) {
  clear(journey.group);
  journey.lines = [];
  journey.D = D;
  if (!D.journey) return;
  const J = D.journey, n = J.points.length, S = J.stops.length;
  const c = sparkCloud(n * S);
  J.points.forEach((pts, s) => {
    const g = new THREE.BufferGeometry(), pos = [], col = [];
    pts.forEach(([x, y, z], k) => {
      const cc = style.journeyBy === "token" ? tokColor(s, n, 0.28 + 0.4 * (k / (S - 1))) : pathColor(k / (S - 1));
      pos.push(x, y, z); col.push(cc.r, cc.g, cc.b);
      c.pos.set([x, y, z], 3 * (s * S + k));
      c.color.set([cc.r, cc.g, cc.b], 3 * (s * S + k));
      c.base[s * S + k] = (k === 0 ? 0.05 : 0.032) * style.size;
    });
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    line.frustumCulled = false;
    journey.lines.push(line);
    journey.group.add(line);
  });
  journey.cloud = c;
  journey.group.add(c.points);
  frame(journey.group);
  const pc = (i) => `PC${i + 1} (${(J.explained[i] * 100).toFixed(0)}%)`;
  journey.group.add(at(label(pc(0) + " →", { color: "#e6e8eb" }), 0, -1.2, 1.15), at(label(pc(1), { color: "#e6e8eb" }), -1.32, 0, 1.08),
    at(label(pc(2) + " →", { color: "#e6e8eb" }), 1.3, -1.12, 0));
  journey.tokLabel = null; journey.tokLabelStep = -1;
}

// ---------------------------------------------------------------- cursor
function cursorOf(c) {
  const L = meta.layers, early = c.kind === "input" || c.kind === "media";
  const layersDone = c.layer != null ? c.layer + (c.kind === "attn" ? 0 : 1) : early || c.kind === "embed" ? 0 : L;
  return { step: c.step, layersDone, embedDone: !early };
}

function tokenLabel(obj, D, step, x, y, z) {
  if (obj.tokLabelStep !== step) {
    if (obj.tokLabel) { obj.group.remove(obj.tokLabel); obj.tokLabel.material.map.dispose(); obj.tokLabel.material.dispose(); }
    const t = D.tokens[step];
    obj.tokLabel = label(`${vis1(t.input.text)} → ${vis1(t.chosen.text)}`, { color: "#5eead4", h: 0.085, bold: true });
    obj.tokLabelStep = step;
    obj.group.add(obj.tokLabel);
  }
  obj.tokLabel.position.set(x, y, z);
}
const vis1 = (s) => JSON.stringify(s).slice(1, -1).replace(/ /g, "·").slice(0, 24) || "∅";

function applyCursor() {
  const D = data();
  if (!D || !cursor) return updateChrome();
  const now = performance.now() / 1000, cs = cursor;
  const vis = (s) => (s === cs.step ? 1 : s < cs.step ? pastAlpha(cs.step - s) : 0);
  if (brain.cloud && brain.D === D) {
    const c = brain.cloud, I = brain.info;
    for (let p = 0; p < c.base.length; p++) {
      const s = I.step[p], v = vis(s), on = v > 0 && (s < cs.step || (s === cs.step && I.layer[p] < cs.layersDone));
      if (on && !c.shown[p] && s === cs.step) c.born[p] = now;
      c.shown[p] = on;
      c.size[p] = on ? c.base[p] * (s === cs.step ? 1 : 0.75) : 0;
      c.alpha[p] = on ? v : 0;
      c.cur[p] = s === cs.step ? 1 : 0;
    }
    c.dirty();
    const z = tokZ(brain.n, cs.step);
    brain.now.position.z = z;
    brain.paths.forEach((line, s) => {
      const v = vis(s);
      line.visible = style.lines && v > 0;
      line.geometry.setDrawRange(0, s < cs.step ? brain.L : s === cs.step ? Math.min(cs.layersDone, brain.L) : 0);
      line.material.opacity = s === cs.step ? style.lineAlpha * 0.7 : v * style.lineAlpha * 0.6;
    });
    tokenLabel(brain, D, cs.step, 0, 1.14, z);
  }
  if (journey.cloud && journey.D === D) {
    const J = D.journey, S = J.stops.length, c = journey.cloud, reached = (cs.embedDone ? 1 : 0) + cs.layersDone;
    journey.lines.forEach((line, s) => {
      const v = vis(s), count = v > 0 ? (s < cs.step ? S : s === cs.step ? reached : 0) : 0;
      line.visible = style.lines;
      line.geometry.setDrawRange(0, count);
      line.material.opacity = s === cs.step ? Math.min(1, style.lineAlpha * 1.25) : v * style.lineAlpha;
      for (let k = 0; k < S; k++) {
        const p = s * S + k, on = k < count;
        if (on && !c.shown[p] && s === cs.step) c.born[p] = now;
        c.shown[p] = on;
        const head = s === cs.step && k === count - 1;
        c.size[p] = on ? c.base[p] * (head ? 2.4 : s === cs.step ? 1 : 0.7) : 0;
        c.alpha[p] = on ? v : 0;
        c.cur[p] = s === cs.step ? 1 : 0;
      }
    });
    c.dirty();
    if (cs.step < J.points.length && reached) {
      const [x, y, z] = J.points[cs.step][reached - 1];
      tokenLabel(journey, D, cs.step, x, y + 0.12, z);
    }
  }
  updateChrome();
}

// ---------------------------------------------------------------- chrome (header, legend, slider)
function updateChrome() {
  const D = data();
  brain.group.visible = mode === "brain";
  journey.group.visible = mode === "journey";
  document.querySelectorAll("#space [data-mode]").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  $("spEmpty").hidden = !!D && !(mode === "journey" && !D.journey);
  $("spEmpty").innerHTML = !D ? "Run a prompt, or <b>import</b> a .json / .png journey (or drop one here)."
    : "Computing the PCA space of every hidden state…";
  $("spLegend").innerHTML = mode === "brain"
    ? `each dot = one of the ${D?.brain.top_k ?? 16} strongest MLP neurons of a layer · ${style.brainBy === "sign"
        ? `<span style="color:${style.pos}">●</span> positive <span style="color:${style.neg}">●</span> negative`
        : style.brainBy === "layer" ? `color = layer <span style="color:${style.p0}">●</span> → <span style="color:${style.p2}">●</span>` : "color = token"}
      · size = strength within its layer · lines = each layer's strongest neuron, per token`
    : `one line per token: ${style.journeyBy === "token" ? "color = token, brighter = deeper layer"
        : `its hidden state after the embedding <span style="color:${style.p0}">●</span> → each layer <span style="color:${style.p1}">●</span> → last layer <span style="color:${style.p2}">●</span>`} · space = ${SPACE_NOTE}`;
  $("spLegend").title = $("spLegend").textContent;
  const tok = $("spTok");
  if (!D || !cursor) { $("spInfo").textContent = ""; tok.disabled = true; $("spTokLbl").textContent = ""; return; }
  const n = D.tokens.length, t = D.tokens[cursor.step];
  tok.disabled = false; tok.max = n - 1; tok.value = cursor.step;
  $("spTokLbl").innerHTML = `token <b>${cursor.step + 1}</b>/${n} · ${esc(vis1(t.input.text))} → <b>${esc(vis1(t.chosen.text))}</b>`;
  $("spInfo").textContent = mode === "brain"
    ? `${D.model.id} · ${D.model.layers} layers · ${brain.cloud ? brain.cloud.shown.reduce((a, b) => a + b, 0).toLocaleString() : 0} sparks`
    : D.journey ? `${D.model.id} · PC1–3 hold ${(D.journey.explained.reduce((a, b) => a + b, 0) * 100).toFixed(0)}% of the variance` : D.model.id;
  renderStats(D);
  $("spBanner").hidden = !imported;
  if (imported) $("spBannerText").innerHTML = `Imported journey: <b>${esc(imported.model.id)}</b> · ${n} tokens · “${esc(imported.generated.slice(0, 80))}”`;
}

// ---------------------------------------------------------------- insights (tokens 1..current)
const fmt = (x) => x.toLocaleString();
function renderStats(D) {
  const box = $("spStats");
  box.hidden = !style.stats || !D || !cursor;
  if (box.hidden) return;
  const upto = cursor.step, n = D.tokens.length, L = D.model.layers, t = D.tokens[upto];
  const row = (k, v, tip = "") => `<div${tip ? ` title="${esc(tip)}"` : ""}><span>${k}</span><b>${v}</b></div>`;
  const rows = [row("tokens", `${upto + 1} / ${n} generated · ${fmt(t.position + 1)} in context`,
    "generated tokens replayed so far / total · context length when this token was produced")];
  if (mode === "brain") {
    const sp = D.brain.sparks, widths = D.model.mlp_n, k = D.brain.top_k;
    let total = 0, pos = 0, best = null;
    const perLayer = new Float64Array(L), recur = new Map();
    for (let s = 0; s <= upto; s++) sp[s].forEach((l, li) => {
      const done = s < upto || li < cursor.layersDone;
      if (!done) return;
      for (let j = 0; j < l.length; j += 2) {
        const a = l[j + 1], key = li * 1e7 + l[j];
        total++; if (a > 0) pos++;
        perLayer[li] += Math.abs(a);
        recur.set(key, (recur.get(key) || 0) + 1);
        if (!best || Math.abs(a) > Math.abs(best.a)) best = { a, s, li, nrn: l[j] };
      }
    });
    const neurons = widths.reduce((a, b) => a + b, 0);
    const wMin = Math.min(...widths), wMax = Math.max(...widths);
    rows.push(row("model", `${L} layers × ${wMin === wMax ? fmt(wMin) : `${fmt(wMin)}–${fmt(wMax)}`} MLP neurons = ${fmt(neurons)}`),
      row("activations", `${fmt(total)} sparks shown · top ${k}/layer = ${(100 * k / (neurons / L)).toFixed(2)}% of each layer`,
        "each spark is one of the strongest MLP neurons of one layer for one token"),
      row("sign", total ? `${(100 * pos / total).toFixed(0)}% positive · ${(100 * (total - pos) / total).toFixed(0)}% negative` : "–"));
    if (best) rows.push(row("strongest", `${best.a.toPrecision(4)} · L${best.li} n${best.nrn} · token ${best.s + 1} ${esc(vis1(D.tokens[best.s].chosen.text))}`));
    let peak = 0;
    perLayer.forEach((v, i) => { if (v > perLayer[peak]) peak = i; });
    if (total) rows.push(row("loudest layer", `L${peak} (${(100 * perLayer[peak] / perLayer.reduce((a, b) => a + b, 0)).toFixed(0)}% of all |activation|)`));
    let rk = null, rc = 0;
    recur.forEach((c, key) => { if (c > rc) { rc = c; rk = key; } });
    if (rk != null && upto > 0) rows.push(row("most recurring", `L${Math.floor(rk / 1e7)} n${rk % 1e7} · top-${k} in ${rc}/${upto + 1} tokens`,
      "the neuron that shows up in a layer's strongest set for the most tokens so far"));
  } else if (D.journey) {
    const J = D.journey, ju = Math.min(upto, J.points.length - 1), pts = J.points[ju], S = J.stops.length;
    const reached = Math.max(1, (cursor.embedDone ? 1 : 0) + cursor.layersDone);
    let len = 0, jump = 0, at = 0;
    for (let k = 1; k < reached; k++) {
      const d = Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1], pts[k][2] - pts[k - 1][2]);
      len += d; if (d > jump) { jump = d; at = k; }
    }
    const nm = J.norms?.[ju];
    rows.push(row("model", `${L} layers · hidden ${fmt(D.model.hidden_size)} dims → 3 PCs`),
      row("variance", `PC1 ${(J.explained[0] * 100).toFixed(0)}% · PC2 ${(J.explained[1] * 100).toFixed(0)}% · PC3 ${(J.explained[2] * 100).toFixed(0)}%`),
      row("points", `${fmt(Math.min(upto + 1, J.points.length) * S)} hidden states (${S} per token)`),
      row("path length", `${len.toFixed(2)} (this token, in PC units)`));
    if (jump) rows.push(row("biggest move", `${at === 1 ? "embedding → L0" : `L${at - 2} → L${at - 1}`} (${jump.toFixed(2)})`,
      "the layer that moved this token's hidden state the most in the 3D space"));
    if (nm) rows.push(row("‖h‖", `${(+nm[0]).toFixed(1)} → ${(+nm[reached - 1]).toFixed(1)}`, "length of the hidden state: after the embedding → now"));
  }
  if (t.chosen.prob != null) rows.push(row("chosen", `${esc(vis1(t.chosen.text))} · p = ${(t.chosen.prob * 100).toFixed(1)}%`));
  box.innerHTML = rows.join("");
}

function rebuild() {
  const D = data();
  clear(brain.group); clear(journey.group);
  brain.cloud = journey.cloud = null;
  if (D) { buildBrain(D); buildJourney(D); }
  applyCursor();
}

// ---------------------------------------------------------------- hooks from sim.html
function update(c) {
  if (imported || !meta || !steps.length) return;
  if (!live || live.tokens.length !== steps.length || live.model.id !== meta.model) {
    live = liveData();
    rebuild();
  }
  cursor = cursorOf(c);
  applyCursor();
  scheduleJourney();
}

function reset() {
  clearTimeout(pcaTimer);
  pcaTimer = null; pcaBasis = null; pcaFor = 0; live = null;
  if (!imported) { cursor = null; rebuild(); }
}

// ---------------------------------------------------------------- hover
const ray = new THREE.Raycaster();
canvas.addEventListener("pointermove", (e) => {
  const D = data(), obj = mode === "brain" ? brain : journey;
  if (!D || !obj.cloud || e.buttons) return hideTip();
  const r = canvas.getBoundingClientRect();
  camera.updateMatrixWorld();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  ray.params.Points.threshold = 0.035;
  const hit = ray.intersectObject(obj.cloud.points).filter((h) => obj.cloud.shown[h.index]).sort((a, b) => a.distanceToRay - b.distanceToRay)[0];
  if (!hit) return hideTip();
  const p = hit.index;
  if (mode === "brain") {
    const I = brain.info, t = D.tokens[I.step[p]];
    showTip(e, `token ${I.step[p] + 1} · ${vis1(t.input.text)} → ${vis1(t.chosen.text)}\nlayer ${I.layer[p]} · neuron n${I.neuron[p]} (${(I.neuron[p] / D.model.mlp_n[I.layer[p]]).toFixed(3)} of ${D.model.mlp_n[I.layer[p]].toLocaleString()})\nactivation ${I.act[p].toPrecision(4)}`);
  } else {
    const J = D.journey, S = J.stops.length, s = Math.floor(p / S), k = p % S, t = D.tokens[s], [x, y, z] = J.points[s][k];
    showTip(e, `token ${s + 1} · ${vis1(t.input.text)} → ${vis1(t.chosen.text)}\n${k === 0 ? "after the embedding" : `after layer ${k - 1}`} · ‖h‖ ${J.norms?.[s]?.[k] ?? "?"}\nPC ${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)}`);
  }
});
canvas.addEventListener("pointerleave", hideTip);

// ---------------------------------------------------------------- render loop
function resize() {
  if (!renderer) return;
  const w = view.clientWidth, h = view.clientHeight;
  if (!w || !h) return;
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const scale = (h * renderer.getPixelRatio()) / (2 * Math.tan((camera.fov * Math.PI) / 360));
  for (const c of [brain.cloud, journey.cloud]) if (c) c.points.material.uniforms.uScale.value = scale;
  uScale = scale;
}
let uScale = 400;
new ResizeObserver(resize).observe(view);

function loop(t) {
  requestAnimationFrame(loop);
  if (!renderer || ui.fold.sp || !view.clientHeight) return;
  controls.autoRotate = $("spSpin").checked;
  controls.update();
  for (const c of [brain.cloud, journey.cloud]) if (c) { c.points.material.uniforms.uTime.value = t / 1000; c.points.material.uniforms.uScale.value = uScale; }
  renderer.render(scene, camera);
}
requestAnimationFrame(loop);

// ---------------------------------------------------------------- export / import
function exportData() {
  if (!imported) journeyNow();
  const D = data();
  return { ...D, exported: new Date().toISOString(),
           view: { mode, style, camera: { position: camera.position.toArray().map((x) => +x.toFixed(3)), target: controls.target.toArray().map((x) => +x.toFixed(3)) } } };
}
const fileBase = (D) => `journey-${D.model.id.split("/").pop()}-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}`;

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return (bytes) => { let c = 0xffffffff; for (const b of bytes) c = t[(c ^ b) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
})();
const b64 = (bytes) => { let s = ""; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000)); return btoa(s); };

// PNG with the journey JSON in a tEXt chunk (keyword "virtualcuda", base64 UTF-8 JSON), before IEND.
function withText(png, json) {
  const text = new TextEncoder().encode(`${PNG_KEY}\0${b64(new TextEncoder().encode(json))}`);
  const chunk = new Uint8Array(12 + text.length), dv = new DataView(chunk.buffer);
  dv.setUint32(0, text.length);
  chunk.set(new TextEncoder().encode("tEXt"), 4);
  chunk.set(text, 8);
  dv.setUint32(8 + text.length, CRC(chunk.subarray(4, 8 + text.length)));
  const iend = png.length - 12;
  const out = new Uint8Array(png.length + chunk.length);
  out.set(png.subarray(0, iend)); out.set(chunk, iend); out.set(png.subarray(iend), iend + chunk.length);
  return out;
}

function textFromPng(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset), dec = new TextDecoder("latin1");
  for (let o = 8; o + 8 <= bytes.length;) {
    const len = dv.getUint32(o), type = dec.decode(bytes.subarray(o + 4, o + 8));
    if (type === "tEXt") {
      const body = dec.decode(bytes.subarray(o + 8, o + 8 + len)), cut = body.indexOf("\0");
      if (body.slice(0, cut) === PNG_KEY) return new TextDecoder().decode(Uint8Array.from(atob(body.slice(cut + 1)), (c) => c.charCodeAt(0)));
    }
    o += 12 + len;
  }
  return null;
}

$("spExportJson").onclick = () => {
  if (!data()) return;
  const D = exportData();
  download(new Blob([JSON.stringify(D)], { type: "application/json" }), `${fileBase(D)}.json`);
};

$("spExportPng").onclick = async () => {
  if (!data() || !renderer) return;
  const D = exportData();
  renderer.render(scene, camera);
  const src = renderer.domElement, W = src.width, k = W / 900, head = Math.round(70 * k), foot = Math.round(64 * k);
  const out = document.createElement("canvas");
  out.width = W; out.height = src.height + head + foot;
  const ctx = out.getContext("2d");
  ctx.fillStyle = style.bg; ctx.fillRect(0, 0, out.width, out.height);
  ctx.drawImage(src, 0, head);
  ctx.textBaseline = "top";
  ctx.font = `600 ${18 * k}px ui-monospace, Menlo, monospace`; ctx.fillStyle = "#e6e8eb";
  ctx.fillText(`VirtualCuda · ${D.model.id} · ${mode === "brain" ? "brain (MLP sparks)" : "journey (residual stream)"}`, 16 * k, 14 * k);
  ctx.font = `${12 * k}px ui-monospace, Menlo, monospace`; ctx.fillStyle = "#8a919c";
  ctx.fillText(mode === "brain" ? `x = layer (${D.model.layers}) · y = neuron / MLP width · z = token (${D.tokens.length}) · top ${D.brain.top_k} neurons per layer`
    : `${D.tokens.length} tokens · embedding → ${D.model.layers} layers · PCA axes hold ${(D.journey.explained.reduce((a, b) => a + b, 0) * 100).toFixed(0)}% of variance`, 16 * k, 40 * k);
  ctx.fillStyle = "#5eead4"; ctx.font = `${13 * k}px ui-monospace, Menlo, monospace`;
  const gen = D.generated.replace(/\s+/g, " ").trim();
  ctx.fillText(`“${gen.length > 110 ? gen.slice(0, 110) + "…" : gen}”`, 16 * k, src.height + head + 12 * k);
  ctx.fillStyle = "#4b525d"; ctx.font = `${11 * k}px ui-monospace, Menlo, monospace`;
  ctx.fillText("full journey data is embedded in this PNG: import it in /sim to reopen", 16 * k, src.height + head + 36 * k);
  const blob = await new Promise((r) => out.toBlob(r, "image/png"));
  const png = withText(new Uint8Array(await blob.arrayBuffer()), JSON.stringify(D));
  download(new Blob([png], { type: "image/png" }), `${fileBase(D)}.png`);
};

async function importFile(file) {
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const isPng = bytes[0] === 0x89 && bytes[1] === 0x50;
    const text = isPng ? textFromPng(bytes) : new TextDecoder().decode(bytes);
    if (!text) throw new Error("This PNG has no VirtualCuda journey data inside.");
    const D = JSON.parse(text);
    if (D.format !== FORMAT || !D.brain?.sparks || !D.tokens?.length) throw new Error("Not a VirtualCuda journey file.");
    imported = D;
    cursor = { step: D.tokens.length - 1, layersDone: D.model.layers, embedDone: true };
    if (D.view?.mode) setMode(D.view.mode, false);
    if (D.view?.style) setStyle(D.view.style, false);
    if (D.view?.camera) { camera.position.fromArray(D.view.camera.position); controls.target.fromArray(D.view.camera.target); }
    if (ui.fold.sp) toggleFold("sp");
    rebuild();
  } catch (e) {
    alert(`Couldn't import ${file.name}: ${e.message}`);
  }
}

$("spImport").onclick = () => $("spFile").click();
$("spFile").onchange = (e) => { if (e.target.files[0]) importFile(e.target.files[0]); e.target.value = ""; };
const space = $("space");
space.addEventListener("dragover", (e) => { e.preventDefault(); space.classList.add("drop"); });
space.addEventListener("dragleave", (e) => { if (!space.contains(e.relatedTarget)) space.classList.remove("drop"); });
space.addEventListener("drop", (e) => {
  e.preventDefault(); space.classList.remove("drop");
  if (e.dataTransfer.files[0]) importFile(e.dataTransfer.files[0]);
});
$("spBannerClose").onclick = () => {
  imported = null; cursor = null; live = null;
  style = cleanStyle(ui.spStyle);
  applyPalette(); syncStylePanel();
  rebuild();
  if (meta && steps.length) update(cur());
};

// ---------------------------------------------------------------- controls
function setMode(m, save = true) {
  mode = m;
  if (save) { ui.spMode = m; saveUi(); }
  applyCursor();
  scheduleJourney();
}
document.querySelectorAll("#space [data-mode]").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
$("spHome").onclick = () => { camera.position.copy(HOME); controls.target.set(0, 0, 0); };
$("spTok").oninput = (e) => {
  const s = +e.target.value;
  if (imported) { cursor = { step: s, layersDone: imported.model.layers, embedDone: true }; applyCursor(); }
  else showStepEnd(s);
};
// Unfolding the panel may need the journey computed.
document.addEventListener("click", (e) => { if (e.target.closest("#space [data-fold]")) setTimeout(scheduleJourney); });

// Style panel: palette swatches + every [data-k] input maps to style[k].
const stylePanel = $("spStyle");
$("spStyleBtn").onclick = () => { stylePanel.hidden = !stylePanel.hidden; $("spStyleBtn").classList.toggle("on", !stylePanel.hidden); };
$("spPresets").innerHTML = Object.entries(PRESETS).map(([k, p]) =>
  `<button data-preset="${k}" title="${k}" style="background:linear-gradient(90deg,${p.neg} 0 25%,${p.p0} 25% 50%,${p.p1} 50% 75%,${p.pos} 75%)"></button>`).join("");
stylePanel.addEventListener("click", (e) => {
  const b = e.target.closest("[data-preset]");
  if (b) setStyle({ preset: b.dataset.preset, ...PRESETS[b.dataset.preset] });
});
stylePanel.addEventListener("input", (e) => {
  const el = e.target, k = el.dataset.k;
  if (!k) return;
  const v = el.type === "checkbox" ? el.checked : el.type === "range" || el.type === "number" ? +el.value : el.value;
  if (el.type === "number" && !(v >= 1)) return;
  setStyle({ [k]: v, ...(el.type === "color" ? { preset: "custom" } : {}) });
});
$("spStyleReset").onclick = () => setStyle({ ...DEFAULT_STYLE });

function syncStylePanel() {
  stylePanel.querySelectorAll("[data-k]").forEach((el) => {
    const v = style[el.dataset.k];
    if (el.type === "checkbox") el.checked = v; else if (document.activeElement !== el) el.value = v;
  });
  stylePanel.querySelectorAll("[data-preset]").forEach((b) => b.classList.toggle("on", b.dataset.preset === style.preset));
  $("spTrailN").hidden = style.trail !== "last";
  $("spFadeRow").hidden = style.trail !== "fade";
}

function setStyle(patch, save = true) {
  const build = Object.keys(patch).some((k) => REBUILD.has(k) && patch[k] !== style[k]);
  style = cleanStyle({ ...style, ...patch });
  if (save) { ui.spStyle = style; saveUi(); }
  applyPalette();
  syncStylePanel();
  if (build) rebuild(); else applyCursor();
}
syncStylePanel();

window.Space = { update, reset };
resize();
if (meta && steps.length) update(cur()); else applyCursor();

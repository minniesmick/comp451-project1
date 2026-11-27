// Model Lab: reads outputs/*.json + models/*/train_log.json, live tokenize/predict via serve.py. No dependencies.
const MODELS = [
  { key: "model1_char", short: "Model 1", name: "char tokenizer", color: "var(--m1)", scratch: true },
  { key: "model2_bpe", short: "Model 2", name: "BPE tokenizer", color: "var(--m2)", scratch: true },
  { key: "model3_distilbert", short: "Model 3", name: "DistilBERT fine-tuned", color: "var(--m3)", scratch: false },
];
const CLASSES = ["negative", "neutral", "positive"];
const $ = (s, r = document) => r.querySelector(s);
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") el.className = v;
    else if (k === "style") el.setAttribute("style", v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k.startsWith("aria-") || k.startsWith("data-")) { if (v != null) el.setAttribute(k, String(v)); }
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null) el.append(k.nodeType ? k : document.createTextNode(k));
  return el;
};
const f2 = (x) => (x == null ? "–" : x.toFixed(3));
const getJSON = (url) => fetch(url, { cache: "no-store" }).then((r) => (r.ok ? r.json() : Promise.reject(new Error(url))));
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

$("#theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = (root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark";
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
});

const state = { summary: null, outs: {}, logs: {}, ready: false };

async function load() {
  const tryJSON = (u) => getJSON(u).catch(() => null);
  state.summary = await tryJSON("/outputs/summary.json");
  state.w2v = await tryJSON("/outputs/word2vec.json");     // optional extra: tools/word2vec_baseline.py
  for (const m of MODELS) {
    state.outs[m.key] = await tryJSON(`/outputs/${m.key}.json`);
    state.logs[m.key] = await tryJSON(`/models/${m.key}/train_log.json`);
  }
  state.ready = MODELS.every((m) => state.outs[m.key]);
}

const emptyBox = () => h("div", { class: "box empty" }, "No results yet. Run ", h("code", {}, "bash run.sh"), " first, then reload.");
const modelDot = (m) => h("span", { class: "dot", style: `background:${m.color}` });
const rankLabel = (m) => h("span", { class: "rl" }, modelDot(m), h("b", {}, m.short), h("small", {}, m.name));
const modelLabel = (m) => h("span", {}, modelDot(m), `${m.short} · ${m.name}`);

// ───────────────────────── Compare
function renderCompare(root) {
  root.replaceChildren();
  if (!state.ready) return root.append(emptyBox());
  const stats = MODELS.map((m) => ({ m, t: state.outs[m.key].metrics }));
  const best = [...stats].sort((a, b) => b.t.macro_f1 - a.t.macro_f1)[0];
  const second = [...stats].sort((a, b) => b.t.macro_f1 - a.t.macro_f1)[1];
  const n = state.outs[MODELS[0].key].num_samples;
  root.append(
    h("h2", {}, `Results on ${n} held-out tweets`),
    h("p", { class: "lede" }, "Same test set, same metrics, three models. Models 1 and 2 are small transformers trained from scratch on the training split; Model 3 starts from a pretrained checkpoint and is fine-tuned."),
    h("p", { class: "finding" }, modelLabel(best.m), " leads on macro-F1 at ", h("span", { class: "num" }, best.t.macro_f1.toFixed(3)), `, ${(best.t.macro_f1 - second.t.macro_f1).toFixed(3)} ahead of ${second.m.short}.`,
      best.t.macro_f1 - second.t.macro_f1 < 0.02 ? ` With ${n} tweets that gap is within noise.` : ""),
  );
  root.append(h("div", { class: "grid g2", style: "margin-bottom:28px" }, rankBox(), classPlot()), metricsBox(stats));

  const tbl = h("table", {}, h("thead", {}, h("tr", {}, ["Model", "Tokenizer", "Weights", "Parameters", "Epochs", "Best val F1", "Inference"].map((t, i) => h("th", { class: i > 2 ? "n" : "" }, t)))));
  const body = h("tbody");
  for (const m of MODELS) {
    const lg = state.logs[m.key] || {};
    body.append(h("tr", {}, h("td", {}, modelLabel(m)), h("td", {}, lg.tokenizer || "–"), h("td", {}, m.scratch ? "from scratch (MLM, then classifier)" : "pretrained, fine-tuned"),
      h("td", { class: "n" }, lg.parameters ? `${(lg.parameters / 1e6).toFixed(1)} M` : "–"), h("td", { class: "n" }, lg.epochs_run ?? "–"),
      h("td", { class: "n" }, f2(lg.best_val_macro_f1)), h("td", { class: "n" }, `${state.outs[m.key].inference_seconds} s`)));
  }
  tbl.append(body);
  root.append(h("div", { class: "box", style: "margin-top:20px" }, h("h3", {}, "Setup"), tbl));
  requestAnimationFrame(() => requestAnimationFrame(() => root.querySelectorAll(".pt").forEach((p) => (p.style.left = p.dataset.left))));
}
// Ranking: pick a metric, bars re-order (FLIP-style) and numbers count up
const RANK_METRICS = [["accuracy", "Accuracy", (t) => t.accuracy], ["macro_f1", "Macro-F1", (t) => t.macro_f1], ["weighted_f1", "Weighted-F1", (t) => t.weighted_f1],
  ["negative", "F1 negative", (t) => t.per_class.negative.f1], ["neutral", "F1 neutral", (t) => t.per_class.neutral.f1], ["positive", "F1 positive", (t) => t.per_class.positive.f1]];
function rankBox() {
  const ROW = matchMedia("(max-width: 720px)").matches ? 74 : 54;
  let metric = RANK_METRICS[1];
  const rows = {};
  const stage = h("div", { class: "rank", style: `height:${ROW * MODELS.length}px` });
  for (const m of MODELS) {
    rows[m.key] = { shown: 0, el: h("div", { class: "rank-row", style: `height:${ROW - 8}px` }, h("span", { class: "rank-n" }), rankLabel(m),
      h("div", { class: "track" }, h("div", { class: "fill", style: `background:${m.color}` })), h("b", { class: "v" }, "0.000")) };
    stage.append(rows[m.key].el);
  }
  const countTo = (r, to) => {
    const from = r.shown, t0 = performance.now(), v = $(".v", r.el);
    if (reduce) { v.textContent = f2(to); r.shown = to; return; }
    const tick = (now) => { const k = Math.min(1, (now - t0) / 600), e = 1 - Math.pow(1 - k, 3); r.shown = from + (to - from) * e; v.textContent = f2(r.shown); if (k < 1) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  };
  const layout = () => {
    const vals = MODELS.map((m) => ({ m, v: metric[2](state.outs[m.key].metrics) })).sort((a, b) => b.v - a.v);
    vals.forEach(({ m, v }, i) => {
      const r = rows[m.key];
      r.el.style.transform = `translateY(${i * ROW}px)`;
      $(".rank-n", r.el).textContent = i + 1;
      $(".fill", r.el).style.width = `${(v * 100).toFixed(1)}%`;
      countTo(r, v);
    });
  };
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Metric" }, RANK_METRICS.map((mt) => h("button", { "aria-pressed": mt === metric, onclick: (e) => {
    metric = mt; seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b === e.currentTarget)); layout(); } }, mt[1])));
  requestAnimationFrame(() => requestAnimationFrame(layout));
  return h("div", { class: "box" }, h("div", { class: "box-head" }, h("h3", {}, "Ranking"), seg), stage);
}
// per-class F1 as a dot plot: one row per class, one dot per model, ring on the best
function classPlot() {
  const lo = 0.4, hi = 0.85, at = (v) => `${(((v - lo) / (hi - lo)) * 100).toFixed(2)}%`;
  const ticks = [0.5, 0.6, 0.7, 0.8];
  const grid = ticks.map((t) => `linear-gradient(var(--line), var(--line)) ${at(t)} 0 / 1px 100% no-repeat`).join(", ");
  const rows = CLASSES.map((c) => {
    const vals = MODELS.map((m) => ({ m, v: state.outs[m.key].metrics.per_class[c].f1 })), top = Math.max(...vals.map((x) => x.v)), min = Math.min(...vals.map((x) => x.v));
    const track = h("div", { class: "dot-track", style: `background:${grid}` }, h("i", { class: "span", style: `left:${at(min)};width:calc(${at(top)} - ${at(min)})` }));
    vals.forEach(({ m, v }, i) => {
      const text = `${m.short} · ${m.name}: ${f2(v)}`;
      const pt = h("button", { class: `pt${v === top ? " top" : ""}`, type: "button", style: `--c:${m.color};--d:${i * 70}ms`, "data-left": at(v), "aria-label": `${c}, ${text}` });
      pt.addEventListener("pointerenter", (e) => showTip([h("div", {}, h("span", { class: "dot", style: `background:${m.color}` }), text)], e.clientX, e.clientY));
      pt.addEventListener("pointermove", (e) => showTip([h("div", {}, h("span", { class: "dot", style: `background:${m.color}` }), text)], e.clientX, e.clientY));
      pt.addEventListener("pointerleave", hideTip);
      pt.addEventListener("focus", () => { const r = pt.getBoundingClientRect(); showTip([h("div", {}, text)], r.left + r.width / 2, r.top); });
      pt.addEventListener("blur", hideTip);
      track.append(pt);
    });
    return h("div", { class: "dot-row" }, h("div", { class: "lab" }, c, h("small", {}, `n=${state.outs[MODELS[0].key].metrics.per_class[c].support}`)), track);
  });
  const axis = h("div", { class: "dot-axis" }, ticks.map((t) => h("span", { style: `left:${at(t)}` }, t.toFixed(1))));
  return h("div", { class: "box" }, h("div", { class: "box-head" }, h("h3", {}, "F1 per class")), h("div", { class: "legend" }, MODELS.map((m) => h("span", {}, modelLabel(m)))),
    h("div", { class: "dots" }, rows, axis), h("p", { class: "dot-note" }, "Ring marks the best model in each class. Hover or focus a dot for its value."));
}
// every metric for every model in one table; the best value per row is bold
function metricsBox(stats) {
  const rows = [["Accuracy", (t) => t.accuracy], ["Macro-F1", (t) => t.macro_f1], ["Weighted-F1", (t) => t.weighted_f1]];
  const classRows = CLASSES.map((c) => [`${c} (n=${stats[0].t.per_class[c].support})`, (t) => t.per_class[c].f1]);
  const base = state.w2v?.variants?.[state.w2v.chosen_by_validation]?.metrics;      // static Word2Vec baseline, if it was run
  const cols = base ? [...stats, { t: base }] : stats;
  const tr = ([label, get]) => { const vals = cols.map((s) => get(s.t)), top = Math.max(...vals);
    return h("tr", {}, h("td", { class: "k" }, label), vals.map((v) => h("td", { class: `n${v === top ? " best" : ""}` }, f2(v)))); };
  const body = h("tbody", {}, rows.map(tr), h("tr", { class: "grp" }, h("td", { colspan: cols.length + 1 }, "F1 per class")), classRows.map(tr));
  const head = h("thead", {}, h("tr", {}, h("th"), MODELS.map((m) => h("th", { class: "mh" }, modelDot(m), m.short)), base ? h("th", { class: "mh" }, "Word2Vec") : null));
  return h("div", { class: "box", style: "margin-bottom:20px" }, h("h3", {}, "All metrics"), h("table", { class: "mt" }, head, body),
    base ? h("p", { class: "dot-note" }, "Word2Vec is the extra static-vector baseline (average word vector, small classifier); it is not one of the three assignment models.") : null);
}

// ───────────────────────── line chart helper
const tip = $("#tip");
function showTip(html, x, y) {
  tip.replaceChildren(...html);
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(x + 14, innerWidth - r.width - 8)}px`;
  tip.style.top = `${Math.max(8, y - r.height - 10)}px`;
}
const hideTip = () => (tip.hidden = true);

function lineChart({ series, yMin, yMax, yTicks, xMax, yFmt = (v) => v.toFixed(2), title }) {
  const W = 560, H = 260, L = 44, R = 14, T = 12, B = 30;
  const X = (x) => L + ((x - 1) / Math.max(1, xMax - 1)) * (W - L - R);
  const Y = (y) => T + (1 - (y - yMin) / (yMax - yMin)) * (H - T - B);
  const NS = "http://www.w3.org/2000/svg";
  const s = (tag, attrs = {}) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
  const gGrid = s("g", { class: "grid" });
  for (let i = 0; i <= yTicks; i++) {
    const v = yMin + ((yMax - yMin) * i) / yTicks;
    gGrid.append(s("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v) }));
    const t = s("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end" }); t.textContent = yFmt(v); gGrid.append(t);
  }
  const step = xMax > 12 ? 2 : 1;
  for (let x = 1; x <= xMax; x += step) { const t = s("text", { x: X(x), y: H - 8, "text-anchor": "middle" }); t.textContent = x; gGrid.append(t); }
  const lbl = s("text", { x: L - 10, y: H - 8, "text-anchor": "end" }); lbl.textContent = "epoch"; gGrid.append(lbl);
  svg.append(gGrid);
  const hover = s("line", { class: "hover-line", y1: T, y2: H - B, opacity: 0 });
  svg.append(hover);
  for (const sr of series) {
    const d = sr.points.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
    const path = s("path", { d, class: `path${sr.dash ? " dash" : ""}`, stroke: sr.color });
    svg.append(path);
    if (!reduce && !sr.dash) { path.setAttribute("pathLength", "1"); path.style.setProperty("--len", 1); path.classList.add("draw"); }   // normalised length: the draw-on animation always covers the whole line
    if (sr.points.length <= 4 && !sr.dash) for (const p of sr.points) svg.append(s("circle", { class: "pt-dot", cx: X(p.x), cy: Y(p.y), r: 3.5, fill: sr.color }));
    if (sr.best != null) { const p = sr.points.find((q) => q.x === sr.best); if (p) svg.append(s("circle", { class: "best", cx: X(p.x), cy: Y(p.y), r: 5, fill: sr.color })); }
  }
  const wrap = h("div", { class: "chart" }, svg);
  svg.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const ep = Math.max(1, Math.min(xMax, Math.round(1 + ((px - L) / (W - L - R)) * (xMax - 1))));
    hover.setAttribute("x1", X(ep)); hover.setAttribute("x2", X(ep)); hover.setAttribute("opacity", 1);
    const rows = series.map((sr) => { const p = sr.points.find((q) => q.x === ep); return p ? h("div", {}, h("span", { class: "dot", style: `background:${sr.color}` }), `${sr.name}: `, h("b", {}, yFmt(p.y))) : null; }).filter(Boolean);
    showTip([h("div", {}, `epoch ${ep}`), ...rows], e.clientX, e.clientY);
  });
  svg.addEventListener("pointerleave", () => { hover.setAttribute("opacity", 0); hideTip(); });
  return wrap;
}

// ───────────────────────── Training
let lossModel = "model3_distilbert";
function renderTrain(root) {
  root.replaceChildren();
  if (!state.ready) return root.append(emptyBox());
  const logs = MODELS.map((m) => ({ m, lg: state.logs[m.key] })).filter((x) => x.lg);
  if (!logs.length) return root.append(emptyBox());
  const xMax = Math.max(...logs.map((x) => x.lg.history.length));
  const bestEpoch = (lg) => lg.history.reduce((b, e) => (e.val_macro_f1 > (b?.val_macro_f1 ?? -1) ? e : b), null).epoch;
  root.append(h("h2", {}, "How each model learned"), h("p", { class: "lede" }, "Validation macro-F1 picks the saved checkpoint (dot). Hover a chart to read a single epoch. Model 3 trains for 3 epochs; the from-scratch models train longer and have to learn language first."));
  const f1 = lineChart({ title: "Validation macro-F1 by epoch", yMin: 0.3, yMax: 0.8, yTicks: 5, xMax,
    series: logs.map(({ m, lg }) => ({ name: `${m.short}`, color: m.color, best: bestEpoch(lg), points: lg.history.map((e) => ({ x: e.epoch, y: e.val_macro_f1 })) })) });
  const box1 = h("div", { class: "box" }, h("h3", {}, "Validation macro-F1"), h("div", { class: "legend" }, logs.map(({ m }) => h("span", {}, modelLabel(m)))), f1);

  const lossBox = h("div", { class: "box" });
  const drawLoss = () => {
    const x = logs.find((l) => l.m.key === lossModel) || logs[0];
    const ys = x.lg.history.flatMap((e) => [e.train_loss, e.val_loss]);
    const yMin = Math.floor(Math.min(...ys) * 10) / 10, yMax = Math.ceil(Math.max(...ys) * 10) / 10;
    lossBox.replaceChildren(h("h3", {}, "Training vs validation loss"),
      h("div", { class: "seg", style: "margin-bottom:12px" }, logs.map(({ m }) => h("button", { "aria-pressed": m.key === x.m.key, onclick: () => { lossModel = m.key; drawLoss(); } }, m.short))),
      h("div", { class: "legend", style: "margin-top:12px" }, h("span", {}, h("span", { class: "dot", style: `background:${x.m.color}` }), "solid: training"), h("span", {}, h("span", { class: "dot", style: `background:${x.m.color};opacity:.45` }), "dashed: validation")),
      lineChart({ title: "Loss by epoch", yMin, yMax, yTicks: 4, xMax: x.lg.history.length,
        series: [{ name: "train loss", color: x.m.color, points: x.lg.history.map((e) => ({ x: e.epoch, y: e.train_loss })) }, { name: "val loss", color: x.m.color, dash: true, points: x.lg.history.map((e) => ({ x: e.epoch, y: e.val_loss })) }] }));
  };
  drawLoss();
  const grid = h("div", { class: "grid g2" }, box1, lossBox);
  root.append(grid);
  const mlm = logs.filter((x) => x.lg.mlm_history && x.lg.mlm_history.length);
  if (mlm.length) {
    const mx = Math.max(...mlm.map((x) => x.lg.mlm_history.length));
    const ys = mlm.flatMap((x) => x.lg.mlm_history.map((e) => e.mlm_loss));
    const chart = h("div", { class: "box" }, h("h3", {}, "Masked-token pretraining loss"), h("div", { class: "legend" }, mlm.map(({ m }) => h("span", {}, modelLabel(m)))),
      lineChart({ title: "MLM loss by epoch", yMin: Math.floor(Math.min(...ys) * 2) / 2, yMax: Math.ceil(Math.max(...ys) * 2) / 2, yTicks: 4, xMax: mx,
        series: mlm.map(({ m, lg }) => ({ name: m.short, color: m.color, points: lg.mlm_history.map((e) => ({ x: e.epoch, y: e.mlm_loss })) })) }));
    const note = h("div", { class: "box" }, h("h3", {}, "Why only two models"), h("p", { class: "note" }, "Models 1 and 2 start with random weights, so they first learn language by predicting hidden tokens, then learn sentiment. Model 3 already knows English and skips this step."),
      h("p", { class: "note" }, "A lower loss means better guesses at the hidden tokens. The two vocabularies differ (characters against subwords), so compare the shape of the curves, not the absolute values."));
    root.append(h("div", { class: "grid g2", style: "margin-top:20px" }, chart, note));
  }
}

// ───────────────────────── Tokenizers
const SAMPLE = ["Can't wait for the weekend, finally some sunshine #happy", "worst customer service ever, never ordering again!!", "Meeting moved to 3pm tomorrow in room B12", "lol this is soooo goood 😂😂"];
function examples() {
  const o = state.outs[MODELS[2].key];
  if (!o) return SAMPLE;
  const pick = [];
  for (const c of CLASSES) { const p = o.predictions.find((q) => q.gold === c && q.text.length > 25 && q.text.length < 110); if (p) pick.push(p.text); }
  return [...pick, SAMPLE[3]];
}
async function api(path, text) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "request failed");
  return j;
}
function inputBlock(onSubmit, placeholder, label) {
  const inp = h("input", { type: "text", maxlength: 280, placeholder, "aria-label": placeholder });
  const btn = h("button", { class: "btn", onclick: () => onSubmit(inp.value, btn) }, label);
  inp.addEventListener("keydown", (e) => e.key === "Enter" && onSubmit(inp.value, btn));
  const chips = h("div", { class: "chips" }, examples().map((t) => h("button", { class: "chip", title: t, onclick: () => { inp.value = t; onSubmit(t, btn); } }, t)));
  return { inp, btn, node: h("div", {}, h("div", { class: "field" }, inp, btn), chips) };
}
function renderTokens(root) {
  root.replaceChildren();
  const out = h("div");
  const err = h("div", { class: "err", role: "alert" });
  const run = async (text, btn) => {
    if (!text.trim()) return;
    btn.disabled = true; err.textContent = ""; err.classList.remove("info");
    const slow = setTimeout(() => { err.classList.add("info"); err.textContent = "Loading the models. The first request takes a few seconds…"; }, 600);
    try {
      const data = await api("/api/tokenize", text);
      out.replaceChildren(...MODELS.filter((m) => data[m.key]).map((m) => {
        const d = data[m.key];
        const row = h("div", { class: "toks", style: `--c:${m.color}` });
        d.tokens.forEach((t, i) => row.append(h("span", { class: `tk c${i % 2}${t.ws ? " ws" : ""}${t.special ? " sp" : ""}`, style: `animation-delay:${reduce ? 0 : Math.min(i * 6, 500)}ms` }, t.t)));
        return h("div", { class: "tokrow" }, h("div", { class: "head" }, modelLabel(m), h("span", { class: "num" }, `${d.count} tokens`)), row);
      }));
    } catch (e) { err.textContent = `Could not tokenize: ${e.message}. Is serve.py running with trained models?`; }
    clearTimeout(slow); err.classList.remove("info"); if (err.textContent.startsWith("Loading the models")) err.textContent = "";
    btn.disabled = false;
  };
  const ib = inputBlock(run, "Type a tweet…", "Tokenize");
  root.append(h("h2", {}, "Same text, three tokenizers"), h("p", { class: "lede" }, "Characters give the longest sequences and no vocabulary problem. BPE merges frequent pieces into a few dozen tokens. WordPiece (Model 3) marks word continuations with ##. Shorter sequences are cheaper for attention, but rare words get split."), ib.node, err, out);
  run(examples()[0], ib.btn); ib.inp.value = examples()[0];
}

// ───────────────────────── Live prediction
const DICE_SVG = '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><rect x="3.5" y="3.5" width="17" height="17" rx="4"/><circle cx="8.5" cy="8.5" r="1.1" fill="currentColor"/><circle cx="15.5" cy="8.5" r="1.1" fill="currentColor"/><circle cx="12" cy="12" r="1.1" fill="currentColor"/><circle cx="8.5" cy="15.5" r="1.1" fill="currentColor"/><circle cx="15.5" cy="15.5" r="1.1" fill="currentColor"/></svg>';
function renderLive(root) {
  root.replaceChildren();
  const out = h("div", { class: "box" });
  const err = h("div", { class: "err", role: "alert" });
  const run = async (text, btn, gold) => {
    if (!text.trim()) return;
    btn.disabled = true; err.textContent = ""; err.classList.remove("info");
    const slow = setTimeout(() => { err.classList.add("info"); err.textContent = "Loading the models. The first request takes a few seconds…"; }, 600);
    try {
      const data = await api("/api/predict", text);
      const rows = MODELS.filter((m) => data[m.key]).map((m) => {
        const d = data[m.key];
        const stack = h("div", { class: "stack", role: "img", "aria-label": CLASSES.map((c) => `${c} ${(d.scores[c] * 100).toFixed(0)}%`).join(", ") }, CLASSES.map((c) => h("i", { class: c.slice(0, 3), "data-w": `${(d.scores[c] * 100).toFixed(1)}%`, title: c })));
        const pills = h("div", { class: "pills" }, CLASSES.map((c) => h("span", {}, `${c.slice(0, 3)} ${(d.scores[c] * 100).toFixed(0)}%`)));
        return h("div", { class: "pred" }, modelLabel(m), h("div", {}, stack, pills), h("span", {}, h("span", { class: `badge ${d.label}` }, d.label), gold ? h("span", { class: `mini ${d.label === gold ? "ok" : "bad"}`, style: "margin-left:8px" }, d.label === gold ? "correct" : "wrong") : null));
      });
      out.replaceChildren(h("div", { class: "box-head" }, h("h3", {}, "Predicted probabilities"), gold ? h("span", { class: "muted" }, "Gold label: ", h("span", { class: `badge ${gold}` }, gold)) : null), ...rows);
      requestAnimationFrame(() => requestAnimationFrame(() => out.querySelectorAll("i[data-w]").forEach((i) => (i.style.width = i.dataset.w))));
    } catch (e) { err.textContent = `Could not predict: ${e.message}. Is serve.py running with trained models?`; }
    clearTimeout(slow); err.classList.remove("info"); if (err.textContent.startsWith("Loading the models")) err.textContent = "";
    btn.disabled = false;
  };
  const ib = inputBlock(run, "Type a tweet…", "Predict");
  const pool = state.outs[MODELS[2].key]?.predictions || [];
  if (pool.length) {
    let last = -1;
    const dice = h("button", { class: "dice", title: "Random tweet from the test set", "aria-label": "Random tweet from the test set" });
    dice.innerHTML = DICE_SVG;
    dice.addEventListener("click", () => {
      let i; do { i = Math.floor(Math.random() * pool.length); } while (i === last && pool.length > 1);
      last = i;
      dice.classList.remove("roll"); void dice.offsetWidth; dice.classList.add("roll");
      ib.inp.value = pool[i].text;
      run(pool[i].text, ib.btn, pool[i].gold);
    });
    ib.node.querySelector(".field").insertBefore(dice, ib.btn);
  }
  out.append(h("p", { class: "muted", style: "margin:0" }, "Results appear here. The dice draws a random tweet from the held-out test set and shows its gold label."));
  root.append(h("h2", {}, "Try the models"), h("p", { class: "lede" }, "All three saved models score the same text through the Hugging Face pipeline. Disagreement between them is the interesting part."), ib.node, err, out);
}

// ───────────────────────── Attention
// /api/attention returns raw weights [layer][head][query][key] per model. Everything below (head mean, rollout, shading) is computed here.
const att = { data: null, text: "", key: MODELS[2].key, layer: null, head: -1, q: 0 };
const meanHeads = (d, l) => {
  const H = d.attn[l], n = d.tokens.length, out = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const m of H) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i][j] += m[i][j] / H.length;
  return out;
};
const weights = (d, l, hd) => (hd < 0 ? meanHeads(d, l) : d.attn[l][hd]);
// attention rollout (Abnar & Zuidema 2020), CLS row only: v = e0 · A'_L · … · A'_1 with A' = ½A + ½I
function rollout(d) {
  const n = d.tokens.length;
  let v = new Array(n).fill(0); v[0] = 1;
  for (let l = d.layers - 1; l >= 0; l--) {
    const A = meanHeads(d, l), nv = new Array(n).fill(0);
    for (let i = 0; i < n; i++) { if (!v[i]) continue; for (let j = 0; j < n; j++) nv[j] += v[i] * (0.5 * A[i][j] + (i === j ? 0.5 : 0)); }
    v = nv;
  }
  return v;
}
const tokText = (t) => (t.ws ? " " : "") + t.t;
function shadedTokens(d, vals, { color, onpick, sel }) {
  const real = d.tokens.map((t, i) => (t.special ? 0 : vals[i]));
  const max = Math.max(...real, 1e-9);
  const row = h("div", { class: "toks", style: `--c:${color}` });
  d.tokens.forEach((t, i) => {
    const a = t.special ? 0 : Math.min(1, vals[i] / max);
    const el = h(onpick ? "button" : "span", { class: `tk at${t.ws ? " ws" : ""}${t.special ? " sp" : ""}${sel === i ? " sel" : ""}`, style: `--a:${a.toFixed(3)}`, "data-i": i,
      title: `${t.t}  ·  ${vals[i].toFixed(3)}`, ...(onpick ? { type: "button", "aria-pressed": sel === i } : {}) }, t.t);
    if (onpick) el.addEventListener("click", () => onpick(i));
    row.append(el);
  });
  return row;
}
function summaryBox(data) {
  const rows = MODELS.filter((m) => data[m.key]).map((m) => {
    const d = data[m.key], r = rollout(d);
    const real = d.tokens.map((t, i) => ({ t, i, v: r[i] })).filter((x) => !x.t.special);
    const wordy = real.reduce((s, x) => s + x.t.t.replace(/^##/, "").length, 0) / Math.max(1, real.length) >= 2;
    const top = wordy ? real.slice().sort((a, b) => b.v - a.v).slice(0, 3).map((x) => x.t.t.replace(/^##/, "")) : [];
    return h("div", { class: "tokrow" },
      h("div", { class: "head" }, modelLabel(m), h("span", { class: `badge ${d.label}` }, d.label), h("span", { class: "num" }, `${(d.scores[d.label] * 100).toFixed(0)}%`),
        top.length ? h("span", { class: "muted" }, "most weight: ", top.join(", ")) : h("span", { class: "muted" }, "single characters: no one token stands out")),
      shadedTokens(d, r, { color: m.color }));
  });
  return h("div", { class: "box" }, h("div", { class: "box-head" }, h("h3", {}, "What feeds the [CLS] vector")),
    h("p", { class: "note", style: "max-width:68ch" }, "The classifier reads only the [CLS] position. Attention rollout follows [CLS] back through every layer (mean over heads, residual connection included), so darker tokens contributed more to the final decision."), ...rows);
}

function heatmap(d, W, q, onpick) {
  const n = d.tokens.length, cs = Math.max(4, Math.min(22, Math.floor(540 / n))), lab = cs >= 11;
  const left = lab ? 84 : 6, top = lab ? 84 : 6, S = n * cs;
  const parts = [`<rect x="${left}" y="${top}" width="${S}" height="${S}" fill="var(--panel-2)"/>`];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const o = Math.min(1, Math.sqrt(W[i][j]));
    if (o > 0.04) parts.push(`<rect x="${left + j * cs}" y="${top + i * cs}" width="${cs}" height="${cs}" fill="var(--accent)" fill-opacity="${o.toFixed(2)}"/>`);
  }
  parts.push(`<rect x="${left - 0.5}" y="${top + q * cs - 0.5}" width="${S + 1}" height="${cs + 1}" fill="none" stroke="var(--red)" stroke-width="1.5"/>`);
  if (lab) d.tokens.forEach((t, i) => {
    const s = (t.t.length > 9 ? t.t.slice(0, 8) + "…" : t.t).replace(/&/g, "&amp;").replace(/</g, "&lt;");
    parts.push(`<text x="${left - 6}" y="${top + i * cs + cs / 2 + 4}" text-anchor="end" class="hl${i === q ? " on" : ""}">${s}</text>`);
    parts.push(`<text transform="translate(${left + i * cs + cs / 2 + 4},${top - 6}) rotate(-90)" class="hl">${s}</text>`);
  });
  const svg = h("div", { class: "hm" });
  svg.innerHTML = `<svg viewBox="0 0 ${left + S + 6} ${top + S + 6}" role="img" aria-label="Attention matrix, ${n} by ${n} tokens. Rows are the token looking, columns the token looked at." style="width:${Math.min(left + S + 6, 640)}px">${parts.join("")}</svg>`;
  const el = svg.firstChild;
  const cell = (e) => { const r = el.getBoundingClientRect(), k = r.width / el.viewBox.baseVal.width; const x = (e.clientX - r.left) / k - left, y = (e.clientY - r.top) / k - top; const i = Math.floor(y / cs), j = Math.floor(x / cs); return i >= 0 && j >= 0 && i < n && j < n ? [i, j] : null; };
  el.addEventListener("mousemove", (e) => { const c = cell(e); if (!c) return hideTip(); showTip([`${d.tokens[c[0]].t}  →  ${d.tokens[c[1]].t}  ·  ${W[c[0]][c[1]].toFixed(3)}`], e.clientX, e.clientY); });
  el.addEventListener("mouseleave", hideTip);
  el.addEventListener("click", (e) => { const c = cell(e); if (c) onpick(c[0]); });
  return svg;
}

function insideBox() {
  const m = MODELS.find((x) => x.key === att.key), d = att.data[att.key];
  if (att.layer == null || att.layer >= d.layers) att.layer = d.layers - 1;
  if (att.head >= d.heads) att.head = -1;
  if (att.q >= d.tokens.length) att.q = 0;
  const box = h("div", { class: "box" });
  const seg = (items, cur, pick, label) => h("div", { class: "seg", role: "group", "aria-label": label }, items.map(([v, t]) => h("button", { type: "button", "aria-pressed": v === cur, onclick: () => pick(v) }, t)));
  const redraw = () => { hideTip(); box.replaceWith(insideBox()); };
  const W = weights(d, att.layer, att.head), row = W[att.q];
  const q = d.tokens[att.q];
  box.append(
    h("div", { class: "box-head" }, h("h3", {}, "Look inside one model"), seg(MODELS.filter((x) => att.data[x.key]).map((x) => [x.key, x.short]), att.key, (v) => { att.key = v; att.q = 0; redraw(); }, "Model")),
    h("div", { class: "ctl" }, h("span", { class: "muted" }, "Layer"), seg(Array.from({ length: d.layers }, (_, i) => [i, String(i + 1)]), att.layer, (v) => { att.layer = v; redraw(); }, "Layer")),
    h("div", { class: "ctl" }, h("span", { class: "muted" }, "Head"), seg([[-1, "mean"], ...Array.from({ length: d.heads }, (_, i) => [i, String(i + 1)])], att.head, (v) => { att.head = v; redraw(); }, "Head")),
    h("p", { class: "note", style: "max-width:70ch;margin-top:14px" }, "Pick the token that is looking. The tokens below are shaded by how much attention it gives each one (",
      h("b", {}, `${q.t}`), `, layer ${att.layer + 1}, ${att.head < 0 ? "mean of " + d.heads + " heads" : "head " + (att.head + 1)}). The [CLS] token (first) is the one the classifier reads.`),
    shadedTokens(d, row, { color: m.color, onpick: (i) => { att.q = i; redraw(); }, sel: att.q }),
    h("div", { class: "hm-wrap" }, heatmap(d, W, att.q, (i) => { att.q = i; redraw(); }),
      h("p", { class: "axis-note" }, "Rows: the token that looks. Columns: the token it looks at. Each row sums to 1; colour is √weight. Click a row to select it, hover a cell for the value.")));
  return box;
}

function renderAttention(root) {
  root.replaceChildren();
  const out = h("div");
  const err = h("div", { class: "err", role: "alert" });
  const run = async (text, btn) => {
    if (!text.trim()) return;
    btn.disabled = true; err.textContent = ""; err.classList.remove("info");
    const slow = setTimeout(() => { err.classList.add("info"); err.textContent = "Loading the models. The first request takes a few seconds…"; }, 600);
    try {
      att.data = await api("/api/attention", text); att.text = text; att.q = 0;
      out.replaceChildren(summaryBox(att.data), insideBox());
    } catch (e) { err.textContent = `Could not run attention: ${e.message}. Is serve.py running with trained models?`; }
    clearTimeout(slow); if (err.textContent.startsWith("Loading the models")) err.textContent = ""; err.classList.remove("info");
    btn.disabled = false;
  };
  const ib = inputBlock(run, "Type a tweet…", "Show attention");
  root.append(h("h2", {}, "What the models look at"),
    h("p", { class: "lede" }, "Self-attention lets every token read every other token. Each layer has several heads with their own weights. Here the real weights of the three saved models are shown for one tweet. Attention is a view into the model, not a proof of why it decided: heads can look at punctuation or [SEP] just to do nothing."),
    ib.node, err, out);
  run(examples()[0], ib.btn); ib.inp.value = examples()[0];
}

// ───────────────────────── Test set: three models side by side
const view = { cell: null, mode: "all" };
function agreementItems() {
  const preds = MODELS.map((x) => state.outs[x.key].predictions);
  return preds[0].map((p, k) => { const per = MODELS.map((x, i) => preds[i][k]); return { text: p.text, gold: p.gold, per, right: per.filter((q) => q.correct).length }; });
}
function matrixBox(m) {
  const o = state.outs[m.key], cm = o.metrics.confusion_matrix.matrix, max = Math.max(...cm.flat());
  const sel = (i, j) => view.cell && view.cell.k === m.key && view.cell.i === i && view.cell.j === j;
  const table = h("table", { class: "cm sm" }, h("thead", {}, h("tr", {}, h("th"), CLASSES.map((c) => h("th", {}, c.slice(0, 3))))));
  const body = h("tbody");
  cm.forEach((row, i) => body.append(h("tr", {}, h("th", { class: "rowh" }, CLASSES[i].slice(0, 3)), row.map((v, j) => h("td", {}, h("button", {
    style: `--a:${(v / max).toFixed(2)}`, "data-ok": i === j, "aria-pressed": sel(i, j), "aria-label": `${m.short}: gold ${CLASSES[i]}, predicted ${CLASSES[j]}: ${v}`,
    onclick: () => { view.cell = sel(i, j) ? null : { k: m.key, i, j }; drawErrors(); } }, v))))));
  table.append(body);
  return h("div", { class: "box" }, h("h3", {}, modelLabel(m)), table,
    h("p", { class: "axis-note" }, h("b", { class: "num" }, f2(o.metrics.accuracy)), " accuracy · ", h("b", { class: "num" }, f2(o.metrics.macro_f1)), " macro-F1"));
}
function agreementBar(items) {
  const defs = [[3, "all 3 right", "c3"], [2, "2 of 3", "c2"], [1, "1 of 3", "c1"], [0, "none", "c0"]];
  const total = items.length;
  const segs = defs.map(([n, label, key]) => ({ n, label, key, c: items.filter((it) => it.right === n).length }));
  return h("div", { class: "box", style: "margin-bottom:20px" }, h("h3", {}, "How many of the three models get each tweet right"),
    h("div", { class: "agree" }, segs.map((s) => h("button", { class: `ag ${s.key}`, style: `flex:${Math.max(s.c, 0.0001)}`, "aria-pressed": view.mode === s.key, title: `${s.label}: ${s.c}`, onclick: () => { view.mode = view.mode === s.key ? "all" : s.key; view.cell = null; drawErrors(); } }, s.c ? h("span", {}, `${s.c}`) : null))),
    h("div", { class: "agree-legend" }, segs.map((s) => h("span", {}, h("i", { class: `ag ${s.key}` }), `${s.label} · ${s.c} (${Math.round((s.c / total) * 100)}%)`))));
}
function listBox(items) {
  let rows = items;
  if (view.cell) { const mi = MODELS.findIndex((x) => x.key === view.cell.k); rows = rows.filter((it) => it.gold === CLASSES[view.cell.i] && it.per[mi].predicted === CLASSES[view.cell.j]); }
  if (/^c\d$/.test(view.mode)) rows = rows.filter((it) => it.right === +view.mode[1]);
  if (view.mode === "split") rows = rows.filter((it) => new Set(it.per.map((p) => p.predicted)).size > 1);
  const modes = h("div", { class: "seg" }, [["all", "All"], ["split", "Models disagree"], ["c0", "All wrong"], ["c3", "All right"]].map(([k, t]) => h("button", { "aria-pressed": view.mode === k, onclick: () => { view.mode = k; drawErrors(); } }, t)));
  const chips = [];
  if (view.cell) { const m = MODELS.find((x) => x.key === view.cell.k); chips.push(h("button", { class: "chip", onclick: () => { view.cell = null; drawErrors(); } }, `${m.short}: gold ${CLASSES[view.cell.i]} → predicted ${CLASSES[view.cell.j]}  ×`)); }
  const list = h("div", { class: "list" });
  if (!rows.length) list.append(h("div", { class: "empty" }, "Nothing matches this filter."));
  for (const it of rows.slice(0, 150)) {
    list.append(h("div", { class: "row3" }, h("div", { class: "t" }, it.text), h("span", { class: `badge ${it.gold}` }, it.gold),
      h("div", { class: "votes" }, MODELS.map((x, i) => h("span", { class: `vote ${it.per[i].correct ? "ok" : "bad"}`, title: `${x.short}: ${it.per[i].predicted} (${(it.per[i].confidence * 100).toFixed(0)}%)` }, h("i", { style: `background:${x.color}` }), it.per[i].predicted.slice(0, 3), h("small", {}, `${(it.per[i].confidence * 100).toFixed(0)}`))))));
  }
  return h("div", { class: "box" }, h("div", { class: "box-head" }, h("h3", {}, `${rows.length} ${rows.length === 1 ? "tweet" : "tweets"}`), modes), chips.length ? h("div", { class: "chips", style: "margin-bottom:10px" }, chips) : null, list);
}
function drawErrors() {
  const root = $("#tab-errors");
  root.replaceChildren();
  if (!state.ready) return root.append(emptyBox());
  const items = agreementItems();
  root.append(h("h2", {}, "Where the models are right and wrong"),
    h("p", { class: "lede" }, "Rows are the gold label, columns the prediction; green diagonal cells are correct. Click a cell, or a bar segment, to list those tweets."),
    agreementBar(items), h("div", { class: "grid g3" }, MODELS.map((m) => matrixBox(m))), h("div", { style: "margin-top:20px" }, listBox(items)));
}
function renderErrors() { drawErrors(); }

// ───────────────────────── tabs
const renderers = { compare: renderCompare, train: renderTrain, tokens: renderTokens, attention: renderAttention, live: renderLive, errors: renderErrors };
const rendered = new Set();
function show(tab) {
  document.querySelectorAll(".tabs [role=tab]").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === tab));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== `tab-${tab}`));
  const root = $(`#tab-${tab}`);
  if (tab === "compare" || tab === "train" || !rendered.has(tab)) renderers[tab](root);   // charts re-animate on each visit
  rendered.add(tab);
  try { history.replaceState(null, "", `#${tab}`); } catch {}
  if (typeof moveInk === "function") moveInk();
}
document.querySelector(".tabs").addEventListener("click", (e) => { const b = e.target.closest("[role=tab]"); if (b) show(b.dataset.tab); });
document.querySelector(".tabs").addEventListener("keydown", (e) => {
  if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
  const tabs = [...document.querySelectorAll(".tabs [role=tab]")], i = tabs.findIndex((b) => b.getAttribute("aria-selected") === "true");
  const nx = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length]; nx.focus(); show(nx.dataset.tab);
});
await load();
const nav = document.querySelector(".tabs"), ink = nav.querySelector(".ink");
function moveInk() {
  const on = nav.querySelector('[aria-selected="true"]');
  if (!on) return;
  ink.style.width = `${on.offsetWidth}px`;
  ink.style.transform = `translateX(${on.offsetLeft}px)`;
}
new ResizeObserver(moveInk).observe(nav);
document.fonts?.ready.then(() => { moveInk(); requestAnimationFrame(() => ink.classList.add("ready")); });
const fromHash = () => (location.hash.slice(1) in renderers ? location.hash.slice(1) : "compare");
show(fromHash());
addEventListener("hashchange", () => show(fromHash()));

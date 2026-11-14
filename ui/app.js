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
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null) el.append(k.nodeType ? k : document.createTextNode(k));
  return el;
};
const f2 = (x) => (x == null ? "–" : x.toFixed(3));
const getJSON = (url) => fetch(url, { cache: "no-store" }).then((r) => (r.ok ? r.json() : Promise.reject(new Error(url))));
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = { summary: null, outs: {}, logs: {}, ready: false };

async function load() {
  const tryJSON = (u) => getJSON(u).catch(() => null);
  state.summary = await tryJSON("/outputs/summary.json");
  for (const m of MODELS) {
    state.outs[m.key] = await tryJSON(`/outputs/${m.key}.json`);
    state.logs[m.key] = await tryJSON(`/models/${m.key}/train_log.json`);
  }
  state.ready = MODELS.every((m) => state.outs[m.key]);
}

const emptyBox = () => h("div", { class: "box empty" }, "No results yet. Run ", h("code", {}, "bash run.sh"), " first, then reload.");
const modelDot = (m) => h("span", { class: "dot", style: `background:${m.color}` });
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
    h("div", { class: "winner" }, h("b", { class: "num" }, best.t.macro_f1.toFixed(3)), h("span", {}, modelLabel(best.m), ` has the best macro-F1, ${(best.t.macro_f1 - second.t.macro_f1).toFixed(3)} ahead of ${second.m.short}.`)),
  );
  const legend = h("div", { class: "legend" }, MODELS.map((m) => h("span", {}, modelLabel(m))));
  const overall = h("div", { class: "box" }, h("h3", {}, "Overall"), legend);
  for (const [key, title] of [["accuracy", "Accuracy"], ["macro_f1", "Macro-F1"], ["weighted_f1", "Weighted-F1"]]) overall.append(group(title, stats.map((s) => [s.m, s.t[key]])));
  const perClass = h("div", { class: "box" }, h("h3", {}, "F1 per class"), h("div", { class: "legend" }, MODELS.map((m) => h("span", {}, modelLabel(m)))));
  for (const c of CLASSES) perClass.append(group(`${c} (n=${stats[0].t.per_class[c].support})`, stats.map((s) => [s.m, s.t.per_class[c].f1])));
  root.append(rankBox(), h("div", { class: "grid g2" }, overall, perClass));

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
  requestAnimationFrame(() => requestAnimationFrame(() => root.querySelectorAll(".fill").forEach((f) => (f.style.width = f.dataset.w))));
}
// Ranking: pick a metric, bars re-order (FLIP-style) and numbers count up
const RANK_METRICS = [["accuracy", "Accuracy", (t) => t.accuracy], ["macro_f1", "Macro-F1", (t) => t.macro_f1], ["weighted_f1", "Weighted-F1", (t) => t.weighted_f1],
  ["negative", "F1 negative", (t) => t.per_class.negative.f1], ["neutral", "F1 neutral", (t) => t.per_class.neutral.f1], ["positive", "F1 positive", (t) => t.per_class.positive.f1]];
function rankBox() {
  const ROW = 46;
  let metric = RANK_METRICS[1];
  const rows = {};
  const stage = h("div", { class: "rank", style: `height:${ROW * MODELS.length}px` });
  for (const m of MODELS) {
    rows[m.key] = { shown: 0, el: h("div", { class: "rank-row", style: `height:${ROW - 8}px` }, h("span", { class: "rank-n" }), h("span", {}, modelLabel(m)),
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
  return h("div", { class: "box", style: "margin-bottom:20px" }, h("div", { class: "box-head" }, h("h3", {}, "Ranking"), seg), stage);
}
function group(title, rows) {
  const g = h("div", { class: "metric-group" }, h("h4", {}, title));
  for (const [m, v] of rows) g.append(h("div", { class: "metric-row" }, h("span", {}, m.short), h("div", { class: "track" }, h("div", { class: "fill", style: `background:${m.color}`, "data-w": `${(v * 100).toFixed(1)}%` })), h("span", { class: "v" }, f2(v))));
  return g;
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
  const lbl = s("text", { x: W - R, y: H - 8, "text-anchor": "end" }); lbl.textContent = "epoch"; gGrid.append(lbl);
  svg.append(gGrid);
  const hover = s("line", { class: "hover-line", y1: T, y2: H - B, opacity: 0 });
  svg.append(hover);
  for (const sr of series) {
    const d = sr.points.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
    const path = s("path", { d, class: `path${sr.dash ? " dash" : ""}`, stroke: sr.color });
    svg.append(path);
    if (!reduce && !sr.dash) { const len = Math.ceil(sr.points.length * 90); path.style.setProperty("--len", len); path.classList.add("draw"); }
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
    root.append(h("div", { class: "box", style: "margin-top:20px" }, h("h3", {}, "Language-model pretraining (masked tokens, from scratch)"),
      h("p", { class: "muted", style: "margin:0 0 10px" }, "Lower loss means the model predicts hidden tokens better. Token vocabularies differ, so compare the shapes, not the absolute values."),
      h("div", { class: "legend" }, mlm.map(({ m }) => h("span", {}, modelLabel(m)))),
      lineChart({ title: "MLM loss by epoch", yMin: Math.floor(Math.min(...ys) * 2) / 2, yMax: Math.ceil(Math.max(...ys) * 2) / 2, yTicks: 4, xMax: mx,
        series: mlm.map(({ m, lg }) => ({ name: m.short, color: m.color, points: lg.mlm_history.map((e) => ({ x: e.epoch, y: e.mlm_loss })) })) })));
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
    btn.disabled = true; err.textContent = "";
    try {
      const data = await api("/api/tokenize", text);
      out.replaceChildren(...MODELS.filter((m) => data[m.key]).map((m) => {
        const d = data[m.key];
        const row = h("div", { class: "toks", style: `--c:${m.color}` });
        d.tokens.forEach((t, i) => row.append(h("span", { class: `tk c${i % 2}${t.ws ? " ws" : ""}${t.special ? " sp" : ""}`, style: `animation-delay:${reduce ? 0 : Math.min(i * 6, 500)}ms` }, t.t)));
        return h("div", { class: "tokrow" }, h("div", { class: "head" }, modelLabel(m), h("span", { class: "num" }, `${d.count} tokens`)), row);
      }));
    } catch (e) { err.textContent = `Could not tokenize: ${e.message}. Is serve.py running with trained models?`; }
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
    btn.disabled = true; err.textContent = "";
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
  return h("div", { class: "box", style: `border-top:3px solid ${m.color}` }, h("h3", {}, modelLabel(m)), table,
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
  return h("div", { class: "box" }, h("div", { class: "box-head" }, h("h3", {}, `${rows.length} tweets`), modes), chips.length ? h("div", { class: "chips", style: "margin-bottom:10px" }, chips) : null, list);
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
const renderers = { compare: renderCompare, train: renderTrain, tokens: renderTokens, live: renderLive, errors: renderErrors };
const rendered = new Set();
function show(tab) {
  document.querySelectorAll(".tabs [role=tab]").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === tab));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== `tab-${tab}`));
  const root = $(`#tab-${tab}`);
  if (tab === "compare" || tab === "train" || !rendered.has(tab)) renderers[tab](root);   // charts re-animate on each visit
  rendered.add(tab);
  try { history.replaceState(null, "", `#${tab}`); } catch {}
}
document.querySelector(".tabs").addEventListener("click", (e) => { const b = e.target.closest("[role=tab]"); if (b) show(b.dataset.tab); });
document.querySelector(".tabs").addEventListener("keydown", (e) => {
  if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
  const tabs = [...document.querySelectorAll(".tabs [role=tab]")], i = tabs.findIndex((b) => b.getAttribute("aria-selected") === "true");
  const nx = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length]; nx.focus(); show(nx.dataset.tab);
});
await load();
show(location.hash.slice(1) in renderers ? location.hash.slice(1) : "compare");

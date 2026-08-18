"use strict";

const state = {
  features: [],
  queries: [],
  current: null,
  editors: { milvus: null, es: null },
  // Text currently occupying the hybrid snippet's lexical-query slot, so a
  // picker change knows what string to swap out of the editors.
  hybridText: null,
  // Which ES snippet the pane shows when the feature has an es_alt rewrite.
  esVariant: "rrf",
};

// The authored placeholder in the hybrid snippets (webapp/snippets.py). On the
// hybrid tab the lexical side is rewritten to describe the same product the
// picker feeds into `qv`, so both retrievers of the fusion agree on the target.
const HYBRID_DEFAULT_TEXT = "wireless earbuds";

// First words of the selected product's title, made safe to sit inside the
// snippets' double-quoted Python strings (drop `"` and `\`).
function queryText() {
  const q = state.queries[Number(els.query.value || 0)];
  if (!q || !q.title) return null;
  return q.title.replace(/["\\]/g, "").split(/\s+/).slice(0, 6).join(" ");
}

// Features with an `es_alt` snippet get a two-step demo: run the authored ES
// pane first (hybrid's RRF retriever 403s on the basic licence), then swap in
// the licence-free rewrite with the pane-header button and run again.
function setEsVariant(variant) {
  const f = state.current;
  if (!f || !f.es_alt) return;
  state.esVariant = variant;
  let src = variant === "alt" ? f.es_alt : f.es;
  if (state.hybridText && state.hybridText !== HYBRID_DEFAULT_TEXT) {
    src = src.split(HYBRID_DEFAULT_TEXT).join(state.hybridText);
  }
  state.editors.es.setValue(src);
  els.esVariant.textContent =
    variant === "alt" ? "Restore RRF version" : "Rewrite for basic licence";
}

function syncHybridText() {
  if (!state.current || state.current.key !== "hybrid" || !state.hybridText) return;
  const next = queryText();
  if (!next || next === state.hybridText) return;
  for (const ed of [state.editors.milvus, state.editors.es]) {
    const v = ed.getValue();
    // Hand-edited text no longer matches -- leave the user's edit alone.
    if (v.includes(state.hybridText)) ed.setValue(v.split(state.hybridText).join(next));
  }
  state.hybridText = next;
}

const els = {
  tabs: document.getElementById("tabs"),
  title: document.getElementById("feat-title"),
  subtitle: document.getElementById("feat-subtitle"),
  runs: document.getElementById("runs"),
  query: document.getElementById("query"),
  queryLabel: document.getElementById("query-label"),
  run: document.getElementById("run"),
  esVariant: document.getElementById("es-variant"),
  cached: document.getElementById("show-cached"),
  status: document.getElementById("status"),
  results: document.getElementById("results"),
  main: document.querySelector("main"),
  intro: document.getElementById("intro"),
};

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

function editor(id) {
  return CodeMirror.fromTextArea(document.getElementById(id), {
    mode: "python",
    theme: "material-darker",
    lineNumbers: true,
    viewportMargin: Infinity, // grow to content -- no inner scrollbar on stage
  });
}

// The Query picker only does anything when the snippet binds a ground-truth
// query vector -- the backend injects `qv` / `img_qv` and nothing else keys off
// query_index. Text-query features (bm25, synonyms, aggregate) ignore it, so
// hide the control for them. Mirrors QUERY_NAMES in webapp/metrics.py.
function usesQueryVector(f) {
  return /\b(qv|img_qv)\b/.test(`${f.milvus || ""}\n${f.es || ""}`);
}

function selectFeature(key) {
  const f = state.features.find((x) => x.key === key);
  if (!f) return;
  state.current = f;
  els.title.textContent = f.title;
  els.subtitle.textContent = f.subtitle;
  els.queryLabel.hidden = !usesQueryVector(f);
  state.editors.milvus.setValue(f.milvus);
  state.editors.es.setValue(f.es);
  state.hybridText = f.key === "hybrid" ? HYBRID_DEFAULT_TEXT : null;
  state.esVariant = "rrf";
  els.esVariant.hidden = !f.es_alt;
  els.esVariant.textContent = "Rewrite for basic licence";
  syncHybridText();
  els.results.innerHTML = "";
  els.status.textContent = "";
  els.main.classList.remove("mode-intro");
  // CodeMirror can't measure a display:none editor, so the setValue above didn't
  // lay out while the intro tab was showing. Now that the editors are visible
  // again, force a remeasure/redraw -- without this the FIRST feature opened from
  // the intro tab renders blank until the next navigation.
  state.editors.milvus.refresh();
  state.editors.es.refresh();
  setActiveTab(key);
}

/* The intro is a single-viewport poster: one symmetric diagram carries the
   whole environment story (identical vessels, live RAM fill, one dataset
   feeding both), and a glyph strip carries the live validation. Hardcoded hex
   mirrors the CSS tokens because the SVG is built as a string. */
const C = { blue: "#175FFF", slate: "#3A4357", ink: "#101428", mute: "#6B7285",
            line: "#E1E5EE", card: "#F4F6FB", white: "#fff" };
const MONO = "ui-monospace, Menlo, monospace";

function archSvg(env) {
  const m = env.milvus, e = env.es, d = env.dataset;
  const GB8 = 8 * 1024 ** 3;
  const mVer = esc(m.version || m.configured_tag || "?");
  const eVer = esc(e.version || e.configured_tag || "?");
  const lic = esc(e.license || "unknown");

  /* RAM gauge: y 192..400 spans 0..8 GB, fill = live usage. The two gauges sit
     either side of the "=" so the levels read as one comparison. */
  const gauge = (x, used, color) => {
    const frac = Math.max(0, Math.min(1, (used || 0) / GB8));
    const h = Math.round(208 * frac);
    const top = 400 - h;
    const fill = used ? `
      <rect x="${x + 2}" y="${top}" width="66" height="${Math.max(h - 2, 0)}" rx="6" fill="${color}" fill-opacity=".18"/>
      <line x1="${x + 2}" y1="${top}" x2="${x + 68}" y2="${top}" stroke="${color}" stroke-width="2"/>
      <text x="${x + 35}" y="${top - 9}" text-anchor="middle" font-size="12" font-weight="600" font-family="${MONO}" fill="${color}">${fmtBytes(used)}</text>` : `
      <text x="${x + 35}" y="300" text-anchor="middle" font-size="12" font-family="${MONO}" fill="${C.mute}">—</text>`;
    return `
      <rect x="${x}" y="192" width="70" height="208" rx="8" fill="${C.white}" stroke="${C.line}"/>
      <text x="${x + 35}" y="184" text-anchor="middle" font-size="10" fill="${C.mute}">RAM · 8 GB cap</text>${fill}`;
  };

  const spec = (x, y, label, value, color) => `
      <text x="${x}" y="${y}" font-size="9.5" letter-spacing="1" fill="${C.mute}">${label}</text>
      <text x="${x}" y="${y + 19}" font-size="12.5" font-family="${MONO}" fill="${color || C.ink}">${value}</text>`;

  const chip = (x, y, w, label) => `
      <rect x="${x}" y="${y}" width="${w}" height="24" rx="6" fill="${C.card}" stroke="${C.line}"/>
      <text x="${x + w / 2}" y="${y + 16}" text-anchor="middle" font-size="11" fill="${C.mute}">${label}</text>`;

  const heapY = 400 - 104;   // 4 GB mark on the ES gauge
  return `
<svg viewBox="0 0 1200 516" role="img" aria-label="Two identical containers, one dataset"
     font-family="system-ui, sans-serif" preserveAspectRatio="xMidYMid meet">
  <defs>
    <marker id="arw" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="${C.mute}"/>
    </marker>
  </defs>

  <g stroke="${C.mute}" stroke-width="1.4" marker-end="url(#arw)">
    <line x1="600" y1="42" x2="600" y2="56"/>
    <line x1="560" y1="106" x2="304" y2="146"/>
    <line x1="640" y1="106" x2="896" y2="146"/>
    <line x1="300" y1="452" x2="300" y2="428"/>
    <line x1="900" y1="452" x2="900" y2="428"/>
  </g>

  <rect x="510" y="8" width="180" height="34" rx="8" fill="${C.white}" stroke="${C.line}"/>
  <text x="600" y="30" text-anchor="middle" font-size="13" fill="${C.ink}">Browser (you)</text>

  <rect x="430" y="60" width="340" height="46" rx="8" fill="${C.ink}"/>
  <text x="600" y="79" text-anchor="middle" font-size="13" fill="${C.white}">demo app · FastAPI · :8080</text>
  <text x="600" y="96" text-anchor="middle" font-size="10.5" fill="#9aa3b8">runs each snippet on both engines, sequentially</text>

  <rect x="40" y="150" width="480" height="270" rx="12" fill="${C.white}" stroke="${C.blue}" stroke-width="2"/>
  <text x="62" y="179" font-size="17" font-weight="600" fill="${C.blue}">Milvus ${mVer}</text>
  <text x="62" y="197" font-size="11" fill="${C.mute}">standalone · gRPC :19530</text>
  ${gauge(430, m.mem_usage_bytes, C.blue)}
  ${spec(62, 238, "DENSE INDEX", "IVF_RABITQ + SQ8 refine", C.blue)}
  ${spec(62, 290, "HEAP", "none — native Go process", C.mute)}
  <text x="62" y="342" font-size="9.5" letter-spacing="1" fill="${C.mute}">SIDECARS</text>
  ${chip(62, 352, 56, "etcd")}
  ${chip(126, 352, 60, "minio")}

  <rect x="680" y="150" width="480" height="270" rx="12" fill="${C.white}" stroke="${C.slate}" stroke-width="2"/>
  <text x="792" y="179" font-size="17" font-weight="600" fill="${C.slate}">Elasticsearch ${eVer}</text>
  <text x="792" y="197" font-size="11" fill="${C.mute}">single-node · REST :9200</text>
  ${gauge(700, e.os_mem_used_bytes, C.slate)}
  <line x1="702" y1="${heapY}" x2="768" y2="${heapY}" stroke="${C.slate}" stroke-width="1.5" stroke-dasharray="4 3"/>
  ${spec(792, 238, "DENSE INDEX", "bbq_hnsw m=16 efc=100 · 3× oversample", C.slate)}
  ${spec(792, 290, "HEAP", "JVM 4 GB — ½ of container", C.mute)}
  <text x="792" y="342" font-size="9.5" letter-spacing="1" fill="${C.mute}">LICENCE</text>
  ${chip(792, 352, 110, lic)}

  <circle cx="600" cy="290" r="24" fill="${C.card}" stroke="${C.line}"/>
  <text x="600" y="299" text-anchor="middle" font-size="26" fill="${C.mute}">=</text>
  <text x="600" y="338" text-anchor="middle" font-size="11.5" font-weight="600" fill="${C.ink}">8 GB · 4 CPU each</text>
  <text x="600" y="353" text-anchor="middle" font-size="9.5" fill="${C.mute}">ceilings, not reservations</text>

  <rect x="40" y="452" width="1120" height="56" rx="10" fill="${C.card}" stroke="${C.line}"/>
  <text x="600" y="475" text-anchor="middle" font-size="13.5" font-weight="600" fill="${C.ink}">one dataset — ${esc(d.collection)} · ${fmtInt(m.docs ?? e.docs)} docs · ${d.dense_dim}-d dense · ${esc((d.metric || "").toUpperCase())}</text>
  <text x="600" y="494" text-anchor="middle" font-size="10.5" fill="${C.mute}">one bundled Parquet, same floats both sides · same BM25 analyzer chain: standard · lowercase · asciifolding · stemmer · stopwords</text>
</svg>`;
}

function renderIntro(env) {
  els.intro.innerHTML = "";
  els.intro.appendChild(el("p", "intro-lead",
    "One machine, two containers with identical limits, one Parquet loaded into both — everything below is read live from the running rig."));

  const arch = el("div", "arch");
  arch.innerHTML = archSvg(env);
  els.intro.appendChild(arch);

  const strip = el("div", "checks-strip");
  strip.appendChild(el("span", "strip-label", "live checks"));
  (env.checks || []).forEach((c) => {
    const item = el("span", `check ${c.state}`);
    item.appendChild(el("span", "glyph",
      c.state === "good" ? "✓" : c.state === "warn" ? "!" : "–"));
    item.appendChild(document.createTextNode(c.label));
    if (c.detail) item.appendChild(el("span", "detail", c.detail));
    strip.appendChild(item);
  });
  els.intro.appendChild(strip);
}

function setActiveTab(key) {
  [...els.tabs.children].forEach((b) => b.classList.toggle("active", b.dataset.key === key));
}

async function selectIntro() {
  state.current = null;
  els.main.classList.add("mode-intro");
  setActiveTab("__intro__");
  els.intro.innerHTML = '<p class="intro-lead">Reading environment…</p>';
  try {
    renderIntro(await getJSON("/api/environment"));
  } catch (e) {
    els.intro.innerHTML = `<div class="errbox">could not load environment: ${esc(e.message)}</div>`;
  }
}

async function pollHealth() {
  try {
    const h = await getJSON("/api/health");
    for (const [k, id] of [["milvus", "dot-milvus"], ["es", "dot-es"]]) {
      const el = document.getElementById(id);
      el.classList.toggle("ok", h[k].ok);
      el.classList.toggle("down", !h[k].ok);
      el.title = h[k].ok ? `${k}: reachable` : `${k}: ${h[k].error || "down"}`;
    }
  } catch (e) {
    /* server itself is gone; leave the dots as they were */
  }
}

async function run() {
  if (!state.current) return;
  els.run.disabled = true;
  const n = Number(els.runs.value);
  els.status.textContent = `running ${n}x on Milvus, then ${n}x on ES...`;
  try {
    const payload = await getJSON("/api/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        feature: state.current.key,
        milvus: state.editors.milvus.getValue(),
        es: state.editors.es.getValue(),
        runs: n,
        query_index: Number(els.query.value || 0),
      }),
    });
    els.status.textContent = "";
    renderResult(payload);
  } catch (e) {
    els.status.textContent = `request failed: ${e.message}`;
  } finally {
    els.run.disabled = false;
  }
}

async function showCached() {
  if (!state.current) return;
  try {
    renderResult(await getJSON(`/api/cached/${state.current.key}`));
    els.status.textContent = "";
  } catch (e) {
    els.status.textContent = "no cached run for this feature yet";
  }
}

const ENGINES = [["milvus", "Milvus"], ["es", "Elasticsearch"]];

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s === null || s === undefined ? "" : String(s);
  return d.innerHTML;
}

function pillClass(state) {
  return state === "good" ? "pill good" : state === "warn" ? "pill warn" : "pill na";
}

function fmtBytes(n) {
  if (typeof n !== "number" || !isFinite(n)) return "—";
  const gb = n / 1024 ** 3;
  return gb >= 1 ? gb.toFixed(1) + " GB" : (n / 1024 ** 2).toFixed(0) + " MB";
}

function fmtInt(n) {
  return typeof n === "number" && isFinite(n) ? n.toLocaleString("en-US") : "—";
}

/* One shared x-domain across both engines. Autoscaling each would make a 4ms
   spread and a 400ms spread look identical -- implying a parity that isn't
   there. */
function stripPlot(samples, domain, color) {
  const w = 100, h = 46, pad = 3;
  const [lo, hi] = domain;
  const span = hi - lo || 1;
  const x = (v) => pad + ((v - lo) / span) * (w - 2 * pad);
  const dots = samples
    .map((s) => `<circle cx="${x(s).toFixed(2)}" cy="23" r="1.7" fill="${color}" fill-opacity=".55"/>`)
    .join("");
  const med = samples.length ? [...samples].sort((a, b) => a - b)[Math.floor(samples.length / 2)] : lo;
  return `<svg class="plot" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <line x1="${pad}" y1="23" x2="${w - pad}" y2="23" stroke="#E1E5EE" stroke-width=".5"/>
    ${dots}
    <line x1="${x(med).toFixed(2)}" y1="10" x2="${x(med).toFixed(2)}" y2="36" stroke="${color}" stroke-width="1"/>
  </svg>`;
}

function latencySection(payload) {
  const all = [];
  for (const [k] of ENGINES) {
    const l = payload.engines[k].latency;
    if (l) all.push(...l.samples);
  }
  const domain = all.length ? [Math.min(...all), Math.max(...all)] : [0, 1];
  const wrap = el("div");
  wrap.appendChild(el("h3", null, `Latency — ${payload.runs} runs per engine, 1 warm-up discarded, run sequentially`));

  const grid = el("div", "lat");
  for (const [k, label] of ENGINES) {
    const e = payload.engines[k];
    const box = el("div", `box ${k}`);
    box.appendChild(el("div", "name", label));
    if (e.latency) {
      box.appendChild(el("div", "big", `${e.latency.median_ms} ms`));
      box.appendChild(el("div", "sub", `min ${e.latency.min_ms} · max ${e.latency.max_ms} · n=${e.latency.samples.length}`));
      box.insertAdjacentHTML("beforeend", stripPlot(e.latency.samples, domain, k === "milvus" ? "#175FFF" : "#3A4357"));
    } else {
      box.appendChild(el("div", "big", "—"));
      box.appendChild(el("div", "sub", "no timed runs completed"));
    }
    if (e.partial) box.appendChild(el("div", "warnbox", "partial: the run failed part-way; samples shown are the ones that completed"));
    grid.appendChild(box);
  }
  wrap.appendChild(grid);
  wrap.appendChild(el("p", "caveat",
    "Single-machine Docker figures — relative SDK feel, not a benchmark. Dominated by JVM warmup, container CPU limits, and cold cache. For performance claims cite the published 160M-vector study, not this box."));
  return wrap;
}

function correctnessSection(payload) {
  const c = payload.correctness;
  const wrap = el("div");
  wrap.appendChild(el("h3", null, "Correctness"));
  const line = el("div");

  if (c.overlap === null) {
    line.appendChild(el("span", "pill na", "agreement n/a"));
  } else {
    line.appendChild(el("span", "pill good", `top-10 agreement: ${c.overlap}/10`));
  }
  if (c.recall_milvus !== null) line.appendChild(el("span", "pill", `Milvus recall@10: ${c.recall_milvus}`));
  if (c.recall_es !== null) line.appendChild(el("span", "pill", `ES recall@10: ${c.recall_es}`));
  wrap.appendChild(line);
  if (c.recall_note) wrap.appendChild(el("p", "note", c.recall_note));
  return wrap;
}

function fmtNum(v, digits) {
  // float32 values arrive as 111.98999786… / 3.5999999… — pin to a stable
  // decimal count so the cards don't show storage noise.
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(digits) : null;
}

// Format a table cell by what the column holds, so prices/ratings/counts read
// the same everywhere they appear (cards and rows tables). Column keys vary by
// engine — Milvus "avg(price)"/"count(*)", ES "avg_price"/"count" — so match on
// substring. Non-numbers (category, id) pass through untouched.
function fmtCell(col, v) {
  if (typeof v !== "number" || !Number.isFinite(v)) return v ?? "";
  const c = col.toLowerCase();
  if (c.includes("price")) return "$" + fmtNum(v, 2);
  if (c.includes("rating")) return "★" + fmtNum(v, 1);
  if (c.includes("count")) return fmtNum(v, 0);
  return fmtNum(v, 2);
}

function hitCard(h, display, agree) {
  const d = display[h.id] || {};
  const card = el("div", "hit" + (agree.has(h.id) ? " agree" : ""));
  card.appendChild(el("div", "rank", `#${h.rank}`));
  if (d.image_url) {
    const img = el("img");
    img.src = d.image_url;
    img.loading = "lazy";
    img.onerror = () => img.remove();
    card.appendChild(img);
  }
  const t = el("div", "t");
  const price = fmtNum(d.price, 2);
  const rating = fmtNum(d.average_rating, 1);
  const meta = [d.store, price != null ? "$" + price : null,
                rating != null ? "★" + rating : null]
    .filter(Boolean).map(esc).join(" · ");
  // main_category is the collapse / group_by key -- surface it on every card so
  // it's visually obvious the grouping actually returned one row per category.
  const cat = d.main_category ? `<span class="cat">${esc(d.main_category)}</span>` : "";
  t.innerHTML = `<b>${esc(d.title || h.id || "(no id)")}</b>` +
                `<span>${cat}${cat && meta ? " · " : ""}${meta}</span>`;
  // Fragments arrive carrying engine-generated <em> markup. Escape the whole
  // string, then re-admit exactly the tag pair both engines are configured to
  // emit -- an allowlist, so a product title containing "<script>" stays inert.
  const frags = Object.values(h.highlight || {}).flat();
  if (frags.length) {
    const hl = el("div", "hl");
    hl.innerHTML = frags
      .map((f) => esc(f).replace(/&lt;em&gt;/g, "<em>")
                        .replace(/&lt;\/em&gt;/g, "</em>"))
      .join(" … ");
    t.appendChild(hl);
  }
  card.appendChild(t);
  card.appendChild(el("div", "sc", h.score != null ? h.score.toFixed(3) : ""));
  return card;
}

function rowsTable(rows) {
  const wrap = el("div", "tablewrap");
  if (!rows.length) { wrap.appendChild(el("p", "note", "no rows")); return wrap; }
  // Keep the first column (the id / group key) as the anchor; sort the rest
  // alphabetically so metric columns land in a predictable order.
  const raw = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  const cols = raw.length <= 1 ? raw
    : [raw[0], ...raw.slice(1).sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()))];
  const t = el("table", "rows");
  const thead = el("thead"), tr = el("tr");
  cols.forEach((c) => tr.appendChild(el("th", null, c)));
  thead.appendChild(tr); t.appendChild(thead);
  const tb = el("tbody");
  for (const r of rows) {
    const row = el("tr");
    cols.forEach((c) => {
      row.appendChild(el("td", null, fmtCell(c, r[c])));
    });
    tb.appendChild(row);
  }
  t.appendChild(tb); wrap.appendChild(t);
  return wrap;
}

function resultsSection(payload) {
  const agree = new Set(payload.correctness.agreement_ids || []);
  const wrap = el("div");
  wrap.appendChild(el("h3", null, payload.metric === "rows" ? "Rows returned" : "Results"));
  const grid = el("div", "hits");
  for (const [k, label] of ENGINES) {
    const e = payload.engines[k];
    const pane = el("div");
    pane.appendChild(el("div", "name", label));

    if (e.error && !e.ok) {
      pane.appendChild(el("div", "errbox", e.error));
    }
    for (const w of e.warnings || []) pane.appendChild(el("div", "warnbox", w));

    if (e.ok) {
      if (payload.metric === "rows") {
        pane.appendChild(rowsTable(e.rows.length ? e.rows : e.hits.map((h) => ({ id: h.id, ...h.fields }))));
      } else {
        const list = el("div", "hitlist");
        e.hits.forEach((h) => list.appendChild(hitCard(h, payload.display, agree)));
        pane.appendChild(list);
      }
    }

    const det = el("details", "raw");
    det.appendChild(el("summary", null, `raw ${label} response`));
    det.appendChild(el("pre", null, JSON.stringify(e.raw, null, 2)));
    pane.appendChild(det);
    grid.appendChild(pane);
  }
  wrap.appendChild(grid);
  return wrap;
}

function analyzeSection(payload) {
  const a = payload.analyze;
  if (!a) return null;
  const wrap = el("div", "analyze");
  wrap.appendChild(el("div", "analyze-head", "Analyzer output (what each engine indexes/searches)"));
  const grid = el("div", "analyze-grid");
  for (const [engine, label, tokens, text] of [
    ["Milvus", a.milvus_label, a.milvus, a.milvus_text],
    ["Elasticsearch", a.es_label, a.es, a.es_text],
  ]) {
    const pane = el("div", `analyze-pane ${engine === "Milvus" ? "milvus" : "es"}`);
    pane.appendChild(el("div", "analyze-label", `${engine} · ${label || ""}`));
    if (text != null) pane.appendChild(el("div", "analyze-query", `"${text}"`));
    const chips = el("div", "chips");
    if (tokens && tokens.length) {
      for (const tok of tokens) chips.appendChild(el("span", "chip", tok));
    } else {
      chips.appendChild(el("span", "chip empty", "—"));
    }
    pane.appendChild(chips);
    grid.appendChild(pane);
  }
  wrap.appendChild(grid);
  if (a.warnings && a.warnings.length) {
    wrap.appendChild(el("div", "analyze-warn", a.warnings.join(" · ")));
  }
  return wrap;
}

function renderResult(payload) {
  els.results.innerHTML = "";
  const wrap = el("div", "res");
  if (payload.cached) {
    wrap.appendChild(el("div", "stamp", `CACHED — NOT LIVE · ${payload.ts} · ${payload.runs} runs`));
  }
  wrap.appendChild(latencySection(payload));
  if (payload.metric !== "rows") wrap.appendChild(correctnessSection(payload));
  const a = analyzeSection(payload);
  if (a) wrap.appendChild(a);
  wrap.appendChild(resultsSection(payload));
  els.results.appendChild(wrap);
}

async function boot() {
  state.editors.milvus = editor("code-milvus");
  state.editors.es = editor("code-es");

  const introBtn = document.createElement("button");
  introBtn.textContent = "Introduction";
  introBtn.dataset.key = "__intro__";
  introBtn.onclick = () => selectIntro();
  els.tabs.appendChild(introBtn);

  state.features = await getJSON("/api/features");
  for (const f of state.features) {
    const b = document.createElement("button");
    b.textContent = f.title;
    b.dataset.key = f.key;
    b.onclick = () => selectFeature(f.key);
    els.tabs.appendChild(b);
  }

  try {
    state.queries = await getJSON("/api/queries");
    for (const q of state.queries) {
      const o = document.createElement("option");
      o.value = q.index;
      o.textContent = q.label;
      els.query.appendChild(o);
    }
  } catch (e) {
    els.query.disabled = true;
  }
  els.query.onchange = syncHybridText;
  els.esVariant.onclick = () =>
    setEsVariant(state.esVariant === "alt" ? "rrf" : "alt");

  els.run.onclick = run;
  els.cached.onclick = showCached;
  selectIntro();

  pollHealth();
  setInterval(pollHealth, 3000);
}

window.__demo = { state, els, renderResult };
boot();

"use strict";
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const col = v => v >= 75 ? "var(--good)" : v >= 50 ? "var(--mid)" : "var(--bad)";
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } }
};
let file = null, last = null, prevScore = null, hist = store.get("rl_hist", []);

/* theme */
function applyTheme(t) { if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; }
applyTheme(store.get("rl_theme", null));
$("#theme").onclick = () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme:dark)").matches;
  const t = dark ? "light" : "dark"; applyTheme(t); store.set("rl_theme", t);
};

/* inputs */
const words = s => (s.trim().match(/\S+/g) || []).length;
const counts = () => { $("#rc").textContent = words($("#resume").value) + " words"; $("#jc").textContent = words($("#job").value) + " words"; };
$("#resume").oninput = () => { counts(); if ($("#resume").value) clearFile(); };
$("#job").oninput = counts;
function mode(paste) {
  $("#m-up").setAttribute("aria-selected", !paste); $("#m-paste").setAttribute("aria-selected", paste);
  $("#drop").hidden = paste; $("#pasteBox").hidden = !paste;
}
$("#m-up").onclick = () => mode(false); $("#m-paste").onclick = () => mode(true);
function setFile(f) {
  file = f; $("#drop").classList.toggle("has", !!f);
  $("#dropTitle").textContent = f ? f.name : "Drop a PDF, DOCX or TXT here";
  $("#dropSub").textContent = f ? Math.max(1, Math.round(f.size / 1024)) + " KB · ready to analyze" : "or click to browse · up to 5 MB";
}
function clearFile() { file = null; $("#file").value = ""; setFile(null); }
$("#file").onchange = e => { if (e.target.files[0]) { $("#resume").value = ""; counts(); setFile(e.target.files[0]); } };
const drop = $("#drop");
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) { $("#resume").value = ""; counts(); setFile(f); } });
$("#demo").onclick = async () => {
  try {
    const r = await (await fetch("/api/sample")).json();
    clearFile(); mode(true); $("#resume").value = r.resume; $("#job").value = r.job; counts(); run();
  } catch (e) { $("#err").textContent = "Couldn’t reach the server. Is it running?"; }
};
$("#clear").onclick = () => { clearFile(); $("#resume").value = ""; $("#job").value = ""; counts(); $("#err").textContent = ""; };
$("#go").onclick = run;

async function run() {
  $("#err").textContent = "";
  const b = $("#go"); b.disabled = true; b.textContent = "Analyzing…";
  $("#out").innerHTML = '<div class="skel"></div><div class="skel" style="height:220px"></div><div class="skel"></div>';
  const fd = new FormData();
  fd.append("resume_text", $("#resume").value); fd.append("job_text", $("#job").value);
  if (file) fd.append("resume_file", file);
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Something went wrong.");
    render(data);
  } catch (e) {
    $("#err").textContent = e.message === "Failed to fetch" || e.message === "Load failed" ? "Couldn’t reach the server. Is it running?" : e.message;
    $("#out").innerHTML = '<div class="card empty"><h2>No results yet</h2><p>Fix the issue above and try again.</p></div>';
  }
  b.disabled = false; b.textContent = "Analyze";
}

/* charts */
function radar(sc) {
  const ax = [["Fit", sc.fit], ["Quality", sc.quality], ["Structure", sc.structure], ["Keywords", sc.keywords], ["Readability", sc.readability]];
  const cx = 130, cy = 124, R = 78, pt = (i, f) => { const a = -Math.PI / 2 + i * 2 * Math.PI / 5; return [cx + Math.cos(a) * R * f, cy + Math.sin(a) * R * f]; };
  const poly = f => ax.map((_, i) => pt(i, f).map(n => n.toFixed(1)).join(",")).join(" ");
  const vals = ax.map(([, v], i) => pt(i, (v || 0) / 100).map(n => n.toFixed(1)).join(",")).join(" ");
  return `<svg class="radar" viewBox="0 0 260 250" role="img" aria-label="Radar chart of ${ax.map(([n, v]) => n + " " + (v == null ? "n/a" : v)).join(", ")}">
   ${[.33, .66, 1].map(f => `<polygon points="${poly(f)}" fill="none" stroke="var(--line)"/>`).join("")}
   ${ax.map((_, i) => `<line x1="${cx}" y1="${cy}" x2="${pt(i, 1)[0]}" y2="${pt(i, 1)[1]}" stroke="var(--line)"/>`).join("")}
   <polygon points="${vals}" fill="var(--accent)" fill-opacity=".22" stroke="var(--accent)" stroke-width="2"/>
   ${ax.map(([n, v], i) => { const [x, y] = pt(i, 1.2); return `<text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="middle">${n}${v == null ? " (n/a)" : ""}</text>`; }).join("")}</svg>`;
}
const bar = (label, v, note) => `<div><div class="bar-h"><span>${label}</span><span>${v == null ? "n/a" : v + "/100"}${note ? " · " + esc(note) : ""}</span></div><div class="track"><div class="fill" style="width:${v || 0}%;background:${col(v || 0)}"></div></div></div>`;

/* render */
function render(d) {
  const delta = prevScore == null ? null : d.score - prevScore;
  last = d; saveHistory(d); prevScore = d.score;
  const C = 2 * Math.PI * 57, verdict = d.score >= 80 ? "Interview ready" : d.score >= 60 ? "Solid, with gaps" : d.score >= 40 ? "Needs work" : "Start with the basics";
  const lvl = { strong: "var(--good)", partial: "var(--mid)", gap: "var(--bad)" }, lbl = { strong: "Covered", partial: "Partly covered", gap: "No evidence" };
  const S = d.skills, extra = S.resume.filter(x => !S.job.includes(x));
  const rd = d.readability;
  const overview = `<div class="hero"><div class="ring" role="img" aria-label="Score ${d.score} of 100"><svg width="132" height="132"><circle class="bg" cx="66" cy="66" r="57"/><circle class="fg" id="fg" cx="66" cy="66" r="57" stroke="${col(d.score)}" stroke-dasharray="${C}" stroke-dashoffset="${C}"/></svg><b>${d.score}</b></div>
   <div><div class="verdict">${verdict}</div><div class="sub">${d.words} words · ${d.bullets.length} bullets · similarity: ${esc(d.engine)}</div>
   ${delta == null ? "" : `<span class="delta ${delta >= 0 ? "up" : "down"}">${delta >= 0 ? "▲ +" : "▼ "}${delta} vs previous run</span>`}</div></div>
   <div class="two"><div>${radar(d.scores)}</div><div>
    ${bar("Job fit", d.scores.fit, d.scores.fit == null ? "add a job description" : "")}${bar("Bullet quality", d.scores.quality)}${bar("Structure", d.scores.structure, d.sections.length + " of 5 headings")}${bar("Keywords", d.scores.keywords)}${bar("Readability", d.scores.readability)}</div></div>
   <h2>Predicted target role</h2>${d.roles.map(r => bar(r.role, Math.round(r.prob * 100))).join("")}
   <p class="sub" style="margin-top:8px">Seniority signal: <b>${esc(d.seniority.label)}</b> (${Object.entries(d.seniority.probs).map(([k, v]) => k + " " + Math.round(v * 100) + "%").join(", ")})</p>
   ${d.low_confidence ? '<div class="warn" role="note">Low model confidence: your resume doesn’t clearly signal one role, so treat the role prediction loosely. The models are trained on small, mostly synthetic data.</div>' : ""}
   ${d.sections_inferred.length ? `<p class="sub" style="margin-top:8px">Content found without a heading: ${d.sections_inferred.map(esc).join(", ")}.</p>` : ""}
   <h2>What to fix first</h2><ul class="tips">${d.tips.map(t => `<li class="${t.level}">${esc(t.text)}</li>`).join("") || '<li class="good">Nothing urgent.</li>'}</ul>
   <h2>Tone and readability</h2><p class="sub">Average bullet length ${rd.avg_bullet_words} words · ${rd.passive_voice} passive phrases · ${rd.first_person} first-person pronouns · ${rd.long_bullets} very long bullets${rd.buzzwords.length ? " · buzzwords: " + rd.buzzwords.map(esc).join(", ") : ""}.</p>`;
  const skills = `${S.job.length ? `<h2 style="margin-top:0">Skills the job asks for</h2><div class="chips">${S.matched.map(k => `<span class="chip ok">${esc(k)}</span>`).join("")}${S.missing.map(k => `<span class="chip miss">${esc(k)}</span>`).join("")}</div>` : '<p class="sub">Add a job description to compare skills.</p>'}
   ${d.keywords.length ? `<h2>Weighted keywords from the job description</h2><div class="chips">${d.keywords.map(k => `<span class="chip ${k.present ? "ok" : "miss"}" title="importance ${k.weight}">${esc(k.term)} · ${Math.round(k.weight * 100)}</span>`).join("")}</div>` : ""}
   ${extra.length ? `<h2>Other skills found</h2><div class="chips">${extra.map(k => `<span class="chip extra">${esc(k)}</span>`).join("")}</div>` : ""}`;
  const reqs = d.requirements.length ? d.requirements.map(r => `<div class="item"><b>${esc(r.requirement)}</b><small style="color:${lvl[r.level]}">${lbl[r.level]}${r.evidence ? ": " + esc(r.evidence) : ". Add a bullet that shows this."}</small><div class="m"><i style="width:${Math.round(r.score * 100)}%;background:${lvl[r.level]}"></i></div></div>`).join("") : '<p class="sub">Add a job description to see requirement matching.</p>';
  const bullets = `<div class="tools"><select id="bf" aria-label="Filter bullets"><option value="all">All bullets</option><option value="weak">Weak only</option><option value="strong">Strong only</option></select><select id="bs" aria-label="Sort bullets"><option value="weak">Weak first</option><option value="orig">Original order</option></select></div><div id="bl"></div>`;
  const rewrites = d.bullets.filter(b => b.rewrite).map((b, i) => `<div class="item"><span class="sub">Before</span><div>${esc(b.text)}</div><div class="sug" id="rw${i}">${esc(b.rewrite)}</div><div class="row" style="margin-top:8px"><button class="btn sm" data-copy="${i}">Copy</button></div></div>`).join("") || '<p class="sub">Every bullet is already strong. Nice.</p>';
  const tabs = [["Overview", overview], ["Skills", skills], ["Requirements", reqs], ["Bullets", bullets], ["Rewrite", `<p class="sub" style="margin-bottom:10px">Starting points only. Replace the [placeholders] with your real numbers.</p>${rewrites}`], ["Resume", viewer(d)]];
  $("#out").innerHTML = `<div class="card"><div class="tabs" role="tablist">${tabs.map((t, i) => `<button role="tab" data-t="${i}" aria-selected="${i === 0}">${t[0]}</button>`).join("")}</div>
   ${tabs.map((t, i) => `<div class="panel" data-p="${i}" ${i ? "hidden" : ""}>${t[1]}</div>`).join("")}
   <div class="row"><button class="btn sm" id="exp">Export JSON</button><button class="btn sm" id="prt">Print / save PDF</button></div></div>`;
  drawBullets();
  requestAnimationFrame(() => requestAnimationFrame(() => { const f = $("#fg"); if (f) f.style.strokeDashoffset = C * (1 - d.score / 100); }));
  if (matchMedia("(max-width:899px)").matches) $("#out").scrollIntoView({ behavior: "smooth", block: "start" });
  renderHistory();
}

function viewer(d) {
  const pats = [...d.highlights.weak.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")), "\\$?\\d[\\d,.]*\\s?(?:%|[kKmMxX]\\b)?\\+?"];
  const terms = [...new Set(d.highlights.terms)].filter(Boolean).map(t => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${pats[0] || "(?!)"})|(${pats[1]})` + (terms.length ? `|\\b(${terms.join("|")})\\b` : ""), "gi");
  const html = esc(d.resume_text).replace(re, (m, w, n) => w ? `<span class="weak">${m}</span>` : n ? `<mark class="num">${m}</mark>` : `<mark class="hit">${m}</mark>`);
  return `<div class="legend"><span><mark class="hit">Matches job</mark></span><span><mark class="num">Metric</mark></span><span><span class="weak">Weak phrase</span></span></div><div class="paper">${html}</div>`;
}

function drawBullets() {
  if (!last || !$("#bl")) return;
  const f = $("#bf").value, s = $("#bs").value;
  let list = last.bullets.map((b, i) => ({ ...b, i })).filter(b => f === "all" || (f === "weak" ? b.quality < .5 : b.quality >= .5));
  if (s === "weak") list.sort((a, b) => a.quality - b.quality);
  $("#bl").innerHTML = list.map(b => `<details class="item bul ${b.quality < .5 ? "weak" : b.quality < .75 ? "mid" : ""}"><summary>${esc(b.text)}<small>${Math.round(b.quality * 100)}% strong${b.confidence < .3 ? " · model unsure" : ""}</small></summary>
    <div class="sub" style="margin-top:8px">${b.reasons.length ? b.reasons.map(esc).join(" · ") : "No issues found."}</div>
    ${b.why.strong.length || b.why.weak.length ? `<div class="sub">Model signals: ${b.why.strong.map(t => `<span class="chip ok">${esc(t)}</span>`).join(" ")} ${b.why.weak.map(t => `<span class="chip miss">${esc(t)}</span>`).join(" ")}</div>` : ""}
    <div class="ed" id="ed${b.i}" hidden><textarea aria-label="Edit bullet">${esc(b.text)}</textarea><button class="btn sm pri" data-apply="${b.i}">Apply and re-analyze</button></div>
    <div class="row" style="margin-top:8px"><button class="btn sm" data-edit="${b.i}">Edit</button></div></details>`).join("") || '<p class="sub">No bullets match this filter.</p>';
}

/* events inside results */
$("#out").addEventListener("click", e => {
  const t = e.target.closest("button"); if (!t) return;
  if (t.dataset.t != null) {
    document.querySelectorAll("#out .tabs button").forEach(b => b.setAttribute("aria-selected", b === t));
    document.querySelectorAll("#out .panel").forEach(p => p.hidden = p.dataset.p !== t.dataset.t);
  } else if (t.dataset.edit != null) { const el = $("#ed" + t.dataset.edit); el.hidden = !el.hidden; }
  else if (t.dataset.apply != null) {
    const i = +t.dataset.apply, nt = $("#ed" + i + " textarea").value.trim(), old = last.bullets[i].text;
    if (!nt) return;
    clearFile(); mode(true); $("#resume").value = last.resume_text.replace(old, nt); counts(); run();
  } else if (t.dataset.copy != null) { const txt = $("#rw" + t.dataset.copy).textContent; navigator.clipboard && navigator.clipboard.writeText(txt); t.textContent = "Copied"; }
  else if (t.id === "exp") {
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(last, null, 2)], { type: "application/json" }));
    a.download = "resumelens-report.json"; a.click(); URL.revokeObjectURL(a.href);
  } else if (t.id === "prt") window.print();
});
$("#out").addEventListener("change", e => { if (e.target.id === "bf" || e.target.id === "bs") drawBullets(); });

/* history + compare */
function saveHistory(d) {
  const title = (d.resume_text.split("\n")[0] || "Resume").slice(0, 40);
  hist.unshift({ id: Date.now(), t: new Date().toLocaleString(), title, score: d.score, scores: d.scores, role: d.roles[0].role });
  hist = hist.slice(0, 12); store.set("rl_hist", hist);
}
function renderHistory() {
  const h = $("#hist");
  if (!hist.length) { h.innerHTML = "<p class='sub'>No runs yet.</p>"; return; }
  h.innerHTML = `<div class="lbl" style="margin-top:0"><b>History</b><span><button class="btn sm" id="cmp">Compare selected</button> <button class="btn sm ghost" id="hclr">Clear</button></span></div>
   ${hist.map(r => `<label class="item" style="display:flex;gap:10px;align-items:center"><input type="checkbox" data-h="${r.id}"><span style="flex:1"><b>${esc(r.title)}</b><small>${esc(r.t)} · ${esc(r.role)}</small></span><b>${r.score}</b></label>`).join("")}<div id="cmpOut"></div>`;
}
$("#hist").addEventListener("click", e => {
  if (e.target.id === "hclr") { hist = []; store.set("rl_hist", hist); renderHistory(); }
  if (e.target.id === "cmp") {
    const ids = [...document.querySelectorAll("#hist [data-h]:checked")].map(c => +c.dataset.h);
    if (ids.length !== 2) { $("#cmpOut").innerHTML = '<p class="sub">Tick exactly two runs to compare.</p>'; return; }
    const [a, b] = ids.map(id => hist.find(r => r.id === id)).sort((x, y) => x.id - y.id);
    const rows = [["Overall", a.score, b.score], ...["fit", "quality", "structure", "keywords", "readability"].map(k => [k[0].toUpperCase() + k.slice(1), a.scores[k], b.scores[k]])];
    $("#cmpOut").innerHTML = `<div class="tbl"><table><tr><th></th><th>Earlier</th><th>Later</th><th>Change</th></tr>${rows.map(([n, x, y]) => `<tr><td>${n}</td><td>${x == null ? "n/a" : x}</td><td>${y == null ? "n/a" : y}</td><td>${x == null || y == null ? "" : (y - x >= 0 ? "+" : "") + (y - x)}</td></tr>`).join("")}</table></div>`;
  }
});
$("#histBtn").onclick = () => { const h = $("#hist"); h.hidden = !h.hidden; $("#histBtn").setAttribute("aria-expanded", !h.hidden); if (!h.hidden) renderHistory(); };
mode(false); counts();

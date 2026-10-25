// RealGrounds Lab - plain JavaScript, no build step. Every number comes from app.py's JSON API (the repository's
// tests and run logs); every action either runs the real backend or replays a recorded run.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pl = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
const usd = (v) => (v ? "$" + (v < 0.01 ? v.toFixed(4) : v.toFixed(3)) : "$0");
const data = { results: null, experiments: null, overview: null };
// a published snapshot (export_site.py sets RG_STATIC): read the exported JSON, never call the backend
const STATIC = !!window.RG_STATIC;
if (STATIC) document.body.classList.add("static");
const SNAPSHOT_MSG = "This is a published snapshot. To run the agent, start the app locally: python app.py";
async function snapshot(path, body) {
  if (body !== undefined) throw new Error(SNAPSHOT_MSG);
  const u = new URL(path, "http://x");
  const f = { "/api/overview": "data/overview.json", "/api/results": "data/results.json", "/api/experiments": "data/experiments.json" }[u.pathname]
    || (u.pathname === "/api/room" ? "data/rooms/" + (u.searchParams.get("id") || "").replace(/[^A-Za-z0-9_-]/g, "_") + ".json" : null);
  if (!f) throw new Error(SNAPSHOT_MSG);
  const r = await fetch(f);
  if (!r.ok) throw new Error("This room is not in the published snapshot.");
  return r.json();
}

// ================================================================ plumbing
async function api(path, body) {
  if (STATIC) return snapshot(path, body);
  let r;
  try {
    r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch (e) { throw new Error("The app server is not reachable. Is `python app.py` still running?"); }
  let j = {};
  try { j = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) throw new Error(j.error || `request failed (${r.status})`);
  return j;
}

let toastTimer;
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = "on" + (err ? " err" : "");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.className = ""), 4200);
}

// follow a backend job; onLine gets each new log line; resolves with the finished job
function follow(id, onLine) {
  let since = 0;
  return new Promise((resolve) => {
    const tick = async () => {
      let j;
      try { j = await api(`/api/job?id=${id}&since=${since}`); } catch (e) { return setTimeout(tick, 1500); }
      since += j.lines.length;
      j.lines.forEach((l) => onLine && onLine(l));
      if (j.status === "done" || j.status === "failed") { refreshBudget(); return resolve(j); }
      setTimeout(tick, 800);
    };
    tick();
  });
}

// an honest progress panel: what is happening, elapsed time, what to expect, the real log
function progress(el, title, note) {
  el.classList.remove("hidden");
  el.innerHTML = `<div class="ph"><span>${esc(title)}</span><span class="t">0 s</span></div><div class="bar-ind"></div>
    <div class="note">${esc(note)}</div><div class="log"></div>`;
  const t0 = Date.now(), t = $(".t", el), log = $(".log", el);
  const iv = setInterval(() => (t.textContent = `${Math.round((Date.now() - t0) / 1000)} s`), 500);
  return {
    line(l) { if (l.trim()) { log.textContent += l + "\n"; log.scrollTop = log.scrollHeight; } },
    done(hide = true) { clearInterval(iv); if (hide) el.classList.add("hidden"); else $(".bar-ind", el)?.remove(); },
  };
}

async function refreshBudget() {
  try {
    data.overview = await api("/api/overview");
    $("#budget").textContent = STATIC ? `Snapshot · ${data.overview.snapshot_time || ""}` : `Claude budget ${usd(data.overview.spent_usd)} / $${data.overview.cap_usd.toFixed(2)}`;
  } catch (e) { $("#budget").textContent = "server offline"; }
  return data.overview;
}

const LABEL = {
  solvable: ["good", "✓ Solved in simulation"], impossible: ["info", "⊘ Impossible (geometry)"],
  unresolved: ["warn", "! Stopped, asked for help"], already_true: ["slate", "= Already true"],
  rejected: ["bad", "✗ Not a task for this room"], skipped: ["slate", "– Skipped (room budget)"],
};
const OUTCOME = {
  simulated_success: "goal reached in the simulated room", infeasible: "impossible: no route even with every movable thing removed",
  stuck: "stuck: nothing it tried measurably helped", budget: "out of budget", loop: "stopped: would return to an earlier state",
  planner_error: "the planner failed",
};
const tag = (label) => { const [c, w] = LABEL[label] || ["slate", label]; return `<span class="tag ${c}">${esc(w)}</span>`; };
const src = (s) => s && s.startsWith("hand") ? '<span class="src hand">hand-made</span>' : '<span class="src photo">photo-built</span>';
const plannerName = (p) => p === "fixed rules" ? "fixed rules (no AI)" : (p || "").replace("claude-", "Claude ").replace(/-(\d)-(\d)$/, " $1.$2").replace("haiku", "Haiku").replace("sonnet", "Sonnet").replace("opus", "Opus");
const clean = (s) => String(s || "").replace(/^[✗✓↩\s]+/, "").replace(/^refused:\s*/, "");

// ================================================================ router
const VIEWS = ["overview", "experiments", "lab", "safety", "results", "run"];
const ALIAS = { try: "lab", stops: "experiments", runs: "results" };
const LOADERS = { overview: loadOverview, experiments: loadExperiments, lab: loadLab, safety: loadSafety, results: loadResults, run: () => {} };
function route() {
  let [v, q] = location.hash.slice(1).split("?");
  v = ALIAS[v] || v;
  if (!VIEWS.includes(v)) v = "overview";
  $$(".view").forEach((s) => s.classList.toggle("on", s.id === "view-" + v));
  $$("#nav a").forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + v));
  LOADERS[v](new URLSearchParams(q || ""));
}
window.addEventListener("hashchange", () => { route(); window.scrollTo({ top: 0 }); });

async function getResults(force) { if (force || !data.results) data.results = await api("/api/results"); return data.results; }
async function getExperiments(force) { if (force || !data.experiments) data.experiments = await api("/api/experiments"); return data.experiments; }

// ================================================================ overview
function latestComplete(runs) { return runs.find((r) => r.finished && r.rooms >= 5); }
function longest(runs) { return [...runs].sort((a, b) => (b.rooms_planned || 0) - (a.rooms_planned || 0))[0]; }
function verified(runs) { return runs.filter((r) => r.verify).map((r) => ({ run: r.run, ...r.verify })); }

async function loadOverview() {
  let r;
  try { r = await getResults(true); } catch (e) { $("#metrics").innerHTML = `<div class="skeleton">${esc(e.message)}</div>`; return; }
  const m = [];
  const t = r.tests;
  m.push(t ? `<div class="metric"><div class="v">${t.passed}<small> / ${t.total}</small></div><div class="k">automated tests pass</div>
      <div class="s">run ${esc(t.time.replace("T", " "))} · <a href="#results">see them</a></div></div>`
    : `<div class="metric"><div class="v">–</div><div class="k">automated tests</div><div class="s">not run yet · <a href="#results">run them</a></div></div>`);
  const v = verified(r.runs)[0];
  m.push(v ? `<div class="metric"><div class="v">${v.held}<small> / ${v.replayed}</small></div><div class="k">solutions re-checked, goal holds again</div>
      <div class="s">run ${esc(v.run)} · replayed on fresh copies</div></div>`
    : `<div class="metric"><div class="v">–</div><div class="k">re-checked solutions</div><div class="s">not checked yet · <a href="#results">check</a></div></div>`);
  const c = latestComplete(r.runs);
  m.push(c ? `<div class="metric"><div class="v">${c.solvable}<small> solved</small></div><div class="k">latest complete evaluation: ${c.rooms} real rooms</div>
      <div class="split"><span>${c.impossible} impossible</span><span>${c.stuck + c.budget} stopped, asked for help</span><span>${c.already_true} already true</span></div>
      <div class="s">${esc(c.run)} · ${esc(plannerName(c.model))} · ${usd(c.usd)}</div></div>`
    : `<div class="metric"><div class="v">–</div><div class="k">unattended evaluation</div><div class="s">none finished yet</div></div>`);
  const L = longest(r.runs);
  m.push(L && L.rooms_planned > (c ? c.rooms : 0) ? `<div class="metric"><div class="v">${L.rooms}<small> / ${L.rooms_planned}</small></div>
      <div class="k">rooms in the ${L.finished ? "finished" : "running"} unattended run</div>
      <div class="split"><span>${L.hours >= 1 ? L.hours.toFixed(1) + " h" : Math.round(L.hours * 60) + " min"} unattended</span><span>${L.solvable} solved</span><span>${L.help} help requests</span></div>
      <div class="s">${esc(L.run)} · ${usd(L.usd)} so far · ${L.finished ? esc(L.stopped || "finished") : "in progress"}</div></div>`
    : `<div class="metric"><div class="v">${r.runs.length}</div><div class="k">unattended runs recorded</div><div class="s"><a href="#results">see them</a></div></div>`);
  $("#metrics").innerHTML = m.join("");

  const f = r.featured;
  if (!f) { $("#pipeline").innerHTML = '<div class="skeleton">No re-checked photo-built example yet: run an unattended evaluation, then "Re-check".</div>'; return; }
  $("#pipe-sub").innerHTML = `A real ${esc((f.type || "room").replace("_", " "))} from the NYU Depth V2 dataset (${esc(f.room.split("/").pop())}), rebuilt from
    <b>one photo</b>, solved by the agent during an unattended run, then <b>re-checked</b> by replaying the kept actions on a fresh copy.`;
  const fig = (body, title, cap) => `<figure>${body}<figcaption><b>${title}</b>${cap}</figcaption></figure>`;
  $("#pipeline").innerHTML =
    fig(`<img src="${f.photo}" alt="the original photo">`, "1 · One ordinary photo", "lens and depth estimated from the photo itself") +
    fig(`<img src="${f.detections}" alt="objects found">`, "2 · Objects found", `${f.objects} objects named and outlined`) +
    fig(`<img class="plan" src="${f.floor_plan}" alt="floor plan">`, "3 · Measured room", f.room_m ? `${f.room_m[0]} × ${f.room_m[1]} m, every object a 3D box` : "seen from above") +
    fig(`<video src="${f.video}" ${f.stills ? `poster="${f.stills}"` : ""} autoplay muted loop playsinline preload="metadata" aria-label="physics replay"></video>`, "4 · Physics + agent",
      `task "${esc(f.task)}": kept ${esc((f.kept || []).join(", "))}; on replay the goal holds again (simulated)`);
}
$("#cta-upload").onclick = () => { location.hash = "#lab"; setTimeout(() => $("#photo").click(), 60); };

// ================================================================ timeline (shared by experiments, safety, lab)
function timeline(rec) {
  const items = [];
  items.push(`<li class="k-goal"><span class="lbl">GOAL</span>${esc(rec.task?.goal || rec.task?.title || "")}
    <span class="why">measured at the start: ${esc(rec.task?.start || "-")}</span></li>`);
  const rules = rec.planner === "fixed rules";
  (rec.steps || []).forEach((s) => {
    if (s.type === "plan") {
      items.push(`<li class="k-plan"><span class="lbl">PLAN</span>${rules ? "Fixed rules proposed" : esc(plannerName(rec.planner)) + " proposed"}
        ${s.usd ? `<span class="muted">· ${usd(s.usd)}</span>` : ""}
        ${s.reasoning ? `<span class="q">"${esc(s.reasoning)}"</span>` : ""}
        <span class="why">${s.proposals.length ? s.proposals.map((p) => `<span class="act">${esc(p)}</span>`).join(" ") : "no actions: nothing it could try"}</span></li>`);
    } else if (s.type === "try") {
      const cls = s.status === "done" ? "ok" : s.status === "refused" ? "bad" : "warn";
      const moved = Object.entries(s.moved_cm || {}).map(([k, v]) => `${k} moved ${Math.round(v)} cm`).join(", ");
      items.push(`<li class="k-try ${cls}"><span class="lbl">TRY</span><span class="act">${esc(s.action)}</span> → <b>${esc(s.status)}</b>
        <span class="why">${s.status === "done" ? esc(moved || "done") : esc(clean(s.result))}</span>
        <span class="why">goal after: ${esc(s.goal_after)} · run on a snapshot, then undone</span></li>`);
    } else if (s.type === "refused") {
      items.push(`<li class="k-refused"><span class="lbl">REFUSED</span><span class="act">${esc(s.action)}</span><span class="why">${esc(s.why)}</span></li>`);
    } else if (s.type === "commit") {
      items.push(`<li class="k-commit"><span class="lbl">KEPT</span><span class="act">${esc(s.action)}</span> → goal: ${esc(s.goal_after)}
        <span class="why">${s.matches_try ? "ended exactly as its try predicted" : "differed from its try"}</span></li>`);
    }
  });
  const end = { simulated_success: "good", infeasible: "info" }[rec.outcome] || "warn";
  items.push(`<li class="k-end ${end}"><span class="lbl">SUPERVISOR</span><b>${esc(OUTCOME[rec.outcome] || rec.outcome || rec.label)}</b>
    ${rec.reason && rec.outcome !== "simulated_success" ? `<span class="why">${esc(rec.reason)}</span>` : ""}
    ${rec.help ? `<span class="why"><b>Asks for help:</b> ${esc(rec.help)}</span>` : ""}</li>`);
  return `<div class="tl-head"><b>Step by step</b><button class="btn small" data-replay>▶ Replay</button></div><ol class="tl">${items.join("")}</ol>`;
}

function bindReplay(root) {
  const btn = $("[data-replay]", root);
  if (!btn) return;
  clearInterval(root._replay);                         // a redrawn panel never inherits an old replay
  btn.onclick = () => {
    clearInterval(root._replay);
    const lis = $$(".tl li", root), video = $("video", root);
    lis.forEach((l) => l.classList.add("pending"));
    if (video) { try { video.currentTime = 0; } catch (e) { /* not loaded yet */ } video.play().catch(() => {}); }
    btn.disabled = true;
    let i = 0;
    const iv = root._replay = setInterval(() => {
      lis.forEach((l) => l.classList.remove("now"));
      if (i >= lis.length || !btn.isConnected) { clearInterval(iv); btn.disabled = false; lis.forEach((l) => l.classList.remove("pending")); return; }
      lis[i].classList.remove("pending"); lis[i].classList.add("now");
      lis[i].scrollIntoView({ block: "nearest", behavior: "smooth" });
      i++;
    }, 950);
  };
}

function media(rec, cls = "") {
  if (rec.video) return `<video class="${cls}" src="${rec.video}" ${cls ? "controls" : `autoplay muted loop ${rec.stills ? `poster="${rec.stills}"` : ""}`} playsinline preload="metadata"></video>`;
  if (rec.stills) return `<img class="${cls || "strip"}" src="${rec.stills}" alt="the simulated room, before and after">`;
  return "";
}

function stats(rec) {
  return `<div class="stats"><div><b>${rec.tries}</b><span>attempts</span></div><div><b>${rec.commits}</b><span>kept</span></div>
    <div><b>${(rec.sim_seconds || 0).toFixed(0)} s</b><span>simulated</span></div>
    <div><b>${rec.planner === "fixed rules" ? "free" : usd(rec.usd)}</b><span>${rec.planner === "fixed rules" ? "no AI" : "Claude cost"}</span></div></div>`;
}

// ================================================================ experiments
const EXP = {
  light: { name: "Light", blocks: "A box (1 kg, a catalog value) and a chair stand in the gap.",
    expect: "Solves it by moving the box out of the way.",
    limits: "Hand-made scene with known answers. The 1 kg mass is a catalog typical value. Success is in simulation only." },
  heavy: { name: "Heavy", blocks: "A box set to 40 kg stands in the gap. The simulated arm lifts at most 15 kg.",
    expect: "Stops as stuck, and asks for help with what it tried.",
    limits: "40 kg is a test value set by hand, not estimated. The 15 kg limit belongs to the simulated body. Pushing and pulling are excluded by the box's catalog affordances." },
  wall: { name: "Wall", blocks: "Fixed furniture fills the gap.",
    expect: "Stops at once: impossible by geometry, nothing worth trying.",
    limits: "Impossible means there is no route for a 0.4 m wide body even if every movable thing were removed, in this simulated layout." },
};
const isDefault = (r) => r.limits && r.limits.max_tries === 12 && r.limits.max_usd === 0.05;
const pick = (recs) => recs.find((r) => isDefault(r) && r.planner !== "fixed rules") || recs.find(isDefault) || recs[0];
const recName = (r) => `${plannerName(r.planner)} · ${r.time.slice(11, 16)}${isDefault(r) ? "" : r.limits.max_tries < 12 ? ` · ${r.limits.max_tries}-try limit` : ` · $${r.limits.max_usd} budget`}`;
const chosen = {};

async function loadExperiments(q, force) {
  let ex;
  try { ex = await getExperiments(force); } catch (e) { $("#exp-grid").innerHTML = `<div class="skeleton">${esc(e.message)}</div>`; return; }
  $("#exp-grid").innerHTML = Object.entries(EXP).map(([v, meta]) => {
    const recs = ex[v].records;
    const r = recs.find((x) => x.time === chosen[v]) || pick(recs);
    if (r) chosen[v] = r.time;
    return `<article class="exp" id="exp-${v}">
      <div class="media">${r ? media(r) : ""}</div>
      <div class="body">
        <h3>${meta.name} ${r ? tag(r.label) : '<span class="tag slate">not recorded yet</span>'}</h3>
        <p>${esc(meta.blocks)}</p>
        <p class="muted">Expected: ${esc(meta.expect)}</p>
        ${r ? stats(r) : '<p class="muted">No recorded run yet. Run it live.</p>'}
        ${recs.length > 1 ? `<label>Recording <select data-rec="${v}">${recs.map((x) => `<option value="${esc(x.time)}" ${r && x.time === r.time ? "selected" : ""}>${esc(recName(x))}</option>`).join("")}</select></label>` : ""}
        <div class="acts">${r ? `<button class="btn primary small" data-open="${v}">Step by step</button>` : ""}
          <button class="btn small live-only" data-live="${v}" data-planner="claude">Run live with Claude</button>
          <button class="btn ghost small live-only" data-live="${v}" data-planner="rules">Run with rules</button></div>
        <div class="progress hidden" id="exp-prog-${v}"></div>
      </div></article>`;
  }).join("");
  $$("[data-rec]").forEach((s) => s.onchange = () => { chosen[s.dataset.rec] = s.value; loadExperiments(); openExp(s.dataset.rec); });
  $$("[data-open]").forEach((b) => b.onclick = () => openExp(b.dataset.open, true));
  $$("[data-live]").forEach((b) => b.onclick = () => runLive(b.dataset.live, b.dataset.planner, b));
  const open = q && q.get && q.get("open");
  if (open && EXP[open]) openExp(open, true);
}

function openExp(v, scroll) {
  const recs = data.experiments[v].records;
  const r = recs.find((x) => x.time === chosen[v]) || pick(recs);
  if (!r) return;
  $$(".exp").forEach((e) => e.classList.toggle("sel", e.id === "exp-" + v));
  const d = $("#exp-detail");
  d.classList.remove("hidden");
  d.innerHTML = detailHTML(EXP[v].name + ": " + r.task.title, r, EXP[v].limits);
  bindReplay(d);
  if (scroll) d.scrollIntoView({ behavior: "smooth", block: "start" });
}

function detailHTML(title, r, limits) {
  return `<div>${detailLeft(title, r, limits)}</div><div>${timeline(r)}</div>`;
}

function detailLeft(title, r, limits) {
  return `
      <h3>${esc(title)}</h3>
      <div class="result-head">${tag(r.label)} ${src(r.room_source)} <span class="muted">recorded ${esc(r.time.replace("T", " "))}</span></div>
      ${media(r, "big")}
      <dl class="kv"><dt>Task</dt><dd>${esc(r.task.title)}</dd><dt>Goal</dt><dd>${esc(r.task.goal)}</dd>
        <dt>Planner</dt><dd>${esc(plannerName(r.planner))}</dd>
        <dt>Limits</dt><dd>${r.limits.max_tries} tries · ${usd(r.limits.max_usd)} per task</dd></dl>
      ${stats(r)}
      <div class="box"><b>Supervisor decision:</b> ${esc(OUTCOME[r.outcome] || r.outcome)}${r.reason && r.outcome !== "simulated_success" ? `: ${esc(r.reason)}` : "."}
        ${r.committed?.length ? `<br><b>Kept:</b> ${r.committed.map(esc).join(" → ")}` : ""}
        <br><span class="muted">${esc((r.verdict || "").split("\n")[0])}</span></div>
      ${limits ? `<div class="box lim"><b>Limitations:</b> ${esc(limits)}</div>` : ""}
      <p class="muted"><a href="${r.transcript}" target="_blank" rel="noopener">full transcript</a></p>`;
}

async function runLive(v, planner, btn) {
  const pr = progress($(`#exp-prog-${v}`), `Running ${EXP[v].name} with ${planner === "claude" ? "Claude" : "fixed rules"} …`,
    "Usually 5–15 seconds. Tries run on a snapshot of the room and are undone.");
  $$(`[data-live="${v}"]`).forEach((b) => (b.disabled = true));
  try {
    const { job } = await api("/api/run", { room: `test/${v}`, task: { kind: "route" }, settings: { planner, video: true } });
    const j = await follow(job, pr.line);
    pr.done();
    if (j.status !== "done") throw new Error(j.error || "the run failed");
    chosen[v] = j.result.time;
    await loadExperiments(null, true);
    openExp(v, true);
    toast(`${EXP[v].name}: ${(LABEL[j.result.label] || [, j.result.label])[1]}`);
  } catch (e) { pr.done(); toast(e.message, true); $$(`[data-live="${v}"]`).forEach((b) => (b.disabled = false)); }
}

// ================================================================ live lab
let room = null, labReady = false;
async function loadLab(q) {
  const o = await refreshBudget();
  if (!o) return;
  if (!labReady) {
    $("#model").innerHTML = o.models.map((m) => `<option value="${m}">${esc(plannerName(m))}</option>`).join("");
    labReady = true;
  }
  fillRooms(o.rooms, q && q.get && q.get("room"));
}

function fillRooms(rooms, select) {
  const groups = {};
  rooms.forEach((r) => (groups[r.group] = groups[r.group] || []).push(r));
  const cur = select || $("#room").value;
  $("#room").innerHTML = '<option value="">Choose a room …</option>' + Object.entries(groups).map(([g, rs]) =>
    `<optgroup label="${esc(g)}">${rs.map((r) => `<option value="${esc(r.id)}">${esc(r.name)}</option>`).join("")}</optgroup>`).join("");
  if (cur) { $("#room").value = cur; if (!room || room.id !== cur) showRoom(cur); }
}

function prov(o, isPhoto) {
  const mass = String(o.mass || "");
  const massP = /TEST VALUE|set by hand/i.test(mass) ? '<span class="prov hand">set by hand</span>' : mass ? '<span class="prov est">catalog</span>' : "";
  return {
    name: isPhoto ? '<span class="prov seen">observed</span>' : '<span class="prov hand">set by hand</span>',
    size: isPhoto ? '<span class="prov est">estimated</span>' : '<span class="prov hand">set by hand</span>',
    mass: massP,
  };
}

async function showRoom(id) {
  if (!id) { $("#roomview").classList.add("hidden"); return; }
  try { room = await api(`/api/room?id=${encodeURIComponent(id)}`); } catch (e) { toast(e.message, true); return; }
  const isPhoto = !id.startsWith("test/");
  $("#roomview").classList.remove("hidden");
  $("#runcard").classList.add("hidden");
  $("#ideas").innerHTML = "";
  $("#room-src").outerHTML = `<span class="src ${isPhoto ? "photo" : "hand"}" id="room-src">${isPhoto ? "photo-built" : "hand-made"}</span>`;
  const fig = (s, cap, cls = "") => s ? `<figure><img class="${cls}" src="${s}" alt="${esc(cap)}" loading="lazy"><figcaption>${esc(cap)}</figcaption></figure>` : "";
  $("#imgs").innerHTML = isPhoto ? fig(room.photo, "the photo") + fig(room.detections, "objects found") + fig(room.floor_plan, "measured room, from above", "plan")
    : '<p class="muted">A hand-made test scene: no photo. Its sizes are set by hand so the right answer is known.</p>';
  const ph = room.physics || {};
  $("#roommeta").textContent = [room.room_m ? `room ${room.room_m.slice(0, 2).join(" × ")} m` : "", `${room.objects.length} objects`,
    ph.settle_ok === false ? "some objects moved when it settled" : ph.settle_ok ? "settles in physics" : "",
    ph.agent_standing === false ? "the body cannot stand here" : "", room.source].filter(Boolean).join(" · ");
  $("#nobj").textContent = room.objects.length;
  const inf = room.objects.some((o) => (o.inferred || []).length);
  $("#objects").innerHTML = `<tr><th>object</th><th>movable</th><th>rests on <span class="muted">(physics contacts)</span></th><th>size, m</th><th>mass</th>${inf ? "<th>inferred, not seen</th>" : ""}</tr>` +
    room.objects.map((o) => { const p = prov(o, isPhoto); return `<tr><td>${esc(o.label)} ${p.name}</td><td>${o.movable_in_simulation ? "yes" : "fixed"}</td>
      <td>${esc((o.rests_on_now || []).join(", ") || "-")}</td><td>${o.footprint_m.join(" × ")} × ${o.height_m} ${p.size}</td>
      <td>${esc(String(o.mass || "-").replace(/\s*\[.*\]/, ""))} ${p.mass}</td>${inf ? `<td class="muted">${esc((o.inferred || []).join("; "))}</td>` : ""}</tr>`; }).join("");
  $("#obj").innerHTML = room.objects.map((o) => `<option>${esc(o.label)}</option>`).join("");
  $("#tgt").innerHTML = ["floor", ...room.surfaces].map((s) => `<option>${esc(s)}</option>`).join("");
  $("#kind").value = id.startsWith("test/") ? "route" : (room.movable.length ? "rests_on" : "route");
  if (room.movable.length) $("#obj").value = room.movable[0];
  syncKind();
  history.replaceState(null, "", `#lab?room=${encodeURIComponent(id)}`);
}

function syncKind() {
  const k = $("#kind").value;
  $("#objwrap").classList.toggle("hidden", k === "route");
  $("#tgtwrap").classList.toggle("hidden", k !== "rests_on");
}
$("#kind").onchange = syncKind;
$("#room").onchange = (e) => showRoom(e.target.value);

$("#photo").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  e.target.value = "";
  const busy = (data.overview && data.overview.busy_runs) || [];
  const pr = progress($("#build"), `Rebuilding the room from ${f.name}`,
    "A new photo usually takes about a minute: depth and lens, floor and walls, objects and outlines, then physics. The first upload also loads the networks (about 20 s more). A photo seen before takes seconds."
    + (busy.length ? ` Right now an unattended run (${busy.join(", ")}) is sharing the GPU, so this can take several minutes.` : ""));
  try {
    pr.line(`uploading ${(f.size / 1e6).toFixed(1)} MB …`);
    const r = await fetch("/api/upload", { method: "POST", headers: { "X-Filename": encodeURIComponent(f.name) }, body: f });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || "upload failed");
    const done = await follow(j.job, pr.line);
    if (done.status !== "done") throw new Error(done.error || "the room could not be built from this photo");
    pr.done(false);
    const o = await refreshBudget();
    fillRooms(o.rooms, done.result.room);
    toast("Room rebuilt. Give it a task below.");
  } catch (err) { pr.done(false); pr.line("✗ " + err.message); toast(err.message, true); }
};

$("#suggest").onclick = async () => {
  const b = $("#suggest"); b.disabled = true; b.textContent = "Claude is reading the room …";
  $("#ideas").innerHTML = "";
  try {
    const { job } = await api("/api/suggest", { room: room.id, n: 3, model: $("#model").value });
    const j = await follow(job);
    if (j.status !== "done") throw new Error(j.error || "no ideas");
    $("#ideas").innerHTML = j.result.tasks.map((t, i) => `<button class="chip" data-i="${i}" title="${esc(t.why)}">${esc(t.title)}</button>`).join("")
      + (j.result.dropped.length ? `<span class="muted">${j.result.dropped.length} idea(s) dropped before trying: not possible here</span>` : "")
      + `<span class="muted">${usd(j.result.usd)}</span>`;
    $$(".chip", $("#ideas")).forEach((c) => c.onclick = () => {
      const t = j.result.tasks[+c.dataset.i];
      $("#kind").value = t.kind; syncKind();
      if (t.object) $("#obj").value = t.object;
      if (t.target) $("#tgt").value = t.target;
      toast(`Task set: ${t.title}`);
    });
  } catch (err) { toast(err.message, true); }
  b.disabled = false; b.textContent = "Ask Claude for task ideas";
};

$("#go").onclick = async () => {
  if (!room) return toast("Choose a room first.", true);
  const go = $("#go");
  go.disabled = true;
  $("#runcard").classList.remove("hidden");
  $("#result").innerHTML = "";
  const pr = progress($("#run-progress"), "The agent is working …",
    "Usually 5–30 seconds. Each try runs on a snapshot and is undone; only what measurably helps is kept. Recording video adds a few seconds.");
  $("#runcard").scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const task = { kind: $("#kind").value, object: $("#obj").value, target: $("#tgt").value };
    const settings = { planner: $("#planner").value, model: $("#model").value, task_usd: +$("#usd").value, max_tries: +$("#tries").value, video: $("#video").checked };
    const { job } = await api("/api/run", { room: room.id, task, settings });
    const j = await follow(job, pr.line);
    pr.done();
    if (j.status !== "done") throw new Error(j.error || "the run failed");
    const r = j.result;
    if (r.label === "rejected") {
      $("#result").innerHTML = `<div class="result-head">${tag("rejected")}</div><p>${esc(r.reason)}</p><p class="muted">Checked against the room before any attempt, so nothing was tried.</p>`;
    } else {
      $("#result").innerHTML = `<div class="result-grid"><div>${detailLeft(r.task.title, r, "")}</div><div>${timeline(r)}</div></div>`;
      bindReplay($("#result"));
    }
  } catch (err) { pr.done(false); toast(err.message, true); }
  go.disabled = false;
};

// ================================================================ safety
const RULES = [
  { title: "Stuck detection", what: "If no tried action measurably helps for two rounds, or nothing untried is left, the task stops and asks for help.",
    signal: "measured progress per round: cm of route still blocked, contacts under an object",
    tests: ["stops and asks for help when nothing works", "a planner that repeats itself is refused then stopped"],
    real: (c, L) => [c && `${c.run}: ${pl(c.stuck, "task")} stopped as stuck`, L && L !== c && `${L.run}: ${L.stuck} so far`], exp: "heavy" },
  { title: "Repeated-action refusal", what: "An action already tried in exactly the same situation is refused before it runs. The simulation is deterministic, so it would end the same way.",
    signal: "pairs of (simulated-state fingerprint, action) already tried",
    tests: ["repeats loops progress and budgets", "a planner that repeats itself is refused then stopped"],
    real: (c, L) => [c && `${c.run}: ${pl(c.refused_repeats, "repeated proposal")} refused`, L && L !== c && `${L.run}: ${L.refused_repeats} so far`] },
  { title: "Loop prevention", what: "A change is kept only if it strictly improves the measured goal, and never if it returns the room to a state seen before.",
    signal: "strictly better goal measurement; fingerprints of every kept state",
    tests: ["repeats loops progress and budgets"],
    real: () => ["Has not triggered in a real run yet: because only strict improvement is kept, a loop can't form. Repeats are refused earlier (card above)."] },
  { title: "Attempt limits", what: "Each task has a cap on tries (each on a snapshot, then undone), on kept actions and on simulated time.",
    signal: "counts of tries, kept actions, simulated seconds",
    tests: ["repeats loops progress and budgets"],
    real: (c, L) => [c && `${c.run}: ${pl(c.budget, "task")} stopped on a budget`], record: (ex) => ex.heavy.records.find((r) => r.limits.max_tries === 1),
    recordLabel: "Recorded: heavy room with a 1-try limit", live: { v: "heavy", planner: "rules", max_tries: 1, label: "Run it (free, no AI)" } },
  { title: "Compute and API budgets", what: "Every Claude call is priced from its token usage. Each task, room, unattended run and this app has a dollar cap, and nothing runs past it.",
    signal: "tokens × list price, counted per call",
    tests: ["money budget stops the task", "cost is counted at list prices", "money time and failure streaks stop the night", "the apps money cap refuses claude"],
    real: (c, L) => [c && `${c.run}: ${usd(c.usd)} for ${c.rooms} rooms`, L && L !== c && `${L.run}: ${usd(L.usd)} so far`],
    record: (ex) => ex.light.records.find((r) => r.limits.max_usd < 0.001), recordLabel: "Recorded: light room with a $0.0005 budget",
    live: { v: "light", planner: "claude", task_usd: 0.0005, label: "Run it (under $0.001)" } },
  { title: "Objective goal checks", what: "Success is measured by geometry and physics contacts, never by the AI's own claim. Tries are undone bit for bit, and a try predicts its commit exactly.",
    signal: "walking-route search for a 0.4 m body; contacts under an object",
    tests: ["route checks are deterministic and objective", "tries leave the world bit identical", "a try predicts its commit and the commit stays", "reach uses the arms reach not the objects centre"],
    real: (c, L, v) => [v && `${v.held} of ${v.replayed} solutions hold again when replayed (${v.run})`] },
  { title: "Stop or ask for help", what: "Impossible goals stop at once. Stuck, budget and planner failures stop with a request that says what was tried and why it failed. An unattended run also stops itself on money, time or failure streaks.",
    signal: "goal feasibility (no route even with every movable thing removed); the supervisors' limits",
    tests: ["stops at once when the goal is infeasible", "planner failure stops the task with a reason", "refusal and cut off answers raise", "money time and failure streaks stop the night"],
    real: (c, L) => [c && `${c.run}: ${pl(c.help, "help request")}, ${c.impossible} impossible`, L && L !== c && `${L.run}: ${L.help} help requests so far`], exp: "wall" },
];

async function loadSafety() {
  let r, ex;
  try { [r, ex] = await Promise.all([getResults(true), getExperiments(true)]); } catch (e) { $("#rules").innerHTML = `<div class="skeleton">${esc(e.message)}</div>`; return; }
  const byName = Object.fromEntries((r.tests?.tests || []).map((t) => [t.name, t.status]));
  const c = latestComplete(r.runs), L = longest(r.runs), v = verified(r.runs)[0];
  $("#rules").innerHTML = RULES.map((rule, i) => {
    const rec = rule.record && rule.record(ex);
    const tests = rule.tests.map((t) => byName[t] ? `<div class="t ${byName[t] === "ok" ? "" : "no"}">test: ${esc(t)}</div>` : `<div class="t no">test: ${esc(t)} (not run yet)</div>`).join("");
    const real = rule.real(c, L, v).filter(Boolean).map((s) => `<div class="r">${esc(s)}</div>`).join("");
    return `<div class="rule"><h3>${esc(rule.title)}</h3><p>${esc(rule.what)}</p>
      <div class="sig"><b>Decides from:</b> ${esc(rule.signal)}</div>
      <div class="ev">${tests}${real}</div>
      <div class="acts">${rule.exp ? `<a class="btn small" href="#experiments?open=${rule.exp}">See it: ${EXP[rule.exp].name} room</a>` : ""}
        ${rec ? `<button class="btn small" data-rec-i="${i}">${esc(rule.recordLabel)}</button>` : ""}
        ${rule.live ? `<button class="btn ghost small live-only" data-live-i="${i}">${esc(rule.live.label)}</button>` : ""}</div></div>`;
  }).join("");
  $$("[data-rec-i]").forEach((b) => b.onclick = () => showSafety(RULES[+b.dataset.recI].record(data.experiments), RULES[+b.dataset.recI].title));
  $$("[data-live-i]").forEach((b) => b.onclick = async () => {
    const L2 = RULES[+b.dataset.liveI].live;
    b.disabled = true;
    try {
      const { job } = await api("/api/run", { room: `test/${L2.v}`, task: { kind: "route" }, settings: { planner: L2.planner, video: true, ...(L2.max_tries ? { max_tries: L2.max_tries } : {}), ...(L2.task_usd ? { task_usd: L2.task_usd } : {}) } });
      const j = await follow(job);
      if (j.status !== "done") throw new Error(j.error || "the run failed");
      showSafety(j.result, RULES[+b.dataset.liveI].title + ", just now");
      data.experiments = null;
    } catch (e) { toast(e.message, true); }
    b.disabled = false;
  });
}

function showSafety(rec, title) {
  if (!rec) return toast("No recorded run yet: use the run button.", true);
  const d = $("#safety-detail");
  d.classList.remove("hidden");
  d.innerHTML = detailHTML(title, rec, "A hand-made test scene, used here to make one rule fire on purpose.");
  bindReplay(d);
  d.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ================================================================ results
const RUN_NOTES = {   // what the logs alone can't say (see RESULTS.md)
  test10: "first run, before the reach fix: 1 of its 3 impossible was wrong",
  "live-rehearsal": "our own dorm photo", smoke1: "one-room smoke test",
};

function testsHTML(t) {
  if (!t) return '<p class="muted">Not run yet. Press "Run all tests".</p>';
  const sections = {};
  t.tests.forEach((x) => (sections[x.section] = sections[x.section] || []).push(x));
  return `<div class="tests-sum">${t.ok ? '<span class="tag good">✓ all passed</span>' : '<span class="tag bad">✗ failures</span>'}
    <b>${t.passed} of ${t.total}</b><span class="muted">${esc(t.summary)} · ${esc(t.time.replace("T", " "))} · no network, no cost: Claude is replaced by a fake API in tests</span></div>` +
    Object.entries(sections).map(([s, xs]) => `<details class="tsec"><summary>${esc(s)} <span class="muted">${xs.filter((x) => x.status === "ok").length}/${xs.length}</span></summary>
      <table class="tbl">${xs.map((x) => `<tr><td class="${x.status === "ok" ? "" : "fail"}">${esc(x.name)}</td><td class="num muted">${esc(x.status)}</td></tr>`).join("")}</table></details>`).join("");
}

async function loadResults() {
  let r;
  try { r = await getResults(true); } catch (e) { $("#tests").innerHTML = `<p class="muted">${esc(e.message)}</p>`; return; }
  $("#tests").innerHTML = testsHTML(r.tests);
  const vs = verified(r.runs);
  $("#verify").innerHTML = (vs.length ? vs.map((v) => `<p><span class="tag good">${v.held} / ${v.replayed}</span> solutions of <b>${esc(v.run)}</b> hold again on a fresh copy
      <span class="muted">· checked ${esc(v.time.replace("T", " "))}${v.failed.length ? " · failed: " + esc(v.failed.join("; ")) : ""}</span></p>`).join("") : '<p class="muted">Not checked yet.</p>')
    + `<div class="row live-only">${r.runs.filter((x) => x.solvable).map((x) => `<button class="btn small" data-verify="${esc(x.run)}">Re-check ${esc(x.run)} (${x.solvable} solved)</button>`).join("")}</div>
       <div class="progress hidden" id="verify-prog"></div>`;
  $("#runs").innerHTML = r.runs.length ? `<div class="scroll"><table class="tbl"><tr><th>run</th><th class="num">rooms</th><th class="num">time</th><th class="num">solved</th>
      <th class="num">impossible</th><th class="num">stopped</th><th class="num">already true</th><th class="num">repeats refused</th><th class="num">help</th><th class="num">cost</th><th></th></tr>
      ${r.runs.map((x) => `<tr><td><b>${esc(x.run)}</b><br><span class="muted">${esc(plannerName(x.model))} · ${x.finished ? esc(x.stopped || "finished") : "running"}${RUN_NOTES[x.run] ? " · " + esc(RUN_NOTES[x.run]) : ""}</span></td>
      <td class="num">${x.rooms}${x.rooms_planned && x.rooms_planned !== x.rooms ? " / " + x.rooms_planned : ""}</td>
      <td class="num">${x.hours >= 1 ? x.hours.toFixed(1) + " h" : Math.round(x.hours * 60) + " min"}</td><td class="num">${x.solvable}</td>
      <td class="num">${x.impossible}</td><td class="num">${x.stuck + x.budget}</td><td class="num">${x.already_true}</td><td class="num">${x.refused_repeats}</td>
      <td class="num">${x.help}</td><td class="num">${usd(x.usd)}</td><td>${x.dashboard ? `<a href="${x.dashboard}" target="_blank" rel="noopener">run page</a>` : ""}</td></tr>`).join("")}</table></div>`
    : '<p class="muted">No unattended runs yet. Start a small one.</p>';
  $("#appruns").innerHTML = r.app_runs.length ? `<div class="scroll"><table class="tbl"><tr><th>when</th><th>room</th><th>task</th><th>planner</th><th>outcome</th>
      <th class="num">tries</th><th class="num">kept</th><th class="num">cost</th></tr>${r.app_runs.map((x) => `<tr><td class="muted">${esc((x.time || "").slice(5, 16).replace("T", " "))}</td>
      <td>${src(x.room_source)} ${esc(x.room.split("/").pop())}</td><td>${esc(x.task?.title || "")}</td><td>${esc(plannerName(x.planner))}</td><td>${tag(x.label)}${x.label === "unresolved" ? ` <span class="muted">${esc(x.outcome)}</span>` : ""}</td>
      <td class="num">${x.tries}</td><td class="num">${x.commits}</td><td class="num">${x.planner === "fixed rules" ? "free" : usd(x.usd)}</td></tr>`).join("")}</table></div>`
    : '<p class="muted">No recorded runs yet.</p>';
  const tbl = (b) => `<p class="muted">${esc(b.source)}</p><table class="tbl">${b.rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num"><b>${esc(v)}</b></td></tr>`).join("")}</table>`;
  $("#bench").innerHTML = tbl(r.benchmark);
  $("#cmds").innerHTML = tbl(r.commands);
  $$("[data-verify]").forEach((b) => b.onclick = async () => {
    b.disabled = true;
    const pr = progress($("#verify-prog"), `Re-checking ${b.dataset.verify} …`, "Replays each solution's kept actions on a fresh copy of its room. A few seconds.");
    try {
      const { job } = await api("/api/verify", { run: b.dataset.verify });
      const j = await follow(job, pr.line);
      pr.done();
      if (j.status !== "done") throw new Error(j.error);
      toast(`${j.result.held} of ${j.result.replayed} solutions hold again`);
      loadResults();
    } catch (e) { pr.done(false); toast(e.message, true); b.disabled = false; }
  });
}

$("#runtests").onclick = async () => {
  const b = $("#runtests"); b.disabled = true; b.textContent = "Running the tests …";
  try {
    const { job } = await api("/api/tests", {});
    const j = await follow(job);
    if (!j.result) throw new Error(j.error || "the tests did not run");
    $("#tests").innerHTML = testsHTML(j.result);
    data.results = null;
    toast(`${j.result.passed} of ${j.result.total} tests passed`);
  } catch (e) { toast(e.message, true); }
  b.disabled = false; b.textContent = "Run all tests";
};

$("#startrun").onclick = async () => {
  const b = $("#startrun"); b.disabled = true;
  const n = +$("#nrooms").value;
  const pr = progress($("#nightprog"), `Unattended run on ${n} real rooms …`,
    `About 1–2 minutes per new room (rebuilding from the photo takes most of it). It continues if you leave this page.`);
  try {
    const { job } = await api("/api/night", { rooms: n });
    const j = await follow(job, pr.line);
    pr.done(false);
    if (j.status !== "done") throw new Error(j.error || "the run stopped");
    toast("Run finished");
    loadResults();
  } catch (e) { pr.done(false); toast(e.message, true); }
  b.disabled = false;
};

// ================================================================ start
refreshBudget();
route();

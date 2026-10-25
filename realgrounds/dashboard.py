"""The night as one page: dashboard.html in the run folder (open it in a browser; it refreshes while the run goes).

Built only from what the run wrote (results.jsonl, timeline.jsonl, the room folders): every number on the page
can be traced to a line in those files. Images are linked by relative path, so the page works where it lies.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

LABELS = {   # label -> (icon, words); never colour alone
    "solvable": ("✓", "solvable in simulation"),
    "impossible": ("⊘", "impossible (geometry)"),
    "unresolved": ("?", "stopped, asked for help"),
    "already_true": ("=", "already true"),
    "skipped": ("–", "skipped (room budget)"),
}

CSS = """
:root { color-scheme: light; --surface-0:#f4f3ef; --surface-1:#fcfcfb; --border:#e2e0d9; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --text-muted:#7a7873; --series-1:#2a78d6; --good:#008300; --warn:#b26f00; --accent-bg:#eef4fc; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark; --surface-0:#121211;
  --surface-1:#1a1a19; --border:#2e2d2a; --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#8f8d86;
  --series-1:#3987e5; --good:#3fae3f; --warn:#d99a2b; --accent-bg:#1c2633; } }
:root[data-theme="dark"] { color-scheme: dark; --surface-0:#121211; --surface-1:#1a1a19; --border:#2e2d2a;
  --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#8f8d86; --series-1:#3987e5; --good:#3fae3f;
  --warn:#d99a2b; --accent-bg:#1c2633; }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface-0); color:var(--text-primary);
  font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 28px 16px 60px; }
h1 { font-size: 28px; margin: 0 0 4px; } h2 { font-size: 19px; margin: 34px 0 12px; }
.sub { color: var(--text-secondary); margin: 0 0 18px; }
.banner { background: var(--accent-bg); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px;
  color: var(--text-secondary); font-size: 14px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 18px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .v { font-size: 32px; font-weight: 650; line-height: 1.1; font-variant-numeric: tabular-nums; }
.tile .k { color: var(--text-secondary); font-size: 13px; margin-top: 4px; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.card h3 { font-size: 15px; margin: 0 0 8px; }
.chart { position: relative; } .chart svg { width: 100%; height: 190px; display: block; }
.chart .axis { stroke: var(--border); } .chart text { fill: var(--text-muted); font-size: 11px; }
.chart .line { fill: none; stroke: var(--series-1); stroke-width: 2; }
.chart .dot { fill: var(--series-1); stroke: var(--surface-1); stroke-width: 2; }
.chart .cross { stroke: var(--text-muted); stroke-dasharray: 3 3; }
.tip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 6px; padding: 4px 8px; font-size: 12px; color: var(--text-primary); display: none; white-space: nowrap; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { color: var(--text-secondary); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.chip { display: inline-block; border: 1px solid var(--border); border-radius: 999px; padding: 0 8px; font-size: 12px;
  color: var(--text-secondary); white-space: nowrap; }
.chip.solvable { color: var(--good); border-color: var(--good); } .chip.unresolved { color: var(--warn); border-color: var(--warn); }
.rooms { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 12px; }
.room img { width: 100%; border-radius: 6px; border: 1px solid var(--border); display: block; }
.room .imgs { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 8px 0; }
.room ul { margin: 6px 0 0; padding-left: 0; list-style: none; } .room li { margin: 4px 0; font-size: 14px; }
.muted { color: var(--text-muted); font-size: 13px; }
a { color: var(--series-1); }
@media (max-width: 520px) { .tile .v { font-size: 26px; } }
"""

CHART_JS = """
document.querySelectorAll('.chart').forEach(function (box) {
  var pts = JSON.parse(box.dataset.points), svg = box.querySelector('svg'), tip = box.querySelector('.tip');
  var cross = svg.querySelector('.cross'), W = 600, H = 190, L = 40, R = 10, T = 10, B = 26;
  if (!pts.length) return;
  var xmax = Math.max(pts[pts.length - 1][0], 1e-9), ymax = Math.max.apply(null, pts.map(function (p) { return p[1]; })) || 1;
  function X(v) { return L + (W - L - R) * v / xmax; } function Y(v) { return H - B - (H - T - B) * v / ymax; }
  svg.addEventListener('mousemove', function (e) {
    var r = svg.getBoundingClientRect(), x = (e.clientX - r.left) * W / r.width, best = pts[0];
    pts.forEach(function (p) { if (Math.abs(X(p[0]) - x) < Math.abs(X(best[0]) - x)) best = p; });
    cross.setAttribute('x1', X(best[0])); cross.setAttribute('x2', X(best[0])); cross.style.display = 'block';
    tip.style.display = 'block'; tip.textContent = best[2];
    tip.style.left = Math.min(r.width - 150, X(best[0]) * r.width / W + 8) + 'px'; tip.style.top = (Y(best[1]) * r.height / H - 30) + 'px';
  });
  svg.addEventListener('mouseleave', function () { tip.style.display = 'none'; cross.style.display = 'none'; });
});
"""


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _cut(s: str, n: int = 240) -> str:
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"


def _chart(title: str, points: list[tuple[float, float, str]], x_label: str, y_fmt: str = "{:.0f}") -> str:
    """One series, one axis: a line over time with a crosshair tooltip."""
    W, H, L, R, T, B = 600, 190, 40, 10, 10, 26
    if not points:
        return f'<div class="card"><h3>{_esc(title)}</h3><p class="muted">no data yet</p></div>'
    xmax = max(points[-1][0], 1e-9)
    ymax = max(p[1] for p in points) or 1
    X = lambda v: L + (W - L - R) * v / xmax                       # noqa: E731
    Y = lambda v: H - B - (H - T - B) * v / ymax                   # noqa: E731
    path = "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y, _ in [(0.0, 0.0, "")] + points)
    dots = "".join(f'<circle class="dot" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4"/>' for x, y, _ in points[-1:])
    ticks = "".join(f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{y_fmt.format(v)}</text>'
                    f'<line class="axis" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" opacity=".5"/>'
                    for v in (0, ymax / 2, ymax))
    xt = "".join(f'<text x="{X(v):.1f}" y="{H - 8}" text-anchor="middle">{v:.1f}</text>' for v in (0, xmax / 2, xmax))
    pts = json.dumps([[x, y, t] for x, y, t in points])
    return (f'<div class="card"><h3>{_esc(title)}</h3><div class="chart" data-points="{_esc(pts)}">'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(title)}">{ticks}{xt}'
            f'<path class="line" d="{path}"/>{dots}<line class="cross" y1="{T}" y2="{H - B}" style="display:none"/>'
            f'</svg><div class="tip"></div></div><div class="muted">x: {_esc(x_label)}</div></div>')


def _task_file(run: Path, room: str, k: int, t: dict) -> str | None:
    if t.get("file"):
        return t["file"]
    hits = sorted((run / "rooms" / room / "tasks").glob(f"{k}_*.txt"))
    return str(hits[0].relative_to(run)) if hits else None


def build(run: Path, refresh: bool = False) -> Path:
    rows = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines()] if (run / "results.jsonl").exists() else []
    events = [json.loads(l) for l in (run / "timeline.jsonl").read_text().splitlines()] if (run / "timeline.jsonl").exists() else []
    start = next((e for e in events if e["event"] == "night_start"), {})
    model = start.get("model", "?")
    total = start.get("rooms", len(rows))
    done = [e for e in events if e["event"] == "room_done"]
    elapsed_h = (events[-1]["elapsed_s"] / 3600) if events else 0.0
    stopped = next((e["reason"] for e in reversed(events) if e["event"] == "night_stopped"), "")
    finished = any(e["event"] == "night_end" for e in events) or bool(stopped)
    tasks = [(r, k, t) for r in rows for k, t in enumerate(r.get("tasks", []), 1)]
    count = lambda key, v: sum(t.get(key) == v for _, _, t in tasks)   # noqa: E731
    usd = sum(r.get("usd", 0) for r in rows)
    helps = [e for e in events if e["event"] == "help_request"]
    refused = sum(t.get("refused_repeats", 0) for _, _, t in tasks)
    dropped = sum(len(r.get("dropped", [])) for r in rows)
    built = [r for r in rows if r.get("built")]

    hours = elapsed_h >= 2                               # short runs read better in minutes
    unit, scale = ("hours", 3600) if hours else ("minutes", 60)
    kpis = [(f"{len(rows)} / {total}", "rooms processed"),
            (f"{elapsed_h:.1f} h" if hours else f"{elapsed_h * 60:.0f} min", "unattended"),
            (count("label", "solvable"), "tasks solved in simulation"), (count("label", "impossible"), "proven impossible"),
            (len(helps), "times it asked for help"), (refused, "repeated actions it refused"),
            (f"${usd:.2f}", f"Claude spend ({_esc(model)})")]

    # charts: cumulative rooms and spend over elapsed hours (two charts: never two y-axes)
    cum_rooms, cum_usd, n, s = [], [], 0, 0.0
    for e in done:
        n += 1
        s += e.get("usd") or 0.0
        h = e["elapsed_s"] / scale
        cum_rooms.append((h, n, f"{h:.1f} {unit}: {n} rooms ({e['room']})"))
        cum_usd.append((h, s, f"{h:.1f} {unit}: ${s:.4f}"))

    # one real example of each way it stopped itself
    examples = []
    for outcome, words in (("stuck", "stuck: nothing it tried helped"), ("budget", "out of budget (tries or money)"),
                           ("infeasible", "impossible: geometry, not effort"), ("planner_error", "the planner failed")):
        hit = next(((r, k, t) for r, k, t in tasks if t.get("outcome") == outcome), None)
        if hit:
            r, k, t = hit
            f = _task_file(run, r["room"], k, t)
            examples.append(f"<tr><td>{_esc(words)}</td><td>{_esc(r['room'])}: {_esc(t['title'])}</td>"
                            f"<td>{_esc(_cut(t.get('help') or t.get('reason') or ''))}</td>"
                            f"<td>{f'<a href={chr(34)}{_esc(f)}{chr(34)}>transcript</a>' if f else ''}</td></tr>")
    stop_rows = [("stuck", count("outcome", "stuck")), ("budget", count("outcome", "budget")),
                 ("impossible (geometry)", count("label", "impossible")), ("planner error", count("outcome", "planner_error")),
                 ("repeated actions refused", refused), ("bad task proposals dropped before proof", dropped),
                 ("loops (back to an earlier state)", count("outcome", "loop"))]

    cards = []
    for r in reversed(rows):
        room = run / "rooms" / r["room"]
        photo = next(iter(sorted(room.glob("photo_*.jpg"))), None)
        det = next(iter(sorted(room.glob("detections_*.png"))), None)
        imgs = "".join(f'<img src="{_esc(p.relative_to(run))}" alt="{alt}" loading="lazy">'
                       for p, alt in ((photo, "photo"), (det, "objects found")) if p)
        plan = room / "floor_plan.png"
        items = []
        for k, t in enumerate(r.get("tasks", []), 1):
            icon, words = LABELS.get(t["label"], ("·", t["label"]))
            f = _task_file(run, r["room"], k, t)
            link = f' <a href="{_esc(f)}">transcript</a>' if f else ""
            vid = run / "rooms" / r["room"] / "tasks" / f"{k}.mp4"
            if vid.exists():
                link += f' <a href="{_esc(vid.relative_to(run))}">video</a>'
            items.append(f'<li><span class="chip {t["label"]}">{icon} {_esc(words)}</span> {_esc(t["title"])}'
                         f'{" <span class=muted>(" + _esc(t["outcome"]) + ")</span>" if t["label"] == "unresolved" else ""}{link}</li>')
        status = ("build failed: " + _esc(r.get("error", ""))[:120]) if not r.get("built") else (
            f'{r.get("objects")} objects · room {"×".join(str(v) for v in (r.get("room_m") or [])[:2])} m · '
            f'{"settles in physics" if r.get("settles") else "did not settle: " + _esc(", ".join(r.get("unstable", [])))}')
        cards.append(f'<div class="card room"><h3>{_esc(r["room"])} <span class="muted">{_esc(r.get("type", ""))} · '
                     f'${r.get("usd", 0):.4f} · {r.get("seconds", 0) / 60:.1f} min</span></h3><div class="imgs">{imgs}</div>'
                     + (f'<img src="{_esc(plan.relative_to(run))}" alt="floor plan" loading="lazy">' if plan.exists() else "")
                     + f'<div class="muted">{status}</div><ul>{"".join(items)}</ul></div>')

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">{'<meta http-equiv="refresh" content="60">' if refresh and not finished else ''}
<title>Night Shift Report</title><style>{CSS}</style></head><body><main>
<h1>Night Shift: {_esc(run.name)}</h1>
<p class="sub">{'Running' if not finished else 'Finished'} · model {_esc(model)} · {len(rows)} of {total} rooms ·
{_esc(stopped) if stopped else ('all rooms done' if finished else 'refreshes every minute')}</p>
<div class="banner">Everything on this page happened in <b>simulation</b>: each room was rebuilt from one ordinary photo
(sizes estimated, masses catalog values). "Solvable" means solved in the simulated room - a plan to check, not a fact
about the real room. Every number below comes from the run's own log files.</div>
<div class="kpis">{''.join(f'<div class="tile"><div class="v">{_esc(v)}</div><div class="k">{_esc(k)}</div></div>' for v, k in kpis)}</div>
<h2>Working on its own</h2><div class="charts">{_chart("Rooms finished", cum_rooms, unit + " since start")}
{_chart("Claude spend (USD)", cum_usd, unit + " since start", "{:.3f}")}</div>
<h2>It knows when to stop</h2><div class="card"><table><tr><th>how a task or proposal was stopped</th><th class="num">count</th></tr>
{''.join(f'<tr><td>{_esc(k)}</td><td class="num">{v}</td></tr>' for k, v in stop_rows)}</table>
<p class="muted">Loops cannot be committed: an action is only kept if it measurably improves the goal, and a repeat in the
same state is refused before it runs (the simulation is deterministic).</p></div>
{('<div class="card" style="margin-top:12px"><h3>One real example of each</h3><table><tr><th>stop</th><th>task</th><th>what it said</th><th></th></tr>' + ''.join(examples) + '</table></div>') if examples else ''}
<h2>Asked for help ({len(helps)})</h2><div class="card"><table><tr><th>room</th><th>what</th><th>why</th></tr>
{''.join(f"<tr><td>{_esc(h['room'])}</td><td>{_esc(h['what'])}</td><td>{_esc(_cut(h['why']))}</td></tr>" for h in reversed(helps[-60:]))}</table></div>
<h2>Rooms ({len(rows)}, newest first)</h2><div class="rooms">{''.join(cards)}</div>
<p class="muted" style="margin-top:30px">Rooms built: {len(built)}; settle in physics: {sum(bool(r.get('settles')) for r in built)}.
Tries are run on a snapshot and undone (checked bit for bit); only actions that measurably help are kept.</p>
</main><script>{CHART_JS}</script></body></html>"""
    out = run / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    return out

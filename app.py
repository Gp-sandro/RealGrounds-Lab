"""RealGrounds Lab: the whole project in a browser - try it on a photo, experiment with the stop rules, watch the
unattended runs, run the tests and see every measured result.

USAGE (from the realgrounds-voice/ folder, venv active)
    python app.py                    # http://127.0.0.1:8000 on this Mac
    python app.py --share            # also reachable from phones on the same Wi-Fi (upload straight from the camera)
    python app.py --cap 2            # the app's money cap for Claude (default $1): one more supervisor

No extra packages: Python's own web server. Jobs that use the GPU or physics run one at a time (a queue), so
several people can use it at once. Everything happens in simulation; every page says so.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from realgrounds.config import ROOT

OUT = ROOT / "outputs"
APP = OUT / "app"
WEB = ROOT / "web"
NIGHT = OUT / "night"
MAX_UPLOAD = 25 * 1024 * 1024
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic"}
MODELS = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")
SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")


# ------------------------------------------------------------------------------------------------ jobs
@dataclass
class Job:
    kind: str
    title: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "queued"                         # queued / running / done / failed
    lines: list[str] = field(default_factory=list)
    result: Any = None
    error: str = ""
    created: float = field(default_factory=time.time)
    finished: float = 0.0

    def say(self, s: str) -> None:
        self.lines.append(s)

    def view(self, since: int = 0) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status, "error": self.error,
                "lines": self.lines[since:], "n_lines": len(self.lines), "result": self.result,
                "seconds": round((self.finished or time.time()) - self.created, 1)}


class Lab:
    """The app's state: jobs, the queue, the money cap, the loaded networks."""

    def __init__(self, cap_usd: float) -> None:
        self.cap_usd, self.spent_usd = cap_usd, 0.0
        self.jobs: dict[str, Job] = {}
        self.queue: Queue[tuple[Job, Callable[[Job], Any]]] = Queue()
        self.lock = threading.Lock()
        self.builder = None
        threading.Thread(target=self._worker, daemon=True).start()

    # -- jobs
    def submit(self, kind: str, title: str, fn: Callable[[Job], Any], queued: bool = True) -> Job:
        job = Job(kind, title)
        self.jobs[job.id] = job
        if queued:
            ahead = self.queue.qsize()
            if ahead:
                job.say(f"waiting: {ahead} job(s) ahead (physics and the GPU run one job at a time)")
            self.queue.put((job, fn))
        else:
            threading.Thread(target=self._run, args=(job, fn), daemon=True).start()
        return job

    def _worker(self) -> None:
        while True:
            job, fn = self.queue.get()
            self._run(job, fn)

    def _run(self, job: Job, fn: Callable[[Job], Any]) -> None:
        job.status = "running"
        try:
            job.result = fn(job)
            job.status = "done"
        except Exception as e:                                         # noqa: BLE001 - a job fails, the app lives
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            job.say(traceback.format_exc(limit=3))
        job.finished = time.time()

    # -- money: the app's own supervisor
    def planner(self, model: str):
        from realgrounds.claude_planner import ClaudePlanner

        if self.spent_usd >= self.cap_usd:
            raise RuntimeError(f"the app's Claude budget is spent (${self.spent_usd:.4f} of ${self.cap_usd:.2f}); "
                               f"restart with --cap to raise it")
        return ClaudePlanner(model=model if model in MODELS else MODELS[0])

    def charge(self, planner) -> None:
        with self.lock:
            self.spent_usd += getattr(planner, "spent_usd", 0.0)


# ------------------------------------------------------------------------------------------------ rooms
def test_rooms() -> dict[str, Path]:
    from realgrounds.agent.scenes import obstacle_room

    out = {}
    for v in ("light", "heavy", "wall"):
        d = APP / "scenes" / v
        if not (d / "route.json").exists():
            obstacle_room(d, v)
        out[f"test/{v}"] = d
    return out


def list_rooms() -> list[dict[str, Any]]:
    rooms = [{"id": k, "group": "Test rooms (hand-made, to see the stop rules)", "name": {
        "test/light": "light: a box and a chair block the way", "test/heavy": "heavy: a 40 kg box blocks the way (test value)",
        "test/wall": "wall: fixed furniture blocks the way"}[k]} for k in ("test/light", "test/heavy", "test/wall")]
    for d in sorted((APP / "rooms").glob("*/summary.json"), key=lambda p: -p.stat().st_mtime):
        rooms.append({"id": f"upload/{d.parent.name}", "group": "Your photos", "name": d.parent.name})
    for run in sorted(NIGHT.glob("*/results.jsonl"), key=lambda p: -p.stat().st_mtime):
        for line in run.read_text().splitlines():
            r = json.loads(line)
            if r.get("built") and r.get("body_stands"):
                rooms.append({"id": f"night/{run.parent.name}/{r['room']}", "group": f"Real rooms from run {run.parent.name}",
                              "name": f"{r['room']} ({r.get('type', '')}, {r.get('objects')} objects)"})
    return rooms


def room_dir(room_id: str) -> Path:
    parts = room_id.split("/")
    if not parts or not all(SAFE.match(p) for p in parts):
        raise ValueError("bad room id")
    if parts[0] == "test" and len(parts) == 2:
        d = test_rooms()[room_id]
    elif parts[0] == "upload" and len(parts) == 2:
        d = APP / "rooms" / parts[1]
    elif parts[0] == "night" and len(parts) == 3:
        d = NIGHT / parts[1] / "rooms" / parts[2]
    else:
        raise ValueError("bad room id")
    if not (d / "objects.json").exists():
        raise ValueError("no such room")
    return d


def rel(p: Path) -> str:
    return "/files/" + str(p.relative_to(OUT))


def room_info(room_id: str) -> dict[str, Any]:
    from realgrounds.sandbox import SimWorld

    d = room_dir(room_id)
    env = SimWorld(d)
    obs = env.observe()
    summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    w = env.world
    surfaces = sorted(t.label for t in w.things.values() if t.is_surface)
    return {
        "id": room_id, "provenance": obs["provenance"], "notes": obs["notes"], "source": room_source(room_id),
        "photo": next((rel(p) for p in sorted(d.glob("photo_*.jpg"))), None),
        "detections": next((rel(p) for p in sorted(d.glob("detections_*.png"))), None),
        "floor_plan": rel(d / "floor_plan.png") if (d / "floor_plan.png").exists() else None,
        "room_m": summary.get("room_m"), "physics": {k: summary.get("physics", {}).get(k) for k in
                                                     ("settle_ok", "agent_standing", "start_overlaps_left")},
        "objects": obs["objects"], "surfaces": surfaces,
        "movable": sorted(o["label"] for o in obs["objects"] if o["movable_in_simulation"]),
    }


# ------------------------------------------------------------------------------------------------ actions
def do_build(lab: Lab, photo: Path, room_id: str) -> Callable[[Job], Any]:
    def run(job: Job) -> Any:
        import night_shift

        if lab.builder is None:
            job.say("loading the networks (once): depth + lens, floor/walls, objects, outlines ...")
            lab.builder = night_shift.Builder()
        job.say("measuring the room from the photo (about a minute the first time) ...")
        t0 = time.time()
        lab.builder(photo, room_id, APP)
        s = json.loads((APP / "rooms" / room_id / "summary.json").read_text())
        ph = s.get("physics", {})
        job.say(f"built in {time.time() - t0:.0f} s: {s.get('objects')} objects, room {s.get('room_m')} m; physics: "
                f"{'settles' if ph.get('settle_ok') else 'some objects moved when it settled'}; body "
                f"{'stands' if ph.get('agent_standing') else 'cannot stand'}")
        return {"room": f"upload/{room_id}"}
    return run


def do_suggest(lab: Lab, room_id: str, n: int, model: str) -> Callable[[Job], Any]:
    def run(job: Job) -> Any:
        from realgrounds.sandbox import SimWorld
        from realgrounds.tasks import claude_tasks, far_side_route

        env = SimWorld(room_dir(room_id))
        p = lab.planner(model)
        job.say(f"asking {p.model} to write {n} tasks for this room ...")
        try:
            tasks, dropped = claude_tasks(env, p, n)
        finally:
            lab.charge(p)
        route = far_side_route(env)
        out = ([{"kind": "route", "object": "", "target": "", "title": route.title, "why": "automatic"}] if route else [])
        out += [{"kind": t.kind, "object": t.spec.get("object", ""), "target": t.spec.get("target", ""),
                 "title": t.title, "why": t.why} for t in tasks]
        for d in dropped:
            job.say(f"dropped before any proof: {d}")
        job.say(f"${p.spent_usd:.4f}")
        return {"tasks": out, "dropped": dropped, "usd": round(p.spent_usd, 5)}
    return run


def do_run(lab: Lab, room_id: str, spec: dict[str, Any], settings: dict[str, Any]) -> Callable[[Job], Any]:
    def run(job: Job) -> Any:
        from realgrounds.autonomy import rule_based_planner
        from realgrounds.sandbox import SimWorld
        from realgrounds.supervisor import Limits
        from realgrounds.tasks import make_task, prove

        d = room_dir(room_id)
        env0 = SimWorld(d)
        task, why_not = make_task(env0, spec.get("kind", ""), spec.get("object", ""), spec.get("target", ""))
        if task is None:
            job.say(f"not a task for this room: {why_not}")
            return {"label": "rejected", "reason": why_not}
        use_claude = settings.get("planner", "claude") == "claude"
        planner = lab.planner(settings.get("model", MODELS[0])) if use_claude else rule_based_planner
        limits = Limits(max_tries=int(min(max(settings.get("max_tries", 12), 1), 40)),
                        max_usd=float(min(max(settings.get("task_usd", 0.05), 0.0), 0.5)))
        stamp = datetime.now().strftime("%H%M%S")
        folder = APP / "runs" / f"{stamp}-{job.id}"
        folder.mkdir(parents=True, exist_ok=True)
        video = folder / "video.mp4" if settings.get("video", True) else None
        job.say(f"task: {task.title}  ·  planner: {getattr(planner, 'model', 'fixed rules')}  ·  budget: "
                f"{limits.max_tries} tries, ${limits.max_usd}")
        try:
            label, r = prove(d, task, planner, limits, say=job.say, video=video)
        finally:
            if use_claude:
                lab.charge(planner)
        (folder / "transcript.txt").write_text(f"{task.title}\n{r.verdict()}\n\n" + "\n".join(job.lines) + "\n")
        result = {"label": label, "outcome": r.outcome, "verdict": r.verdict(), "reason": r.reason, "help": r.help_request,
                  "tries": r.tries, "commits": r.commits, "committed": r.committed, "usd": round(r.spent_usd, 5),
                  "refused_repeats": sum(e["event"] == "refused" for e in r.log), "sim_seconds": r.sim_seconds,
                  # a video only if physics ran (a task stopped before any try has stills, no frames)
                  "video": rel(video) if video and video.exists() and video.stat().st_size > 20_000 else None,
                  "stills": rel(video.with_suffix(".png")) if video and video.with_suffix(".png").exists() else None,
                  "transcript": rel(folder / "transcript.txt"), "steps": steps(r.log)}
        record = {**result, "room": room_id, "room_source": room_source(room_id),
                  "task": {**spec, "title": task.title, "goal": task.goal.describe(),
                           "start": brief(env0.check(task.goal))},
                  "planner": getattr(planner, "model", "fixed rules"),
                  "limits": {"max_tries": limits.max_tries, "max_usd": limits.max_usd},
                  "seconds": round(time.time() - job.created, 1), "time": datetime.now().isoformat(timespec="seconds")}
        (folder / "result.json").write_text(json.dumps(record, indent=1, default=str))
        return record                      # the result plus the task, goal and planner (a superset)
    return run


def brief(g: dict[str, Any] | None) -> str:
    if not g:
        return "-"
    if g.get("satisfied"):
        return "satisfied"
    return g.get("why") or "not satisfied"


def steps(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The supervisor's log as a timeline: plan / try / refused / commit."""
    out = []
    for e in log:
        if e["event"] == "plan":
            out.append({"type": "plan", "usd": e.get("usd", 0), "proposals": e.get("proposals", []),
                        "reasoning": e.get("reasoning", "")})
        elif e["event"] == "try":
            out.append({"type": "try", "action": e["action"], "status": e["status"], "result": e.get("result", ""),
                        "moved_cm": e.get("moved_cm") or {}, "goal_after": brief(e.get("goal_after")),
                        "sim_s": e.get("sim_s")})
        elif e["event"] == "refused":
            out.append({"type": "refused", "action": e["action"], "why": e["why"]})
        elif e["event"] == "commit":
            out.append({"type": "commit", "action": e["action"], "status": e["status"],
                        "goal_after": brief(e.get("goal_after")), "matches_try": e.get("matches_try")})
    return out


def room_source(room_id: str) -> str:
    return ("hand-made test room (sizes set by hand)" if room_id.startswith("test/")
            else "rebuilt from one photo (sizes estimated)")


def run_records(room: str | None = None) -> list[dict[str, Any]]:
    """Recorded task runs (newest first), each with its timeline - what the experiment pages replay."""
    out = []
    for f in sorted((APP / "runs").glob("*/result.json"), key=lambda p: -p.stat().st_mtime):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if room is None or r.get("room") == room:
            out.append(r)
    return out


def experiments() -> dict[str, Any]:
    return {v: {"room": f"test/{v}", "records": [r for r in run_records(f"test/{v}")
                                                 if r.get("task", {}).get("kind") == "route"][:40]}
            for v in ("light", "heavy", "wall")}


def featured() -> dict[str, Any] | None:
    """A photo-built room with a re-checked solution on video: the photo-to-physics example."""
    for video in sorted(NIGHT.glob("*/rooms/*/tasks/*_replay.mp4"), key=lambda p: (p.parent.parent.name != "nyu_0588", -p.stat().st_mtime)):
        room = video.parent.parent
        run = room.parent.parent
        k = int(video.name.split("_")[0])
        rows = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines()]
        rec = next((r for r in rows if r["room"] == room.name), None)
        if not rec or len(rec.get("tasks", [])) < k:
            continue
        t = rec["tasks"][k - 1]
        return {"room": f"night/{run.name}/{room.name}", "run": run.name, "type": rec.get("type"),
                "objects": rec.get("objects"), "room_m": rec.get("room_m"), "task": t["title"], "kept": t.get("committed"),
                "photo": next((rel(p) for p in sorted(room.glob("photo_*.jpg"))), None),
                "detections": next((rel(p) for p in sorted(room.glob("detections_*.png"))), None),
                "floor_plan": rel(room / "floor_plan.png") if (room / "floor_plan.png").exists() else None,
                "video": rel(video), "stills": rel(video.with_suffix(".png")) if video.with_suffix(".png").exists() else None}
    return None


def do_verify(run_name: str) -> Callable[[Job], Any]:
    """Replay every solved task's kept actions on a fresh copy of its room and check the goal again."""
    def run(job: Job) -> Any:
        from realgrounds.goals import ObjectRestsOn, PathClear, ReachObject
        from realgrounds.sandbox import SimWorld

        folder = NIGHT / run_name
        rows = [json.loads(l) for l in (folder / "results.jsonl").read_text().splitlines()]
        held, total, fails = 0, 0, []
        for r in rows:
            for i, t in enumerate(r.get("tasks", []), 1):
                if t.get("label") != "solvable" or not t.get("committed"):
                    continue
                d = folder / "rooms" / r["room"]
                tr = next(iter(sorted((d / "tasks").glob(f"{i}_*.txt"))), None)
                kind = t.get("kind") or (tr.stem.split("_", 1)[1] if tr else "route")
                env = SimWorld(d)
                spec = t["spec"]
                goal = (PathClear(tuple(spec["start"]), tuple(spec["target"]), 0.3) if kind == "route" else
                        ObjectRestsOn(spec["object"], spec["target"]) if kind == "rests_on" else
                        ReachObject(spec["object"], tuple(float(v) for v in env.world.base()[0][:2])))
                for a in t["committed"]:
                    env.commit_action(a, goal)
                ok = bool(env.check(goal).get("satisfied"))
                total += 1
                held += ok
                job.say(f"{'✓' if ok else '✗'} {r['room']}: {t['title']}")
                if not ok:
                    fails.append(f"{r['room']}: {t['title']}")
        res = {"run": run_name, "replayed": total, "held": held, "failed": fails,
               "time": datetime.now().isoformat(timespec="seconds")}
        (APP / f"verify_{run_name}.json").write_text(json.dumps(res, indent=1))
        job.say(f"{held} of {total} solutions hold again on a fresh copy")
        return res
    return run


def do_subprocess(cmd: list[str], parse: Callable[[list[str]], Any] | None = None) -> Callable[[Job], Any]:
    def run(job: Job) -> Any:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        out = []
        for line in p.stdout:
            line = line.rstrip()
            out.append(line)
            if line and "Warning" not in line and "autocast" not in line and "Loading weights" not in line:
                job.say(line[:300])
        p.wait()
        res = parse(out) if parse else None
        if p.returncode and not parse:
            raise RuntimeError(f"exit code {p.returncode}")
        return res
    return run


SECTIONS = {
    "test_room": "Room understanding (photos -> measured room)",
    "test_agent": "The simulated body follows commands, checked by physics",
    "test_autonomy": "Sandbox, goals, supervisor and the autonomous loop",
    "test_night": "Claude planner, tasks and the night's supervisor (fake API)",
    "test_app": "The web app: API, recorded runs, a free task run, money cap, file safety",
}


def parse_tests(lines: list[str]) -> dict[str, Any]:
    rows, pending = [], None
    for l in lines:
        m = re.match(r"^(test_\w+) \(tests\.(\w+)\.\w+\.\w+\)(?: \.\.\.)?\s*(ok|FAIL|ERROR|skipped)?", l.strip())
        if m:
            pending = {"name": m.group(1)[5:].replace("_", " "), "section": SECTIONS.get(m.group(2), m.group(2)),
                       "status": m.group(3) or "?"}
            rows.append(pending)
        elif pending is not None and pending["status"] == "?" and l.strip() in ("ok", "FAIL", "ERROR"):
            pending["status"] = l.strip()          # a warning printed between the name and the result
    summary = next((l for l in reversed(lines) if l.startswith("Ran ")), "")
    res = {"tests": rows, "passed": sum(r["status"] == "ok" for r in rows), "total": len(rows), "summary": summary,
           "ok": any(l.strip() == "OK" or l.startswith("OK ") for l in lines),
           "time": datetime.now().isoformat(timespec="seconds")}
    APP.mkdir(parents=True, exist_ok=True)
    (APP / "tests.json").write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------------------------------------ results
BENCHMARK = {
    "source": "40 real rooms the system had never seen (NYU Depth V2 / SUN RGB-D test split, lens unknown); measured "
              "in the research repo against human-labelled ground truth",
    "rows": [["depth error (AbsRel, lower is better)", "0.062"], ["rooms more than 15 % off in size", "0 %"],
             ["3D box centre error", "0.31 m"], ["objects found and named right", "50 %"],
             ["'rests on the floor' right", "92 %"], ["'rests on another object' right", "42 %"],
             ["'hangs on a wall' right", "62 %"], ["time to build a room from one photo", "~1 min (M5 MacBook)"]],
}
COMMANDS = {
    "source": "plain-English commands run by the simulated body, every result checked in physics (tests/test_agent.py "
              "and the task files in realgrounds/agent/)",
    "rows": [["real dorm room from photos, zone 1", "13 / 14 ended as expected"],
             ["real dorm room from photos, zone 2", "9 / 10 ended as expected"],
             ["hand-made test room", "13 / 13 ended as expected"]],
}


def run_stats(folder: Path) -> dict[str, Any]:
    rows = [json.loads(l) for l in (folder / "results.jsonl").read_text().splitlines()] if (folder / "results.jsonl").exists() else []
    ev = [json.loads(l) for l in (folder / "timeline.jsonl").read_text().splitlines()] if (folder / "timeline.jsonl").exists() else []
    tasks = [t for r in rows for t in r.get("tasks", [])]
    lab = lambda v: sum(t.get("label") == v for t in tasks)          # noqa: E731
    start = next((e for e in ev if e["event"] == "night_start"), {})
    stopped = next((e["reason"] for e in reversed(ev) if e["event"] == "night_stopped"), "")
    verify = APP / f"verify_{folder.name}.json"
    return {
        "run": folder.name, "model": start.get("model"), "rooms_planned": start.get("rooms"), "rooms": len(rows),
        "built": sum(bool(r.get("built")) for r in rows), "settle": sum(bool(r.get("settles")) for r in rows),
        "hours": round(ev[-1]["elapsed_s"] / 3600, 2) if ev else 0, "finished": bool(stopped) or any(e["event"] == "night_end" for e in ev),
        "stopped": stopped, "solvable": lab("solvable"), "impossible": lab("impossible"), "unresolved": lab("unresolved"),
        "already_true": lab("already_true"), "skipped": lab("skipped"),
        "stuck": sum(t.get("outcome") == "stuck" for t in tasks), "budget": sum(t.get("outcome") == "budget" for t in tasks),
        "refused_repeats": sum(t.get("refused_repeats", 0) for t in tasks),
        "dropped": sum(len(r.get("dropped", [])) for r in rows),
        "help": sum(e["event"] == "help_request" for e in ev), "usd": round(sum(r.get("usd", 0) for r in rows), 4),
        "dashboard": rel(folder / "dashboard.html") if (folder / "dashboard.html").exists() else None,
        "verify": json.loads(verify.read_text()) if verify.exists() else None,
    }


def busy_runs() -> list[str]:
    """Unattended runs working right now (their timeline changed in the last 3 minutes and they haven't ended):
    they share the GPU, so a photo takes longer to build meanwhile."""
    out = []
    for t in NIGHT.glob("*/timeline.jsonl"):
        if time.time() - t.stat().st_mtime < 180:
            tail = t.read_text().splitlines()[-1:]
            if tail and json.loads(tail[0])["event"] not in ("night_end", "night_stopped"):
                out.append(t.parent.name)
    return out


def results() -> dict[str, Any]:
    tests = json.loads((APP / "tests.json").read_text()) if (APP / "tests.json").exists() else None
    runs = [run_stats(p.parent) for p in sorted(NIGHT.glob("*/results.jsonl"), key=lambda p: -p.stat().st_mtime)]
    keep = ("room", "room_source", "task", "planner", "label", "outcome", "tries", "commits", "usd", "sim_seconds", "time")
    return {"benchmark": BENCHMARK, "commands": COMMANDS, "tests": tests, "runs": runs, "featured": featured(),
            "app_runs": [{k: r.get(k) for k in keep} for r in run_records()][:40]}


# ------------------------------------------------------------------------------------------------ http
def make_handler(lab: Lab):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RealGroundsLab/1"

        def log_message(self, fmt: str, *args: Any) -> None:       # quiet
            pass

        def _json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path, base: Path) -> None:
            try:
                path = path.resolve()
                path.relative_to(base.resolve())
            except ValueError:
                return self._json({"error": "not found"}, 404)
            if not path.is_file():
                return self._json({"error": "not found"}, 404)
            size = path.stat().st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix == ".txt":
                ctype = "text/plain; charset=utf-8"
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
            if m and (m.group(1) or m.group(2)):                      # video seeking (Safari / iPhone need it)
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(size - int(m.group(2)), 0)
                end = min(end, size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self) -> None:
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self._file(WEB / "index.html", WEB)
                if u.path.startswith("/web/"):
                    return self._file(WEB / unquote(u.path[5:]), WEB)
                if u.path.startswith("/files/"):
                    return self._file(OUT / unquote(u.path[7:]), OUT)
                if u.path == "/api/overview":
                    recent = sorted(lab.jobs.values(), key=lambda j: -j.created)[:15]
                    return self._json({"spent_usd": round(lab.spent_usd, 5), "cap_usd": lab.cap_usd, "models": MODELS,
                                       "rooms": list_rooms(), "jobs": [j.view(len(j.lines)) for j in recent],
                                       "queue": lab.queue.qsize(), "busy_runs": busy_runs()})
                if u.path == "/api/room":
                    return self._json(room_info(q.get("id", "")))
                if u.path == "/api/job":
                    job = lab.jobs.get(q.get("id", ""))
                    return self._json(job.view(int(q.get("since", 0))) if job else {"error": "no such job"}, 200 if job else 404)
                if u.path == "/api/results":
                    return self._json(results())
                if u.path == "/api/experiments":
                    return self._json(experiments())
                return self._json({"error": "not found"}, 404)
            except ValueError as e:
                return self._json({"error": str(e)}, 400)

        def do_POST(self) -> None:
            u = urlparse(self.path)
            try:
                if u.path == "/api/upload":
                    n = int(self.headers.get("Content-Length") or 0)
                    name = unquote(self.headers.get("X-Filename") or "photo.jpg")
                    ext = Path(name).suffix.lower()
                    if ext not in PHOTO_EXT:
                        return self._json({"error": f"send a photo ({', '.join(sorted(PHOTO_EXT))})"}, 400)
                    if not 0 < n <= MAX_UPLOAD:
                        return self._json({"error": "photo missing or larger than 25 MB"}, 400)
                    room_id = "photo_" + datetime.now().strftime("%m%d_%H%M%S")
                    folder = APP / "uploads" / room_id
                    folder.mkdir(parents=True, exist_ok=True)
                    photo = folder / f"{room_id}{ext}"
                    photo.write_bytes(self.rfile.read(n))
                    if ext == ".heic":
                        from build_room import convert_heic
                        convert_heic(folder)
                        photo = photo.with_suffix(".jpg")
                        if not photo.exists():
                            return self._json({"error": "could not convert the HEIC photo; send a JPEG"}, 400)
                    job = lab.submit("build", f"build a room from {name}", do_build(lab, photo, room_id))
                    return self._json({"job": job.id})
                b = self._body()
                if u.path == "/api/suggest":
                    room_dir(b.get("room", ""))
                    job = lab.submit("suggest", "Claude writes tasks", do_suggest(lab, b["room"], int(min(max(b.get("n", 3), 1), 5)),
                                                                           b.get("model", MODELS[0])))
                    return self._json({"job": job.id})
                if u.path == "/api/run":
                    room_dir(b.get("room", ""))
                    job = lab.submit("run", f"task in {b['room']}", do_run(lab, b["room"], b.get("task", {}), b.get("settings", {})))
                    return self._json({"job": job.id})
                if u.path == "/api/verify":
                    run = b.get("run", "")
                    if not SAFE.match(run) or not (NIGHT / run / "results.jsonl").exists():
                        return self._json({"error": "no such run"}, 400)
                    job = lab.submit("verify", f"re-check every solution of {run}", do_verify(run))
                    return self._json({"job": job.id})
                if u.path == "/api/tests":
                    job = lab.submit("tests", "run the automated tests",
                                     do_subprocess([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
                                                   parse_tests), queued=False)
                    return self._json({"job": job.id})
                if u.path == "/api/night":
                    n = int(min(max(b.get("rooms", 3), 1), 20))
                    left = max(lab.cap_usd - lab.spent_usd, 0.0)
                    if left < 0.05:
                        return self._json({"error": "the app's Claude budget is spent"}, 400)
                    name = "lab-" + datetime.now().strftime("%m%d-%H%M%S")
                    job = lab.submit("night", f"unattended run on {n} real rooms ({name})", do_subprocess(
                        [sys.executable, "-u", "night_shift.py", "--nyu", str(n), "--split", "dev", "--run", name,
                         "--budget-usd", f"{min(left, 0.5):.2f}", "--room-usd", "0.05"]), queued=False)
                    return self._json({"job": job.id, "run": name})
                return self._json({"error": "not found"}, 404)
            except (ValueError, KeyError) as e:
                return self._json({"error": str(e)}, 400)

    return Handler


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RealGrounds Lab in a browser.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--share", action="store_true", help="listen on the local network too (phones on the same Wi-Fi)")
    ap.add_argument("--cap", type=float, default=1.0, help="the app's money cap for Claude, in USD")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args(argv)
    for d in (APP, NIGHT):
        d.mkdir(parents=True, exist_ok=True)
    test_rooms()
    lab = Lab(args.cap)
    host = "0.0.0.0" if args.share else "127.0.0.1"
    server = ThreadingHTTPServer((host, args.port), make_handler(lab))
    print(f"RealGrounds Lab: http://127.0.0.1:{args.port}")
    if args.share:
        print(f"  on this Wi-Fi:  http://{lan_ip()}:{args.port}   (anyone on the network can use it - and spend up to ${args.cap})")
    print(f"  Claude budget for this session: ${args.cap:.2f}   (Ctrl+C to stop)")
    if not args.no_open:
        import webbrowser
        threading.Timer(0.8, webbrowser.open, [f"http://127.0.0.1:{args.port}"]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

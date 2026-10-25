"""RealGrounds Night Shift: photos of real rooms in, simulation rooms with proven tasks out - unattended.

USAGE (from the realgrounds-voice/ folder)
    python night_shift.py --nyu 10 --run test10                 # 10 real rooms from the NYU dev split
    python night_shift.py --photos inputs/my_rooms --run home   # every photo in the folder is one room
    python night_shift.py --photos ~/Downloads/IMG_1234.HEIC --run live --video   # one photo, with task videos
    python night_shift.py --nyu 200 --split train+dev --types all --run long      # the long unattended run
    (run it again with the same --run to resume: finished rooms are skipped)

FOR EVERY ROOM
    build    photo -> measured room -> physics (the lens is estimated from the photo, as for any photo)
    inspect  does the room settle, can the body stand in it
    tasks    one automatic route across the room + tasks Claude writes for this room
    prove    the autonomous loop solves each task in a fresh simulated copy: try (undone, checked) -> commit
             what measurably helps -> stop when solved, impossible, stuck, looping or out of budget
Two supervisors watch: one per task (realgrounds/supervisor.py), one for the night (realgrounds/run_supervisor.py).

OUTPUT  outputs/night/<run>/
    report.md        the night: what was built, proven, flagged, spent, and every self-stop
    timeline.jsonl   every event with its time
    results.jsonl    one line per room (also the checkpoint)
    help_queue.md    what a person should look at, and why
    dashboard.html   the night as one page (refreshes while it runs)
    rooms/<id>/      the room (objects.json, room_mujoco.xml, floor_plan.png...) + tasks/<n>.txt transcripts

Everything is SIMULATED: rooms are estimated from photos, masses are catalog values. "Solvable" means solved in
the simulated room - a plan to check, not a fact about the real room.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

from realgrounds.claude_planner import DEFAULT_MODEL, ClaudePlanner, PlannerError
from realgrounds import dashboard
from realgrounds.config import ROOT
from realgrounds.run_supervisor import NightLimits, NightSupervisor
from realgrounds.sandbox import SimWorld
from realgrounds.supervisor import Limits
from realgrounds.tasks import claude_tasks, far_side_route, prove

OUT = ROOT / "outputs" / "night"
NYU = ROOT.parent / "pixel" / "datasets" / "rooms_nyu"
ROOM_TYPES = ("bedroom", "living_room", "office", "dining_room")


def nyu_rooms(n: int, splits: str = "dev", types: str = "homes") -> list[tuple[str, Path, str]]:
    """One photo per distinct room. splits: "dev", "train", "train+dev" (the test split is kept for scoring the
    pipeline); types: "homes" (bedrooms, living rooms, offices, dining rooms) or "all" (kitchens, bathrooms,
    classrooms, shops... too)."""
    rows = json.loads((NYU / "index.json").read_text())
    use = set(splits.split("+"))
    seen, out = set(), []
    for r in rows:
        if r["split"] in use and (types == "all" or r["scene_type"] in ROOM_TYPES) and r["floor_seen"] \
                and r["scene"] not in seen:
            seen.add(r["scene"])
            out.append((r["id"], NYU / r["split"] / r["id"] / "photo.jpg", r["scene_type"]))
    return out[:n]


def folder_rooms(folder: Path) -> list[tuple[str, Path, str]]:
    """Every photo in a folder (or one photo) is one room. iPhone .HEIC photos are converted to .jpg first."""
    from build_room import convert_heic
    from realgrounds.images import list_images

    if folder.is_file():
        if folder.suffix.lower() == ".heic":
            convert_heic(folder.parent)
            folder = folder.with_suffix(".jpg")
        return [(folder.stem, folder, "photo")]
    convert_heic(folder)
    return [(p.stem, p, "photo") for p in list_images(folder)]


class Builder:
    """Loads the networks once, on the first room that needs building."""

    def __init__(self) -> None:
        self.models = None

    def __call__(self, photo: Path, room_id: str, out: Path) -> Path:
        import build_room

        stage = out / "photos" / room_id
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copy(photo, stage / f"{room_id}{photo.suffix.lower()}")    # only the photo: lens estimated
        if self.models is None:
            rc, catalog, _ = build_room.load_config()
            self.models = build_room.models_for(rc, catalog, cache=True)
        return build_room.build(stage, name=room_id, models=self.models, out_root=out / "rooms")


def do_room(room_id: str, photo: Path, kind: str, out: Path, builder: Builder, planner: ClaudePlanner,
            night: NightSupervisor, args) -> dict:
    t0, usd0 = time.time(), planner.spent_usd
    rec: dict = {"room": room_id, "type": kind, "tasks": [], "dropped": []}
    room_dir = out / "rooms" / room_id
    try:
        if not (room_dir / "summary.json").exists():
            builder(photo, room_id, out)
        summary = json.loads((room_dir / "summary.json").read_text())
    except Exception as e:                                              # a room that can't be built
        rec.update(built=False, error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t0, 1), usd=0.0)
        night.event("build_failed", room=room_id, error=rec["error"])
        night.ask_help(room_id, "room could not be built", rec["error"])
        (out / "errors.log").open("a").write(f"--- {room_id}\n{traceback.format_exc()}\n")
        return rec
    ph = summary.get("physics", {})
    unstable = sorted(k for k, v in (ph.get("settle") or {}).items() if not v.get("stayed"))
    rec.update(built=True, objects=summary.get("objects"), room_m=summary.get("room_m"),
               build_s=summary.get("seconds"), settles=ph.get("settle_ok"), unstable=unstable,
               body_stands=ph.get("agent_standing"))
    night.event("built", room=room_id, objects=rec["objects"], settles=rec["settles"], body_stands=rec["body_stands"])
    if unstable:
        night.ask_help(room_id, "objects moved when the room settled", ", ".join(unstable))
    if not (ph.get("ran") and ph.get("agent_standing")):
        night.ask_help(room_id, "room not usable", "the body cannot stand in the simulated room")
        rec.update(seconds=round(time.time() - t0, 1), usd=0.0)
        return rec

    env = SimWorld(room_dir)
    tasks = []
    route = far_side_route(env)
    if route:
        tasks.append(route)
    try:
        written, dropped = claude_tasks(env, planner, args.tasks_per_room)
        tasks += written
        rec["dropped"] = dropped
        night.event("tasks_written", room=room_id, tasks=[t.title for t in tasks], dropped=len(dropped))
    except PlannerError as e:
        rec["planner_error"] = str(e)
        night.event("planner_error", room=room_id, error=str(e))

    (room_dir / "tasks").mkdir(exist_ok=True)
    limits = Limits(max_tries=args.task_tries, max_commits=4, max_usd=args.task_usd)
    for k, task in enumerate(tasks, 1):
        room_spent = planner.spent_usd - usd0
        if room_spent >= night.limits.room_usd:
            rec["tasks"].append({"title": task.title, "source": task.source, "label": "skipped",
                                 "reason": f"room budget spent (${room_spent:.4f})"})
            night.ask_help(room_id, f"task not attempted: {task.title}", "this room spent its budget")
            continue
        lines: list[str] = []
        u0 = planner.spent_usd
        video = room_dir / "tasks" / f"{k}.mp4" if args.video else None
        label, r = prove(room_dir, task, planner, limits, say=lines.append, video=video)
        refused = sum(e["event"] == "refused" for e in r.log)
        t = {"title": task.title, "kind": task.kind, "source": task.source, "why": task.why, "spec": task.spec, "label": label,
             "outcome": r.outcome, "reason": r.reason, "help": r.help_request, "tries": r.tries,
             "commits": r.commits, "committed": r.committed, "refused_repeats": refused,
             "usd": round(planner.spent_usd - u0, 5), "sim_s": r.sim_seconds,
             "file": f"rooms/{room_id}/tasks/{k}_{task.kind}.txt"}
        rec["tasks"].append(t)
        if r.outcome == "planner_error":
            rec["planner_error"] = r.reason
        (room_dir / "tasks" / f"{k}_{task.kind}.txt").write_text(
            f"{task.title}  [{task.source}]  {task.why}\n{r.verdict()}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
        night.event("task_done", room=room_id, task=task.title, label=label, outcome=r.outcome, tries=r.tries,
                    commits=r.commits, refused_repeats=refused, usd=t["usd"])
        if label == "unresolved":
            night.ask_help(room_id, f"unresolved: {task.title}", r.help_request or r.reason)
    rec.update(seconds=round(time.time() - t0, 1), usd=round(planner.spent_usd - usd0, 5))
    return rec


def report(out: Path, rows: list[dict], night: NightSupervisor, model: str, total_rooms: int) -> None:
    tasks = [t for r in rows for t in r.get("tasks", [])]
    by = lambda key, v: sum(t.get(key) == v for t in tasks)          # noqa: E731
    built = [r for r in rows if r.get("built")]
    usd = sum(r.get("usd", 0) for r in rows)
    secs = sum(r.get("seconds", 0) for r in rows)
    solved = by("label", "solvable")
    md = [f"# Night Shift: {out.name}", "",
          f"Model: `{model}`. Rooms processed: {len(rows)} of {total_rooms}. "
          f"Stopped: {night.stopped or 'finished all rooms'}.", "",
          "> Everything here is SIMULATED: rooms were built from photos (sizes estimated, masses catalog values). "
          "\"Solvable\" means solved in the simulated room - a plan to check, not a fact about the real room.", "",
          "## The night in numbers", "",
          "| | |", "|---|---|",
          f"| rooms built / failed | {len(built)} / {len(rows) - len(built)} |",
          f"| rooms that settle in physics | {sum(bool(r.get('settles')) for r in built)} of {len(built)} |",
          f"| tasks: solvable / impossible / unresolved / already true / skipped | {solved} / {by('label', 'impossible')} / "
          f"{by('label', 'unresolved')} / {by('label', 'already_true')} / {by('label', 'skipped')} |",
          f"| tasks written by Claude / automatic | {by('source', 'claude')} / {by('source', 'auto')} |",
          f"| task proposals dropped before proof (bad object, fixed, already true) | {sum(len(r.get('dropped', [])) for r in rows)} |",
          f"| repeated actions refused by the supervisor | {sum(t.get('refused_repeats', 0) for t in tasks)} |",
          f"| tasks stopped: stuck / budget / loop / planner error | {by('outcome', 'stuck')} / {by('outcome', 'budget')} / "
          f"{by('outcome', 'loop')} / {by('outcome', 'planner_error')} |",
          f"| tries (sandboxed, undone) / commits | {sum(t.get('tries', 0) for t in tasks)} / {sum(t.get('commits', 0) for t in tasks)} |",
          f"| Claude spend | ${usd:.4f} (${usd / max(len(rows), 1):.4f} per room"
          + (f", ${usd / solved:.4f} per solvable task)" if solved else ")") + " |",
          f"| time | {secs / 60:.1f} min ({secs / 60 / max(len(rows), 1):.1f} min per room) |", "",
          "## Rooms", "", "| room | type | objects | settles | tasks (label: title) | $ | min |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("built"):
            md.append(f"| {r['room']} | {r['type']} | - | build failed | {r.get('error', '')[:80]} | 0 | {r.get('seconds', 0) / 60:.1f} |")
            continue
        ts = "<br>".join(f"**{t['label']}**: {t['title']}" + (f" ({t['outcome']})" if t['label'] == 'unresolved' else "")
                         for t in r.get("tasks", [])) or "-"
        md.append(f"| {r['room']} | {r['type']} | {r.get('objects')} | {'yes' if r.get('settles') else 'no'} | {ts} | "
                  f"{r.get('usd', 0):.4f} | {r.get('seconds', 0) / 60:.1f} |")
    md += ["", f"## Help queue ({len(night.help_queue)})", "", "See help_queue.md.", ""]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    hq = [f"# What a person should look at ({len(night.help_queue)})", ""]
    hq += [f"- **{h['room']}** - {h['what']}: {h['why']}" for h in night.help_queue]
    (out / "help_queue.md").write_text("\n".join(hq) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Photos of real rooms -> simulation rooms with proven tasks, unattended.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--nyu", type=int, help="this many real rooms from the NYU dev split")
    src.add_argument("--photos", help="a folder of photos (one room each)")
    ap.add_argument("--run", required=True, help="run name (same name = resume)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--budget-usd", type=float, default=2.0, help="the whole night")
    ap.add_argument("--room-usd", type=float, default=0.20)
    ap.add_argument("--task-usd", type=float, default=0.05)
    ap.add_argument("--task-tries", type=int, default=12)
    ap.add_argument("--tasks-per-room", type=int, default=2, help="tasks Claude writes (plus one automatic route)")
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--split", default="dev", help="NYU split(s): dev, train, train+dev")
    ap.add_argument("--types", default="homes", choices=("homes", "all"), help="NYU room types")
    ap.add_argument("--video", action="store_true", help="a video of every task's proof (slower)")
    args = ap.parse_args(argv)

    rooms = nyu_rooms(args.nyu, args.split, args.types) if args.nyu else folder_rooms(Path(args.photos).expanduser())
    out = OUT / args.run
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "results.jsonl"
    rows = [json.loads(l) for l in ckpt.read_text().splitlines()] if ckpt.exists() else []
    done = {r["room"] for r in rows}
    night = NightSupervisor(NightLimits(args.budget_usd, args.room_usd, args.max_hours), out)
    night.spent_usd = sum(r.get("usd", 0) for r in rows)
    planner = ClaudePlanner(model=args.model)
    builder = Builder()
    night.event("night_start", rooms=len(rooms), already_done=len(done), model=args.model, budget_usd=args.budget_usd)
    for room_id, photo, kind in rooms:
        if room_id in done:
            continue
        why = night.may_start_room()
        if why:
            night.stop(why)
            break
        night.event("room_start", room=room_id, type=kind)
        rec = do_room(room_id, photo, kind, out, builder, planner, night, args)
        rows.append(rec)
        with ckpt.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        night.room_finished(rec.get("usd", 0.0), rec.get("built", False), bool(rec.get("planner_error")))
        night.event("room_done", room=room_id, usd=rec.get("usd"), minutes=round(rec.get("seconds", 0) / 60, 1),
                    labels=[t["label"] for t in rec.get("tasks", [])])
        report(out, rows, night, args.model, len(rooms))
        dashboard.build(out, refresh=True)
    else:
        night.event("night_end", rooms=len(rows), usd=round(night.spent_usd, 4))
    report(out, rows, night, args.model, len(rooms))
    page = dashboard.build(out)
    print(f"\noutput: {out.relative_to(ROOT)}/  (dashboard.html, report.md, help_queue.md, timeline.jsonl)")
    print(f"open {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

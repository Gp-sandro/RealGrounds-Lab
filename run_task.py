"""Give the agent a goal; it works on its own in the SIMULATED room and stops when done, stuck or blocked.

USAGE (from the realgrounds-voice/ folder)
    python run_task.py --scene light            # test room: a box and a chair block the route -> cleared
    python run_task.py --scene heavy            # the box is set to 40 kg (a test value) -> stops, asks for help
    python run_task.py --scene wall             # fixed furniture blocks the route -> stops at once
    python run_task.py --scene light --planner claude   # Claude plans instead of the fixed rules (uses the API)
    python run_task.py --scene outputs/rooms/dorm/zone_1 --start 0.2,0.5 --target 1.0,2.5
    (add --video for a video of the tries and commits)

WHAT HAPPENS
    goal check (geometry + collisions)  ->  the planner proposes actions  ->  each is TRIED from a snapshot of
    the simulated room and undone (checked bit for bit)  ->  the supervisor commits the best one to the
    persistent SIMULATED room only if it measurably helps  ->  repeat until the supervisor stops it.

    Nothing here acts in the real room. A "simulated success" means the goal holds in a copy of the room whose
    sizes were estimated from photos and whose masses are catalog values - it is a plan to check, not a fact.

OUTPUT  outputs/tasks/<scene>/
    report.md        verdict, what was tried and committed, and which properties are estimates
    transcript.txt   the run, line by line
    log.json         every try and commit with its measurements
    stills.png       the simulated room at the start, after each commit, at the end
    video.mp4        (--video) every try (undone afterwards) and commit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from realgrounds.agent.run import Recorder
from realgrounds.agent.scenes import obstacle_room
from realgrounds.autonomy import rule_based_planner, run
from realgrounds.config import ROOT
from realgrounds.goals import PathClear
from realgrounds.sandbox import SimWorld
from realgrounds.supervisor import Limits
from realgrounds.tasks import label_actions

OUT = ROOT / "outputs" / "tasks"
TEST_SCENES = ("light", "heavy", "wall")


def _xy(s: str) -> tuple[float, float]:
    x, y = (float(v) for v in s.split(","))
    return x, y


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="An autonomous agent clears a walking route - in simulation.")
    ap.add_argument("--scene", default="light", help="light / heavy / wall (test rooms) or a room folder")
    ap.add_argument("--start", type=_xy, help="x,y in metres (default for test rooms: 0,0.3)")
    ap.add_argument("--target", type=_xy, help="x,y in metres (default for test rooms: 0,3.0)")
    ap.add_argument("--radius", type=float, default=0.3, help="how close to the target counts as arrived (m)")
    ap.add_argument("--max-tries", type=int, default=Limits.max_tries)
    ap.add_argument("--max-commits", type=int, default=Limits.max_commits)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--planner", choices=("rules", "claude"), default="rules",
                    help="rules (no language model) or claude (model from REALGROUNDS_MODEL, default Haiku)")
    args = ap.parse_args(argv)

    if args.scene in TEST_SCENES:
        name, room_dir = args.scene, obstacle_room(OUT / "scenes" / args.scene, args.scene)
        start, target = args.start or (0.0, 0.3), args.target or (0.0, 3.0)
        source = "a hand-made test room (sizes and positions set by hand; masses from the catalog or marked test values)"
    else:
        room_dir = Path(args.scene) if Path(args.scene).is_absolute() else ROOT / args.scene
        name = "_".join(room_dir.relative_to(ROOT / "outputs" / "rooms").parts) \
            if (ROOT / "outputs" / "rooms") in room_dir.parents else room_dir.name
        if not (args.start and args.target):
            ap.error("a room folder needs --start x,y and --target x,y (see floor_plan.png for coordinates)")
        start, target = args.start, args.target
        source = "built from photos by build_room.py (sizes and positions estimated from photos; masses catalog values)"
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    env = SimWorld(room_dir)
    goal = PathClear(start, target, args.radius, "a clear walking route")
    if args.video:
        rec = Recorder(env.world, out / "video.mp4")
        env.world.hooks.append(rec)
        env.record_tries = True
    else:
        rec = _StillsOnly(env)
    label_actions(env, rec)
    rec.text = "start (simulated room)"
    rec.still()

    if args.planner == "claude":
        from realgrounds.claude_planner import ClaudePlanner
        planner = ClaudePlanner()
    else:
        planner = rule_based_planner
    report = run(env, goal, planner=planner, limits=Limits(max_tries=args.max_tries, max_commits=args.max_commits))
    if args.planner == "claude":
        print(f"Claude ({planner.model}): {planner.calls} calls, ${planner.spent_usd:.4f}")
    rec.text = f"end: {report.outcome}"
    rec.still()
    rec.close(out / "stills.png")

    (out / "transcript.txt").write_text("\n".join(report.transcript) + "\n", encoding="utf-8")
    (out / "log.json").write_text(json.dumps(
        {"goal": report.goal, "outcome": report.outcome, "reason": report.reason, "help_request": report.help_request,
         "committed": report.committed, "tries": report.tries, "commits": report.commits,
         "sim_seconds": report.sim_seconds, "final_goal": report.final_goal, "provenance": report.provenance,
         "objects": env.observe()["objects"], "log": report.log}, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(_markdown(name, report, env, source), encoding="utf-8")
    print(f"\n{report.verdict()}\n\noutput: {out.relative_to(ROOT)}/  ({time.time() - t0:.0f} s)")
    return 0


class _StillsOnly:
    """The stills sheet without a video (same pictures as the Recorder)."""

    def __init__(self, env: SimWorld) -> None:
        import tempfile

        self._tmp = Path(tempfile.mkdtemp()) / "none.mp4"
        self.r = Recorder(env.world, self._tmp)
        self.text = ""

    def still(self) -> None:
        self.r.text = self.text
        self.r.still()

    def close(self, sheet: Path) -> None:
        self.r.close(sheet)
        self._tmp.unlink(missing_ok=True)


def _markdown(name: str, r, env: SimWorld, source: str) -> str:
    md = [f"# Autonomous task in simulation: {name}", "",
          f"**Goal:** {r.goal}", "", f"**Verdict:** {r.verdict().splitlines()[0]}", "",
          f"**The simulated room:** {source}.", "",
          "> Everything below happened in the simulated room. No real-world action was taken and no real-world "
          "result is claimed.", "",
          f"{r.tries} tries (each undone and checked bit for bit), {r.commits} commit(s), "
          f"{r.sim_seconds:.0f} simulated seconds.", ""]
    if r.committed:
        md += ["## Committed to the simulated room", ""] + [f"{i}. {a}" for i, a in enumerate(r.committed, 1)] + [""]
    md += ["## Every try and commit (measured in the simulation)", "",
           "| # | event | action | status | moved (cm) | goal after |", "|---|---|---|---|---|---|"]
    acts = [e for e in r.log if e["event"] in ("try", "commit")]
    for i, e in enumerate(acts, 1):
        g = e.get("goal_after") or {}
        after = "satisfied" if g.get("satisfied") else (f"{g.get('progress_cm')} cm blocked"
                                                        if g.get("progress_cm") is not None else g.get("why", "-"))
        if e["status"] != "done" and g.get("satisfied"):
            after += " - not counted: the action itself failed"
        moved = ", ".join(f"{k} {v:.0f}" for k, v in (e.get("moved_cm") or {}).items()) or "-"
        md.append(f"| {i} | {e['event']} | {e['action']} | {e['status']} | {moved} | {after} |")
    plans = [e for e in r.log if e["event"] in ("plan", "refused")]
    if plans:
        md += ["", "## The planner's proposals, and repeats the supervisor refused", ""]
        for e in plans:
            md.append(f"- planner (${e['usd']:.4f}): {e['reasoning'] or '-'} -> {e['proposals'] or 'no actions'}"
                      if e["event"] == "plan" else f"- **refused** {e['action']!r}: {e['why']}")
    md += ["", "## What the simulation assumed (estimates, not confirmed facts)", "",
           "| object | movable in simulation | size (m, w x d x h) | mass | material |", "|---|---|---|---|---|"]
    for o in env.observe()["objects"]:
        w, d = o["footprint_m"]
        md.append(f"| {o['label']} | {'yes' if o['movable_in_simulation'] else 'no (fixed)'} | "
                  f"{w} x {d} x {o['height_m']} | {o['mass'] or '-'} | {o['material'] or '-'} |")
    md += ["", "None of these was measured in the real room. Before acting there, a person should check the route "
           "and the objects' real size and weight.", ""]
    if r.help_request:
        md += ["## Request for help", "", r.help_request, ""]
    return "\n".join(md)


if __name__ == "__main__":
    sys.exit(main())

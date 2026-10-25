"""The hackathon demo, one command per moment (see DEMO.md for the script).

USAGE (from the realgrounds-voice/ folder, venv active)
    python demo.py check                   # before going on stage: key, models, API, physics, tests
    python demo.py dashboard [run]         # open the long run's page (default: the newest night run)
    python demo.py live PHOTO              # a photo taken here -> room -> Claude writes and proves tasks (videos)
    python demo.py stops                   # the three stop rules on the test rooms: success / stuck / impossible
    python demo.py replay [run] [room] [k] # re-run a solved task's kept actions on a fresh copy, with video, and
                                           # check the goal again (the physics is deterministic: same result)

Everything happens in simulation; nothing is claimed about the real room.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from realgrounds.config import ROOT

NIGHT = ROOT / "outputs" / "night"


def _open(path: Path) -> None:
    print(f"open {path}")
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)


def check() -> int:
    ok = True

    def line(good: bool, what: str, note: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  {'✓' if good else '✗'} {what}" + (f" - {note}" if note else ""))

    print("pre-stage check")
    line(os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-"), "API key loaded (from .env)")
    for m in ("moge-2-vitl-normal", "segformer-b0-ade", "grounding-dino-base", "sam-vit-base"):
        line((ROOT / "models" / m).exists(), f"model {m}")
    try:
        from realgrounds.claude_planner import ClaudePlanner

        p = ClaudePlanner()
        out = p.ask("Answer in JSON.", "Say ok.", {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                                    "required": ["ok"], "additionalProperties": False}, max_tokens=500)
        line(bool(out.get("ok")), f"Claude answers ({p.model})", f"${p.spent_usd:.5f}")
    except Exception as e:                                             # noqa: BLE001 - report, don't crash
        line(False, "Claude answers", str(e)[:120])
    try:
        import tempfile

        from realgrounds.agent.scenes import obstacle_room
        from realgrounds.sandbox import SimWorld, same, take

        env = SimWorld(obstacle_room(Path(tempfile.mkdtemp()) / "light", "light"))
        before = take(env.world, env.cmd)
        r = env.try_action("put box #2 on dresser #1")
        line(r.status == "done" and same(take(env.world, env.cmd), before), "physics: a try runs and is undone exactly")
    except Exception as e:                                             # noqa: BLE001
        line(False, "physics", str(e)[:120])
    t = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=ROOT,
                       capture_output=True, text=True)
    line(t.returncode == 0, "automated tests", (t.stderr.strip().splitlines() or [""])[-3])
    runs = sorted(NIGHT.glob("*/dashboard.html"), key=lambda p: p.stat().st_mtime)
    line(bool(runs), "a night run to show", runs[-1].parent.name if runs else "run night_shift.py first")
    print("READY" if ok else "NOT READY - fix the ✗ lines")
    return 0 if ok else 1


def dashboard(run: str | None) -> int:
    from realgrounds import dashboard as dash

    if run:
        folder = NIGHT / run
    else:
        runs = sorted((p.parent for p in NIGHT.glob("*/timeline.jsonl")), key=lambda p: p.stat().st_mtime)
        if not runs:
            print("no night run yet: python night_shift.py --nyu 10 --run first")
            return 1
        folder = runs[-1]
    _open(dash.build(folder))
    return 0


def live(photo: str) -> int:
    import night_shift

    name = "live-" + datetime.now().strftime("%H%M%S")
    t0 = time.time()
    code = night_shift.main(["--photos", photo, "--run", name, "--video", "--tasks-per-room", "3",
                             "--budget-usd", "0.5", "--room-usd", "0.3"])
    print(f"live run: {time.time() - t0:.0f} s")
    _open(NIGHT / name / "dashboard.html")
    return code


def replay(run: str | None, room: str | None, k: int | None) -> int:
    """Re-check a solution: replay only the committed actions on a fresh copy of the room, record it, check the goal."""
    import json

    from realgrounds.agent.run import Recorder
    from realgrounds.goals import ObjectRestsOn, PathClear, ReachObject
    from realgrounds.sandbox import SimWorld
    from realgrounds.tasks import label_actions

    runs = [NIGHT / run] if run else sorted((p.parent for p in NIGHT.glob("*/results.jsonl")),
                                            key=lambda p: p.stat().st_mtime, reverse=True)
    for folder in runs:
        rows = [json.loads(l) for l in (folder / "results.jsonl").read_text().splitlines()]
        hits = [(r, i, t) for r in rows if room in (None, r["room"]) for i, t in enumerate(r.get("tasks", []), 1)
                if t.get("label") == "solvable" and t.get("committed") and k in (None, i)]
        if hits:
            break
    else:
        print("no solved task found to replay")
        return 1
    r, i, t = hits[0]
    room_dir = folder / "rooms" / r["room"]
    transcript = next(iter(sorted((room_dir / "tasks").glob(f"{i}_*.txt"))), None)     # older runs: kind from the name
    kind = t.get("kind") or (transcript.stem.split("_", 1)[1] if transcript else "route")
    env = SimWorld(room_dir)
    spec = t["spec"]
    goal = {"route": lambda: PathClear(tuple(spec["start"]), tuple(spec["target"]), 0.3, "a route across the room"),
            "rests_on": lambda: ObjectRestsOn(spec["object"], spec["target"]),
            "reach": lambda: ReachObject(spec["object"], tuple(float(v) for v in env.world.base()[0][:2]))}[kind]()
    video = room_dir / "tasks" / f"{i}_replay.mp4"
    rec = Recorder(env.world, video)
    env.world.hooks.append(rec)
    label_actions(env, rec)
    rec.text = f"replay of a solved task: {t['title']}"
    rec.still()
    print(f"{folder.name} / {r['room']} / task {i}: {t['title']}")
    print(f"  before: {env.check(goal).get('why')}")
    for a in t["committed"]:
        c = env.commit_action(a, goal)
        print(f"  commit {a!r}: {c.status}")
    after = env.check(goal)
    rec.text = "goal holds again (simulated)" if after.get("satisfied") else "goal does NOT hold on replay"
    rec.still()
    rec.close(video.with_suffix(".png"))
    print(f"  after:  {'SATISFIED again' if after.get('satisfied') else 'NOT satisfied - ' + str(after.get('why'))}"
          f"  [{after['provenance']}]")
    _open(video)
    return 0 if after.get("satisfied") else 1


def stops() -> int:
    import run_task

    for scene in ("light", "heavy", "wall"):
        print(f"\n=== {scene} ===")
        run_task.main(["--scene", scene, "--planner", "claude", "--video"])
    for scene in ("light", "heavy"):
        _open(ROOT / "outputs" / "tasks" / scene / "video.mp4")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("check", "dashboard", "live", "stops", "replay"):
        print(__doc__)
        return 1
    cmd, rest = argv[0], argv[1:]
    if cmd == "check":
        return check()
    if cmd == "dashboard":
        return dashboard(rest[0] if rest else None)
    if cmd == "live":
        if not rest:
            print("usage: python demo.py live PHOTO (a .jpg or iPhone .HEIC, or a folder of them)")
            return 1
        return live(rest[0])
    if cmd == "replay":
        return replay(rest[0] if rest else None, rest[1] if len(rest) > 1 else None,
                      int(rest[2]) if len(rest) > 2 else None)
    return stops()


if __name__ == "__main__":
    sys.exit(main())

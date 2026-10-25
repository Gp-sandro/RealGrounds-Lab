"""Phase 2 experiment: tell the agent what to do, watch it do it in physics, check it was right.

USAGE (from the realgrounds-voice/ folder)
    python -m realgrounds.agent.run --scene simple                    # tasks_simple.yaml
    python -m realgrounds.agent.run --scene outputs/rooms/dorm/zone_1 --tasks tasks_dorm_zone_1.yaml
    python -m realgrounds.agent.run --scene simple --say "pick up the book and put it on the nightstand"
    (add --no-video to skip the video)
    python -m realgrounds.agent.run --scene simple --trials 10   # robustness: 10 sessions, the
                                                                         # agent starting up to 10 cm elsewhere

WHAT IT DOES (in plain words)
    Loads a room made by the pipeline (or the hand-made simple room), adds the agent's body,
    and runs the sentences one after another in the same world (like one session). A task file
    lists, for each sentence, what SHOULD happen:
        done     - it should be done, and physics must confirm it (e.g. the bottle rests on the
                   nightstand afterwards);
        refused  - it should be refused because of an object's attributes (static furniture,
                   not a surface, not in the room...);
        failed   - it should be tried and fail in physics (too heavy, in the way).
    Accuracy = how many sentences ended the way they should.

OUTPUT  outputs/agent/<scene>/
    report.md   the table: sentence, expected, got, what physics measured
    log.json    everything, step by step
    video.mp4   the physics view, with the sentence on screen
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from ..config import ROOT
from .commands import Commander
from .skills import Agent
from .world import World

HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "agent"


class Recorder:
    """Video of the physics view: furniture blue-grey, movable things orange, agent red, held thing green."""

    def __init__(self, world: World, path: Path, fps: int = 20, size: tuple[int, int] = (800, 600)) -> None:
        import cv2
        import mujoco

        self.cv2, self.mujoco = cv2, mujoco
        self.w = world
        m = world.model
        m.vis.global_.offwidth, m.vis.global_.offheight = size
        self.renderer = mujoco.Renderer(m, height=size[1], width=size[0])
        self.every = max(1, int(round(1.0 / fps / m.opt.timestep)))
        self.n = 0
        self.text = ""
        # H.264 plays everywhere (QuickTime, Safari); the older MPEG-4 format is the fallback
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
        if not self.writer.isOpened():
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.stills: list[np.ndarray] = []
        self._camera()

    def _camera(self) -> None:
        m, d = self.w.model, self.w.data
        f = m.geom("floor").id
        centre = d.geom_xpos[f].copy()
        half = m.geom_size[f][:2]
        self.cam = self.mujoco.MjvCamera()
        self.cam.lookat[:] = [centre[0], centre[1], 0.6]
        self.cam.distance = float(np.linalg.norm(half)) * 2.3 + 1.0
        self.cam.azimuth, self.cam.elevation = 90.0 - 25.0, -48.0   # from behind the photographer, above
        a, e = math.radians(self.cam.azimuth), math.radians(self.cam.elevation)
        cam_xy = centre[:2] - self.cam.distance * math.cos(e) * np.array([math.cos(a), math.sin(a)])
        self.opt = self.mujoco.MjvOption()
        for g, on in {0: False, 1: True, 2: True, 3: True, 4: False}.items():
            self.opt.geomgroup[g] = int(on)
        # see-through: collision walls/floor (the room shell shows them), and the walls between camera and room
        R = d.geom_xmat[f].reshape(3, 3)
        local = R[:2, :2].T @ (cam_xy - centre[:2])
        for g in range(m.ngeom):
            name = m.geom(g).name
            if name == "floor" or (name.startswith("wall_")):
                m.geom_rgba[g][3] = 0.0
            if name.startswith("wallvis_"):
                side = name.split("_")[1]
                out = {"u-": local[0] < -half[0], "u+": local[0] > half[0], "v-": local[1] < -half[1], "v+": local[1] > half[1]}
                if out.get(side):
                    m.geom_rgba[g][3] = 0.0

    def __call__(self) -> None:
        self.n += 1
        if self.n % self.every:
            return
        self.writer.write(self.frame()[..., ::-1])

    def frame(self) -> np.ndarray:
        m, held = self.w.model, self.w.held
        saved = {}
        if held is not None:                       # the held thing in green
            for g in held.geoms:
                saved[g] = m.geom_rgba[g].copy()
                m.geom_rgba[g] = [0.2, 0.75, 0.3, 1.0]
        self.renderer.update_scene(self.w.data, camera=self.cam, scene_option=self.opt)
        img = self.renderer.render().copy()
        for g, c in saved.items():
            m.geom_rgba[g] = c
        cv2 = self.cv2
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (20, 20, 24), -1)
        cv2.putText(img, self.text[:90], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def still(self) -> None:
        self.stills.append(self.frame())

    def close(self, sheet: Path) -> None:
        self.writer.release()
        self.renderer.close()
        if self.stills:
            cv2 = self.cv2
            small = [cv2.resize(s, (400, 300)) for s in self.stills]
            while len(small) % 3:
                small.append(np.zeros_like(small[0]))
            rows = [np.hstack(small[i:i + 3]) for i in range(0, len(small), 3)]
            cv2.imwrite(str(sheet), np.vstack(rows)[..., ::-1])


def scene_dir(scene: str) -> Path:
    if scene == "simple":
        from .scenes import simple_room

        return simple_room(OUT / "scenes" / "simple")
    p = Path(scene)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2: an agent that does what it is told, checked by physics.")
    ap.add_argument("--scene", default="simple", help="'simple' or a room folder (lab/outputs/rooms/<name>[/zone_k])")
    ap.add_argument("--tasks", help="task file (yaml) in this folder; default tasks_<scene>.yaml")
    ap.add_argument("--say", action="append", help="sentence(s) to run instead of a task file")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--trials", type=int, default=1, help="repeat the session N times from slightly different starts")
    ap.add_argument("--jitter", type=float, default=0.1, help="how far (m) the start may move in --trials")
    args = ap.parse_args(argv)

    room_dir = scene_dir(args.scene)
    name = "simple" if args.scene == "simple" else "_".join(room_dir.relative_to(ROOT / "outputs" / "rooms").parts) \
        if (ROOT / "outputs" / "rooms") in room_dir.parents else room_dir.name
    if args.say:
        tasks = [{"say": s} for s in args.say]
    else:
        tf = HERE / (args.tasks or f"tasks_{name}.yaml")
        tasks = yaml.safe_load(tf.read_text())["tasks"]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    if args.trials > 1:
        return robustness(room_dir, name, tasks, args.trials, args.jitter, out)

    t0 = time.time()
    world = World(room_dir)
    rec = None if args.no_video else Recorder(world, out / "video.mp4")
    if rec:
        world.hooks.append(rec)
    agent = Agent(world)
    boss = Commander(world, agent)
    print(f"=== {name}: {len(world.things)} objects, agent at {world.base()[0].round(2).tolist()} ===")
    rows, log = [], []
    for task in tasks:
        text = task["say"]
        if rec:
            rec.text = text
        sim0 = world.data.time
        o = boss.run(text)
        if rec:
            rec.still()
        expect = task.get("expect")
        right = expect is None or expect == o.status
        mark = "" if expect is None else ("  [as expected]" if right else f"  [EXPECTED {expect.upper()}]")
        print(f"\n> {text}\n  -> {o.status.upper()}{mark}")
        for line in o.lines:
            print("    " + line)
        rows.append((text, expect or "-", o.status, right, "; ".join(o.lines)))
        log.append({"say": text, "expect": expect, "status": o.status, "as_expected": right, "lines": o.lines,
                    "facts": o.facts, "sim_seconds": round(world.data.time - sim0, 1)})
    if rec:
        rec.close(out / "stills.png")
    scored = [r for r in rows if r[1] != "-"]
    acc = sum(r[3] for r in scored)
    print(f"\n=== {name}: {acc}/{len(scored)} sentences ended as expected · {time.time() - t0:.0f} s ===")
    (out / "log.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    md = [f"# Agent session: {name}", "", f"{acc}/{len(scored)} sentences ended as expected.", "",
          "| # | sentence | expected | got | what happened (measured in physics) |", "|---|---|---|---|---|"]
    for i, (text, exp, got, right, lines) in enumerate(rows, 1):
        md.append(f"| {i} | {text} | {exp} | {'✓ ' if right else '✗ '}{got} | {lines.replace('|', '/')} |")
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"output: {out.relative_to(ROOT)}/")
    return 0 if acc == len(scored) else 1


def robustness(room_dir: Path, name: str, tasks: list[dict], trials: int, jitter: float, out: Path) -> int:
    """Run the same session `trials` times, the agent starting up to `jitter` m away from its usual
    spot. A sentence is robust when it ends as expected in (nearly) every trial."""
    rng = np.random.default_rng(0)
    hits = np.zeros(len(tasks), int)
    t0 = time.time()
    for k in range(trials):
        offset = (0.0, 0.0) if k == 0 else tuple(rng.uniform(-jitter, jitter, 2))
        world = World(room_dir, start_offset=offset)
        boss = Commander(world, Agent(world))
        for i, task in enumerate(tasks):
            o = boss.run(task["say"])
            hits[i] += task.get("expect") in (None, o.status)
        print(f"  trial {k + 1}/{trials} (start moved {np.hypot(*offset) * 100:.0f} cm): {int(sum(1 for i in range(len(tasks)) if hits[i] == k + 1))}/{len(tasks)} as expected so far in every trial")
    lines = [f"# Robustness: {name}", "", f"{trials} sessions, the agent starting up to {jitter * 100:.0f} cm from its usual spot.", "",
             "| # | sentence | expected | as expected |", "|---|---|---|---|"]
    for i, task in enumerate(tasks):
        lines.append(f"| {i + 1} | {task['say']} | {task.get('expect', '-')} | {hits[i]}/{trials} |")
    total = int(hits.sum())
    lines += ["", f"Overall: {total}/{trials * len(tasks)} = {100 * total / (trials * len(tasks)):.0f}%"]
    (out / "robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"=== {name}: {total}/{trials * len(tasks)} sentence runs ended as expected ({100 * total / (trials * len(tasks)):.0f}%) "
          f"over {trials} trials · {time.time() - t0:.0f} s ===")
    for i, task in enumerate(tasks):
        if hits[i] < trials:
            print(f"  {hits[i]}/{trials}  {task['say']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

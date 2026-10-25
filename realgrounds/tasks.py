"""Tasks for a room, and their proof in simulation.

    far_side_route()   automatic: "walk from where you stand to the farthest floor spot of the room"
                       (the target is chosen with every movable thing removed, so it is a place the body could
                       reach if the room were tidy - whether it can now is what the task finds out)
    claude_tasks()     Claude reads the room and writes tasks a household robot might be asked: put an object
                       on a surface / on the floor, or get to an object. Tasks that name a missing object, move
                       fixed furniture, or are already true are dropped (and logged) before any proof.
    prove()            runs the autonomous loop on a FRESH copy of the room, so tasks never affect each other.

Labels (never stronger than the evidence):
    solvable     the goal was reached in the simulated room; the actions that did it are recorded
    impossible   geometry shows no action can reach it (route blocked by walls / fixed furniture only)
    unresolved   stuck, out of budget, or the planner failed - with what was tried and why it failed
    already_true the goal held before any action
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .agent.skills import CELL, FloorGrid
from .autonomy import Report, run
from .claude_planner import ClaudePlanner, room_for_prompt
from .goals import ObjectRestsOn, PathClear, ReachObject
from .sandbox import SimWorld
from .supervisor import Limits


@dataclass
class Task:
    kind: str                         # route / reach / rests_on
    title: str
    source: str                       # auto / claude
    goal: Any
    why: str = ""
    spec: dict[str, Any] = field(default_factory=dict)


def far_side_route(env: SimWorld, min_m: float = 1.0) -> Task | None:
    designed = env.room_dir / "route.json"            # a test room declares the route it was built for
    if designed.exists():
        d = json.loads(designed.read_text())
        return Task("route", d["title"], "auto", PathClear(tuple(d["start"]), tuple(d["target"]), d["radius"], d["title"]),
                    spec={"start": tuple(d["start"]), "target": tuple(d["target"])})
    w = env.world
    start = w.base()[0][:2].astype(float)
    relaxed = FloorGrid(w, ignore={t.label for t in w.things.values() if t.body is not None})
    reach = relaxed.reach_map(start)
    if reach is None:
        return None
    dist = np.where(np.isfinite(reach[0]), reach[0], -1.0)
    cell = np.unravel_index(int(np.argmax(dist)), dist.shape)
    if dist[cell] * CELL < min_m:
        return None
    target = relaxed.origin + (np.array([cell[1], cell[0]]) + 0.5) * CELL
    s, t = (round(float(v), 2) for v in start), (round(float(v), 2) for v in target)
    start_xy, target_xy = tuple(s), tuple(t)
    return Task("route", "walk across the room to its farthest floor spot", "auto",
                PathClear(start_xy, target_xy, 0.3, "a route across the room"),
                spec={"start": start_xy, "target": target_xy, "walk_m": round(float(dist[cell]) * CELL, 2)})


TASK_SYSTEM = """You write tasks for a simulated robot body working in a SIMULATED copy of a real room (built from \
a photo; sizes are estimates, masses catalog values). Write tasks a person might really ask a household robot, \
that can be checked by measurement:
  rests_on: put a movable object on a surface object (or "floor")   -> object, target
  reach:    get the body next to an object                           -> object, target = ""
Rules: use object labels exactly as given; only objects marked movable_in_simulation can be moved; the target \
of rests_on must be something things can rest on (a table, desk, shelf, bed, the floor...); do not write a task \
that is already true (look at rests_on_now). Prefer tasks that need real work over trivial ones."""

TASK_SCHEMA = {
    "type": "object",
    "properties": {"tasks": {"type": "array", "items": {
        "type": "object",
        "properties": {"kind": {"type": "string", "enum": ["rests_on", "reach"]},
                       "object": {"type": "string"}, "target": {"type": "string"},
                       "why": {"type": "string", "description": "one sentence: why someone would ask this"}},
        "required": ["kind", "object", "target", "why"], "additionalProperties": False}}},
    "required": ["tasks"],
    "additionalProperties": False,
}


def claude_tasks(env: SimWorld, planner: ClaudePlanner, n: int = 2) -> tuple[list[Task], list[str]]:
    """(tasks, why some proposals were dropped)."""
    out = planner.ask(TASK_SYSTEM, json.dumps({"room": room_for_prompt(env), "ask": f"Write {n} tasks."}, default=str),
                      TASK_SCHEMA)
    tasks, dropped = [], []
    for p in out.get("tasks", [])[:n]:
        task, why_not = make_task(env, p.get("kind", ""), p.get("object", ""), p.get("target", ""), p.get("why", ""),
                                  "claude")
        if task is None:
            dropped.append(f"{p}: {why_not}")
        else:
            tasks.append(task)
    return tasks, dropped


def make_task(env: SimWorld, kind: str, obj: str = "", target: str = "", why: str = "",
              source: str = "you") -> tuple[Task | None, str]:
    """A task from its parts, checked against the room before any proof: (task, "") or (None, why not)."""
    w = env.world
    if kind == "route":
        t = far_side_route(env)
        return (t, "") if t else (None, "no floor spot at least 1 m away to walk to")
    t = w.things.get(obj)
    if t is None:
        return None, f"no object {obj!r} in the room"
    if kind == "reach":
        goal = ReachObject(obj, tuple(round(float(v), 2) for v in w.base()[0][:2]))
        if goal.check(env).get("satisfied"):
            return None, "already true (a route to it exists now)"
        return Task("reach", f"get next to {obj}", source, goal, why, {"object": obj}), ""
    if kind != "rests_on":
        return None, f"unknown kind of task {kind!r}"
    if t.body is None:
        return None, f"{obj} is fixed in this simulation"
    if not {"grasp", "lift", "carry"} & set(t.affordances):            # the command layer's own rule
        return None, f"{obj} can't be picked up (its catalog affordances: {', '.join(t.affordances)})"
    if target != "floor" and (target not in w.things or not w.things[target].is_surface):
        return None, f"{target!r} is not a surface in the room"
    goal = ObjectRestsOn(obj, target)
    if goal.check(env).get("satisfied"):
        return None, "already true"
    return Task("rests_on", f"put {obj} on {target}", source, goal, why, {"object": obj, "target": target}), ""


def label_actions(env: SimWorld, rec) -> None:
    """Put the action on the video - TRY (undone afterwards) or COMMIT - and a still after every commit."""
    try_action, commit_action = env.try_action, env.commit_action

    def tried(action, goal=None):
        rec.text = f"TRY in a sandbox (undone afterwards): {action}"
        return try_action(action, goal)

    def committed(action, goal=None):
        rec.text = f"COMMIT to the simulated room: {action}"
        r = commit_action(action, goal)
        rec.still()
        return r

    env.try_action, env.commit_action = tried, committed


def prove(room_dir: Path, task: Task, planner, limits: Limits, say=lambda s: None,
          video: Path | None = None) -> tuple[str, Report]:
    env = SimWorld(room_dir)
    rec = None
    if video is not None:                                  # every try (undone) and commit, on screen
        from .agent.run import Recorder

        rec = Recorder(env.world, video)
        env.world.hooks.append(rec)
        env.record_tries = True
        label_actions(env, rec)
        rec.text = f"task: {task.title}"
        rec.still()
    already = env.check(task.goal).get("satisfied")
    r = run(env, task.goal, planner=planner, limits=limits, say=say)
    if rec is not None:
        rec.text = f"{task.title}: {r.outcome}"
        rec.still()
        rec.close(video.with_suffix(".png"))
    if already:
        return "already_true", r
    return {"simulated_success": "solvable", "infeasible": "impossible"}.get(r.outcome, "unresolved"), r

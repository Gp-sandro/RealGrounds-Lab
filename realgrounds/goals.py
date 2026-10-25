"""Objective, deterministic goal checks on the SIMULATED room - geometry and collisions, no model opinion.

PathClear       a walking route for the agent's body (0.4 m wide + 3 cm margin, things lower than 20 cm stepped
                over) from a start point to a target area exists. Also measures:
                  progress  cm of the best possible route still blocked by movable things (0 = clear)
                  feasible  could it ever be cleared by moving things? (no route even with every movable
                            thing removed = walls / fixed furniture block it: no action can succeed)
ReachObject     the same, where the target area is every floor spot within the arm's reach of an object's
                outline (the body's own rule for picking things up) - "get to the water bottle on the sofa"
ObjectRestsOn   an object rests on a target (from the physics contacts under it, not from the record)

Every result says it is a simulated check: the room was estimated from photos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .agent.skills import CELL, REACH, FloorGrid
from .sandbox import SIMULATED


def _circle(target: np.ndarray, radius: float):
    def mask(grid: FloorGrid) -> np.ndarray:
        rr, cc = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
        xy = grid.origin + (np.stack([cc, rr], -1) + 0.5) * CELL
        return np.linalg.norm(xy - target, axis=-1) <= radius
    return mask


def _route(grid: FloorGrid, start: np.ndarray, target_mask) -> list[tuple[int, int]] | None:
    """Shortest walking route (grid cells) from start into the target area (target_mask(grid)), or None."""
    reach = grid.reach_map(start)
    if reach is None:
        return None
    dist, prev, free = reach
    goal = target_mask(grid) & free & np.isfinite(dist)
    if not goal.any():
        return None
    cell = tuple(int(v) for v in np.unravel_index(np.argmin(np.where(goal, dist, np.inf)), dist.shape))
    path = [cell]
    while path[-1] in prev:
        path.append(prev[path[-1]])
    return path[::-1]


@dataclass
class PathClear:
    start: tuple[float, float]
    target: tuple[float, float]
    radius: float = 0.3
    name: str = "a clear walking route"

    def describe(self) -> str:
        return (f"{self.name}: from ({self.start[0]:.2f}, {self.start[1]:.2f}) to within {self.radius:.2f} m of "
                f"({self.target[0]:.2f}, {self.target[1]:.2f}) for a 0.4 m wide body")

    def check(self, env) -> dict[str, Any]:
        return route_check(env, self.describe(), np.array(self.start, float),
                           _circle(np.array(self.target, float), self.radius))


def route_check(env, description: str, start: np.ndarray, target_mask) -> dict[str, Any]:
    """The route check behind PathClear and ReachObject (see the file header)."""
    w = env.world
    out: dict[str, Any] = {"goal": description, "provenance": SIMULATED}
    if w.held is not None:
        # a route only counts with free hands (the held thing would be put down somewhere)
        out.update(satisfied=False, feasible=True, progress_cm=None, why=f"holding {w.held.label}")
        return out
    full = FloorGrid(w)
    route = _route(full, start, target_mask)
    if route is not None:
        out.update(satisfied=True, feasible=True, progress_cm=0.0, route_m=round(len(route) * CELL, 2),
                   blocked_by=[], why="a route exists in the simulated room")
        return out
    movable = {t.label for t in w.things.values() if t.body is not None}
    relaxed = FloorGrid(w, ignore=movable)
    best = _route(relaxed, start, target_mask)
    if best is None:
        out.update(satisfied=False, feasible=False, progress_cm=None, blocked_by=[],
                   why="no route even with every movable thing removed: walls or fixed furniture block it")
        return out
    blocked_cells = [c for c in best if not full.free[c]]
    blockers = []
    for lab in sorted(movable):                      # which movable things sit on that best route
        one = FloorGrid(w, ignore=movable - {lab})
        if any(not one.free[c] for c in best):
            blockers.append(lab)
    out.update(satisfied=False, feasible=True, progress_cm=round(len(blocked_cells) * CELL * 100, 1),
               blocked_by=blockers, why=f"{len(blocked_cells) * CELL * 100:.0f} cm of the best route is blocked")
    return out


@dataclass
class ReachObject:
    obj: str
    start: tuple[float, float]
    reach: float = REACH[1] - 0.2                        # what the body itself uses to pick things up

    def describe(self) -> str:
        return (f"get within arm's reach of {self.obj}: stand within {self.reach:.2f} m of its outline, walking "
                f"from ({self.start[0]:.2f}, {self.start[1]:.2f})")

    def check(self, env) -> dict[str, Any]:
        w = env.world
        t = w.things.get(self.obj)
        if t is None:
            return {"goal": self.describe(), "provenance": SIMULATED, "satisfied": False, "feasible": False,
                    "progress_cm": None, "blocked_by": [], "why": "no such object in the simulated room"}
        hull = w.extent(t)["hull"]
        out = route_check(env, self.describe(), np.array(self.start, float),
                          lambda grid: grid.near(grid.region(hull), self.reach))
        out["object"] = self.obj
        return out


@dataclass
class ObjectRestsOn:
    obj: str
    target: str                                          # an object label, or "floor"

    def describe(self) -> str:
        return f"{self.obj} rests on {self.target}"

    def check(self, env) -> dict[str, Any]:
        w = env.world
        t = w.things.get(self.obj)
        out: dict[str, Any] = {"goal": self.describe(), "provenance": SIMULATED, "object": self.obj,
                               "target": self.target}
        if t is None or (self.target != "floor" and self.target not in w.things):
            out.update(satisfied=False, feasible=False, progress_cm=None, why="no such object in the simulated room")
            return out
        on = w.rests_on(t, self.target)
        out.update(satisfied=bool(on and w.held is not t), feasible=t.body is not None or on,
                   progress_cm=0.0 if on else 100.0, rests_on_now=w.resting_on(t),
                   why="from the contacts under it" if on else f"it rests on {', '.join(w.resting_on(t)) or 'nothing'}")
        return out

"""The simulated world as tools for an agent: look, try an action in isolation, commit it, check a goal.

Everything here happens in the SIMULATED copy of the room. Nothing is a real-world action and no result is
real-world verified: the copy was estimated from photos (sizes) and a catalog (masses, materials).

    observe()             what is in the simulated room now (with where each property came from)
    try_action(text)      run an action from a snapshot, measure what happened, restore the snapshot -
                          the persistent world is unchanged afterwards (verified on every call)
    commit_action(text)   run the action on the persistent SIMULATED world (it stays)
    check(goal)           an objective, deterministic goal check (realgrounds/goals.py)

Isolation: a Snapshot holds every piece of state the action code can change - MuJoCo's full integration
state (positions, velocities, controls, active welds, warm start, time), the model arrays changed at run
time (weld poses, collision bits of a held object), the body's Python state (what it holds, where its hand
and base are told to be) and the command layer's memory ("it", where the held thing came from).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .agent.commands import Commander
from .agent.skills import Agent
from .agent.world import World

SIMULATED = "simulated (estimated from photos; not verified in the real world)"
SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


# ------------------------------------------------------------------------------------------------
# snapshot / restore
# ------------------------------------------------------------------------------------------------

@dataclass
class Snapshot:
    physics: np.ndarray
    eq_data: np.ndarray
    contype: np.ndarray
    conaffinity: np.ndarray
    world: dict[str, Any]
    commander: dict[str, Any]


def take(world: World, cmd: Commander) -> Snapshot:
    m, d = world.model, world.data
    state = np.empty(mujoco.mj_stateSize(m, SPEC))
    mujoco.mj_getState(m, d, state, SPEC)
    return Snapshot(
        physics=state, eq_data=m.eq_data.copy(), contype=m.geom_contype.copy(), conaffinity=m.geom_conaffinity.copy(),
        world={"held": world.held.label if world.held else None, "hand_above_bottom": world.hand_above_bottom,
               "cmd_hand": world.cmd_hand.copy(), "cmd_base": world.cmd_base.copy(),
               "_bits": dict(getattr(world, "_bits", {})), "unhooked": getattr(world, "unhooked", None)},
        commander={"last": cmd.last.label if cmd.last else None, "_picked_from": cmd._picked_from, "missing": cmd.missing})


def restore(world: World, cmd: Commander, s: Snapshot) -> None:
    m, d = world.model, world.data
    m.eq_data[:] = s.eq_data
    m.geom_contype[:] = s.contype
    m.geom_conaffinity[:] = s.conaffinity
    mujoco.mj_setState(m, d, s.physics, SPEC)
    mujoco.mj_forward(m, d)
    w = s.world
    world.held = world.things[w["held"]] if w["held"] else None
    world.hand_above_bottom = w["hand_above_bottom"]
    world.cmd_hand, world.cmd_base = w["cmd_hand"].copy(), w["cmd_base"].copy()
    world._bits, world.unhooked = dict(w["_bits"]), w["unhooked"]
    c = s.commander
    cmd.last = world.things[c["last"]] if c["last"] else None
    cmd._picked_from, cmd.missing = c["_picked_from"], c["missing"]


def same(a: Snapshot, b: Snapshot) -> bool:
    """Bit-identical (physics arrays) and equal (Python state)."""
    return (np.array_equal(a.physics, b.physics) and np.array_equal(a.eq_data, b.eq_data)
            and np.array_equal(a.contype, b.contype) and np.array_equal(a.conaffinity, b.conaffinity)
            and a.world["held"] == b.world["held"] and a.commander == b.commander
            and np.array_equal(a.world["cmd_hand"], b.world["cmd_hand"])
            and np.array_equal(a.world["cmd_base"], b.world["cmd_base"]))


# ------------------------------------------------------------------------------------------------
# the tools
# ------------------------------------------------------------------------------------------------

@dataclass
class ActionResult:
    action: str
    mode: str                                # "try" (restored afterwards) or "commit" (kept)
    status: str                              # done / refused / failed (from the command layer)
    lines: list[str]
    moved: dict[str, float]                  # objects that moved, cm (measured in the simulation)
    goal: dict[str, Any] | None              # the goal check after the action (if a goal was given)
    sim_seconds: float
    state_after: str                         # hash of the resulting simulated state
    restored: bool | None = None             # try only: the persistent world is bit-identical afterwards
    provenance: str = SIMULATED
    facts: list[dict[str, Any]] = field(default_factory=list)


class SimWorld:
    """A room (built from photos by build_room.py, or a hand-made test scene) as an agent environment."""

    def __init__(self, room_dir: Path) -> None:
        self.room_dir = Path(room_dir)
        self.world = World(self.room_dir)
        self.agent = Agent(self.world)
        self.cmd = Commander(self.world, self.agent)
        self.record_tries = False                # video of tries too (rendering reads the physics, never writes it)

    # -------------------------------------------------------------- looking
    def observe(self) -> dict[str, Any]:
        w = self.world
        objects = []
        for t in sorted(w.things.values(), key=lambda t: t.index):
            e = w.extent(t)
            rec = t.record
            objects.append({
                "label": t.label,
                "movable_in_simulation": t.body is not None,
                "state": rec.get("State"),
                "affordances": t.affordances,
                "position_m": [round(float(v), 2) for v in e["center"]],
                "footprint_m": [round(float(v), 2) for v in (e["xy_max"] - e["xy_min"])],
                "height_m": round(e["zmax"] - e["zmin"], 2),
                "rests_on_now": w.resting_on(t),
                "mass": rec.get("Mass"),                      # e.g. "6 kg [typical]": a catalog value, not measured
                "material": rec.get("Material"),
                "inferred": rec.get("Inferred", []),
            })
        xy, yaw = w.base()
        return {"provenance": SIMULATED,
                "notes": "sizes and positions are estimated from photos; masses and materials are catalog typical values",
                "agent": {"position_m": [round(float(v), 2) for v in xy], "heading_rad": round(yaw, 3),
                          "holding": w.held.label if w.held else None},
                "objects": objects}

    def state_hash(self) -> str:
        """The simulated state, rounded to what matters (1 cm, 0.05 rad, what is held): equal hashes = the
        same situation (for loop detection; tiny physics jitter does not count as a new state)."""
        w = self.world
        parts = [f"held={w.held.label if w.held else ''}"]
        for t in sorted(w.things.values(), key=lambda t: t.index):
            if t.body is None:
                continue
            p = w.data.xpos[t.body]
            q = w.data.xquat[t.body]
            yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            parts.append(f"{t.label}:{p[0]:.2f},{p[1]:.2f},{p[2]:.2f},{round(yaw / 0.05)}")
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]

    def _positions(self) -> dict[str, np.ndarray]:
        w = self.world
        return {t.label: w.data.xpos[t.body].copy() for t in w.things.values() if t.body is not None}

    def _run(self, action: str, goal, mode: str) -> ActionResult:
        before = self._positions()
        t0 = self.world.data.time
        out = self.cmd.run(action)
        after = self._positions()
        moved = {k: round(float(np.linalg.norm(after[k][:2] - before[k][:2])) * 100, 1) for k in before
                 if np.linalg.norm(after[k][:2] - before[k][:2]) > 0.02}
        g = goal.check(self) if goal is not None else None
        return ActionResult(action, mode, out.status, out.lines, moved, g, round(self.world.data.time - t0, 2),
                            self.state_hash(), facts=out.facts)

    # -------------------------------------------------------------- acting
    def try_action(self, action: str, goal=None) -> ActionResult:
        """Run `action` from a snapshot, measure, restore. The persistent world is unchanged afterwards
        (checked bit for bit; a failed restore raises - a try must never leak into the world)."""
        snap = take(self.world, self.cmd)
        hooks = self.world.hooks
        if not self.record_tries:
            self.world.hooks = []                               # no video frames from tries
        try:
            r = self._run(action, goal, "try")
        finally:
            restore(self.world, self.cmd, snap)
            self.world.hooks = hooks
        r.restored = same(take(self.world, self.cmd), snap)
        if not r.restored:
            raise RuntimeError(f"try_action({action!r}) did not restore the simulated world exactly")
        return r

    def commit_action(self, action: str, goal=None) -> ActionResult:
        """Run `action` on the persistent SIMULATED world. This is not a real-world action."""
        return self._run(action, goal, "commit")

    def check(self, goal) -> dict[str, Any]:
        return goal.check(self)

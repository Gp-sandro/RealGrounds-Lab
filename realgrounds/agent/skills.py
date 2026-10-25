"""What the agent can do: walk, turn, pick up, put on, pull out, push, drop.

WHAT THIS FILE DOES (in plain words)
    Every skill MOVES THE BODY in physics (MuJoCo) and then CHECKS the result in physics:

      walk_to     plans a path on the floor plan (around furniture, using the body's size) and
                  follows it; blocked = it says what blocked it.
      pick_up     walks within reach and puts the hand on top of the object, grips, lifts 20 cm.
                  Done only if the object really rose (a 40 kg box doesn't: the arm is too weak).
                  No room above it (a bag hanging on a loft bed's rail)? It takes it from its
                  open side and draws it off the hook instead.
      put_on      finds a free spot on the target (a blanket on a bed counts as the bed), carries
                  the object there, lowers it, lets go. Done only if it then RESTS ON the target.
      pull_out    grips a chair (from the front if it is tucked under the desk top) and walks
                  backwards out of the desk; a leg in the way: sideways first, or at an angle.
      push        grips and walks forwards. Done if it moved.
      drop        lets go; says where it landed.

    Each returns a Result: ok, a sentence, and the measured facts (cm, kg, contacts).

HOW WALKING IS PLANNED
    The floor is a grid of 5 cm cells. A cell is blocked when anything lower than the agent's
    head stands there (a desk, a loft bed - you can't stand under them) or it is outside the
    walls; things lower than 20 cm (shoes) are stepped over. Blocked cells are grown by the
    body's radius, so a free cell = the body fits there. When carrying something (hugged against
    the chest), the boxes at carrying height also count, grown by how far it sticks out; carrying
    is tried at 1.2 m, then lower (0.95, 0.75 m) if that is what fits (under a loft bed). The
    shortest path (Dijkstra, 8 directions) leads to the nearest cell within arm's reach.

HOW PUTTING DOWN IS CHECKED BEFORE MOVING
    A spot must hold the object's real footprint (turned the way it will be, seen from where the
    body will stand), a hand's width from its neighbours, with room above it for object + hand,
    a place to stand within reach, and a clear way down (beside the body or onto the spot). The
    best three spots are tried in turn.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from .world import HAND_HALF, HEIGHT, LIFT, LIFT_KG, RADIUS, REACH, SIDE, STEP, Thing, World, outline, reach_at

CELL = 0.05           # floor grid (m)
SPEED = 0.5           # walking speed (m/s)
CTRL_DT = 0.02        # how often the controller updates its targets (s)
CARRY_HEIGHTS = (1.2, 0.95, 0.75)  # carried things travel with their bottom this high: first above desks
                                  # and what's on them; lower if that doesn't fit (under a loft bed)
ARM_IN = 0.4          # hand distance in front of the body when reaching out
HUG = 0.22            # ...and while carrying: things are held close against the chest
GAP = 0.04            # space left between a put-down thing and its neighbours (a hand fits between)
SOFT_CARRY_M = 0.4    # a soft thing carried against the chest is at most this wide (it folds)
TUCKED = 0.15         # ...and with empty hands: inside the torso outline, so it can't catch on furniture


@dataclass
class Result:
    ok: bool
    message: str
    facts: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------------------------
# floor plan + path
# ------------------------------------------------------------------------------------------

class FloorGrid:
    """Where the body can stand (see the file header)."""

    def __init__(self, world: World, ignore: set[str] = frozenset(), carry: tuple[float, float, float] | None = None) -> None:
        """carry = (bottom, top, how far it sticks out from the body centre) of a carried thing."""
        m, d = world.model, world.data
        floor = m.geom("floor").id
        half = m.geom_size[floor][:2]
        R = d.geom_xmat[floor].reshape(3, 3)
        c = d.geom_xpos[floor][:2]
        room = np.array([c + R[:2, :2] @ (half * np.array(s)) for s in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        self.origin = room.min(axis=0) - 0.1
        size = room.max(axis=0) + 0.1 - self.origin
        self.shape = (int(math.ceil(size[1] / CELL)), int(math.ceil(size[0] / CELL)))
        inside = np.zeros(self.shape, np.uint8)
        cv2.fillPoly(inside, [self.px(room)], 1)
        body = np.zeros(self.shape, np.uint8)     # things the body bumps into
        high = np.zeros(self.shape, np.uint8)     # things a carried object would bump into
        for t in world.things.values():
            if t.label in ignore or (world.held is not None and t.label == world.held.label):
                continue
            for hull, bottom, top in world.boxes(t):      # box by box: a loft bed's slab vs its posts
                if bottom < HEIGHT and top > STEP:          # shoes are stepped over
                    cv2.fillPoly(body, [self.px(hull)], 1)
                if carry is not None and bottom < carry[1] and top > carry[0]:
                    cv2.fillPoly(high, [self.px(hull)], 1)
        self.solid = (body > 0) | (inside == 0)
        self.free = ~self._grow(self.solid, RADIUS + 0.03)
        if carry is not None:                       # the carried thing must fit too
            self.free &= ~self._grow((high > 0) | (inside == 0), carry[2])

    def _grow(self, mask: np.ndarray, radius: float) -> np.ndarray:
        r = max(1, int(math.ceil(radius / CELL)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(mask.astype(np.uint8), k) > 0

    def px(self, pts: np.ndarray) -> np.ndarray:
        """World (x, y) points -> integer pixel (col, row) points for drawing."""
        return np.round((np.asarray(pts)[:, :2] - self.origin) / CELL - 0.5).astype(np.int32)

    def cell(self, xy: np.ndarray) -> tuple[int, int]:
        c = np.floor((np.asarray(xy) - self.origin) / CELL).astype(int)
        return int(np.clip(c[1], 0, self.shape[0] - 1)), int(np.clip(c[0], 0, self.shape[1] - 1))

    def xy(self, rc: tuple[int, int]) -> np.ndarray:
        return self.origin + (np.array([rc[1], rc[0]]) + 0.5) * CELL

    def near(self, region: np.ndarray, reach: float) -> np.ndarray:
        """Cells whose centre is within `reach` of a region mask."""
        dist = cv2.distanceTransform((region == 0).astype(np.uint8), cv2.DIST_L2, 5) * CELL
        return dist <= reach

    def region(self, hull: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.shape, np.uint8)
        cv2.fillPoly(mask, [self.px(hull)], 1)
        if not mask.any():                       # tiny thing: at least its own cell
            mask[self.cell(hull.mean(axis=0))] = 1
        return mask

    def reach_map(self, start: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray] | None:
        """Walking distance (cells) from `start` to every free cell (Dijkstra, 8 directions):
        (distances, previous cell, the free mask used). None = the body is stuck."""
        s = self.cell(start)
        free = self.free.copy()
        if not free[s]:
            # the body starts touching something: step to the nearest free cell first
            cells = np.argwhere(free)
            if not len(cells):
                return None
            near = cells[np.argmin(np.hypot(*(cells - np.array(s)).T))]
            if np.hypot(*(near - np.array(s))) * CELL > 0.4:
                return None
            free[s] = True
            self._line(free, s, tuple(near))
        dist = np.full(self.shape, np.inf)
        prev: dict[tuple[int, int], tuple[int, int]] = {}
        dist[s] = 0.0
        heap = [(0.0, s)]
        steps = [(dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        while heap:
            dcur, cur = heapq.heappop(heap)
            if dcur > dist[cur]:
                continue
            for dr, dc, w in steps:
                n = (cur[0] + dr, cur[1] + dc)
                if 0 <= n[0] < self.shape[0] and 0 <= n[1] < self.shape[1] and free[n] and dcur + w < dist[n]:
                    dist[n] = dcur + w
                    prev[n] = cur
                    heapq.heappush(heap, (dcur + w, n))
        return dist, prev, free

    def path_to(self, prev: dict, cell: tuple[int, int]) -> list[np.ndarray]:
        path = [cell]
        while path[-1] in prev:
            path.append(prev[path[-1]])
        return self._simplify([self.xy(rc) for rc in reversed(path)])

    def nearest(self, reach: tuple[np.ndarray, dict, np.ndarray], goal: np.ndarray) -> tuple[int, int] | None:
        """The goal cell that is the shortest walk away (None = none reachable)."""
        dist, _, free = reach
        d = np.where(goal & free, dist, np.inf)
        if not np.isfinite(d).any():
            return None
        return tuple(int(v) for v in np.unravel_index(np.argmin(d), d.shape))

    def plan(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray] | None:
        """Shortest path from `start` (x, y) to any cell of the `goal` mask, as waypoints."""
        reach = self.reach_map(start)
        if reach is None:
            return None
        cell = self.nearest(reach, goal)
        return None if cell is None else self.path_to(reach[1], cell)

    def _line(self, mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
        for t in np.linspace(0, 1, 20):
            mask[int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t))] = True

    def _clear(self, a: np.ndarray, b: np.ndarray) -> bool:
        n = max(2, int(np.linalg.norm(b - a) / (CELL / 2)))
        return all(self.free[self.cell(a + (b - a) * t)] for t in np.linspace(0, 1, n))

    def _simplify(self, pts: list[np.ndarray]) -> list[np.ndarray]:
        """Keep only the corners of the path (straight lines where nothing is in the way)."""
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._clear(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out


# ------------------------------------------------------------------------------------------
# the agent
# ------------------------------------------------------------------------------------------

class Agent:
    """Skills on top of a World. `say` receives one line per thing done (for the log)."""

    def __init__(self, world: World, say: Callable[[str], None] = lambda s: None) -> None:
        self.w = world
        self.say = say

    # ---------------------------------------------------------------- low level
    def _wait(self, seconds: float, until: Callable[[], bool] | None = None) -> bool:
        t = 0.0
        while t < seconds:
            self.w.step(CTRL_DT)
            t += CTRL_DT
            if until is not None and until():
                return True
        return until is None

    def _move_hand(self, goal: np.ndarray, speed: float = 0.5, until: Callable[[], bool] | None = None) -> bool:
        """Move the hand target smoothly to `goal` (body frame). Stops early if `until()`."""
        start = self.w.cmd_hand.copy()
        goal = np.array([np.clip(goal[0], *REACH), np.clip(goal[1], *SIDE), np.clip(goal[2], *LIFT)])
        n = max(1, int(np.linalg.norm(goal - start) / (speed * CTRL_DT)))
        for i in range(1, n + 1):
            self.w.set_hand(start + (goal - start) * i / n)
            self.w.step(CTRL_DT)
            if until is not None and until():
                return True
        def settled() -> bool:
            near = np.linalg.norm(self.w.to_local(self.w.hand()) - goal) < 0.02
            return near or (until is not None and until())
        self._wait(1.0, settled)
        return until() if until is not None else True

    def _turn_to(self, point: np.ndarray) -> None:
        xy, yaw = self.w.base()
        want = math.atan2(point[1] - xy[1], point[0] - xy[0])
        target = self.w.cmd_base[2] + (want - yaw + math.pi) % (2 * math.pi) - math.pi
        self.w.set_base(self.w.cmd_base[:2], target)
        self._wait(3.0, lambda: abs((self.w.base()[1] - want + math.pi) % (2 * math.pi) - math.pi) < math.radians(2))

    def _carry_pose(self, bottom: float = CARRY_HEIGHTS[0]) -> None:
        """Before walking: empty hand tucked in at chest height; anything held is hugged close in
        front of the body with its bottom `bottom` m high."""
        if self.w.held is None:
            self._move_hand(np.array([TUCKED, 0.0, self.w.cmd_hand[2]]))
            self._move_hand(np.array([TUCKED, 0.0, 1.1]))
            return
        z = bottom + self.w.hand_above_bottom
        self._move_hand(np.array([self.w.cmd_hand[0], self.w.cmd_hand[1], max(self.w.cmd_hand[2], z)]))
        self._move_hand(np.array([HUG, 0.0, max(self.w.cmd_hand[2], z)]))
        self._move_hand(np.array([HUG, 0.0, z]))

    def _follow(self, waypoints: list[np.ndarray]) -> Result:
        yaw = self.w.cmd_base[2]
        target = self.w.cmd_base[:2].copy()
        for wp in waypoints[1:]:
            stuck = 0.0
            while np.linalg.norm(wp - target) > 1e-4:
                d = wp - target
                target = target + d * min(1.0, SPEED * CTRL_DT / np.linalg.norm(d))
                self.w.set_base(target, yaw)
                self.w.step(CTRL_DT)
                lag = np.linalg.norm(self.w.base()[0] - target)
                stuck = stuck + CTRL_DT if lag > 0.25 else 0.0
                if stuck > 1.0:
                    mine = [g for g, o in self.w.owner.items() if o == "agent"] + (self.w.held.geoms if self.w.held else [])
                    held = {self.w.held.label} if self.w.held else set()
                    who = sorted({o for o, _ in self.w.touching(mine)} - {"agent", "floor"} - held)
                    self.w.set_base(self.w.base()[0], yaw)
                    why = f"blocked by {', '.join(who)}" if who else "could not keep up with the planned path (nothing touched)"
                    return Result(False, why, {"at": self.w.base()[0].round(2).tolist()})
        self._wait(2.0, lambda: np.linalg.norm(self.w.base()[0] - target) < 0.02)
        return Result(True, "arrived")

    def _grid(self, bottom: float = CARRY_HEIGHTS[0]) -> FloorGrid:
        """Where the body can stand - carrying the held thing with its bottom `bottom` m high
        (the band from there up to the top of the hand on it must be free too)."""
        carry = None
        if self.w.held is not None:
            e = self.w.extent(self.w.held)
            across = float(np.linalg.norm(e["xy_max"] - e["xy_min"]))
            if self.w.held.info["physics"]["hardness"] < 0.1:
                # soft things (a plastic bag, a blanket, a pillow) fold against the chest when carried:
                # they don't stick out as the rigid box they were estimated as
                across = min(across, SOFT_CARRY_M)
            stick_out = HUG + 0.5 * across + 0.03
            carry = (bottom - 0.02, bottom + self.w.hand_above_bottom + HAND_HALF[2] + 0.02, stick_out)
        return FloorGrid(self.w, carry=carry)

    def _plan(self, region_of: Callable[[FloorGrid], np.ndarray], reach: float,
              extra: Callable[[FloorGrid], np.ndarray] | None = None) -> tuple[list[np.ndarray] | None, float]:
        """A path to where `region` is within `reach`: ([] = already there, None = none) and the
        carrying height that made it possible (tried high first, then lower)."""
        heights = CARRY_HEIGHTS if self.w.held is not None else CARRY_HEIGHTS[:1]
        for bottom in heights:
            grid = self._grid(bottom)
            goal = grid.near(region_of(grid), reach)
            if extra is not None:
                goal &= extra(grid)
            start = self.w.base()[0]
            if goal[grid.cell(start)] and grid.free[grid.cell(start)]:
                return [], bottom
            path = grid.plan(start, goal)
            if path is not None:
                return path, bottom
        return None, heights[0]

    def _go_near(self, region_of: Callable[[FloorGrid], np.ndarray], reach: float, what: str,
                 extra: Callable[[FloorGrid], np.ndarray] | None = None) -> Result:
        """Walk to the nearest spot from which `region` is within `reach` of the body centre."""
        if self.w.held is None:
            self._carry_pose()
        path, bottom = self._plan(region_of, reach, extra)
        if self.w.held is not None:
            self._carry_pose(bottom)
        if path == []:
            return Result(True, f"already next to {what}", {"walked_m": 0.0})
        if path is None:
            return Result(False, f"no free path to {what} (the body is {2 * RADIUS:.1f} m wide)")
        walked = float(sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])))
        r = self._follow(path)
        if not r.ok:
            return Result(False, f"on the way to {what}: {r.message}", r.facts)
        return Result(True, f"walked {walked:.1f} m to {what}", {"walked_m": round(walked, 2)})

    # ---------------------------------------------------------------- skills
    def walk_to(self, t: Thing) -> Result:
        r = self._go_near(lambda g: g.region(self.w.extent(t)["hull"]), REACH[1] - 0.2, t.label)
        if not r.ok and r.message.startswith("no free path"):
            return self._closest_to(t)
        if not r.ok:
            return r
        self._turn_to(self.w.extent(t)["center"])
        e = self.w.extent(t)
        gap = cv2.pointPolygonTest(e["hull"].astype(np.float32), tuple(map(float, self.w.base()[0])), True)
        r.facts["distance_m"] = round(max(0.0, -gap), 2)
        r.ok = -gap <= REACH[1]
        if not r.ok:
            r.message = f"ended {-gap:.2f} m from {t.label}: out of reach"
        return r

    def _closest_to(self, t: Thing) -> Result:
        """Nowhere within arm's reach of `t` (a window behind a dresser): walk to the closest spot
        the body can reach, and say what stands in between."""
        self._carry_pose()
        grid = self._grid()
        reach = grid.reach_map(self.w.base()[0])
        if reach is None:
            return Result(False, "the body is stuck")
        dist_to = cv2.distanceTransform((grid.region(self.w.extent(t)["hull"]) == 0).astype(np.uint8), cv2.DIST_L2, 5) * CELL
        ok = np.isfinite(reach[0]) & reach[2]
        if not ok.any():
            return Result(False, f"no free path to {t.label}")
        cell = tuple(int(v) for v in np.unravel_index(np.argmin(np.where(ok, dist_to, np.inf)), ok.shape))
        r = self._follow(grid.path_to(reach[1], cell))
        if not r.ok:
            return r
        self._turn_to(self.w.extent(t)["center"])
        me = self.w.base()[0]
        tc = self.w.extent(t)["center"][:2]
        between = sorted({o.label for o in self.w.things.values() if o is not t and o.body is None and any(
            cv2.pointPolygonTest(h.astype(np.float32), tuple(map(float, me + (tc - me) * k)), False) >= 0
            for h, bottom, top in self.w.boxes(o) for k in np.linspace(0.2, 0.9, 8))})
        gap = float(dist_to[cell])
        return Result(True, f"walked as close to {t.label} as the room allows: {gap:.1f} m away"
                            + (f", {', '.join(between)} in between" if between else ""),
                      {"distance_m": round(gap, 2), "in_between": between})

    def pick_up(self, t: Thing) -> Result:
        if self.w.held is t:
            return Result(True, f"already holding {t.label}")
        if self.w.held is not None:
            return Result(False, f"hands full: holding {self.w.held.label}")
        r = self._go_near(lambda g: g.region(self.w.extent(t)["hull"]), REACH[1] - 0.15, t.label)
        if not r.ok and r.message.startswith("no free path") and reach_at(self.w.extent(t)["zmax"]) > REACH[1]:
            # can't get close enough (it lies deep on a low bed): stand at the edge and bend over
            r = self._go_near(lambda g: g.region(self.w.extent(t)["hull"]), reach_at(self.w.extent(t)["zmax"]) - 0.15, t.label)
        if not r.ok:
            return r
        e = self.w.extent(t)
        top = e["zmax"]
        xy = self._grasp_xy(t)
        if xy is None:
            on = sorted(o.label for o in self.w.things.values() if o is not t and t.label in self.w.resting_on(o))
            return Result(False, f"can't get a hand on {t.label}: its top is covered ({', '.join(on) or 'something'} on it)")
        grasp = np.array([*xy, top])
        self._turn_to(grasp)
        if not self._room_above(xy, top, exclude=t):
            # something right above it (a bag hanging on a loft bed's rail): take it from the side
            return self._take_from_side(t)
        local = self.w.to_local(grasp)
        if not (REACH[0] <= local[0] <= reach_at(top) and SIDE[0] <= local[1] <= SIDE[1]):
            return Result(False, f"{t.label} is out of the arm's reach", {"hand_target_body_frame": local.round(2).tolist()})
        if top + 0.02 > LIFT[1]:
            return Result(False, f"{t.label} is too high ({top:.2f} m; the hand reaches {LIFT[1]} m)")
        return self._grip_and_lift(t, local, top)

    def _exposed_face(self, t: Thing) -> tuple[np.ndarray, np.ndarray, float] | None:
        """The side of `t` a hand can come in from: (middle of that face, its outward direction,
        grip height). The grip goes below anything overhead in the way in (a loft bed's top over a
        hanging bag); the corridor in front must be clear at that height and the body must be able
        to stand on that side within reach. The nearest such face wins."""
        w = self.w
        e = w.extent(t)
        hull = e["hull"]
        grid = FloorGrid(w, ignore={t.label})
        me = w.base()[0]
        boxes = [(h.astype(np.float32), bottom, top) for o in w.things.values() if o is not t
                 for h, bottom, top in w.boxes(o)]
        best, best_d = None, 1e9
        for a, b in zip(hull, np.roll(hull, -1, axis=0)):
            edge = b - a
            if np.linalg.norm(edge) < 0.05:
                continue
            n = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
            mid = 0.5 * (a + b)
            if (mid - e["center"][:2]) @ n < 0:
                n = -n                                              # outward
            corridor = [mid + n * k for k in (0.03, 0.08, 0.15, 0.25, 0.35)]
            near = lambda h: any(cv2.pointPolygonTest(h, (float(q[0]), float(q[1])), True) > -HAND_HALF[0] for q in corridor)  # noqa: E731
            over = [bottom for h, bottom, top in boxes if e["zmin"] < bottom < e["zmax"] + 0.05 and near(h)]
            z = min(0.5 * (e["zmin"] + e["zmax"]), min(over, default=9.0) - HAND_HALF[2] - 0.01)
            if z < e["zmin"] + 0.01:
                continue
            if any(bottom < z + HAND_HALF[2] and top > z - HAND_HALF[2] and near(h) for h, bottom, top in boxes):
                continue
            # somewhere to stand on that side, within arm's reach of the face
            spots = grid.near(self._point_mask(grid, mid), REACH[1] - 0.25) & self._side_mask(grid, mid, n) & grid.free
            if not spots.any():
                continue
            d = float(np.linalg.norm(mid + n * 0.6 - me))
            if d < best_d:
                best, best_d = (mid, n, float(z)), d
        return best

    def _take_from_side(self, t: Thing) -> Result:
        """Grip `t` from its exposed side, then draw it back towards the body (off its hook / out
        from under what is above it). Done if it came away and touches nothing it was on."""
        w = self.w
        e = w.extent(t)
        side = self._exposed_face(t)
        if side is None:
            return Result(False, f"no side of {t.label} a hand can get to (something is in the way on every side)")
        face, out, z = side
        r = self._go_near(lambda g: self._point_mask(g, face + out * 0.6), 0.2, f"the open side of {t.label}")
        if not r.ok:
            r = self._go_near(lambda g: self._point_mask(g, face), REACH[1] - 0.25, f"the open side of {t.label}",
                              extra=lambda g: self._side_mask(g, face, out))
            if not r.ok:
                return r
        self._turn_to(face)
        failed = self._hand_onto_side(t, face, out, z)
        if failed is not None:
            return failed
        before = w.extent(t)["center"].copy()
        below0 = w.resting_on(t)
        w.grip(t)
        h = w.cmd_hand
        self._move_hand(np.array([max(REACH[0], h[0] - 0.25), h[1], h[2] - 0.03]), speed=0.25)
        self._wait(0.5)
        moved = float(np.linalg.norm(w.extent(t)["center"][:2] - before[:2]))
        still = [x for x, _ in w.touching(t.geoms) if x not in ("agent", t.label)]
        mass = w.mass(t) or 0.0
        if moved < 0.1 or (t.hangs_on and t.hangs_on in still):
            w.release()
            self._wait(0.5)
            return Result(False, f"could not draw {t.label} away (moved {moved * 100:.0f} cm; still touching {', '.join(still) or 'nothing'})",
                          {"moved_cm": round(moved * 100, 1)})
        source = f"took it off {w.unhooked}" if w.unhooked else f"from {', '.join(below0) or 'the air'}"
        return Result(True, f"picked up {t.label} ({mass:.1f} kg) from the side, {source}: drew it {moved * 100:.0f} cm away",
                      {"moved_cm": round(moved * 100, 1), "mass_kg": round(mass, 2), "unhooked_from": w.unhooked, "grip": "side"})

    def _grasp_xy(self, t: Thing, prefer: np.ndarray | None = None) -> np.ndarray | None:
        """Where to put the hand on the top of `t`: a spot not covered by things lying on it, as
        close as possible to `prefer` (default: the centre if within reach, else the near side)."""
        w = self.w
        hull, top = w.top_face(t)
        me = w.base()[0]
        centre = hull.mean(axis=0)
        if prefer is None:
            prefer = centre if np.linalg.norm(centre - me) <= REACH[1] - 0.15 else me
        covers = [w.extent(o)["hull"].astype(np.float32) for o in w.things.values()
                  if o is not t and abs(w.extent(o)["zmin"] - top) < 0.05
                  and cv2.pointPolygonTest(hull.astype(np.float32), tuple(map(float, w.extent(o)["center"][:2])), False) >= 0]
        poly = hull.astype(np.float32)
        lo, hi = hull.min(axis=0), hull.max(axis=0)
        thick = 2 * max(0.0, cv2.pointPolygonTest(poly, tuple(map(float, centre)), True))  # true width, even turned
        inset = min(0.04, 0.4 * thick)
        step = max(0.005, min(0.02, thick / 8))                    # fine enough for a bottle cap or a thin bag
        cand = [tuple(map(float, centre))] + [(float(x), float(y)) for x in np.arange(lo[0], hi[0] + 1e-6, step)
                                              for y in np.arange(lo[1], hi[1] + 1e-6, step)]
        best, best_d = None, 1e9
        for p in cand:
            if cv2.pointPolygonTest(poly, p, True) < inset:
                continue
            if float(np.hypot(p[0] - me[0], p[1] - me[1])) < REACH[0] + 0.15:   # not under its own feet (a rug)
                continue
            if any(cv2.pointPolygonTest(c, p, True) > -0.06 for c in covers):   # hand half-width + 1 cm
                continue
            d = float(np.hypot(p[0] - prefer[0], p[1] - prefer[1]))
            if d < best_d:
                best, best_d = np.array(p), d
        return best

    def _hand_onto(self, t: Thing, local: np.ndarray, top: float) -> Result | None:
        """Hand over the top of `t`, then down until it touches. None = the hand is on it."""
        w = self.w
        hz = top + 0.02 + 0.12                                   # hand bottom 12 cm above the top
        self._move_hand(np.array([ARM_IN, 0.0, max(w.cmd_hand[2], hz)]))
        self._move_hand(np.array([local[0], local[1], max(w.cmd_hand[2], hz)]))
        self._move_hand(np.array([local[0], local[1], top + 0.02 - 0.03]), speed=0.2, until=lambda: bool(w.hand_touches()))
        touched = w.hand_touches()
        if any(self._part_of(x, t) for x in touched):
            return None
        hand_bottom = w.hand()[2] - 0.02
        if not touched and abs(hand_bottom - w.extent(t)["zmax"]) <= 0.02:
            return None
        in_way = [x for x in touched if not self._part_of(x, t)]
        self._move_hand(np.array([ARM_IN, 0.0, w.cmd_hand[2] + 0.1]))
        return Result(False, f"could not get the hand onto {t.label}" + (f": {', '.join(in_way)} in the way" if in_way else ""))

    def _part_of(self, label: str, t: Thing) -> bool:
        """The thing itself, or something that rides on its body (a backpack hanging on the chair)."""
        o = self.w.things.get(label)
        return label == t.label or (o is not None and o in self.w.riders(t))

    def _room_above(self, xy: np.ndarray, top: float, exclude: Thing, clearance: float = 0.2) -> bool:
        """Is the space above a point free (up to `clearance` over `top`) - room for the hand?"""
        p = (float(xy[0]), float(xy[1]))
        for o in self.w.things.values():
            if o is exclude:
                continue
            for h, bottom, top_ in self.w.boxes(o):
                if bottom < top + clearance and top_ > top + 0.005 and \
                        cv2.pointPolygonTest(h.astype(np.float32), p, True) > -0.06:
                    return False
        return True

    def _hand_onto_side(self, t: Thing, face: np.ndarray, out: np.ndarray, z: float) -> Result | None:
        """Hand to `z` high, 15 cm outside the face of `t`, then straight in until it touches."""
        w = self.w
        start = w.to_local(np.array([*(face + out * 0.15), z]))
        end = w.to_local(np.array([*(face - out * 0.05), z]))
        self._move_hand(np.array([TUCKED, 0.0, z]))
        self._move_hand(np.array([start[0], start[1], z]))
        self._move_hand(end, speed=0.15, until=lambda: bool(w.hand_touches()))
        touched = w.hand_touches()
        if any(self._part_of(x, t) for x in touched):
            return None
        self._move_hand(np.array([ARM_IN, 0.0, w.cmd_hand[2] + 0.1]))
        others = [x for x in touched if not self._part_of(x, t)]
        return Result(False, f"could not get the hand onto {t.label} from the front" + (f": {', '.join(others)} in the way" if others else ""))

    def _grip_and_lift(self, t: Thing, local: np.ndarray, top: float, lift_by: float = 0.2) -> Result:
        """Hand onto the top, grip, lift. Checks in physics that it really rose."""
        w = self.w
        failed = self._hand_onto(t, local, top)
        if failed is not None:
            return failed
        bottom0 = w.extent(t)["zmin"]
        below0 = w.resting_on(t)
        w.grip(t)
        self._move_hand(np.array([w.cmd_hand[0], w.cmd_hand[1], w.cmd_hand[2] + lift_by]), speed=0.25)
        self._wait(0.5)
        rose = w.extent(t)["zmin"] - bottom0
        mass = w.mass(t)
        if rose < 0.08:
            w.release()
            self._wait(0.5)
            self._move_hand(np.array([ARM_IN, 0.0, w.cmd_hand[2] + 0.05]))
            load = [o for o in w.things.values() if o is not t and o.body is not None and w.rests_on(o, t.label)]
            total = (mass or 0) + sum(w.mass(o) or 0 for o in load)
            extra = f" + {len(load)} thing(s) on it = {total:.0f} kg" if load else ""
            return Result(False, f"too heavy: {t.label} ({mass:.0f} kg{extra}) rose only {rose * 100:.0f} cm - the arm lifts at most "
                                 f"{LIFT_KG:.0f} kg", {"rose_cm": round(rose * 100, 1), "mass_kg": round(mass or 0, 1),
                                                       "load_kg": round(total, 1)})
        source = f"took it off {w.unhooked}" if w.unhooked else f"from {', '.join(below0) or 'the air'}"
        return Result(True, f"picked up {t.label} ({mass:.1f} kg), {source}: lifted {rose * 100:.0f} cm",
                      {"rose_cm": round(rose * 100, 1), "mass_kg": round(mass or 0, 2), "was_on": below0,
                       "unhooked_from": w.unhooked})

    def has_room(self, t: Thing, target: Thing | None) -> bool:
        """Is there a free spot for `t` on `target`? (asked BEFORE picking it up)"""
        return bool(self._free_spots(target, t)) and not self.too_high(t, target)

    def too_high(self, t: Thing, target: Thing | None, margin: float = 0.0) -> str | None:
        """The hand must still be above the object when it touches down: surface + object <= reach
        (minus `margin`: what is comfortably in reach)."""
        if target is None:
            return None
        e = self.w.extent(t)
        above = self.w.hand_above_bottom if self.w.held is t else (e["zmax"] - e["zmin"]) + HAND_HALF[2] * 2
        top = self.w.top_face(target)[1]
        if top + above + 0.01 > LIFT[1] - margin:
            return f"{target.label} is too high: its top is at {top:.2f} m, the hand reaches {LIFT[1]:.2f} m"
        return None

    def _surface(self, target: Thing | None, held: Thing) -> dict[str, Any]:
        """Everything needed to test where `held` fits on `target` (None = the floor)."""
        w = self.w
        e = w.extent(held)
        height = e["zmax"] - e["zmin"]
        hand_above = w.hand_above_bottom if w.held is held else height + 0.02
        if target is None:
            f = w.model.geom("floor").id
            R = w.data.geom_xmat[f].reshape(3, 3)
            hs = w.model.geom_size[f][:2]
            hull = np.array([w.data.geom_xpos[f][:2] + R[:2, :2] @ (hs * np.array(sg)) for sg in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            layers = [(hull.astype(np.float32), 0.0, "floor")]
        else:
            hull, top = w.top_face(target)
            # a flat thing lying on the target (a blanket on the bed, a mouse pad on the desk) is
            # part of its surface: things can go on it
            layers = [(hull.astype(np.float32), top, target.label)]
            for o in w.things.values():
                if o is held or o is target or not o.is_surface:
                    continue
                oe = w.extent(o)
                if oe["zmax"] - oe["zmin"] < 0.08 and target.label in w.resting_on(o):
                    h, t_ = w.top_face(o)
                    layers.append((h.astype(np.float32), t_, o.label))
        names = {lab for _, _, lab in layers}
        base = min(t_ for _, t_, _ in layers)
        surface_top = max(t_ for _, t_, _ in layers)
        # room for the object with the hand on top of it (hand_above already includes its height),
        # slid in 8 cm above the surface - it is lowered near the body first
        headroom = surface_top + hand_above + HAND_HALF[2] + 0.1
        # in the way: anything in the space above the surface (pillows lying on it, a desk top over
        # the floor, the part of a chair seat under a desk top)
        blockers = [h.astype(np.float32) for o in w.things.values() if o is not held and o.label not in names
                    for h, bottom, top_ in w.boxes(o) if bottom < headroom and top_ > base + 0.005]
        return {"hull": hull, "poly": hull.astype(np.float32), "layers": layers, "blockers": blockers,
                "fp": e["hull"] - e["center"][:2], "height": height, "target": target, "held": held,
                "half": 0.5 * (e["xy_max"] - e["xy_min"]).max() + 0.02}

    def _placed(self, S: dict[str, Any], p: np.ndarray, turn: float) -> np.ndarray:
        """The held thing's footprint if put at `p` after the body turns by `turn`."""
        c_, s_ = math.cos(turn), math.sin(turn)
        return S["fp"] @ np.array([[c_, s_], [-s_, c_]]) + p

    def _fits(self, S: dict[str, Any], p: np.ndarray, turn: float) -> bool:
        """Does it fit at `p` (turned by `turn`): all of it on the surface, a hand's width from others?"""
        placed = self._placed(S, p, turn)
        if any(cv2.pointPolygonTest(S["poly"], (float(q[0]), float(q[1])), True) < 0.01 for q in placed):
            return False
        grow = np.array([[a, b] for a in (-GAP, GAP) for b in (-GAP, GAP)])
        ring = outline(np.concatenate([placed + g for g in grow])).astype(np.float32)
        return not any(cv2.intersectConvexConvex(ring, b)[0] > 1e-6 for b in S["blockers"])

    def _surface_z(self, S: dict[str, Any], p: np.ndarray) -> float:
        return max(t_ for h, t_, _ in S["layers"] if cv2.pointPolygonTest(h, (float(p[0]), float(p[1])), False) >= 0)

    def _free_spots(self, target: Thing | None, held: Thing) -> list[tuple[np.ndarray, float]]:
        """Free points on the target's top (None = the floor) where `held` fits: [(point, surface
        height)], best first. The turn the body will make is estimated from where it stands now;
        put_on() checks it again from where the body will really stand."""
        w = self.w
        S = self._surface(target, held)
        me, my_yaw = w.base()
        lo, hi = S["hull"].min(axis=0), S["hull"].max(axis=0)
        step = 0.03
        if target is None:                                         # the floor: near the body only
            lo, hi, step = np.maximum(lo, me - 1.2), np.minimum(hi, me + 1.2), 0.05
        spots = []
        for x in np.arange(lo[0], hi[0], step):
            for y in np.arange(lo[1], hi[1], step):
                p = np.array([x, y])
                d = float(np.hypot(x - me[0], y - me[1]))
                if d < RADIUS + S["half"] + 0.12:                     # not at the agent's own feet
                    continue
                if cv2.pointPolygonTest(S["poly"], (float(x), float(y)), True) < 0:
                    continue
                turn = math.atan2(y - me[1], x - me[0]) - my_yaw if w.held is held else 0.0
                if self._fits(S, p, turn):
                    spots.append((d, p, self._surface_z(S, p)))
        if target is None and spots:
            # on the floor: out of the walkway - near a wall or furniture, not in the middle of the room
            grid = FloorGrid(w, ignore={held.label})
            room = cv2.distanceTransform((~grid.solid).astype(np.uint8), cv2.DIST_L2, 5) * CELL
            spots = [(d + 1.5 * float(room[grid.cell(p)]), p, z) for d, p, z in spots]
        # spots open above (it can be lowered straight onto them) before spots under something
        # (the floor under a desk top, a bed under a loft bed: there it has to be slid in)
        reach_top = CARRY_HEIGHTS[0] + S["height"]
        open_above = lambda p, z: self._column_clear(S["fp"] + p, z + S["height"], reach_top, held, ignore=target)  # noqa: E731
        spots.sort(key=lambda t: (not open_above(t[1], t[2]), t[0]))
        return [(p, z) for _, p, z in spots]

    def put_on(self, t: Thing, target: Thing | None) -> Result:
        """Put the held thing `t` on `target` (None = the floor)."""
        w = self.w
        where = target.label if target else "the floor"
        if w.held is not t:
            return Result(False, f"not holding {t.label}")
        high = self.too_high(t, target)
        if high:
            return Result(False, high)
        spots = self._free_spots(target, t)
        if not spots:
            return Result(False, f"no free space on {where} for {t.label}")
        # The best spot the body can actually get to (the chair may block the desk front) - checked
        # with the turn it will really make from where it will stand, and with a clear way down for
        # the object: lowered beside the body, or straight down onto the spot.
        S = self._surface(target, t)
        here_local = w.to_local(w.extent(t)["center"])               # where it is held, in the body frame
        me, yaw_now = w.base()
        maps = []                                                    # one walking-distance map per carrying height
        for bottom in CARRY_HEIGHTS:
            grid = self._grid(bottom)
            maps.append((bottom, grid, grid.reach_map(me)))
        chosen = []
        for lean in (False, True):              # stand close if possible; bend over a low surface only if not
            for p, top in spots[:400]:
                if lean and reach_at(top) <= REACH[1]:
                    continue
                arm = (reach_at(top) if lean else REACH[1]) - 0.3
                for bottom, grid, reach in maps:
                    if reach is None:
                        continue
                    cell = grid.nearest(reach, grid.near(self._point_mask(grid, p), arm))
                    if cell is None:
                        continue
                    stand = grid.xy(cell)
                    face = math.atan2(p[1] - stand[1], p[0] - stand[0])
                    turn = face - yaw_now
                    if not self._fits(S, p, turn):
                        continue
                    c_, s_ = math.cos(face), math.sin(face)
                    beside = stand + np.array([c_ * here_local[0] - s_ * here_local[1], s_ * here_local[0] + c_ * here_local[1]])
                    if not (self._column_clear(self._placed(S, beside, turn), top + 0.01, bottom, t)
                            or self._column_clear(self._placed(S, p, turn), top + 0.01, bottom, t, ignore=target)):
                        continue
                    chosen.append((p, top))
                    break
                if len(chosen) >= 3:
                    break
            if chosen:
                break
        if not chosen:
            return Result(False, f"the free space on {where} ({len(spots)} spots where {t.label} fits) can't be reached "
                                 f"from anywhere the body can stand, or not with a clear way down")
        last = Result(False, f"could not put {t.label} on {where}")
        for p, top in chosen:                                        # the best first; the next if this one fails
            r = self._place_at(t, target, where, p, top)
            if r is not None:
                return r
            last = Result(False, f"no clear way to lower {t.label} onto {where}")
        return last

    def _place_at(self, t: Thing, target: Thing | None, where: str, p: np.ndarray, top: float) -> Result | None:
        """Walk to `p`, put `t` down there, check it. None = no clear way to lower it here (try another spot)."""
        w = self.w
        r = self._go_near(lambda g: self._point_mask(g, p), REACH[1] - 0.3, f"a free spot on {where}")
        if not r.ok and r.message.startswith("no free path") and reach_at(top) > REACH[1]:
            r = self._go_near(lambda g: self._point_mask(g, p), reach_at(top) - 0.3, f"a free spot on {where}")
        if not r.ok:
            return r
        self._turn_to(p)
        # the hand holds the object where its top was free (maybe off-centre): steer the OBJECT's
        # centre onto the spot, i.e. the hand to the spot + the hand's offset from that centre
        offset = w.hand()[:2] - w.extent(t)["center"][:2]
        local = w.to_local(np.array([p[0] + offset[0], p[1] + offset[1], 0.0]))
        if not (REACH[0] <= local[0] <= reach_at(top) and SIDE[0] <= local[1] <= SIDE[1]):
            return Result(False, f"the free spot on {where} is out of reach", {"spot": p.round(2).tolist()})
        # Either lower it close to the body to just above the surface and slide it over the spot (it
        # may have to go in under something, like a bed under a loft bed), or - if lowering there
        # would land it on something (a desk next to the body) - move it over the spot first.
        above = top + w.hand_above_bottom + 0.08
        e = w.extent(t)
        here = e["hull"]
        there = here + (np.array(p) - e["center"][:2])
        if self._column_clear(here, top + 0.01, e["zmin"], t):
            self._move_hand(np.array([w.cmd_hand[0], w.cmd_hand[1], above]))
            self._move_hand(np.array([local[0], local[1], above]))
        elif self._column_clear(there, top + 0.01, e["zmin"], t, ignore=target):
            self._move_hand(np.array([local[0], local[1], w.cmd_hand[2]]))
            self._move_hand(np.array([local[0], local[1], above]))
        else:
            return None
        ground = lambda: any(o != "agent" for o, _ in w.touching(t.geoms))  # noqa: E731
        self._move_hand(np.array([local[0], local[1], top + w.hand_above_bottom + 0.005]), speed=0.15, until=ground)
        w.release()
        self._move_hand(np.array([w.cmd_hand[0], w.cmd_hand[1], w.cmd_hand[2] + 0.1]))
        self._move_hand(np.array([ARM_IN, 0.0, w.cmd_hand[2]]))
        self._wait(1.0)
        now = w.resting_on(t)
        e = w.extent(t)
        off = float(np.linalg.norm(e["center"][:2] - p))
        facts = {"rests_on": now, "cm_from_chosen_spot": round(off * 100, 1)}
        ok = (target is None and "floor" in now) or (target is not None and w.rests_on(t, target.label))
        if ok:
            return Result(True, f"put {t.label} on {where} (it rests on {', '.join(now)}, {off * 100:.0f} cm from the chosen spot)", facts)
        return Result(False, f"{t.label} did not stay on {where}: it rests on {', '.join(now) or 'nothing'}", facts)

    def _column_clear(self, footprint: np.ndarray, z_lo: float, z_hi: float, held: Thing,
                      ignore: Thing | None = None) -> bool:
        """Is the space below/above a footprint free between two heights (room to lower a thing)?"""
        poly = footprint.astype(np.float32)
        for o in self.w.things.values():
            if o is held or o is ignore:
                continue
            for h, bottom, top in self.w.boxes(o):
                if bottom < z_hi and top > z_lo and cv2.intersectConvexConvex(poly, h.astype(np.float32))[0] > 1e-5:
                    return False
        return True

    def _point_mask(self, g: FloorGrid, p: np.ndarray) -> np.ndarray:
        m = np.zeros(g.shape, np.uint8)
        m[g.cell(p)] = 1
        return m

    def pull_out(self, t: Thing, under: Thing | None) -> Result:
        """Grip `t` (a chair) on its exposed side and walk backwards out of `under` (the desk)."""
        w = self.w
        if w.held is not None:
            return Result(False, f"hands full: holding {w.held.label}")
        e = w.extent(t)
        need = 0.45                                             # how far to pull it
        if under is not None:
            # out of the nearest edge of the furniture that has open floor beyond it, far enough
            # to bring the thing all the way out
            ue = w.extent(under)
            box = under.record["Boundary"]["box_m"]                 # furniture doesn't move: its record is exact
            yaw = math.radians(box["yaw_deg"])
            axes = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
            half = np.array(box["size"][:2]) / 2
            local = axes @ (e["center"][:2] - ue["center"][:2])
            grid = FloorGrid(w, ignore={t.label})
            options = []
            for k in (0, 1):
                for sign in (1.0, -1.0):
                    d = axes[k] * sign
                    own = float(((e["hull"] - e["center"][:2]) @ d).max())       # its own half length along d
                    to_edge = half[k] - sign * local[k]                         # from its centre to the edge
                    beyond = ue["center"][:2] + d * (half[k] + 0.6)             # where the agent would stand
                    if grid.free[grid.cell(beyond)] or grid.free[grid.cell(beyond + d * 0.2)]:
                        options.append((to_edge, d, to_edge + own + 0.05))
            if not options:
                return Result(False, f"no open side to pull {t.label} out from under {under.label}")
            _, out, need = min(options, key=lambda o: o[0])
            need = float(np.clip(need, 0.3, 1.0))
        else:
            d = e["center"][:2] - w.base()[0]
            out = -d / max(np.linalg.norm(d), 1e-6)            # towards the agent
        stick = float(((e["hull"] - e["center"][:2]) @ out).max())  # how far its outline reaches along `out`
        edge = e["center"][:2] + out * max(0.0, stick - 0.08)       # the exposed edge
        grasp_xy = self._grasp_xy(t, prefer=edge)
        if grasp_xy is None:
            grasp_xy = edge
        face = e["center"][:2] + out * stick
        top = e["zmax"]
        from_top = self._room_above(grasp_xy, top, exclude=t)
        if from_top:
            side = lambda g: self._side_mask(g, e["center"][:2], out)  # noqa: E731
            r = self._go_near(lambda g: self._point_mask(g, grasp_xy), REACH[1] - 0.3, t.label, extra=side)
        else:   # stand square in front of the face it will be pulled out of
            stand = face + out * 0.6
            r = self._go_near(lambda g: self._point_mask(g, stand), 0.15, f"the front of {t.label}")
            if not r.ok:
                side = lambda g: self._side_mask(g, e["center"][:2], out)  # noqa: E731
                r = self._go_near(lambda g: self._point_mask(g, face), REACH[1] - 0.3, t.label, extra=side)
        if not r.ok:
            return r
        self._turn_to(face if not from_top else grasp_xy)
        if from_top:
            failed = self._hand_onto(t, w.to_local(np.array([*grasp_xy, top])), top)
            grip = "top"
        else:   # tucked in under the desk top: take it from the front, like a person does
            failed = self._hand_onto_side(t, face, out, top - 0.08)
            grip = "front"
        if failed is not None:
            return failed
        w.grip(t)
        start = w.extent(t)["center"][:2].copy()
        plans, why = self._pull_moves(t, out, need)          # a desk leg in the way: sideways first
        grid = FloorGrid(w, ignore={t.label})
        moves: list[np.ndarray] = []
        for plan in plans[:5]:
            # act and watch: if the chair stops following (the arm stretches instead), it is stuck -
            # step back to where the arm had its length and try the next way
            moves, stuck = plan, False
            for move in plan:
                xy, yaw = w.base()
                length = float(np.linalg.norm(move))
                d = move / max(length, 1e-9)
                while length > 0.1 and not grid._clear(xy, xy + d * length):   # room for the body behind it
                    length -= 0.05
                c0 = w.extent(t)["center"][:2].copy()
                for i in range(1, 31):
                    w.set_base(xy + d * length * i / 30, yaw)
                    w.step(length / SPEED / 30)
                self._wait(0.5)
                gone = float((w.extent(t)["center"][:2] - c0) @ d)
                if gone < 0.5 * length:
                    w.set_base(xy + d * max(gone, 0.0), yaw)
                    self._wait(0.8)
                    stuck = True
                    break
            if not stuck:
                break
        self._wait(0.5)
        w.release()
        self._move_hand(np.array([ARM_IN, 0.0, w.cmd_hand[2] + 0.1]))
        self._wait(1.0)
        moved = float(np.linalg.norm(w.extent(t)["center"][:2] - start))
        tilt = self._tilt(t)
        facts = {"moved_cm": round(moved * 100, 1), "tilt_deg": round(tilt, 1), "grip": grip,
                 "moves": "sideways, then out" if len(moves) > 1 else "straight out"}
        if under is not None:
            facts["still_under_pct"] = round(100 * self._overlap(t, under), 0)
        ok = bool(moved > 0.2 and tilt < 20 and facts.get("still_under_pct", 0) < 50)
        msg = f"pulled {t.label} {moved * 100:.0f} cm" + (f" out from under {under.label}" if under else "")
        fail = f"{t.label} moved only {moved * 100:.0f} cm (tilt {tilt:.0f} deg)" + (f": {why}" if why else "")
        return Result(ok, msg if ok else fail, facts)

    def push(self, t: Thing) -> Result:
        w = self.w
        if w.held is not None:
            return Result(False, f"hands full: holding {w.held.label}")
        r = self._go_near(lambda g: g.region(w.extent(t)["hull"]), REACH[1] - 0.3, t.label)
        if not r.ok:
            return r
        e = w.extent(t)
        self._turn_to(e["center"])
        xy = self._grasp_xy(t)
        if xy is None:
            return Result(False, f"can't get a hand on {t.label}: its top is covered")
        local = w.to_local(np.array([*xy, e["zmax"]]))
        failed = self._hand_onto(t, local, e["zmax"])
        if failed is not None:
            return failed
        w.grip(t)
        start = e["center"][:2].copy()
        xy, yaw = w.base()
        ahead = np.array([math.cos(yaw), math.sin(yaw)])
        for i in range(1, 21):
            w.set_base(xy + ahead * 0.3 * i / 20, yaw)
            w.step(0.3 / SPEED / 20)
        self._wait(0.8)
        w.release()
        self._move_hand(np.array([ARM_IN, 0.0, w.cmd_hand[2] + 0.1]))
        self._wait(0.8)
        moved = float(np.linalg.norm(w.extent(t)["center"][:2] - start))
        return Result(bool(moved > 0.15), f"pushed {t.label} {moved * 100:.0f} cm", {"moved_cm": round(moved * 100, 1)})

    def drop(self) -> Result:
        t = self.w.release()
        if t is None:
            return Result(False, "not holding anything")
        self._wait(1.5)
        return Result(True, f"dropped {t.label}: it landed on {', '.join(self.w.resting_on(t)) or 'nothing'}",
                      {"rests_on": self.w.resting_on(t)})

    # ---------------------------------------------------------------- measuring
    def _fixed_near(self, t: Thing, margin: float = 0.02) -> list[np.ndarray]:
        """Outlines of fixed boxes at `t`'s height (desk legs, walls of furniture), grown by `margin`
        (a chair rubbing along a desk leg jams by friction even without overlapping)."""
        riders = self.w.riders(t)
        zs = np.concatenate([self.w.corners(o)[:, 2] for o in [t, *riders]])
        lo, hi = float(zs.min()) + 0.02, float(zs.max()) - 0.02
        grow = np.array([[a, b] for a in (-margin, margin) for b in (-margin, margin)])
        return [(o.label, outline(np.concatenate([h + g for g in grow])).astype(np.float32))
                for o in self.w.things.values() if o.body is None and o is not t and o not in riders
                for h, bottom, top in self.w.boxes(o) if bottom < hi and top > lo]

    def _slides(self, hull: np.ndarray, move: np.ndarray, fixed: list[tuple[str, np.ndarray]]) -> str | None:
        """Can this footprint slide by `move` without running into a fixed box? None = yes, else
        what it runs into. Something it already touches is fine if it moves away from it."""
        start = [cv2.intersectConvexConvex(hull.astype(np.float32), f)[0] for _, f in fixed]
        for k in np.linspace(0.1, 1.0, 10):
            moved = (hull + move * k).astype(np.float32)
            for (label, f), s0 in zip(fixed, start):
                a = cv2.intersectConvexConvex(moved, f)[0]
                if a > 1e-6 and (s0 <= 1e-6 or a >= s0 - 1e-6):
                    return label
        return None

    def _pull_moves(self, t: Thing, out: np.ndarray, dist: float) -> tuple[list[list[np.ndarray]], str]:
        """Ways to get `t` out, best first: straight; first sideways along the edge (a desk leg in
        front of its corner), then straight; at an angle. Each move is checked against the fixed
        boxes at its height. Returns (plans, "" or why none is clear)."""
        hull, fixed = self.w.body_outline(t), self._fixed_near(t)    # what hangs on it moves too
        straight = out * dist
        side = np.array([-out[1], out[0]])
        plans = [[straight]]
        plans += [[side * sign * s, straight] for s in np.arange(0.04, 0.31, 0.03) for sign in (1.0, -1.0)]
        for deg in (15, -15, 30, -30, 45, -45):
            a = math.radians(deg)
            plans.append([np.array([out[0] * math.cos(a) - out[1] * math.sin(a), out[0] * math.sin(a) + out[1] * math.cos(a)]) * dist])

        def clear(h: np.ndarray, fx: list[tuple[str, np.ndarray]], plan: list[np.ndarray]) -> str | None:
            for move in plan:
                hit = self._slides(h, move, fx)
                if hit:
                    return hit
                h = h + move
            return None

        first_hit = clear(hull, fixed, plans[0])
        good = [plan for plan in plans if clear(hull, fixed, plan) is None]
        if good:
            return good, ""
        why = f"it catches on {first_hit}"
        riders = self.w.riders(t)
        if riders:                                          # would it come out without what hangs on it?
            alone = self.w.extent(t)["hull"]
            if any(clear(alone, fixed, plan) is None for plan in plans):
                names = ", ".join(r.label for r in riders)
                why = f"{names} hanging on it catches on {first_hit} - take it off first"
        return [[straight]], why                            # nothing is clear: try anyway, physics decides

    def _side_mask(self, g: FloorGrid, center: np.ndarray, out: np.ndarray) -> np.ndarray:
        rows, cols = np.indices(g.shape)
        xy = g.origin + (np.stack([cols, rows], -1) + 0.5) * CELL
        return ((xy - center) @ out) > 0.15

    def _tilt(self, t: Thing) -> float:
        if t.body is None:
            return 0.0
        R = self.w.data.xmat[t.body].reshape(3, 3)
        return math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))

    def _overlap(self, a: Thing, b: Thing) -> float:
        """Share of a's footprint inside b's footprint."""
        ea, eb = self.w.extent(a), self.w.extent(b)
        lo = np.minimum(ea["xy_min"], eb["xy_min"]) - 0.1
        n = 120
        scale = (np.maximum(ea["xy_max"], eb["xy_max"]) + 0.1 - lo).max() / n
        ma, mb = np.zeros((n + 1, n + 1), np.uint8), np.zeros((n + 1, n + 1), np.uint8)
        cv2.fillPoly(ma, [np.round((ea["hull"] - lo) / scale).astype(np.int32)], 1)
        cv2.fillPoly(mb, [np.round((eb["hull"] - lo) / scale).astype(np.int32)], 1)
        return float((ma & mb).sum()) / max(1, ma.sum())

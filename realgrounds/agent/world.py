"""The agent's world: a room built by the pipeline, plus a body that can act in it.

WHAT THIS FILE DOES (in plain words)
    World(room_dir) loads what lab/room_world.py made (room_mujoco.xml + objects.json) and adds
    the agent's body:

      * a base that walks (slides over the floor) and turns, 1.75 m tall; it steps over things
        lower than 20 cm (shoes) - its legs are drawn but don't collide;
      * one arm with a hand: forward/back, left/right, up/down;
      * grasping = a "weld" between hand and object, switched on only when the hand touches the
        object (the usual simplification in robot simulators: no finger physics);
      * the arm's lifting FORCE is limited (LIFT_KG), so "too heavy" is decided by physics, not
        by a rule;
      * things that HANG (a backpack on a chair, a bag on a bed rail) get their own body, held by
        a "hang" weld; taking one switches that weld off (unhooking).

    It also answers questions about the CURRENT physics state - where an object is, its top,
    what it rests on (from the contacts) - so what the agent knows stays true after it acts.

WHERE IT FITS
    skills.py moves this body; commands.py turns sentences into skills; run.py runs task lists.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
LAB = PKG  # (name kept from the research repo)
CATALOG: dict[str, dict[str, Any]] = yaml.safe_load((PKG / "catalog.yaml").read_text())["objects"]

# --- the body (metres, kilograms) ---------------------------------------------------------
RADIUS = 0.2                         # torso radius
HEIGHT = 1.75                        # top of the head: anything lower than this is in the way
STEP = 0.2                           # things lower than this are stepped over (shoes), not walked around
SHOULDER = np.array([0.3, 0.0, 1.0])  # hand rest position in the body frame (forward, left, up)
REACH = (0.15, 0.95)                 # how far in front of the body centre the hand can be
LEAN = 0.2                           # ...plus this much when the hand works low: the body bends at the hips
LEAN_BELOW = 1.2                     # ...below this height (over a low bed, into a low shelf); only when it can't get closer
SIDE = (-0.35, 0.35)                 # how far left/right of the body axis
LIFT = (0.02, 1.95)                  # how high above the floor
LIFT_KG = 15.0                       # heaviest load the arm can lift (its force limit)
HAND_HALF = np.array([0.05, 0.05, 0.02])  # hand box half sizes
HAND_KG = 1.0


def reach_at(z: float) -> float:
    """How far in front of the body centre the hand reaches at height z (further when bending over low things)."""
    return REACH[1] + (LEAN if z < LEAN_BELOW else 0.0)


def _agent_xml(x: float, y: float) -> str:
    """The body as MuJoCo XML, standing at (x, y). Joints are relative to the rest pose."""
    r = (REACH[0] - SHOULDER[0], REACH[1] + LEAN - SHOULDER[0])
    z = (LIFT[0] - SHOULDER[2], LIFT[1] - SHOULDER[2])
    return f"""
<body name="agent" pos="{x:.3f} {y:.3f} 0">
  <joint name="agent_x" type="slide" axis="1 0 0" damping="1300"/>
  <joint name="agent_y" type="slide" axis="0 1 0" damping="1300"/>
  <joint name="agent_yaw" type="hinge" axis="0 0 1" damping="120"/>
  <geom name="agent_torso" type="capsule" fromto="0 0 {STEP + RADIUS + 0.02:.2f} 0 0 {HEIGHT - 0.3:.2f}" size="{RADIUS}" mass="70" rgba="0.85 0.2 0.2 1" group="2" contype="2" conaffinity="1"/>
  <geom name="agent_legs" type="capsule" fromto="0 0 0.05 0 0 {STEP + RADIUS:.2f}" size="0.09" mass="0" rgba="0.6 0.15 0.15 1" contype="0" conaffinity="0" group="2"/>
  <geom name="agent_head" type="sphere" pos="0 0 {HEIGHT - 0.12:.2f}" size="0.12" mass="5" rgba="0.85 0.2 0.2 1" group="2" contype="2" conaffinity="1"/>
  <geom name="agent_nose" type="box" pos="{RADIUS:.2f} 0 {HEIGHT - 0.12:.2f}" size="0.05 0.03 0.03" mass="0.1" rgba="1 1 1 1" contype="0" conaffinity="0" group="2"/>
  <body name="hand" pos="{SHOULDER[0]} {SHOULDER[1]} {SHOULDER[2]}">
    <joint name="arm_reach" type="slide" axis="1 0 0" range="{r[0]:.3f} {r[1]:.3f}" damping="250"/>
    <joint name="arm_side" type="slide" axis="0 1 0" range="{SIDE[0]} {SIDE[1]}" damping="250"/>
    <joint name="arm_lift" type="slide" axis="0 0 1" range="{z[0]:.3f} {z[1]:.3f}" damping="350"/>
    <geom name="agent_hand" type="box" size="{HAND_HALF[0]} {HAND_HALF[1]} {HAND_HALF[2]}" mass="{HAND_KG}" rgba="1 0.85 0.15 1" group="2" contype="2" conaffinity="1"/>
  </body>
</body>"""


def _actuators_xml() -> str:
    lift_force = (HAND_KG + LIFT_KG) * 9.81 * 1.15     # holds LIFT_KG with a little to spare
    return f"""
<actuator>
  <position name="go_x" joint="agent_x" kp="6000" forcerange="-1500 1500"/>  <!-- top speed 1500/1300 = 1.15 m/s -->
  <position name="go_y" joint="agent_y" kp="6000" forcerange="-1500 1500"/>
  <position name="turn" joint="agent_yaw" kp="600" forcerange="-200 200"/>
  <position name="reach" joint="arm_reach" kp="3000" forcerange="-300 300"/>
  <position name="side" joint="arm_side" kp="3000" forcerange="-300 300"/>
  <position name="lift" joint="arm_lift" kp="6000" forcerange="-{lift_force:.0f} {lift_force:.0f}"/>
</actuator>"""


@dataclass
class Thing:
    """One object of the room, as the agent knows it: its record (the agreed format) + its body."""

    index: int
    label: str                    # e.g. "keyboard #24"
    kind: str                     # catalog key, e.g. "keyboard"
    record: dict[str, Any]        # Object, Position, Material, State, Affordances, Relationships, ...
    info: dict[str, Any]          # catalog entry (support, size, ...)
    body: int | None              # free body id (it can move) - None = fixed in place
    geoms: list[int] = field(default_factory=list)
    hangs_on: str | None = None   # what it hangs on (a chair, the loft bed) ...
    hang: int | None = None       # ... held by this weld until it is taken off
    grip: int | None = None       # weld constraint id used to hold it in the hand

    @property
    def name(self) -> str:
        return self.label.split(" #")[0]

    @property
    def affordances(self) -> list[str]:
        return list(self.record.get("Affordances", []))

    @property
    def movable(self) -> bool:
        return self.record.get("State") == "movable"

    @property
    def is_surface(self) -> bool:
        """Can things be put on it?"""
        return bool(self.info.get("support")) or "place_on" in self.affordances


def outline(xy: np.ndarray) -> np.ndarray:
    """Convex outline of points seen from above, without repeated points (a box's top and bottom
    corners have the same x, y - repeats make OpenCV's polygon intersection silently return 0)."""
    pts = np.unique(np.round(np.asarray(xy, float), 5), axis=0).astype(np.float32)
    return cv2.convexHull(pts).reshape(-1, 2).astype(float)


def _mass(rec: dict[str, Any]) -> float | None:
    try:
        return float(str(rec.get("Mass", "")).split()[0])
    except ValueError:
        return None


def _free_the_hanging(world_el: ET.Element, records: list[dict[str, Any]]) -> list[tuple[str, str | None]]:
    """Give every hanging thing its own body (the room export fixes them in place or puts them
    inside the body they hang on). Returns [(its body, the body it hangs on - None = fixed furniture)];
    a "hang" weld will hold each one where it is until it is taken off."""
    hangs: list[tuple[str, str | None]] = []
    for body in list(world_el.findall("body")):                  # inside a movable body (chair)
        parent = body.get("name", "")
        if not parent.startswith("obj"):
            continue
        bpos = np.array([float(v) for v in body.get("pos").split()])
        byaw = float((body.get("euler") or "0 0 0").split()[2])
        for g in list(body.findall("geom")):
            name = g.get("name", "")
            if name == f"{parent}_g" or not name.startswith("obj"):
                continue
            child = name[:-2] if name.endswith("_g") else name
            gpos = np.array([float(v) for v in (g.get("pos") or "0 0 0").split()])
            gyaw = float((g.get("euler") or "0 0 0").split()[2])
            c, s_ = math.cos(byaw), math.sin(byaw)
            wpos = bpos + np.array([c * gpos[0] - s_ * gpos[1], s_ * gpos[0] + c * gpos[1], gpos[2]])
            body.remove(g)
            nb = ET.SubElement(world_el, "body", name=child, pos=" ".join(f"{v:.4f}" for v in wpos), euler=f"0 0 {byaw + gyaw:.4f}")
            ET.SubElement(nb, "freejoint", name=f"{child}_free")
            g.set("pos", "0 0 0")
            g.set("euler", "0 0 0")
            nb.append(g)
            hangs.append((child, parent))
    for g in list(world_el.findall("geom")):                      # fixed in place (bag on the loft bed)
        name = g.get("name", "")
        if not name.startswith("obj") or "_" not in name or not name[3:name.index("_")].isdigit():
            continue
        i = int(name[3:name.index("_")])
        if i >= len(records) or not str(records[i].get("Position", "")).lower().startswith("hanging on"):
            continue
        world_el.remove(g)
        nb = ET.SubElement(world_el, "body", name=name, pos=g.get("pos"), euler=g.get("euler") or "0 0 0")
        ET.SubElement(nb, "freejoint", name=f"{name}_free")
        for k in ("contype", "conaffinity"):
            g.attrib.pop(k, None)                                  # it is a movable thing now: it collides
        g.set("pos", "0 0 0")
        g.set("euler", "0 0 0")
        g.set("name", f"{name}_g")
        g.set("mass", f"{_mass(records[i]) or 0.5:.3f}")
        nb.append(g)
        hangs.append((name, None))
    return hangs


def _kind_of(rest: str) -> str:
    """'keyboard' / 'desk_top' / 'loft_bed_leg-1-1' / 'backpack_g' -> catalog key."""
    keys = [k for k in CATALOG if rest == k or rest.startswith(k + "_")]
    return max(keys, key=len) if keys else rest


class World:
    """The room + the agent body in MuJoCo, and questions about the current physics state."""

    def __init__(self, room_dir: Path, start_offset: tuple[float, float] = (0.0, 0.0)) -> None:
        """start_offset moves where the agent starts (robustness tests)."""
        self.room_dir = Path(room_dir)
        records = json.loads((self.room_dir / "objects.json").read_text())
        root = ET.parse(self.room_dir / "room_mujoco.xml").getroot()
        # The photo-textured surfaces are pictures only (and don't move when an object does):
        # leave them out - the agent works with the physics bodies.
        asset = root.find("asset")
        if asset is not None:
            for el in list(asset):
                if el.tag in ("mesh", "material") or (el.tag == "texture" and el.get("type") == "2d"):
                    asset.remove(el)
        world = root.find("worldbody")
        for g in list(world):
            if g.tag == "geom" and g.get("type") == "mesh":
                world.remove(g)
        old = world.find("body[@name='agent']")
        x, y = (float(v) for v in old.get("pos").split()[:2])
        x, y = x + start_offset[0], y + start_offset[1]
        world.remove(old)
        world.append(ET.fromstring(_agent_xml(x, y)))
        root.append(ET.fromstring(_actuators_xml()))
        hangs = _free_the_hanging(world, records)
        free_bodies = [b.get("name") for b in world.findall("body") if b.find("freejoint") is not None]
        eq = ET.SubElement(root, "equality")
        for name in free_bodies:
            ET.SubElement(eq, "weld", name=f"grip_{name}", body1="hand", body2=name, active="false", solref="0.01 1")
        for child, parent in hangs:
            attrs = {"name": f"hang_{child}", "active": "false", "solref": "0.01 1"}
            attrs.update({"body1": parent, "body2": child} if parent else {"body1": child})
            ET.SubElement(eq, "weld", **attrs)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.spawn = np.array([x, y])
        self.hooks: list[Callable[[], None]] = []      # called after every physics step (video)
        self.held: Thing | None = None
        self.hand_above_bottom = 0.0                   # hand height above the held thing's bottom
        self._build_things(records, hangs)
        self.act = {n: self.model.actuator(n).id for n in ("go_x", "go_y", "turn", "reach", "side", "lift")}
        self.cmd_hand = SHOULDER.copy()                # where the hand is told to be (body frame)
        self.cmd_base = np.array([x, y, 0.0])          # where the base is told to be (x, y, yaw)
        mujoco.mj_forward(self.model, self.data)
        for t in self.things.values():                 # hang things up exactly where they are
            if t.hang is not None:
                m = self.model
                self._weld_now(t.hang, m.eq_obj1id[t.hang], m.eq_obj2id[t.hang])
        self._settle_objects()

    # ---------------------------------------------------------------- set-up
    def _build_things(self, records: list[dict[str, Any]], hangs: list[tuple[str, str | None]]) -> None:
        m = self.model
        self.things: dict[str, Thing] = {}
        self.owner: dict[int, str] = {}                # geom id -> thing label / floor / wall / agent
        by_index: dict[int, Thing] = {}
        for i, rec in enumerate(records):
            by_index[i] = Thing(i, rec["Id"], "", rec, {}, None)
        for g in range(m.ngeom):
            name = m.geom(g).name
            if name.startswith("agent"):
                self.owner[g] = "agent"
            elif name.startswith("floor"):
                self.owner[g] = "floor"
            elif name.startswith("wall"):
                self.owner[g] = "wall"
            elif name.startswith("obj") and "_" in name:
                i = int(name[3:name.index("_")])
                t = by_index.get(i)
                if t is None:
                    continue
                t.kind = t.kind or _kind_of(name[name.index("_") + 1:])
                t.geoms.append(g)
                self.owner[g] = t.label
        for t in by_index.values():
            if not t.geoms:
                continue
            t.info = CATALOG.get(t.kind, {})
            try:
                t.body = m.body(f"obj{t.index}_{t.kind}").id
                t.grip = m.equality(f"grip_obj{t.index}_{t.kind}").id
            except KeyError:
                t.body = None                                  # fixed geometry
            self.things[t.label] = t
        for child, parent in hangs:
            t = by_index[int(child[3:child.index("_")])]
            t.hang = m.equality(f"hang_{child}").id
            if parent:
                t.hangs_on = by_index[int(parent[3:parent.index("_")])].label
            else:
                t.hangs_on = ((t.record.get("Relationships") or {}).get("hanging_on") or ["?"])[0]

    def _settle_objects(self, seconds: float = 0.5) -> None:
        """Let things come to rest before the agent starts (boxes are estimates)."""
        self.step(seconds)

    # ---------------------------------------------------------------- time
    def step(self, seconds: float) -> None:
        for _ in range(max(1, int(round(seconds / self.model.opt.timestep)))):
            mujoco.mj_step(self.model, self.data)
            for hook in self.hooks:
                hook()

    # ---------------------------------------------------------------- the body
    def base(self) -> tuple[np.ndarray, float]:
        """(x, y) and heading (radians) of the body now."""
        b = self.model.body("agent").id
        R = self.data.xmat[b].reshape(3, 3)
        return self.data.xpos[b][:2].copy(), math.atan2(R[1, 0], R[0, 0])

    def hand(self) -> np.ndarray:
        return self.data.xpos[self.model.body("hand").id].copy()

    def set_base(self, xy: np.ndarray, yaw: float) -> None:
        self.cmd_base = np.array([xy[0], xy[1], yaw])
        self.data.ctrl[self.act["go_x"]] = xy[0] - self.spawn[0]
        self.data.ctrl[self.act["go_y"]] = xy[1] - self.spawn[1]
        self.data.ctrl[self.act["turn"]] = yaw

    def set_hand(self, local: np.ndarray) -> np.ndarray:
        """Tell the hand where to be, in the body frame (forward, left, up); clipped to its reach."""
        local = np.array([np.clip(local[0], *REACH), np.clip(local[1], *SIDE), np.clip(local[2], *LIFT)])
        self.cmd_hand = local
        self.data.ctrl[self.act["reach"]] = local[0] - SHOULDER[0]
        self.data.ctrl[self.act["side"]] = local[1] - SHOULDER[1]
        self.data.ctrl[self.act["lift"]] = local[2] - SHOULDER[2]
        return local

    def to_local(self, p: np.ndarray) -> np.ndarray:
        """World point -> body frame (forward, left, up)."""
        xy, yaw = self.base()
        d = np.asarray(p[:2]) - xy
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], p[2]])

    def to_world(self, local: np.ndarray) -> np.ndarray:
        xy, yaw = self.base()
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([xy[0] + c * local[0] - s * local[1], xy[1] + s * local[0] + c * local[1], local[2]])

    # ---------------------------------------------------------------- objects now
    def corners(self, t: Thing) -> np.ndarray:
        """(N, 3) world corners of all of a thing's boxes, as they are NOW."""
        pts = []
        signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
        for g in t.geoms:
            if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            R = self.data.geom_xmat[g].reshape(3, 3)
            pts.append(self.data.geom_xpos[g] + (signs * self.model.geom_size[g]) @ R.T)
        return np.concatenate(pts) if pts else np.zeros((0, 3))

    def extent(self, t: Thing) -> dict[str, Any]:
        """Footprint (convex outline from above), bottom, top and centre of a thing now."""
        c = self.corners(t)
        hull = outline(c[:, :2])
        return {"hull": hull, "zmin": float(c[:, 2].min()), "zmax": float(c[:, 2].max()),
                "center": c.mean(axis=0), "xy_min": c[:, :2].min(axis=0), "xy_max": c[:, :2].max(axis=0)}

    def boxes(self, t: Thing, solid_only: bool = True) -> list[tuple[np.ndarray, float, float]]:
        """Each of a thing's boxes separately: (outline from above, bottom, top) - a desk is a top
        slab high up AND legs down to the floor. solid_only: skip boxes nothing collides with
        (string lights, a curtain)."""
        out = []
        signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
        for g in t.geoms:
            if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            if solid_only and self.model.geom_contype[g] == 0 and self.model.geom_conaffinity[g] == 0:
                continue
            R = self.data.geom_xmat[g].reshape(3, 3)
            c = self.data.geom_xpos[g] + (signs * self.model.geom_size[g]) @ R.T
            out.append((outline(c[:, :2]), float(c[:, 2].min()), float(c[:, 2].max())))
        return out

    def body_outline(self, t: Thing) -> np.ndarray:
        """Footprint of everything that moves together with `t` (a chair and the backpack on it)."""
        if t.body is None:
            return self.extent(t)["hull"]
        pts = np.concatenate([self.corners(o) for o in [t, *self.riders(t)]])
        return outline(pts[:, :2])

    def top_face(self, t: Thing) -> tuple[np.ndarray, float]:
        """The surface things can be put on: outline of its highest box (a desk's top slab) and its height."""
        best, best_top = None, -1.0
        signs = np.array([[a, b, 1] for a in (-1, 1) for b in (-1, 1)], float)
        for g in t.geoms:
            if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            R = self.data.geom_xmat[g].reshape(3, 3)
            c = self.data.geom_xpos[g] + (signs * self.model.geom_size[g]) @ R.T
            if c[:, 2].max() > best_top:
                best, best_top = c, float(c[:, 2].max())
        return outline(best[:, :2]), best_top

    def touching(self, geoms: list[int]) -> list[tuple[str, np.ndarray]]:
        """What touches these geoms now: [(owner, contact point)]."""
        mine = set(geoms)
        out = []
        for con in self.data.contact[: self.data.ncon]:
            if con.dist > 0.003:
                continue
            if con.geom1 in mine and con.geom2 not in mine:
                out.append((self.owner.get(con.geom2, "?"), np.array(con.pos)))
            elif con.geom2 in mine and con.geom1 not in mine:
                out.append((self.owner.get(con.geom1, "?"), np.array(con.pos)))
        return out

    def resting_on(self, t: Thing) -> list[str]:
        """What a thing rests on NOW, from the physics contacts under it (not from the old record)."""
        if self.held is t:
            return ["agent's hand"]
        bottom = self.extent(t)["zmin"]
        under = {who for who, p in self.touching(t.geoms) if p[2] < bottom + 0.03 and who not in ("agent", t.label)}
        return sorted(under)

    def rests_on(self, t: Thing, target: str) -> bool:
        """True if t rests on `target`, directly or on something that rests on it (a toy on the
        blanket on the bed is on the bed)."""
        direct = self.resting_on(t)
        if target in direct:
            return True
        return any(target in self.resting_on(self.things[d]) for d in direct if d in self.things)

    # ---------------------------------------------------------------- holding
    def hand_touches(self) -> list[str]:
        return sorted({who for who, _ in self.touching([self.model.geom("agent_hand").id]) if who != "agent"})

    def _weld_now(self, eq: int, b1: int, b2: int) -> None:
        """Switch on a weld that holds body b2 exactly where it is now relative to b1 (no jump)."""
        m, d = self.model, self.data
        R1 = d.xmat[b1].reshape(3, 3)
        q1c = np.array([d.xquat[b1][0], *(-d.xquat[b1][1:])])
        relq = np.zeros(4)
        mujoco.mju_mulQuat(relq, q1c, d.xquat[b2])
        m.eq_data[eq][3:6] = R1.T @ (d.xpos[b2] - d.xpos[b1])
        m.eq_data[eq][6:10] = relq
        d.eq_active[eq] = 1

    def hanging(self, t: Thing) -> bool:
        """Still hanging where it was (not taken off yet)?"""
        return t.hang is not None and bool(self.data.eq_active[t.hang])

    def riders(self, t: Thing) -> list[Thing]:
        """Things hanging on `t` right now (they move with it)."""
        return [o for o in self.things.values() if o.hangs_on == t.label and self.hanging(o)]

    def grip(self, t: Thing) -> None:
        """Weld the thing to the hand exactly where it is now (no jump). A hanging thing is taken off.

        While held it can be hugged against the body: its collision bits change so it still hits
        the room (contype 4 & room conaffinity 1) but not the agent (agent contype 2, conaffinity 1)."""
        m, d = self.model, self.data
        self._bits = {g: (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in t.geoms}
        for g in t.geoms:
            if m.geom_contype[g] or m.geom_conaffinity[g]:
                m.geom_contype[g], m.geom_conaffinity[g] = 4, 1
        self._weld_now(t.grip, m.body("hand").id, t.body)
        self.unhooked = None
        if self.hanging(t):                            # take it off its hook / chair
            d.eq_active[t.hang] = 0
            self.unhooked = t.hangs_on
        self.held = t
        self.hand_above_bottom = float(self.hand()[2] - self.extent(t)["zmin"])

    def release(self) -> Thing | None:
        t = self.held
        if t is not None:
            self.data.eq_active[t.grip] = 0
            for g, (ct, ca) in getattr(self, "_bits", {}).items():
                self.model.geom_contype[g], self.model.geom_conaffinity[g] = ct, ca
        self.held = None
        return t

    def mass(self, t: Thing) -> float | None:
        """Real mass in the simulation (with anything riding on it), else the record's."""
        if t.body is not None:
            return float(self.model.body_subtreemass[t.body])
        txt = str(t.record.get("Mass", ""))
        try:
            return float(txt.split()[0])
        except ValueError:
            return None

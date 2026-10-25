"""Room -> MuJoCo physics world, physics proofs, renders and the floor plan.

WHAT THIS FILE DOES (in plain words)
    to_mjcf()        writes the room as a MuJoCo world:
                       floor + 4 walls (seen walls are invisible colliders because the photo
                       surface shows them; walls the photo never saw are translucent),
                       static furniture as fixed geometry (a desk/table = top slab + 4 legs, so
                       a chair fits under it), movable things as free bodies with their mass,
                       resting on their support, the photo-textured 2.5D surface as visuals,
                       and an agent standing where the photographer stood.
    depenetrate()    nudges movable bodies apart if two boxes start overlapping a little.
    physics_proofs() runs the checks that make it a training ground:
                       1. SETTLE - every movable object stays on its support (no falling through,
                          no sliding off): the `supported_by` relationships hold physically.
                       2. PUSH - a 80 N pull moves the chair out from under its desk
                          (state: movable is real).
                       3. LIFT - a small object rises when lifted with more than its weight
                          (grasp/lift affordance is physically possible).
    render_room()    4 pictures: the photo's camera, a cut-away, the physics bodies, first person.

GEOM GROUPS (which picture shows what)
    0 photo surface   1 room shell (walls, floor colour)   2 agent   3 physics bodies
    4 photo surface without the ceiling (cut-away only)
    floor_plan()     top-down plan with objects, relationships, camera and agent.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .room import RoomBox, RoomObject, platform_slab



def _rgba(c, a: float = 1.0) -> str:
    return f"{c[0] / 255:.3f} {c[1] / 255:.3f} {c[2] / 255:.3f} {a}"


def _rgb_only(c) -> str:
    """Indoors there is no sky: anything not covered by the photo surface shows the wall colour."""
    return f"{c[0] / 255:.3f} {c[1] / 255:.3f} {c[2] / 255:.3f}"


def _color(o: RoomObject) -> tuple[int, int, int]:
    return (120, 140, 170) if o.info["state"] == "static" else (235, 140, 40)


def _is_free(o: RoomObject) -> bool:
    """A free body in physics (falls, slides, can be pushed); everything else is fixed geometry."""
    return not (o.info["state"] == "static" or o.info.get("platform") or o.attached_to or o.raised
                or o.support in ("unseen", "wall") or getattr(o, "inner_support", False))


def _object_xml(o: RoomObject, idx: int, others: list[RoomObject] = (),
                riders: list[tuple[int, RoomObject]] = ()) -> str:
    """MuJoCo XML for one object (static geometry or a free body).

    `riders` hang on this (free) object - a backpack on a chair: their boxes become part of its
    body, so they move with it when it is pushed."""
    cx, cy, _ = o.center
    sx, sy, sz = (float(v) for v in o.size)
    zmin, top = o.zmin, o.top
    yaw = o.yaw
    phys = o.info["physics"]
    fr = f'friction="{phys["friction"]:.2f} 0.005 0.0001"'
    soft = phys["hardness"] < 0.1
    # Static soft things (curtain, string lights) don't block anything; movable soft things
    # (blanket, plastic bag) must still land on what they lie on - they collide, but are light.
    col = ' contype="0" conaffinity="0"' if soft and o.info["state"] == "static" else ""
    name = f"obj{idx}_{o.kind}"
    if not _is_free(o):
        if (o.info.get("platform") or o.raised) and top > 0.3:
            slab = platform_slab(o)
            leg = 0.04 if o.kind != "loft_bed" else 0.07
            parts = [f'<geom name="{name}_top" type="box" pos="{cx:.3f} {cy:.3f} {top - slab / 2:.3f}" euler="0 0 {yaw:.4f}" '
                     f'size="{sx / 2:.3f} {sy / 2:.3f} {slab / 2:.3f}" rgba="{_rgba(_color(o))}" {fr} group="3"/>']
            c, s = math.cos(yaw), math.sin(yaw)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                lx = cx + a * (sx / 2 - leg / 2) * c - b * (sy / 2 - leg / 2) * s   # posts in the very corners
                ly = cy + a * (sx / 2 - leg / 2) * s + b * (sy / 2 - leg / 2) * c
                h = (top - slab) / 2
                # A post that would stand inside something under the platform (the lower bed under
                # a loft) is left out: the top is fixed geometry and doesn't need it to stay up.
                if any(q is not o and q.zmin < top - slab and q.contains_xy(np.array([lx, ly]), leg / 2) for q in others):
                    continue
                parts.append(f'<geom name="{name}_leg{a}{b}" type="box" pos="{lx:.3f} {ly:.3f} {h:.3f}" euler="0 0 {yaw:.4f}" '
                             f'size="{leg / 2:.3f} {leg / 2:.3f} {h:.3f}" rgba="{_rgba(_color(o))}" {fr} group="3"/>')
            return "\n    ".join(parts)
        return (f'<geom name="{name}" type="box" pos="{cx:.3f} {cy:.3f} {(zmin + top) / 2:.3f}" euler="0 0 {yaw:.4f}" '
                f'size="{sx / 2:.3f} {sy / 2:.3f} {max(top - zmin, 0.01) / 2:.3f}" rgba="{_rgba(_color(o))}" {fr}{col} group="3"/>')
    mass = o.info.get("mass_kg") or max(0.05, float(np.prod(o.size)) * 300)
    hz = max(top - zmin, 0.01) / 2
    pos = np.array([cx, cy, zmin + hz + 0.002])
    c, s = math.cos(yaw), math.sin(yaw)
    rider_geoms = ""
    for j, r in riders:
        d = np.array([r.center[0], r.center[1], (r.zmin + r.top) / 2]) - pos      # in the body's own frame:
        local = (d[0] * c + d[1] * s, -d[0] * s + d[1] * c, d[2])                  # undo the body's yaw
        rz = max(r.top - r.zmin, 0.01) / 2
        rmass = r.info.get("mass_kg") or 0.5
        rider_geoms += (f'<geom name="obj{j}_{r.kind}_g" type="box" pos="{local[0]:.3f} {local[1]:.3f} {local[2]:.3f}" '
                        f'euler="0 0 {r.yaw - yaw:.4f}" size="{r.size[0] / 2:.3f} {r.size[1] / 2:.3f} {rz:.3f}" '
                        f'mass="{rmass:.3f}" rgba="{_rgba(_color(r))}" group="3"/>')
    return (f'<body name="{name}" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" euler="0 0 {yaw:.4f}">'
            f'<freejoint name="{name}_free"/>'
            f'<geom name="{name}_g" type="box" size="{sx / 2:.3f} {sy / 2:.3f} {hz:.3f}" mass="{mass:.3f}" '
            f'rgba="{_rgba(_color(o))}" {fr}{col} group="3"/>{rider_geoms}</body>')


def to_mjcf(path: Path, box: RoomBox, objects: list[RoomObject], meshes: list[tuple[str, str, str]],
            floor_rgb, wall_rgb, agent_xy: tuple[float, float]) -> Path:
    """Write the room as a MuJoCo model. `meshes` = [(name, obj file, texture file)] visual layers."""
    c, s = math.cos(box.axis), math.sin(box.axis)
    U, V = np.array([c, s]), np.array([-s, c])
    mid = (box.lo + box.hi) / 2
    centre = mid[0] * U + mid[1] * V
    half = (box.hi - box.lo) / 2
    h = box.height
    wall_t = 0.05
    walls = []
    for side, (a, b) in {"u-": (-1, 0), "u+": (1, 0), "v-": (0, -1), "v+": (0, 1)}.items():
        pos = centre + a * (half[0] + wall_t) * U + b * (half[1] + wall_t) * V
        size = (wall_t, half[1] + 2 * wall_t) if a else (half[0] + 2 * wall_t, wall_t)
        # Collision wall (invisible) ...
        walls.append(f'<geom name="wall_{side}" type="box" pos="{pos[0]:.3f} {pos[1]:.3f} {h / 2:.3f}" euler="0 0 {box.axis:.4f}" '
                     f'size="{size[0]:.3f} {size[1]:.3f} {h / 2:.3f}" group="3"/>')
        # ... and what you see: a seen wall is a solid surface 1 cm behind the photo surface (fills
        # cut-outs with wall colour); an inferred wall is only a low dark outline (cut-away model).
        vis = pos + (a * U + b * V) * 0.01
        if box.sides_seen[side]:
            walls.append(f'<geom name="wallvis_{side}" type="box" pos="{vis[0]:.3f} {vis[1]:.3f} {h / 2:.3f}" euler="0 0 {box.axis:.4f}" '
                         f'size="{size[0]:.3f} {size[1]:.3f} {h / 2:.3f}" rgba="{_rgba(wall_rgb)}" contype="0" conaffinity="0" group="1"/>')
        else:
            walls.append(f'<geom name="wallvis_{side}" type="box" pos="{vis[0]:.3f} {vis[1]:.3f} 0.05" euler="0 0 {box.axis:.4f}" '
                         f'size="{size[0]:.3f} {size[1]:.3f} 0.05" rgba="0.25 0.25 0.28 1" contype="0" conaffinity="0" group="1"/>')
    assets, visuals = [], []
    for name, obj, tex in meshes:
        assets.append(f'<mesh name="m_{name}" file="{obj}" inertia="shell"/>')
        assets.append(f'<texture name="t_{name}" type="2d" file="{tex}"/>')
        assets.append(f'<material name="mat_{name}" texture="t_{name}" emission="0.45" specular="0" shininess="0"/>')
        group = 4 if name.endswith("_cut") else 0
        visuals.append(f'<geom name="v_{name}" type="mesh" mesh="m_{name}" material="mat_{name}" contype="0" conaffinity="0" group="{group}"/>')
    # Things hanging on a free object (backpack on a chair) ride along inside its body.
    labels = {o.label: o for o in objects}
    riders: dict[str, list[tuple[int, RoomObject]]] = {}
    for i, o in enumerate(objects):
        parent = labels.get(o.attached_to or "")
        if parent is not None and _is_free(parent):
            riders.setdefault(parent.label, []).append((i, o))
    riding = {o.label for group in riders.values() for _, o in group}
    bodies = [_object_xml(o, i, objects, riders.get(o.label, ())) for i, o in enumerate(objects) if o.label not in riding]
    ax, ay = agent_xy
    xml = f"""<mujoco model="pixel_world_room">
  <!-- Generated by Pixel World (lab/toolkit/room_export.py). metres; x right, y forward, z up; camera at (0, 0). -->
  <compiler angle="radian"/>  <!-- all yaw angles below are in radians -->
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><headlight ambient="0.55 0.55 0.55" diffuse="0.35 0.35 0.35"/><map znear="0.01"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="{_rgb_only(wall_rgb)}" rgb2="{_rgb_only(wall_rgb)}" width="16" height="16"/>
    {chr(10).join('    ' + a for a in assets).strip()}
  </asset>
  <worldbody>
    <light pos="{centre[0]:.2f} {centre[1]:.2f} {h + 1:.2f}" dir="0 0 -1" directional="true" castshadow="false"/>
    <geom name="floor" type="plane" pos="{centre[0]:.3f} {centre[1]:.3f} 0" euler="0 0 {box.axis:.4f}" size="{half[0]:.3f} {half[1]:.3f} 0.1" friction="0.7 0.005 0.0001" group="3"/>
    <geom name="floor_vis" type="box" pos="{centre[0]:.3f} {centre[1]:.3f} -0.008" euler="0 0 {box.axis:.4f}" size="{half[0]:.3f} {half[1]:.3f} 0.005" rgba="{_rgba(floor_rgb)}" contype="0" conaffinity="0" group="1"/>
    {chr(10).join('    ' + w for w in walls).strip()}
    {chr(10).join('    ' + v for v in visuals).strip()}
    {chr(10).join('    ' + b for b in bodies).strip()}
    <body name="agent" pos="{ax:.3f} {ay:.3f} 0.92"><freejoint name="agent_free"/>
      <geom name="agent_body" type="capsule" size="0.25 0.65" mass="70" rgba="0.9 0.2 0.2 1" group="2"/></body>
  </worldbody>
</mujoco>
"""
    path.write_text(xml, encoding="utf-8")
    return path


def depenetrate(model, data, mujoco, rounds: int = 300, max_move: float = 0.3) -> int:
    """Move free bodies out of SMALL start overlaps (boxes are estimates). Returns overlaps left.

    A body is moved at most `max_move` metres in total: a big overlap is a modelling error to fix
    upstream, and pushing further only bounces it between a wall and the furniture."""
    free = {model.jnt_bodyid[j]: model.jnt_qposadr[j] for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    start = {b: data.qpos[adr:adr + 3].copy() for b, adr in free.items()}

    def far(b: int) -> bool:
        return b in free and float(np.linalg.norm(data.qpos[free[b]:free[b] + 3] - start[b])) > max_move

    for _ in range(rounds):
        mujoco.mj_forward(model, data)
        moved = False
        for con in data.contact[: data.ncon]:
            if con.dist >= -0.002:
                continue
            b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
            if far(b1) or far(b2):
                continue
            n = np.array(con.frame[:3])
            push = -con.dist + 0.001
            if b1 in free and b2 in free and abs(n[2]) > 0.7 and push < 0.02:
                # One already rests ON the other (mouse on a mouse pad) and sinks in a few mm: lift
                # the upper one - sliding them apart sideways pushes the lower one off its desk.
                upper = b2 if data.xpos[b2][2] >= data.xpos[b1][2] else b1
                data.qpos[free[upper] + 2] += push
                moved = True
            elif b1 in free and b2 in free:
                # Two objects on the same surface overlapping (estimated boxes): slide them apart
                # ALONG the surface - never stack one onto the other.
                horiz = np.array([n[0], n[1], 0.0])
                if np.linalg.norm(horiz) < 0.3:
                    horiz = np.array([*(data.xpos[b2][:2] - data.xpos[b1][:2]), 0.0])
                horiz /= max(np.linalg.norm(horiz), 1e-9)
                data.qpos[free[b2]:free[b2] + 3] += horiz * push / 2
                data.qpos[free[b1]:free[b1] + 3] -= horiz * push / 2
                moved = True
            elif b2 in free:   # normal points from geom1 to geom2: move geom2's body along it
                data.qpos[free[b2]:free[b2] + 3] += n * push
                moved = True
            elif b1 in free:
                data.qpos[free[b1]:free[b1] + 3] -= n * push
                moved = True
        if not moved:
            break
    # Never leave a body pushed into the floor.
    for body, adr in free.items():
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
        half_h = max((model.geom_size[g][2] for g in geoms if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX), default=0.0)
        if data.qpos[adr + 2] < half_h:
            data.qpos[adr + 2] = half_h + 0.001
    mujoco.mj_forward(model, data)
    return sum(1 for con in data.contact[: data.ncon] if con.dist < -0.005)


def physics_proofs(xml: Path, objects: list[RoomObject]) -> dict[str, Any]:
    """Load the room in MuJoCo and run SETTLE, PUSH and LIFT (see the file header)."""
    try:
        import mujoco
    except ImportError:
        return {"ran": False, "reason": "mujoco not installed"}
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    left = depenetrate(model, data, mujoco)
    start_qpos = data.qpos.copy()
    bodies = {o.label: model.body(f"obj{i}_{o.kind}").id for i, o in enumerate(objects) if _is_free(o)}
    start = {k: data.xpos[b].copy() for k, b in bodies.items()}
    for _ in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    settle = {}
    for k, b in bodies.items():
        d = data.xpos[b] - start[k]
        settle[k] = {"moved_cm": round(float(np.linalg.norm(d)) * 100, 1), "dropped_cm": round(float(-d[2]) * 100, 1),
                     "stayed": bool(np.linalg.norm(d[:2]) < 0.05 and d[2] > -0.03)}
    result: dict[str, Any] = {"ran": True, "start_overlaps_left": left, "settle": settle,
                              "settle_ok": all(v["stayed"] for v in settle.values())}
    agent = model.body("agent").id
    result["agent_standing"] = bool(0.7 < data.xpos[agent][2] < 1.1)

    def run_force(label: str, force: np.ndarray, seconds: float) -> tuple[np.ndarray, float]:
        """Push one body for `seconds`, let it come to rest; return (final move, highest rise)."""
        data.qpos[:] = start_qpos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        b = bodies[label]
        p0 = data.xpos[b].copy()
        peak = 0.0
        for _ in range(int(seconds / model.opt.timestep)):
            data.xfrc_applied[b, :3] = force
            mujoco.mj_step(model, data)
            peak = max(peak, float(data.xpos[b][2] - p0[2]))
        data.xfrc_applied[b, :3] = 0
        for _ in range(int(0.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            peak = max(peak, float(data.xpos[b][2] - p0[2]))
        return data.xpos[b] - p0, peak

    chairs = [o for o in objects if o.label in bodies and o.kind == "chair"]
    if chairs:
        # Pull the chair out from under its desk (as an agent would), else towards the camera.
        chair = chairs[0]
        by_label = {o.label: o for o in objects}
        over = [by_label[k] for k in chair.relations.get("under", []) + chair.relations.get("partly_under", []) if k in by_label]
        ways = [np.array([0.0, -1.0])]
        if over:
            # straight out of a desk edge (not diagonally into a leg): first the edge the chair sticks
            # out of most, then the edge facing the camera (open room), then the other edge. A chair in
            # a corner of the desk is nearly as far out of two edges; physics decides which way is free.
            desk = over[0]
            axes = np.array([[math.cos(desk.yaw), math.sin(desk.yaw)], [-math.sin(desk.yaw), math.cos(desk.yaw)]])
            local = axes @ (chair.center[:2] - desk.center[:2]) / np.maximum(desk.size[:2] / 2, 1e-3)
            to_cam = axes @ (-desk.center[:2])                     # the camera stands at (0, 0)
            ways = []
            for k in [int(np.argmax(np.abs(local))), int(np.argmax(np.abs(to_cam))), int(np.argmin(np.abs(local)))]:
                sign = local[k] if abs(local[k]) > 0.05 else to_cam[k]
                w = axes[k] * (1.0 if sign >= 0 else -1.0)
                if not any(np.allclose(w, x) for x in ways):
                    ways.append(w)
        for tried, way in enumerate(ways, 1):
            d, _ = run_force(chair.label, np.array([*(80.0 * way), 0.0]), 0.6)
            result["push"] = {"object": chair.label, "force_N": 80, "moved_cm": round(float(np.linalg.norm(d[:2])) * 100, 1),
                              "ok": bool(np.linalg.norm(d[:2]) > 0.05), "ways_tried": tried}
            if result["push"]["ok"]:
                break
    graspable = [o for o in objects if o.label in bodies and "lift" in o.info["affordances"]]
    if graspable:
        o = min(graspable, key=lambda o: o.info.get("mass_kg") or 1.0)
        m = o.info.get("mass_kg") or 0.5
        # The object rises, then falls back once the force stops (or hits a loft bed above it),
        # so the proof is the highest point it reached.
        _, rose = run_force(o.label, np.array([0.0, 0.0, 2.5 * m * 9.81]), 0.25)
        result["lift"] = {"object": o.label, "mass_kg": m, "force_N": round(2.5 * m * 9.81, 1),
                          "rose_cm": round(rose * 100, 1), "ok": bool(rose > 0.02)}
    return result


def floor_plan(path: Path, box: RoomBox, objects: list[RoomObject], walls, agent_xy, fov_deg: float, title: str) -> Path:
    """Top-down plan: room (solid = seen walls, dashed = inferred), objects, supports, camera, agent."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(8, 8))
    corners = box.corners()
    sides = {"v-": (0, 1), "u+": (1, 2), "v+": (2, 3), "u-": (3, 0)}
    for side, (i, j) in sides.items():
        seen = box.sides_seen[side]
        ax.plot(*zip(corners[i], corners[j]), color="#333", lw=3 if seen else 1.5, ls="-" if seen else "--")
    order = sorted(objects, key=lambda o: o.top)
    for o in order:
        col = np.array(_color(o)) / 255
        fill = 0.85 if o.support in ("floor", None, "wall") else 0.6
        ax.add_patch(Polygon(o.footprint(), closed=True, fc=(*col, fill), ec="k", lw=0.8, zorder=2 + o.top))
        ax.text(*o.center[:2], f"{o.name}\n#{o.id}", ha="center", va="center", fontsize=7, zorder=10 + o.top,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    # camera + field of view
    half = math.radians(fov_deg) / 2
    reach = 1.2
    ax.add_patch(Polygon([(0, 0), (reach * math.sin(-half), reach * math.cos(half)), (reach * math.sin(half), reach * math.cos(half))],
                         closed=True, fc=(1, 0, 0, 0.12), ec="red", lw=0.8))
    ax.plot(0, 0, marker="^", color="red", ms=12, label="camera (photo)")
    ax.plot(*agent_xy, marker="o", color="#c00", ms=9, mec="k", label="agent spawn")
    ax.plot([], [], color=(120 / 255, 140 / 255, 170 / 255), lw=8, label="static (furniture)")
    ax.plot([], [], color=(235 / 255, 140 / 255, 40 / 255), lw=8, label="movable")
    ax.plot([], [], color="#333", lw=3, label="wall seen")
    ax.plot([], [], color="#333", lw=1.5, ls="--", label="wall inferred")
    ax.set_aspect("equal")
    allp = np.concatenate([corners, [[0, 0]]])
    ax.set_xlim(allp[:, 0].min() - 0.3, allp[:, 0].max() + 0.3)
    ax.set_ylim(allp[:, 1].min() - 0.3, allp[:, 1].max() + 0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path

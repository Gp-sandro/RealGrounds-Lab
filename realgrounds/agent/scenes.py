"""A very simple room to start with, built by hand - in exactly the format real rooms have.

WHAT THIS FILE DOES (in plain words)
    simple_room() places a few objects in a 4 x 3.5 m room and runs them through the SAME code
    as photos do (relationships, records in the agreed format, MuJoCo export), so the agent can't
    tell it apart from a photographed room:

        a table with a water bottle and a book on it, a chair tucked in under the table,
        a nightstand, a light cardboard box and a 40 kg box on the floor, a poster on the wall.

    The 40 kg box is there on purpose: the record says it can be lifted (a box affords "lift"),
    but the arm can only lift 15 kg - physics has to catch that, not a rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .world import CATALOG, LAB

from .. import room, physics as room_export  # noqa: E402


def _obj(i: int, kind: str, center, size, zmin: float, top: float, name: str | None = None, **info) -> room.RoomObject:
    return room.RoomObject(i, kind, name or kind.replace("_", " "), 0.9, np.array(center, float), np.array(size, float),
                           0.0, zmin, top, np.zeros((4, 4), bool), {**CATALOG[kind], **info}, seen_in=["hand-made"],
                           points=100)


def simple_room(out: Path) -> Path:
    """Write the simple room to `out` (room_mujoco.xml, objects.json, summary.json). Returns `out`."""
    out.mkdir(parents=True, exist_ok=True)
    walls = [room.Wall(np.array([0.0, -1.0]), 3.0, False, 1000),    # y = 3   (back)
             room.Wall(np.array([0.0, 1.0]), 0.5, False, 1000),     # y = -0.5 (behind the camera)
             room.Wall(np.array([1.0, 0.0]), 2.0, False, 1000),     # x = -2
             room.Wall(np.array([-1.0, 0.0]), 2.0, False, 1000)]    # x = 2
    objects = [
        _obj(0, "table", [0.0, 2.2, 0.375], [1.2, 0.7, 0.75], 0.0, 0.75),
        _obj(1, "water_bottle", [-0.3, 2.15, 0.87], [0.07, 0.07, 0.24], 0.75, 0.99),
        _obj(2, "book", [0.25, 2.3, 0.765], [0.16, 0.24, 0.03], 0.75, 0.78),
        _obj(3, "chair", [0.1, 1.9, 0.45], [0.45, 0.45, 0.9], 0.0, 0.9),       # tucked in under the table
        _obj(4, "nightstand", [1.55, 2.6, 0.275], [0.45, 0.4, 0.55], 0.0, 0.55),
        _obj(5, "box", [-1.3, 0.9, 0.15], [0.4, 0.3, 0.3], 0.0, 0.3),
        _obj(6, "box", [1.3, 0.7, 0.2], [0.45, 0.35, 0.4], 0.0, 0.4, name="heavy box", mass_kg=40),
        _obj(7, "poster", [-1.0, 2.99, 1.5], [0.5, 0.02, 0.7], 1.15, 1.85),
    ]
    geo = {**room_config()["geometry"], **room_config()["multi_view"], "seed": 0}
    room.relate(objects, walls, geo, CATALOG)
    box = room.room_box(walls, 0.0, np.zeros((0, 2)), objects, 2.5)
    room.fit_inside(objects, box)
    room_export.to_mjcf(out / "room_mujoco.xml", box, objects, [], (150, 135, 115), (225, 222, 215), (0.0, 0.3))
    (out / "objects.json").write_text(json.dumps([room.record(o) for o in objects], indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"title": "simple test room", "objects": len(objects)}, indent=2), encoding="utf-8")
    return out


def room_config() -> dict:
    import yaml

    return yaml.safe_load((LAB.parent / "config.yaml").read_text())


def obstacle_room(out: Path, variant: str = "light") -> Path:
    """A test room for autonomous path clearing (same pipeline code as photo rooms).

    A desk (static) and a dresser (static) leave a 1 m gap; things on the floor block it. The task: a walking
    route from the agent's start (0, 0.3) to the far side (0, 3.0).
        light   a box (catalog 1 kg) in the gap and a chair beside it   -> can be cleared
        heavy   a box set to 40 kg in the gap (a TEST VALUE, not an estimate) -> the arm lifts 15 kg: can't
        wall    a nightstand (static furniture) in the gap              -> nothing movable blocks it
    """
    out.mkdir(parents=True, exist_ok=True)
    walls = [room.Wall(np.array([0.0, -1.0]), 3.5, False, 1000),    # y = 3.5 (back)
             room.Wall(np.array([0.0, 1.0]), 0.5, False, 1000),     # y = -0.5 (behind the start)
             room.Wall(np.array([1.0, 0.0]), 2.0, False, 1000),     # x = -2
             room.Wall(np.array([-1.0, 0.0]), 2.0, False, 1000)]    # x = 2
    objects = [
        _obj(0, "desk", [-1.25, 1.8, 0.375], [1.5, 0.7, 0.75], 0.0, 0.75),
        _obj(1, "dresser", [1.25, 1.8, 0.375], [1.5, 0.6, 0.75], 0.0, 0.75),
    ]
    if variant == "light":
        objects += [_obj(2, "box", [0.0, 1.8, 0.15], [0.4, 0.3, 0.3], 0.0, 0.3),
                    _obj(3, "chair", [0.35, 1.2, 0.45], [0.45, 0.45, 0.9], 0.0, 0.9)]
    elif variant == "heavy":
        objects += [_obj(2, "box", [0.0, 1.8, 0.2], [0.5, 0.4, 0.4], 0.0, 0.4, name="heavy box", mass_kg=40)]
    elif variant == "wall":
        objects += [_obj(2, "nightstand", [0.0, 1.8, 0.275], [0.6, 0.45, 0.55], 0.0, 0.55)]
    else:
        raise ValueError(variant)
    geo = {**room_config()["geometry"], **room_config()["multi_view"], "seed": 0}
    room.relate(objects, walls, geo, CATALOG)
    box = room.room_box(walls, 0.0, np.zeros((0, 2)), objects, 2.5)
    room.fit_inside(objects, box)
    room_export.to_mjcf(out / "room_mujoco.xml", box, objects, [], (150, 135, 115), (225, 222, 215), (0.0, 0.3))
    records = [room.record(o) for o in objects]
    if variant == "heavy":
        records[2]["Mass"] = "40 kg [TEST VALUE set by hand - not estimated from a photo]"
    (out / "objects.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"title": f"obstacle test room ({variant})", "objects": len(objects)},
                                                 indent=2), encoding="utf-8")
    # the route this room was designed for (realgrounds/tasks.py uses it instead of "the farthest floor spot")
    (out / "route.json").write_text(json.dumps({"start": [0.0, 0.3], "target": [0.0, 3.0], "radius": 0.3,
                                                "title": "walk through the gap to the far side"}), encoding="utf-8")
    return out

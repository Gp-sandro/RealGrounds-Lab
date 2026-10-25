"""Automatic checks for the room model (realgrounds/room.py, support.py, physics.py).

WHAT THIS FILE DOES (in plain words)
    Builds small rooms by hand (a desk against a wall, a book on it, a chair under it) and
    checks that the relationships, the object records and the physics come out right. Needs
    no neural networks; runs in a few seconds.

HOW TO RUN (from the pixel/ folder)
    python -m unittest discover -s tests -t . -v      (from the realgrounds-voice/ folder)
"""

from __future__ import annotations

import importlib.util
import math
import shutil
import sys
import unittest

import numpy as np
import yaml

from realgrounds import room
from realgrounds.config import ROOT


CATALOG = yaml.safe_load((ROOT / "realgrounds/catalog.yaml").read_text())["objects"]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
GEO = {**CFG["geometry"], **CFG["multi_view"], "seed": 0}


def obj(i: int, kind: str, center, size, zmin: float, top: float) -> room.RoomObject:
    return room.RoomObject(i, kind, kind.replace("_", " "), 0.9, np.array(center, float), np.array(size, float), 0.0,
                           zmin, top, np.zeros((4, 4), bool), CATALOG[kind], seen_in=["test"], points=100)


def desk_scene() -> tuple[list[room.RoomObject], list[room.Wall]]:
    """A desk 5 cm from a wall at y = 2, a book and a bottle on it, a chair half under it."""
    wall = room.Wall(np.array([0.0, -1.0]), 2.0, False, 1000)      # wall line y = 2, room on the y < 2 side
    desk = obj(0, "desk", [0.0, 1.6, 0.375], [1.2, 0.7, 0.75], 0.0, 0.75)
    book = obj(1, "book", [-0.3, 1.6, 0.765], [0.2, 0.25, 0.03], 0.752, 0.78)
    bottle = obj(2, "water_bottle", [0.1, 1.6, 0.87], [0.07, 0.07, 0.24], 0.75, 0.99)
    chair = obj(3, "chair", [0.3, 1.35, 0.45], [0.5, 0.5, 0.9], 0.3, 0.9)  # half under the desk; legs hidden
    return [desk, book, bottle, chair], [wall]


class TestRelationships(unittest.TestCase):
    """supported_by / next_to / under / against come out of plain geometry."""

    def test_desk_scene(self) -> None:
        objects, walls = desk_scene()
        room.relate(objects, walls, GEO)
        desk, book, bottle, chair = objects
        self.assertEqual(book.relations["supported_by"], ["desk #0"])
        self.assertEqual(bottle.relations["supported_by"], ["desk #0"])
        self.assertIn("water bottle #2", book.relations.get("next_to", []))
        self.assertEqual(chair.relations["supported_by"], ["floor"])        # hidden legs -> on the floor
        self.assertEqual(chair.zmin, 0.0)
        self.assertIn("desk #0", chair.relations.get("partly_under", []) + chair.relations.get("under", []))
        self.assertLessEqual(chair.top, desk.top - 0.04 + 1e-9)            # collides only below the desk top
        self.assertIn("against", desk.relations)
        self.assertAlmostEqual(float(np.max(desk.footprint()[:, 1])), 2.0, places=2)  # desk extended to the wall
        self.assertNotIn("under", book.relations)                           # a desk item is never "under" another

    def test_overlapping_boxes_are_separated(self) -> None:
        objects, walls = desk_scene()
        objects[2].center[:2] = [-0.25, 1.62]                                # bottle box inside the book box
        room.relate(objects, walls, GEO)
        book, bottle = objects[1], objects[2]
        d = np.abs(book.center[:2] - bottle.center[:2])
        reach = (book.size[:2] + bottle.size[:2]) / 2
        self.assertTrue((d >= reach - 1e-6).any())                          # apart along at least one axis

    def test_record_has_the_agreed_fields(self) -> None:
        objects, walls = desk_scene()
        room.relate(objects, walls, GEO)
        r = room.record(objects[2])
        for key in ("Object", "Boundary", "Position", "Material", "State", "Affordances", "Relationships",
                    "Collision", "Mass"):
            self.assertIn(key, r)
        self.assertEqual(r["Object"], "Water Bottle")
        self.assertTrue(r["Material"].endswith("[estimated]"))
        self.assertEqual(r["State"], "movable")
        self.assertIn("grasp", r["Affordances"])
        self.assertIn("on desk surface", r["Position"])
        self.assertIn("Object: Water Bottle", room.record_text(objects[2]))


class TestDormRules(unittest.TestCase):
    """Rules added for a lived-in dorm: loft bed, ladder, hanging, same-thing-twice, room walls."""

    def test_loft_bed_ladder_and_storage(self) -> None:
        wall = room.Wall(np.array([0.0, -1.0]), 3.0, False, 1000)
        loft = obj(0, "bed", [0.0, 2.5, 0.9], [2.0, 1.0, 1.8], 0.0, 1.8)          # a "bed" reaching 1.8 m
        ladder = obj(1, "ladder", [1.15, 2.2, 0.75], [0.3, 0.1, 1.5], 0.0, 1.5)
        bin_ = obj(2, "storage_bin", [-0.5, 2.5, 0.15], [0.5, 0.4, 0.3], 0.0, 0.3)
        objects = [loft, ladder, bin_]
        room.relate(objects, [wall], GEO, CATALOG)
        self.assertEqual(loft.kind, "loft_bed")                                  # renamed: platform on posts
        self.assertIn("loft bed #0", ladder.relations.get("leaning_on", []))
        self.assertIn("loft bed #0", bin_.relations.get("under", []))

    def test_backpack_hangs_on_chair_and_rides_along(self) -> None:
        chair = obj(0, "chair", [0.0, 1.0, 0.33], [0.5, 0.5, 0.66], 0.0, 0.66)
        backpack = obj(1, "backpack", [0.0, 0.8, 0.43], [0.4, 0.2, 0.33], 0.27, 0.6)  # draped over its back
        objects = [chair, backpack]
        room.relate(objects, [], GEO, CATALOG)
        self.assertEqual(backpack.relations.get("hanging_on"), ["chair #0"])
        from realgrounds import physics as room_export

        box = room.room_box([], 0.0, np.zeros((0, 2)), objects, 2.5)
        out = ROOT / "outputs" / "_tests_room"
        out.mkdir(parents=True, exist_ok=True)
        try:
            xml = room_export.to_mjcf(out / "room.xml", box, objects, [], (140, 125, 110), (225, 222, 215), (-1.5, -1.0))
            text = xml.read_text()
            chair_body = text[text.index('<body name="obj0_chair"'):]
            self.assertIn("obj1_backpack_g", chair_body[:chair_body.index("</body>")])  # moves with the chair
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_same_thing_twice_and_lying_inside(self) -> None:
        bed = obj(0, "bed", [0.0, 2.0, 0.3], [1.0, 2.0, 0.6], 0.0, 0.6)
        mattress = obj(1, "mattress", [0.0, 2.0, 0.28], [0.9, 1.9, 0.55], 0.0, 0.55)  # the bed's own mattress
        blanket = obj(2, "blanket", [0.1, 2.1, 0.2], [0.8, 1.2, 0.4], 0.0, 0.4)       # depth put it too low
        objects = [bed, mattress, blanket]
        room.relate(objects, [], GEO, CATALOG)
        self.assertNotIn(mattress, objects)                                     # removed: seen twice
        self.assertEqual(blanket.relations["supported_by"], ["bed #0"])           # lies ON the bed
        self.assertAlmostEqual(blanket.zmin, bed.top)

    def test_nothing_pokes_through_a_wall(self) -> None:
        box = room.RoomBox(0.0, np.array([-2.0, -1.0]), np.array([2.0, 3.0]), 2.5, dict.fromkeys(("u-", "u+", "v-", "v+"), True))
        bed = obj(0, "bed", [1.5, 2.6, 0.3], [1.2, 2.0, 0.6], 0.0, 0.6)          # reaches 0.1 m past x = 2, 0.6 m past y = 3
        room.fit_inside([bed], box)
        fp = bed.footprint()
        self.assertLessEqual(fp[:, 0].max(), 2.0 + 1e-6)
        self.assertLessEqual(fp[:, 1].max(), 3.0 + 1e-6)
        self.assertAlmostEqual(fp[:, 0].min(), 0.9)                              # the near side stays put


class TestSupportCommonSense(unittest.TestCase):
    """What rests on what: geometry AND how often that kind really rests on that kind (room benchmark).
    Each case is a failure seen on real rooms (lab/experiments/room_benchmark/notes.md)."""

    def setUp(self) -> None:
        self.assertTrue(GEO.get("support_prior"), "these rules are switched on by room_config.yaml")

    def test_pillow_on_a_sofa_seat_inside_its_box(self) -> None:
        # the sofa's box reaches its backrest (0.85 m); the pillow lies on the seat (0.45 m) inside it
        sofa = obj(0, "sofa", [0.0, 2.0, 0.425], [2.0, 0.9, 0.85], 0.0, 0.85)
        pillow = obj(1, "pillow", [0.3, 1.9, 0.55], [0.45, 0.35, 0.2], 0.45, 0.65)
        room.relate([sofa, pillow], [], GEO, CATALOG)
        self.assertEqual(pillow.support, "sofa #0")
        self.assertTrue(pillow.inner_support)
        self.assertAlmostEqual(pillow.zmin, 0.45, places=2)      # stays on the seat, not lifted to 0.85

    def test_picture_high_on_a_wall_hangs_on_it(self) -> None:
        wall = room.Wall(np.array([0.0, -1.0]), 3.0, False, 1000)   # wall line y = 3
        picture = obj(0, "picture_frame", [0.0, 2.97, 1.5], [0.5, 0.04, 0.4], 1.3, 1.7)
        room.relate([picture], [wall], GEO, CATALOG)
        self.assertEqual(picture.support, "wall")

    def test_picture_frame_standing_on_a_table_is_on_the_table(self) -> None:
        wall = room.Wall(np.array([0.0, -1.0]), 3.0, False, 1000)
        table = obj(0, "table", [0.0, 2.7, 0.375], [1.0, 0.5, 0.75], 0.0, 0.75)
        picture = obj(1, "picture_frame", [0.0, 2.85, 0.86], [0.2, 0.05, 0.22], 0.75, 0.97)
        room.relate([table, picture], [wall], GEO, CATALOG)
        self.assertEqual(picture.support, "table #0")

    def test_pillow_on_the_blanket_on_the_mattress(self) -> None:
        mattress = obj(0, "mattress", [0.0, 2.0, 0.25], [0.9, 1.9, 0.5], 0.0, 0.5)
        blanket = obj(1, "blanket", [0.0, 2.0, 0.53], [0.9, 1.6, 0.05], 0.5, 0.55)
        pillow = obj(2, "pillow", [0.0, 2.6, 0.62], [0.5, 0.35, 0.14], 0.55, 0.69)
        room.relate([mattress, blanket, pillow], [], GEO, CATALOG)
        self.assertEqual(blanket.support, "mattress #0")
        self.assertEqual(pillow.support, "blanket #1")            # a cover takes the role of what it covers

    def test_keyboard_estimated_low_still_rests_on_the_desk(self) -> None:
        # its bottom came out 12 cm under the desk top (depth smear): never on the floor (0 of 25 in the data)
        desk = obj(0, "desk", [0.0, 1.6, 0.375], [1.2, 0.7, 0.75], 0.0, 0.75)
        keyboard = obj(1, "keyboard", [0.0, 1.5, 0.70], [0.44, 0.14, 0.1], 0.63, 0.77)
        room.relate([desk, keyboard], [], GEO, CATALOG)
        self.assertEqual(keyboard.support, "desk #0")

    def test_bag_hangs_on_the_chair_not_the_desk_beside_it(self) -> None:
        desk = obj(0, "desk", [0.0, 1.6, 0.375], [1.2, 0.7, 0.75], 0.0, 0.75)
        chair = obj(1, "chair", [0.0, 1.0, 0.33], [0.5, 0.5, 0.66], 0.0, 0.66)
        backpack = obj(2, "backpack", [0.0, 0.8, 0.5], [0.4, 0.2, 0.33], 0.34, 0.67)   # bottom above hang_min_m
        room.relate([desk, chair, backpack], [], GEO, CATALOG)
        self.assertEqual(backpack.relations.get("hanging_on"), ["chair #1"])

    def test_fan_on_an_unseen_shelf_is_not_put_on_the_wall(self) -> None:
        wall = room.Wall(np.array([0.0, -1.0]), 3.0, False, 1000)
        fan = obj(0, "fan", [0.0, 2.88, 1.35], [0.25, 0.2, 0.3], 1.2, 1.5)
        room.relate([fan], [wall], GEO, CATALOG)
        self.assertEqual(fan.support, "unseen")                   # fans rarely hang: say what we don't know


class TestAlignment(unittest.TestCase):
    """Two photos: recover turn + shift + scale between their world frames."""

    def test_yaw_similarity(self) -> None:
        rng = np.random.default_rng(0)
        b = rng.uniform([-2, 0, 0], [2, 4, 2], (50, 3))
        truth = room.Alignment("x", True, 1.1, math.radians(30), np.array([0.5, -0.3, 0.0]))
        a = truth.apply(b)
        s, yaw, t = room._fit_yaw_similarity(a, b)
        self.assertAlmostEqual(s, 1.1, places=3)
        self.assertAlmostEqual(math.degrees(yaw), 30, places=2)
        np.testing.assert_allclose(t, [0.5, -0.3, 0.0], atol=1e-6)


class TestRoomPhysics(unittest.TestCase):
    """The room loads in MuJoCo; things rest on their supports; the chair can be pushed."""

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "mujoco not installed")
    def test_settle_push_lift(self) -> None:
        from realgrounds import physics as room_export

        objects, walls = desk_scene()
        room.relate(objects, walls, GEO)
        box = room.room_box(walls, 0.0, np.zeros((0, 2)), objects, 2.5)
        out = ROOT / "outputs" / "_tests_room"
        out.mkdir(parents=True, exist_ok=True)
        try:
            xml = room_export.to_mjcf(out / "room.xml", box, objects, [], (140, 125, 110), (225, 222, 215), (-0.8, 0.3))
            proofs = room_export.physics_proofs(xml, objects)
            self.assertTrue(proofs["settle_ok"], proofs["settle"])
            self.assertTrue(proofs["agent_standing"])
            self.assertTrue(proofs["push"]["ok"])
            self.assertTrue(proofs["lift"]["ok"])
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

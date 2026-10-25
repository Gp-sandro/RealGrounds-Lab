"""Automatic checks for the agent (realgrounds/agent/): an agent that does
what it is told in a physics world, and is checked by physics.

WHAT THIS FILE DOES (in plain words)
    1. Language: sentences are split into the right steps and objects ("pick up the book and put
       it on the chair", "put down", "take the backpack off the chair").
    2. The hand-made simple room: the whole task list ends as expected - things are really put
       where they were told (contact under them), static furniture is refused, a 40 kg box is too
       heavy for the arm, a chair under a table is pulled out.
    Needs MuJoCo (no neural networks); runs in a few seconds.

HOW TO RUN (from the realgrounds-voice/ folder)
    python -m unittest tests.test_agent -v
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from realgrounds.agent.commands import parse
from realgrounds.config import ROOT

EXP = ROOT / "realgrounds" / "agent"

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None


class TestParse(unittest.TestCase):
    """Sentence -> steps."""

    def test_steps(self) -> None:
        steps = parse("Pick up the book and put it on the chair, then walk to the desk")
        self.assertEqual([(s.verb, s.obj, s.target) for s in steps],
                         [("pick", "the book", ""), ("place", "it", "the chair"), ("walk", "the desk", "")])

    def test_put_down_and_off(self) -> None:
        self.assertEqual(parse("put down the pillow")[0].target, "the floor")
        step = parse("take the backpack off the chair and put it on the floor")
        self.assertEqual((step[0].verb, step[0].obj), ("pick", "the backpack off the chair"))
        self.assertEqual((step[1].verb, step[1].target), ("place", "the floor"))

    def test_unknown_verb_is_refused_with_the_skill_list(self) -> None:
        msg = parse("climb the ladder")
        self.assertIsInstance(msg, str)
        self.assertIn("pick up", msg)


@unittest.skipUnless(HAS_MUJOCO, "mujoco not installed")
class TestSimpleRoom(unittest.TestCase):
    """The task list of the simple room ends exactly as expected, measured in physics."""

    @classmethod
    def setUpClass(cls) -> None:
        from realgrounds.agent.commands import Commander
        from realgrounds.agent.scenes import simple_room
        from realgrounds.agent.skills import Agent
        from realgrounds.agent.world import World

        cls.tmp = Path(tempfile.mkdtemp())
        cls.world = World(simple_room(cls.tmp / "simple"))
        cls.boss = Commander(cls.world, Agent(cls.world))
        cls.tasks = yaml.safe_load((EXP / "tasks_simple.yaml").read_text())["tasks"]
        cls.outcomes = [cls.boss.run(t["say"]) for t in cls.tasks]

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_sentence_ends_as_expected(self) -> None:
        for task, o in zip(self.tasks, self.outcomes):
            self.assertEqual(o.status, task["expect"], f"{task['say']}: {o.lines}")

    def test_placements_are_checked_by_contact_and_accurate(self) -> None:
        placed = [f for o in self.outcomes for f in o.facts if "rests_on" in f and f.get("ok")]
        self.assertGreaterEqual(len(placed), 4)
        for f in placed:
            self.assertLessEqual(f["cm_from_chosen_spot"], 5.0, f)

    def test_too_heavy_is_decided_by_physics(self) -> None:
        heavy = next(o for t, o in zip(self.tasks, self.outcomes) if "heavy" in t["say"])
        self.assertIn("too heavy", heavy.lines[-1])
        self.assertLess(heavy.facts[-1]["rose_cm"], 2.0)

    def test_world_state_after_the_session(self) -> None:
        w = self.world
        self.assertIn("floor", w.resting_on(w.things["water bottle #1"]))
        self.assertTrue(w.rests_on(w.things["box #5"], "desk #0"))
        self.assertTrue(w.rests_on(w.things["book #2"], "chair #3"))
        self.assertIsNone(w.held)


if __name__ == "__main__":
    unittest.main()

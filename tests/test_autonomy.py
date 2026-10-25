"""Direct tests of the agent tools and the autonomous loop - all in SIMULATION (MuJoCo, no neural networks).

    1. try_action() runs in isolation: the persistent simulated world is bit-identical afterwards
    2. a try predicts its commit exactly (deterministic simulation); commit_action() changes the world
    3. goal checks are objective and deterministic (route clear / infeasible / blocked by what)
    4. the supervisor enforces its limits on its own (repeats, loops, no progress, budgets)
    5. end to end: one simulated success, one appropriate stop (stuck), one immediate stop (infeasible)

HOW TO RUN (from the realgrounds-voice/ folder)
    python -m unittest tests.test_autonomy -v
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from realgrounds.agent.scenes import obstacle_room
from realgrounds.autonomy import Report, run
from realgrounds.goals import ObjectRestsOn, PathClear
from realgrounds.sandbox import SimWorld, same, take
from realgrounds.supervisor import Limits, Supervisor

GOAL = PathClear((0.0, 0.3), (0.0, 3.0), 0.3, "route to the far side")


class Scenes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp())
        cls.rooms = {v: obstacle_room(cls.tmp / v, v) for v in ("light", "heavy", "wall")}

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestSandbox(Scenes):
    def test_tries_leave_the_world_bit_identical(self) -> None:
        env = SimWorld(self.rooms["light"])
        before, h = take(env.world, env.cmd), env.state_hash()
        for a in ("put box #2 on the floor", "put box #2 on dresser #1", "pick up box #2", "pull out chair #3",
                  "push chair #3", "pick up desk #0"):
            r = env.try_action(a, GOAL)
            self.assertTrue(r.restored, a)
            self.assertEqual(r.mode, "try")
        self.assertTrue(same(take(env.world, env.cmd), before))
        self.assertEqual(env.state_hash(), h)
        self.assertIsNone(env.world.held)                      # the try that picked something up left nothing held

    def test_a_try_predicts_its_commit_and_the_commit_stays(self) -> None:
        env = SimWorld(self.rooms["light"])
        before = take(env.world, env.cmd)
        t = env.try_action("put box #2 on dresser #1", GOAL)
        c = env.commit_action("put box #2 on dresser #1", GOAL)
        self.assertEqual((t.status, t.moved, t.state_after), (c.status, c.moved, c.state_after))
        self.assertFalse(same(take(env.world, env.cmd), before))
        self.assertTrue(env.check(ObjectRestsOn("box #2", "dresser #1"))["satisfied"])
        self.assertIn("simulated", c.provenance)


class TestGoals(Scenes):
    def test_route_checks_are_deterministic_and_objective(self) -> None:
        light, heavy, wall = (SimWorld(self.rooms[v]) for v in ("light", "heavy", "wall"))
        a, b = light.check(GOAL), light.check(GOAL)
        self.assertEqual(a, b)
        self.assertFalse(a["satisfied"])
        self.assertTrue(a["feasible"])
        self.assertEqual(a["blocked_by"], ["box #2", "chair #3"])
        self.assertEqual(heavy.check(GOAL)["blocked_by"], ["heavy box #2"])
        w = wall.check(GOAL)
        self.assertFalse(w["feasible"])                       # fixed furniture: nothing to move
        self.assertIn("simulated", w["provenance"])


class _R:   # a stand-in for an ActionResult (only what the supervisor reads)
    def __init__(self, status="done", goal=None, state="s1", sim=1.0, lines=()):
        self.status, self.goal, self.state_after, self.sim_seconds, self.lines = status, goal, state, sim, list(lines)
        self.moved, self.restored, self.action = {}, True, "a"


class TestSupervisor(unittest.TestCase):
    def test_repeats_loops_progress_and_budgets(self) -> None:
        s = Supervisor(Limits(max_tries=3, max_commits=1, patience=2))
        s.start("s0")
        self.assertIsNone(s.may_try("s0", "a"))
        s.record_try("s0", "a", _R(goal={"progress_cm": 50.0}))
        self.assertIn("already tried", s.may_try("s0", "a"))                 # repeated action, same state
        self.assertIsNone(s.may_try("s1", "a"))                               # ...fine in another state
        cur = {"satisfied": False, "feasible": True, "progress_cm": 50.0}
        self.assertIn("not measurably improve", s.may_commit(cur, _R(goal={"progress_cm": 50.0})))
        self.assertIn("loop", s.may_commit(cur, _R(goal={"progress_cm": 10.0}, state="s0")))
        self.assertIn("succeeded", s.may_commit(cur, _R(status="failed", goal={"satisfied": True})))
        s.end_round(False); s.end_round(False)
        self.assertEqual(s.decide(cur, untried_left=5).outcome, "stuck")     # no progress for `patience` rounds
        s2 = Supervisor(Limits(max_tries=1))
        s2.record_try("s0", "a", _R(goal={"progress_cm": 50.0}))
        self.assertEqual(s2.may_try("s0", "b"), "try budget spent")
        self.assertEqual(s2.decide(cur, untried_left=5).outcome, "budget")
        self.assertEqual(Supervisor().decide({"satisfied": True}, 0).outcome, "simulated_success")
        self.assertEqual(Supervisor().decide({"satisfied": False, "feasible": False}, 3).outcome, "infeasible")


class TestEndToEnd(Scenes):
    def _run(self, v: str) -> Report:
        return run(SimWorld(self.rooms[v]), GOAL, say=lambda s: None)

    def test_clears_the_route_in_simulation(self) -> None:
        r = self._run("light")
        self.assertEqual(r.outcome, "simulated_success")
        self.assertTrue(r.final_goal["satisfied"])
        self.assertGreaterEqual(r.commits, 1)
        self.assertTrue(all(e["matches_try"] for e in r.log if e["event"] == "commit"))
        self.assertTrue(r.verdict().startswith("SIMULATED SUCCESS"))         # never claims the real room
        self.assertIn("Nothing was done in the real room", r.verdict())

    def test_stops_and_asks_for_help_when_nothing_works(self) -> None:
        r = self._run("heavy")
        self.assertEqual(r.outcome, "stuck")
        self.assertEqual(r.commits, 0)
        self.assertIn("too heavy", r.help_request)
        self.assertLessEqual(r.tries, Limits().max_tries)

    def test_stops_at_once_when_the_goal_is_infeasible(self) -> None:
        r = self._run("wall")
        self.assertEqual((r.outcome, r.tries, r.commits), ("infeasible", 0, 0))


if __name__ == "__main__":
    unittest.main()

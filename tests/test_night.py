"""Offline tests of the Claude planner, task writing and the night's supervisor - a fake API client, no network,
no cost. (The real API is exercised by night_shift.py; these check the plumbing and the stop rules.)

HOW TO RUN (from the realgrounds-voice/ folder)
    python -m unittest tests.test_night -v
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from realgrounds.agent.scenes import obstacle_room
from realgrounds.autonomy import run
from realgrounds.claude_planner import ClaudePlanner, PlannerError, cost_usd
from realgrounds.goals import PathClear
from realgrounds.run_supervisor import NightLimits, NightSupervisor
from realgrounds.sandbox import SimWorld
from realgrounds.supervisor import Limits
from realgrounds.tasks import claude_tasks, far_side_route

GOAL = PathClear((0.0, 0.3), (0.0, 3.0), 0.3, "route to the far side")


class FakeClient:
    """Answers each call with the next canned reply (a dict -> JSON text), 1000 tokens in / 200 out."""

    def __init__(self, replies, stop_reason="end_turn"):
        self.replies, self.stop_reason, self.calls = list(replies), stop_reason, 0
        self.messages = self

    def create(self, **kw):
        self.calls += 1
        reply = self.replies[min(self.calls, len(self.replies)) - 1]
        return SimpleNamespace(stop_reason=self.stop_reason, stop_details=None,
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
                               content=[SimpleNamespace(type="text", text=json.dumps(reply))])


def planner(replies, **kw) -> ClaudePlanner:
    return ClaudePlanner(model="claude-haiku-5-5", client=FakeClient(replies, **kw))


class Rooms(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp())
        cls.rooms = {v: obstacle_room(cls.tmp / v, v) for v in ("light", "heavy", "wall")}

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestPlanner(Rooms):
    def test_cost_is_counted_at_list_prices(self) -> None:
        self.assertAlmostEqual(cost_usd("claude-haiku-5-5", SimpleNamespace(input_tokens=1_000_000, output_tokens=0)), 0.10)
        p = planner([{"reasoning": "r", "actions": ["put box #2 on dresser #1"]}])
        self.assertEqual(p(SimWorld(self.rooms["light"]), {"goal": "g"}), ["put box #2 on dresser #1"])
        self.assertAlmostEqual(p.spent_usd, (1000 * 0.10 + 200 * 0.50) / 1e6)

    def test_refusal_and_cut_off_answers_raise(self) -> None:
        for stop in ("refusal", "max_tokens"):
            with self.assertRaises(PlannerError):
                planner([{"reasoning": "", "actions": []}], stop_reason=stop)(SimWorld(self.rooms["light"]), {})

    def test_the_loop_solves_with_claude_and_counts_money(self) -> None:
        p = planner([{"reasoning": "move the box", "actions": ["put box #2 on dresser #1"]}])
        r = run(SimWorld(self.rooms["light"]), GOAL, planner=p, say=lambda s: None)
        self.assertEqual(r.outcome, "simulated_success")
        self.assertAlmostEqual(r.spent_usd, p.spent_usd)

    def test_a_planner_that_repeats_itself_is_refused_then_stopped(self) -> None:
        p = planner([{"reasoning": "lift it", "actions": ["pick up heavy box #2"]}])     # the same answer forever
        r = run(SimWorld(self.rooms["heavy"]), GOAL, planner=p, say=lambda s: None)
        self.assertEqual(r.outcome, "stuck")
        self.assertEqual(r.tries, 1)                                     # tried once, never again in that state
        self.assertGreaterEqual(sum(e["event"] == "refused" for e in r.log), 1)

    def test_money_budget_stops_the_task(self) -> None:
        p = planner([{"reasoning": "", "actions": ["walk to desk #0"]}, {"reasoning": "", "actions": ["walk to dresser #1"]}])
        r = run(SimWorld(self.rooms["light"]), GOAL, planner=p, limits=Limits(max_usd=0.0001), say=lambda s: None)
        self.assertEqual(r.outcome, "budget")
        self.assertEqual(p.client.calls, 1)                              # no second call once the budget is spent

    def test_planner_failure_stops_the_task_with_a_reason(self) -> None:
        p = planner([{"reasoning": "", "actions": []}], stop_reason="refusal")
        r = run(SimWorld(self.rooms["light"]), GOAL, planner=p, say=lambda s: None)
        self.assertEqual(r.outcome, "planner_error")


class TestTasks(Rooms):
    def test_far_side_route_and_task_validation(self) -> None:
        env = SimWorld(self.rooms["light"])
        t = far_side_route(env)
        self.assertIsNotNone(t)
        self.assertFalse(env.check(t.goal)["satisfied"])                 # the box and chair are in the way
        p = planner([{"tasks": [
            {"kind": "rests_on", "object": "box #2", "target": "desk #0", "why": "tidy"},
            {"kind": "rests_on", "object": "ghost #9", "target": "desk #0", "why": "x"},     # no such object
            {"kind": "rests_on", "object": "desk #0", "target": "floor", "why": "x"},        # fixed furniture
            {"kind": "rests_on", "object": "chair #3", "target": "floor", "why": "x"}]}])    # already true
        tasks, dropped = claude_tasks(env, p, n=4)
        self.assertEqual([x.title for x in tasks], ["put box #2 on desk #0"])
        self.assertEqual(len(dropped), 3)


class TestReach(Rooms):
    def test_reach_uses_the_arms_reach_not_the_objects_centre(self) -> None:
        from realgrounds.goals import ReachObject

        env = SimWorld(self.rooms["light"])
        self.assertTrue(env.check(ReachObject("dresser #1", (0.0, 0.3)))["satisfied"])  # 0.6 m deep: from its edge
        self.assertFalse(env.check(ReachObject("ghost #9", (0.0, 0.3)))["feasible"])


class TestNightSupervisor(unittest.TestCase):
    def test_money_time_and_failure_streaks_stop_the_night(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            n = NightSupervisor(NightLimits(max_usd=1.0, room_usd=0.3, build_failures_in_row=2), Path(d), say=lambda s: None)
            self.assertIsNone(n.may_start_room())
            n.room_finished(0.5, built=True, planner_error=False)
            self.assertIsNone(n.may_start_room())                         # 0.5 + a room's 0.3 still fits in 1.0
            n.room_finished(0.3, built=True, planner_error=False)
            self.assertIn("money", n.may_start_room())                    # 0.8 + 0.3 does not
        with tempfile.TemporaryDirectory() as d:
            n = NightSupervisor(NightLimits(max_usd=1.0, room_usd=0.3, build_failures_in_row=2), Path(d), say=lambda s: None)
            n.room_finished(0.0, built=False, planner_error=False)
            n.room_finished(0.0, built=False, planner_error=False)
            self.assertIn("pipeline", n.may_start_room())
            n.event("x", a=1)
            self.assertTrue((Path(d) / "timeline.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

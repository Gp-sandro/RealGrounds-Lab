"""The autonomous loop: measure the goal, propose actions, try them in isolation, commit the best - in SIMULATION.

    planner      proposes actions (sentences for the command layer): planner(env, goal, history). The
                 rule_based_planner below needs no language model; realgrounds/claude_planner.py is Claude. It
                 only proposes; its spend (if it reports `spent_usd`) counts against the supervisor's budget.
    supervisor   (realgrounds/supervisor.py) approves every try and commit and alone decides when to stop.
    environment  (realgrounds/sandbox.py) runs tries from snapshots and commits on the persistent simulated world.

The result is a report whose outcome is one of: simulated_success, infeasible, stuck, loop, budget - never a
claim about the real room. The world was estimated from photos; a simulated success is a plan to check, not a
fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .claude_planner import PlannerError
from .sandbox import SIMULATED, SimWorld
from .supervisor import Limits, Supervisor, _rank


def rule_based_planner(env: SimWorld, goal: dict[str, Any], history: list[dict[str, Any]] | None = None) -> list[str]:
    """For each thing blocking the route: put it on the floor elsewhere, on one of the nearest surfaces, pull it
    out, push it. Things that can't do an action are refused by the command layer (that is measured too)."""
    w = env.world
    actions: list[str] = []
    if goal.get("target") and goal.get("object"):                     # "X rests on Y": the one obvious command
        target = "the floor" if goal["target"] == "floor" else goal["target"]
        actions.append(f"pick up {goal['object']} and put it on {target}")
    elif goal.get("object"):                                         # "get to X"
        actions.append(f"walk to {goal['object']}")
    for label in goal.get("blocked_by") or []:
        t = w.things.get(label)
        if t is None:
            continue
        c = w.extent(t)["center"][:2]
        surfaces = sorted((s for s in w.things.values() if s is not t and s.is_surface and s.body is None),
                          key=lambda s: float(np.linalg.norm(w.extent(s)["center"][:2] - c)))[:2]
        actions.append(f"put {label} on the floor")
        actions += [f"put {label} on {s.label}" for s in surfaces]
        actions += [f"pull out {label}", f"push {label}"]
    return actions


@dataclass
class Report:
    goal: str
    outcome: str
    reason: str
    help_request: str
    committed: list[str]
    tries: int
    commits: int
    sim_seconds: float
    final_goal: dict[str, Any]
    transcript: list[str] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)
    spent_usd: float = 0.0
    provenance: str = SIMULATED

    def verdict(self) -> str:
        if self.outcome == "simulated_success":
            return ("SIMULATED SUCCESS - the goal holds in the simulated copy of the room. Nothing was done in the "
                    "real room and nothing there is verified: the simulation's sizes and masses are estimates, "
                    "not measurements.")
        return f"STOPPED ({self.outcome}) - {self.reason}.\nAsks for help: {self.help_request}"


def run(env: SimWorld, goal, planner: Callable[..., list[str]] = rule_based_planner,
        limits: Limits | None = None, say: Callable[[str], None] = print) -> Report:
    sup = Supervisor(limits or Limits())
    sup.start(env.state_hash())
    transcript: list[str] = []

    def note(s: str) -> None:
        transcript.append(s)
        say(s)

    note(f"goal: {goal.describe()}  [{SIMULATED}]")
    committed: list[str] = []
    while True:
        g = env.check(goal)
        note(f"  measured: {'satisfied' if g.get('satisfied') else g.get('why')}"
             + (f" (blocked by {', '.join(g['blocked_by'])})" if g.get("blocked_by") else ""))
        state = env.state_hash()
        proposals: list[str] = []
        if not g.get("satisfied") and g.get("feasible") is not False and sup.decide(g, 1).stop is False:
            before = getattr(planner, "spent_usd", 0.0)
            try:
                proposals = planner(env, g, sup.log)
            except PlannerError as e:
                note(f"supervisor: STOP - planner_error: {e}")
                help_ = sup._help(f"The planner failed ({e}); the task is unfinished.")
                return Report(goal.describe(), "planner_error", str(e), help_, committed, sup.tries, sup.commits,
                              round(sup.sim_seconds, 1), g, transcript, sup.log, spent_usd=round(sup.spent_usd, 6))
            spent = getattr(planner, "spent_usd", 0.0) - before
            why = getattr(planner, "last_reasoning", "")
            sup.record_spend(spent, proposals, why)
            if why or spent:
                note(f"  planner (${spent:.4f}): {why}" + (f" -> {proposals}" if proposals else " -> no actions"))
        untried = []
        for a in proposals:
            refused = sup.may_try(state, a)
            if refused is None:
                untried.append(a)
            elif "already tried" in refused:
                sup.record_refused(a, refused)
                note(f"  supervisor: refused {a!r} ({refused})")
        d = sup.decide(g, len(untried))
        if d.stop:
            note(f"supervisor: STOP - {d.outcome}: {d.reason}")
            if d.help_request:
                note(f"supervisor: asks for help - {d.help_request}")
            return Report(goal.describe(), d.outcome, d.reason, d.help_request, committed, sup.tries, sup.commits,
                          round(sup.sim_seconds, 1), g, transcript, sup.log, spent_usd=round(sup.spent_usd, 6))
        results = []
        for a in untried:
            why = sup.may_try(state, a)
            if why is not None:
                note(f"  supervisor: no more tries this round ({why})")
                break
            r = env.try_action(a, goal)
            sup.record_try(state, a, r)
            rg = r.goal or {}
            outcome = (r.lines[-1] if r.status != "done" else
                       f"moved {', '.join(f'{k} {v:.0f} cm' for k, v in r.moved.items()) or 'nothing'}; goal "
                       + ("SATISFIED" if rg.get("satisfied") else f"{rg.get('progress_cm')} cm blocked"))
            note(f"  try (sandboxed, then undone) {a!r}: {r.status} - {outcome}")
            results.append(r)
            if r.status == "done" and rg.get("satisfied"):      # (a failed action that happens to clear it doesn't count)
                break
        done = [r for r in results if r.status == "done"]
        best = min(done, key=lambda r: _rank(r.goal)) if done else None
        why = sup.may_commit(g, best) if best else "no tried action succeeded"
        if why:
            note(f"  supervisor: nothing committed this round ({why})")
            sup.end_round(False)
            continue
        c = env.commit_action(best.action, goal)
        sup.record_commit(best, c)
        sup.end_round(True)
        committed.append(best.action)
        note(f"  COMMIT in simulation {best.action!r}: {c.status}, matches its try: {c.state_after == best.state_after}")

"""An independent supervisor for an autonomous agent working in the SIMULATED room.

The planner (rule-based or a language model) only proposes; every try and every commit goes through here, and
this - not the planner - decides when to stop. All its signals are measured, none is the planner's opinion:

  attempt limits      max tries (sandboxed actions), max commits, max simulated seconds, max dollars spent on
                      the planner (a language model)
  repeated actions    the same action tried again in the same simulated state is refused (it would give the
                      same result: the simulation is deterministic)
  repeated states     a commit that returns the world to a state seen before = a loop -> stop
  progress            after every commit the goal's measured progress must improve; `patience` rounds of tries
                      without any improving action = stuck -> stop
  stopping            goal satisfied (simulated success) / goal infeasible / budget spent / loop / stuck - and
                      for the failures, a request for help that says what was tried and why it failed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Limits:
    max_tries: int = 24
    max_commits: int = 6
    max_sim_seconds: float = 900.0
    max_usd: float = 0.10                # planner spend per task (language-model tokens at list prices)
    patience: int = 2                    # rounds of tries without an improving action before "stuck"


@dataclass
class Decision:
    stop: bool
    outcome: str = ""                    # simulated_success / infeasible / stuck / loop / budget / planner_error
    reason: str = ""
    help_request: str = ""


def _rank(goal: dict[str, Any]) -> tuple:
    """Lower = better: satisfied first, then less of the route blocked."""
    if goal.get("satisfied"):
        return (0, 0.0)
    p = goal.get("progress_cm")
    return (1, float("inf") if p is None else float(p))


@dataclass
class Supervisor:
    limits: Limits = field(default_factory=Limits)
    tries: int = 0
    commits: int = 0
    sim_seconds: float = 0.0
    spent_usd: float = 0.0
    stale_rounds: int = 0
    seen_states: list[str] = field(default_factory=list)
    tried: dict[tuple[str, str], Any] = field(default_factory=dict)     # (state, action) -> result
    failures: list[str] = field(default_factory=list)                    # "action: why it failed"
    log: list[dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------- gates (the planner must pass these)
    def may_try(self, state: str, action: str) -> str | None:
        """None = allowed, else why not."""
        if self.tries >= self.limits.max_tries:
            return "try budget spent"
        if self.sim_seconds >= self.limits.max_sim_seconds:
            return "simulated-time budget spent"
        if self.spent_usd >= self.limits.max_usd:
            return "money budget spent"
        if (state, action) in self.tried:
            return "already tried in this exact state (deterministic: same result)"
        return None

    def record_try(self, state: str, action: str, result) -> None:
        self.tries += 1
        self.sim_seconds += result.sim_seconds
        self.tried[(state, action)] = result
        if result.status != "done":
            why = next((l for l in reversed(result.lines) if l.startswith("✗")), result.status)
            self.failures.append(f"{action}: {why.lstrip('✗ ').strip()}")
        self.log.append({"event": "try", "action": action, "status": result.status, "moved_cm": result.moved,
                         "result": result.lines[-1] if result.lines else "", "goal_after": result.goal,
                         "sim_s": result.sim_seconds, "restored": result.restored})

    def record_spend(self, usd: float, proposals: list[str], reasoning: str = "") -> None:
        self.spent_usd += usd
        self.log.append({"event": "plan", "usd": round(usd, 6), "proposals": proposals, "reasoning": reasoning})

    def record_refused(self, action: str, why: str) -> None:
        """A proposal refused before it ran (e.g. the planner repeated an action in the same state)."""
        self.log.append({"event": "refused", "action": action, "why": why})

    def may_commit(self, current_goal: dict[str, Any], tried_result) -> str | None:
        if self.commits >= self.limits.max_commits:
            return "commit budget spent"
        if tried_result.status != "done":
            return "only actions that succeeded when tried may be committed"
        if tried_result.goal is None or _rank(tried_result.goal) >= _rank(current_goal):
            return "the tried action does not measurably improve the goal"
        if tried_result.state_after in self.seen_states:
            return "it would return the world to a state seen before (loop)"
        return None

    def record_commit(self, tried_result, committed_result) -> None:
        self.commits += 1
        self.sim_seconds += committed_result.sim_seconds
        self.seen_states.append(committed_result.state_after)
        agreed = committed_result.state_after == tried_result.state_after
        self.log.append({"event": "commit", "action": committed_result.action, "status": committed_result.status,
                         "moved_cm": committed_result.moved, "goal_after": committed_result.goal,
                         "matches_try": agreed, "sim_s": committed_result.sim_seconds})
        if not agreed:
            raise RuntimeError("the committed action ended differently from its try - the simulation is not "
                               "deterministic here; stop and inspect")

    # ---------------------------------------------------------- the stop decision
    def start(self, state: str) -> None:
        self.seen_states.append(state)

    def end_round(self, improved: bool) -> None:
        self.stale_rounds = 0 if improved else self.stale_rounds + 1

    def decide(self, goal: dict[str, Any], untried_left: int) -> Decision:
        if goal.get("satisfied"):
            return Decision(True, "simulated_success",
                            f"goal reached in the simulated room after {self.commits} committed action(s)")
        if goal.get("feasible") is False:
            return Decision(True, "infeasible", goal.get("why", "the goal cannot be reached"),
                            self._help("Nothing I can move changes this. A person needs to decide (e.g. move fixed "
                                       "furniture or pick another route)."))
        if self.tries >= self.limits.max_tries or self.commits >= self.limits.max_commits \
                or self.sim_seconds >= self.limits.max_sim_seconds or self.spent_usd >= self.limits.max_usd:
            return Decision(True, "budget", f"budget spent ({self.tries} tries, {self.commits} commits, "
                                            f"{self.sim_seconds:.0f} simulated s, ${self.spent_usd:.4f})",
                            self._help("I ran out of my action budget before reaching the goal."))
        if self.stale_rounds >= self.limits.patience:
            return Decision(True, "stuck", f"no improving action in {self.stale_rounds} rounds of tries",
                            self._help("Every action I tried failed or did not bring the goal measurably closer."))
        if untried_left == 0:
            return Decision(True, "stuck", "no untried actions left",
                            self._help("I have no other action to try."))
        return Decision(False)

    def _help(self, lead: str) -> str:
        if not self.tries:
            return f"{lead} I did not try any action: none could change this."
        last = self.failures[-6:]
        tried = "; ".join(last) if last else "every try succeeded but none improved the goal"
        if len(self.failures) > len(last):
            tried = f"{tried} (and {len(self.failures) - len(last)} earlier)"
        return f"{lead} What I tried (in simulation): {tried}."

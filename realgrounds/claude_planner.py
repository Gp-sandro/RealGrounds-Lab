"""Claude as the planner: reads the simulated room and the measured goal, proposes the next actions.

It only proposes. Every proposal is a plain-English command that goes through the same command layer, sandbox
and supervisor as the rule-based planner's (realgrounds/autonomy.py): a proposal that names a missing object or
an impossible action is refused or fails there, and that is measured. Claude never decides that it succeeded.

Spend is counted from the API's token usage at list prices (see PRICES), so the supervisor can stop a task, and
the night run can stop itself, on money as well as on tries.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic

from . import config  # noqa: F401  (loads the API key from .env)
from .sandbox import SimWorld

DEFAULT_MODEL = os.environ.get("REALGROUNDS_MODEL", "claude-haiku-5-5")

# USD per million tokens (input, output) - Anthropic list prices, October 2026. Cache reads are billed at 0.1x
# and cache writes at 1.25x the input price.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

SYSTEM = """You plan actions for a simulated robot body in a SIMULATED copy of a real room. The copy was built \
from a photo: object sizes and positions are estimates, masses are catalog values. Nothing you propose happens \
in the real room.

The body: about 0.4 m wide, walks on the floor, steps over things lower than 20 cm, holds one thing at a time, \
and its arm lifts at most 15 kg. Fixed furniture cannot be moved.

Write each action as one plain-English command, using object labels exactly as given (e.g. "box #2"):
  pick up <object>
  put <object> on <surface object>        (or: put <object> on the floor)
  pull out <object>        push <object>        walk to <object>        drop <object>
Commands can be joined: "pick up book #4 and put it on shelf #1".

How your proposals are used: each one is TRIED on a snapshot of the simulated room, measured, and undone. Only \
an action that measurably brings the goal closer is then kept. An action already tried in the same situation is \
refused (the simulation is deterministic: it would end the same way). So do not repeat tried actions; learn from \
their measured results and propose something different, or propose nothing if no action can help."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "one or two sentences: why these actions"},
        "actions": {"type": "array", "items": {"type": "string"}, "description": "best first; empty if none can help"},
    },
    "required": ["reasoning", "actions"],
    "additionalProperties": False,
}


class PlannerError(RuntimeError):
    """The planner could not answer (API unreachable, refused, cut off). The supervisor stops the task."""


def room_for_prompt(env: SimWorld) -> dict[str, Any]:
    """The observation, trimmed to what planning needs (smaller prompt, same facts)."""
    obs = env.observe()
    return {
        "note": obs["notes"],
        "agent": obs["agent"],
        "objects": [{k: o[k] for k in ("label", "movable_in_simulation", "affordances", "position_m", "footprint_m",
                                         "height_m", "rests_on_now", "mass")} for o in obs["objects"]],
    }


def cost_usd(model: str, usage) -> float:
    pin, pout = PRICES.get(model, PRICES["claude-opus-5-5"])
    cached_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cached_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (usage.input_tokens * pin + cached_read * pin * 0.1 + cached_write * pin * 1.25
            + usage.output_tokens * pout) / 1e6


@dataclass
class ClaudePlanner:
    model: str = DEFAULT_MODEL
    effort: str = "medium"
    max_actions: int = 4
    spent_usd: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    last_reasoning: str = ""
    client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = anthropic.Anthropic(max_retries=4)

    def ask(self, system: str, user: str, schema: dict, max_tokens: int = 8000) -> dict[str, Any]:
        """One structured call; returns the parsed JSON. Counts tokens and dollars."""
        try:
            r = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": user}])
        except anthropic.APIConnectionError as e:
            raise PlannerError(f"cannot reach the API: {e}") from e
        except anthropic.RateLimitError as e:
            raise PlannerError("rate limited by the API (after retries)") from e
        except anthropic.APIStatusError as e:
            raise PlannerError(f"API error {e.status_code}: {e.message}") from e
        self.calls += 1
        self.input_tokens += r.usage.input_tokens
        self.output_tokens += r.usage.output_tokens
        self.spent_usd += cost_usd(self.model, r.usage)
        if r.stop_reason == "refusal":
            raise PlannerError(f"the model declined ({getattr(r.stop_details, 'category', None)})")
        if r.stop_reason == "max_tokens":
            raise PlannerError("the answer was cut off (max_tokens)")
        text = next((b.text for b in r.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise PlannerError(f"unreadable answer: {text[:120]!r}") from e

    def __call__(self, env: SimWorld, goal: dict[str, Any], history: list[dict[str, Any]] | None = None) -> list[str]:
        tried = [{"action": h["action"], "status": h["status"], "result": h.get("result", ""),
                  "goal_after": _goal_brief(h.get("goal_after"))} for h in history or [] if h["event"] == "try"]
        kept = [h["action"] for h in history or [] if h["event"] == "commit"]
        user = json.dumps({
            "goal": {"description": goal.get("goal"), "measured_now": _goal_brief(goal),
                     "blocked_by": goal.get("blocked_by") or []},
            "room": room_for_prompt(env),
            "already_kept_actions": kept,
            "already_tried_actions": tried,
            "ask": f"Propose up to {self.max_actions} actions, best first.",
        }, indent=1, default=str)
        out = self.ask(SYSTEM, user, SCHEMA)
        self.last_reasoning = str(out.get("reasoning", ""))[:400]
        return [str(a).strip() for a in out.get("actions", []) if str(a).strip()][:self.max_actions]


def _goal_brief(g: dict[str, Any] | None) -> str:
    if not g:
        return "-"
    if g.get("satisfied"):
        return "satisfied"
    return g.get("why") or (f"{g.get('progress_cm')} cm blocked" if g.get("progress_cm") is not None else "not satisfied")

"""The night's supervisor: watches the whole run, not one task, and stops it when the RUN goes wrong.

    money        the night's dollar budget; a room is only started if a full room's budget is left; a room that
                 spends its own budget has its remaining tasks skipped (and queued for a person)
    time         a wall-clock limit
    failure      N rooms in a row failing to build = the pipeline itself is broken; N planner errors in a row =
    streaks      the API is unreachable. Either stops the night instead of burning through every room.
    timeline     every event with its time, as JSON lines (timeline.jsonl) - the evidence of what it did alone
    help queue   what a person must look at in the morning, each item with the reason
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class NightLimits:
    max_usd: float = 2.0
    room_usd: float = 0.20
    max_hours: float = 12.0
    build_failures_in_row: int = 3
    planner_errors_in_row: int = 3


@dataclass
class NightSupervisor:
    limits: NightLimits
    out: Path
    say: Any = print
    spent_usd: float = 0.0
    rooms_done: int = 0
    build_fail_streak: int = 0
    planner_err_streak: int = 0
    stopped: str = ""
    help_queue: list[dict[str, Any]] = field(default_factory=list)
    t0: float = field(default_factory=time.time)

    def event(self, kind: str, **data: Any) -> None:
        row = {"time": datetime.now().isoformat(timespec="seconds"), "elapsed_s": round(time.time() - self.t0, 1),
               "event": kind, **data}
        with (self.out / "timeline.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        brief = ", ".join(f"{k}={v}" for k, v in data.items() if not isinstance(v, (list, dict)))[:220]
        self.say(f"[{row['time'][11:]}] {kind}: {brief}")

    def ask_help(self, room: str, what: str, why: str) -> None:
        self.help_queue.append({"room": room, "what": what, "why": why})
        self.event("help_request", room=room, what=what, why=why)

    def may_start_room(self) -> str | None:
        hours = (time.time() - self.t0) / 3600
        if self.spent_usd + self.limits.room_usd > self.limits.max_usd:
            return (f"money: ${self.spent_usd:.2f} spent of ${self.limits.max_usd:.2f}; not enough left for another "
                    f"room (${self.limits.room_usd:.2f})")
        if hours >= self.limits.max_hours:
            return f"time: {hours:.1f} h of {self.limits.max_hours} h used"
        if self.build_fail_streak >= self.limits.build_failures_in_row:
            return f"{self.build_fail_streak} rooms in a row failed to build - the pipeline itself may be broken"
        if self.planner_err_streak >= self.limits.planner_errors_in_row:
            return f"{self.planner_err_streak} planner errors in a row - the API may be unreachable"
        return None

    def room_finished(self, usd: float, built: bool, planner_error: bool) -> None:
        self.rooms_done += 1
        self.spent_usd += usd
        self.build_fail_streak = 0 if built else self.build_fail_streak + 1
        self.planner_err_streak = self.planner_err_streak + 1 if planner_error else 0

    def stop(self, reason: str) -> None:
        self.stopped = reason
        self.event("night_stopped", reason=reason)

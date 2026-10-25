"""From a sentence to actions: "pick up the keyboard and put it on the bed".

WHAT THIS FILE DOES (in plain words)
    1. parse()    splits the sentence into steps (then / and / commas) and finds the verb and
                  the objects of each step: "put X on Y" -> (place, X, Y).
    2. resolve()  finds which object of the room is meant, using the records the pipeline made:
                  names ("the toy" -> stuffed toy), ids ("pillow #6"), "it" (the last object),
                  relations ("the pillow on the blanket"). Several matches -> the nearest one,
                  and it says so.
    3. check()    asks the object's ATTRIBUTES before moving at all: static furniture can't be
                  picked up, a poster is not a surface, a hanging bag is fixed in this
                  simulation... A refusal says why. This is the core idea of the project: the
                  attributes decide what is possible.
    4. execute()  runs the skills (skills.py) step by step and stops at the first failure.

    No language model: a small fixed grammar, so every decision can be traced. (A language model
    could later write these steps from free speech; the checks and skills stay the same.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .skills import Agent, Result
from .world import Thing, World

VERBS = {   # verb -> the ways to say it (longest first when matching)
    "walk": ["walk over to", "walk to", "go to", "go over to", "approach", "come to", "move to"],
    "pick": ["pick up", "pick", "grab", "take", "lift", "get", "hold"],
    "place": ["put down", "put", "place", "set", "bring", "move"],
    "pull": ["pull out", "pull"],
    "push": ["push"],
    "drop": ["drop", "let go of", "let go", "release"],
}
KNOWN = sorted(((s, v) for v, says in VERBS.items() for s in says), key=lambda t: -len(t[0]))
SYNONYMS = {"toy": ["stuffed toy"], "plushie": ["stuffed toy"], "teddy": ["stuffed toy"], "bed": ["bed", "loft bed", "mattress"],
            "table": ["table", "desk"], "screen": ["monitor"], "computer": ["monitor", "laptop"], "bottle": ["water bottle"],
            "bag": ["backpack", "tote bag", "duffel bag", "plastic bag"], "lights": ["string lights"],
            "sneakers": ["shoes"], "shoe": ["shoes"], "picture": ["picture frame", "poster"],
            "mug": ["cup"], "coffee mug": ["cup"], "glass": ["cup"], "speakers": ["speaker"], "carpet": ["rug"], "mat": ["rug", "mouse pad"],
            "trash": ["trash can"], "garbage can": ["trash can"], "bin": ["storage bin", "trash can"], "wastebasket": ["trash can"],
            "pc": ["computer tower"], "tower": ["computer tower"], "light": ["lamp", "ceiling light"]}
RELATIONS = {"on": "on", "on top of": "on", "off": "on", "off of": "on", "from": "on", "under": "under", "beneath": "under",
             "next to": "near", "near": "near", "by": "near"}
FLOOR_WORDS = ("floor", "ground")


@dataclass
class Step:
    verb: str                       # walk / pick / place / pull / push / drop
    obj: str = ""                   # the words for the object ("the pillow on the blanket")
    target: str = ""                # place: the words for where it goes
    text: str = ""                  # the original words


@dataclass
class Outcome:
    """What happened to one sentence."""

    text: str
    status: str                     # done / refused / failed
    lines: list[str] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)


# ------------------------------------------------------------------------------------------
# 1. parse
# ------------------------------------------------------------------------------------------

def parse(text: str) -> list[Step] | str:
    """Sentence -> steps. Returns an error sentence if a part can't be understood."""
    t = text.lower().strip().rstrip(".!?")
    t = re.sub(r"\b(please|can you|could you|now)\b", " ", t)
    verbs = "|".join(re.escape(s) for s, _ in KNOWN)
    parts = re.split(rf"\s*(?:,|;|\bthen\b|\band then\b|\band\b(?=\s+(?:{verbs})\b))\s*", t)
    steps = []
    for part in (p.strip() for p in parts):
        if not part or part in ("and", "then"):
            continue
        hit = next(((s, v) for s, v in KNOWN if part == s or part.startswith(s + " ")), None)
        if hit is None:
            can = "walk to, pick up, put ... on ..., pull out, push, drop"
            return f"I don't know how to '{part.split()[0]}' yet (I can: {can})"
        said, verb = hit
        rest = part[len(said):].strip()
        if said == "put down":
            verb, rest = "place", rest + " on the floor"
        if verb == "place":
            m = re.match(r"(.*?)\s+(?:on top of|onto|on|into|in|to|at)\s+(.+)$", rest)
            if m is None:
                if said == "move":                     # "move the chair" = push it
                    steps.append(Step("push", rest, text=part))
                    continue
                return f"where should '{rest}' go? (say: put {rest} on ...)"
            steps.append(Step("place", m.group(1).strip(), m.group(2).strip(), part))
        elif verb == "drop":
            steps.append(Step("drop", rest, text=part))
        else:
            steps.append(Step(verb, rest, text=part))
    return steps or "nothing to do"


# ------------------------------------------------------------------------------------------
# 2. resolve
# ------------------------------------------------------------------------------------------

def _clean(words: str) -> str:
    return re.sub(r"\b(the|a|an|my|your|this|that|some)\b", " ", words).strip()


def _names(t: Thing) -> set[str]:
    n = {t.name, t.kind.replace("_", " ")}
    if t.info.get("prompt"):
        n.add(str(t.info["prompt"]).lower())
    return n


def resolve(words: str, world: World, last: Thing | None, near: np.ndarray,
            rank: Any = None) -> tuple[Thing | None, str]:
    """Words -> (thing, note). thing None = not found (note says why).
    Several matches: the best by `rank(thing)` (lower = better; default: the nearest)."""
    w = _clean(words.lower())
    if w in ("it", "them", "") and last is not None:
        return last, ""
    if w in ("it", "them", ""):
        return None, "which object? ('it' has nothing to refer to yet)"
    m = re.search(r"#\s*(\d+)", w)
    if m:
        for t in world.things.values():
            if t.label.endswith(f"#{m.group(1)}"):
                return t, ""
        return None, f"there is no object #{m.group(1)}"
    # split off a relation: "pillow on the blanket" -> head "pillow", (on, "blanket")
    rel, other = None, None
    for say, kind in sorted(RELATIONS.items(), key=lambda kv: -len(kv[0])):
        mm = re.match(rf"(.+?)\s+{re.escape(say)}\s+(.+)$", w)
        if mm:
            w, rel, other = mm.group(1).strip(), kind, mm.group(2).strip()
            break
    wanted = {w, w.rstrip("s")} | set(SYNONYMS.get(w, [])) | set(SYNONYMS.get(w.rstrip("s"), []))
    exact = [t for t in world.things.values() if t.name in (w, w.rstrip("s"))]   # "box" means the box, not the heavy box
    found = exact or [t for t in world.things.values() if _names(t) & wanted]
    if not found:   # last word only: "the blue pillow" -> pillow
        tail = w.split()[-1] if w.split() else w
        found = [t for t in world.things.values() if any(n.split()[-1] in (tail, tail.rstrip("s")) for n in _names(t))]
    if not found:
        have = ", ".join(sorted({t.name for t in world.things.values()}))
        return None, f"I don't see any '{w}' here (I see: {have})"
    if rel is not None:
        # "the chair next to the bed" with two beds: next to ANY bed, not just the nearest one
        refs = _all_named(other, world, last)
        if not refs:
            _, note = resolve(other, world, last, near)
            return None, note

        def related(t: Thing, ref: Thing) -> bool:
            if rel == "on":                              # resting on it, or hanging on it
                return world.rests_on(t, ref.label) or (t.hangs_on == ref.label and world.hanging(t))
            if rel == "under":
                r = t.record["Relationships"] or {}
                return ref.label in r.get("under", []) + r.get("partly_under", [])
            rh = world.extent(ref)["hull"].astype(np.float32)  # near = within half a metre of its edge
            return min(-cv2.pointPolygonTest(rh, (float(q[0]), float(q[1])), True) for q in world.extent(t)["hull"]) < 0.5

        found = [t for t in found if t not in refs and any(related(t, r) for r in refs)]
        if not found:
            return None, f"no {w} {rel} {' / '.join(r.label for r in refs)}"
    if len(found) == 1:
        return found[0], ""
    dist = lambda t: float(np.linalg.norm(world.extent(t)["center"][:2] - near))  # noqa: E731
    found.sort(key=(lambda t: (rank(t), dist(t))) if rank else dist)
    why = "the nearest one I can use" if rank else "the nearest"
    return found[0], f"{len(found)} match '{w}' - chose {why}, {found[0].label}"


def _all_named(words: str, world: World, last: Thing | None) -> list[Thing]:
    """Every object these words could mean (no choosing yet)."""
    w = _clean(words.lower())
    if w in ("it", "them") and last is not None:
        return [last]
    m = re.search(r"#\s*(\d+)", w)
    if m:
        return [t for t in world.things.values() if t.label.endswith(f"#{m.group(1)}")]
    wanted = {w, w.rstrip("s")} | set(SYNONYMS.get(w, [])) | set(SYNONYMS.get(w.rstrip("s"), []))
    exact = [t for t in world.things.values() if t.name in (w, w.rstrip("s"))]
    return exact or [t for t in world.things.values() if _names(t) & wanted]


# ------------------------------------------------------------------------------------------
# 3. check the attributes
# ------------------------------------------------------------------------------------------

def check(verb: str, t: Thing, world: World, target: Thing | None = None) -> str | None:
    """Can this be done to `t`, by its attributes? None = yes, else the reason why not."""
    mass = world.mass(t)
    kg = f", ~{mass:.0f} kg" if mass else ""
    if verb in ("pick", "place", "pull", "push"):
        if not t.movable:
            return f"{t.label} is static furniture ({t.record.get('Material', '?')}{kg}) - it stays where it is"
        if t.body is None:
            where = (t.record.get("Position") or "").lower()
            return f"{t.label} is held in place in this simulation ({where}) - its support was not seen, so it can't be moved safely"
    if verb in ("pick", "place") and not ({"grasp", "lift", "carry"} & set(t.affordances)):
        return f"{t.label} can't be picked up (it affords: {', '.join(t.affordances)})"
    if verb == "pull" and not ({"pull", "move", "pull_out"} & set(t.affordances)):
        return f"{t.label} can't be pulled (it affords: {', '.join(t.affordances)})"
    if verb == "push" and not ({"push", "move"} & set(t.affordances)):
        return f"{t.label} can't be pushed (it affords: {', '.join(t.affordances)})"
    if verb == "place" and target is not None:
        if target is t:
            return f"{t.label} can't be put on itself"
        if not target.is_surface:
            return f"{target.label} is not something to put things on ({target.record.get('Position', '')})"
    return None


# ------------------------------------------------------------------------------------------
# 4. execute
# ------------------------------------------------------------------------------------------

class Commander:
    """Keeps 'it' between sentences and runs them on one agent."""

    def __init__(self, world: World, agent: Agent) -> None:
        self.w, self.agent = world, agent
        self.last: Thing | None = None
        self._picked_from: str | None = None       # where the held thing was taken from
        self.missing = ""                          # words of the last object that could not be found

    def run(self, text: str) -> Outcome:
        out = Outcome(text, "done")
        steps = parse(text)
        if isinstance(steps, str):
            return Outcome(text, "refused", [steps])
        held_before = self.w.held
        for step in steps:
            r = self._one(step, out)
            if r is None or not r.ok:
                if r is not None:
                    out.lines.append("✗ " + r.message)
                    out.facts.append({"step": step.text, "ok": False, **r.facts})
                out.status = "refused" if r is None else "failed"
                self._put_back(held_before, out)
                return out
            out.lines.append("✓ " + r.message)
            out.facts.append({"step": step.text, "ok": True, **r.facts})
        return out

    def _put_back(self, held_before: Thing | None, out: Outcome, why: str = "") -> None:
        """Put the held thing back where it came from: when a sentence failed half-way (so the
        next one starts with free hands), or when the hands are needed for something else."""
        t = self.w.held
        if t is None or t is held_before:
            return
        src = self._picked_from or "floor"
        target = self.w.things.get(src) if src != "floor" else None
        r = self.agent.put_on(t, target)
        if not r.ok and target is not None:
            r = self.agent.put_on(t, None)
        if not r.ok:
            r = self.agent.drop()
        lead = f"put it back ({why}): " if why else "put it back: "
        out.lines.append(("  ↩ " if r.ok else "  ✗ ") + (lead if r.ok else "") + r.message)

    def _one(self, step: Step, out: Outcome) -> Result | None:
        me = self.w.base()[0]
        held = self.w.held
        if step.verb == "drop" and not step.obj:
            return self.agent.drop()
        if _clean(step.obj.lower()) in ("it", "them", "that") and held is None and self.missing:
            # "pick up the mug" found no mug: "put it on the floor" must not grab something else
            out.lines.append(f"✗ 'it' would be '{self.missing}', which I could not find")
            return None
        t, note = resolve(step.obj, self.w, held or self.last, me)
        if t is None:
            out.lines.append("✗ " + note)
            self.missing, self.last = step.obj, None
            return None
        self.missing = ""
        if note:
            out.lines.append("  " + note)
        self.last = t
        target = None
        if step.verb == "place" and _clean(step.target) not in FLOOR_WORDS:
            # "the bed" in a dorm with a loft bed: prefer a surface the arm can reach that has room
            usable = lambda c: (not c.is_surface, bool(self.agent.too_high(t, c, margin=0.15)),  # noqa: E731
                                c is not t and not self.agent.has_room(t, c))
            target, note = resolve(step.target, self.w, None, me, rank=usable)
            if target is None:
                out.lines.append("✗ " + note)
                return None
            if note:
                out.lines.append("  " + note)
        why = check(step.verb, t, self.w, target) if step.verb != "walk" else None
        if why:
            out.lines.append("✗ refused: " + why)
            return None
        if step.verb in ("pick", "place", "pull", "push") and self.w.held is not None and self.w.held is not t:
            # hands are needed but full: put the held thing back where it came from first
            self._put_back(None, out, why="hands were full")
        if step.verb == "walk":
            return self.agent.walk_to(t)
        if step.verb == "pick":
            return self._pick(t)
        if step.verb == "place":
            high = self.agent.too_high(t, target)      # look before picking it up
            if high:
                return Result(False, high)
            if not self.agent.has_room(t, target):
                return Result(False, f"no free space on {target.label if target else 'the floor'} for {t.label}")
            if self.w.held is not t:
                r = self._pick(t)
                if not r.ok:
                    return r
                out.lines.append("✓ " + r.message)
                out.facts.append({"step": f"pick up {t.label}", "ok": True, **r.facts})
            return self.agent.put_on(t, target)
        if step.verb == "pull":
            rel = t.record.get("Relationships") or {}
            under = [self.w.things[x] for x in rel.get("under", []) + rel.get("partly_under", []) if x in self.w.things]
            return self.agent.pull_out(t, under[0] if under else None)
        if step.verb == "push":
            return self.agent.push(t)
        if step.verb == "drop":
            if self.w.held is not t:
                return Result(False, f"not holding {t.label}")
            return self.agent.drop()
        return Result(False, f"unknown step {step.verb}")

    def _pick(self, t: Thing) -> Result:
        below = self.w.resting_on(t)
        r = self.agent.pick_up(t)
        if r.ok and self.w.held is t:
            self._picked_from = next((b for b in below if b in self.w.things), "floor" if "floor" in below else None)
        return r

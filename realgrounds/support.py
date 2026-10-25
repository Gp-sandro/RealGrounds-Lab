"""What does an object rest on? Choose among ALL candidates (the floor and every surface) by geometry AND common sense.

WHY (room benchmark, lab/experiments/room_benchmark/notes.md)
    The first rule that matched used to win: "bottom within 8 cm of a surface's top, centre over it, the
    highest such surface", else the floor if low, else "support not seen". On 40 never-seen rooms that
    got "rests on the floor" right 95 % of the time but "rests on another object" only 32 %: a keyboard
    whose estimated bottom was 20 cm low went to the floor, a monitor whose stand wasn't seen went to "not
    seen", a pillow went onto the pillow next to it (its top is higher than the bed's).

HOW
    Every candidate gets a score = how well the heights meet  x  how much of the object lies over it
    x  how often that kind of thing really rests on that kind of thing (support_prior.yaml, counted from
    the dataset's TRAIN split: a keyboard rests on a desk 68 %, on the floor 0 %; a pillow on a bed 53 %,
    on another pillow 4 %). The best candidate wins if its geometry is still plausible (within
    `support_above_max_m` above / `support_below_max_m` below the surface, `support_min_overlap` of the
    footprint over it); otherwise the caller falls back to its other rules (raised platform, wall,
    hanging, "support not seen").
"""

from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PRIOR_FILE = Path(__file__).resolve().parent / "support_prior.yaml"
ALIAS = {"mattress": "bed", "loft_bed": "bed"}   # our finer kinds -> the dataset's label for the same thing
_PRIOR: "SupportPrior | None" = None


class SupportPrior:
    """P(how a kind is held | kind): resting on each supporter kind ("below") or hanging on a wall
    ("behind"), smoothed towards how common each answer is overall (kinds with few examples)."""

    def __init__(self, below: dict[str, dict[str, int]], behind: dict[str, dict[str, int]], alpha: float):
        self.below, self.alpha = below, alpha
        self.wall = {k: v.get("wall", 0) for k, v in behind.items()}
        total = collections.Counter()
        for counts in below.values():
            total.update(counts)
        n_wall = sum(self.wall.values())
        n = (sum(total.values()) + n_wall) or 1
        self.base = {k: v / n for k, v in total.items()}
        self.base_wall = n_wall / n

    def _n(self, kind: str) -> int:
        return sum(self.below.get(kind, {}).values()) + self.wall.get(kind, 0)

    def p(self, kind: str, supporter: str) -> float:
        """P(it rests on `supporter`) - a kind, or "floor". A mattress / loft bed counts as the "bed" the
        dataset labels (it labels the whole bed, mattress and bedding, as one thing)."""
        best = 0.0
        for sup in {supporter, ALIAS.get(supporter, supporter)}:
            base = max(self.base.get(sup, 0.0), 0.002)       # never quite impossible
            best = max(best, (self.below.get(kind, {}).get(sup, 0) + self.alpha * base) / (self._n(kind) + self.alpha))
        return best

    def p_wall(self, kind: str) -> float:
        """P(it hangs on / is fixed to a wall)."""
        return (self.wall.get(kind, 0) + self.alpha * self.base_wall) / (self._n(kind) + self.alpha)


def prior(alpha: float = 5.0) -> SupportPrior:
    global _PRIOR
    if _PRIOR is None or _PRIOR.alpha != alpha:
        doc = yaml.safe_load(PRIOR_FILE.read_text()) if PRIOR_FILE.exists() else {}
        _PRIOR = SupportPrior(doc.get("below") or {}, doc.get("behind") or {}, alpha)
    return _PRIOR


def _height_fit(dz: float, sigma: float, above_max: float, below_max: float) -> float:
    """dz = object's bottom minus the surface's top. 1 when they meet; 0 outside the allowed window.
    Above (a gap) is the usual estimate error: a monitor's thin stand is not seen, so its bottom looks
    high. Below (sinking in) happens with soft supports and depth smear - allowed but less."""
    if dz > above_max or dz < -below_max:
        return 0.0
    s = sigma if dz >= 0 else sigma * 0.75
    return math.exp(-0.5 * (dz / s) ** 2)


INNER_FIT = 0.6     # how well "rests on an inner surface" fits, against 1.0 for resting exactly on the top


def inner_surface(a, b, over: float, sink: float) -> bool:
    """Does A's bottom lie on a surface INSIDE B's box (seat, shelf board, desk top under a hutch)?
    Seats, desk tops and beds have a typical height (catalog surface_m): A's bottom must be near it and A
    almost entirely over B (a backpack hanging off a chair's back is not on its seat). Shelves and
    cabinets have boards at any height."""
    if not b.info.get("inner_surface") or not (b.zmin + 0.1 <= a.zmin < b.top - sink):
        return False
    level = b.info.get("surface_m")
    if level is not None:
        # resting on a seat means its bottom is AT the seat (a little estimate noise below, more above for a
        # thick cushion): a backpack whose bottom hangs below the seat is hanging, not sitting there
        return over >= 0.8 and -0.06 <= a.zmin - b.zmin - level <= 0.15
    return over >= 0.6


def choose(a, objects: list, cfg: dict[str, Any], floor_ok: bool, excluded, walls=()) -> tuple[Any, float]:
    """Best support for `a`: a RoomObject, "floor", "wall", or None (nothing plausible). Also returns its score.

    `excluded(b)` says a candidate may never hold `a` (e.g. furniture on a book). `floor_ok`: the floor
    is a candidate (it always is for things near it; the caller decides)."""
    from .room import _grid  # (late import: room imports this module)

    pr = prior(cfg.get("support_prior_alpha", 5.0))
    sigma = cfg.get("support_sigma_m", 0.08)
    above_max = cfg.get("support_above_max_m", 0.35)
    below_max = cfg.get("support_below_max_m", 0.25)
    min_over = cfg.get("support_min_overlap", 0.3)
    pts = _grid(a, 5)
    by_label = {o.label: o for o in objects}
    best, best_score = None, 0.0
    inside_box = False          # A lies inside another thing's box (a picture inside a bookcase)
    sits_on = 0.0               # how well A's bottom sits on a surface's top (a picture frame standing on a table)
    for b in objects:
        if b is a:
            continue
        if b.info.get("support") or b.info["state"] == "static":
            o_ = float(np.mean([b.contains_xy(p, 0.05) for p in pts]))
            inside_box |= o_ >= 0.6 and b.zmin + 0.1 <= a.zmin < b.top - 0.02
        if not b.info.get("support") or excluded(b):
            continue
        # how much of A's footprint lies over B (a little margin: boxes are estimates)
        over = float(np.mean([b.contains_xy(p, 0.05) for p in pts]))
        if over < min_over and not b.contains_xy(a.center[:2], 0.05):
            continue
        # a thin support (a book) can't have things sunk into it
        sink = min(below_max, 0.5 * (b.top - b.zmin) + 0.02)
        h = _height_fit(a.zmin - b.top, sigma, above_max, sink) if b.top < a.top else 0.0
        if h <= 0 and b.kind in (a.info.get("set_into") or []) and over >= 0.6:
            # set INTO it (a sink in a counter): its top is level with the surface, the basin hangs below
            h = _height_fit(a.top - b.top, sigma, above_max, below_max)
        if h <= 0 and inner_surface(a, b, over, sink):
            # a box hides surfaces INSIDE it: a sofa's seat under its back, a bookcase's boards, a desk
            # top under its hutch - A lies well inside the box, mostly over it: it rests in there
            h = INNER_FIT
        if h <= 0:
            continue
        # a cover (blanket, rug, mouse pad) lying on something takes its role: a pillow on the blanket
        # on a mattress is as likely as a pillow on the mattress
        under = by_label.get(b.support or "") if b.info.get("cover") else None
        kind = under.kind if under is not None else ("floor" if b.info.get("cover") and b.support == "floor" else b.kind)
        score = h * max(over, 0.3) * max(pr.p(a.kind, b.kind), pr.p(a.kind, kind))
        if over >= 0.5 and h != INNER_FIT:
            sits_on = max(sits_on, h)
        if score > best_score:
            best, best_score = b, score
    if floor_ok:
        h = _height_fit(a.zmin, sigma * 1.5, cfg.get("floor_tol_m", 0.15) + 0.1, 0.3)
        score = h * pr.p(a.kind, "floor")
        if score > best_score:
            best, best_score = "floor", score
    # the wall: for things clearly above the floor, right against a wall that was seen - or very high
    # up when no wall was found (a picture at 1.4 m hangs on something behind it)
    # (only kinds that commonly hang: pictures, windows, curtains, mirrors, tote bags - not a fan or a blanket)
    # ...and not when it hangs right at something made for hanging (a tote bag on a loft bed's rail): the
    # hanging rule (room.relate) knows that better than "bags often hang on walls"
    from .room import _footprint_gap
    at_hook = any(b is not a and b.info.get("hang_point") and _footprint_gap(a, b) < cfg.get("hang_gap_m", 0.12)
                  and b.zmin - 0.05 <= a.top <= b.top + 0.15 for b in objects)
    # ...and not from inside another thing's box: a picture inside a bookcase stands on a shelf board
    if (a.zmin > cfg.get("wall_min_m", 0.3) and pr.p_wall(a.kind) >= cfg.get("wall_prior_min", 0.25) and not at_hook
            and not inside_box):
        near = min((float(np.min(wl.distance(a.footprint()))) for wl in walls), default=None)
        h = 1.0 if near is not None and near < cfg.get("wall_tol_m", 0.15) + 0.15 else \
            0.5 if a.zmin > cfg.get("wall_unseen_min_m", 0.9) else 0.0
        # A hanging thing's bottom lines up with a surface's top only by coincidence; a standing one always
        # does: that evidence outweighs "pictures usually hang"
        if sits_on >= 0.8:      # bottom right on the top (within ~5 cm)
            h *= cfg.get("wall_if_sitting", 0.03)
        elif sits_on >= 0.5:
            h *= 0.1
        score = h * pr.p_wall(a.kind)
        if score > best_score:
            best, best_score = "wall", score
    return best, best_score

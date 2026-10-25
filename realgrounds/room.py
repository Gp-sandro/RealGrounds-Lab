"""Indoor training ground: one or more room photos -> 3D room + objects with properties & relationships.

WHAT THIS FILE DOES (in plain words)
    The outdoor pipeline maps a park from above. A room is different: it is a box (floor,
    walls, ceiling) with furniture standing in it and small things resting ON the furniture.
    This file builds exactly that, from what the photo shows:

      1. WHICH WAY IS UP + the floor: fitted to floor pixels; if little floor is visible, to
         the top of a desk/table/bed (then the floor is put the catalog height below it); if
         neither, to two walls (rooms are boxes: two walls' directions give "up").
      2. WALLS + CEILING: wall pixels -> vertical planes (as lines on the floor plan, robust);
         walls are snapped to right angles (rooms are rectangular) -> the room rectangle.
         Sides of the room the photo never saw are closed with walls flagged `inferred`.
      3. OBJECTS: every detected outline (SAM mask) + depth -> 3D points -> a box aligned with
         the room. One photo only shows an object's front: if it looks too shallow, the back
         is completed from the catalog's typical size (flagged).
      4. RELATIONSHIPS from geometry: supported_by (rests on top of), next_to, under,
         against / leaning_on a wall, mounted_on a wall. Objects are snapped onto their support.
      5. RECORDS in the agreed format: Object, Boundary, Position, Material [estimated],
         State, Affordances, Relationships, Collision, Mass (+ size, confidence, seen in which photos).

    Several photos of the same room: each is built on its own, then aligned (matching image
    features + their 3D positions; floor = z 0 in both, so only turn + shift + scale), and the
    same object seen in two photos becomes ONE object seen twice (persistence).

WHERE IT FITS
    lab/room_world.py runs it and exports physics, 3D surface, renders and the presentation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .geometry import GroundPlane, backproject, depth_edges, intrinsics, to_world
from .support import choose as choose_support

log = logging.getLogger(__name__)

# Furniture that stands on the floor even when its legs are hidden (e.g. a chair behind a desk).
FLOOR_STANDING = ("bed", "loft_bed", "ladder", "desk", "table", "chair", "dresser", "nightstand", "wardrobe", "shelf",
                  "mini_fridge", "trash_can", "door", "sofa", "ottoman", "refrigerator", "stove", "dishwasher", "toilet",
                  "bathtub")


FLOOR_ONLY_SUPPORTS = ("storage_bin", "box", "book", "notebook")  # furniture never stands on these


# (part, whole): a part's box inside its whole's box is the same thing seen twice.
PART_OF = {("mattress", "bed"), ("mattress", "loft_bed")}


def platform_slab(o: "RoomObject") -> float:
    """Thickness of a platform's top: a desk top is thin, a loft bed's (frame + mattress) is thick,
    a raised mattress is as thick as it was measured."""
    if getattr(o, "raised", 0.0):
        return float(min(o.raised, 0.35))
    return 0.25 if o.kind == "loft_bed" else 0.03

# ADE20K class names (SegFormer) that make up the room's structure.
FLOOR_ADE = ("floor", "rug")
WALL_ADE = ("wall",)
CEILING_ADE = ("ceiling",)


# ------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------

MIN_THICK_M = 0.01   # thinnest box in physics (room_export pads thinner things to this)


@dataclass
class Wall:
    """A vertical wall, as a line on the floor plan: points X with n . X_xy + d = 0."""

    normal: np.ndarray        # (2,) unit, pointing INTO the room (towards the camera side)
    offset: float
    inferred: bool
    points: int = 0           # wall pixels that support it (0 for inferred walls)

    def distance(self, xy: np.ndarray) -> np.ndarray:
        return xy @ self.normal + self.offset


@dataclass
class RoomObject:
    id: int
    kind: str                         # catalog key
    name: str                         # display name, e.g. "water bottle"
    score: float                      # detector confidence
    center: np.ndarray                # (3,) world centre of the box (x, y, z)
    size: np.ndarray                  # (3,) box extent along (room u, room v, z)
    yaw: float                        # box rotation = room axis angle (radians)
    zmin: float
    top: float
    mask: np.ndarray                  # (h, w) bool outline in the analysis image (Boundary)
    info: dict[str, Any]              # catalog entry
    back_inferred: bool = False
    bottom_inferred: bool = False
    relations: dict[str, list[str]] = field(default_factory=dict)
    support: str | None = None        # name of what it rests on ("floor", "desk #3", ...)
    attached_to: str | None = None    # hangs on this (tote bag on a bed rail): fixed in physics
    raised: float = 0.0               # >0: a raised surface on a hidden frame (mattress on a bed frame); slab thickness
    seen_in: list[str] = field(default_factory=list)
    points: int = 0
    notes: list[str] = field(default_factory=list)   # what was inferred rather than seen (uncertainty)
    inner_support: bool = False       # rests on a surface INSIDE its support's box (sofa seat, shelf board)

    @property
    def label(self) -> str:
        return f"{self.name} #{self.id}"

    def footprint(self) -> np.ndarray:
        """(4, 2) corners of the box seen from above."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        u, v = np.array([c, s]), np.array([-s, c])
        hx, hy = self.size[0] / 2, self.size[1] / 2
        cxy = self.center[:2]
        return np.array([cxy + a * hx * u + b * hy * v for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

    def contains_xy(self, xy: np.ndarray, pad: float = 0.0) -> bool:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        d = xy - self.center[:2]
        return abs(d @ np.array([c, s])) <= self.size[0] / 2 + pad and abs(d @ np.array([-s, c])) <= self.size[1] / 2 + pad


@dataclass
class RoomView:
    """Everything built from ONE photo (in that photo's own world frame)."""

    name: str
    image: np.ndarray                 # full photo (RGB)
    small: np.ndarray                 # analysis-size photo
    depth: np.ndarray                 # (h, w) metres (scaled)
    k: np.ndarray                     # camera matrix at analysis size
    fov_deg: float
    fov_source: str
    plane: GroundPlane                # floor in camera coordinates (normal = up, offset = camera height)
    up_source: str
    world: np.ndarray                 # (h, w, 3) world points (planar pixels snapped)
    snapped: np.ndarray               # (h, w) bool: pixel lies exactly on floor / wall / ceiling
    stuff: np.ndarray                 # (h, w) structure label: 0 none, 1 floor, 2 wall, 3 ceiling
    walls: list[Wall]
    room_axis: float                  # angle (rad) of the room's x axis on the floor plan
    ceiling: float
    ceiling_inferred: bool
    objects: list[RoomObject]
    scale: float
    scale_note: str
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------------------
# small geometry helpers
# ------------------------------------------------------------------------------------------

def ransac_plane(pts: np.ndarray, thresh: float, iters: int, rng: np.random.Generator,
                 accept=None) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Best plane n . p + d = 0 through 3D points (RANSAC + least-squares refit). `accept(n)` filters."""
    if len(pts) < 50:
        return None
    best, best_count = None, 0
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if accept is not None and not accept(n):
            continue
        d = -float(n @ s[0])
        count = int((np.abs(pts @ n + d) < thresh).sum())
        if count > best_count:
            best, best_count = (n, d), count
    if best is None:
        return None
    n, d = best
    inl = np.abs(pts @ n + d) < thresh
    c = pts[inl].mean(axis=0)
    n = np.linalg.svd(pts[inl] - c, full_matrices=False)[2][-1]
    d = -float(n @ c)
    if accept is not None and not accept(n):
        n, d = best
    return n, d, np.abs(pts @ n + d) < thresh


def ransac_line(xy: np.ndarray, thresh: float, iters: int, rng: np.random.Generator) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Best 2D line n . p + d = 0 through floor-plan points (a vertical wall seen from above)."""
    if len(xy) < 30:
        return None
    best, best_count = None, 0
    for _ in range(iters):
        a, b = xy[rng.choice(len(xy), 2, replace=False)]
        t = b - a
        if np.linalg.norm(t) < 1e-6:
            continue
        n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
        d = -float(n @ a)
        count = int((np.abs(xy @ n + d) < thresh).sum())
        if count > best_count:
            best, best_count = (n, d), count
    n, d = best
    inl = np.abs(xy @ n + d) < thresh
    c = xy[inl].mean(axis=0)
    direction = np.linalg.svd(xy[inl] - c, full_matrices=False)[2][0]
    n = np.array([-direction[1], direction[0]])
    d = -float(n @ c)
    return n, d, np.abs(xy @ n + d) < thresh


def snap_to_world_plane(points: np.ndarray, cam: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """Slide world points along their viewing rays (from the camera `cam`) onto plane n . X + d = 0."""
    rays = points - cam
    denom = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -(cam @ n + d) / denom
    ok = np.isfinite(t) & (t > 0)
    out = points.copy()
    out[ok] = cam + rays[ok] * t[ok, None]
    return out


# ------------------------------------------------------------------------------------------
# 1. up + floor
# ------------------------------------------------------------------------------------------

def find_up(cam_pts: np.ndarray, valid: np.ndarray, stuff: np.ndarray, objects_px: dict[str, np.ndarray],
            catalog: dict, cfg: dict, rng: np.random.Generator) -> tuple[GroundPlane, str]:
    """Which way is up, and how high the camera is above the floor (camera coordinates)."""
    up_cam = np.array([0.0, -1.0, 0.0])
    horizontal = lambda n: abs(n @ up_cam) > math.cos(math.radians(cfg["max_tilt_deg"]))  # noqa: E731

    def as_plane(n: np.ndarray, d: float, inlier_ratio: float) -> tuple[np.ndarray, float]:
        if n @ up_cam < 0:
            n, d = -n, -d
        return n, d

    floor_px = (stuff == 1) & valid
    if floor_px.mean() >= cfg["min_floor_fraction"]:
        fit = ransac_plane(cam_pts[floor_px], cfg["plane_thresh_m"], 300, rng, horizontal)
        if fit is not None:
            n, d = as_plane(fit[0], fit[1], float(fit[2].mean()))
            if d > 0.3:
                return GroundPlane(n, d, float(fit[2].mean()), True), "floor"
    # Little floor visible: use the top of a desk / table / bed, then put the floor below it.
    for kind in ("desk", "table", "bed", "dresser", "nightstand"):
        m = objects_px.get(kind)
        if m is None or (m & valid).sum() < 400:
            continue
        steep = lambda n: abs(n @ up_cam) > math.cos(math.radians(cfg["max_surface_tilt_deg"]))  # noqa: E731
        fit = ransac_plane(cam_pts[m & valid], cfg["plane_thresh_m"], 300, rng, steep)
        if fit is None or fit[2].mean() < 0.3:
            continue
        n, d = as_plane(fit[0], fit[1], float(fit[2].mean()))
        height = catalog[kind]["size_m"][2]
        if d > 0.05:  # the camera is above this surface: floor = surface - typical height
            return GroundPlane(n, d + height, float(fit[2].mean()), True), f"top of the {kind} (floor {height} m below, inferred)"
    # Neither: two walls (box-shaped room) -> up = cross product of their normals.
    wall_px = (stuff == 2) & valid
    if wall_px.sum() > 1000:
        vertical = lambda n: abs(n @ up_cam) < math.sin(math.radians(cfg["max_tilt_deg"]))  # noqa: E731
        pts = cam_pts[wall_px]
        first = ransac_plane(pts, cfg["plane_thresh_m"], 300, rng, vertical)
        if first is not None:
            rest = pts[~first[2]]
            second = ransac_plane(rest, cfg["plane_thresh_m"], 300, rng,
                                  lambda n: vertical(n) and abs(n @ first[0]) < 0.3)
            if second is not None:
                up = np.cross(first[0], second[0])
                up /= np.linalg.norm(up)
                if up @ up_cam < 0:
                    up = -up
                h = cfg["assumed_camera_height_m"]
                return GroundPlane(up, h, 0.0, False), f"two walls (camera height {h} m assumed)"
    return GroundPlane(up_cam, cfg["assumed_camera_height_m"], 0.0, False), "level camera assumed"


# ------------------------------------------------------------------------------------------
# 2. walls + ceiling
# ------------------------------------------------------------------------------------------

def find_walls(world: np.ndarray, wall_px: np.ndarray, cfg: dict, rng: np.random.Generator) -> tuple[list[Wall], float]:
    """Vertical walls as floor-plan lines (sequential RANSAC), snapped to right angles."""
    xy = world[wall_px][:, :2]
    walls: list[Wall] = []
    remaining = xy
    for _ in range(cfg["max_walls"]):
        fit = ransac_line(remaining, cfg["plane_thresh_m"] * 2, 300, rng)
        if fit is None or fit[2].sum() < max(cfg["min_wall_points"], 0.05 * len(xy)):
            break
        n, d, inl = fit
        if d < 0:  # camera (origin) must be on the positive side: normal points into the room
            n, d = -n, -d
        walls.append(Wall(n, d, False, int(inl.sum())))
        remaining = remaining[~inl]
    if not walls:
        return [], 0.0
    # Rooms are rectangular: the strongest wall sets the room axis; others snap to 0/90 degrees.
    main = max(walls, key=lambda w: w.points)
    axis = math.atan2(-main.normal[0], main.normal[1])  # direction ALONG the main wall
    snapped = []
    for w in walls:
        ang = math.atan2(w.normal[1], w.normal[0])
        rel = (ang - (axis + math.pi / 2)) % (math.pi / 2)
        if min(rel, math.pi / 2 - rel) < math.radians(cfg["snap_deg"]):
            k = round((ang - (axis + math.pi / 2)) / (math.pi / 2))
            ang = axis + math.pi / 2 + k * math.pi / 2
            n = np.array([math.cos(ang), math.sin(ang)])
            pts = xy[np.abs(xy @ w.normal + w.offset) < cfg["plane_thresh_m"] * 2]
            d = -float(np.median(pts @ n)) if len(pts) else w.offset
            if d < 0:
                n, d = -n, -d
            snapped.append(Wall(n, d, False, w.points))
    # Merge near-duplicates (same direction, within 15 cm).
    merged: list[Wall] = []
    for w in sorted(snapped, key=lambda w: -w.points):
        if any(abs(w.normal @ m.normal) > 0.99 and abs(w.offset - m.offset * (w.normal @ m.normal)) < 0.15 for m in merged):
            continue
        merged.append(w)
    return merged, axis


# ------------------------------------------------------------------------------------------
# 3. objects
# ------------------------------------------------------------------------------------------

def object_box(kind: str, pts: np.ndarray, axis: float, info: dict, cam_xy: np.ndarray,
               cfg: dict) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
    """Room-aligned box around an object's 3D points; back completed from the catalog if too shallow."""
    c, s = math.cos(axis), math.sin(axis)
    u = pts[:, :2] @ np.array([c, s])
    v = pts[:, :2] @ np.array([-s, c])
    lo_u, hi_u = np.percentile(u, [cfg["box_pct"], 100 - cfg["box_pct"]])
    lo_v, hi_v = np.percentile(v, [cfg["box_pct"], 100 - cfg["box_pct"]])
    zmin, top = np.percentile(pts[:, 2], [2, 98])
    w_cat, d_cat, _ = info["size_m"]
    # Which room axis points away from the camera here? Its far side is the unseen back.
    centre = np.array([(lo_u + hi_u) / 2, (lo_v + hi_v) / 2])
    cam_uv = np.array([cam_xy @ np.array([c, s]), cam_xy @ np.array([-s, c])])
    away = centre - cam_uv
    back_inferred = False
    min_depth = min(w_cat, d_cat)
    if abs(away[0]) > abs(away[1]):  # depth runs along u
        if hi_u - lo_u < 0.7 * min_depth:
            if away[0] > 0:
                hi_u = lo_u + min_depth
            else:
                lo_u = hi_u - min_depth
            back_inferred = True
    else:
        if hi_v - lo_v < 0.7 * min_depth:
            if away[1] > 0:
                hi_v = lo_v + min_depth
            else:
                lo_v = hi_v - min_depth
            back_inferred = True
    # Sideways, something in front can hide part of a RIGID thing (a chair half behind a desk came out
    # 19 cm wide): never thinner than min_footprint_frac x its catalog's smaller side, widened evenly
    # (which side was hidden is unknown). Soft things (blankets, bags) really can be that thin.
    floor_w = cfg.get("min_footprint_frac", 0.0) * min_depth
    if floor_w > 0 and info["physics"]["hardness"] >= 0.3:
        if abs(away[0]) > abs(away[1]) and hi_v - lo_v < floor_w:      # lateral = v
            mid = (lo_v + hi_v) / 2
            lo_v, hi_v = mid - floor_w / 2, mid + floor_w / 2
            back_inferred = True
        elif abs(away[0]) <= abs(away[1]) and hi_u - lo_u < floor_w:   # lateral = u
            mid = (lo_u + hi_u) / 2
            lo_u, hi_u = mid - floor_w / 2, mid + floor_w / 2
            back_inferred = True
    # Depth noise smears objects along the line of sight: never longer than max_size_factor x the
    # catalog's largest size - trim the FAR side (the near side is the surface we actually saw).
    cap = max(w_cat, d_cat) * cfg["max_size_factor"]
    if hi_u - lo_u > cap:
        lo_u, hi_u = (lo_u, lo_u + cap) if away[0] > 0 else (hi_u - cap, hi_u)
    if hi_v - lo_v > cap:
        lo_v, hi_v = (lo_v, lo_v + cap) if away[1] > 0 else (hi_v - cap, hi_v)
    size = np.array([max(hi_u - lo_u, 0.02), max(hi_v - lo_v, 0.02), max(top - zmin, 0.01)])
    cu, cv = (lo_u + hi_u) / 2, (lo_v + hi_v) / 2
    cxy = cu * np.array([c, s]) + cv * np.array([-s, c])
    return np.array([cxy[0], cxy[1], (zmin + top) / 2]), size, float(zmin), float(top), back_inferred


def build_objects(view_name: str, found, world: np.ndarray, valid: np.ndarray, depth: np.ndarray,
                  axis: float, catalog: dict, cfg: dict, first_id: int = 0) -> list[RoomObject]:
    h, w = depth.shape
    edges = depth_edges(depth, 5, 0.06)
    cam_xy = np.zeros(2)
    out = []
    for f in found:
        m = cv2.resize(f.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        sel = (inner if inner.sum() > 30 else m) & valid & ~edges
        if sel.sum() < 20:
            continue
        pts = world[sel]
        kind = f.kind
        info = catalog[kind]
        center, size, zmin, top, back = object_box(kind, pts, axis, info, cam_xy, cfg)
        # Some kinds ARE their height (a counter top is 0.75-1.1 m high, a toilet stands on the floor): a
        # detection far outside that is a wrong name - a "counter" high on the wall is a wall cabinet; the
        # rest is dropped rather than kept as an impossible object that misleads support and physics.
        rng = info.get("top_m")
        if rng is not None and not (rng[0] - 0.15 <= top <= rng[1] + 0.15):
            alt = info.get("else_kind")
            log.info("%s with its top at %.2f m is not plausible (%s-%s m) -> %s", kind, top, rng[0], rng[1], alt or "dropped")
            if alt is None or alt not in catalog:
                continue
            kind, info = alt, catalog[alt]
            center, size, zmin, top, back = object_box(kind, pts, axis, info, cam_xy, cfg)
        name = kind.replace("_", " ")
        out.append(RoomObject(first_id + len(out), kind, name, f.score, center, size, axis, zmin, top, m, info,
                              back_inferred=back, seen_in=[view_name], points=int(sel.sum())))
    return out


# ------------------------------------------------------------------------------------------
# 4. relationships
# ------------------------------------------------------------------------------------------

def relate(objects: list[RoomObject], walls: list[Wall], cfg: dict, catalog: dict | None = None) -> None:
    """Fill supported_by / next_to / under / against / leaning_on / mounted_on, and snap objects down."""
    # A "table" with a chair at it, or a computer on it, is a desk (dorm vocabulary) - and gets
    # the desk's knowledge.
    chairs = [o for o in objects if o.kind == "chair"]
    computer = [o for o in objects if o.kind in ("monitor", "keyboard", "laptop", "mouse")]
    for o in objects:
        at_it = any(np.linalg.norm(c.center[:2] - o.center[:2]) < max(o.size[:2]) for c in chairs)
        on_it = any(o.contains_xy(c.center[:2], 0.1) and c.zmin > o.top - 0.2 for c in computer)
        if o.kind == "table" and (at_it or on_it):
            o.kind, o.name = "desk", "desk"
            if catalog:
                o.info = catalog["desk"]

    for o in objects:
        o.relations = {}
        o.notes = []
        o.raised = 0.0
        o.support = None          # decided fresh every time (relate runs again after merging photos)
        o.attached_to = None
        o.inner_support = False
        # A "bed" reaching higher than 1.3 m is a loft bed (a platform on posts, space under it).
        if o.kind == "bed" and o.top > cfg["loft_min_top_m"]:
            o.kind, o.name = "loft_bed", "loft bed"
            if catalog:
                o.info = catalog["loft_bed"]

    # Beds, mattresses, desks against a wall reach it (their back is hidden behind what lies on
    # them) - done FIRST, so the pillows lying by the wall find the mattress under them.
    for a in objects:
        if not a.info.get("support") or a.size[0] * a.size[1] < 0.25 or a.kind == "chair":  # chairs stand free
            continue
        # a back the photo never showed (filled in from a typical size) leaves an invented gap: big
        # furniture stands against the wall, nobody leaves 50 cm behind a bed (and it can't be walked)
        reach = cfg.get("support_to_wall_inferred_m", cfg["support_to_wall_m"]) if a.back_inferred else cfg["support_to_wall_m"]
        for wl in walls:
            dist = float(np.min(wl.distance(a.footprint())))
            if 0 < dist < reach:
                _extend_to_wall(a, wl, dist)

    # The bed under a loft bed fills the loft's footprint (that is how loft/bunk beds are built);
    # one photo often sees it end-on and far too narrow.
    lofts = [o for o in objects if o.kind == "loft_bed"]
    for a in objects:
        if a.kind not in ("mattress", "bed", "blanket"):
            continue
        for loft in lofts:
            if loft.contains_xy(a.center[:2], 0.1) and a.top < loft.top - platform_slab(loft):
                inset = 0.08  # stay inside the loft's posts
                c, s_ = math.cos(loft.yaw), math.sin(loft.yaw)
                axes = np.array([[c, s_], [-s_, c]])
                lo = np.minimum((a.footprint() @ axes.T).min(axis=0), (loft.center[:2] @ axes.T) - loft.size[:2] / 2 + inset)
                hi = np.maximum((a.footprint() @ axes.T).max(axis=0), (loft.center[:2] @ axes.T) + loft.size[:2] / 2 - inset)
                lo = np.maximum(lo, (loft.center[:2] @ axes.T) - loft.size[:2] / 2 + inset)
                hi = np.minimum(hi, (loft.center[:2] @ axes.T) + loft.size[:2] / 2 - inset)
                a.yaw = loft.yaw
                a.size[:2] = hi - lo
                a.center[:2] = ((lo + hi) / 2) @ axes
                a.back_inferred = True
                break

    def find_support(a: RoomObject, below_tol: float, above_tol: float):
        best = None
        for b in objects:
            if b is a or not b.info.get("support") or b.kind in FLOOR_ONLY_SUPPORTS and a.kind in FLOOR_STANDING:
                continue
            below = min(below_tol, 0.5 * (b.top - b.zmin))  # a thin support (book) can't have things sunk into it
            if b.top - below <= a.zmin <= b.top + above_tol and b.contains_xy(a.center[:2], 0.05) and a.top > b.top:
                if best is None or b.top > best.top:
                    best = b
        return best

    # supported_by: lowest point of A ~ top of a supporting B, and A's centre above B's top.
    # With `support_prior` (default), every candidate - the floor and each surface - is scored by geometry
    # AND how often that kind really rests on that kind (toolkit/support.py); otherwise the first rule that
    # matches wins (the previous behaviour, kept for comparison).
    use_prior = cfg.get("support_prior", False)
    for a in sorted(objects, key=lambda o: o.zmin):
        choice = None
        if a.kind in FLOOR_STANDING:
            best = None  # beds, desks, ladders... stand on the floor, never on a bin or a box
        elif use_prior:
            choice, _ = choose_support(a, objects, cfg, True, lambda b: False, walls)
            best = choice if choice not in (None, "floor", "wall") else None
        else:
            best = find_support(a, cfg["support_tol_m"], cfg["support_tol_m"] * 2)
        if best is not None:
            a.support = best.label
            if use_prior and best.info.get("inner_surface") and a.zmin < best.top - 0.05:
                # on a seat / shelf inside the box: it stays at its own height (the box top is higher)
                a.inner_support = True
                a.notes.append(f"rests on a surface inside {best.label} (seat / shelf); held in place in "
                               f"physics, whose box shape fills that space")
            else:
                a.zmin = best.top
            a.relations.setdefault("supported_by", []).append(best.label)
        elif choice == "floor" or a.kind in FLOOR_STANDING or (not use_prior and a.zmin < cfg["floor_tol_m"]):
            # legs may be hidden (chair behind a desk)
            if a.zmin > 0.02:
                a.bottom_inferred = True  # legs hidden: extend down to the floor
            a.support = "floor"
            a.zmin = 0.0
            a.relations.setdefault("supported_by", []).append("floor")
        elif choice != "wall" and a.info.get("support") and a.zmin < cfg["raised_max_m"] and (not use_prior or a.size[0] * a.size[1] >= 0.4):
            # (only big things: a bed or mattress on a frame - not a pillow or a book in the air)
            # A surface floating low above the floor (a mattress) rests on a frame the photo can't
            # separate: model it as a raised platform on posts; things can be under it (bins).
            a.raised = max(a.top - a.zmin, 0.1)
            a.bottom_inferred = True
            a.support = "floor"
            a.zmin = 0.0
            a.relations.setdefault("supported_by", ["floor (on its frame)"])
        elif choice == "wall" or a.kind in ("poster", "window", "curtain", "door", "mirror", "whiteboard", "clock"):
            a.support = "wall"
        else:
            # Soft things sink into soft supports and boxes are rough: one more, looser try
            # (the prior-based choice already looked that far).
            best = None if use_prior else find_support(a, cfg["support_loose_m"], cfg["support_loose_m"])
            if best is None:
                # Resting at exactly the top height of a support whose back was never seen, just
                # behind it: that support continues back there (its depth was only a typical size).
                best = _support_behind(a, objects, cfg)
            if best is not None:
                a.support = best.label
                a.zmin = best.top
                a.relations.setdefault("supported_by", []).append(best.label)
        a.size[2] = max(a.top - a.zmin, 0.01)
        a.center[2] = (a.top + a.zmin) / 2

    # A bed/mattress standing on the floor over smaller floor things (storage bins under the bed)
    # is really raised on a frame: make it a raised platform so they fit under it.
    for b in objects:
        if b.kind not in ("bed", "mattress") or b.support != "floor" or b.raised:
            continue
        # only HARD things are evidence (a bin, a suitcase): a soft thing inside a bed's box (a
        # blanket, a dark pillow put too low by depth) is lying ON it, not under it
        below = [a for a in objects if a is not b and a.support == "floor" and a.top < b.top - 0.15
                 and not _soft(a) and b.contains_xy(a.center[:2]) and a.size[0] * a.size[1] < b.size[0] * b.size[1]]
        if below:
            b.raised = float(np.clip(b.top - max(a.top for a in below) - 0.03, 0.1, 0.35))
            b.bottom_inferred = True

    # under: below a support's top, with overlapping footprints (chair under desk).
    for a in objects:
        for b in objects:
            if a is b or not b.info.get("support") or a.support == b.label:
                continue
            # "under" = on the floor beneath a piece of furniture (chair under a desk, box under a
            # bed) - never one desk item "under" another (that is only depth smear of their boxes).
            bigger = b.size[0] * b.size[1] > 1.5 * a.size[0] * a.size[1] or bool(b.info.get("platform"))
            # only furniture with open space beneath (desk, table, loft bed, a bed on its frame) -
            # a dresser or a bed box is solid to the floor in physics
            furniture = (b.info.get("platform") or b.raised > 0) and a.support == "floor" and a.kind != "ladder"
            if furniture and bigger and a.zmin < b.top - 0.1 and b.contains_xy(a.center[:2]) and a.kind not in ("door", "window", "curtain"):
                if a.top > b.top:
                    a.relations.setdefault("partly_under", []).append(b.label)
                else:
                    a.relations.setdefault("under", []).append(b.label)

    # A ladder leans on the loft bed it stands against.
    for a in objects:
        if a.kind == "ladder":
            lofts = [b for b in objects if b.kind == "loft_bed" and _footprint_gap(a, b) < 0.35]
            if lofts:
                a.relations.setdefault("leaning_on", []).append(lofts[0].label)

    # next_to: same support, footprints close (centres within half their sizes + gap).
    for a in objects:
        near = []
        for b in objects:
            if a is b or a.support is None or a.support != b.support:
                continue
            gap = np.linalg.norm(a.center[:2] - b.center[:2]) - (max(a.size[:2]) + max(b.size[:2])) / 2
            if gap < cfg["next_to_gap_m"]:
                near.append((gap, b.label))
        if near:
            a.relations["next_to"] = [lab for _, lab in sorted(near)[:3]]

    # walls: against (close), leaning_on (close at the top, not at the bottom), mounted_on (off the floor).
    for a in objects:
        for i, wl in enumerate(walls):
            dist = float(np.min(wl.distance(a.footprint())))
            big = a.size[0] * a.size[1] > 0.25
            if (dist < cfg["wall_tol_m"] and a.info.get("support") and (a.info["state"] == "static" or big) and dist > 0
                    and a.kind != "chair"):
                # Furniture standing against a wall reaches it (the camera can't see the gap behind).
                _extend_to_wall(a, wl, dist)
                dist = 0.0
            if dist < cfg["wall_tol_m"]:
                rel = "against"
                if a.support == "wall" or (a.support is None and a.zmin > 0.3):
                    rel = "mounted_on"
                elif a.kind in ("picture_frame", "poster") and a.support not in (None, "wall"):
                    rel = "leaning_on"
                a.relations.setdefault(rel, []).append(f"wall {i + 1}")
    # Hanging: rests on nothing, but sits right against a piece of furniture (tote bag on a bed
    # rail, backpack on a chair back) -> hanging_on that furniture; fixed in place in physics.
    # Things hang on what is made for it (catalog hang_point: chair backs, bed rails, ladders, doors,
    # wardrobes) - preferred over any other furniture that happens to be close, at any height (a
    # backpack on a chair back reaches down over the seat). Other furniture only for things well
    # above the floor (hang_min_m).
    for a in objects:
        a.attached_to = None
        if a.support is not None or a.info["state"] != "movable":
            continue
        best = None
        for b in objects:
            if b is a or b.attached_to is not None:
                continue
            point = bool(b.info.get("hang_point"))
            if a.zmin < cfg["hang_min_m"] and not point:
                continue
            gap = _footprint_gap(a, b)
            if gap < cfg["hang_gap_m"] and b.zmin - 0.05 <= a.top <= b.top + 0.15:
                rank = (not point, gap)
                if best is None or rank < best[0]:
                    best = (rank, b)
        if best is not None:
            a.attached_to = best[1].label
            a.support = f"hanging:{best[1].label}"
            a.relations.setdefault("hanging_on", []).append(best[1].label)
    # Still resting on nothing (its support wasn't recognised, e.g. a shelf or chair the detector
    # missed): hold it in place and say so, instead of letting it fall in physics.
    for a in objects:
        if a.support is None and a.info["state"] == "static" and "mounted_on" in a.relations:
            a.support = "wall"
        if a.support is None and a.info["state"] == "movable":
            a.support = "unseen"
            a.relations.setdefault("support_not_seen", ["held in place"])
    keep_on_support(objects)
    resolve_overlaps(objects)
    # Supports may have changed above (capped under a desk top, extended to a wall, a blanket put
    # on a bed): what rests on something rests on its FINAL top - nothing floats or sinks in.
    # Nothing is thinner than physics makes it (1 cm, room_export): a 4 mm mouse pad is 1 cm thick
    # in MuJoCo, so the mouse on it must sit on the 1 cm top, not inside the pad.
    for o in objects:
        if o.top - o.zmin < MIN_THICK_M:
            o.top = o.zmin + MIN_THICK_M
            o.size[2] = MIN_THICK_M
            o.center[2] = o.zmin + MIN_THICK_M / 2
    by_label = {o.label: o for o in objects}
    for o in sorted(objects, key=lambda o: o.zmin):
        b = by_label.get(o.support or "")
        if b is not None and abs(o.zmin - b.top) > 0.005 and not o.inner_support:
            height = o.top - o.zmin
            o.zmin, o.top = b.top, b.top + height
            o.center[2] = b.top + height / 2


def _support_behind(a: RoomObject, objects: list[RoomObject], cfg: dict) -> RoomObject | None:
    """A piece of furniture `a` rests on although the photo showed it too shallow: `a`'s bottom is at
    the furniture's top height and `a` is just behind it, seen from the camera (a cup behind the
    visible front of a dresser). The furniture is extended back under `a`; both get a note saying
    this was inferred, not seen."""
    best = None
    for b in objects:
        # one photo never really shows a back: up to 45 cm behind a support is allowed even when its
        # depth looked complete, 60 cm when its back was filled in from a typical size
        if b is a or not b.info.get("support") or b.info["state"] != "static" or abs(a.zmin - b.top) > 1.5 * cfg["support_tol_m"]:
            continue
        gap = _footprint_gap(a, b)
        view = b.center[:2] / max(float(np.linalg.norm(b.center[:2])), 1e-6)   # camera -> support (camera at 0, 0)
        if gap > (0.6 if b.back_inferred else 0.45) or (a.center[:2] - b.center[:2]) @ view <= 0:
            continue
        if best is None or gap < best[0]:
            best = (gap, b, view)
    if best is None:
        return None
    gap, b, view = best
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    axes = (np.array([c, s]), np.array([-s, c]))
    k = int(np.argmax([abs(ax @ view) for ax in axes]))
    ax = axes[k] * (1.0 if axes[k] @ view >= 0 else -1.0)
    front = b.center[:2] - ax * b.size[k] / 2
    reach = float(((a.footprint() - front) @ ax).max()) + 0.03
    if reach > b.size[k]:
        b.size[k] = reach
        b.center[:2] = front + ax * reach / 2
    b.notes.append(f"back extended {gap * 100:.0f} cm to hold {a.label} (its back is not visible in the photo)")
    a.notes.append(f"support inferred: {b.label}, whose unseen back reaches under it")
    return b


def _footprint_gap(a: RoomObject, b: RoomObject) -> float:
    """Horizontal gap between two boxes (0 if their footprints touch or overlap), room-aligned."""
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    d = a.center[:2] - b.center[:2]
    gu = abs(d @ np.array([c, s])) - (a.size[0] + b.size[0]) / 2
    gv = abs(d @ np.array([-s, c])) - (a.size[1] + b.size[1]) / 2
    return float(math.hypot(max(gu, 0.0), max(gv, 0.0)))


def _extend_to_wall(o: RoomObject, wall: Wall, gap: float) -> None:
    """Grow a box `gap` metres towards a wall (along whichever box axis faces the wall)."""
    c, s = math.cos(o.yaw), math.sin(o.yaw)
    axes = [np.array([c, s]), np.array([-s, c])]
    k = int(np.argmax([abs(ax @ wall.normal) for ax in axes]))
    o.size[k] += gap
    o.center[:2] -= wall.normal * gap / 2      # the wall is on the side opposite to its normal


def keep_on_support(objects: list[RoomObject]) -> None:
    """Things resting on something stay over it: move their centre inside the support's top."""
    by_label = {o.label: o for o in objects}
    for a in objects:
        b = by_label.get(a.support or "")
        if b is None:
            continue
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        U, V = np.array([c, s]), np.array([-s, c])
        d = a.center[:2] - b.center[:2]
        du, dv = d @ U, d @ V
        lim_u = max(b.size[0] / 2 - 0.25 * min(a.size[0], a.size[1]), 0.0)
        lim_v = max(b.size[1] / 2 - 0.25 * min(a.size[0], a.size[1]), 0.0)
        du, dv = float(np.clip(du, -lim_u, lim_u)), float(np.clip(dv, -lim_v, lim_v))
        a.center[:2] = b.center[:2] + du * U + dv * V
        # Under-the-desk things (chair) collide only up to the desk's underside.
    for a in objects:
        for lab in a.relations.get("partly_under", []) + a.relations.get("under", []):
            b = by_label.get(lab)
            if b is not None and a.top > b.top - platform_slab(b) - 0.02:
                a.top = b.top - platform_slab(b) - 0.02
                a.size[2] = max(a.top - a.zmin, 0.05)
                a.center[2] = (a.top + a.zmin) / 2


def resolve_overlaps(objects: list[RoomObject], gap: float = 0.002) -> None:
    """Two things on the same surface can't share space. Their boxes overlap only because depth
    smear makes them too deep: shrink them along the overlapping direction (room axes).

    A movable thing lying mostly INSIDE a piece of furniture is not squashed flat:
      * a part of it seen twice (a "mattress" box inside the bed's box: one space belongs to one
        object) is removed from `objects` (in place);
      * anything else inside a supporting piece (a blanket inside the bed's box: depth put it too
        low) lies ON it."""
    def fixed(o: RoomObject) -> bool:
        return o.info["state"] == "static" or bool(o.info.get("platform")) or bool(o.raised)

    def under(a: RoomObject, b: RoomObject) -> bool:
        beneath = a.relations.get("under", []) + a.relations.get("partly_under", []) + \
            b.relations.get("under", []) + b.relations.get("partly_under", [])
        return a.label in beneath or b.label in beneath

    twice = set()
    for m in objects:
        for f in objects:
            if m is f or fixed(m) or m.support != f.support or m.support in (None, "wall") or under(m, f):
                continue
            # furniture, or (for a soft thing like a blanket) a support at least about as big as it
            if not (fixed(f) or (_soft(m) and f.info.get("support") and not _soft(f)
                                 and f.size[0] * f.size[1] >= 0.8 * m.size[0] * m.size[1])):
                continue
            if m.zmin >= f.top or f.zmin >= m.top:
                continue
            inside = np.mean([f.contains_xy(p) for p in _grid(m)])
            if inside < 0.6:
                continue
            if (m.kind, f.kind) in PART_OF:
                log.info("%s lies inside %s: the same thing seen twice -> removed", m.label, f.label)
                twice.add(m.label)
            elif m.kind in FLOOR_STANDING and f.info["state"] == "static" and not f.info.get("platform") and f.top > m.top:
                # a chair standing inside a "dresser": it has open space below (it is desk-like) -
                # the chair stays on the floor, under it, and the furniture stands on legs
                log.info("%s stands inside %s: it has open space underneath", m.label, f.label)
                f.raised = float(np.clip(f.top - m.top - 0.03, 0.03, 0.35))
                f.notes.append(f"open space underneath inferred: {m.label} stands inside its outline")
                m.relations.setdefault("under", []).append(f.label)
            elif f.info.get("support") and m.kind not in FLOOR_STANDING:
                # (furniture that stands on the floor - a chair - is never lifted onto what its box overlaps)
                height = min(m.top - m.zmin, (m.info.get("size_m") or [0, 0, 0.3])[2] or 0.3)
                log.info("%s lies inside %s: it rests on it", m.label, f.label)
                m.zmin, m.top = f.top, f.top + height
                m.size[2], m.center[2] = height, f.top + height / 2
                m.support = f.label
                m.relations["supported_by"] = [f.label]
                for rel in ("under", "partly_under"):
                    m.relations.pop(rel, None)
                if f.label in m.relations.get("next_to", []):
                    m.relations["next_to"].remove(f.label)
                    if not m.relations["next_to"]:
                        del m.relations["next_to"]
                # what already lay on that furniture where it now lies (toys on the bed) rests on it
                for o in objects:
                    if o is not m and o.support == f.label and m.contains_xy(o.center[:2]) and m.info.get("support"):
                        o.zmin, o.top = o.zmin + height, o.top + height
                        o.center[2] += height
                        o.support = m.label
                        o.relations["supported_by"] = [m.label]
            break
    objects[:] = [o for o in objects if o.label not in twice]

    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            if a.support is None or a.support != b.support or a.support in ("wall",):
                continue
            if under(a, b):
                continue  # a chair under a desk legitimately shares the desk's footprint
            if b.zmin >= a.top or a.zmin >= b.top:
                continue
            c, s = math.cos(a.yaw), math.sin(a.yaw)
            axes = (np.array([c, s]), np.array([-s, c]))
            d = b.center[:2] - a.center[:2]
            over = [(a.size[k] + b.size[k]) / 2 - abs(d @ axes[k]) for k in (0, 1)]
            if min(over) <= 0:
                continue
            k = int(np.argmin(over))              # separate along the smaller overlap
            sign = 1.0 if d @ axes[k] >= 0 else -1.0
            still = {o.label for o in (a, b) if fixed(o)}
            if len(still) == 2:
                continue                          # two pieces of furniture: positions are trusted
            movers = [(o, dirn) for o, dirn in ((a, -sign), (b, sign)) if o.label not in still]
            cut = over[k] / len(movers) + gap     # one mover takes the whole overlap, two share it
            for o, direction in movers:
                o.size[k] = max(o.size[k] - cut, 0.02)
                o.center[:2] += axes[k] * direction * cut / 2


def _soft(o: RoomObject) -> bool:
    """Blanket, pillow, plush toy, clothes, plastic bag: they lie on things and take their shape."""
    return o.info["physics"]["hardness"] < 0.1


def _grid(o: RoomObject, n: int = 5) -> np.ndarray:
    """n x n points spread over an object's footprint (to measure how much of it lies inside another)."""
    c, s = math.cos(o.yaw), math.sin(o.yaw)
    t = (np.arange(n) + 0.5) / n - 0.5
    a, b = np.meshgrid(t * o.size[0], t * o.size[1])
    return o.center[:2] + np.stack([a.ravel(), b.ravel()], 1) @ np.array([[c, s], [-s, c]])


def fit_inside(objects: list[RoomObject], box: "RoomBox") -> None:
    """Nothing pokes through a wall: trim object boxes to the room (depth errors can push the far
    end of a bed past the wall it stands against)."""
    c, s = math.cos(box.axis), math.sin(box.axis)
    room_axes = (np.array([c, s]), np.array([-s, c]))
    for o in objects:
        oc, os_ = math.cos(o.yaw), math.sin(o.yaw)
        for k, ax in enumerate((np.array([oc, os_]), np.array([-os_, oc]))):
            dots = [abs(ax @ r) for r in room_axes]
            r = int(np.argmax(dots))
            if dots[r] < 0.95:
                continue                          # not lined up with the room: leave it
            mid = float(o.center[:2] @ room_axes[r])
            lo, hi = mid - o.size[k] / 2, mid + o.size[k] / 2
            new_lo, new_hi = max(lo, box.lo[r]), min(hi, box.hi[r])
            if new_lo == lo and new_hi == hi:
                continue
            if new_hi - new_lo < 0.02:            # (almost) entirely outside: keep a sliver at the wall
                new_lo, new_hi = (box.lo[r], box.lo[r] + 0.02) if mid < box.lo[r] else (box.hi[r] - 0.02, box.hi[r])
            o.size[k] = new_hi - new_lo
            o.center[:2] += room_axes[r] * ((new_lo + new_hi) / 2 - mid)


# ------------------------------------------------------------------------------------------
# 5. records (the agreed format)
# ------------------------------------------------------------------------------------------

def position_text(o: RoomObject) -> str:
    if o.support == "unseen":
        return "support not seen (held in place)"
    if o.attached_to:
        return f"hanging on {o.attached_to.split(' #')[0]} ({o.attached_to})"
    if o.support and o.support not in ("floor", "wall") and o.inner_support:
        return f"on a surface inside {o.support.split(' #')[0]} ({o.support}: seat / shelf)"
    if o.support and o.support not in ("floor", "wall"):
        return f"on {o.support.split(' #')[0]} surface ({o.support})"
    if o.support == "floor":
        return "on the floor"
    if o.support == "wall":
        return "on the wall"
    return "unknown support"


def record(o: RoomObject) -> dict[str, Any]:
    """One object in the format agreed for the proof of concept."""
    info = o.info
    vol = float(np.prod(o.size))
    mass = info.get("mass_kg")
    return {
        "Object": o.name.title(),
        "Id": o.label,
        "Boundary": {"mask": "pixel outline (SAM)", "mask_pixels": int(o.mask.sum()),
                     "box_m": {"center": [round(float(v), 2) for v in o.center],
                               "size": [round(float(v), 2) for v in o.size],
                               "yaw_deg": round(math.degrees(o.yaw), 1)},
                     "back_side": "inferred (catalog size)" if o.back_inferred else "seen",
                     "bottom": "inferred (hidden legs)" if o.bottom_inferred else "seen"},
        "Position": position_text(o),
        "Material": f"{info['material']} [estimated]",
        "State": info["state"],
        "Affordances": list(info["affordances"]),
        "Relationships": o.relations,
        # same rule as the physics export: soft static things (curtain) don't block; soft movable
        # things (blanket, plush toy) collide so they rest on their support, but are light and soft
        "Collision": ("enabled" if info["physics"]["hardness"] > 0.1 else
                      "enabled (soft)" if info["state"] == "movable" else "off (soft - things pass through)"),
        "Mass": f"{mass} kg [typical]" if mass is not None else "unknown",
        "Size_measured_m": [round(float(v), 2) for v in o.size],
        "Volume_m3": round(vol, 4),
        "Confidence": round(o.score, 2),
        "Seen_in": o.seen_in,
        **({"Inferred": list(o.notes)} if o.notes else {}),
    }


def record_text(o: RoomObject) -> str:
    """The record as plain text, laid out like the example card."""
    r = record(o)
    rel = r["Relationships"]
    lines = [f"Object: {r['Object']}  ({r['Id']})",
             f"Boundary: [mask: {r['Boundary']['mask_pixels']} px] box {r['Boundary']['box_m']['size']} m, back {r['Boundary']['back_side']}",
             f"Position: {r['Position']}",
             f"Material: {r['Material']}",
             f"State: {r['State']}",
             "Affordances:"] + [f"  - {a}" for a in r["Affordances"]] + ["Relationships:"]
    lines += [f"  - {k}: {', '.join(v)}" for k, v in rel.items()] or ["  - (none found)"]
    lines += [f"Collision: {r['Collision']}", f"Mass: {r['Mass']}",
              f"Confidence: {r['Confidence']}  |  seen in: {', '.join(r['Seen_in'])}"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------
# one photo -> RoomView
# ------------------------------------------------------------------------------------------

@dataclass
class Models:
    """The networks, loaded once: depth, structure segmentation, open-vocabulary detector."""

    depth: Any
    seg: Any
    det: Any


def structure_map(ade: np.ndarray, class_names: list[str]) -> np.ndarray:
    """ADE20K class ids -> 0 other, 1 floor, 2 wall, 3 ceiling."""
    out = np.zeros(ade.shape, np.uint8)
    for code, names in ((1, FLOOR_ADE), (2, WALL_ADE), (3, CEILING_ADE)):
        ids = [i for i, n in enumerate(class_names) if n in names]
        out[np.isin(ade, ids)] = code
    return out


def _geometry(name, image, small, depth, k, fov, fov_src, stuff, found, catalog, cfg, rng, plane=None, up_source=""):
    """Everything after depth: up/floor, world points, walls, ceiling, objects, relations."""
    h, w = depth.shape
    cam = backproject(depth, k)
    valid = (depth > 0.05) & (depth < cfg["max_depth_m"])
    obj_px: dict[str, np.ndarray] = {}
    for f in found:
        m = cv2.resize(f.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        obj_px[f.kind] = obj_px.get(f.kind, np.zeros((h, w), bool)) | m
    any_obj = np.zeros((h, w), bool)
    for kind, m in obj_px.items():
        # objects win over wall/floor labels - except what you walk on: a rug IS the floor for
        # geometry (a carpeted room detected as one big "rug" must not lose its floor)
        if "walk_on" not in (catalog.get(kind, {}).get("affordances") or []):
            any_obj |= m
    stuff = np.where(any_obj, 0, stuff).astype(np.uint8)
    if plane is None:
        plane, up_source = find_up(cam, valid, stuff, obj_px, catalog, cfg, rng)
    world = to_world(cam, plane)
    cam_w = np.array([0.0, 0.0, plane.offset])
    snapped = np.zeros((h, w), bool)
    # Floor pixels close to z = 0 -> exactly on the floor (crisp floor, like the park ground).
    fl = (stuff == 1) & valid & (np.abs(world[..., 2]) < cfg["snap_m"])
    world[fl] = snap_to_world_plane(world[fl], cam_w, np.array([0.0, 0.0, 1.0]), 0.0)
    snapped |= fl
    # Walls.
    wall_px = (stuff == 2) & valid & (world[..., 2] > 0.05)
    walls, axis = find_walls(world, wall_px, cfg, rng)
    if not walls:
        axis = 0.0
    for wl in walls:
        near = wall_px & (np.abs(world[..., :2] @ wl.normal + wl.offset) < cfg["snap_m"])
        n3 = np.array([wl.normal[0], wl.normal[1], 0.0])
        world[near] = snap_to_world_plane(world[near], cam_w, n3, wl.offset)
        snapped |= near
    # Ceiling.
    ce = (stuff == 3) & valid & (world[..., 2] > 1.5)
    if ce.sum() > cfg["min_ceiling_px"]:
        ceiling, ceiling_inf = float(np.median(world[ce][:, 2])), False
        near = ce & (np.abs(world[..., 2] - ceiling) < cfg["snap_m"])
        world[near] = snap_to_world_plane(world[near], cam_w, np.array([0.0, 0.0, 1.0]), -ceiling)
        snapped |= near
    else:
        ceiling, ceiling_inf = cfg["default_ceiling_m"], True
    objects = build_objects(name, found, world, valid, depth, axis, catalog, cfg)
    relate(objects, walls, cfg, catalog)
    return plane, up_source, world, snapped, stuff, walls, axis, ceiling, ceiling_inf, objects


def build_view(name: str, image: np.ndarray, path, models: Models, catalog: dict, cfg: dict) -> RoomView:
    """One photo -> a RoomView (see the file header). Runs twice if the scale needs correcting."""
    from .lens import resolve_fov  # camera.yaml > EXIF > the depth model's estimate

    rng = np.random.default_rng(cfg["seed"])
    h = int(round(cfg["analysis_width"] * image.shape[0] / image.shape[1]))
    w = cfg["analysis_width"]
    small = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    fov, fov_src = resolve_fov(path, cfg["default_fov_deg"])
    if getattr(models.depth, "gives_lens", False):
        # the depth model also reads the lens off the photo: give it the lens when known (camera.yaml /
        # EXIF) so depth and lens agree, otherwise use its own estimate instead of a fixed guess
        known = None if fov_src.startswith("config") else fov
        depth, est = models.depth.predict_with_lens(image, (h, w), known)
        if known is None:
            fov, fov_src = est, "estimated from the photo (MoGe-2)"
    else:
        depth = models.depth.predict(image, (h, w))
    ade = models.seg.predict_classes(image, (h, w))
    stuff = structure_map(ade, models.seg.class_names)
    found = models.det.detect(image)
    k = intrinsics(w, h, fov)

    geo = _geometry(name, image, small, depth, k, fov, fov_src, stuff, found, catalog, cfg, rng)
    plane, up_source = geo[0], geo[1]
    # Scale check: standard heights (desk/table top ~0.75 m, door ~2.03 m), only when the floor
    # itself was seen (otherwise the floor was placed FROM the desk height - circular).
    scale, note = 1.0, "not calibrated (floor not seen directly)" if up_source != "floor" else "not calibrated (no reference object)"
    if not cfg.get("scale_from_furniture", True):
        note = "metric depth as the model gives it (no furniture-height correction)"
    elif up_source == "floor":
        ratios, used = [], []
        for o in geo[9]:
            if o.info.get("scale_ref") and o.support == "floor":
                ratios.append(o.top / o.info["size_m"][2])   # standard furniture height
                used.append(o.kind)
        if not geo[8]:  # ceiling seen: typical room height
            ratios.append(geo[7] / cfg["typical_ceiling_m"])
            used.append("ceiling")
        if ratios:
            raw = 1.0 / float(np.median(ratios))
            scale = float(np.clip(raw, *cfg["scale_clamp"]))
            note = f"x{scale:.2f} from {len(ratios)} reference{'s' if len(ratios) > 1 else ''} ({', '.join(sorted(set(used)))})"
            if scale != raw:
                note += f" - wanted x{raw:.2f}, limited"
            if abs(scale - 1.0) > 0.03:
                depth = depth * scale
                scaled = GroundPlane(plane.normal, plane.offset * scale, plane.inlier_ratio, plane.fitted)
                geo = _geometry(name, image, small, depth, k, fov, fov_src, stuff, found, catalog, cfg, rng, scaled, up_source)
            else:
                scale = 1.0
    plane, up_source, world, snapped, stuff2, walls, axis, ceiling, ceiling_inf, objects = geo
    return RoomView(name, image, small, depth, k, fov, fov_src, plane, up_source, world, snapped, stuff2,
                    walls, axis, ceiling, ceiling_inf, objects, scale, note)


# ------------------------------------------------------------------------------------------
# the room rectangle (closed, for physics)
# ------------------------------------------------------------------------------------------

@dataclass
class RoomBox:
    axis: float                 # room x-axis angle on the floor plan
    lo: np.ndarray              # (2,) min (u, v) in room axes
    hi: np.ndarray              # (2,) max (u, v)
    height: float
    sides_seen: dict[str, bool]  # "u-", "u+", "v-", "v+" -> wall observed?

    def corners(self) -> np.ndarray:
        c, s = math.cos(self.axis), math.sin(self.axis)
        U, V = np.array([c, s]), np.array([-s, c])
        return np.array([a * U + b * V for a, b in ((self.lo[0], self.lo[1]), (self.hi[0], self.lo[1]),
                                                     (self.hi[0], self.hi[1]), (self.lo[0], self.hi[1]))])


def room_box(walls: list[Wall], axis: float, points_xy: np.ndarray, objects: list[RoomObject], height: float,
             margin: float = 0.3) -> RoomBox:
    """Close the room: seen walls where they exist, otherwise just beyond everything seen (inferred)."""
    c, s = math.cos(axis), math.sin(axis)
    U, V = np.array([c, s]), np.array([-s, c])
    pts = [points_xy] + [o.footprint() for o in objects]
    allp = np.concatenate([p for p in pts if len(p)]) if any(len(p) for p in pts) else np.zeros((1, 2))
    allp = np.concatenate([allp, np.zeros((1, 2))])  # the camera stands inside the room
    u, v = allp @ U, allp @ V
    lo = np.array([np.percentile(u, 1) - margin, np.percentile(v, 1) - margin])
    hi = np.array([np.percentile(u, 99) + margin, np.percentile(v, 99) + margin])
    # The photographer stood in the room: keep their spot inside with room to stand (0.7 m).
    lo, hi = np.minimum(lo, -0.7), np.maximum(hi, 0.7)
    seen = {"u-": False, "u+": False, "v-": False, "v+": False}
    for wl in walls:
        n = wl.normal
        # wall plane in room coordinates: n . X + d = 0 -> along U or V
        nu, nv = n @ U, n @ V
        if abs(nu) > 0.9:   # wall perpendicular to U: u = -d / nu
            pos = -wl.offset / nu
            side = "u-" if nu > 0 else "u+"   # normal points into the room
            if side == "u-":
                lo[0] = pos
            else:
                hi[0] = pos
            seen[side] = True
        elif abs(nv) > 0.9:
            pos = -wl.offset / nv
            side = "v-" if nv > 0 else "v+"
            if side == "v-":
                lo[1] = pos
            else:
                hi[1] = pos
            seen[side] = True
    return RoomBox(axis, lo, hi, height, seen)


# ------------------------------------------------------------------------------------------
# several photos -> one room (registration + persistence)
# ------------------------------------------------------------------------------------------

FAMILY = {"table": "desk"}  # names the detector uses for the same kind of thing


def family(kind: str) -> str:
    return FAMILY.get(kind, kind)


@dataclass
class Alignment:
    """How photo j's world frame maps into the reference photo's: X_ref = s * Rz(yaw) X_j + t."""

    view: str
    ok: bool
    scale: float = 1.0
    yaw: float = 0.0
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    matches: int = 0
    inliers: int = 0
    note: str = ""

    def apply(self, pts: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return self.scale * pts @ r.T + self.t


def _fit_yaw_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Least squares: a ~ s * Rz(yaw) b + t (t has no z: both floors are z = 0)."""
    ca, cb = a[:, :2].mean(0), b[:, :2].mean(0)
    pa, pb = a[:, :2] - ca, b[:, :2] - cb
    h = pb.T @ pa
    yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    c, s = math.cos(yaw), math.sin(yaw)
    rb = pb @ np.array([[c, -s], [s, c]]).T
    # scale from horizontal spread and heights together
    num = float((rb * pa).sum() + (a[:, 2] * b[:, 2]).sum())
    den = float((rb * rb).sum() + (b[:, 2] * b[:, 2]).sum())
    scale = num / den if den > 1e-9 else 1.0
    t = np.array([*(ca - scale * (np.array([[c, -s], [s, c]]) @ cb)), 0.0])
    return scale, yaw, t


def align(ref: RoomView, other: RoomView, cfg: dict, rng: np.random.Generator) -> Alignment:
    """Find how `other` sits relative to `ref` from matching image features (SIFT) + depth."""
    sift = cv2.SIFT_create(nfeatures=cfg["align_features"])
    ga, gb = (cv2.cvtColor(v.small, cv2.COLOR_RGB2GRAY) for v in (ref, other))
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return Alignment(other.name, False, note="too few image features")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    A, B = [], []
    for m in good:
        xa, ya = (int(round(v)) for v in ka[m.trainIdx].pt)
        xb, yb = (int(round(v)) for v in kb[m.queryIdx].pt)
        if 0 <= ya < ref.world.shape[0] and 0 <= xa < ref.world.shape[1] and 0 <= yb < other.world.shape[0] and 0 <= xb < other.world.shape[1]:
            pa, pb = ref.world[ya, xa], other.world[yb, xb]
            if np.isfinite(pa).all() and np.isfinite(pb).all() and ref.depth[ya, xa] < cfg["max_depth_m"] and other.depth[yb, xb] < cfg["max_depth_m"]:
                A.append(pa)
                B.append(pb)
    A, B = np.array(A), np.array(B)
    if len(A) < cfg["align_min_inliers"]:
        return Alignment(other.name, False, matches=len(A), note=f"only {len(A)} feature matches with depth")
    best, best_inl = None, np.zeros(len(A), bool)
    for _ in range(cfg["align_iters"]):
        i, j = rng.choice(len(A), 2, replace=False)
        if np.linalg.norm(B[i, :2] - B[j, :2]) < 0.2:
            continue
        s, yaw, t = _fit_yaw_similarity(A[[i, j]], B[[i, j]])
        if not 0.5 < s < 2.0:
            continue
        al = Alignment(other.name, True, s, yaw, t)
        inl = np.linalg.norm(A - al.apply(B), axis=1) < cfg["align_thresh_m"]
        if inl.sum() > best_inl.sum():
            best, best_inl = al, inl
    if best is None or best_inl.sum() < max(cfg["align_min_inliers"], cfg["align_min_ratio"] * len(A)):
        n = int(best_inl.sum())
        return Alignment(other.name, False, matches=len(A), inliers=n, note=f"only {n}/{len(A)} matches agree")
    s, yaw, t = _fit_yaw_similarity(A[best_inl], B[best_inl])
    return Alignment(other.name, True, s, yaw, t, len(A), int(best_inl.sum()),
                     f"{int(best_inl.sum())}/{len(A)} matches agree; turn {math.degrees(yaw):.0f} deg, scale x{s:.2f}")


def compose(a: Alignment, b: Alignment) -> Alignment:
    """The alignment doing b first, then a (photo -> middle photo -> reference)."""
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Alignment(b.view, True, a.scale * b.scale, a.yaw + b.yaw, a.scale * r @ b.t + a.t,
                     b.matches, b.inliers, f"{b.note} (chained)")


def invert(a: Alignment, view: str) -> Alignment:
    """The opposite alignment (reference -> photo becomes photo -> reference)."""
    c, s = math.cos(-a.yaw), math.sin(-a.yaw)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Alignment(view, True, 1.0 / a.scale, -a.yaw, -(r @ a.t) / a.scale, a.matches, a.inliers, a.note)


def plan_groups(views: list[RoomView], cfg: dict) -> tuple[list[tuple[RoomView, dict[str, Alignment]]], list[Alignment]]:
    """Which photos can be put together, and how. Every pair is tried; photos linked directly or
    through a chain form one group (one zone of the room). Returns [(reference, {photo: transform
    into the reference})] (largest group first) and every attempted pair (for the report)."""
    rng = np.random.default_rng(cfg["seed"])
    edges: dict[tuple[int, int], Alignment] = {}
    tried: list[Alignment] = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            al = align(views[i], views[j], cfg, rng)          # maps photo j into photo i
            al.note = f"{views[j].name} -> {views[i].name}: {al.note}"
            tried.append(al)
            if al.ok:
                edges[(i, j)] = al
    seen: set[int] = set()
    groups = []
    for start in sorted(range(len(views)), key=lambda k: -len(views[k].objects)):
        if start in seen:
            continue
        # breadth-first over aligned pairs: transform of every member into `start`'s frame
        to_ref = {start: Alignment(views[start].name, True)}
        queue = [start]
        while queue:
            u = queue.pop(0)
            for (i, j), al in edges.items():
                if i == u and j not in to_ref:
                    to_ref[j] = compose(to_ref[u], al)
                    queue.append(j)
                elif j == u and i not in to_ref:
                    to_ref[i] = compose(to_ref[u], invert(al, views[i].name))
                    queue.append(i)
        seen |= set(to_ref)
        groups.append((views[start], {views[k].name: t for k, t in to_ref.items() if k != start}))
    return groups, tried


def merge_views(views: list[RoomView], cfg: dict, catalog: dict | None = None) -> tuple[RoomView, list[RoomObject], list[Wall], list[Alignment], list[RoomView]]:
    """All photos that can be aligned -> one room (the largest group). Kept for single-group use."""
    groups, tried = plan_groups(views, cfg)
    ref, transforms = groups[0]
    members = [v for v in views if v.name in transforms]
    objects, walls = merge_group(ref, members, transforms, cfg, catalog)
    return ref, objects, walls, tried, [ref] + members


def merge_group(ref: RoomView, members: list[RoomView], transforms: dict[str, Alignment], cfg: dict,
                catalog: dict | None = None) -> tuple[list[RoomObject], list[Wall]]:
    """Put the member photos into the reference photo's frame; merge objects seen more than once."""
    objects = [o for o in ref.objects]
    walls = list(ref.walls)
    for v in members:
        al = transforms[v.name]
        # walls into the reference frame
        c, s = math.cos(al.yaw), math.sin(al.yaw)
        r2 = np.array([[c, -s], [s, c]])
        for wl in v.walls:
            n = r2 @ wl.normal
            point = al.apply(np.array([*(-wl.offset * wl.normal), 0.0]))[:2]
            d = -float(n @ point)
            if d < 0:
                n, d = -n, -d
            if not any(abs(n @ m.normal) > 0.98 and abs(d - m.offset) < 0.2 for m in walls):
                walls.append(Wall(n, d, False, wl.points))
        # objects into the reference frame; same kind close together = the same object (persistence).
        # Matching is one-to-one per photo and nearest-first (two notebooks stay two notebooks).
        matched: set[int] = set()
        moved = []
        for o in v.objects:
            c_o = al.apply(o.center[None])[0]
            dists = [np.linalg.norm(m.center[:2] - c_o[:2]) for m in objects if family(m.kind) == family(o.kind)]
            moved.append((min(dists) if dists else 1e9, o))
        for _, o in sorted(moved, key=lambda t: t[0]):
            centre = al.apply(o.center[None])[0]
            size = o.size * al.scale
            yaw = o.yaw + al.yaw
            twin = None
            probe = RoomObject(-1, o.kind, o.name, o.score, centre, size, yaw, 0.0, 0.0, o.mask, o.info)
            candidates = []
            for m in objects:
                if family(m.kind) != family(o.kind) or id(m) in matched:
                    continue
                dist = float(np.linalg.norm(m.center[:2] - centre[:2]))
                close = dist < max(cfg["same_object_m"], 0.5 * max(m.size[:2]))
                overlap = m.contains_xy(centre[:2], 0.1) or probe.contains_xy(m.center[:2], 0.1)
                if close or overlap:
                    candidates.append((dist, m))
            if candidates:
                twin = min(candidates, key=lambda t: t[0])[1]
                matched.add(id(twin))
            if twin is not None:
                w_a, w_b = twin.points, o.points
                # Same object seen twice: average where it is (bottom and top heights too);
                # take the larger horizontal extent (each photo may see a different side).
                z_lo = (twin.zmin * w_a + o.zmin * al.scale * w_b) / (w_a + w_b)
                z_hi = (twin.top * w_a + o.top * al.scale * w_b) / (w_a + w_b)
                # Horizontal: each photo may see a different PART -> cover both (union of footprints).
                c, s_ = math.cos(twin.yaw), math.sin(twin.yaw)
                axes = np.array([[c, s_], [-s_, c]])
                corners = np.concatenate([twin.footprint(), probe.footprint()]) @ axes.T
                lo, hi = corners.min(axis=0), corners.max(axis=0)
                twin.size[:2] = hi - lo
                twin.center[:2] = ((lo + hi) / 2) @ axes
                twin.zmin, twin.top = float(z_lo), float(z_hi)
                twin.size[2] = max(z_hi - z_lo, 0.01)
                twin.center[2] = (z_lo + z_hi) / 2
                twin.seen_in = sorted(set(twin.seen_in + o.seen_in))
                if family(twin.kind) == "desk" and "desk" in (twin.kind, o.kind):
                    twin.kind, twin.name = "desk", "desk"
                    if catalog:
                        twin.info = catalog["desk"]
                twin.score = max(twin.score, o.score)
                twin.points += o.points
            else:
                objects.append(RoomObject(len(objects), o.kind, o.name, o.score, centre, size, yaw, o.zmin * al.scale,
                                          o.top * al.scale, o.mask, o.info, o.back_inferred, o.bottom_inferred,
                                          seen_in=list(o.seen_in), points=o.points))
    for i, o in enumerate(objects):
        o.id = i
    relate(objects, walls, cfg, catalog)
    return objects, walls

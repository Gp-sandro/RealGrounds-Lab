"""Photos of a room -> a measured room: every object with its 3D box, what it rests on, and a physics check.

USAGE (from the realgrounds-voice/ folder)
    python build_room.py --input inputs/dorm            # all photos in the folder
    python build_room.py --input inputs/dorm --cache    # reuse saved network outputs (rules changes: seconds)

OUTPUT  outputs/rooms/<name>/  (one folder per zone when photos don't line up: zone_1, zone_2, ...)
    objects.json / objects.md   every object: name, 3D box, position ("on desk surface"), material, state,
                                affordances, relationships (supported_by, next_to, under, against, hanging_on),
                                mass, and what was inferred rather than seen
    summary.json                lens, camera height, room size, physics checks
    room_mujoco.xml             the room for the MuJoCo physics engine (the agent loads it)
    floor_plan.png              top-down plan (for developers; the product itself is screenless)
    detections_<photo>.png      each object's outline on the photo (for developers)

HOW   depth + lens (MoGe-2) + floor / walls / ceiling (SegFormer) + objects by name (Grounding DINO) with exact
outlines (SAM) -> room planes and 3D boxes -> what rests on / hangs on what (realgrounds/support.py: geometry +
common sense learned from 6,397 labelled real-room relations) -> records -> MuJoCo + physics checks.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml

from realgrounds import physics, room
from realgrounds.config import ROOT
from realgrounds.images import list_images, load_image, save_image

log = logging.getLogger("room")
PKG = ROOT / "realgrounds"


def convert_heic(folder: Path) -> None:
    """iPhone photos (.HEIC) -> .jpg next to them (macOS `sips`, keeps the EXIF lens info)."""
    for heic in sorted(list(folder.glob("*.HEIC")) + list(folder.glob("*.heic"))):
        jpg = heic.with_suffix(".jpg")
        if not jpg.exists():
            subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "92", str(heic), "--out", str(jpg)],
                           capture_output=True, check=False)


def detector_catalog(rc: dict, catalog: dict) -> dict:
    """The kinds the detector asks for: those whose catalog 'vocab' tag (default 'base') is switched on."""
    on = set(rc["detector"].get("vocabulary") or ["base"])
    return {k: v for k, v in catalog.items() if v.get("vocab", "base") in on}


def apply_overrides(rc: dict, sets: list[str] | None) -> dict:
    """--set section.key=value (YAML value), e.g. --set geometry.support_prior=false"""
    for item in sets or []:
        path, _, raw = item.partition("=")
        node, keys = rc, path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return rc


def load_config(sets: list[str] | None = None) -> tuple[dict, dict, dict]:
    """(config.yaml with overrides, the object catalog, the flat geometry settings the room code uses)."""
    rc = apply_overrides(yaml.safe_load((ROOT / "config.yaml").read_text()), sets)
    catalog = yaml.safe_load((PKG / "catalog.yaml").read_text())["objects"]
    cfg = {**rc["geometry"], "seed": rc["seed"], "analysis_width": rc["analysis_width"],
           "default_fov_deg": rc["default_fov_deg"], **rc["multi_view"]}
    return rc, catalog, cfg


def load_models(rc: dict, catalog: dict) -> room.Models:
    from realgrounds.depth import MoGeDepth
    from realgrounds.detect import OpenDetector
    from realgrounds.segformer import SegformerLabeler

    m = rc["models"]
    seg = SegformerLabeler({"model_dir": m["segformer_dir"], "device": m["device"]})
    det = OpenDetector({**rc["detector"], "grounding_dino_dir": m["grounding_dino_dir"], "sam_dir": m["sam_dir"],
                        "device": m["device"]}, detector_catalog(rc, catalog))
    return room.Models(MoGeDepth({"model_dir": m["moge_dir"], "device": m["device"]}), seg, det)


def models_for(rc: dict, catalog: dict, cache: bool):
    if not cache:
        return load_models(rc, catalog)
    from realgrounds.model_cache import CachedModels
    return CachedModels(lambda: load_models(rc, catalog), rc, catalog)


def agent_spot(box: room.RoomBox, objects: list[room.RoomObject]) -> tuple[float, float]:
    """A free floor spot for the agent: nothing within 35 cm, closest to where the photographer stood."""
    c, s = math.cos(box.axis), math.sin(box.axis)
    U, V = np.array([c, s]), np.array([-s, c])
    floor_things = [o for o in objects if o.support in ("floor", None)]
    best, best_d = None, 1e9
    roomiest, most_room = (0.0, 0.0), -1.0
    for a in np.arange(box.lo[0] + 0.1, box.hi[0] - 0.1, 0.05):
        for b in np.arange(box.lo[1] + 0.1, box.hi[1] - 0.1, 0.05):
            p = a * U + b * V
            clear_wall = min(a - box.lo[0], box.hi[0] - a, b - box.lo[1], box.hi[1] - b)
            clear = min(clear_wall, min((_gap(o, p) for o in floor_things), default=9.0))
            if clear > most_room:
                roomiest, most_room = (float(p[0]), float(p[1])), clear
            if clear >= 0.35 and float(np.linalg.norm(p)) < best_d:
                best, best_d = (float(p[0]), float(p[1])), float(np.linalg.norm(p))
    return best if best is not None else roomiest


def _gap(o: room.RoomObject, p: np.ndarray) -> float:
    c, s = math.cos(o.yaw), math.sin(o.yaw)
    d = p - o.center[:2]
    du = max(abs(d @ np.array([c, s])) - o.size[0] / 2, 0.0)
    dv = max(abs(d @ np.array([-s, c])) - o.size[1] / 2, 0.0)
    return float(math.hypot(du, dv))


def detections_picture(v: room.RoomView, objects: list[room.RoomObject], path: Path) -> None:
    img = v.small.copy()
    rng = np.random.default_rng(7)
    for o in objects:
        if v.name not in o.seen_in or o.mask.shape != img.shape[:2]:
            continue
        col = tuple(int(x) for x in rng.integers(60, 255, 3))
        tint = img.astype(np.float32)
        tint[o.mask] = 0.6 * tint[o.mask] + 0.4 * np.array(col)
        img = tint.astype(np.uint8)
        cnts, _ = cv2.findContours(o.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, col, 2)
        ys, xs = np.nonzero(o.mask)
        cv2.putText(img, f"#{o.id} {o.name}", (int(np.median(xs)) - 40, int(np.median(ys))), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
    save_image(path, img)


def export_zone(out: Path, title: str, ref: room.RoomView, members: list[room.RoomView], transforms: dict,
                cfg: dict, catalog: dict, tried: list, t0: float) -> dict:
    """Merge one group of aligned photos into a room and write every output for it."""
    out.mkdir(parents=True, exist_ok=True)
    objects, walls = room.merge_group(ref, members, transforms, cfg, catalog)
    used = [ref] + members
    pts = [ref.world[(ref.stuff == 1)][:, :2]]
    for v in members:
        fl = v.world[v.stuff == 1]
        if len(fl):
            pts.append(transforms[v.name].apply(fl)[:, :2])
    box = room.room_box(walls, ref.room_axis, np.concatenate(pts) if any(len(p) for p in pts) else np.zeros((0, 2)),
                        objects, ref.ceiling)
    room.fit_inside(objects, box)
    floor_rgb = ref.small[ref.stuff == 1].mean(axis=0) if (ref.stuff == 1).mean() > 0.02 else np.array([140, 125, 110])
    wall_rgb = ref.small[ref.stuff == 2].mean(axis=0) if (ref.stuff == 2).mean() > 0.02 else np.array([225, 222, 215])
    spot = agent_spot(box, objects)
    xml = physics.to_mjcf(out / "room_mujoco.xml", box, objects, [], floor_rgb, wall_rgb, spot)
    proofs = physics.physics_proofs(xml, objects)
    physics.floor_plan(out / "floor_plan.png", box, objects, walls, spot, ref.fov_deg,
                       f"{title}: {box.hi[0] - box.lo[0]:.1f} x {box.hi[1] - box.lo[1]:.1f} m")
    for v in used:
        detections_picture(v, objects, out / f"detections_{v.name}.png")
        save_image(out / f"photo_{v.name}.jpg", v.small)
    (out / "objects.json").write_text(json.dumps([room.record(o) for o in objects], indent=2), encoding="utf-8")
    (out / "objects.md").write_text("\n\n".join(room.record_text(o) for o in objects) + "\n", encoding="utf-8")
    names = {v.name for v in used}
    summary = {
        "title": title, "folder": out.name, "photos": [v.name for v in used], "reference_photo": ref.name,
        "views": [{"photo": v.name, "fov_deg": round(v.fov_deg, 1), "fov_source": v.fov_source, "up_from": v.up_source,
                   "camera_height_m": round(v.plane.offset, 2), "scale": v.scale_note, "objects": len(v.objects),
                   "ceiling_m": round(v.ceiling, 2), "ceiling_seen": not v.ceiling_inferred, "walls_seen": len(v.walls)}
                  for v in used],
        "alignment": [{"photo": a.view, "ok": a.ok, "note": a.note} for a in tried if any(n in a.note.split(":")[0] for n in names)],
        "room_m": [round(float(box.hi[0] - box.lo[0]), 2), round(float(box.hi[1] - box.lo[1]), 2), round(box.height, 2)],
        "walls_seen": box.sides_seen, "objects": len(objects),
        "by_kind": {k: sum(o.kind == k for o in objects) for k in sorted({o.kind for o in objects})},
        "physics": proofs, "agent_spawn": spot, "seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n=== {title}: {len(objects)} objects, room {summary['room_m']} m, {len(used)} photo(s) ===")
    for o in objects:
        rel = "; ".join(f"{k}: {', '.join(v)}" for k, v in o.relations.items())
        print(f"  #{o.id:<2} {o.name:15} {o.info['state']:8} {room.position_text(o):38} {rel}")
    if proofs.get("ran"):
        print(f"  physics: settle {'OK' if proofs['settle_ok'] else 'FAILED'}, overlaps left {proofs['start_overlaps_left']}")
    return summary


def build(folder: Path, name: str | None = None, cache: bool = False, sets: list[str] | None = None,
          models=None, out_root: Path | None = None) -> Path:
    """Build the room from every photo in `folder`; returns the output folder. `models`: already loaded
    networks (many rooms in one run load them once); `out_root`: where the room folder goes (default
    outputs/rooms)."""
    rc, catalog, cfg = load_config(sets)
    convert_heic(folder)
    photos = list_images(folder)
    if not photos:
        raise FileNotFoundError(f"No photos in {folder}")
    out = (out_root or ROOT / "outputs" / "rooms") / (name or folder.name)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    models = models or models_for(rc, catalog, cache)
    views = []
    for p in photos:
        v = room.build_view(p.stem, load_image(p, rc["max_side"]), p, models, catalog, cfg)
        log.info("%s: lens %.1f deg (%s), camera %.2f m, %d objects", p.name, v.fov_deg, v.fov_source, v.plane.offset, len(v.objects))
        views.append(v)
    groups, tried = room.plan_groups(views, cfg)
    for k, (ref, transforms) in enumerate(groups):
        members = [v for v in views if v.name in transforms]
        zone_out = out if len(groups) == 1 else out / f"zone_{k + 1}"
        export_zone(zone_out, folder.name + ("" if len(groups) == 1 else f" - zone {k + 1}"), ref, members, transforms,
                    cfg, catalog, tried, t0)
    if len(groups) > 1:
        (out / "summary.json").write_text(json.dumps({
            "zones": [f"zone_{k + 1}" for k in range(len(groups))], "photos": [p.name for p in photos],
            "alignment": [{"photo": a.view, "ok": a.ok, "note": a.note} for a in tried],
            "note": "photos that could not be aligned with each other are separate zones of the room",
            "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    print(f"  total {round(time.time() - t0, 1)} s, output: {out}/")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Room photos -> a measured room (objects, relations, physics).")
    ap.add_argument("--input", required=True, help="folder with one or more photos of the same room")
    ap.add_argument("--name", help="output name (default: folder name)")
    ap.add_argument("--cache", action="store_true", help="reuse saved network outputs (depth, structure, detections)")
    ap.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE", help="override a config.yaml setting")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("transformers", "httpx", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    folder = Path(args.input) if Path(args.input).is_absolute() else ROOT / args.input
    build(folder, args.name, args.cache, args.set)
    return 0


if __name__ == "__main__":
    sys.exit(main())

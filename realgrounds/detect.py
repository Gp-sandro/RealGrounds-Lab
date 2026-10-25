"""Open-vocabulary objects: find ANY named object in a photo, with its exact outline.

WHAT THIS FILE DOES (in plain words)
    YOLO only knows 80 fixed classes (no "notebook", "calendar", "picture frame", "desk").
    This uses two newer models instead:

      Grounding DINO  you give it words ("water bottle. desk. calendar.") and it finds those
                      things in the photo (boxes + how sure it is). Any word works, so the
                      vocabulary is just the list in catalog.yaml.
      SAM             "Segment Anything": given a box, it draws the object's exact outline
                      (a pixel mask) - the object's BOUNDARY.

    Both load only from the local models/ folder.

    How the words are asked (detect):
      1. word groups - similar things in one pass (backpack / tote bag / duffel bag), so the
         detector chooses the best name for each thing;
      2. second look - Grounding DINO is unsure with lists: a word can find nothing in its group
         yet score well alone (a dorm poster: nothing in a list of 8 words, 0.50 alone). Each
         kind not found yet is asked again ALONE, with a stricter bar;
      3. context look - things that come in pairs (a chair at a desk, a keyboard by a monitor)
         are accepted with a lower bar when the partner is right there.
    Looks 2 and 3 only ADD objects: they never replace or rename one already found.
    A word asked alone tends to stick to SOMETHING (a poster with dates on it found as
    "calendar"), so each object added by look 2 gets a name check: its crop is shown to the
    detector with its whole word group, and the best-scoring name wins (detect, then verify).
    Look 3 needs no check - the partner next to it is the evidence (a chair's crop shows mostly desk).

WHERE IT FITS
    lab/room_world.py (indoor training grounds) calls detect() for each photo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import get_device, resolve

log = logging.getLogger(__name__)


@dataclass
class Found:
    """One object found in one photo."""

    kind: str               # catalog key, e.g. "water_bottle"
    phrase: str             # what the detector matched, e.g. "water bottle"
    score: float            # 0..1 detector confidence
    box: np.ndarray         # [x0, y0, x1, y1] pixels in the photo
    mask: np.ndarray        # (H, W) bool outline from SAM (the Boundary)
    mask_score: float       # SAM's own quality estimate


def _box_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Pixels between two boxes [x0, y0, x1, y1] (0 if they touch or overlap)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(np.hypot(dx, dy))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class OpenDetector:
    """Grounding DINO (find by words) + SAM (exact outline)."""

    def __init__(self, cfg: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

        self.torch = torch
        self.device = get_device(cfg.get("device", "auto"))
        gd = resolve(cfg["grounding_dino_dir"])
        sam = resolve(cfg["sam_dir"])
        self.gd_proc = AutoProcessor.from_pretrained(gd, local_files_only=True)
        self.gd = AutoModelForZeroShotObjectDetection.from_pretrained(gd, local_files_only=True).eval().to(self.device)
        self.sam_proc = SamProcessor.from_pretrained(sam, local_files_only=True)
        self.sam = SamModel.from_pretrained(sam, local_files_only=True).eval().to(self.device)
        self.cfg = cfg
        self.catalog = catalog
        # prompt words -> catalog key ("water bottle" -> "water_bottle")
        self.by_prompt = {v["prompt"].lower(): k for k, v in catalog.items()}
        log.info("open-vocabulary detector on %s: %s + %s", self.device, Path(gd).name, Path(sam).name)

    def _key_for(self, phrase: str) -> str | None:
        """Map the detector's matched phrase back to a catalog key (exact, then partial match)."""
        p = phrase.lower().strip(" .")
        if p in self.by_prompt:
            return self.by_prompt[p]
        best = [k for prompt, k in self.by_prompt.items() if prompt in p or p in prompt]
        return max(best, key=lambda k: len(self.catalog[k]["prompt"])) if best else None

    def find_boxes(self, image: np.ndarray, kinds: list[str], apply_min: bool = True) -> list[tuple[str, str, float, np.ndarray]]:
        """Grounding DINO with one short list of words. apply_min=False keeps hits below a class's min_score."""
        torch = self.torch
        h, w = image.shape[:2]
        out = []
        for group in [kinds]:
            text = " ".join(f"{self.catalog[k]['prompt']}." for k in group)
            inputs = self.gd_proc(images=image, text=text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.gd(**inputs)
            res = self.gd_proc.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.cfg["box_threshold"], text_threshold=self.cfg["text_threshold"],
                target_sizes=[(h, w)])[0]
            labels = res["text_labels"] if "text_labels" in res else res["labels"]
            for box, score, phrase in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), labels):
                key = self._key_for(str(phrase))
                if key is None:
                    continue
                min_score = self.cfg.get("min_score", {}).get(key, self.cfg["box_threshold"])
                if float(score) >= min_score or not apply_min:
                    out.append((key, str(phrase), float(score), box.astype(np.float32)))
        return out

    def outline(self, image: np.ndarray, boxes: list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
        """SAM: one exact mask per box."""
        torch = self.torch
        if not boxes:
            return [], []
        inputs = self.sam_proc(image, input_boxes=[[b.tolist() for b in boxes]], return_tensors="pt")
        # The Mac GPU has no 64-bit floats: send only the image and boxes, as float32.
        model_in = {"pixel_values": inputs["pixel_values"].to(self.device, dtype=torch.float32),
                    "input_boxes": inputs["input_boxes"].to(self.device, dtype=torch.float32)}
        with torch.no_grad():
            outputs = self.sam(**model_in, multimask_output=False)
        masks = self.sam_proc.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
        masks = masks[:, 0].numpy().astype(bool)
        scores = outputs.iou_scores.cpu().numpy().reshape(-1).tolist()
        return list(masks), scores

    def passes(self) -> list[list[str]]:
        """Word lists for the detector. Similar things go in the SAME pass (backpack vs tote bag),
        so the detector chooses between them instead of finding the same bag twice."""
        groups = self.cfg.get("groups") or []
        listed = {k for g in groups for k in g}
        rest = [k for k in self.catalog if k not in listed]
        size = int(self.cfg["words_per_pass"])
        return [[k for k in g if k in self.catalog] for g in groups] + [rest[i:i + size] for i in range(0, len(rest), size)]

    def _new_boxes(self, hits: list, kept: list, rank=None) -> list:
        """Hits (strongest first) whose box is not already taken (same object found under two names,
        e.g. "desk" and "table": the more confident one stays). `rank(hit)` overrides "strongest"."""
        rank = rank or (lambda h: h[2])
        out: list = []
        for hit in sorted(hits, key=lambda h: -rank(h)):
            if not any(_iou(hit[3], k[3]) > self.cfg["duplicate_iou"] for k in kept + out):
                out.append(hit)
        return out

    def _new_outlines(self, image: np.ndarray, hits: list, accepted: list[Found], rank=None) -> list[Found]:
        """SAM outlines for `hits`; keep those whose pixels are not already someone else's.

        One pixel belongs to one object (the Boundary idea): a weaker detection whose outline lies
        mostly inside a stronger one's outline is the same thing seen twice (e.g. a bed frame also
        called "chair") -> dropped. Things ON other things (book on desk) don't overlap in their
        outlines, only in their boxes, so they stay."""
        rank = rank or (lambda h: h[2])
        masks, mscores = self.outline(image, [h[3] for h in hits])
        found = [(rank(h), Found(*h, m, ms)) for h, m, ms in zip(hits, masks, mscores) if m.sum() > 50]
        new: list[Found] = []
        for _, f in sorted(found, key=lambda t: -t[0]):
            covered = max((float((f.mask & u.mask).sum()) / f.mask.sum() for u in accepted + new), default=0.0)
            if covered < self.cfg["same_pixels_max"]:
                new.append(f)
        return new

    def _more_looks(self, image: np.ndarray, kept: list) -> list[tuple[tuple, bool]]:
        """Looks 2 and 3 (see the file header): kinds not found yet, each asked alone.
        Returns [(hit, next_to_its_partner)]."""
        have = {k[0] for k in kept}
        bar, ctx_bar = self.cfg.get("second_look_min", 0.35), self.cfg.get("context_min", 0.3)
        context = self.cfg.get("context_look") or {}
        w = image.shape[1]
        out = []
        for kind in self.catalog:
            if kind in have:
                continue
            partners = [k[3] for k in kept if k[0] in context.get(kind, ())]
            for key, phrase, score, box in self.find_boxes(image, [kind], apply_min=False):
                if key != kind:
                    continue
                # asked alone, scores run higher: a class with known false alarms (min_score) needs extra
                strict = self.cfg.get("min_score", {}).get(kind)
                alone_ok = score >= max(bar, strict + self.cfg.get("second_look_extra", 0.1) if strict else 0.0)
                # partner "right there": the boxes touch or are less than 5 % of the photo apart
                near = any(_box_gap(box, pb) < 0.05 * w for pb in partners)
                if alone_ok or (near and score >= ctx_bar):
                    out.append(((key, phrase, score, box), near))
        return out

    def _name_check(self, image: np.ndarray, hit: tuple) -> tuple:
        """Re-name one added object when two views agree on a better name:
          * on its crop (box + 15 %) each word of its group is asked alone, and another word beats
            the current name by at least `rename_margin` (a poster: 0.75 vs "calendar" 0.57);
          * and in the whole photo that word also finds this box with `second_look_min`.
        Crops alone are over-confident (a wall thermostat scored 0.76 as "mouse" on its crop),
        hence the second view."""
        kind, phrase, score, box = hit
        group = next((g for g in self.passes() if kind in g), [kind])
        h, w = image.shape[:2]
        mx, my = 0.15 * (box[2] - box[0]), 0.15 * (box[3] - box[1])
        x0, y0 = int(max(box[0] - mx, 0)), int(max(box[1] - my, 0))
        x1, y1 = int(min(box[2] + mx, w)), int(min(box[3] + my, h))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return hit
        inner = box - np.array([x0, y0, x0, y0], np.float32)          # the object's box inside the crop
        crop = image[y0:y1, x0:x1]
        votes: dict[str, float] = {}
        for word in group:
            for k, _, s, b in self.find_boxes(crop, [word], apply_min=False):
                if k == word and _iou(b, inner) > 0.5:
                    votes[word] = max(votes.get(word, 0.0), float(s))
        others = {k: v for k, v in votes.items() if k != kind}
        if not others:
            return hit
        best = max(others, key=others.get)
        if others[best] < votes.get(kind, 0.0) + self.cfg.get("rename_margin", 0.1):
            return hit
        full = [s for k, _, s, b in self.find_boxes(image, [best], apply_min=False) if k == best and _iou(b, box) > 0.5]
        if not full or max(full) < self.cfg.get("second_look_min", 0.35):
            return hit
        log.info("name check: %s %.2f -> %s %.2f", kind, score, best, max(full))
        return best, self.catalog[best]["prompt"], float(max(full)), box

    def detect(self, image: np.ndarray) -> list[Found]:
        """Find every catalog object in the photo, remove duplicates, outline each with SAM."""
        hits = []
        for group in self.passes():
            if group:
                hits += self.find_boxes(image, group)
        kept = self._new_boxes(hits, [])
        found = self._new_outlines(image, kept, [])
        if self.cfg.get("second_look", True):
            looks = self._more_looks(image, kept)
            by_context = {id(h) for h, near in looks if near}
            # a thing next to its partner outranks other names for the same pixels
            # (a keyboard by the monitor, rather than "laptop")
            bonus = self.cfg.get("context_bonus", 0.1)
            rank = lambda h: h[2] + (bonus if id(h) in by_context else 0.0)  # noqa: E731
            extra = self._new_boxes([h for h, _ in looks], kept, rank)
            extra = [h if id(h) in by_context else self._name_check(image, h) for h in extra]
            by_context |= {id(h) for h in extra if id(h) in by_context}
            extra = self._new_boxes(extra, kept, rank)
            added = self._new_outlines(image, extra, found, rank) if extra else []
            if added:
                log.info("second look added: %s", ", ".join(f"{f.kind} {f.score:.2f}" for f in added))
            found += added
        log.info("found %d objects: %s", len(found), ", ".join(f"{f.kind} {f.score:.2f}" for f in found))
        return found

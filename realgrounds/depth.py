"""Depth + lens from one photo with MoGe-2 (Microsoft, MIT licence) - the indoor pipeline's depth model.

WHY (measured on 182 never-seen room photos, lab/experiments/room_benchmark/notes.md)
    The previous model (Depth Anything V2 metric indoor) got the SHAPE of a room right but its SIZE wrong:
    45 % of rooms came out more than 15 % too big or too small, and the photo's lens had to come from EXIF
    or a 70 deg guess. MoGe-2 estimates the lens itself (median error 1.1 deg) and the metric size:
    3 % of rooms more than 15 % off, depth error 0.188 -> 0.070. In the whole pipeline (lens unknown) box
    centres moved from 0.73 m to 0.35 m off.

HOW IT IS USED
    predict(image, size_hw, fov_deg=None) -> (depth in metres, horizontal field of view in degrees).
    If the lens is known (camera.yaml / EXIF), it is given to the model, so depth and lens agree; if not,
    the model's own lens estimate is returned and used for the whole room.
    Same `predict(image, size_hw)` call as depth.DepthEstimator for code that only wants depth.

The weights load from models/moge-2-vitl-normal/model.pt only (never downloaded at run time).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from .config import get_device, resolve

log = logging.getLogger(__name__)


class MoGeDepth:
    """Loads MoGe-2 once; estimates metric depth and the lens of a photo."""

    gives_lens = True

    def __init__(self, cfg: dict[str, Any]) -> None:
        import torch
        from moge.model.v2 import MoGeModel

        path = resolve(cfg["model_dir"]) / "model.pt"
        self.device = get_device(cfg["device"])
        self.model = MoGeModel.from_pretrained(str(path)).to(self.device).eval()
        self._torch = torch
        log.info("depth + lens model MoGe-2 loaded from %s on %s", path.parent.name, self.device)

    def predict_with_lens(self, image: np.ndarray, size_hw: tuple[int, int],
                          fov_deg: float | None = None) -> tuple[np.ndarray, float]:
        """(depth map in metres at `size_hw`, horizontal field of view in degrees)."""
        import cv2

        torch = self._torch
        h, w = image.shape[:2]
        t = torch.tensor(image, dtype=torch.float32, device=self.device).permute(2, 0, 1) / 255.0
        with torch.no_grad():
            # full precision: the Mac GPU (MPS) does not support the model's half-precision path
            out = self.model.infer(t, fov_x=fov_deg, use_fp16=False)
        depth = out["depth"].float().cpu().numpy()
        fx = float(out["intrinsics"][0, 0].cpu()) * w                  # normalised -> pixels
        fov = math.degrees(2 * math.atan(w / 2 / fx))
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)  # sky / invalid -> unknown
        depth = cv2.resize(depth, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)
        return depth, fov

    def predict(self, image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        return self.predict_with_lens(image, size_hw)[0]

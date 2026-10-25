"""Room structure: SegFormer (ADE20K classes) tells floor, wall and ceiling pixels apart (room.structure_map)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import get_device, resolve

log = logging.getLogger(__name__)


class SegformerLabeler:
    """Loads SegFormer once; predict_classes() gives the ADE20K class id of every pixel."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        model_dir = resolve(cfg["model_dir"])
        self.device = get_device(cfg.get("device", "auto"))
        self.processor = SegformerImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_dir, local_files_only=True)
        self.model.eval().to(self.device)
        self._torch = torch
        id2label = {int(i): name.strip() for i, name in self.model.config.id2label.items()}
        self.class_names = [id2label[i] for i in range(len(id2label))]
        log.info("segformer loaded from %s on %s", model_dir.name, self.device)

    def predict_classes(self, image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            logits = torch.nn.functional.interpolate(logits, size=size_hw, mode="bilinear", align_corners=False)
            classes = logits.argmax(dim=1)[0]
        return classes.cpu().numpy()

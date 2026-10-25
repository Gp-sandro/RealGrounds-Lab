"""Cache of the slow model outputs (depth + lens, structure, detections) per photo, for fast iteration.

The room pipeline spends ~50 s per photo in the networks and ~1 s in the rules (geometry, supports,
relations, physics). When only the rules change, the networks' outputs are the same - so they are saved
in .cache/model_outputs/ and reused. Every entry is keyed by the photo's pixels AND the settings that
affect that model (detector words/thresholds and the catalog prompts for detection; the depth model and
the lens given to it for depth), so a change to the vocabulary re-runs detection automatically.

Models are only loaded on the first cache miss: a fully cached run starts instantly.
Used by build_room.py --cache.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np

CACHE = Path(__file__).resolve().parents[1] / ".cache" / "model_outputs"


def _key(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
            h.update(str(p.shape).encode())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:24]


class _Store:
    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def get(self, key: str, make: Callable[[], Any]) -> Any:
        path = self.folder / f"{key}.pkl"
        if path.exists():
            self.hits += 1
            return pickle.loads(path.read_bytes())
        self.misses += 1
        value = make()
        path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        return value


class CachedModels:
    """Same interface as room.Models (depth / seg / det), backed by the disk cache."""

    def __init__(self, load: Callable[[], Any], rc: dict, catalog: dict, folder: Path = CACHE):
        self._load, self._real = load, None
        self.store = _Store(folder)
        m = rc["models"]
        depth_sig = {"model": "moge2", "dir": m["moge_dir"]}
        # only the kinds the detector is asked for count (adding a kind it doesn't ask for changes nothing)
        on = set(rc["detector"].get("vocabulary") or ["base"])
        asked = {k: v.get("prompt") for k, v in sorted(catalog.items()) if v.get("vocab", "base") in on}
        det = {k: v for k, v in rc["detector"].items() if k != "vocabulary"}
        det["groups"] = [[k for k in g if k in asked] for g in det.get("groups") or []]
        det["groups"] = [g for g in det["groups"] if g]
        det_sig = {"det": det, "prompts": asked, "gdino": m["grounding_dino_dir"], "sam": m["sam_dir"]}
        self.depth = _Depth(self, depth_sig)
        self.seg = _Seg(self, {"seg": m["segformer_dir"]})
        self.det = _Det(self, det_sig)

    @property
    def real(self):
        if self._real is None:
            self._real = self._load()
        return self._real


class _Depth:
    def __init__(self, owner: CachedModels, sig: dict):
        self.o, self.sig = owner, sig
        self.gives_lens = True

    def predict_with_lens(self, image, size_hw, fov_deg=None):
        k = _key("depth+lens", self.sig, image, list(size_hw), fov_deg)
        return self.o.store.get(k, lambda: self.o.real.depth.predict_with_lens(image, size_hw, fov_deg))

    def predict(self, image, size_hw):
        k = _key("depth", self.sig, image, list(size_hw))
        return self.o.store.get(k, lambda: self.o.real.depth.predict(image, size_hw))


class _Seg:
    def __init__(self, owner: CachedModels, sig: dict):
        self.o, self.sig = owner, sig
        self._names = None

    @property
    def class_names(self):
        if self._names is None:
            self._names = self.o.store.get(_key("seg-names", self.sig), lambda: list(self.o.real.seg.class_names))
        return self._names

    def predict_classes(self, image, size_hw):
        k = _key("seg", self.sig, image, list(size_hw))
        return self.o.store.get(k, lambda: self.o.real.seg.predict_classes(image, size_hw))


class _Det:
    def __init__(self, owner: CachedModels, sig: dict):
        self.o, self.sig = owner, sig

    def detect(self, image):
        k = _key("det", self.sig, image)
        return self.o.store.get(k, lambda: self.o.real.det.detect(image))

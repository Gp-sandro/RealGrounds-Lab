"""Which lens took the photo (horizontal field of view): camera.yaml next to the photo > EXIF > a default.

When neither says, the depth model's own estimate replaces the default (room.build_view).
"""

from __future__ import annotations

from pathlib import Path

from .geometry import fov_from_exif, fov_from_sidecar


def resolve_fov(path: Path, default_fov: float) -> tuple[float, str]:
    fov = fov_from_sidecar(path)
    if fov:
        return fov, "camera.yaml"
    fov = fov_from_exif(path)
    if fov:
        return fov, "exif"
    return default_fov, "config default (guess)"

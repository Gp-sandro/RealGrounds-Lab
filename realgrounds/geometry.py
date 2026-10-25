"""Geometry: turn "pixel + distance" into real 3D positions standing on the ground.

WHAT THIS FILE DOES (in plain words)
    After depth.py we know, for every pixel, how far away it is. This file uses basic
    camera maths to answer: WHERE in the world is that pixel?

      1. Lens -> "field of view" (how wide the camera sees). Read from camera.yaml or
         the photo's EXIF data (fov_from_sidecar / fov_from_exif).
      2. Field of view -> camera matrix (intrinsics).
      3. Pixel + distance + camera matrix -> 3D point (backproject).
      4. Find the ground among those points (fit_ground_plane) - this also tells us how
         high the camera was and how much it was tilted.
      5. Put ground pixels exactly onto that ground (snap_to_plane) - removes depth noise.
      6. Rotate everything so the ground is flat and "up" is up (to_world).
      Extra: depth_edges finds unreliable "flying pixels" at object borders.

TWO COORDINATE SYSTEMS (all in metres)
    Camera frame: x right, y down, z forward (straight out of the lens).
    World frame:  x right, y forward (along the ground), z up; the ground is z = 0 and
                  the camera stands at (0, 0, camera_height).

WHERE IT FITS
    Used by ground.py, step by step, to build the 3D scene from one photo.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Numeric tags inside a photo's EXIF data (the hidden camera info in JPEG files).
EXIF_IFD = 0x8769          # the "Exif" sub-section
EXIF_FOCAL_35MM = 0xA405   # "focal length in 35 mm film equivalent"


EXIF_ORIENTATION = 0x0112  # 5-8 = the picture is shown turned by 90 degrees (phone held upright)


def fov_from_exif(path: Path) -> float | None:
    """Read the lens from the photo's hidden EXIF data and return the field of view (degrees)
    across the photo's WIDTH, as the photo is displayed.

    The 35 mm-equivalent focal length describes the frame's LONG side. For a portrait photo
    (phone held upright) the width is the short side, so its angle is narrower
    (iPhone 24 mm: 74 deg across a landscape photo, 59 deg across a portrait one).
    Returns None if the photo has no such information (downloaded photos often lose it).
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            exif_root = img.getexif()
            exif = exif_root.get_ifd(EXIF_IFD)
            f35 = exif.get(EXIF_FOCAL_35MM)
            w, h = img.size
            if exif_root.get(EXIF_ORIENTATION, 1) in (5, 6, 7, 8):
                w, h = h, w
    except Exception:  # unreadable EXIF is not an error, just no information
        return None
    if not f35:
        return None
    fov_long = fov_from_focal_35mm(float(f35))
    if w >= h:
        return fov_long
    return math.degrees(2 * math.atan(math.tan(math.radians(fov_long) / 2) * w / h))


SIDECAR_FILE = "camera.yaml"


def fov_from_focal_35mm(focal_35mm: float) -> float:
    """Convert a lens size (35 mm equivalent, e.g. 26 mm phone, 180 mm zoom) to a field of view.

    A 35 mm film frame is 36 mm wide; basic trigonometry gives the horizontal angle.
    Small focal length = wide view (26 mm -> ~69°); large = narrow zoom (180 mm -> ~11°).
    """
    return math.degrees(2 * math.atan(36.0 / (2 * focal_35mm)))


def fov_from_sidecar(path: Path) -> float | None:
    """Read the field of view from a `camera.yaml` file in the same folder as the photo.

    Use this when the photo has no EXIF data. Format (one line per photo):
        photo.jpg: {focal_35mm: 70}     or     photo.jpg: {fov_deg: 29}
    Returns None if there is no camera.yaml or it doesn't mention this photo.
    """
    sidecar = path.parent / SIDECAR_FILE
    if not sidecar.is_file():
        return None
    import yaml

    data = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
    entry = data.get(path.name)
    if not isinstance(entry, dict):
        return None
    if "fov_deg" in entry:
        return float(entry["fov_deg"])
    if "focal_35mm" in entry:
        return fov_from_focal_35mm(float(entry["focal_35mm"]))
    raise ValueError(f"{sidecar}: entry for {path.name} needs fov_deg or focal_35mm")


def intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Build the 3x3 "camera matrix" for an image size and field of view.

    It stores the focal length in pixels (f) and the image centre (width/2, height/2).
    With it, a pixel position can be turned into a viewing direction and back.
    """
    f = (width / 2) / math.tan(math.radians(fov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], np.float64)


def depth_edges(depth: np.ndarray, window: int, max_relative_jump: float) -> np.ndarray:
    """Find "flying pixels": unreliable pixels at the border of objects.

    At the edge of e.g. a tree, the depth model mixes the tree's distance with the
    background's, producing in-between distances. In 3D those pixels become streaks
    floating in the air. A pixel is flagged when, within a small `window`, the distance
    changes by more than `max_relative_jump` (e.g. 8 %) of its own distance.
    Returns a True/False map (True = unreliable, drop it).
    """
    import cv2

    kernel = np.ones((window, window), np.uint8)
    d = depth.astype(np.float32)
    spread = cv2.dilate(d, kernel) - cv2.erode(d, kernel)  # local max - local min
    return spread > max_relative_jump * np.maximum(d, 1e-3)


def backproject(depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Turn every pixel into a 3D point (in the camera frame) using its distance.

    Standard pinhole-camera maths: a pixel at column u, row v with distance z sits at
        x = (u - centre_x) / f * z      (right of the lens axis)
        y = (v - centre_y) / f * z      (below the lens axis)
    Result shape: (height, width, 3) - one (x, y, z) point per pixel.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)  # +0.5 = centre of each pixel
    x = (u - k[0, 2]) / k[0, 0] * depth
    y = (v - k[1, 2]) / k[1, 1] * depth
    return np.stack([x, y, depth], axis=-1).astype(np.float32)


@dataclass
class GroundPlane:
    """The ground, described as a flat plane seen from the camera.

    A plane is stored as a direction pointing straight up out of it (`normal`) and the
    distance from the camera to it (`offset`) - which is exactly the camera's height.
    """

    normal: np.ndarray      # (3,) unit normal in camera frame, pointing up (towards the camera side)
    offset: float           # plane: normal · p + offset = 0; offset = camera height (m)
    inlier_ratio: float     # share of candidate points that lie on this plane (fit quality)
    fitted: bool            # False = fallback (level camera at a default height)

    @property
    def camera_height(self) -> float:
        """How high the camera was above the ground, in metres."""
        return float(self.offset)

    @property
    def pitch_deg(self) -> float:
        """Camera tilt relative to the ground: positive = looking down."""
        forward = np.array([0.0, 0.0, 1.0])
        return float(math.degrees(math.asin(np.clip(-(forward @ self.normal), -1.0, 1.0))))


def fit_ground_plane(
    points: np.ndarray,
    iterations: int,
    threshold: float,
    min_points: int,
    fallback_height: float,
    rng: np.random.Generator,
) -> GroundPlane:
    """Find the flat ground among 3D points that SegFormer labelled grass/path/soil/...

    Method: RANSAC ("random sample consensus"), a standard way to fit a shape while
    ignoring bad points:
      * repeat `iterations` times: pick 3 random points, make a plane through them,
        count how many points lie within `threshold` metres of it ("inliers");
      * keep the plane with the most inliers;
      * refine it with an exact best fit (least squares, via SVD) through its inliers.
    Planes steeper than 45° are rejected - that isn't ground.

    If there are too few ground points or the fit fails, it falls back to "camera level,
    `fallback_height` metres high" and marks the result as not fitted (the report warns).
    """
    up_cam = np.array([0.0, -1.0, 0.0])  # "up" in the camera frame if the camera is held level
    fallback = GroundPlane(up_cam.copy(), fallback_height, 0.0, fitted=False)
    if len(points) < min_points:
        log.warning("only %d ground points; assuming a level camera %.1f m high", len(points), fallback_height)
        return fallback

    best_inliers, best = 0, None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(sample[1] - sample[0], sample[2] - sample[0])  # perpendicular to the 3 points
        norm = np.linalg.norm(n)
        if norm < 1e-9:  # the 3 points are on one line: no plane
            continue
        n /= norm
        if n @ up_cam < 0:  # make the normal point up, not down
            n = -n
        if n @ up_cam < math.cos(math.radians(45)):  # ground can't be steeper than 45° to the camera
            continue
        d = -float(n @ sample[0])
        inliers = int((np.abs(points @ n + d) < threshold).sum())
        if inliers > best_inliers:
            best_inliers, best = inliers, (n, d)

    if best is None or best_inliers < min_points // 2:
        log.warning("ground plane fit failed; assuming a level camera %.1f m high", fallback_height)
        return fallback

    # Refine with least squares on the inliers.
    n, d = best
    inl = points[np.abs(points @ n + d) < threshold]
    centroid = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    n = vt[-1]  # the direction in which the inliers vary least = the plane's normal
    if n @ up_cam < 0:
        n = -n
    d = -float(n @ centroid)
    if d <= 0:  # camera below the "ground": the fit latched onto something else
        log.warning("ground plane is above the camera; assuming a level camera")
        return fallback
    return GroundPlane(n, d, best_inliers / len(points), fitted=True)


def snap_to_plane(points: np.ndarray, plane: GroundPlane, mask: np.ndarray) -> np.ndarray:
    """Put ground pixels exactly on the ground, keeping them on their line of sight.

    Why: the depth model adds small bumps and steps on flat ground (e.g. it put a park
    path 35 cm below the lawn, leaving a gap in the map). Since we already know these
    pixels ARE ground, we slide each one along its viewing ray (the line from the lens
    through that pixel) to where that ray hits the ground plane. That position is more
    exact than the model's guess.

    Only points selected by `mask` move; points whose ray never hits the ground in front
    of the camera (e.g. above the horizon) are left unchanged.
    """
    out = points.copy()
    rays = points[mask]
    denom = rays @ plane.normal
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -plane.offset / denom  # how far along each ray the ground is
    ok = np.isfinite(t) & (t > 0)
    snapped = rays.copy()
    snapped[ok] = rays[ok] * t[ok, None]
    out[mask] = snapped
    return out


def to_world(points: np.ndarray, plane: GroundPlane) -> np.ndarray:
    """Rotate camera-frame points into the level world frame (x right, y forward, z up).

    The camera may have been tilted; this "straightens" the scene using the ground plane:
      z (up)      = height above the ground
      y (forward) = the camera's viewing direction, flattened onto the ground
      x (right)   = perpendicular to both
    After this, flat ground has z = 0 and a 2 m tall tree has z up to 2.
    """
    up = plane.normal
    forward = np.array([0.0, 0.0, 1.0]) - up[2] * up  # viewing direction minus its "up" part
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    x = points @ right
    y = points @ forward
    z = points @ up + plane.offset
    return np.stack([x, y, z], axis=-1).astype(np.float32)

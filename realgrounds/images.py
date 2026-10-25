"""Reading and writing image files.

WHAT THIS FILE DOES (in plain words)
    Finds the photos in an input folder, loads them into memory, and saves pictures
    (previews) back to disk.

    One rule for the whole project: inside the pipeline every image is a NumPy array of
    shape (height, width, 3) with RGB colour values 0-255 (uint8). OpenCV, the library
    used for reading/writing, works in BGR order, so the conversion happens only here.

WHERE IT FITS
    `run.py` uses it to load the input photos; `viewer.py` uses it to save previews.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    """Return all image files in `folder`, sorted by name (other files are ignored)."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(path: Path, max_side: int | None = None) -> np.ndarray:
    """Load one photo as an RGB array.

    Large photos are shrunk so their longest side is at most `max_side` pixels. This keeps
    the pipeline fast; the models do not need more detail than that.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    if max_side is not None:
        h, w = bgr.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:  # only ever shrink, never enlarge
            bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_images(folder: Path, max_side: int | None = None) -> tuple[list[np.ndarray], list[Path]]:
    """Load every photo in a folder. Returns (images, their file paths) in the same order.

    Fails with a clear message if the folder has no photos.
    """
    paths = list_images(folder)
    if not paths:
        raise FileNotFoundError(
            f"No images found in {folder} (supported: {', '.join(sorted(IMAGE_EXTENSIONS))})"
        )
    return [load_image(p, max_side) for p in paths], paths


def save_image(path: Path, rgb: np.ndarray) -> None:
    """Save an RGB array as an image file (the format comes from the file extension, e.g. .png)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image: {path}")


def upscale_nearest(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Enlarge a small grid image so each cell becomes a visible square block.

    "Nearest" means no smoothing: a 96x57 world becomes e.g. 960x570 with sharp pixel edges,
    which is what we want for the pixel-art look of the previews.
    """
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)

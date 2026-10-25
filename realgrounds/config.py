"""Paths, devices and library settings for the whole project.

Everything runs offline from models/ (no downloads at run time); caches stay in .cache/ inside the project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent      # the realgrounds-voice/ folder
CACHE_DIR = ROOT / ".cache"

for _var, _sub in (("HF_HOME", "huggingface"), ("TORCH_HOME", "torch"), ("MPLCONFIGDIR", "matplotlib")):
    os.environ.setdefault(_var, str(CACHE_DIR / _sub))
os.environ.setdefault("HF_HUB_OFFLINE", "1")                 # models are loaded from models/, never downloaded
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")    # if the Mac GPU lacks an operation, use the CPU


def _load_env(path: Path) -> None:
    """KEY=VALUE lines from .env (git-ignored: API keys) into the environment; a variable already set wins."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env(ROOT / ".env")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def get_device(preference: str = "auto") -> Any:
    """cuda > mps (Mac GPU) > cpu with "auto"; any other value forces that device."""
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve(path: str | Path) -> Path:
    """A path from config.yaml: relative paths are relative to the project folder."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p

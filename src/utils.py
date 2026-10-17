"""Shared helpers: config loading, seeding and device selection."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config and expand ``~`` in every value under ``paths``."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg.get("paths", {}).items():
        cfg["paths"][key] = os.path.expanduser(value)
    return cfg


def set_seed(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch (CPU and CUDA). Returns the seed for logging."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def wsi_open_path(path: str | Path) -> Path:
    """File to pass to OpenSlide: the slide file itself, or for a DICOM-WSI series directory
    one of its instances (OpenSlide finds the other instances in the same directory)."""
    path = Path(path)
    if path.is_dir():
        instances = sorted(path.glob("*.dcm"))
        if not instances:
            raise FileNotFoundError(f"no DICOM instances in {path}")
        return instances[0]
    return path


def path_bytes(path: str | Path) -> int:
    """Size of a file, or total size of the files in a directory tree (0 if absent)."""
    path = Path(path)
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size if path.exists() else 0

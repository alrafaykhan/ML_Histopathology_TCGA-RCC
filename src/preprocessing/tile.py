"""Tissue masking and fixed-resolution tiling of whole-slide images."""

from __future__ import annotations

import cv2
import numpy as np
import openslide
from PIL import Image

FALLBACK_MPP = {"40": 0.25, "20": 0.5}


def slide_mpp(slide: openslide.OpenSlide) -> float:
    """Level-0 microns per pixel, from the MPP property or the objective power."""
    mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp:
        return float(mpp)
    objective = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, "")
    if objective.split(".")[0] in FALLBACK_MPP:
        return FALLBACK_MPP[objective.split(".")[0]]
    raise ValueError("slide has neither MPP nor a known objective power")


def tissue_mask(rgb: np.ndarray, min_area_frac: float = 0.0005) -> np.ndarray:
    """Binary tissue mask: Otsu threshold on HSV saturation, morphological cleanup,
    removal of small fragments and of near-black regions (pen, folds)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = cv2.medianBlur(hsv[..., 1], 7)
    thr, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (sat > max(thr, 20)) & (hsv[..., 2] > 40)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_frac * mask.size
    return keep[labels]


def tissue_coords(
    slide: openslide.OpenSlide,
    target_mpp: float = 1.0,
    patch_size: int = 224,
    min_tissue: float = 0.5,
    thumb_px_per_patch: int = 8,
) -> tuple[np.ndarray, float]:
    """Level-0 (x, y) of non-overlapping patches with at least ``min_tissue`` tissue.

    Returns the coordinates and the level-0 patch footprint in pixels.
    """
    footprint = patch_size * target_mpp / slide_mpp(slide)
    w0, h0 = slide.dimensions
    scale = footprint / thumb_px_per_patch
    thumb = slide.get_thumbnail((int(w0 / scale), int(h0 / scale))).convert("RGB")
    mask = tissue_mask(np.asarray(thumb))

    ny, nx = mask.shape[0] // thumb_px_per_patch, mask.shape[1] // thumb_px_per_patch
    k = thumb_px_per_patch
    frac = mask[: ny * k, : nx * k].reshape(ny, k, nx, k).mean(axis=(1, 3))
    iy, ix = np.nonzero(frac >= min_tissue)
    coords = np.stack([ix * footprint, iy * footprint], axis=1).round().astype(np.int64)
    return coords, footprint


def read_patch(
    slide: openslide.OpenSlide, x: int, y: int, footprint: float, patch_size: int = 224
) -> Image.Image:
    """Read one patch at level-0 position (x, y) and resample it to ``patch_size``."""
    level = slide.get_best_level_for_downsample(footprint / patch_size + 1e-6)
    ds = slide.level_downsamples[level]
    size = int(round(footprint / ds))
    region = slide.read_region((int(x), int(y)), level, (size, size)).convert("RGB")
    if size != patch_size:
        region = region.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
    return region


def sample_coords(coords: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Seeded uniform sample of at most ``n`` coordinates, returned in raster order."""
    if len(coords) <= n:
        return coords
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(coords), size=n, replace=False))
    return coords[idx]

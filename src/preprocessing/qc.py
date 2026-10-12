"""Slide quality-control measures: pen-mark detection and blur scoring."""

from __future__ import annotations

import cv2
import numpy as np


def pen_mask(rgb: np.ndarray, min_sat: int = 60) -> np.ndarray:
    """Pixels of blue or green marker ink.

    H&E stains occupy pink-to-purple hues (OpenCV hue roughly 125-180 and 0-15); marker
    ink used on TCGA slides is mostly blue or green (hue 35-125) and well saturated.
    Black ink is already excluded by the value threshold in ``tissue_mask``.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ink = (hue >= 35) & (hue <= 125) & (sat >= min_sat) & (val >= 30)
    ink = cv2.morphologyEx(
        ink.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    return ink.astype(bool)


def laplacian_variance(rgb: np.ndarray) -> float:
    """Variance of the Laplacian of the grayscale image; low values indicate blur."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

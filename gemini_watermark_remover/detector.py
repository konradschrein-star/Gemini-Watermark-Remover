"""
Watermark detection and calibrated alpha map extraction for Gemini sparkle and Veo logos.
Based on the GargantuaX & Dearabhin reverse alpha blending standard catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np


class WatermarkType(str, Enum):
    GEMINI_SPARKLE = "gemini_sparkle"
    VEO_LOGO = "veo_logo"
    AUTO = "auto"
    MANUAL = "manual"


@dataclass
class BoundingBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


TEMPLATES_DIR = Path(__file__).parent / "templates"


def load_calibrated_alpha_template() -> np.ndarray:
    """
    Loads the official 96x96 float32 alpha map.
    First checks for the binary .npy array, then falls back to PNG assets.
    """
    npy_path = TEMPLATES_DIR / "alpha_96_20260520.npy"
    if npy_path.exists():
        return np.load(str(npy_path))

    png_2026 = TEMPLATES_DIR / "bg_96_20260520.png"
    if png_2026.exists():
        img = cv2.imread(str(png_2026))
        return np.max(img, axis=2).astype(np.float32) / 255.0

    png_std = TEMPLATES_DIR / "bg_96.png"
    if png_std.exists():
        img = cv2.imread(str(png_std))
        # Scaled to match video calibrated opacity
        return (np.max(img, axis=2).astype(np.float32) / 255.0) * (0.365 / 0.514)

    # Fallback synthetic star
    grid = np.zeros((96, 96), dtype=np.float32)
    cv2.circle(grid, (48, 48), 24, 0.35, -1)
    return grid


def compute_ncc(roi: np.ndarray, alpha_template: np.ndarray) -> float:
    """Computes Normalized Cross Correlation between luminance and alpha template."""
    if roi.size == 0 or alpha_template.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_zero = gray - np.mean(gray)
    a_zero = alpha_template - np.mean(alpha_template)
    norm = np.linalg.norm(g_zero) * np.linalg.norm(a_zero)
    if norm <= 1e-6:
        return 0.0
    return float(np.sum(g_zero * a_zero) / norm)


def resolve_watermark_candidates(w: int, h: int) -> List[Tuple[str, int, int, int]]:
    """
    Returns candidate (label, size, margin_right, margin_bottom).
    Matches GargantuaX & dearabhin catalogs.
    """
    candidates = []

    # 1080p exact candidates
    if w == 1920 and h == 1080:
        candidates.append(("veo-1080p-inset", 72, 144, 144))
        candidates.append(("veo-1080p-standard", 72, 108, 108))
    # 720p exact candidates
    elif w == 1280 and h == 720:
        candidates.append(("veo-720p-inset", 48, 96, 96))
        candidates.append(("veo-720p-standard", 48, 72, 72))

    # General image and proportional candidates
    if w >= 1024 and h >= 1024:
        candidates.append(("gemini-image-large", 96, 64, 64))
    else:
        candidates.append(("gemini-image-small", 48, 32, 32))

    # Scale-proportional candidates for arbitrary resolutions
    base = min(w, h)
    prop_size = max(16, round(base * (72 / 1080)))
    prop_margin = max(16, round(base * (144 / 1080)))
    candidates.append(("proportional-inset", prop_size, prop_margin, prop_margin))

    return candidates


def compute_veo_roi(frame: np.ndarray) -> Tuple[BoundingBox, np.ndarray, np.ndarray]:
    """
    ROI and mask for legacy Veo text logo.
    """
    h, w = frame.shape[:2]
    x0 = int(w * 0.83)
    y0 = int(h * 0.88)
    x1 = w
    y1 = h
    bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    diff = cv2.subtract(gray, blur)
    bright_th = float(np.percentile(gray, 85))
    raw = ((gray >= bright_th) & (diff >= 6)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(raw, kernel, iterations=1)
    # Binary alpha
    alpha_map = (mask.astype(np.float32) / 255.0) * 0.5
    return bbox, mask, alpha_map


def detect_watermark(
    frame: np.ndarray,
    watermark_type: WatermarkType = WatermarkType.AUTO,
    manual_region: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[WatermarkType, BoundingBox, np.ndarray, np.ndarray]:
    """
    Detects the watermark region and creates the corresponding alpha map and binary mask.
    Returns: (watermark_type, bbox, binary_mask, alpha_map)
    """
    h, w = frame.shape[:2]
    base_alpha = load_calibrated_alpha_template()

    if manual_region is not None:
        mx, my, mw, mh = manual_region
        bbox = BoundingBox(x0=mx, y0=my, x1=min(w, mx + mw), y1=min(h, my + mh))
        alpha_map = cv2.resize(base_alpha, (bbox.width, bbox.height), interpolation=cv2.INTER_AREA)
        mask = (alpha_map > 0.02).astype(np.uint8) * 255
        return WatermarkType.MANUAL, bbox, mask, alpha_map

    if watermark_type == WatermarkType.VEO_LOGO:
        bbox, mask, alpha_map = compute_veo_roi(frame)
        return WatermarkType.VEO_LOGO, bbox, mask, alpha_map

    # Evaluate candidate catalog with NCC
    candidates = resolve_watermark_candidates(w, h)
    best_ncc = -1.0
    best_candidate = None
    best_bbox = None
    best_alpha = None

    for label, size, mr, mb in candidates:
        x0 = w - mr - size
        y0 = h - mb - size
        if x0 < 0 or y0 < 0 or x0 + size > w or y0 + size > h:
            continue
        roi = frame[y0:y0+size, x0:x0+size]
        a_resized = cv2.resize(base_alpha, (size, size), interpolation=cv2.INTER_AREA)
        ncc = compute_ncc(roi, a_resized)

        if ncc > best_ncc:
            best_ncc = ncc
            best_candidate = (label, size)
            best_bbox = BoundingBox(x0=x0, y0=y0, x1=x0+size, y1=y0+size)
            best_alpha = a_resized

    # Fine-tune coordinates around peak candidate
    if best_bbox is not None and best_alpha is not None and best_ncc > 0.15:
        size = best_candidate[1]
        best_x = best_bbox.x0
        best_y = best_bbox.y0
        peak_ncc = best_ncc

        for dy in range(-3, 4):
            for dx in range(-3, 4):
                cx = best_bbox.x0 + dx
                cy = best_bbox.y0 + dy
                if cx < 0 or cy < 0 or cx + size > w or cy + size > h:
                    continue
                roi = frame[cy:cy+size, cx:cx+size]
                ncc = compute_ncc(roi, best_alpha)
                if ncc > peak_ncc:
                    peak_ncc = ncc
                    best_x = cx
                    best_y = cy

        best_bbox = BoundingBox(x0=best_x, y0=best_y, x1=best_x+size, y1=best_y+size)
        mask = (best_alpha > 0.015).astype(np.uint8) * 255
        return WatermarkType.GEMINI_SPARKLE, best_bbox, mask, best_alpha

    # If no sparkle detected and AUTO, test legacy Veo text logo
    if watermark_type == WatermarkType.AUTO:
        bbox, mask, alpha_map = compute_veo_roi(frame)
        if int((mask > 0).sum()) > 50:
            return WatermarkType.VEO_LOGO, bbox, mask, alpha_map

    # Default fallback to the primary candidate (e.g. veo-1080p-inset)
    if candidates:
        _, size, mr, mb = candidates[0]
        x0 = w - mr - size
        y0 = h - mb - size
        bbox = BoundingBox(x0=x0, y0=y0, x1=x0+size, y1=y0+size)
        alpha_map = cv2.resize(base_alpha, (size, size), interpolation=cv2.INTER_AREA)
        mask = (alpha_map > 0.015).astype(np.uint8) * 255
        return WatermarkType.GEMINI_SPARKLE, bbox, mask, alpha_map

    # Ultimate fallback
    bbox = BoundingBox(x0=max(0, w-144-72), y0=max(0, h-144-72), x1=w-144, y1=h-144)
    alpha_map = cv2.resize(base_alpha, (bbox.width, bbox.height), interpolation=cv2.INTER_AREA)
    mask = (alpha_map > 0.015).astype(np.uint8) * 255
    return WatermarkType.GEMINI_SPARKLE, bbox, mask, alpha_map

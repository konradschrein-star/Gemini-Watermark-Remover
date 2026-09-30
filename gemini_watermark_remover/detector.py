"""
Watermark detection and mask generation for Gemini sparkle and Veo logos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional

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


def get_template_mask(name: str) -> np.ndarray:
    path = TEMPLATES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Template mask not found at {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to decode template image at {path}")
    return mask


def compute_gemini_sparkle_roi(width: int, height: int) -> Tuple[BoundingBox, np.ndarray]:
    """
    Computes the scaled bounding box and scaled binary mask for the standard Google Gemini
    4-pointed star watermark.
    
    Baseline 1080p reference:
      Full resolution: 1920x1080
      Watermark box: x in [1701, 1781] (width 80), y in [864, 954] (height 90)
      Center: x ~ 1741, y ~ 909
      Offset from right margin: 179 px (9.32% of 1920)
      Offset from bottom margin: 171 px (15.83% of 1080)
    """
    base_mask = get_template_mask("gemini_sparkle_mask.png")
    
    scale_x = width / 1920.0
    scale_y = height / 1080.0
    scale = (scale_x + scale_y) / 2.0

    target_w = max(16, int(round(80 * scale)))
    target_h = max(18, int(round(90 * scale)))

    margin_right = int(round(179 * scale_x))
    margin_bottom = int(round(171 * scale_y))

    center_x = width - margin_right
    center_y = height - margin_bottom

    x0 = max(0, center_x - target_w // 2)
    y0 = max(0, center_y - target_h // 2)
    x1 = min(width, x0 + target_w)
    y1 = min(height, y0 + target_h)

    bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    resized_mask = cv2.resize(base_mask, (bbox.width, bbox.height), interpolation=cv2.INTER_LINEAR)
    _, binary_mask = cv2.threshold(resized_mask, 50, 255, cv2.THRESH_BINARY)
    
    # Slight dilation (1-2px) to guarantee swallow of anti-aliasing edges
    kernel_size = max(3, int(round(5 * scale)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    return bbox, dilated_mask


def compute_veo_roi(frame: np.ndarray) -> Tuple[BoundingBox, np.ndarray]:
    """
    Computes the ROI and adaptive bright-pixel mask for Google Veo video clips
    (Konrad's adaptive inpaint strategy from ReelForge).
    """
    h, w = frame.shape[:2]
    # Fixed Veo corner: rightmost ~17% x bottom ~12%
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

    return bbox, mask


def detect_watermark(
    frame: np.ndarray,
    watermark_type: WatermarkType = WatermarkType.AUTO,
    manual_region: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[WatermarkType, BoundingBox, np.ndarray]:
    """
    Detects the watermark region and creates the corresponding inpainting mask.
    """
    h, w = frame.shape[:2]

    if manual_region is not None:
        mx, my, mw, mh = manual_region
        bbox = BoundingBox(x0=mx, y0=my, x1=min(w, mx + mw), y1=min(h, my + mh))
        mask = np.ones((bbox.height, bbox.width), dtype=np.uint8) * 255
        return WatermarkType.MANUAL, bbox, mask

    if watermark_type == WatermarkType.GEMINI_SPARKLE:
        bbox, mask = compute_gemini_sparkle_roi(w, h)
        return WatermarkType.GEMINI_SPARKLE, bbox, mask

    if watermark_type == WatermarkType.VEO_LOGO:
        bbox, mask = compute_veo_roi(frame)
        return WatermarkType.VEO_LOGO, bbox, mask

    # WatermarkType.AUTO:
    # Heuristic test between Veo text and Gemini sparkle
    # Check Gemini sparkle position first
    sparkle_bbox, sparkle_mask = compute_gemini_sparkle_roi(w, h)
    roi_sparkle = frame[sparkle_bbox.y0:sparkle_bbox.y1, sparkle_bbox.x0:sparkle_bbox.x1]
    
    # Check if there is significant brightness / edge presence in the sparkle zone
    if roi_sparkle.size > 0:
        gray_sp = cv2.cvtColor(roi_sparkle, cv2.COLOR_BGR2GRAY)
        blur_sp = cv2.GaussianBlur(gray_sp, (9, 9), 0)
        diff_sp = cv2.subtract(gray_sp, blur_sp)
        # In the presence of a sparkle, local highpass contrast is elevated
        if np.mean(diff_sp) > 1.2 or np.max(diff_sp) > 15:
            return WatermarkType.GEMINI_SPARKLE, sparkle_bbox, sparkle_mask

    # Default to Gemini sparkle for images/modern Google generations, or Veo fallback
    return WatermarkType.GEMINI_SPARKLE, sparkle_bbox, sparkle_mask

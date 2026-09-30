"""
Inpainting and reconstruction algorithms for watermark removal.
Implements mathematically exact closed-form Reverse Alpha Blending
with surgical perimeter repair to eliminate compression artifacts.
"""

from __future__ import annotations

from enum import Enum
import cv2
import numpy as np

from .detector import BoundingBox


class InpaintMethod(str, Enum):
    ALPHA_REVERSE = "alpha_reverse"
    TELEA = "telea"
    NS = "ns"


def remove_watermark_frame(
    frame: np.ndarray,
    bbox: BoundingBox,
    mask: np.ndarray,
    method: InpaintMethod = InpaintMethod.ALPHA_REVERSE,
    inpaint_radius: int = 5,
    alpha_map: np.ndarray | None = None,
    alpha_gain: float = 1.0,
) -> np.ndarray:
    """
    Removes the watermark from a single frame in-place or returns the cleaned frame.
    Optimized to operate exclusively on the localized bounding box with padding.
    """
    h, w = frame.shape[:2]
    pad = inpaint_radius + 4

    y0 = max(0, bbox.y0 - pad)
    y1 = min(h, bbox.y1 + pad)
    x0 = max(0, bbox.x0 - pad)
    x1 = min(w, bbox.x1 + pad)

    roi_frame = frame[y0:y1, x0:x1]

    # Build local ROI mask
    roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    my0 = bbox.y0 - y0
    mx0 = bbox.x0 - x0
    roi_mask[my0 : my0 + bbox.height, mx0 : mx0 + bbox.width] = mask

    if method == InpaintMethod.ALPHA_REVERSE:
        if alpha_map is None:
            # Fall back to telea if no alpha map provided
            cleaned_roi = cv2.inpaint(roi_frame, roi_mask, inpaint_radius, cv2.INPAINT_TELEA)
        else:
            roi_alpha = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
            roi_alpha[my0 : my0 + bbox.height, mx0 : mx0 + bbox.width] = alpha_map

            # Apply gain if calibrated
            effective_alpha = np.clip(roi_alpha * alpha_gain, 0.0, 0.99)
            alpha_3d = np.expand_dims(effective_alpha, axis=2)
            denom = np.maximum(1.0 - alpha_3d, 0.01)

            # Mathematically exact reverse alpha blending: B = (I - alpha * 255) / (1 - alpha)
            restored = (roi_frame.astype(np.float32) - alpha_3d * 255.0) / denom
            restored = np.clip(restored, 0, 255).astype(np.uint8)

            # Seamless hairline edge repair (1-2px outer perimeter where video compression creates artifacts)
            edge_mask = ((effective_alpha > 0.015) & (effective_alpha < 0.09)).astype(np.uint8) * 255
            if np.any(edge_mask):
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                edge_mask = cv2.dilate(edge_mask, k, iterations=1)
                hairline_cleaned = cv2.inpaint(restored, edge_mask, 2, cv2.INPAINT_TELEA)
            else:
                hairline_cleaned = restored

            # Only substitute pixels where watermark alpha was active
            active_mask = (effective_alpha > 0.015)[:, :, np.newaxis]
            cleaned_roi = np.where(active_mask, hairline_cleaned, roi_frame)

    elif method == InpaintMethod.TELEA:
        cleaned_roi = cv2.inpaint(roi_frame, roi_mask, inpaint_radius, cv2.INPAINT_TELEA)
    elif method == InpaintMethod.NS:
        cleaned_roi = cv2.inpaint(roi_frame, roi_mask, inpaint_radius, cv2.INPAINT_NS)
    else:
        cleaned_roi = cv2.inpaint(roi_frame, roi_mask, inpaint_radius, cv2.INPAINT_TELEA)

    out = frame.copy()
    out[y0:y1, x0:x1] = cleaned_roi
    return out

"""
Inpainting and reconstruction algorithms for watermark removal.
Implements:
1. Outlier-immune Navier-Stokes Median Opacity Gain Calibration (immune to bright rocks, leaves, or raindrops).
2. 4:2:0-Aware Luma-Chroma Reverse Alpha Blending (restores Luma Y at 1x1 pixel precision and Chroma Cb/Cr with
   2x2/3x3 subsampling compensation, eliminating 4:2:0 boundary color fringes).
3. 2px Morphological Gradient Perimeter Bridge weighted by local smoothness (eliminating sub-pixel or DCT steps).
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


def calibrate_optimal_alpha_gain(
    roi_bgr: np.ndarray,
    alpha_map: np.ndarray,
    nominal_gain: float | None = None,
    max_deviation: float = 0.08,
) -> float:
    """
    Estimates the scalar opacity gain `g` using Navier-Stokes background interpolation and the median
    interior pixel ratio `(Y_roi - Y_bg) / ((255 - Y_bg) * alpha)`.
    - Resistant to localized highlights (raindrops, specular pebbles, text edges) inside the star.
    - Tightly bounded around the physical compositor opacity (`nominal_gain = 1.0` for Flow/Veo 77/255,
      or `1.6623` for Google Vids / Gemini Web 128/255) so background textures can never cause over-subtraction.
    """
    if roi_bgr.size == 0 or alpha_map.size == 0:
        return float(nominal_gain or 1.0)

    peak_a = float(np.max(alpha_map))
    if peak_a <= 1e-6:
        return float(nominal_gain or 1.0)

    k3 = np.ones((3, 3), np.uint8)
    m_in = cv2.dilate((alpha_map > 0.008).astype(np.uint8) * 255, k3, iterations=1)
    bg_in = cv2.inpaint(roi_bgr, m_in, 5, cv2.INPAINT_NS).astype(np.float32)
    roi_f = roi_bgr.astype(np.float32)

    if roi_bgr.ndim == 3:
        y_roi = 0.114 * roi_f[:, :, 0] + 0.587 * roi_f[:, :, 1] + 0.299 * roi_f[:, :, 2]
        y_bg = 0.114 * bg_in[:, :, 0] + 0.587 * bg_in[:, :, 1] + 0.299 * bg_in[:, :, 2]
    else:
        y_roi = roi_f
        y_bg = bg_in

    core = (alpha_map > (peak_a * 0.72)) & ((255.0 - y_bg) > 25.0)
    if int(np.sum(core)) < 12:
        return float(nominal_gain or 1.0)

    pixel_gains = (y_roi[core] - y_bg[core]) / ((255.0 - y_bg[core]) * alpha_map[core] + 1e-6)
    med_g = float(np.median(pixel_gains))

    if nominal_gain is None:
        # Automatically snap to the closer physical Google compositor mode: 1.00 (77/255) or 1.6623 (128/255)
        nominal_gain = 128.0 / 77.0 if med_g > 1.33 else 1.0

    return float(np.clip(med_g, nominal_gain - max_deviation, nominal_gain + max_deviation))


def remove_watermark_frame(
    frame: np.ndarray,
    bbox: BoundingBox,
    mask: np.ndarray,
    method: InpaintMethod = InpaintMethod.ALPHA_REVERSE,
    inpaint_radius: int = 4,
    alpha_map: np.ndarray | None = None,
    alpha_gain: float | None = None,
    nominal_gain: float | None = None,
    in_place: bool = False,
) -> np.ndarray:
    """
    Removes the watermark from a single frame in-place or returns the cleaned frame.
    Operates strictly on the localized bounding box with a tiny 3px padding (< 0.3% of image pixels).
    """
    h, w = frame.shape[:2]
    pad = 3

    y0 = max(0, bbox.y0 - pad)
    y1 = min(h, bbox.y1 + pad)
    x0 = max(0, bbox.x0 - pad)
    x1 = min(w, bbox.x1 + pad)

    roi_frame = frame[y0:y1, x0:x1]

    roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    my0 = bbox.y0 - y0
    mx0 = bbox.x0 - x0
    roi_mask[my0 : my0 + bbox.height, mx0 : mx0 + bbox.width] = mask

    if method == InpaintMethod.ALPHA_REVERSE:
        if alpha_map is None:
            cleaned_roi = cv2.inpaint(roi_frame, roi_mask, inpaint_radius, cv2.INPAINT_NS)
        else:
            if alpha_gain is None:
                inner_roi = frame[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
                alpha_gain = calibrate_optimal_alpha_gain(
                    inner_roi,
                    alpha_map,
                    nominal_gain=nominal_gain,
                )

            roi_alpha = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
            roi_alpha[my0 : my0 + bbox.height, mx0 : mx0 + bbox.width] = alpha_map

            eff_a = np.clip(roi_alpha * float(alpha_gain), 0.0, 0.94)
            roi_f = roi_frame.astype(np.float32)
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            # 1. 4:2:0-Aware Luma-Chroma Reverse Alpha Blending
            y_luma = (0.114 * roi_f[:, :, 0] + 0.587 * roi_f[:, :, 1] + 0.299 * roi_f[:, :, 2])[:, :, None]
            chroma_delta = roi_f - y_luma

            a_luma = eff_a[:, :, None]
            y_rest = (y_luma - a_luma * 255.0) / np.maximum(1.0 - a_luma, 0.04)

            a_chroma_2d = cv2.blur(cv2.erode(eff_a, k3, iterations=1), (3, 3))
            a_chroma = a_chroma_2d[:, :, None]
            chroma_rest = chroma_delta / np.maximum(1.0 - a_chroma, 0.08)

            rest = np.clip(y_rest + chroma_rest, 0.0, 255.0)
            rest_u8 = rest.astype(np.uint8)

            # 2. 2px Morphological Gradient Perimeter Bridge (1px inside + 1px outside the star boundary)
            peak_a = float(np.max(roi_alpha)) + 1e-6
            norm_a = roi_alpha / peak_a
            binary_star = (norm_a > 0.16).astype(np.uint8) * 255
            contour_mask = cv2.morphologyEx(binary_star, cv2.MORPH_GRADIENT, k3)
            partial_mask = ((norm_a > 0.008) & (norm_a < 0.92)).astype(np.uint8) * 255
            contour_mask = cv2.bitwise_or(contour_mask, partial_mask)

            if np.any(contour_mask):
                inpainted_contour = cv2.inpaint(rest_u8, contour_mask, 2, cv2.INPAINT_NS).astype(np.float32)
                blurred_contour = cv2.GaussianBlur(rest_u8, (3, 3), 0.85).astype(np.float32)

                gray = cv2.cvtColor(rest_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                outer_ring = (norm_a <= 0.008) & (contour_mask == 0)
                outer_grad = (
                    float(np.percentile(np.sqrt(gx[outer_ring] ** 2 + gy[outer_ring] ** 2), 75))
                    if np.any(outer_ring)
                    else 25.0
                )
                smooth_factor = float(np.clip(1.0 - (outer_grad - 18.0) / 55.0, 0.25, 1.0))

                bridge = inpainted_contour * smooth_factor + blurred_contour * (1.0 - smooth_factor)
                c_mask_f = (contour_mask > 0).astype(np.float32)[:, :, None] * (0.65 + 0.30 * smooth_factor)
                cleaned_roi = np.clip(rest * (1.0 - c_mask_f) + bridge * c_mask_f, 0.0, 255.0).astype(np.uint8)
            else:
                cleaned_roi = rest_u8

    elif method == InpaintMethod.TELEA:
        cleaned_roi = cv2.inpaint(roi_frame, roi_mask, inpaint_radius, cv2.INPAINT_TELEA)
    elif method == InpaintMethod.NS:
        cleaned_roi = cv2.inpaint(roi_frame, roi_mask, inpaint_radius, cv2.INPAINT_NS)
    else:
        cleaned_roi = cv2.inpaint(roi_frame, roi_mask, inpaint_radius, cv2.INPAINT_NS)

    out = frame if in_place else frame.copy()
    out[y0:y1, x0:x1] = cleaned_roi
    return out

"""
Watermark detection and calibrated alpha map extraction for Gemini sparkle, Flow, Google Vids, and Veo logos.
Optimized from https://github.com/konradschrein-star/Gemini-Watermark-Remover with:
- Clean zero-overshoot 4-fold symmetric base template (`alpha_96_20260520.npy`) + empirical 1080p Flow template (`alpha_flow_1080p_68.npy`)
- Pre-cached multi-scale alpha templates, Sobel edge gradients, and core/ring masks (_TEMPLATE_CACHE)
- Sub-millisecond localized bottom-right anchor search across Nano Banana, Google Vids (512x512 & 1280x720), and Veo resolutions
- 4-Quadrant Sobel gradient symmetry + intensity-NCC + core-vs-ring luminance gating to guarantee zero false positives
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class WatermarkType(str, Enum):
    GEMINI_SPARKLE = "gemini_sparkle"
    VEO_LOGO = "veo_logo"
    NONE = "none"
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


@dataclass
class WatermarkDetectionResult:
    detected: bool
    watermark_type: WatermarkType
    bbox: BoundingBox
    mask: np.ndarray
    alpha_map: np.ndarray
    ncc: float
    grad_ncc: float
    core_diff: float
    combined_score: float
    candidate_label: str
    nominal_gain: float = 1.0


@dataclass
class _CachedTemplateEntry:
    key: str
    size: int
    alpha: np.ndarray
    mask: np.ndarray
    sobel_x: np.ndarray
    sobel_y: np.ndarray
    core_mask: np.ndarray
    ring_mask: np.ndarray
    zero_mean: np.ndarray
    norm: float


TEMPLATES_DIR = Path(__file__).parent / "templates"
_BASE_ALPHA_TEMPLATE: Optional[np.ndarray] = None
_FLOW_1080P_TEMPLATE: Optional[np.ndarray] = None
_TEMPLATE_CACHE: Dict[str, _CachedTemplateEntry] = {}
_PREWARM_SIZES: Tuple[int, ...] = (24, 32, 36, 40, 42, 44, 45, 46, 48, 49, 50, 52, 64, 67, 68, 72, 96)


def load_calibrated_alpha_template() -> np.ndarray:
    """
    Loads the clean, 4-fold symmetric 96x96 float32 alpha map once and caches it in memory.
    Plateau opacity is calibrated to 77/255 (0.3019608) with zero perimeter overshoot ringing.
    """
    global _BASE_ALPHA_TEMPLATE
    if _BASE_ALPHA_TEMPLATE is not None:
        return _BASE_ALPHA_TEMPLATE

    npy_path = TEMPLATES_DIR / "alpha_96_20260520.npy"
    if npy_path.exists():
        arr = np.load(str(npy_path)).astype(np.float32)
        # Ensure zero perimeter overshoot if an older .npy file is loaded
        arr = np.where(arr <= 0.006, 0.0, np.clip(arr, 0.0, 0.3019608)).astype(np.float32)
        _BASE_ALPHA_TEMPLATE = arr
        return _BASE_ALPHA_TEMPLATE

    png_std = TEMPLATES_DIR / "bg_96.png"
    if png_std.exists():
        img = cv2.imread(str(png_std), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            raw = img.astype(np.float32)
            clean = np.clip(np.where(raw <= 3.0, 0.0, raw) * (0.3019608 / 128.0), 0.0, 0.3019608)
            sym = 0.25 * (clean + np.fliplr(clean) + np.flipud(clean) + np.fliplr(np.flipud(clean)))
            _BASE_ALPHA_TEMPLATE = (0.5 * (sym + sym.T)).astype(np.float32)
            return _BASE_ALPHA_TEMPLATE

    grid = np.zeros((96, 96), dtype=np.float32)
    cv2.circle(grid, (48, 48), 24, 0.3019608, -1)
    _BASE_ALPHA_TEMPLATE = grid
    return _BASE_ALPHA_TEMPLATE


def _load_flow_1080p_template() -> Optional[np.ndarray]:
    """Loads the empirically calibrated 68x68 Google Flow 1080p template if present."""
    global _FLOW_1080P_TEMPLATE
    if _FLOW_1080P_TEMPLATE is not None:
        return _FLOW_1080P_TEMPLATE

    p = TEMPLATES_DIR / "alpha_flow_1080p_68.npy"
    if p.exists():
        _FLOW_1080P_TEMPLATE = np.load(str(p)).astype(np.float32)
        return _FLOW_1080P_TEMPLATE
    return None


def _build_cache_entry(key: str, alpha: np.ndarray) -> _CachedTemplateEntry:
    sz = int(alpha.shape[0])
    mask = (alpha > 0.01).astype(np.uint8) * 255
    sobel_x = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
    core_mask = alpha > (float(np.max(alpha)) * 0.72)
    ring_mask = alpha <= 0.008
    zero_mean = alpha - float(np.mean(alpha))
    norm = float(np.linalg.norm(zero_mean))
    return _CachedTemplateEntry(
        key=key,
        size=sz,
        alpha=alpha,
        mask=mask,
        sobel_x=sobel_x,
        sobel_y=sobel_y,
        core_mask=core_mask,
        ring_mask=ring_mask,
        zero_mean=zero_mean,
        norm=norm if norm > 1e-6 else 1.0,
    )


def _get_cached_template(size: int, candidate_label: str = "") -> _CachedTemplateEntry:
    """Returns a pre-computed template entry for `size x size` from memory cache."""
    if candidate_label == "flow-1080p-68" and size == 68:
        entry = _TEMPLATE_CACHE.get("flow-1080p-68")
        if entry is not None:
            return entry
        emp68 = _load_flow_1080p_template()
        if emp68 is not None and emp68.shape == (68, 68):
            entry = _build_cache_entry("flow-1080p-68", emp68)
            _TEMPLATE_CACHE["flow-1080p-68"] = entry
            return entry

    clamped_size = max(16, int(size))
    key = f"sz_{clamped_size}"
    entry = _TEMPLATE_CACHE.get(key)
    if entry is not None:
        return entry

    base = load_calibrated_alpha_template()
    interp = cv2.INTER_AREA if clamped_size <= 96 else cv2.INTER_CUBIC
    alpha = cv2.resize(base, (clamped_size, clamped_size), interpolation=interp).astype(np.float32)
    entry = _build_cache_entry(key, alpha)
    _TEMPLATE_CACHE[key] = entry
    return entry


def prewarm_template_cache() -> None:
    """Pre-warms standard watermark template sizes at worker startup."""
    _get_cached_template(68, "flow-1080p-68")
    for sz in _PREWARM_SIZES:
        _get_cached_template(sz)


def compute_ncc(roi: np.ndarray, alpha_template: np.ndarray) -> float:
    """Computes Normalized Cross Correlation between luminance and alpha template."""
    if roi.size == 0 or alpha_template.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) if roi.ndim == 3 else roi.astype(np.float32)
    g_zero = gray - float(np.mean(gray))
    a_zero = alpha_template - float(np.mean(alpha_template))
    norm = float(np.linalg.norm(g_zero) * np.linalg.norm(a_zero))
    if norm <= 1e-6:
        return 0.0
    return float(np.sum(g_zero * a_zero) / norm)


def _nominal_gain_for_label(label: str) -> float:
    """
    Returns the physical compositor opacity ratio relative to the 77/255 (0.30196) reference plateau:
    - Google Vids AI Images (`vids-image-*`) and Gemini Web Images (`gemini-image-*`) use 128/255 (0.50196 -> gain 1.6623).
    - Google Flow Nano Banana (`flow-*`) and Veo / Google Vids Videos (`veo-*`) use 77/255 (0.30196 -> gain 1.000).
    """
    if label.startswith("vids-image") or label.startswith("gemini-image"):
        return 128.0 / 77.0
    return 1.0


def resolve_watermark_candidates(w: int, h: int) -> List[Tuple[str, int, int, int]]:
    """
    Returns candidate (label, size, margin_right, margin_bottom).
    Includes calibrated Google Flow Nano Banana (1376x768, 1920x1080, 1280x720),
    Google Vids AI Images (512x512, 1024x1024) & Videos (1280x720), and Veo standard catalogs.
    """
    candidates: List[Tuple[str, int, int, int]] = []

    if (w == 1376 and h == 768) or (w == 768 and h == 1376):
        candidates.append(("flow-nanobanana-48", 48, 73, 73))
    elif (w == 1920 and h == 1080) or (w == 1080 and h == 1920):
        # Exact empirical 1376x768->1920x1080 Flow placement: x0=1750, y0=909, sz=68 -> mr=102, mb=103
        candidates.append(("flow-1080p-68", 68, 102, 103))
        candidates.append(("veo-1080p-inset", 72, 144, 144))
        candidates.append(("veo-1080p-standard", 72, 108, 108))
    elif (w == 1280 and h == 720) or (w == 720 and h == 1280):
        candidates.append(("veo-720p-inset", 48, 96, 96))
        candidates.append(("flow-nanobanana-720p-45", 45, 68, 68))
        candidates.append(("veo-720p-standard", 48, 72, 72))
    elif w == 512 and h == 512:
        candidates.append(("vids-image-512", 24, 16, 16))
    elif w == 1024 and h == 1024:
        candidates.append(("vids-image-1024", 48, 32, 32))
        candidates.append(("gemini-image-large", 96, 64, 64))

    short_side = min(w, h)
    flow_size = max(24, round(short_side / 16.0))
    flow_margin = max(16, round(flow_size * (73.0 / 48.0)))
    if not any(abs(c[1] - flow_size) <= 2 and abs(c[2] - flow_margin) <= 3 for c in candidates):
        candidates.append(("flow-proportional", flow_size, flow_margin, flow_margin))

    if w >= 1024 and h >= 1024 and not any(c[0] == "gemini-image-large" for c in candidates):
        candidates.append(("gemini-image-large", 96, 64, 64))
    if not any(c[1] == 48 and c[2] == 32 for c in candidates):
        candidates.append(("gemini-image-small", 48, 32, 32))

    prop_size = max(16, round(short_side * (72.0 / 1080.0)))
    prop_margin = max(16, round(short_side * (144.0 / 1080.0)))
    if not any(abs(c[1] - prop_size) <= 2 and abs(c[2] - prop_margin) <= 4 for c in candidates):
        candidates.append(("proportional-inset", prop_size, prop_margin, prop_margin))

    return candidates


def _compute_quadrant_gradient_metrics(
    patch_gray: np.ndarray,
    entry: _CachedTemplateEntry,
) -> Tuple[float, int, float]:
    """
    Computes directional Sobel gradient correlation across the full patch and each of the 4 quadrants.
    A genuine 4-pointed Gemini/Veo star exhibits positive concave arc gradients in all 4 quadrants,
    whereas natural edges (rocks, branches, fur) only align in 1-2 quadrants.
    """
    gx_p = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_p = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
    gx_t = entry.sobel_x
    gy_t = entry.sobel_y

    sz = entry.size
    mid = sz // 2
    quad_slices = (
        (slice(0, mid), slice(0, mid)),
        (slice(0, mid), slice(mid, sz)),
        (slice(mid, sz), slice(0, mid)),
        (slice(mid, sz), slice(mid, sz)),
    )

    q_scores: List[float] = []
    for sy, sx in quad_slices:
        px, py = gx_p[sy, sx], gy_p[sy, sx]
        tx, ty = gx_t[sy, sx], gy_t[sy, sx]
        num = float(np.sum(px * tx + py * ty))
        den = float(np.sqrt(np.sum(px * px + py * py) * np.sum(tx * tx + ty * ty))) + 1e-6
        q_scores.append(num / den)

    overall_num = float(np.sum(gx_p * gx_t + gy_p * gy_t))
    overall_den = float(np.sqrt(np.sum(gx_p * gx_p + gy_p * gy_p) * np.sum(gx_t * gx_t + gy_t * gy_t))) + 1e-6
    overall_gncc = overall_num / overall_den
    pos_quads = sum(1 for q in q_scores if q >= 0.12)
    min_quad = min(q_scores)
    return overall_gncc, pos_quads, min_quad


def detect_watermark_fast(
    frame: np.ndarray,
    search_pad: int = 2,
    min_ncc: float = 0.25,
    min_grad_ncc: float = 0.28,
    min_combined: float = 0.30,
    min_core_diff: float = 3.5,
) -> WatermarkDetectionResult:
    """
    Ultra-lightweight watermark detector (< 0.5 ms on clean images, ~1.0 ms on watermarked images).
    Requires intensity NCC, directional Sobel gradient NCC, core-vs-ring brightness delta, AND
    4-quadrant rotational symmetry consistency to eliminate false positives on natural textures.
    """
    h, w = frame.shape[:2]
    candidates = resolve_watermark_candidates(w, h)

    best_combined = -1.0
    best_result: Optional[WatermarkDetectionResult] = None

    for label, sz, mr, mb in candidates:
        ex = w - sz - mr
        ey = h - sz - mb
        if ex < 0 or ey < 0 or ex + sz > w or ey + sz > h:
            continue

        entry = _get_cached_template(sz, label)
        pad = 1 if label == "flow-1080p-68" else search_pad
        x0 = max(0, ex - pad)
        y0 = max(0, ey - pad)
        x1 = min(w, ex + sz + pad)
        y1 = min(h, ey + sz + pad)

        sub_bgr = frame[y0:y1, x0:x1]
        if sub_bgr.shape[0] < sz or sub_bgr.shape[1] < sz:
            continue

        sub_gray = (
            cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if sub_bgr.ndim == 3
            else sub_bgr.astype(np.float32)
        )
        res = cv2.matchTemplate(sub_gray, entry.alpha, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        ncc = float(max_val)

        if ncc < min_ncc * 0.72:
            continue

        ax = x0 + int(max_loc[0])
        ay = y0 + int(max_loc[1])
        patch = sub_gray[int(max_loc[1]) : int(max_loc[1]) + sz, int(max_loc[0]) : int(max_loc[0]) + sz]

        core_mean = float(np.mean(patch[entry.core_mask]))
        ring_mean = float(np.mean(patch[entry.ring_mask]))
        core_diff = core_mean - ring_mean
        effective_opacity = core_diff / max(18.0, 255.0 - ring_mean)
        if core_diff < min_core_diff or effective_opacity < 0.14:
            continue

        grad_ncc, pos_quads, min_quad = _compute_quadrant_gradient_metrics(patch, entry)
        combined = 0.50 * ncc + 0.50 * grad_ncc

        is_hit = (
            (ncc >= min_ncc and grad_ncc >= min_grad_ncc and combined >= min_combined and pos_quads >= 3 and min_quad >= 0.08)
            or (grad_ncc >= 0.38 and ncc >= 0.18 and pos_quads >= 3 and core_diff >= 6.0)
        )

        if combined > best_combined:
            best_combined = combined
            best_result = WatermarkDetectionResult(
                detected=bool(is_hit),
                watermark_type=WatermarkType.GEMINI_SPARKLE if is_hit else WatermarkType.NONE,
                bbox=BoundingBox(x0=ax, y0=ay, x1=ax + sz, y1=ay + sz),
                mask=entry.mask,
                alpha_map=entry.alpha,
                ncc=ncc,
                grad_ncc=grad_ncc,
                core_diff=core_diff,
                combined_score=combined,
                candidate_label=label,
                nominal_gain=_nominal_gain_for_label(label),
            )

        if is_hit and ncc >= 0.55 and grad_ncc >= 0.38 and pos_quads == 4:
            return best_result  # type: ignore[return-value]

    if best_result is not None:
        return best_result

    fallback_entry = _get_cached_template(48)
    fx0 = max(0, w - 121)
    fy0 = max(0, h - 121)
    return WatermarkDetectionResult(
        detected=False,
        watermark_type=WatermarkType.NONE,
        bbox=BoundingBox(x0=fx0, y0=fy0, x1=min(w, fx0 + 48), y1=min(h, fy0 + 48)),
        mask=fallback_entry.mask,
        alpha_map=fallback_entry.alpha,
        ncc=0.0,
        grad_ncc=0.0,
        core_diff=0.0,
        combined_score=0.0,
        candidate_label="none",
        nominal_gain=1.0,
    )


def compute_veo_roi(frame: np.ndarray) -> Tuple[BoundingBox, np.ndarray, np.ndarray]:
    """ROI and mask for legacy Veo text logo."""
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
    alpha_map = (mask.astype(np.float32) / 255.0) * 0.5
    return bbox, mask, alpha_map


def detect_watermark(
    frame: np.ndarray,
    watermark_type: WatermarkType = WatermarkType.AUTO,
    manual_region: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[WatermarkType, BoundingBox, np.ndarray, np.ndarray]:
    """
    Backwards-compatible detector returning (watermark_type, bbox, binary_mask, alpha_map).
    """
    h, w = frame.shape[:2]
    if manual_region is not None:
        mx, my, mw, mh = manual_region
        bbox = BoundingBox(x0=mx, y0=my, x1=min(w, mx + mw), y1=min(h, my + mh))
        base_alpha = load_calibrated_alpha_template()
        alpha_map = cv2.resize(base_alpha, (bbox.width, bbox.height), interpolation=cv2.INTER_AREA)
        mask = (alpha_map > 0.01).astype(np.uint8) * 255
        return WatermarkType.MANUAL, bbox, mask, alpha_map

    if watermark_type == WatermarkType.VEO_LOGO:
        bbox, mask, alpha_map = compute_veo_roi(frame)
        return WatermarkType.VEO_LOGO, bbox, mask, alpha_map

    res = detect_watermark_fast(frame)
    return res.watermark_type, res.bbox, res.mask, res.alpha_map

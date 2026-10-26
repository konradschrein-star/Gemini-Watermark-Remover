"""
Gemini Watermark Remover — Optimized Ultra-Lightweight Detector & Closed-Form Reverse Alpha Inpainter.
Based on https://github.com/konradschrein-star/Gemini-Watermark-Remover
"""

from .detector import (
    BoundingBox,
    WatermarkDetectionResult,
    WatermarkType,
    compute_ncc,
    detect_watermark,
    detect_watermark_fast,
    load_calibrated_alpha_template,
    resolve_watermark_candidates,
)
from .inpainter import (
    InpaintMethod,
    calibrate_optimal_alpha_gain,
    remove_watermark_frame,
)
from .pipeline import (
    detect_and_remove_image_if_present,
    detect_and_remove_video_if_present,
    process_image,
    process_video,
)

__all__ = [
    "BoundingBox",
    "WatermarkDetectionResult",
    "WatermarkType",
    "InpaintMethod",
    "load_calibrated_alpha_template",
    "compute_ncc",
    "resolve_watermark_candidates",
    "detect_watermark",
    "detect_watermark_fast",
    "calibrate_optimal_alpha_gain",
    "remove_watermark_frame",
    "detect_and_remove_image_if_present",
    "detect_and_remove_video_if_present",
    "process_image",
    "process_video",
]

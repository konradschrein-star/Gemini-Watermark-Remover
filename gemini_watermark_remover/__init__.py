"""
Gemini & Veo Watermark Remover.

High-fidelity watermark removal for Google Gemini (Imagen 3) sparkle logos
and Google Veo video watermarks with optional AI/VLM verification (e.g. Jev-Omni).
"""

__version__ = "1.0.0"

from .detector import detect_watermark, WatermarkType, BoundingBox
from .inpainter import remove_watermark_frame, InpaintMethod
from .pipeline import process_video, process_image
from .verifier import verify_watermark_removal, VerificationResult

__all__ = [
    "detect_watermark",
    "WatermarkType",
    "BoundingBox",
    "remove_watermark_frame",
    "InpaintMethod",
    "process_video",
    "process_image",
    "verify_watermark_removal",
    "VerificationResult",
]

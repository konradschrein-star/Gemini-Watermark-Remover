"""
End-to-end processing pipelines for video and image watermark removal.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple

import cv2
import numpy as np

from .detector import detect_watermark, WatermarkType, BoundingBox
from .inpainter import remove_watermark_frame, InpaintMethod
from .verifier import verify_watermark_removal, VerificationResult


def process_image(
    input_path: str,
    output_path: str,
    watermark_type: WatermarkType = WatermarkType.AUTO,
    method: InpaintMethod = InpaintMethod.TELEA,
    manual_region: Optional[Tuple[int, int, int, int]] = None,
    verify: bool = False,
    verifier_backend: str = "jev-omni",
) -> Dict[str, Any]:
    """
    Removes watermark from an image file (PNG, JPG, WebP, etc.).
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not open image file at {input_path}")

    w_type, bbox, mask = detect_watermark(img, watermark_type=watermark_type, manual_region=manual_region)
    cleaned = remove_watermark_frame(img, bbox=bbox, mask=mask, method=method)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, cleaned)

    verif_res = None
    if verify:
        verif_res = verify_watermark_removal(cleaned, bbox=bbox, backend=verifier_backend)

    return {
        "status": "success",
        "input": input_path,
        "output": output_path,
        "watermark_type": w_type.value,
        "bbox": {"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1},
        "verification": verif_res.__dict__ if verif_res else None,
    }


def process_video(
    input_path: str,
    output_path: str,
    watermark_type: WatermarkType = WatermarkType.AUTO,
    method: InpaintMethod = InpaintMethod.TELEA,
    manual_region: Optional[Tuple[int, int, int, int]] = None,
    ffmpeg_binary: str = "ffmpeg",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    verify: bool = False,
    verifier_backend: str = "jev-omni",
) -> Dict[str, Any]:
    """
    Processes an entire video clip:
    1. Samples frame to lock watermark mask and ROI.
    2. Inpaints frames at maximum throughput using localized ROI processing.
    3. Re-muxes with FFmpeg preserving all original audio channels (AAC/MP3/Opus).
    4. Performs optional quality verification with Jev-Omni or local CV.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read mid-clip frame to build static watermark mask
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_frames // 2))
    ret, mid_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ret or mid_frame is None:
        cap.release()
        raise ValueError("Could not read sample frame from video")

    w_type, bbox, mask = detect_watermark(
        mid_frame, watermark_type=watermark_type, manual_region=manual_region
    )

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    temp_video = os.path.join(out_dir, f".temp_inpaint_{os.getpid()}_{os.path.basename(output_path)}.avi")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to initialize OpenCV VideoWriter")

    processed = 0
    sample_cleaned_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            cleaned = remove_watermark_frame(frame, bbox=bbox, mask=mask, method=method)
            writer.write(cleaned)

            if processed == n_frames // 2:
                sample_cleaned_frame = cleaned

            processed += 1
            if progress_callback:
                progress_callback(processed, n_frames)

    finally:
        cap.release()
        writer.release()

    # Mux back with ffmpeg preserving audio
    cmd = [
        ffmpeg_binary,
        "-y",
        "-loglevel",
        "error",
        "-i",
        temp_video,
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(temp_video):
            try:
                os.remove(temp_video)
            except OSError:
                pass

    verif_res = None
    if verify and sample_cleaned_frame is not None:
        verif_res = verify_watermark_removal(
            sample_cleaned_frame, bbox=bbox, backend=verifier_backend
        )

    return {
        "status": "success",
        "input": input_path,
        "output": output_path,
        "frames_processed": processed,
        "fps": fps,
        "resolution": f"{w}x{h}",
        "watermark_type": w_type.value,
        "bbox": {"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1},
        "verification": verif_res.__dict__ if verif_res else None,
    }

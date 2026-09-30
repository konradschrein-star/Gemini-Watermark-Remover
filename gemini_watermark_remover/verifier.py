"""
Visual quality assurance and watermark verification module.

Supports evaluating output images or video frames against:
1. Jev-Omni (akhilaaa3/jev-omni) multimodal decision classifier on Hugging Face Spaces.
2. Local Computer Vision edge and high-pass residual analysis (offline fallback).
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Dict, Any, Optional

import cv2
import numpy as np

from .detector import BoundingBox


@dataclass
class VerificationResult:
    is_clean: bool
    confidence: float
    label: str
    backend: str
    details: Dict[str, Any]


def verify_with_local_cv(
    frame: np.ndarray,
    bbox: BoundingBox,
) -> VerificationResult:
    """
    Offline CV verification: checks high-pass edge energy and coherent gradient structures
    inside the cleaned watermark bounding box compared to the local surrounding context.
    """
    h, w = frame.shape[:2]
    crop = frame[bbox.y0:bbox.y1, bbox.x0:bbox.x1]
    if crop.size == 0:
        return VerificationResult(
            is_clean=True,
            confidence=1.0,
            label="Clean (empty ROI)",
            backend="local_cv",
            details={},
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    diff = cv2.subtract(gray, blur)

    # Sharp bright pixels characteristic of residual watermark outlines
    residual_pixels = (diff >= 18).sum()
    total_pixels = crop.shape[0] * crop.shape[1]
    ratio = float(residual_pixels) / float(total_pixels)

    # If sharp edge ratio is low (< 3%), the watermark is cleanly removed
    is_clean = ratio < 0.035
    confidence = max(0.5, 1.0 - (ratio * 10.0))

    return VerificationResult(
        is_clean=is_clean,
        confidence=confidence,
        label="Clean (No residual detected)" if is_clean else "Artifact detected",
        backend="local_cv",
        details={"residual_ratio": ratio, "residual_pixels": int(residual_pixels)},
    )


def verify_with_jev_omni(image_path: str, timeout: int = 15) -> VerificationResult:
    """
    Queries the Jev-Omni multimodal decision classifier on Hugging Face Spaces
    (akhilaaa3/jev-omni) via Gradio API.
    """
    upload_url = "https://akhilaaa3-jev-omni.hf.space/gradio_api/upload"
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

    with open(image_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{os.path.basename(image_path)}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(upload_url, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        upload_resp = json.loads(resp.read().decode("utf-8"))
        remote_path = upload_resp[0]

    decide_url = "https://akhilaaa3-jev-omni.hf.space/gradio_api/call/decide"
    payload = {
        "data": [
            "You are an automated visual QA inspector checking for unwanted watermarks or logos.",
            "Is there any visible watermark or logo in this image?",
            "No watermark visible\nWatermark is visible",
            {"path": remote_path, "meta": {"_type": "gradio.FileData"}},
            "image",
        ]
    }
    decide_req = urllib.request.Request(
        decide_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(decide_req, timeout=timeout) as d_resp:
        event_id = json.loads(d_resp.read().decode("utf-8")).get("event_id")

    result_url = f"https://akhilaaa3-jev-omni.hf.space/gradio_api/call/decide/{event_id}"
    last_data = None
    with urllib.request.urlopen(result_url, timeout=timeout) as sse_resp:
        for line in sse_resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data:"):
                try:
                    parsed = json.loads(line_str[5:].strip())
                    if parsed is not None:
                        last_data = parsed
                except Exception:
                    pass

    if not last_data or not isinstance(last_data, list):
        raise ValueError(f"Unexpected response from Jev-Omni: {last_data}")

    result_obj = last_data[0]
    best_label = result_obj.get("label", "")
    confidences = result_obj.get("confidences", [])

    conf_map = {item["label"]: item["confidence"] for item in confidences if "label" in item}
    clean_conf = conf_map.get("No watermark visible", 0.5)
    is_clean = "No watermark visible" in best_label

    return VerificationResult(
        is_clean=is_clean,
        confidence=clean_conf if is_clean else conf_map.get("Watermark is visible", 0.5),
        label=best_label,
        backend="jev-omni",
        details={"confidences": conf_map, "raw_summary": last_data[1] if len(last_data) > 1 else ""},
    )


def verify_watermark_removal(
    frame_or_path: str | np.ndarray,
    bbox: Optional[BoundingBox] = None,
    backend: str = "jev-omni",
) -> VerificationResult:
    """
    High-level entrypoint for verifying whether a watermark was cleanly removed.
    Automatically handles temporary file generation and falls back to local CV
    if the external API is unreachable or rate-limited.
    """
    if isinstance(frame_or_path, str):
        image_path = frame_or_path
        frame = cv2.imread(image_path)
    else:
        frame = frame_or_path
        image_path = None

    if backend == "jev-omni":
        temp_file = None
        try:
            if image_path is None:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.close()
                temp_file = tmp.name
                # Save cropped ROI or full frame
                if bbox is not None:
                    pad = 20
                    h, w = frame.shape[:2]
                    crop = frame[
                        max(0, bbox.y0 - pad) : min(h, bbox.y1 + pad),
                        max(0, bbox.x0 - pad) : min(w, bbox.x1 + pad),
                    ]
                    cv2.imwrite(temp_file, crop)
                else:
                    cv2.imwrite(temp_file, frame)
                target_path = temp_file
            else:
                target_path = image_path

            return verify_with_jev_omni(target_path)
        except Exception as e:
            # Fall back to local CV if Jev-Omni network / GPU error occurs
            if bbox is not None and frame is not None:
                res = verify_with_local_cv(frame, bbox)
                res.details["jev_omni_fallback_reason"] = str(e)
                return res
            raise e
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    # Default to local CV
    if bbox is not None and frame is not None:
        return verify_with_local_cv(frame, bbox)

    return VerificationResult(
        is_clean=True,
        confidence=1.0,
        label="Skipped verification",
        backend="none",
        details={},
    )

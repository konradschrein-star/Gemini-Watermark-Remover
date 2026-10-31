"""
High-level processing pipelines for images and video streams.
Includes:
- `detect_and_remove_image_if_present` for conditional, sub-millisecond image detection & removal.
- `detect_and_remove_video_if_present` for single-frame yes/no video detection & direct-pipe FFmpeg removal.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, Optional, Tuple

import cv2

from .detector import (
    WatermarkType,
    compute_ncc,
    detect_watermark,
    detect_watermark_fast,
)
from .inpainter import (
    InpaintMethod,
    calibrate_optimal_alpha_gain,
    remove_watermark_frame,
)
from .verifier import verify_watermark_removal


def detect_and_remove_image_if_present(
    input_path: str,
    output_path: Optional[str] = None,
    jpeg_quality: int = 96,
) -> Dict[str, Any]:
    """
    Ultra-lightweight conditional image pipeline:
    1. Reads image and runs `detect_watermark_fast` on the bottom-right corner (< 0.8 ms).
    2. If no Gemini watermark is present, returns immediately without modifying or re-encoding the file.
    3. If a Gemini watermark is detected, calibrates `alpha_gain`, applies closed-form reverse alpha
       blending + 2px hairline edge repair strictly on the watermark bbox, and atomically writes `output_path`.
    """
    t0 = time.perf_counter()
    target_path = output_path or input_path

    if not os.path.exists(input_path):
        return {
            "status": "error",
            "kind": "image",
            "detected": False,
            "error": f"File not found: {input_path}",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    img = cv2.imread(input_path)
    if img is None:
        return {
            "status": "error",
            "kind": "image",
            "detected": False,
            "error": f"Could not decode image: {input_path}",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    det = detect_watermark_fast(img)
    if not det.detected:
        if target_path != input_path and not os.path.exists(target_path):
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            shutil.copyfile(input_path, target_path)
        return {
            "status": "clean_untouched",
            "kind": "image",
            "detected": False,
            "input": input_path,
            "output": target_path,
            "watermark_type": det.watermark_type.value,
            "candidate_label": det.candidate_label,
            "bbox": {"x0": det.bbox.x0, "y0": det.bbox.y0, "x1": det.bbox.x1, "y1": det.bbox.y1},
            "ncc_before": round(det.ncc, 4),
            "grad_ncc_before": round(det.grad_ncc, 4),
            "core_diff": round(det.core_diff, 2),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    inner_roi = img[det.bbox.y0 : det.bbox.y1, det.bbox.x0 : det.bbox.x1]
    alpha_gain = calibrate_optimal_alpha_gain(
        inner_roi,
        det.alpha_map,
        nominal_gain=det.nominal_gain,
    )

    remove_watermark_frame(
        img,
        bbox=det.bbox,
        mask=det.mask,
        method=InpaintMethod.ALPHA_REVERSE,
        alpha_map=det.alpha_map,
        alpha_gain=alpha_gain,
        nominal_gain=det.nominal_gain,
        in_place=True,
    )

    ncc_after = compute_ncc(img[det.bbox.y0 : det.bbox.y1, det.bbox.x0 : det.bbox.x1], det.alpha_map)

    out_abs = os.path.abspath(target_path)
    out_dir = os.path.dirname(out_abs)
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(out_abs)[1].lower() or ".jpg"
    tmp_path = os.path.join(out_dir, f".wm_tmp_{os.getpid()}_{int(time.time() * 1000)}{ext}")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)] if ext in (".jpg", ".jpeg") else []
    ok = cv2.imwrite(tmp_path, img, encode_params)
    if not ok:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return {
            "status": "error",
            "kind": "image",
            "detected": True,
            "error": f"Failed to write cleaned image to {tmp_path}",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    os.replace(tmp_path, out_abs)

    return {
        "status": "removed",
        "kind": "image",
        "detected": True,
        "input": input_path,
        "output": out_abs,
        "watermark_type": det.watermark_type.value,
        "candidate_label": det.candidate_label,
        "bbox": {"x0": det.bbox.x0, "y0": det.bbox.y0, "x1": det.bbox.x1, "y1": det.bbox.y1},
        "alpha_gain": round(alpha_gain, 3),
        "ncc_before": round(det.ncc, 4),
        "grad_ncc_before": round(det.grad_ncc, 4),
        "ncc_after": round(ncc_after, 4),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }


def detect_and_remove_video_if_present(
    input_path: str,
    output_path: Optional[str] = None,
    ffmpeg_binary: str = "ffmpeg",
) -> Dict[str, Any]:
    """
    Single-frame yes/no video watermark detector + direct-pipe FFmpeg remover:
    1. Samples a single frame (midpoint frame `n_frames // 2`) and runs `detect_watermark_fast`.
    2. If `detected == False`, closes the video immediately (~80 ms, zero full-video decode,
       zero FFmpeg spawn, original MP4 left 100% untouched).
    3. If `detected == True`, calibrates `alpha_gain` on the sample frame and streams in-place
       ROI-cleaned BGR24 frames directly into `ffmpeg` (`-preset superfast -crf 18 -c:a copy`),
       eliminating intermediate `.avi` disk I/O.
    """
    t0 = time.perf_counter()
    target_path = output_path or input_path

    if not os.path.exists(input_path):
        return {
            "status": "error",
            "kind": "video",
            "detected": False,
            "error": f"File not found: {input_path}",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {
            "status": "error",
            "kind": "video",
            "detected": False,
            "error": f"Could not open video file: {input_path}",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_indices = sorted(
        set(
            idx
            for idx in (
                0,
                max(0, n_frames // 6),
                max(0, (n_frames * 2) // 6),
                max(0, n_frames // 2),
                max(0, (n_frames * 4) // 6),
                max(0, (n_frames * 5) // 6),
            )
            if 0 <= idx < max(1, n_frames)
        )
    )

    best_sample_frame = None
    best_det = None
    best_idx = 0

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_cand = cap.read()
        if not ret or frame_cand is None:
            continue
        det_cand = detect_watermark_fast(frame_cand)
        if best_det is None:
            best_det = det_cand
            best_sample_frame = frame_cand
            best_idx = idx
        elif det_cand.detected and (
            not best_det.detected
            or (det_cand.ncc + det_cand.grad_ncc) > (best_det.ncc + best_det.grad_ncc)
        ):
            best_det = det_cand
            best_sample_frame = frame_cand
            best_idx = idx

    if best_sample_frame is None or best_det is None:
        cap.release()
        return {
            "status": "error",
            "kind": "video",
            "detected": False,
            "error": f"Could not read sample frame from video: {input_path}",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    sample_frame = best_sample_frame
    det = best_det
    mid_idx = best_idx
    detect_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    if not det.detected:
        cap.release()
        if target_path != input_path and not os.path.exists(target_path):
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            shutil.copyfile(input_path, target_path)
        return {
            "status": "clean_untouched",
            "kind": "video",
            "detected": False,
            "input": input_path,
            "output": target_path,
            "watermark_type": det.watermark_type.value,
            "candidate_label": det.candidate_label,
            "bbox": {"x0": det.bbox.x0, "y0": det.bbox.y0, "x1": det.bbox.x1, "y1": det.bbox.y1},
            "ncc_before": round(det.ncc, 4),
            "grad_ncc_before": round(det.grad_ncc, 4),
            "core_diff": round(det.core_diff, 2),
            "detect_ms": detect_ms,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    inner_roi = sample_frame[det.bbox.y0 : det.bbox.y1, det.bbox.x0 : det.bbox.x1]
    alpha_gain = calibrate_optimal_alpha_gain(
        inner_roi,
        det.alpha_map,
        nominal_gain=det.nominal_gain,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_abs = os.path.abspath(target_path)
    out_dir = os.path.dirname(out_abs)
    os.makedirs(out_dir, exist_ok=True)
    tmp_video = os.path.join(out_dir, f".wm_vid_tmp_{os.getpid()}_{int(time.time() * 1000)}.mp4")

    cmd = [
        ffmpeg_binary,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "superfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        tmp_video,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frames_processed = 0
    ncc_after = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            remove_watermark_frame(
                frame,
                bbox=det.bbox,
                mask=det.mask,
                method=InpaintMethod.ALPHA_REVERSE,
                alpha_map=det.alpha_map,
                alpha_gain=alpha_gain,
                nominal_gain=det.nominal_gain,
                in_place=True,
            )

            if frames_processed == mid_idx:
                ncc_after = compute_ncc(
                    frame[det.bbox.y0 : det.bbox.y1, det.bbox.x0 : det.bbox.x1],
                    det.alpha_map,
                )

            if proc.stdin is not None:
                proc.stdin.write(frame.tobytes())
            frames_processed += 1
    finally:
        cap.release()
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass

    stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
    rc = proc.wait()
    if rc != 0 or not os.path.exists(tmp_video):
        if os.path.exists(tmp_video):
            try:
                os.remove(tmp_video)
            except OSError:
                pass
        return {
            "status": "error",
            "kind": "video",
            "detected": True,
            "error": f"FFmpeg exited with code {rc}: {stderr_bytes.decode('utf-8', 'ignore')[:300]}",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    os.replace(tmp_video, out_abs)

    return {
        "status": "removed",
        "kind": "video",
        "detected": True,
        "input": input_path,
        "output": out_abs,
        "watermark_type": det.watermark_type.value,
        "candidate_label": det.candidate_label,
        "frames_processed": frames_processed,
        "resolution": f"{w}x{h}",
        "bbox": {"x0": det.bbox.x0, "y0": det.bbox.y0, "x1": det.bbox.x1, "y1": det.bbox.y1},
        "alpha_gain": round(alpha_gain, 3),
        "ncc_before": round(det.ncc, 4),
        "grad_ncc_before": round(det.grad_ncc, 4),
        "ncc_after": round(ncc_after, 4),
        "detect_ms": detect_ms,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }


def process_image(
    input_path: str,
    output_path: str,
    watermark_type: WatermarkType = WatermarkType.AUTO,
    method: InpaintMethod = InpaintMethod.ALPHA_REVERSE,
    manual_region: Optional[Tuple[int, int, int, int]] = None,
    verify: bool = False,
    verifier_backend: str = "local",
) -> Dict[str, Any]:
    """Removes watermark from an image file (PNG, JPG, WebP, etc.)."""
    if manual_region is None and watermark_type == WatermarkType.AUTO and not verify:
        return detect_and_remove_image_if_present(input_path, output_path)

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not open image file at {input_path}")

    w_type, bbox, mask, alpha_map = detect_watermark(
        img, watermark_type=watermark_type, manual_region=manual_region
    )
    if w_type == WatermarkType.NONE and manual_region is None:
        if os.path.abspath(input_path) != os.path.abspath(output_path):
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, img)
        return {
            "status": "clean_untouched",
            "input": input_path,
            "output": output_path,
            "watermark_type": w_type.value,
            "bbox": {"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1},
            "verification": None,
        }

    cleaned = remove_watermark_frame(img, bbox=bbox, mask=mask, method=method, alpha_map=alpha_map)

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
    method: InpaintMethod = InpaintMethod.ALPHA_REVERSE,
    manual_region: Optional[Tuple[int, int, int, int]] = None,
    ffmpeg_binary: str = "ffmpeg",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    verify: bool = False,
    verifier_backend: str = "local",
) -> Dict[str, Any]:
    """
    Processes a video clip: runs the single-frame detector first when in AUTO mode,
    and only re-encodes when a watermark is present.
    """
    if (
        manual_region is None
        and watermark_type == WatermarkType.AUTO
        and method == InpaintMethod.ALPHA_REVERSE
        and progress_callback is None
        and not verify
    ):
        return detect_and_remove_video_if_present(
            input_path=input_path,
            output_path=output_path,
            ffmpeg_binary=ffmpeg_binary,
        )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_frames // 2))
    ret, mid_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ret or mid_frame is None:
        cap.release()
        raise ValueError("Could not read sample frame from video")

    w_type, bbox, mask, alpha_map = detect_watermark(
        mid_frame, watermark_type=watermark_type, manual_region=manual_region
    )
    if w_type == WatermarkType.NONE and manual_region is None:
        cap.release()
        if os.path.abspath(input_path) != os.path.abspath(output_path):
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            shutil.copyfile(input_path, output_path)
        return {
            "status": "clean_untouched",
            "input": input_path,
            "output": output_path,
            "frames_processed": 0,
            "fps": fps,
            "resolution": f"{w}x{h}",
            "watermark_type": w_type.value,
            "bbox": {"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1},
            "verification": None,
        }

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    temp_video = os.path.join(out_dir, f".temp_inpaint_{os.getpid()}_{os.path.basename(output_path)}.mp4")

    cmd = [
        ffmpeg_binary,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "superfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        temp_video,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    processed = 0
    sample_cleaned_frame = None
    alpha_gain = calibrate_optimal_alpha_gain(mid_frame[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1], alpha_map)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            cleaned = remove_watermark_frame(
                frame,
                bbox=bbox,
                mask=mask,
                method=method,
                alpha_map=alpha_map,
                alpha_gain=alpha_gain,
                in_place=True,
            )
            if proc.stdin is not None:
                proc.stdin.write(cleaned.tobytes())

            if processed == n_frames // 2:
                sample_cleaned_frame = cleaned.copy()

            processed += 1
            if progress_callback:
                progress_callback(processed, n_frames)
    finally:
        cap.release()
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass

    stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
    rc = proc.wait()
    if rc != 0:
        if os.path.exists(temp_video):
            try:
                os.remove(temp_video)
            except OSError:
                pass
        raise RuntimeError(f"FFmpeg failed ({rc}): {stderr_bytes.decode('utf-8', 'ignore')}")

    os.replace(temp_video, os.path.abspath(output_path))

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

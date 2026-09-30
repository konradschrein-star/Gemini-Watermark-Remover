#!/usr/bin/env python3
"""
CLI for Gemini & Veo Watermark Remover.

Examples:
  # Clean a video with auto-detection and audio preservation
  python cli.py video.mp4 -o video_clean.mp4

  # Clean an image and verify with Jev-Omni QA classifier
  python cli.py image.png --verify

  # Force specific watermark type
  python cli.py video.mp4 --type gemini_sparkle --method telea
"""

import argparse
import json
import os
import sys
from pathlib import Path

from gemini_watermark_remover import (
    process_video,
    process_image,
    WatermarkType,
    InpaintMethod,
)


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def parse_region(region_str: str):
    try:
        parts = [int(p.strip()) for p in region_str.split(",")]
        if len(parts) != 4:
            raise ValueError()
        return tuple(parts)
    except Exception:
        raise argparse.ArgumentTypeError("Region must be formatted as x,y,w,h (e.g. 1700,860,90,90)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gemini & Veo Watermark Remover with optional Jev-Omni QA Verification."
    )
    parser.add_argument("input", help="Path to input video or image file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output file. Defaults to <input_basename>_cleaned.<ext>.",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=[t.value for t in WatermarkType],
        default=WatermarkType.AUTO.value,
        help="Watermark profile to detect/remove (default: auto).",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=[m.value for m in InpaintMethod],
        default=InpaintMethod.TELEA.value,
        help="Inpainting / reconstruction algorithm (default: telea).",
    )
    parser.add_argument(
        "--region",
        type=parse_region,
        default=None,
        help="Manual bounding box x,y,w,h (overrides auto-detection).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run post-removal QA verification (Jev-Omni classifier).",
    )
    parser.add_argument(
        "--verifier",
        choices=["jev-omni", "local_cv"],
        default="jev-omni",
        help="Verification backend engine (default: jev-omni).",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="Path or name of ffmpeg binary.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit result as JSON to stdout (ideal for CLI/API wrappers).",
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: input file not found at {input_path}", file=sys.stderr)
        return 1

    ext = Path(input_path).suffix.lower()
    if not args.output:
        base = Path(input_path).stem
        output_path = os.path.join(os.path.dirname(input_path), f"{base}_cleaned{ext}")
    else:
        output_path = os.path.abspath(args.output)

    w_type = WatermarkType(args.type)
    method = InpaintMethod(args.method)

    if ext in VIDEO_EXTS:
        def progress(done: int, total: int):
            if not args.json:
                pct = (done / max(1, total)) * 100.0
                sys.stderr.write(f"\r[GeminiWatermark] Processing video: {done}/{total} frames ({pct:.1f}%)")
                sys.stderr.flush()

        result = process_video(
            input_path=input_path,
            output_path=output_path,
            watermark_type=w_type,
            method=method,
            manual_region=args.region,
            ffmpeg_binary=args.ffmpeg,
            progress_callback=progress,
            verify=args.verify,
            verifier_backend=args.verifier,
        )
        if not args.json:
            sys.stderr.write("\n")
    elif ext in IMAGE_EXTS:
        result = process_image(
            input_path=input_path,
            output_path=output_path,
            watermark_type=w_type,
            method=method,
            manual_region=args.region,
            verify=args.verify,
            verifier_backend=args.verifier,
        )
    else:
        print(f"Error: unsupported file extension '{ext}'", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[GeminiWatermark] Successfully cleaned: {result['output']}")
        print(f"[GeminiWatermark] Detected profile: {result['watermark_type']}")
        print(f"[GeminiWatermark] Bounding box: {result['bbox']}")
        if result.get("verification"):
            v = result["verification"]
            verdict = "PASSED" if v.get("is_clean") else "FLAGGED"
            conf = v.get("confidence", 0.0) * 100.0
            print(f"[GeminiWatermark] QA Verification ({v.get('backend')}): {verdict} ({conf:.1f}% confidence - '{v.get('label')}')")

    return 0


if __name__ == "__main__":
    sys.exit(main())

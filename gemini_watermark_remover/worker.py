#!/usr/bin/env python3
"""
Persistent Ultra-Lightweight JSON-RPC Stdio Worker for Gemini / Veo Watermark Detection & Removal.
Supports both images and videos (using a single midpoint frame yes/no detector on videos).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_watermark_remover.detector import prewarm_template_cache  # noqa: E402
from gemini_watermark_remover.pipeline import (  # noqa: E402
    detect_and_remove_image_if_present,
    detect_and_remove_video_if_present,
)

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


def _handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    req_id = req.get("id")
    action = req.get("action", "process_media")
    if action == "ping":
        return {"id": req_id, "ok": True, "status": "pong"}

    input_path = str(req.get("input_path") or "").strip()
    output_path = req.get("output_path")
    kind = str(req.get("kind") or "").strip().lower()
    jpeg_quality = int(req.get("jpeg_quality") or 96)

    if not input_path:
        return {"id": req_id, "ok": False, "error": "Missing input_path"}

    ext = os.path.splitext(input_path)[1].lower()
    is_video = (action == "process_video") or (kind == "video") or (ext in VIDEO_EXTS)

    try:
        if is_video:
            result = detect_and_remove_video_if_present(
                input_path=input_path,
                output_path=str(output_path) if output_path else None,
            )
        else:
            result = detect_and_remove_image_if_present(
                input_path=input_path,
                output_path=str(output_path) if output_path else None,
                jpeg_quality=jpeg_quality,
            )
        return {"id": req_id, "ok": result.get("status") != "error", **result}
    except Exception as exc:
        return {"id": req_id, "ok": False, "status": "error", "error": str(exc)}


def main() -> int:
    prewarm_template_cache()
    sys.stdout.write(json.dumps({"ready": True, "pid": os.getpid()}) + "\n")
    sys.stdout.flush()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            sys.stdout.write(json.dumps({"id": None, "ok": False, "error": f"Invalid JSON: {exc}"}) + "\n")
            sys.stdout.flush()
            continue

        resp = _handle_request(payload)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

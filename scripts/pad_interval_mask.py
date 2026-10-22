#!/usr/bin/env python3
"""Pad a tracked interval mask to a full-length, otherwise empty timeline."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--total-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.mask))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open {args.mask}")
    interval_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_frame = args.start_frame + interval_frames
    if not 0 <= args.start_frame < end_frame <= args.total_frames:
        raise ValueError("Tracked interval does not fit inside the target timeline")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
            "-c:v", "ffv1", str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    if encoder.stdin is None:
        raise RuntimeError("Failed to open FFmpeg stdin")

    empty = np.zeros((height, width), dtype=np.uint8)
    nonempty_frames = 0
    for index in range(args.total_frames):
        if args.start_frame <= index < end_frame:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Interval-mask decode failed at frame {index}")
            mask = (np.max(frame, axis=2) > 32).astype(np.uint8) * 255
            nonempty_frames += int(mask.any())
        else:
            mask = empty
        encoder.stdin.write(mask.tobytes())

    capture.release()
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("Mask encoding failed")

    payload = {
        "method": "causal interval mask padded with empty frames outside held state",
        "input": str(args.mask.resolve()),
        "tracked_interval": [args.start_frame, end_frame],
        "total_frames": args.total_frames,
        "nonempty_frames": nonempty_frames,
        "fps": fps,
        "resolution": [width, height],
        "output": str(args.output.resolve()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

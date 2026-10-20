#!/usr/bin/env python3
"""Composite source pixels over generated video using a tracked region mask."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_info(capture: cv2.VideoCapture) -> dict:
    return {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dilate", type=int, default=2)
    parser.add_argument("--feather", type=int, default=3)
    args = parser.parse_args()

    captures = {
        name: cv2.VideoCapture(str(path))
        for name, path in {
            "source": args.source,
            "generated": args.generated,
            "mask": args.mask,
        }.items()
    }
    if not all(capture.isOpened() for capture in captures.values()):
        raise RuntimeError("Failed to open one or more input videos")
    info = {name: video_info(capture) for name, capture in captures.items()}
    expected_frames = info["generated"]["frames"]
    expected_fps = info["generated"]["fps"]
    for name in ("source", "mask"):
        if info[name]["frames"] != expected_frames:
            raise RuntimeError(f"{name} frame mismatch: {info[name]['frames']} != {expected_frames}")
        if abs(info[name]["fps"] - expected_fps) > 0.001:
            raise RuntimeError(f"{name} FPS mismatch: {info[name]['fps']} != {expected_fps}")

    width = info["generated"]["width"]
    height = info["generated"]["height"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(expected_fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "12",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    if encoder.stdin is None:
        raise RuntimeError("Failed to open FFmpeg stdin")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * args.dilate + 1, 2 * args.dilate + 1),
    )
    areas: list[float] = []
    frame_index = 0
    while frame_index < expected_frames:
        decoded = {name: capture.read() for name, capture in captures.items()}
        if not all(ok for ok, _ in decoded.values()):
            raise RuntimeError(f"Decode failure at frame {frame_index}")
        source = decoded["source"][1]
        generated = decoded["generated"][1]
        mask_frame = decoded["mask"][1]
        source = cv2.resize(source, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = np.max(mask_frame, axis=2) > 32
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        if args.dilate:
            mask = cv2.dilate(mask, kernel)
        alpha = mask.astype(np.float32)
        if args.feather:
            size = 2 * args.feather + 1
            alpha = cv2.GaussianBlur(alpha, (size, size), 0)
        alpha = alpha[..., None]
        composed = np.rint(source * alpha + generated * (1.0 - alpha))
        encoder.stdin.write(np.clip(composed, 0, 255).astype(np.uint8).tobytes())
        areas.append(float(mask.mean()))
        frame_index += 1

    encoder.stdin.close()
    return_code = encoder.wait()
    for capture in captures.values():
        capture.release()
    if return_code:
        raise RuntimeError(f"FFmpeg failed with exit code {return_code}")

    manifest = {
        "method": "SAM3.1 gripper mask; source-pixel interior with feathered boundary",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "generated": str(args.generated.resolve()),
        "generated_sha256": sha256(args.generated),
        "mask": str(args.mask.resolve()),
        "mask_sha256": sha256(args.mask),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "inputs": info,
        "frames": frame_index,
        "dilate_pixels_at_output": args.dilate,
        "feather_pixels_at_output": args.feather,
        "mask_area_fraction": {
            "min": min(areas),
            "max": max(areas),
            "mean": float(np.mean(areas)),
            "empty_frames": sum(area == 0 for area in areas),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

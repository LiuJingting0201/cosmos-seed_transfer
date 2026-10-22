#!/usr/bin/env python3
"""Apply deterministic Lab chroma to generated garment pixels."""

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


def parse_hex_color(value: str) -> tuple[int, int, int]:
    value = value.removeprefix("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError("Expected a color in RRGGBB form")
    try:
        red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a color in RRGGBB form") from error
    return blue, green, red


def target_lab(color: tuple[int, int, int]) -> np.ndarray:
    bgr = np.array([[color]], dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--exclude-mask", type=Path)
    parser.add_argument("--target-color", type=parse_hex_color, required=True)
    parser.add_argument("--chroma-strength", type=float, default=0.9)
    parser.add_argument("--feather", type=int, default=4)
    parser.add_argument("--exclude-dilate", type=int, default=5)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--min-area-fraction", type=float, default=0.0002)
    parser.add_argument("--max-area-fraction", type=float, default=0.65)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 <= args.chroma_strength <= 1.0:
        raise ValueError("--chroma-strength must be in [0, 1]")
    captures = {
        name: cv2.VideoCapture(str(path))
        for name, path in {
            "generated": args.generated,
            "mask": args.mask,
            **({"exclude": args.exclude_mask} if args.exclude_mask else {}),
        }.items()
    }
    if not all(capture.isOpened() for capture in captures.values()):
        raise RuntimeError("Failed to open one or more input videos")
    info = {name: video_info(capture) for name, capture in captures.items()}
    expected = info["generated"]
    end_frame = args.end_frame if args.end_frame is not None else expected["frames"]
    if not 0 <= args.start_frame < end_frame <= expected["frames"]:
        raise ValueError("Invalid --start-frame/--end-frame interval")
    for name, media in info.items():
        if media["frames"] != expected["frames"]:
            raise RuntimeError(f"{name} frame mismatch: {media['frames']} != {expected['frames']}")
        if abs(media["fps"] - expected["fps"]) > 0.001:
            raise RuntimeError(f"{name} FPS mismatch: {media['fps']} != {expected['fps']}")

    width, height = expected["width"], expected["height"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(expected["fps"]), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "12",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    if encoder.stdin is None:
        raise RuntimeError("Failed to open FFmpeg stdin")

    target = target_lab(args.target_color)
    exclude_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * args.exclude_dilate + 1, 2 * args.exclude_dilate + 1),
    )
    areas: list[float] = []
    applied: list[bool] = []
    for index in range(expected["frames"]):
        decoded = {name: capture.read() for name, capture in captures.items()}
        if not all(ok for ok, _ in decoded.values()):
            raise RuntimeError(f"Decode failure at frame {index}")
        frame = decoded["generated"][1]
        mask_frame = decoded["mask"][1]
        region = cv2.resize(
            (np.max(mask_frame, axis=2) > 32).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        if "exclude" in decoded:
            exclude_frame = decoded["exclude"][1]
            excluded = cv2.resize(
                (np.max(exclude_frame, axis=2) > 32).astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            region[cv2.dilate(excluded, exclude_kernel) > 0] = 0

        area = float(region.mean())
        in_interval = args.start_frame <= index < end_frame
        valid = (
            in_interval
            and args.min_area_fraction <= area <= args.max_area_fraction
        )
        areas.append(area)
        applied.append(valid)
        if valid:
            alpha = region.astype(np.float32)
            if args.feather:
                size = 2 * args.feather + 1
                alpha = cv2.GaussianBlur(alpha, (size, size), 0)
            alpha = np.clip(alpha * args.chroma_strength, 0.0, 1.0)[..., None]
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            recolored_lab = lab.copy()
            recolored_lab[..., 1] = target[1]
            recolored_lab[..., 2] = target[2]
            recolored = cv2.cvtColor(recolored_lab, cv2.COLOR_LAB2BGR)
            frame = np.rint(frame * (1.0 - alpha) + recolored * alpha).astype(np.uint8)
        encoder.stdin.write(frame.tobytes())

    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("FFmpeg encoding failed")
    for capture in captures.values():
        capture.release()

    target_rgb_hex = "#{:02X}{:02X}{:02X}".format(
        args.target_color[2], args.target_color[1], args.target_color[0]
    )
    payload = {
        "method": "deterministic Lab chroma transfer preserving generated luminance",
        "generated": str(args.generated.resolve()),
        "generated_sha256": sha256(args.generated),
        "mask": str(args.mask.resolve()),
        "mask_sha256": sha256(args.mask),
        "exclude_mask": str(args.exclude_mask.resolve()) if args.exclude_mask else None,
        "target_color": target_rgb_hex,
        "target_lab_opencv": [int(value) for value in target],
        "chroma_strength": args.chroma_strength,
        "feather": args.feather,
        "active_frame_interval": [args.start_frame, end_frame],
        "area_gate": [args.min_area_fraction, args.max_area_fraction],
        "applied_frames": sum(applied),
        "skipped_frames": len(applied) - sum(applied),
        "mask_area_fraction": {
            "min": min(areas),
            "max": max(areas),
            "mean": float(np.mean(areas)),
        },
        "media": expected,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a garment mask from a stable source-color identity anchor."""

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


def hue_distance(hue: np.ndarray, center: int) -> np.ndarray:
    distance = np.abs(hue.astype(np.int16) - center)
    return np.minimum(distance, 180 - distance)


def filter_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    filtered = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 1
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude-mask", type=Path)
    parser.add_argument("--hue-center", type=int, required=True)
    parser.add_argument("--hue-radius", type=int, default=12)
    parser.add_argument("--min-saturation", type=int, default=65)
    parser.add_argument("--min-value", type=int, default=35)
    parser.add_argument("--max-value", type=int, default=255)
    parser.add_argument("--min-component-area", type=int, default=120)
    parser.add_argument("--close-radius", type=int, default=5)
    parser.add_argument("--open-radius", type=int, default=2)
    parser.add_argument("--exclude-dilate", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if not 0 <= args.hue_center < 180:
        raise ValueError("--hue-center must be in [0, 179]")
    source = cv2.VideoCapture(str(args.source))
    exclude = cv2.VideoCapture(str(args.exclude_mask)) if args.exclude_mask else None
    if not source.isOpened() or (exclude and not exclude.isOpened()):
        raise RuntimeError("Failed to open source or exclude mask")

    frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(source.get(cv2.CAP_PROP_FPS))
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if exclude:
        if int(exclude.get(cv2.CAP_PROP_FRAME_COUNT)) < frames:
            raise RuntimeError("Exclude mask has fewer frames than source")
        exclude_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * args.exclude_dilate + 1, 2 * args.exclude_dilate + 1),
        )

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

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * args.close_radius + 1, 2 * args.close_radius + 1)
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * args.open_radius + 1, 2 * args.open_radius + 1)
    )
    areas: list[float] = []
    contact_frames: list[np.ndarray] = []
    contact_indexes = set(np.linspace(0, max(0, frames - 1), 12).astype(int).tolist())

    for index in range(frames):
        ok, frame = source.read()
        if not ok:
            raise RuntimeError(f"Source decode failed at frame {index}")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = (
            (hue_distance(hsv[..., 0], args.hue_center) <= args.hue_radius)
            & (hsv[..., 1] >= args.min_saturation)
            & (hsv[..., 2] >= args.min_value)
            & (hsv[..., 2] <= args.max_value)
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = filter_components(mask, args.min_component_area)

        if exclude:
            exclude_ok, exclude_frame = exclude.read()
            if not exclude_ok:
                raise RuntimeError(f"Exclude-mask decode failed at frame {index}")
            excluded = cv2.resize(
                (np.max(exclude_frame, axis=2) > 32).astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            mask[cv2.dilate(excluded, exclude_kernel) > 0] = 0

        encoder.stdin.write((mask * 255).astype(np.uint8).tobytes())
        areas.append(float(mask.mean()))
        if index in contact_indexes:
            overlay = frame.copy()
            overlay[mask > 0] = np.rint(
                overlay[mask > 0] * 0.4 + np.array([0, 255, 0]) * 0.6
            ).astype(np.uint8)
            cv2.putText(
                overlay, f"frame {index}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            contact_frames.append(cv2.resize(overlay, (320, 240)))

    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("Mask encoding failed")
    source.release()
    if exclude:
        exclude.release()

    while len(contact_frames) < 12:
        contact_frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
    contact_sheet = np.vstack(
        [np.hstack(contact_frames[offset:offset + 4]) for offset in range(0, 12, 4)]
    )
    args.contact.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.contact), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])

    nonempty = [area for area in areas if area > 0]
    payload = {
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "exclude_mask": str(args.exclude_mask.resolve()) if args.exclude_mask else None,
        "method": "HSV source-color identity anchor with morphology and robot exclusion",
        "frames": frames,
        "fps": fps,
        "resolution": [width, height],
        "hue_center": args.hue_center,
        "hue_radius": args.hue_radius,
        "min_saturation": args.min_saturation,
        "value_range": [args.min_value, args.max_value],
        "min_component_area": args.min_component_area,
        "empty_frames": sum(area == 0 for area in areas),
        "area_fraction": {
            "min_nonempty": min(nonempty) if nonempty else 0.0,
            "max": max(areas),
            "mean": float(np.mean(areas)),
        },
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "contact": str(args.contact.resolve()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

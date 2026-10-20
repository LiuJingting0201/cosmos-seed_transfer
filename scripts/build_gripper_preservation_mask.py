#!/usr/bin/env python3
"""Build reviewed gripper/robot preservation masks from source video."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


def largest_arm_components(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    candidates = []
    height = mask.shape[0]
    for label in range(1, count):
        x, y, width, component_height, area = stats[label]
        touches_lower_region = y + component_height >= int(height * 0.72)
        if area >= 150 and touches_lower_region:
            candidates.append((int(area), label))
    output = np.zeros_like(mask, dtype=np.uint8)
    for _, label in sorted(candidates, reverse=True)[:2]:
        output[labels == label] = 1
    return output


def warp_mask(mask: np.ndarray, source_gray: np.ndarray, target_gray: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    flow = cv2.calcOpticalFlowFarneback(
        target_gray, source_gray, None, 0.5, 3, 25, 3, 5, 1.2, 0
    )
    return cv2.remap(
        mask.astype(np.uint8),
        xx + flow[..., 0],
        yy + flow[..., 1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )


def head_masks(frames: list[np.ndarray], sam_frames: list[np.ndarray]) -> tuple[list[np.ndarray], dict]:
    masks = [
        largest_arm_components(np.max(frame, axis=2) > 32)
        for frame in sam_frames
    ]
    valid = [index for index, mask in enumerate(masks) if mask.any()]
    if not valid:
        raise RuntimeError("SAM produced no valid robot-arm masks")
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    first = valid[0]
    for index in range(first - 1, -1, -1):
        masks[index] = warp_mask(masks[index + 1], gray[index + 1], gray[index])
    for index in range(first + 1, len(masks)):
        if not masks[index].any():
            masks[index] = warp_mask(masks[index - 1], gray[index - 1], gray[index])
    return masks, {
        "method": "SAM3.1 robotic-arm masks; keep two largest lower-connected components; optical-flow gap fill",
        "raw_valid_frames": len(valid),
        "raw_empty_frames": len(masks) - len(valid),
    }


def write_mask_video(path: Path, masks: list[np.ndarray], fps: float) -> None:
    height, width = masks[0].shape
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
        "-c:v", "ffv1", str(path),
    ], stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Failed to open FFmpeg stdin")
    for mask in masks:
        process.stdin.write((mask * 255).astype(np.uint8).tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError("FFmpeg mask encoding failed")


def write_contact(path: Path, frames: list[np.ndarray], masks: list[np.ndarray]) -> None:
    indexes = sorted(set(range(0, len(frames), 30)) | {len(frames) - 1})
    tiles = []
    for index in indexes:
        frame = frames[index].copy()
        overlay = frame.copy()
        overlay[masks[index] > 0] = (
            overlay[masks[index] > 0] * 0.45 + np.array([0, 255, 0]) * 0.55
        ).astype(np.uint8)
        tile = np.hstack([
            cv2.resize(frame, (320, 240)),
            cv2.resize(overlay, (320, 240)),
        ])
        cv2.putText(
            tile, f"frame {index}: source | preserved mask", (6, 232),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )
        tiles.append(tile)
    rows = []
    for offset in range(0, len(tiles), 2):
        row = tiles[offset:offset + 2]
        if len(row) == 1:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=("head",), required=True)
    parser.add_argument("--sam-control", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    frames, fps = read_frames(args.source)
    if not args.sam_control:
        raise SystemExit("--sam-control is required for head mode")
    sam_frames, sam_fps = read_frames(args.sam_control)
    if len(sam_frames) != len(frames) or abs(sam_fps - fps) > 0.001:
        raise RuntimeError("SAM control does not match source video")
    masks, details = head_masks(frames, sam_frames)

    write_mask_video(args.output, masks, fps)
    write_contact(args.contact, frames, masks)
    payload = {
        "source": str(args.source.resolve()),
        "mode": args.mode,
        "frames": len(frames),
        "fps": fps,
        "width": frames[0].shape[1],
        "height": frames[0].shape[0],
        "empty_output_frames": sum(not mask.any() for mask in masks),
        "mean_area_fraction": float(np.mean([mask.mean() for mask in masks])),
        **details,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

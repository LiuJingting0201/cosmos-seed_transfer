#!/usr/bin/env python3
"""Apply a temporally feathered color treatment inside a tracked video mask."""

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


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--region-source", type=Path)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--exclude-mask", type=Path)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--active-frames", type=int)
    parser.add_argument("--fade-frames", type=int, default=5)
    parser.add_argument("--hue", type=int, default=24)
    parser.add_argument("--saturation", type=int, default=180)
    parser.add_argument("--value-offset", type=float, default=70.0)
    parser.add_argument("--value-scale", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    base = cv2.VideoCapture(str(args.base))
    region_source = cv2.VideoCapture(str(args.region_source)) if args.region_source else None
    mask = cv2.VideoCapture(str(args.mask))
    exclude = cv2.VideoCapture(str(args.exclude_mask)) if args.exclude_mask else None
    if (
        not base.isOpened() or not mask.isOpened()
        or (exclude and not exclude.isOpened())
        or (region_source and not region_source.isOpened())
    ):
        raise RuntimeError("Failed to open one or more input videos")
    frames = int(base.get(cv2.CAP_PROP_FRAME_COUNT))
    mask_frames = int(mask.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(base.get(cv2.CAP_PROP_FPS))
    width = int(base.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(base.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.start_frame < 0 or args.start_frame + mask_frames > frames:
        raise RuntimeError("Tracked mask range is outside the base video")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output),
    ], stdin=subprocess.PIPE)
    assert encoder.stdin is not None

    coverage = []
    for frame_index in range(frames):
        ok, frame = base.read()
        if not ok:
            raise RuntimeError(f"Base decode failed at frame {frame_index}")
        if region_source:
            source_ok, source_frame = region_source.read()
            if not source_ok:
                raise RuntimeError(f"Region-source decode failed at frame {frame_index}")
            source_frame = cv2.resize(
                source_frame, (width, height), interpolation=cv2.INTER_LINEAR
            )
        else:
            source_frame = frame
        if exclude:
            exclude_ok, exclude_frame = exclude.read()
            if not exclude_ok:
                raise RuntimeError(f"Exclude-mask decode failed at frame {frame_index}")
        else:
            exclude_frame = None

        local_index = frame_index - args.start_frame
        if 0 <= local_index < mask_frames:
            mask_ok, mask_frame = mask.read()
            if not mask_ok:
                raise RuntimeError(f"Tracked-mask decode failed at local frame {local_index}")
            region = cv2.resize(
                (np.max(mask_frame, axis=2) > 32).astype(np.uint8),
                (width, height), interpolation=cv2.INTER_NEAREST,
            )
            if exclude_frame is not None:
                excluded = cv2.resize(
                    (np.max(exclude_frame, axis=2) > 32).astype(np.uint8),
                    (width, height), interpolation=cv2.INTER_NEAREST,
                )
                excluded = cv2.dilate(
                    excluded,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                )
                region[excluded > 0] = 0

            active = args.active_frames is None or local_index < args.active_frames
            fade_in = smoothstep((local_index + 1) / args.fade_frames)
            fade_out = (
                smoothstep((mask_frames - local_index) / args.fade_frames)
                if args.active_frames is None else 1.0
            )
            alpha = cv2.GaussianBlur(region.astype(np.float32), (9, 9), 0)
            temporal_alpha = min(fade_in, fade_out) if active else 0.0
            alpha = np.clip(alpha * temporal_alpha, 0.0, 1.0)[..., None]

            hsv = cv2.cvtColor(source_frame, cv2.COLOR_BGR2HSV)
            target_hsv = hsv.copy()
            target_hsv[..., 0] = args.hue
            target_hsv[..., 1] = np.maximum(target_hsv[..., 1], args.saturation)
            value = target_hsv[..., 2].astype(np.float32)
            target_hsv[..., 2] = np.clip(
                args.value_offset + args.value_scale * value, 0, 255
            ).astype(np.uint8)
            target = cv2.cvtColor(target_hsv, cv2.COLOR_HSV2BGR)
            frame = np.rint(frame * (1.0 - alpha) + target * alpha).astype(np.uint8)
            coverage.append(float(region.mean()))
        encoder.stdin.write(frame.tobytes())

    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("FFmpeg encoding failed")
    base.release()
    if region_source:
        region_source.release()
    mask.release()
    if exclude:
        exclude.release()

    payload = {
        "base": str(args.base.resolve()),
        "base_sha256": sha256(args.base),
        "region_source": str(args.region_source.resolve()) if args.region_source else None,
        "region_source_sha256": sha256(args.region_source) if args.region_source else None,
        "mask": str(args.mask.resolve()),
        "mask_sha256": sha256(args.mask),
        "exclude_mask": str(args.exclude_mask.resolve()) if args.exclude_mask else None,
        "start_frame": args.start_frame,
        "active_frames": args.active_frames,
        "mask_frames": mask_frames,
        "fade_frames": args.fade_frames,
        "hue": args.hue,
        "saturation": args.saturation,
        "value_offset": args.value_offset,
        "value_scale": args.value_scale,
        "mean_mask_area_fraction": float(np.mean(coverage)),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "frames": frames,
        "fps": fps,
        "resolution": [width, height],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

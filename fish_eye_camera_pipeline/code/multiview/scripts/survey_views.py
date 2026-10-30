#!/usr/bin/env python3
"""Create streaming, one-frame-per-interval contact sheets for per-view videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import imageio.v3 as iio
import numpy as np


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def discover(root: Path, selected: set[str]) -> list[tuple[str, Path]]:
    direct = sorted(p for p in root.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if direct:
        scene = root.name
        return [] if selected and scene not in selected else [(scene, p) for p in direct]
    found: list[tuple[str, Path]] = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if selected and scene_dir.name not in selected:
            continue
        found.extend(
            (scene_dir.name, p)
            for p in sorted(scene_dir.iterdir())
            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
        )
    return found


def decode_samples(path: Path, interval_s: float, scale: float) -> tuple[dict, list[np.ndarray]]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        step = max(1, round(fps * interval_s))
        samples: list[np.ndarray] = []
        frames = 0
        sampled_indices = []
        last_frame = None
        for index, frame in enumerate(container.decode(stream)):
            frames += 1
            last_frame = frame
            if index % step == 0:
                image = frame.to_ndarray(format="rgb24")
                if scale < 1:
                    stride = max(1, round(1 / scale))
                    image = image[::stride, ::stride]
                samples.append(image)
                sampled_indices.append(index)
        if last_frame is not None and sampled_indices[-1] != frames - 1:
            image = last_frame.to_ndarray(format="rgb24")
            if scale < 1:
                image = image[::max(1, round(1 / scale)), ::max(1, round(1 / scale))]
            samples.append(image)
            sampled_indices.append(frames - 1)
        if not samples:
            raise ValueError(f"No decodable frames: {path}")
        return ({
            "source": str(path.resolve()),
            "frames": frames,
            "width": stream.width,
            "height": stream.height,
            "fps": fps,
            "duration_s": frames / fps,
            "sampled_frames": sampled_indices,
            "last_frame_time_s": (frames - 1) / fps,
            "last_frame_interval_end_s": frames / fps,
        }, samples)


def make_sheet(samples: list[np.ndarray], cols: int) -> np.ndarray:
    blank = np.zeros_like(samples[0])
    rows = []
    for start in range(0, len(samples), cols):
        row = samples[start:start + cols]
        row += [blank] * (cols - len(row))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()
    if args.interval_seconds <= 0 or not 0 < args.scale <= 1:
        parser.error("interval must be > 0 and scale must be in (0, 1]")

    args.out.mkdir(parents=True, exist_ok=True)
    videos = discover(args.views_root.resolve(), set(args.scene))
    if not videos:
        raise SystemExit(f"No source videos found under {args.views_root}")

    metadata: dict[str, dict] = {}
    for scene, video in videos:
        info, samples = decode_samples(video, args.interval_seconds, args.scale)
        scene_out = args.out / scene
        scene_out.mkdir(parents=True, exist_ok=True)
        iio.imwrite(scene_out / f"{video.stem}.jpg", make_sheet(samples, args.columns), quality=92)
        metadata[f"{scene}/{video.stem}"] = info
        print(
            f"{scene}/{video.stem}: {info['frames']} frames, "
            f"{info['width']}x{info['height']} @ {info['fps']:.3f} fps, "
            f"{len(samples)} samples",
            flush=True,
        )

    (args.out / "view_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

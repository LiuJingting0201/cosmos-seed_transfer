#!/usr/bin/env python3
"""Create fixed-count, uniformly sampled contact sheets for visual audit."""

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
    found = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if selected and scene_dir.name not in selected:
            continue
        found.extend(
            (scene_dir.name, path)
            for path in sorted(scene_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
    return found


def decode_uniform(path: Path, sample_count: int, scale: float) -> tuple[dict, list[np.ndarray]]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frame_count = stream.frames
        if not frame_count:
            frame_count = sum(1 for _ in container.decode(stream))

    indices = [
        int(index)
        for index in sorted(set(np.linspace(
            0, max(0, frame_count - 1), sample_count, dtype=int
        )))
    ]
    wanted = set(indices)
    samples = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index not in wanted:
                continue
            image = frame.to_ndarray(format="rgb24")
            if scale < 1:
                stride = max(1, round(1 / scale))
                image = image[::stride, ::stride]
            samples[index] = image
    if len(samples) != len(indices):
        raise ValueError(f"Could not decode all audit samples from {path}")
    return ({
        "source": str(path.resolve()),
        "frames": frame_count,
        "width": stream.width,
        "height": stream.height,
        "fps": fps,
        "duration_s": frame_count / fps,
        "sampled_frames": indices,
        "sample_count": len(indices),
    }, [samples[index] for index in indices])


def make_sheet(samples: list[np.ndarray], columns: int) -> np.ndarray:
    blank = np.zeros_like(samples[0])
    rows = []
    for start in range(0, len(samples), columns):
        row = samples[start:start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--sample-count", type=int, default=9)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()
    if args.sample_count < 3 or args.columns < 1 or not 0 < args.scale <= 1:
        parser.error("sample-count >= 3, columns >= 1, and scale in (0, 1] are required")

    videos = discover(args.views_root.resolve(), set(args.scene))
    if not videos:
        raise SystemExit(f"No source videos found under {args.views_root}")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for scene, video in videos:
        info, samples = decode_uniform(video, args.sample_count, args.scale)
        scene_out = args.out / scene
        scene_out.mkdir(parents=True, exist_ok=True)
        iio.imwrite(scene_out / f"{video.stem}.jpg", make_sheet(samples, args.columns), quality=92)
        metadata[f"{scene}/{video.stem}"] = info
        print(f"{scene}/{video.stem}: {len(samples)} uniform audit samples", flush=True)
    (args.out / "view_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

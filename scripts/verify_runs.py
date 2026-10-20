#!/usr/bin/env python3
"""Streaming technical verification plus start/middle/end review sheets."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import av
import imageio.v3 as iio
import numpy as np


def inspect(path: Path, picks: set[int] | None = None) -> tuple[dict, dict[int, np.ndarray]]:
    kept: dict[int, np.ndarray] = {}
    tail: deque[np.ndarray] = deque(maxlen=10)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames = 0
        for index, frame in enumerate(container.decode(stream)):
            frames += 1
            image = frame.to_ndarray(format="rgb24")
            tail.append(image[::16, ::16])
            if picks and index in picks:
                kept[index] = image
        diffs = [
            float(np.mean(np.abs(b.astype(np.int16) - a.astype(np.int16))))
            for a, b in zip(tail, list(tail)[1:])
        ]
        return ({
            "frames": frames,
            "width": stream.width,
            "height": stream.height,
            "fps": fps,
            "duration_s": frames / fps,
            "tail_mean_absdiff": round(float(np.mean(diffs)), 4) if diffs else 0.0,
            "tail_frozen": bool(diffs and max(diffs) < 0.01),
        }, kept)


def source_path(root: Path, scene: str, view: str) -> Path:
    bases = [root / scene, root] if root.name == scene else [root / scene]
    for base in bases:
        for suffix in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
            candidate = base / f"{view}{suffix}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"source missing for {scene}/{view}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--views-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--control", choices=("edge", "depth", "blur"), default="edge")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    scene_rows: dict[str, list[tuple[str, np.ndarray]]] = {}
    for run_dir in sorted(p for p in args.run_root.iterdir() if p.is_dir()):
        scene, sep, view = run_dir.name.partition("__")
        if not sep:
            continue
        entry = {"scene": scene, "view": view, "ok": False}
        try:
            source = source_path(args.views_root, scene, view)
            output = run_dir / "vision.mp4"
            control = run_dir / f"control_{args.control}.mp4"
            source_info, _ = inspect(source)
            picks = {0, source_info["frames"] // 2, source_info["frames"] - 1}
            source_info, source_frames = inspect(source, picks)
            output_info, output_frames = inspect(output, picks)
            control_info, _ = inspect(control)
            entry.update(source=source_info, output=output_info, control=control_info)
            entry["ok"] = (
                source_info["frames"] == output_info["frames"] == control_info["frames"]
                and abs(source_info["fps"] - output_info["fps"]) < 0.01
                and output_info["width"] == source_info["width"]
                and output_info["height"] == source_info["height"]
                and control_info["width"] == source_info["width"]
                and control_info["height"] == source_info["height"]
                and not output_info["tail_frozen"]
            )
            top = np.concatenate([source_frames[i][::2, ::2] for i in sorted(picks)], axis=1)
            bottom = np.concatenate([output_frames[i][::2, ::2] for i in sorted(picks)], axis=1)
            scene_rows.setdefault(scene, []).append((view, np.concatenate([top, bottom], axis=0)))
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)
        report[run_dir.name] = entry

    for scene, rows in sorted(scene_rows.items()):
        images = [image for _, image in sorted(rows)]
        width = min(image.shape[1] for image in images)
        iio.imwrite(args.out / f"{scene}.jpg", np.concatenate([x[:, :width] for x in images]), quality=90)
    (args.out / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    passed = sum(bool(entry["ok"]) for entry in report.values())
    print(f"verified {passed}/{len(report)} views ok")
    for name, entry in report.items():
        if not entry["ok"]:
            print(f"FAIL {name}: {entry.get('error', entry)}")
    raise SystemExit(0 if report and passed == len(report) else 1)


if __name__ == "__main__":
    main()

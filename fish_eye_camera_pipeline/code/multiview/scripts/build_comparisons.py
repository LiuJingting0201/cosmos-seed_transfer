#!/usr/bin/env python3
"""Build per-scene source-top / Transfer-bottom review videos with ffmpeg."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def scenes(root: Path) -> list[tuple[str, list[Path]]]:
    direct = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    if direct:
        return [(root.name, direct)]
    return [
        (d.name, sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES))
        for d in sorted(p for p in root.iterdir() if p.is_dir())
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views-root", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cell-width", type=int, default=552)
    ap.add_argument("--cell-height", type=int, default=408)
    ap.add_argument("--frames", type=int, help="optional output frame limit")
    ap.add_argument("--fps", type=float, help="optional output frame rate override")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--font", type=Path, help="optional font file for source/Transfer labels")
    args = ap.parse_args()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("ffmpeg and ffprobe are required for comparison videos")
    args.out.mkdir(parents=True, exist_ok=True)
    selected = set(args.scene)
    font = args.font.resolve() if args.font else None
    if font and not font.is_file():
        raise SystemExit(f"Font file does not exist: {font}")

    for scene, sources in scenes(args.views_root):
        if selected and scene not in selected:
            continue
        if not sources:
            continue
        inputs: list[str] = []
        filters: list[str] = []
        source_labels: list[str] = []
        transfer_labels: list[str] = []
        for index, source in enumerate(sources):
            transfer = args.run_root / f"{scene}__{source.stem}" / "vision.mp4"
            if not transfer.exists():
                raise SystemExit(f"Missing Transfer output: {transfer}")
            inputs.extend(["-i", str(source), "-i", str(transfer)])
            label_filter = ""
            transfer_filter = ""
            if font:
                common = (
                    f",drawbox=x=0:y=0:w=iw:h=34:color=black@0.65:t=fill,"
                    f"drawtext=fontfile={font}:fontcolor=white:fontsize=22:x=10:y=6:text="
                )
                label_filter = common + f"'SOURCE | {source.stem}'"
                transfer_filter = common + f"'TRANSFER | {source.stem}'"
            filters.append(
                f"[{2 * index}:v]scale={args.cell_width}:{args.cell_height}:flags=lanczos,setsar=1"
                f"{label_filter}[s{index}]"
            )
            filters.append(
                f"[{2 * index + 1}:v]scale={args.cell_width}:{args.cell_height}:flags=lanczos,setsar=1"
                f"{transfer_filter}[t{index}]"
            )
            source_labels.append(f"[s{index}]")
            transfer_labels.append(f"[t{index}]")
        count = len(sources)
        filters.extend([
            f"{''.join(source_labels)}hstack=inputs={count}[top]",
            f"{''.join(transfer_labels)}hstack=inputs={count}[bottom]",
            "[top][bottom]vstack=inputs=2[outv]",
        ])
        final = args.out / f"{scene}__source_top_transfer_bottom.mp4"
        temporary = args.out / f".{scene}.{os.getpid()}.tmp.mp4"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
            "-filter_complex", ";".join(filters), "-map", "[outv]", "-an",
        ]
        if args.frames is not None:
            command.extend(["-frames:v", str(args.frames)])
        if args.fps is not None:
            command.extend(["-r", str(args.fps)])
        command.extend([
            "-shortest", "-c:v", "libx264",
            "-preset", "fast", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(temporary),
        ])
        subprocess.run(command, check=True)
        temporary.replace(final)
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of", "csv=p=0", str(final),
        ], text=True, capture_output=True, check=True).stdout.strip()
        print(f"DONE {scene}: {probe}", flush=True)


if __name__ == "__main__":
    main()

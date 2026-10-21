#!/usr/bin/env python3
"""Build a synchronized source/P1/P2/P3 review matrix for the prompt ablation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


VARIANTS = [
    "P1_source_color_anchor_compact",
    "P2_source_position_trajectory_redundant",
    "P3_identical_prompt_all_views",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def draw_label(font: Path | None, text: str) -> str:
    if not font:
        return ""
    escaped = text.replace(":", r"\:").replace("'", r"\'")
    return (
        ",drawbox=x=0:y=0:w=iw:h=30:color=black@0.7:t=fill,"
        f"drawtext=fontfile='{font}':fontcolor=white:fontsize=18:x=8:y=5:text='{escaped}'"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cell-width", type=int, default=414)
    parser.add_argument("--cell-height", type=int, default=306)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("ffmpeg and ffprobe are required")
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    font = args.font.resolve() if args.font else None
    if font and not font.is_file():
        raise SystemExit(f"Font does not exist: {font}")

    manifest = read_json(root / "manifest.json")
    views = sorted(manifest["variants"][VARIANTS[0]]["views"])
    inputs: list[str] = []
    filters: list[str] = []
    rows: list[str] = []
    row_names = ["SOURCE", "P1 compact", "P2 redundant", "P3 identical"]

    for row_index, row_name in enumerate(row_names):
        cells = []
        for column_index, view in enumerate(views):
            if row_index == 0:
                path = Path(manifest["variants"][VARIANTS[0]]["views"][view]["clip"])
            else:
                variant = VARIANTS[row_index - 1]
                spec = read_json(Path(manifest["variants"][variant]["views"][view]["spec"]))
                path = root / "runs" / variant / spec["name"] / "vision.mp4"
            if not path.is_file():
                raise SystemExit(f"Missing review input: {path}")
            input_index = len(inputs) // 2
            inputs.extend(["-i", str(path)])
            label = f"{row_name} | {view.replace('__file-000', '')}"
            output_label = f"c{row_index}_{column_index}"
            filters.append(
                f"[{input_index}:v]scale={args.cell_width}:{args.cell_height}:flags=lanczos,"
                f"setsar=1{draw_label(font, label)}[{output_label}]"
            )
            cells.append(f"[{output_label}]")
        row_label = f"row{row_index}"
        filters.append(f"{''.join(cells)}hstack=inputs={len(cells)}[{row_label}]")
        rows.append(f"[{row_label}]")
    filters.append(f"{''.join(rows)}vstack=inputs={len(rows)}[matrix]")

    review_video = out / "source_p1_p2_p3_matrix.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
            "-filter_complex", ";".join(filters), "-map", "[matrix]", "-an",
            "-frames:v", "121", "-r", "10", "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(review_video),
        ],
        check=True,
    )

    frames = {}
    for second in (3, 5, 7, 9, 11):
        frame_path = out / f"matrix_{second:02d}s.jpg"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(second), "-i", str(review_video), "-frames:v", "1",
                "-q:v", "2", str(frame_path),
            ],
            check=True,
        )
        frames[str(second)] = str(frame_path)

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of", "json", str(review_video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = {
        "review_video": str(review_video),
        "probe": json.loads(probe.stdout)["streams"][0],
        "frames": frames,
        "row_order": row_names,
        "view_order": views,
    }
    (out / "review_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

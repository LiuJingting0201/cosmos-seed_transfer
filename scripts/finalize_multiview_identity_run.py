#!/usr/bin/env python3
"""Wait for a four-view identity run, preserve robot pixels, and build comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2


VIEWS = (
    "head_stereo_left__file-000",
    "head_stereo_right__file-000",
    "wrist_left__file-000",
    "wrist_right__file-000",
)


def run(command: list[str]) -> None:
    print("+ " + " ".join(command[:3]), flush=True)
    subprocess.run(command, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open {path}")
    result = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-sec", type=int, default=32400)
    args = parser.parse_args()

    started = time.time()
    generated = {
        view: (
            args.experiment / "runs" / f"seed_{args.seed}"
            / f"{args.scene}__{view}__identity_palette_v1" / "vision.mp4"
        )
        for view in VIEWS
    }
    while True:
        ready = [view for view, path in generated.items() if path.is_file() and path.stat().st_size]
        print(f"ready={len(ready)}/4 {ready}", flush=True)
        if len(ready) == len(VIEWS):
            break
        if time.time() - started > args.timeout_sec:
            raise TimeoutError(f"Timed out waiting for generated videos: {sorted(set(VIEWS) - set(ready))}")
        time.sleep(60)

    final_dir = args.root / "final_robot_preserved"
    manifest_dir = args.root / "manifests"
    final_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    composite = args.pipeline_root / "scripts" / "composite_preserved_region.py"
    results = {}
    for view in VIEWS:
        source = args.source_root / f"{view}.mp4"
        mask = args.mask_root / f"final_{view}.mkv"
        output = final_dir / f"{view}.mp4"
        manifest = manifest_dir / f"{view}.json"
        run([
            "python3", str(composite),
            "--source", str(source),
            "--generated", str(generated[view]),
            "--mask", str(mask),
            "--output", str(output),
            "--manifest", str(manifest),
        ])
        media = probe(output)
        if media["frames"] != 300 or abs(media["fps"] - 10.0) > 0.001:
            raise RuntimeError(f"Unexpected final media metadata for {view}: {media}")
        results[view] = {
            "generated": str(generated[view]),
            "generated_sha256": sha256(generated[view]),
            "final": str(output),
            "final_sha256": sha256(output),
            "robot_mask": str(mask),
            "robot_mask_sha256": sha256(mask),
            "media": media,
        }

    archive = args.root / "archive" / args.scene / f"seed_{args.seed}"
    archive.mkdir(parents=True, exist_ok=True)
    final_inputs = []
    source_inputs = []
    for view in VIEWS:
        final_inputs.extend(["-i", str(final_dir / f"{view}.mp4")])
        source_inputs.extend(["-i", str(args.source_root / f"{view}.mp4")])
    filter_graph = (
        "[0:v]scale=544:416[a];[1:v]scale=544:416[b];"
        "[2:v]scale=544:416[c];[3:v]scale=544:416[d];"
        "[a][b][c][d]hstack=inputs=4[v]"
    )
    final_horizontal = archive / "identity_palette_robot_preserved.mp4"
    source_horizontal = archive / "source_horizontal.mp4"
    run([
        "ffmpeg", "-y", "-v", "error", *final_inputs,
        "-filter_complex", filter_graph, "-map", "[v]", "-an",
        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
        str(final_horizontal),
    ])
    run([
        "ffmpeg", "-y", "-v", "error", *source_inputs,
        "-filter_complex", filter_graph, "-map", "[v]", "-an",
        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
        str(source_horizontal),
    ])
    comparison = archive / "source_vs_identity_palette.mp4"
    run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(source_horizontal), "-i", str(final_horizontal),
        "-filter_complex", "[0:v][1:v]vstack=inputs=2[v]", "-map", "[v]",
        "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
        str(comparison),
    ])
    contact = archive / "identity_palette_contact.jpg"
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(final_horizontal),
        "-vf", "select='not(mod(n,30))',tile=2x5", "-frames:v", "1", str(contact),
    ])
    payload = {
        "status": "completed",
        "scene": args.scene,
        "seed": args.seed,
        "method": (
            "Independent four-view Edge+Depth generation with a byte-identical three-garment "
            "identity registry; SAM masks preserve only robot and gripper source pixels."
        ),
        "garment_edit_postprocess": False,
        "views": results,
        "final_horizontal": str(final_horizontal),
        "source_vs_final": str(comparison),
        "contact_sheet": str(contact),
    }
    (archive / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

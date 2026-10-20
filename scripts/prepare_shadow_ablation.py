#!/usr/bin/env python3
"""Prepare the reproducible 8-second Dex1 gripper-shadow ablation."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import av
import cv2
import numpy as np


VARIANTS = {
    "A_baseline_edge_medium": {"prompt": "baseline", "edge": "medium"},
    "B_shadow_prompt_negative_edge_medium": {"prompt": "shadow", "edge": "medium"},
    "C_shadow_prompt_negative_edge_high": {"prompt": "shadow", "edge": "high"},
    "D_shadow_prompt_negative_cleaned_edge": {"prompt": "shadow", "edge": "cleaned"},
    "E_alt_shadow_prompt_edge_depth": {"prompt": "shadow", "edge": "medium", "depth": True},
}

SHADOW_NEGATIVE = (
    "solidified shadows, shadows converted into objects, phantom grippers, "
    "duplicated end effectors, mechanical structures emerging from fabric, "
    "cast shadows with physical volume"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def trim_video(source: Path, output: Path, frame_limit: int) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source)) as source_container:
        source_stream = source_container.streams.video[0]
        fps = float(source_stream.average_rate)
        width = source_stream.width
        height = source_stream.height
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-vf", f"trim=end_frame={frame_limit},setpts=N/({fps}*TB)",
            "-an", "-r", str(fps), "-c:v", "libx264", "-crf", "10",
            "-preset", "slow", "-pix_fmt", "yuv420p", str(output),
        ],
        check=True,
    )
    with av.open(str(output)) as output_container:
        output_stream = output_container.streams.video[0]
        count = sum(1 for _ in output_container.decode(output_stream))
        output_fps = float(output_stream.average_rate)
    if count != frame_limit or abs(output_fps - fps) > 0.001:
        raise RuntimeError(
            f"Expected {frame_limit} frames at {fps} FPS, got {count} at {output_fps} FPS"
        )
    return {
        "frames": count,
        "fps": output_fps,
        "width": width,
        "height": height,
    }


def shadow_prompt(source: dict) -> dict:
    prompt = copy.deepcopy(source)
    subject = prompt["subjects"][0]
    subject["description"] = "the single physical robot angular two-finger gripper"
    subject["appearance_details"] = (
        "Exactly one physical black angular two-finger gripper with a white wrist mounting "
        "plate is present. Its cast projection on fabric is only a flat lighting shadow."
    )
    subject["state_changes"] = (
        "The one physical gripper opens and closes; its material remains black. From 0:01 "
        "to 0:04, any gripper-shaped dark region on fabric remains a flat cast shadow with "
        "no volume, hardware, joints, depth, or independent motion."
    )
    prompt["lighting"]["shadows"] = (
        "Soft, flat cast shadows remain illumination on surfaces. A gripper-shaped region "
        "on the garment from 0:01 to 0:04 is not an object and must never become a second "
        "gripper, robot part, joint, or mechanical structure."
    )
    prompt["temporal_caption"] = (
        "Identity constraint: exactly one physical gripper is visible. During 0:01-0:04 its "
        "gripper-shaped cast shadow lies flat on the garment and has no physical volume or "
        "independent motion. Never create a second gripper or robot component. "
        + prompt["temporal_caption"]
    )
    return prompt


def build_cleaned_edge(source: Path, output: Path, frame_limit: int, fps: float) -> dict:
    capture = cv2.VideoCapture(str(source))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"FFV1"),
        fps,
        (width, height),
        isColor=False,
    )
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    removed = []
    count = 0
    while count < frame_limit:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red = (
            cv2.inRange(hsv, (0, 35, 35), (18, 255, 255))
            | cv2.inRange(hsv, (165, 35, 35), (179, 255, 255))
        )
        garment = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel_close)
        interior = cv2.erode(garment, kernel_erode)
        # The real gripper enters from the lower image boundary. Preserve that region.
        interior[int(height * 0.64) :, :] = 0
        cleaned = edges.copy()
        before = int(np.count_nonzero(cleaned))
        cleaned[interior > 0] = 0
        removed.append(before - int(np.count_nonzero(cleaned)))
        writer.write(cleaned)
        count += 1
    capture.release()
    writer.release()
    if count != frame_limit:
        raise RuntimeError(f"Expected {frame_limit} edge frames, wrote {count}")
    return {
        "method": "Canny 100/200; suppress edges inside eroded red-garment mask; preserve lower 36%",
        "frames": count,
        "fps": fps,
        "mean_removed_edge_pixels": round(float(np.mean(removed)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--source-prompt", type=Path, required=True)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--source-negative", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=80)
    args = parser.parse_args()

    root = args.output_root.resolve()
    clip = root / "inputs" / "wrist_left_first_8s_80f.mp4"
    cleaned_edge = root / "controls" / "wrist_left_cleaned_edge_80f.mkv"
    media = trim_video(args.source_video, clip, args.frames)
    cleaned = build_cleaned_edge(clip, cleaned_edge, args.frames, media["fps"])

    baseline_prompt = read_json(args.source_prompt)
    corrected_prompt = shadow_prompt(baseline_prompt)
    write_json(root / "prompts" / "baseline_prompt.json", baseline_prompt)
    write_json(root / "prompts" / "shadow_aware_prompt.json", corrected_prompt)

    baseline_negative = read_json(args.source_negative)
    corrected_negative = baseline_negative.rstrip(".") + ", " + SHADOW_NEGATIVE + "."
    write_json(root / "negative" / "baseline_negative.json", baseline_negative)
    write_json(root / "negative" / "shadow_aware_negative.json", corrected_negative)

    source_spec = read_json(args.source_spec)
    manifest = {
        "scope": "Dex1 wrist_left first 8 seconds, seed 2026",
        "source_video": str(args.source_video.resolve()),
        "media": media,
        "cleaned_edge": cleaned,
        "blocked": {
            "E_shadow_prompt_negative_edge_depth_seg": (
                "No reviewed segmentation control exists and the production gateway cannot "
                "materialize segmentation. E_alt tests edge+depth instead."
            )
        },
        "variants": {},
    }
    specs_dir = root / "specs"
    for name, variant in VARIANTS.items():
        spec = copy.deepcopy(source_spec)
        spec["name"] = name
        spec["vision_path"] = str(clip)
        spec["num_frames"] = args.frames
        spec["max_frames"] = args.frames
        spec["num_video_frames_per_chunk"] = args.frames
        shadow_aware = variant["prompt"] == "shadow"
        spec["prompt_path"] = str(
            root / "prompts" / ("shadow_aware_prompt.json" if shadow_aware else "baseline_prompt.json")
        )
        spec["negative_prompt_file"] = str(
            root / "negative" / ("shadow_aware_negative.json" if shadow_aware else "baseline_negative.json")
        )
        spec["edge"] = {
            "weight": 0.6 if variant.get("depth") else 1.0,
            "preset_edge_threshold": (
                "medium" if variant["edge"] == "cleaned" else variant["edge"]
            ),
        }
        if variant["edge"] == "cleaned":
            spec["edge"]["control_path"] = str(cleaned_edge)
        if variant.get("depth"):
            spec["depth"] = {"weight": 0.4, "auto_generate": True}
        else:
            spec.pop("depth", None)
        write_json(specs_dir / name / f"{name}.json", spec)
        manifest["variants"][name] = {
            "prompt": variant["prompt"],
            "negative": "shadow-aware" if shadow_aware else "baseline",
            "control": "edge+depth" if variant.get("depth") else f"edge:{variant['edge']}",
            "spec": str(specs_dir / name / f"{name}.json"),
        }
    write_json(root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

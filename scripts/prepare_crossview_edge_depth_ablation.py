#!/usr/bin/env python3
"""Prepare a four-view Edge+Depth run with operation-bound garment identities."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path


GARMENT_IDENTITIES = (
    "Garment identity table, shared by all synchronized cameras: garment A is the first "
    "garment picked up and is golden yellow (#D8A928); garment B is the second garment "
    "picked up and is cobalt blue (#2459C4); garment C is the third garment picked up and "
    "is emerald green (#16845B). These are three different persistent physical objects. "
    "The currently handled garment and every still-visible garment must retain its own "
    "identity and color before grasp, while held, during transport, after release, after "
    "occlusion, and when seen from another camera. Never transfer the active garment's "
    "color to either of the other garments."
)

SHARED_APPEARANCE = (
    "Cross-view canonical appearance is defined by the two synchronized head cameras. "
    "All four cameras observe the same physical robot, garments, basket, washer, and room "
    "at the same source time. Every garment seen by either wrist camera must match that "
    "same garment in the head cameras in color, textile material, folds, and physical "
    "identity. A wrist view is only a close-up of the head-view world, not an independent "
    "restyling. Camera viewpoint, scale, proximity, shadow, and occlusion never change "
    "garment or robot appearance."
)

WRIST_BINDING = (
    "This wrist camera must follow the canonical appearance established by both head "
    "cameras. Preserve the operation-bound color of each individual garment and the same "
    "robot materials at every synchronized moment, including extreme fabric close-ups and "
    "reappearance after occlusion."
)

NEGATIVE_SUFFIX = (
    "wrist-head appearance mismatch, wrist-view garment color change, camera-dependent "
    "garment color, identity swap between first second and third garment, all garments "
    "becoming one color, active garment color bleeding onto other garments, inconsistent "
    "textile material between views, independent wrist-view restyling, robot material "
    "mismatch between cameras"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trim_video(source: Path, output: Path, frames: int, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-vf", f"trim=end_frame={frames},setpts=N/({fps}*TB)",
            "-an", "-r", str(fps), "-c:v", "libx264", "-crf", "10",
            "-preset", "slow", "-pix_fmt", "yuv420p", str(output),
        ],
        check=True,
    )


def remove_all_yellow_assumption(value):
    if isinstance(value, dict):
        return {key: remove_all_yellow_assumption(item) for key, item in value.items()}
    if isinstance(value, list):
        return [remove_all_yellow_assumption(item) for item in value]
    if not isinstance(value, str):
        return value
    replacements = (
        ("All visible garments are yellow fabric", "The three garments retain their distinct operation-bound colors"),
        ("all visible garments are yellow fabric", "the three garments retain their distinct operation-bound colors"),
        ("yellow garments", "the three operation-bound garments"),
        ("yellow garment", "the current operation-bound garment"),
        ("yellow fabric", "the current operation-bound fabric"),
        ("Yellow garments", "The three operation-bound garments"),
        ("Yellow garment", "The current operation-bound garment"),
        ("Yellow fabric", "The current operation-bound fabric"),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    return value.replace("yellow", "garment-specific").replace("Yellow", "Garment-specific")


def bind_prompt(source: dict, view: str) -> dict:
    prompt = remove_all_yellow_assumption(copy.deepcopy(source))
    role = "canonical head-view reference" if view.startswith("head_") else "wrist-view follower"
    prompt["cross_view_consistency"] = {
        "role": role,
        "canonical_views": [
            "head_stereo_left__file-000",
            "head_stereo_right__file-000",
        ],
        "garment_identity_table": GARMENT_IDENTITIES,
        "rule": SHARED_APPEARANCE,
    }
    context = str(prompt.get("context") or "").strip()
    prompt["context"] = f"{GARMENT_IDENTITIES} {SHARED_APPEARANCE} {context}".strip()
    prefix = f"{GARMENT_IDENTITIES} {SHARED_APPEARANCE}"
    if view.startswith("wrist_"):
        prefix = f"{prefix} {WRIST_BINDING}"
    prompt["temporal_caption"] = (
        f"{prefix} From 0:00 to 0:10, garment A is the first handled object and remains "
        "golden yellow in every camera. From 0:10 to 0:23, garment B is the second handled "
        "object and remains cobalt blue in every camera. From 0:23 to 0:30, garment C is "
        "the third handled object and remains emerald green in every camera. Other garments "
        "visible in the basket keep their own identity-bound colors throughout."
    )
    prompt["actions"] = [
        {
            "time": "0:00-0:10",
            "description": (
                "The robot approaches, grasps, transports, and releases garment A, the "
                "golden-yellow first garment. It remains the same golden-yellow object in "
                "both head cameras and both wrist cameras."
            ),
        },
        {
            "time": "0:10-0:23",
            "description": (
                "The robot approaches, grasps, transports, and releases garment B, the "
                "cobalt-blue second garment. It remains the same cobalt-blue object in all "
                "four cameras; garments A and C keep their own colors."
            ),
        },
        {
            "time": "0:23-0:30",
            "description": (
                "The robot approaches, grasps, transports, and releases garment C, the "
                "emerald-green third garment. It remains the same emerald-green object in "
                "all four cameras; garments A and B keep their own colors."
            ),
        },
    ]
    prompt["segments"] = [
        {
            "segment_index": 0,
            "time_range": "0:00-0:10",
            "description": "First operation: persistent golden-yellow garment A.",
            "key_changes": "garment A moves from basket to washer without changing identity",
            "camera": "source camera motion and viewpoint are preserved",
        },
        {
            "segment_index": 1,
            "time_range": "0:10-0:23",
            "description": "Second operation: persistent cobalt-blue garment B.",
            "key_changes": "garment B moves from basket to washer without changing identity",
            "camera": "source camera motion and viewpoint are preserved",
        },
        {
            "segment_index": 2,
            "time_range": "0:23-0:30",
            "description": "Third operation: persistent emerald-green garment C.",
            "key_changes": "garment C moves from basket to washer without changing identity",
            "camera": "source camera motion and viewpoint are preserved",
        },
    ]
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--output-experiment", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source = args.source_experiment.resolve()
    output = args.output_experiment.resolve()
    source_specs = sorted((source / "specs" / f"seed_{args.seed}").glob("*.json"))
    if len(source_specs) != 4:
        raise SystemExit(f"Expected four source specs, found {len(source_specs)}")

    first_spec = read_json(source_specs[0])
    source_negative_path = Path(first_spec["negative_prompt_file"])
    negative = read_json(source_negative_path).rstrip(".") + ", " + NEGATIVE_SUFFIX + "."
    negative_path = output / "negative" / "crossview_shadow_negative.json"
    write_json(negative_path, negative)

    manifest = {
        "method": "E_alt Edge+Depth four-view operation-bound identity prompt ablation",
        "frames": args.frames,
        "seed": args.seed,
        "edge": {"weight": 0.6, "preset_edge_threshold": "medium"},
        "depth": {"weight": 0.4, "auto_generate": True},
        "shared_appearance": SHARED_APPEARANCE,
        "garment_identities": GARMENT_IDENTITIES,
        "wrist_binding": WRIST_BINDING,
        "source_experiment": str(source),
        "views": {},
    }

    for source_spec_path in source_specs:
        source_spec = read_json(source_spec_path)
        scene, view = source_spec["name"].split("__", 1)
        fps = float(source_spec["fps"])
        source_video = Path(source_spec["vision_path"])
        clip_path = output / "inputs" / f"{view}_first_{args.frames}f.mp4"
        trim_video(source_video, clip_path, args.frames, fps)

        source_prompt_path = Path(source_spec["prompt_path"])
        prompt = bind_prompt(read_json(source_prompt_path), view)
        prompt_path = output / "prompts" / scene / view / "repaired_prompt.json"
        write_json(prompt_path, prompt)

        spec = copy.deepcopy(source_spec)
        spec["name"] = f"{scene}__{view}__crossview_e{args.frames}"
        spec["vision_path"] = str(clip_path)
        spec["prompt_path"] = str(prompt_path)
        spec["negative_prompt_file"] = str(negative_path)
        spec["num_frames"] = args.frames
        spec["max_frames"] = args.frames
        spec["num_video_frames_per_chunk"] = args.frames
        spec["seed"] = args.seed
        spec["edge"] = {"weight": 0.6, "preset_edge_threshold": "medium"}
        spec["depth"] = {"weight": 0.4, "auto_generate": True}
        output_spec = output / "specs" / f"seed_{args.seed}" / source_spec_path.name
        write_json(output_spec, spec)

        manifest["views"][view] = {
            "source_video": str(source_video),
            "input_clip": str(clip_path),
            "input_clip_sha256": sha256(clip_path),
            "source_prompt": str(source_prompt_path),
            "prepared_prompt": str(prompt_path),
            "prepared_prompt_sha256": sha256(prompt_path),
            "prepared_spec": str(output_spec),
            "prepared_spec_sha256": sha256(output_spec),
        }

    write_json(output / "provenance" / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

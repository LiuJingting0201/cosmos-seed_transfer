#!/usr/bin/env python3
"""Prepare a controlled four-view garment-identity prompt ablation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path


VARIANTS = {
    "P1_source_color_anchor_compact": {
        "description": "Compact source-color identity mapping with minimal per-view camera text.",
        "redundant": False,
        "identical_across_views": False,
    },
    "P2_source_position_trajectory_redundant": {
        "description": "Source color, initial basket position, time, and trajectory repeated in fields.",
        "redundant": True,
        "identical_across_views": False,
    },
    "P3_identical_prompt_all_views": {
        "description": "Byte-identical compact semantic prompt for all four camera views.",
        "redundant": False,
        "identical_across_views": True,
    },
}

GARMENTS = [
    {
        "id": "garment_A",
        "source": "coral-red garment initially at the front-left of the basket",
        "target": "golden yellow",
        "hex": "#D8A928",
        "interval": "0:00-0:10",
        "trajectory": "first garment grasped, lifted from the basket, carried, and loaded",
    },
    {
        "id": "garment_B",
        "source": "black garment initially near the middle of the basket",
        "target": "cobalt blue",
        "hex": "#2459C4",
        "interval": "0:10-0:22",
        "trajectory": "second garment handled after garment_A",
    },
    {
        "id": "garment_C",
        "source": "dark navy-blue garment initially at the back-right of the basket",
        "target": "emerald green",
        "hex": "#16845B",
        "interval": "0:22-0:30",
        "trajectory": "third garment handled after garment_B",
    },
]

CAMERA_TEXT = {
    "head_stereo_left__file-000": "left head-camera high-angle wide view",
    "head_stereo_right__file-000": "right head-camera high-angle wide view",
    "wrist_left__file-000": "left wrist-camera close view moving with the gripper",
    "wrist_right__file-000": "right wrist-camera close view moving with the gripper",
}

SHARED_RULE = (
    "Use source-video appearance only to identify each physical garment, then replace its "
    "color with the registered target color. The source coral-red garment is garment_A "
    "and must render golden yellow (#D8A928), never blue or green. The source black garment "
    "is garment_B and must render cobalt blue (#2459C4), never yellow or green. The source "
    "dark navy-blue garment is garment_C and must render emerald green (#16845B), never "
    "yellow or blue. Source color is an identity cue, not an output color. Identity follows "
    "the same fabric continuously through folds, scale changes, close-ups, occlusion, "
    "reappearance, grasping, carrying, and release. Camera viewpoint never changes identity."
)

TEST_INTERVAL_RULE = (
    "This test covers 0:00-0:12. Garment_A is the coral-red source garment being handled "
    "from 0:00-0:10 and must stay golden yellow in every frame and every view, including "
    "wrist close-ups around 0:05-0:09. Never render garment_A cobalt blue, navy, green, or "
    "coral-red. Around 0:10-0:12 the robot returns toward the basket; any source-black "
    "garment_B remains cobalt blue."
)

SHADOW_RULE = (
    "Preserve the source geometry, robot, black grippers, washer, basket, and motion. Cast "
    "shadows are flat lighting effects and never become grippers or mechanical objects."
)

NEGATIVE_SUFFIX = (
    "source coral-red garment rendered blue, source coral-red garment rendered green, "
    "source coral-red garment keeping coral color, garment_A color mismatch between views, "
    "garment identity swap after close-up, target color reassignment after occlusion, "
    "camera-dependent garment color, wrist-view color change, garment changing color "
    "mid-trajectory, all garments sharing one color"
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
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", f"trim=end_frame={frames},setpts=N/({fps}*TB)", "-an", "-r", str(fps),
            "-c:v", "libx264", "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv420p",
            str(output),
        ],
        check=True,
    )
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=avg_frame_rate,nb_read_frames", "-of", "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    if int(stream["nb_read_frames"]) != frames:
        raise RuntimeError(f"{output} has {stream['nb_read_frames']} frames, expected {frames}")


def registry() -> dict:
    return {
        garment["id"]: {
            "source_identity_anchor": garment["source"],
            "target_color": garment["target"],
            "target_hex": garment["hex"],
            "handling_interval": garment["interval"],
            "continuous_trajectory": garment["trajectory"],
        }
        for garment in GARMENTS
    }


def garment_subject(garment: dict, *, redundant: bool) -> dict:
    description = (
        f"{garment['id']}: {garment['source']}; output is always "
        f"{garment['target']} ({garment['hex']})"
    )
    if redundant:
        description += (
            f"; active interval {garment['interval']}; {garment['trajectory']}; the same "
            "physical fabric keeps this identity through every camera and occlusion"
        )
    return {
        "description": description,
        "appearance_details": (
            f"source geometry and woven folds, recolored uniformly to {garment['target']} "
            f"({garment['hex']}) without source-color remnants"
        ),
        "action": garment["trajectory"],
        "state_changes": (
            f"position and folds change continuously while identity remains {garment['id']} "
            f"and output color remains {garment['target']} ({garment['hex']})"
        ),
    }


def build_prompt(source: dict, view: str, variant: dict) -> dict:
    redundant = variant["redundant"]
    identical = variant["identical_across_views"]
    camera = (
        "camera viewpoint and framing follow the input reference video"
        if identical
        else CAMERA_TEXT[view]
    )
    prompt = {
        "subjects": [
            {
                "description": "the same dual-arm robot and two physical black grippers",
                "appearance_details": "unchanged source robot geometry, materials, and hardware",
                "action": "follows the exact source-video motion",
                "state_changes": "no added, removed, duplicated, or restyled robot components",
            },
            *(garment_subject(garment, redundant=redundant) for garment in GARMENTS),
        ],
        "background_setting": (
            "The source laundry alcove, open washing machine, and white woven basket remain "
            "unchanged. Replace only the floor surface with realistic matte brown wood grain."
        ),
        "lighting": {
            "description": "natural source lighting with stable exposure and material colors",
            "shadows": SHADOW_RULE,
        },
        "aesthetics": {
            "color_scheme": (
                "garment_A golden yellow #D8A928; garment_B cobalt blue #2459C4; "
                "garment_C emerald green #16845B; silver robot; black grippers; white "
                "basket; dark washer; brown wood-grain floor"
            ),
            "visual_quality": "photorealistic source-video detail and stable colors",
        },
        "cinematography": {
            "camera": camera,
            "motion": "exactly follows the input reference video",
            "framing": "exactly follows the input reference video",
        },
        "style_medium": "photorealistic video",
        "artistic_style": "realistic industrial robot recording",
        "context": (
            "A dual-arm robot transfers three persistent physical garments from a basket "
            "into a washing machine. Garment identity is shared across all four cameras."
        ),
        "actions": [
            {
                "time": garment["interval"],
                "description": (
                    f"{garment['source']} is {garment['id']} and follows its continuous source "
                    f"trajectory while always rendering {garment['target']} ({garment['hex']})."
                ),
            }
            for garment in GARMENTS
        ],
        "temporal_caption": f"{SHARED_RULE} {TEST_INTERVAL_RULE} {SHADOW_RULE}",
        "cross_view_identity_registry": registry(),
        "identity_constraints": f"{SHARED_RULE} {TEST_INTERVAL_RULE}",
        "resolution": source.get("resolution", "704p"),
        "aspect_ratio": source.get("aspect_ratio", "4:3"),
        "duration": "12.1 seconds",
        "fps": 10,
    }
    if redundant:
        prompt["segments"] = [
            {
                "time": garment["interval"],
                "identity_anchor": garment["source"],
                "required_output": (
                    f"the same {garment['id']} remains {garment['target']} "
                    f"({garment['hex']}) in every camera"
                ),
                "trajectory": garment["trajectory"],
            }
            for garment in GARMENTS
        ]
        prompt["identity_constraints"] = (
            f"{SHARED_RULE} {TEST_INTERVAL_RULE} Initial basket position, source color, "
            "handling time, and continuous trajectory jointly identify each garment. "
            "Repeat the registered target color after every occlusion and close-up."
        )
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--frames", type=int, default=121)
    args = parser.parse_args()

    source = args.source_experiment.resolve()
    output = args.output_root.resolve()
    source_specs = sorted((source / "specs" / f"seed_{args.seed}").glob("*.json"))
    if len(source_specs) != 4:
        raise SystemExit(f"Expected four source specs, found {len(source_specs)}")

    source_negative_path = Path(read_json(source_specs[0])["negative_prompt_file"])
    source_negative = read_json(source_negative_path).rstrip(".")
    negative_path = output / "negative" / "garment_prompt_ablation_negative.json"
    write_json(negative_path, source_negative + ", " + NEGATIVE_SUFFIX + ".")

    manifest = {
        "method": "garment identity prompt ablation",
        "controlled_variables": {
            "seed": args.seed,
            "frames": args.frames,
            "fps": 10.0,
            "edge": {"weight": 0.6, "preset_edge_threshold": "medium"},
            "depth": {"weight": 0.4, "auto_generate": True},
        },
        "source_experiment": str(source),
        "garments": GARMENTS,
        "variants": {},
    }
    clips = {}
    for source_spec_path in source_specs:
        spec = read_json(source_spec_path)
        _, view = spec["name"].split("__", 1)
        view = view.split("__identity_palette", 1)[0]
        clip = output / "inputs" / f"{view}__first_{args.frames}f.mp4"
        trim_video(Path(spec["vision_path"]), clip, args.frames, float(spec["fps"]))
        clips[view] = clip

    for variant_name, variant in VARIANTS.items():
        variant_manifest = {
            "description": variant["description"],
            "identical_across_views": variant["identical_across_views"],
            "views": {},
        }
        hashes = []
        for source_spec_path in source_specs:
            spec = read_json(source_spec_path)
            scene, view_with_suffix = spec["name"].split("__", 1)
            view = view_with_suffix.split("__identity_palette", 1)[0]
            source_prompt = read_json(Path(spec["prompt_path"]))
            prompt = build_prompt(source_prompt, view, variant)
            prompt_path = output / "prompts" / variant_name / view / "prompt.json"
            write_json(prompt_path, prompt)

            prepared = copy.deepcopy(spec)
            prepared["name"] = f"{scene}__{view}__{variant_name}"
            prepared["vision_path"] = str(clips[view])
            prepared["prompt_path"] = str(prompt_path)
            prepared["negative_prompt_file"] = str(negative_path)
            prepared["num_frames"] = args.frames
            prepared["max_frames"] = args.frames
            prepared["num_video_frames_per_chunk"] = args.frames
            prepared["edge"] = {"weight": 0.6, "preset_edge_threshold": "medium"}
            prepared["depth"] = {"weight": 0.4, "auto_generate": True}
            spec_path = output / "specs" / variant_name / f"{scene}__{view}.json"
            write_json(spec_path, prepared)
            prompt_hash = sha256(prompt_path)
            hashes.append(prompt_hash)
            variant_manifest["views"][view] = {
                "clip": str(clips[view]),
                "prompt": str(prompt_path),
                "prompt_sha256": prompt_hash,
                "spec": str(spec_path),
                "spec_sha256": sha256(spec_path),
            }
        if variant["identical_across_views"] and len(set(hashes)) != 1:
            raise RuntimeError(f"{variant_name} prompts are not byte-identical")
        variant_manifest["unique_prompt_hashes"] = len(set(hashes))
        manifest["variants"][variant_name] = variant_manifest

    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

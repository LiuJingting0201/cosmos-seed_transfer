#!/usr/bin/env python3
"""Prepare a full-length, four-view Edge+Depth Transfer experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path


SHADOW_NEGATIVE = (
    "solidified shadows, shadows converted into objects, phantom grippers, "
    "duplicated end effectors, mechanical structures emerging from fabric, "
    "cast shadows with physical volume"
)

SHARED_SHADOW_RULE = (
    "Cast shadows are flat lighting effects attached to the surfaces receiving them. "
    "Never turn a gripper-shaped or robot-shaped shadow into a physical object, and "
    "never add or duplicate a gripper, end effector, joint, arm, or robot component."
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


def shadow_aware_prompt(source: dict) -> dict:
    prompt = copy.deepcopy(source)
    lighting = prompt.setdefault("lighting", {})
    original_shadows = str(lighting.get("shadows") or "").strip()
    lighting["shadows"] = f"{original_shadows} {SHARED_SHADOW_RULE}".strip()
    original_caption = str(prompt.get("temporal_caption") or "").strip()
    prompt["temporal_caption"] = f"Shared identity constraint: {SHARED_SHADOW_RULE} {original_caption}"
    for subject in prompt.get("subjects", []):
        if "gripper" not in str(subject.get("description", "")).lower():
            continue
        state = str(subject.get("state_changes") or "").strip()
        subject["state_changes"] = (
            f"{state} Its cast shadows remain flat surface illumination and never become "
            "additional physical robot components."
        ).strip()
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--source-negative", type=Path, required=True)
    parser.add_argument("--output-experiment", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--edge-weight", type=float, default=0.6)
    parser.add_argument("--depth-weight", type=float, default=0.4)
    args = parser.parse_args()

    source_experiment = args.source_experiment.resolve()
    output_experiment = args.output_experiment.resolve()
    source_specs = source_experiment / "specs" / f"seed_{args.seed}"
    spec_paths = sorted(source_specs.glob("*.json"))
    if len(spec_paths) != 4:
        raise SystemExit(f"Expected exactly four source specs, found {len(spec_paths)}")

    baseline_negative = read_json(args.source_negative)
    shadow_negative = baseline_negative.rstrip(".") + ", " + SHADOW_NEGATIVE + "."
    negative_path = output_experiment / "negative" / "shadow_aware_negative.json"
    write_json(negative_path, shadow_negative)

    scene_names: set[str] = set()
    provenance = {
        "method": "full-length four-view Edge+Depth Transfer",
        "seed": args.seed,
        "shared_shadow_rule": SHARED_SHADOW_RULE,
        "edge": {"weight": args.edge_weight, "preset_edge_threshold": "medium"},
        "depth": {"weight": args.depth_weight, "auto_generate": True},
        "source_experiment": str(source_experiment),
        "source_negative": str(args.source_negative.resolve()),
        "views": {},
    }

    for source_spec_path in spec_paths:
        source_spec = read_json(source_spec_path)
        name = source_spec["name"]
        scene, view = name.split("__", 1)
        scene_names.add(scene)
        source_prompt_path = Path(source_spec["prompt_path"])
        prompt = shadow_aware_prompt(read_json(source_prompt_path))
        prompt_path = (
            output_experiment / "prompt_sets" / "edge_depth_e" / scene / view
            / "repaired_prompt.json"
        )
        write_json(prompt_path, prompt)

        spec = copy.deepcopy(source_spec)
        spec["seed"] = args.seed
        spec["prompt_path"] = str(prompt_path)
        spec["negative_prompt_file"] = str(negative_path)
        for key in ("edge", "depth", "seg", "blur", "wsm", "multi_control"):
            spec.pop(key, None)
        spec["edge"] = {
            "weight": args.edge_weight,
            "preset_edge_threshold": "medium",
        }
        spec["depth"] = {
            "weight": args.depth_weight,
            "auto_generate": True,
        }
        output_spec_path = output_experiment / "specs" / f"seed_{args.seed}" / source_spec_path.name
        write_json(output_spec_path, spec)
        provenance["views"][view] = {
            "source_spec": str(source_spec_path),
            "source_spec_sha256": sha256(source_spec_path),
            "source_prompt": str(source_prompt_path),
            "source_prompt_sha256": sha256(source_prompt_path),
            "prepared_prompt": str(prompt_path),
            "prepared_prompt_sha256": sha256(prompt_path),
            "prepared_spec": str(output_spec_path),
            "prepared_spec_sha256": sha256(output_spec_path),
            "vision_path": source_spec["vision_path"],
        }

    if len(scene_names) != 1:
        raise SystemExit(f"Expected one scene, found {sorted(scene_names)}")
    provenance_dir = output_experiment / "provenance" / "edge_depth_e"
    write_json(provenance_dir / "manifest.json", provenance)
    logs = output_experiment / "logs" / "transfer" / f"seed_{args.seed}"
    logs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source_negative, provenance_dir / "source_negative.json")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare four-view Cosmos specs with persistent garment identities and colors."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


GARMENTS = [
    ("garment_A", "first garment handled", "golden yellow", "#D8A928", "0:00-0:10"),
    ("garment_B", "second garment handled", "cobalt blue", "#2459C4", "0:10-0:19"),
    ("garment_C", "third garment handled", "emerald green", "#16845B", "0:19-0:30"),
]

IDENTITY_RULE = (
    "Cross-view garment identity registry: garment_A is the first garment handled and "
    "is always golden yellow (#D8A928); garment_B is the second garment handled and "
    "is always cobalt blue (#2459C4); garment_C is the third garment handled and is "
    "always emerald green (#16845B). Determine identity only from handling order and "
    "continuous motion, never from camera viewpoint, apparent source color, scale, "
    "occlusion, or proximity to a wrist camera. Each garment keeps its assigned color "
    "in every head and wrist view, before grasping, while carried, during close-up "
    "occlusion, after reappearance, and inside the washer. Never swap garment IDs or "
    "transfer a color between garments."
)

IDENTITY_NEGATIVE = (
    "cross-view garment color mismatch, garment identity swap, color reassignment "
    "after occlusion, garment changing color mid-motion, colors bleeding between "
    "garments, treating a wrist close-up as a different garment, assigning color by "
    "camera viewpoint, all garments collapsing into one identity"
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


def deconflict_text(value):
    if isinstance(value, dict):
        return {key: deconflict_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [deconflict_text(item) for item in value]
    if not isinstance(value, str):
        return value
    replacements = [
        ("All visible garments are yellow fabric", "Visible garments follow the shared identity palette"),
        ("all visible handled clothes are yellow", "handled clothes retain their assigned identity colors"),
        ("yellow garments", "identity-tracked garments"),
        ("yellow garment", "identity-tracked garment"),
        ("yellow clothes", "identity-tracked clothes"),
        ("yellow fabric", "identity-tracked fabric"),
        ("Yellow garments", "Identity-tracked garments"),
        ("Yellow garment", "Identity-tracked garment"),
        ("Yellow fabric", "Identity-tracked fabric"),
    ]
    for source, target in replacements:
        value = value.replace(source, target)
    return value


def garment_subject(identifier: str, order: str, color: str, hex_color: str) -> dict:
    return {
        "description": f"{identifier}, the {order}",
        "appearance_details": (
            f"soft wrinkled fabric with stable {color} color ({hex_color}), visible "
            "folds, seams, and woven texture; geometry follows the source garment"
        ),
        "relationship": "one of three persistent garments moved from the basket into the washer",
        "location": "basket, gripper, transfer path, washer opening, or washer drum",
        "relative_size": "varies with camera distance without changing identity",
        "orientation": "irregular folds following the original motion",
        "pose": "crumpled, grasped, carried, or released",
        "action": "follows its source-video grasp, transfer, and release motion",
        "state_changes": (
            f"position and folds change continuously; identity remains {identifier} and "
            f"color remains {color} ({hex_color}) in all four views"
        ),
        "clothing": f"{color} garment",
        "expression": "",
        "gender": "",
        "age": "",
        "skin_tone_and_texture": "",
        "facial_features": "",
        "number_of_subjects": 1,
        "number_of_arms": 0,
        "number_of_legs": 0,
    }


def identity_prompt(source: dict) -> dict:
    prompt = deconflict_text(copy.deepcopy(source))
    subjects = []
    for subject in prompt.get("subjects", []):
        description = str(subject.get("description", "")).lower()
        if "garment" in description or "fabric" in description:
            continue
        subjects.append(subject)
    subjects.extend(garment_subject(*garment[:4]) for garment in GARMENTS)
    prompt["subjects"] = subjects
    prompt["cross_view_identity_registry"] = {
        identifier: {
            "handling_order": order,
            "output_color": color,
            "hex": hex_color,
            "active_interval": interval,
        }
        for identifier, order, color, hex_color, interval in GARMENTS
    }
    prompt["identity_constraints"] = IDENTITY_RULE
    prompt["context"] = (
        "A dual-arm robot transfers three persistent, identity-tracked garments from "
        "a basket into a washing machine. Their colors follow the shared registry."
    )
    prompt["actions"] = [
        {
            "time": interval,
            "description": (
                f"{identifier}, the {order}, follows the original grasp, lift, transfer, "
                f"and release motion while remaining {color} ({hex_color}) in every view."
            ),
        }
        for identifier, order, color, hex_color, interval in GARMENTS
    ]
    view_caption = str(prompt.get("temporal_caption") or "").strip()
    prompt["temporal_caption"] = f"{IDENTITY_RULE} View-specific observation: {view_caption}"
    aesthetics = prompt.setdefault("aesthetics", {})
    aesthetics["color_scheme"] = (
        "garment_A golden yellow (#D8A928), garment_B cobalt blue (#2459C4), garment_C "
        "emerald green (#16845B), white basket, silver robot, black grippers, dark washer, "
        "and brown wood-grain floor"
    )
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--output-experiment", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source = args.source_experiment.resolve()
    output = args.output_experiment.resolve()
    source_specs = sorted((source / "specs" / f"seed_{args.seed}").glob("*.json"))
    if len(source_specs) != 4:
        raise SystemExit(f"Expected four source specs, found {len(source_specs)}")

    source_negative = Path(read_json(source_specs[0])["negative_prompt_file"])
    negative = read_json(source_negative).rstrip(".") + ", " + IDENTITY_NEGATIVE + "."
    negative_path = output / "negative" / "garment_identity_negative.json"
    write_json(negative_path, negative)

    manifest = {
        "method": "four-view persistent garment identity palette",
        "seed": args.seed,
        "identity_rule": IDENTITY_RULE,
        "garments": GARMENTS,
        "source_experiment": str(source),
        "views": {},
    }
    prompt_hashes = {}
    for source_spec_path in source_specs:
        spec = read_json(source_spec_path)
        scene, view = spec["name"].split("__", 1)
        source_prompt_path = Path(spec["prompt_path"])
        prompt = identity_prompt(read_json(source_prompt_path))
        prompt_path = output / "prompt_sets" / "identity_palette_v1" / scene / view / "repaired_prompt.json"
        write_json(prompt_path, prompt)

        prepared = copy.deepcopy(spec)
        prepared["name"] = f"{spec['name']}__identity_palette_v1"
        prepared["prompt_path"] = str(prompt_path)
        prepared["negative_prompt_file"] = str(negative_path)
        spec_path = output / "specs" / f"seed_{args.seed}" / source_spec_path.name
        write_json(spec_path, prepared)
        prompt_hashes[view] = hashlib.sha256(
            json.dumps(prompt["cross_view_identity_registry"], sort_keys=True).encode()
        ).hexdigest()
        manifest["views"][view] = {
            "source_prompt": str(source_prompt_path),
            "prepared_prompt": str(prompt_path),
            "prepared_prompt_sha256": sha256(prompt_path),
            "prepared_spec": str(spec_path),
            "prepared_spec_sha256": sha256(spec_path),
        }

    if len(set(prompt_hashes.values())) != 1:
        raise SystemExit("Cross-view identity registries are not byte-equivalent")
    manifest["shared_identity_registry_sha256"] = next(iter(prompt_hashes.values()))
    write_json(output / "provenance" / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

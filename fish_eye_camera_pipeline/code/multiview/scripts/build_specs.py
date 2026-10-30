#!/usr/bin/env python3
"""Build one Cosmos Transfer spec per view from an official baseline spec."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import av


VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def find_video(views_root: Path, scene: str, view: str) -> Path:
    for suffix in VIDEO_SUFFIXES:
        candidate = views_root / scene / f"{view}{suffix}"
        if candidate.exists():
            return candidate
    if views_root.name == scene:
        for suffix in VIDEO_SUFFIXES:
            candidate = views_root / f"{view}{suffix}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No source video for {scene}/{view} under {views_root}")


def probe(path: Path) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
        return {
            "frames": frames,
            "fps": float(stream.average_rate),
            "width": stream.width,
            "height": stream.height,
        }


def aspect_ratio(width: int, height: int) -> str:
    value = width / height
    common = [(1, 1), (4, 3), (3, 2), (16, 9), (9, 16), (3, 4)]
    numerator, denominator = min(common, key=lambda pair: abs(pair[0] / pair[1] - value))
    return f"{numerator},{denominator}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-root", type=Path, required=True)
    ap.add_argument("--views-root", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--prompt-set", default="v1")
    ap.add_argument("--run-name", default="edge_v1")
    ap.add_argument("--control", choices=("edge", "depth"), default="edge")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--control-root", type=Path)
    ap.add_argument("--negative-prompt-file", type=Path, required=True)
    args = ap.parse_args()

    experiment = args.experiment_root.resolve()
    prompt_root = experiment / "prompt_sets" / args.prompt_set
    audit_marker = prompt_root / "AUDIT_DONE"
    if not audit_marker.exists():
        raise SystemExit(f"Audit gate is closed: missing {audit_marker}")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline.get("emphasize_control_in_prompt") is not False:
        raise SystemExit("Official appearance-Transfer baseline must keep emphasize_control_in_prompt=false")
    if args.control not in baseline:
        raise SystemExit(f"Baseline does not contain the selected {args.control!r} hint")
    if not args.negative_prompt_file.is_file():
        raise SystemExit(f"Negative prompt file does not exist: {args.negative_prompt_file}")
    out = experiment / "specs" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    allowed_changes = {
        "name", "prompt_path", "vision_path", "negative_prompt_file", "seed",
        "num_frames", "max_frames", "fps", "aspect_ratio",
        "edge", "depth", "blur", "seg", "wsm", "multi_control",
    }

    for scene_dir in sorted(p for p in prompt_root.iterdir() if p.is_dir()):
        for view_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            prompt_file = view_dir / "modified_prompt.json"
            if not prompt_file.exists():
                continue
            source = find_video(args.views_root.resolve(), scene_dir.name, view_dir.name)
            media = probe(source)
            spec = copy.deepcopy(baseline)
            spec["name"] = f"{scene_dir.name}__{view_dir.name}"
            if args.seed is not None:
                spec["seed"] = args.seed
            spec["num_frames"] = media["frames"]
            if "max_frames" in spec:
                spec["max_frames"] = media["frames"]
            spec["fps"] = round(media["fps"], 6)
            spec["aspect_ratio"] = aspect_ratio(media["width"], media["height"])

            spec["prompt_path"] = str(prompt_file.resolve())
            spec["vision_path"] = str(source.resolve())
            spec["negative_prompt_file"] = str(args.negative_prompt_file.resolve())
            for key in ("edge", "depth", "blur", "seg", "wsm", "multi_control"):
                spec.pop(key, None)
            if args.control == "edge":
                edge = copy.deepcopy(baseline["edge"])
                edge.pop("control_path", None)
                spec["edge"] = edge
            else:
                if args.control_root is None:
                    raise SystemExit("--control-root is required for depth")
                control_path = args.control_root / scene_dir.name / f"{view_dir.name}.mp4"
                if not control_path.is_file():
                    raise SystemExit(f"Depth control video does not exist: {control_path}")
                depth = copy.deepcopy(baseline["depth"])
                spec["depth"] = {
                    **depth,
                    "control_path": str(control_path.resolve()),
                }
            changed_fixed = [
                key for key in sorted(set(baseline) | set(spec))
                if key not in allowed_changes and baseline.get(key) != spec.get(key)
            ]
            if changed_fixed:
                raise SystemExit(
                    f"Spec {spec['name']} changed fixed baseline fields: {changed_fixed}"
                )
            (out / f"{spec['name']}.json").write_text(
                json.dumps(spec, indent=2) + "\n", encoding="utf-8"
            )
            count += 1
    if count == 0:
        raise SystemExit("No audited modified prompts were found; wrote no specs")
    print(f"specs written: {count} -> {out}")


if __name__ == "__main__":
    main()

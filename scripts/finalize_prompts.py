#!/usr/bin/env python3
"""Finalize Reasoner JSON and structurally inject one shared block per scene."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


TOP_KEYS = [
    "subjects", "background_setting", "lighting", "aesthetics", "cinematography",
    "style_medium", "artistic_style", "context", "actions", "text_and_signage_elements",
    "segments", "transitions", "temporal_caption", "audio_description", "resolution",
    "aspect_ratio", "duration", "fps",
]
SUBJECT_KEYS = [
    "description", "appearance_details", "relationship", "location", "relative_size",
    "orientation", "pose", "action", "state_changes", "clothing", "expression", "gender",
    "age", "skin_tone_and_texture", "facial_features", "number_of_subjects",
    "number_of_arms", "number_of_legs",
]
SHARED_TOP = ["background_setting", "context", "style_medium", "artistic_style"]
SHARED_NESTED = {
    "lighting": ["conditions", "direction", "shadows", "illumination_effect"],
    "aesthetics": ["color_scheme", "mood_atmosphere", "patterns"],
}


def normalized_subject(subject: dict) -> dict:
    return {key: copy.deepcopy(subject.get(key, "")) for key in SUBJECT_KEYS}


def shared_value(plan: dict, canonical: dict, parent: str, child: str | None = None):
    shared = plan.get("shared") or {}
    source = shared if parent in shared else canonical
    value = source.get(parent)
    if child is None:
        return copy.deepcopy(value)
    return copy.deepcopy((value or {}).get(child))


def choose_canonical(views: dict[str, dict], plan: dict) -> str:
    requested = plan.get("canonical_view")
    if requested in views:
        return requested
    for name in ("cam_left_high", "cam_right_high", "cam_0"):
        if name in views:
            return name
    return sorted(views)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-set-dir", type=Path, required=True)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=float)
    ap.add_argument("--duration-secs", type=float)
    ap.add_argument("--scene", action="append", default=[])
    args = ap.parse_args()

    selected = set(args.scene)
    finalized = 0
    for scene_dir in sorted(p for p in args.prompt_set_dir.iterdir() if p.is_dir()):
        if selected and scene_dir.name not in selected:
            continue
        plan_path = scene_dir / "transfer_plan.json"
        if not plan_path.exists():
            raise SystemExit(f"Missing transfer plan: {plan_path}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        views: dict[str, dict] = {}
        for view_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            source = view_dir / "upsampled_prompt.json"
            if source.exists():
                views[view_dir.name] = json.loads(source.read_text(encoding="utf-8"))
        if not views:
            continue

        canonical_name = choose_canonical(views, plan)
        canonical = views[canonical_name]
        shared = plan.get("shared") or {}
        robot = shared.get("robot_subject")
        end_effector = shared.get("end_effector_subject")
        media = plan.get("media") or {}
        width = args.width or media.get("width")
        height = args.height or media.get("height")
        fps = args.fps or media.get("fps")
        duration = args.duration_secs or media.get("duration_s")

        for view_name, prompt in views.items():
            prompt.pop("scene_imagination", None)
            prompt["subjects"] = [normalized_subject(s) for s in prompt.get("subjects", [])]
            for key in SHARED_TOP:
                value = shared_value(plan, canonical, key)
                if value is not None:
                    prompt[key] = value
            for parent, children in SHARED_NESTED.items():
                prompt.setdefault(parent, {})
                for child in children:
                    value = shared_value(plan, canonical, parent, child)
                    if value is not None:
                        prompt[parent][child] = value

            if robot:
                shared_robot = normalized_subject(robot)
                candidates = [
                    sub for sub in prompt["subjects"]
                    if any(
                        word in str(sub.get("description", "")).lower()
                        for word in ("robot arm", "robotic arm", "manipulator")
                    )
                ]
                for sub in candidates:
                    for field in ("description", "appearance_details"):
                        sub[field] = shared_robot[field]
            elif plan.get("shared_subject_mode") != "explicit" and canonical.get("subjects") and prompt["subjects"]:
                for field in ("description", "appearance_details"):
                    prompt["subjects"][0][field] = canonical["subjects"][0].get(field, "")

            if end_effector:
                shared_effector = normalized_subject(end_effector)
                candidates = [sub for sub in prompt["subjects"][1:] if any(word in str(sub.get("description", "")).lower() for word in ("gripper", "end effector", "end-effector", "robotic hand"))]
                if candidates:
                    for sub in candidates:
                        sub["description"] = shared_effector["description"]
                        sub["appearance_details"] = shared_effector["appearance_details"]

            camera = (plan.get("views") or {}).get(view_name, {}).get("camera_motion")
            if camera:
                prompt.setdefault("cinematography", {})["camera_motion"] = camera
            if width and height:
                prompt["resolution"] = {"W": int(width), "H": int(height)}
            if fps:
                prompt["fps"] = float(fps)
            if duration:
                prompt["duration"] = f"{float(duration):g}s"

            for key in TOP_KEYS:
                prompt.setdefault(key, [] if key in {"subjects", "actions", "text_and_signage_elements", "segments", "transitions"} else "")
            ordered = {key: prompt[key] for key in TOP_KEYS}
            ordered.update({key: value for key, value in prompt.items() if key not in ordered})
            output = scene_dir / view_name / "modified_prompt.json"
            output.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            finalized += 1
        print(f"{scene_dir.name}: finalized {len(views)} views from {canonical_name}")

    marker = args.prompt_set_dir / "AUDIT_REQUIRED"
    marker.write_text(
        "Finalization is complete. Independent per-scene audit is required before check/build-specs.\n",
        encoding="utf-8",
    )
    audit_done = args.prompt_set_dir / "AUDIT_DONE"
    if audit_done.exists():
        audit_done.unlink()
    print(f"finalized {finalized} view prompts; wrote {marker}")


if __name__ == "__main__":
    main()

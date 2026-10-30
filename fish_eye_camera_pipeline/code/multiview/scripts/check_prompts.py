#!/usr/bin/env python3
"""Cross-view prompt consistency check for per-view Cosmos transfer generation.

When each camera is generated as its own job, nothing but the prompt text keeps
the views looking like the same restyled scene — the seed only fixes the initial
noise, and every view's control map and input video differ, so the denoising
trajectories diverge from step one.

This script enforces the field split from the cosmos-transfer-per-view skill:
the scene-level fields must be byte-identical across a group's views, per-object
colour phrases must be bound to that object's own subjects[] entry, and
view-relative phrasing must stay out of the shared fields. Describing positions
inside a per-view field is expected and is not flagged.

Usage:
    check_prompts.py <group_dir> [--require-audit]
where <group_dir> holds one subdirectory per view, each with
modified_prompt.json, plus a shared transfer_plan.json in either place.

Exits non-zero when the group is not ready to run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Must be byte-identical across every view of a group.
SHARED_TOP = ["background_setting", "context", "style_medium", "artistic_style"]
SHARED_NESTED = [
    ("lighting", "conditions"),
    ("lighting", "direction"),
    ("lighting", "shadows"),
    ("lighting", "illumination_effect"),
    ("aesthetics", "color_scheme"),
    ("aesthetics", "mood_atmosphere"),
    ("aesthetics", "patterns"),
]

# View-relative phrasing. This is perfectly good writing inside a per-view field —
# each camera should describe what it actually sees, positions included. It is only
# a defect in a SHARED field, because that text is broadcast verbatim to every view
# and the ordering it asserts holds for at most one of them.
POSITIONAL_PHRASES = [
    "leftmost", "rightmost", "left to right", "right to left",
    "ordered ", "from the left", "from the right", "on the far left",
    "on the far right", "in the middle of the row",
]
REQUIRED_TOP = [
    "subjects", "background_setting", "lighting", "aesthetics", "cinematography",
    "style_medium", "artistic_style", "context", "actions", "text_and_signage_elements",
    "segments", "transitions", "temporal_caption", "audio_description", "resolution",
    "aspect_ratio", "duration", "fps",
]
SUBJECT_FIELDS = [
    "description", "appearance_details", "relationship", "location", "relative_size",
    "orientation", "pose", "action", "state_changes", "clothing", "expression", "gender",
    "age", "skin_tone_and_texture", "facial_features", "number_of_subjects",
    "number_of_arms", "number_of_legs",
]


def load_views(group_dir: Path) -> dict[str, dict]:
    views = {}
    for d in sorted(group_dir.iterdir()):
        f = d / "modified_prompt.json" if d.is_dir() else None
        if f and f.exists():
            views[d.name] = json.loads(f.read_text())
    return views


def get(d: dict, path: tuple[str, ...]):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("group_dir", type=Path)
    ap.add_argument("--require-audit", action="store_true")
    args = ap.parse_args()

    views = load_views(args.group_dir)
    if len(views) < 1:
        print(f"FAIL: found {len(views)} view prompt(s) under {args.group_dir}")
        return 2

    plan_path = args.group_dir / "transfer_plan.json"
    if not plan_path.exists():
        cands = sorted(args.group_dir.glob("*/transfer_plan.json"))
        plan_path = cands[0] if cands else None
    plan = json.loads(plan_path.read_text()) if plan_path else {}

    failures: list[str] = []
    names = list(views)
    ref = names[0]

    for name, prompt in views.items():
        if "scene_imagination" in prompt:
            failures.append(f"[schema] {name} still contains scene_imagination")
        missing_top = [key for key in REQUIRED_TOP if key not in prompt]
        if missing_top:
            failures.append(f"[schema] {name} missing top-level keys: {missing_top}")
        for index, subject in enumerate(prompt.get("subjects") or []):
            missing_subject = [key for key in SUBJECT_FIELDS if key not in subject]
            if missing_subject:
                failures.append(
                    f"[schema] {name} subjects[{index}] missing keys: {missing_subject}"
                )
        if args.require_audit:
            report_path = args.group_dir / name / "compliance_report.json"
            if not report_path.exists():
                failures.append(f"[audit] {name} has no compliance_report.json")
            else:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if str(report.get("status", "")).lower() != "pass":
                    failures.append(f"[audit] {name} compliance status is not pass")

    # 1. scene-level fields byte-identical
    for path in [(k,) for k in SHARED_TOP] + SHARED_NESTED:
        label = ".".join(path)
        ref_val = get(views[ref], path)
        for n in names[1:]:
            val = get(views[n], path)
            if val != ref_val:
                failures.append(
                    f"[shared-field] {label} differs\n"
                    f"    {ref}: {str(ref_val)[:160]}\n"
                    f"    {n}: {str(val)[:160]}"
                )

    # 2. the robot subject is part of the shared block: byte-identical across views,
    #    and still carrying the appearance recorded in the plan.
    def robot_sub(v: dict) -> dict | None:
        subs = v.get("subjects") or []
        return subs[0] if subs else None

    shared_plan = plan.get("shared") or {}
    check_first_subject = plan.get("shared_subject_mode") != "explicit" or bool(shared_plan.get("robot_subject"))
    ref_robot = robot_sub(views[ref]) if check_first_subject else None
    if ref_robot is not None:
        for field in ("description", "appearance_details"):
            for n in names[1:]:
                cur = robot_sub(views[n])
                if cur is None or cur.get(field) != ref_robot.get(field):
                    failures.append(
                        f"[shared-field] robot subjects[0].{field} differs\n"
                        f"    {ref}: {str(ref_robot.get(field))[:160]}\n"
                        f"    {n}: {str((cur or {}).get(field))[:160]}"
                    )
    robot = plan.get("robot_preserved_appearance") or (
        (shared_plan.get("robot_subject") or {}).get("appearance_details")
    )
    if robot:
        probe = robot[:60]
        for n, v in views.items():
            if probe not in json.dumps(v, ensure_ascii=False):
                failures.append(
                    f"[robot] the plan's appearance phrase is missing from {n}"
                )

    # 2b. The gripper / hand needs its own subjects[] entry, carrying its real source
    #     appearance, in every view. In a wrist close-up the end effector fills the
    #     foreground and touches the recoloured objects; as a clause inside the arm entry
    #     it drifts to white. The entry is part of the shared block, so it must also be
    #     byte-identical across views.
    gripper = plan.get("gripper_preserved_appearance") or (
        (shared_plan.get("end_effector_subject") or {}).get("appearance_details")
    )
    if gripper:
        probe = gripper[:50].lower()
        found: dict[str, dict] = {}
        for n, v in views.items():
            # subjects[0] is the arm, and the arm's own appearance names the gripper too,
            # so searching from index 0 would match the arm and never see whether a
            # dedicated entry exists. Look past it.
            hits = [
                sub for sub in list(v.get("subjects", []))[1:]
                if probe in f"{sub.get('description','')} {sub.get('appearance_details','')}".lower()
            ]
            if not hits:
                failures.append(
                    f"[gripper] {n} has no subjects[] entry for the gripper/hand carrying "
                    f"its real appearance; a clause inside the arm entry is not enough"
                )
                continue
            found[n] = hits[0]
            blob = json.dumps(hits[0], ensure_ascii=False).lower()
            if "foreground" not in blob:
                failures.append(
                    f"[gripper] {n}: the gripper entry does not say it keeps this colour "
                    f"where it fills the foreground of the wrist views"
                )
        if len(found) > 1:
            ref_name = next(iter(found))
            for n, sub in list(found.items())[1:]:
                for field in ("description", "appearance_details"):
                    if sub.get(field) != found[ref_name].get(field):
                        failures.append(
                            f"[shared-field] gripper subjects[].{field} differs between "
                            f"{ref_name} and {n}"
                        )

    # 3. Colour phrases must be BOUND to that object's own subjects[] entry.
    #    Presence somewhere in the file is not enough: a global substring match
    #    also passes when a view attaches the wrong colour to the wrong object.
    #
    #    Several objects of one class are expected and fine — they are grouped by
    #    source colour, and each source-colour group maps to its own distinct target
    #    colour. Objects within a group share a colour and need no distinguishing;
    #    what must hold is that different groups get different colours and that no
    #    single subject entry carries two of them.
    colours = plan.get("target", {}).get("colors") or {}
    groups: dict[str, dict[str, str]] = {}
    for obj, colour in colours.items():
        groups.setdefault(obj.split()[-1].lower(), {})[obj] = colour

    for head, members in groups.items():
        if len(members) > 1:
            seen: dict[str, str] = {}
            for obj, colour in members.items():
                key = colour.lower()
                if key in seen:
                    failures.append(
                        f"[ambiguous] '{obj}' and '{seen[key]}' are both '{head}' and share "
                        f"the target colour '{colour}', so nothing tells them apart. Give "
                        f"same-class objects distinct target colours."
                    )
                seen[key] = obj

        for n, v in views.items():
            # A subject belongs to the class whose head word appears EARLIEST in its
            # description: "a pegged rack holding three round plates" is a rack, not a
            # plate, even though it names both.
            subs = []
            for sub in v.get("subjects", []):
                desc = str(sub.get("description", "")).lower()
                pos = {h: desc.find(h) for h in groups if desc.find(h) >= 0}
                if pos and min(pos, key=pos.get) == head:
                    subs.append(sub)
            if not subs:
                if head in json.dumps(v, ensure_ascii=False).lower():
                    failures.append(
                        f"[binding] {n} mentions '{head}' outside subjects[]; give each one "
                        f"its own subjects[] entry so its colour can be bound and checked"
                    )
                continue  # not visible in this view -> correctly omitted

            for sub in subs:
                blob = json.dumps(sub, ensure_ascii=False).lower()
                hits = [c for c in members.values() if c.lower() in blob]
                label = str(sub.get("description", ""))[:60]
                if not hits:
                    failures.append(
                        f"[colour] {n}: subject '{label}' is a '{head}' but carries none of "
                        f"the target colours {sorted(set(members.values()))}"
                    )
                elif len(set(h.lower() for h in hits)) > 1:
                    failures.append(
                        f"[colour] {n}: subject '{label}' carries more than one target "
                        f"colour {hits} — one entry per physical object, one colour each"
                    )

            # Two subjects sharing a target colour is correct, not a conflict: objects
            # that had the same source colour map to the same target colour and never
            # needed telling apart. Nothing to check here.

    # 4. view-relative phrasing is fine per view, but must not sit in a shared field
    for path in [(k,) for k in SHARED_TOP] + SHARED_NESTED:
        label = ".".join(path)
        for n, v in views.items():
            val = get(v, path)
            if not isinstance(val, str):
                continue
            for bad in POSITIONAL_PHRASES:
                # Match a phrase at a word boundary: 'bordered by' contains
                # 'ordered ' as a substring but makes no view-relative claim.
                if re.search(r"\b" + re.escape(bad), val.lower()):
                    failures.append(
                        f"[shared-positional] {n}: shared field {label} contains the "
                        f"view-relative phrase '{bad.strip()}'. Shared text is broadcast to "
                        f"every camera, so move this into a per-view field "
                        f"(subjects[].location, cinematography, temporal_caption)."
                    )
                    break
    for n, v in views.items():
            for sub in ((v.get("subjects") or [])[:1] if check_first_subject else []):
                txt = str(sub.get("appearance_details", "")).lower()
                for bad in POSITIONAL_PHRASES:
                    if re.search(r"\b" + re.escape(bad), txt):
                        failures.append(
                            f"[shared-positional] {n}: the robot's appearance_details is a "
                            f"shared string but contains '{bad.strip()}'"
                        )
                        break

    print(f"group {args.group_dir.name}: {len(views)} views ({', '.join(names)})")
    if failures:
        print(f"\nNOT READY — {len(failures)} problem(s):\n")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("consistency check PASSED — shared fields identical, colours and robot pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())

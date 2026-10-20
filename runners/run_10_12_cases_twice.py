#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE = Path(__file__).with_name("run_new_cases_twice.py")
SPEC = importlib.util.spec_from_file_location("shared_double_runner", MODULE)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

BASE.ROOT = Path("/root/fangweiyi/showcases/10-12-new-cases-double")
BASE.CASES = {
    "G1_WBT_Inspire_Put_Drinks_Into_Fridge_v0": {
        "prompt": (
            "将抽屉的内置改成黄色，桌面改成木质桌面，地板改成木质地板，环境改成"
            "暖光，不要变动饮料的颜色。保持机器人和人物外观、物体类别与数量、"
            "几何结构、位置、接触关系、动作、时序及相机运动不变。"
        ),
        "target": "requested drawer interior, tabletop, and floor",
        "color": "requested yellow and wooden appearances",
        "task_scope": (
            "This is a tightly scoped scene restyle. Change only the visible drawer "
            "interior to yellow, the tabletop surface to wood, the floor surface to wood, "
            "and the environment lighting to warm light. Preserve every drink exactly, "
            "including its source color, label, material, geometry, count, position, and "
            "motion. Preserve robots, people, drawer exterior, appliances, and all other "
            "objects exactly. Requested appearances must not bleed across object boundaries."
        ),
        "binding": (
            "Give the drawer interior, tabletop, floor, and each visible drink separate "
            "subjects[] ownership. Bind yellow only to the drawer interior, wood only to "
            "the tabletop and floor, and source colors/materials to every drink. Every "
            "visible robot hand or gripper needs its own preserved subject entry."
        ),
        "scope": (
            "The yellow drawer interior, wooden tabletop, wooden floor, and warm lighting "
            "are intentional requested outcomes. Do not report them as source-fidelity "
            "errors. Drinks must retain their exact source colors and appearances. Audit "
            "all object identities/counts, geometry, pose, contact, motion, timing, "
            "visibility, non-target appearances, and complete robot/person appearance."
        ),
    },
    "G1_WBT_Brainco_Pick_Up_Medicine_v0": {
        "prompt": (
            "帮我将环境改成冷光环境，将地板改成木质地板，将柜子整体改成黄色。"
            "保持机器人和人物外观、物体类别与数量、几何结构、位置、接触关系、"
            "动作、时序及相机运动不变。"
        ),
        "target": "visible cabinet and floor",
        "color": "yellow cabinet and wooden floor appearances",
        "task_scope": (
            "This is a tightly scoped scene restyle. Change only the complete visible "
            "cabinet appearance to yellow, the floor surface to wood, and the environment "
            "lighting to cold light. Preserve medicines and all other objects exactly, "
            "including source colors, labels, materials, identities, counts, geometry, "
            "positions, contacts, and motion. Preserve every robot and person exactly. "
            "Requested appearances must not bleed across object boundaries."
        ),
        "binding": (
            "Give the cabinet, floor, each visible medicine item, and every robot hand or "
            "gripper separate subjects[] ownership. Bind yellow only to the cabinet and "
            "wood only to the floor. Preserve exact source colors and materials on medicine "
            "items, robots, people, and all non-target objects."
        ),
        "scope": (
            "The yellow cabinet, wooden floor, and cold lighting are intentional requested "
            "outcomes. Do not report them as source-fidelity errors. Audit all object "
            "identities/counts, geometry, pose, contact, motion, timing, visibility, "
            "medicine and non-target appearances, and complete robot/person appearance."
        ),
    },
}

ORIGINAL_CONFIG = BASE.config


def config(scene: str, seed: int, experiment: Path) -> dict:
    value = ORIGINAL_CONFIG(scene, seed, experiment)
    case = BASE.CASES[scene]
    value["revision_feedback"]["task_scope"] = case["task_scope"]
    value["revision_feedback"]["binding"] = case["binding"]
    value["revision_feedback"]["length"] = (
        "Hard limit: keep the complete native JSON under 15,000 UTF-8 characters "
        "and under 3,200 model tokens while retaining every required schema key. "
        "Remove repetition and redundant prose, never factual constraints."
    )
    value["source_validation_facts"]["scope"] = case["scope"]
    return value


BASE.config = config


if __name__ == "__main__":
    BASE.main()

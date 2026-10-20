#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE = Path(__file__).with_name("run_new_cases_twice.py")
SPEC = importlib.util.spec_from_file_location("shared_double_runner", MODULE)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

BASE.ROOT = Path("/root/fangweiyi/showcases/10-20-dex1-washing-machine-double")
BASE.CASES = {
    "G1_WBT_Dex1_Put_Clothes_into_Washing_Machine": {
        "prompt": (
            "1. 将衣服颜色调整为黄色。"
            "2. 地板材质替换为木纹质感。"
            "3. 保持多视角下的动作同步和物体交互关系不变。"
        ),
        "target": "clothes",
        "color": "yellow",
    }
}

ORIGINAL_CONFIG = BASE.config


def config(scene: str, seed: int, experiment: Path) -> dict:
    value = ORIGINAL_CONFIG(scene, seed, experiment)
    value["revision_feedback"]["task_scope"] = (
        "This is a tightly scoped restyle. Recolor only the visible clothes to "
        "yellow and replace only the floor appearance with a wood-grain material. "
        "Preserve multi-view action synchronization, object identities and counts, "
        "geometry, positions, contacts, interactions, motion, timing, camera motion, "
        "robots, people, lighting, and every other non-target appearance. Do not let "
        "yellow or wood texture bleed across object boundaries."
    )
    value["source_validation_facts"]["scope"] = (
        "The requested yellow clothes and wood-grain floor are intentional target "
        "outcomes. Lighting is not a requested change and must remain source-faithful. "
        "Audit object identity and count, geometry, pose, contact, interactions, "
        "cross-view action timing, visibility, non-target appearances, robots, and "
        "people."
    )
    return value


BASE.config = config


if __name__ == "__main__":
    BASE.main()

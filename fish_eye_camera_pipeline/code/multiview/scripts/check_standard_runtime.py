#!/usr/bin/env python3
"""Check whether the standard local Cosmos Transfer workflow can run."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def module_status(name: str) -> dict:
    try:
        module = __import__(name, fromlist=["*"])
    except Exception as exc:
        return {"name": name, "available": False, "origin": None, "error": repr(exc)}
    return {"name": name, "available": True, "origin": getattr(module, "__file__", None)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosmos-checkout", type=Path, required=True)
    ap.add_argument("--framework-root", type=Path)
    args = ap.parse_args()

    checkout = args.cosmos_checkout.expanduser().resolve()
    framework = (args.framework_root.expanduser().resolve()
                 if args.framework_root else checkout / "packages" / "cosmos3")
    if framework.is_dir():
        sys.path.insert(0, str(framework))
    transfer = checkout / "cookbooks" / "cosmos3" / "generator" / "transfer"
    files = {
        "cosmos_checkout": checkout,
        "edge_baseline": transfer / "specs" / "edge.json",
        "depth_baseline": transfer / "specs" / "depth.json",
        "negative_prompt": transfer / "assets" / "negative_prompt.json",
    }
    payload = {
        "python": sys.executable,
        "framework_root": str(framework),
        "modules": [
            module_status("cosmos_framework"),
            module_status("cosmos_framework.scripts.inference"),
            module_status("cosmos_control_service"),
        ],
        "files": {
            key: {"path": str(path), "exists": path.exists()}
            for key, path in files.items()
        },
        "commands": {
            "ffmpeg": shutil.which("ffmpeg"),
            "ffprobe": shutil.which("ffprobe"),
        },
    }
    payload["standard_transfer_ready"] = (
        payload["modules"][0]["available"]
        and payload["modules"][1]["available"]
        and files["edge_baseline"].is_file()
        and files["depth_baseline"].is_file()
        and files["negative_prompt"].is_file()
        and payload["commands"]["ffmpeg"] is not None
        and payload["commands"]["ffprobe"] is not None
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["standard_transfer_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

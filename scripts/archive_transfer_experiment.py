#!/usr/bin/env python3
"""Archive an existing Transfer run with the standard showcase runner format."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--started-at-file", type=Path)
    args = parser.parse_args()

    module_spec = importlib.util.spec_from_file_location("showcase_runner", args.runner)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot import runner: {args.runner}")
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    runner.ROOT = args.root.resolve()

    started = time.time()
    if args.started_at_file and args.started_at_file.exists():
        started = float(args.started_at_file.read_text(encoding="utf-8").strip())
    result = runner.archive(args.scene, args.seed, args.experiment.resolve(), started)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

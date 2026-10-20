#!/usr/bin/env python3
"""Config-driven, gated harness for the portable Cosmos per-view Transfer skill."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
HERE = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    config_path = path.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    for key in (
        "cosmos_checkout", "views_root", "experiment_root", "framework_root",
        "transfer_cookbook", "baseline_spec", "negative_prompt_file", "control_root",
        "comparison_font", "transfer_config_file", "python",
    ):
        value = config.get(key)
        if value and key != "python":
            candidate = Path(value).expanduser()
            config[key] = str((base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
        elif value and key == "python" and ("/" in value or value.startswith(".")):
            candidate = Path(value).expanduser()
            config[key] = str((base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
    config["_config_path"] = str(config_path)
    return config


def paths(config: dict) -> tuple[Path, Path, Path, Path, Path, str]:
    missing_keys = [
        key for key in ("cosmos_checkout", "views_root", "experiment_root")
        if not config.get(key)
    ]
    if missing_keys:
        raise SystemExit("Missing required config keys: " + ", ".join(missing_keys))
    checkout = Path(config["cosmos_checkout"])
    views = Path(config["views_root"])
    experiment = Path(config["experiment_root"])
    framework = Path(config.get("framework_root") or checkout / "packages" / "cosmos3")
    cookbook = Path(config.get("transfer_cookbook") or checkout / "cookbooks" / "cosmos3" / "generator" / "transfer")
    for label, path in (
        ("cosmos_checkout", checkout), ("views_root", views),
        ("framework_root", framework), ("transfer_cookbook", cookbook),
    ):
        if not path.is_dir():
            raise SystemExit(f"Configured {label} directory does not exist: {path}")
    python_value = config.get("python") or str(framework / ".venv" / "bin" / "python")
    if "/" in python_value or python_value.startswith("."):
        python_path = Path(python_value)
        if not python_path.is_file() or not os.access(python_path, os.X_OK):
            raise SystemExit(f"Configured Python is not an executable file: {python_path}")
        python = str(python_path)
    else:
        python = shutil.which(python_value)
        if python is None:
            raise SystemExit(f"Configured Python command was not found on PATH: {python_value}")
    return checkout, views, experiment, framework, cookbook, python


def run(command: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def discover(views: Path) -> list[tuple[str, list[Path]]]:
    direct = sorted(p for p in views.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    if direct:
        return [(views.name, direct)]
    return [
        (scene.name, sorted(p for p in scene.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES))
        for scene in sorted(p for p in views.iterdir() if p.is_dir())
        if any(p.is_file() and p.suffix.lower() in SUFFIXES for p in scene.iterdir())
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("stage", choices=(
        "init", "survey", "status", "reasoner", "finalize", "check", "build-specs",
        "transfer", "verify", "compare",
    ))
    args = ap.parse_args()
    config = load_config(args.config)
    checkout, views, experiment, framework, cookbook, python = paths(config)
    prompt_set = config.get("prompt_set", "v1")
    run_name = config.get("run_name", "edge_v1")
    control = config.get("control", "edge")
    prompt_root = experiment / "prompt_sets" / prompt_set

    if args.stage == "init":
        for path in (
            experiment / "survey", prompt_root, experiment / "specs" / run_name,
            experiment / "runs" / run_name, experiment / "evaluations" / run_name,
            experiment / "logs",
        ):
            path.mkdir(parents=True, exist_ok=True)
        scene_index = {}
        for scene, videos in discover(views):
            if args.scene and scene not in args.scene:
                continue
            scene_index[scene] = [video.stem for video in videos]
            scene_dir = prompt_root / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            for video in videos:
                (scene_dir / video.stem).mkdir(parents=True, exist_ok=True)
        if not scene_index:
            raise SystemExit(f"No matching source videos found under {views}")
        index = {
            "config": str(Path(config["_config_path"])), "source_root": str(views),
            "control": control, "prompt_set": prompt_set, "run_name": run_name,
            "seed": config.get("seed", 2026), "no_diffusion_cache": config.get("no_diffusion_cache", True),
            "scenes": scene_index,
        }
        (experiment / "experiment_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        print(f"initialized {len(scene_index)} scenes at {experiment}")
        return

    if args.stage == "survey":
        command = [python, str(HERE / "survey_views.py"), "--views-root", str(views), "--out", str(experiment / "survey")]
        for scene in args.scene:
            command.extend(["--scene", scene])
        run(command)
    elif args.stage == "status":
        counts = {
            "plain": len(list(prompt_root.glob("*/*/plain_prompt.txt"))),
            "upsampled": len(list(prompt_root.glob("*/*/upsampled_prompt.json"))),
            "modified": len(list(prompt_root.glob("*/*/modified_prompt.json"))),
            "audit_reports": len(list(prompt_root.glob("*/*/compliance_report.json"))),
            "specs": len(list((experiment / "specs" / run_name).glob("*.json"))),
            "outputs": len(list((experiment / "runs" / run_name).glob("*/vision.mp4"))),
        }
        print(json.dumps(counts, indent=2))
    elif args.stage == "reasoner":
        command = [python, str(HERE / "run_reasoner.py"), "--prompt-set-dir", str(prompt_root),
                   "--log-dir", str(experiment / "logs" / "reasoner" / prompt_set),
                   "--checkpoint-path", config.get("checkpoint_path", "Cosmos3-Nano")]
        for scene in args.scene:
            command.extend(["--scene", scene])
        run(command, cwd=framework)
    elif args.stage == "finalize":
        command = [python, str(HERE / "finalize_prompts.py"), "--prompt-set-dir", str(prompt_root)]
        for scene in args.scene:
            command.extend(["--scene", scene])
        run(command)
    elif args.stage == "check":
        if args.scene:
            raise SystemExit("check is prompt-set-wide; omit --scene so AUDIT_DONE covers every scene")
        scene_dirs = sorted(p for p in prompt_root.iterdir() if p.is_dir())
        if not scene_dirs:
            raise SystemExit(f"audit/consistency gate failed: no scenes under {prompt_root}")
        failures = 0
        for scene_dir in scene_dirs:
            result = subprocess.run([python, str(HERE / "check_prompts.py"), str(scene_dir), "--require-audit"])
            failures += int(result.returncode != 0)
        if failures:
            raise SystemExit(f"audit/consistency gate failed for {failures} scene(s)")
        (prompt_root / "AUDIT_DONE").write_text(
            "All scene audits and structural checks passed.\n",
            encoding="utf-8",
        )
        (prompt_root / "AUDIT_REQUIRED").unlink(missing_ok=True)
        print("audit/consistency gate PASSED")
    elif args.stage == "build-specs":
        baseline = Path(config.get("baseline_spec") or cookbook / "specs" / f"{control}.json")
        negative = Path(config.get("negative_prompt_file") or cookbook / "assets" / "negative_prompt.json")
        control_root = Path(config.get("control_root") or experiment / "controls")
        command = [python, str(HERE / "build_specs.py"), "--experiment-root", str(experiment),
                   "--views-root", str(views), "--baseline", str(baseline), "--prompt-set", prompt_set,
                   "--run-name", run_name, "--control", control, "--seed", str(config.get("seed", 2026)),
                   "--negative-prompt-file", str(negative), "--control-root", str(control_root)]
        run(command)
    elif args.stage == "transfer":
        transfer_backend = config.get("transfer_backend", "local_cosmos_framework")
        if transfer_backend == "vllm_omni_gateway":
            command = [python, str(HERE / "run_vllm_omni_transfer.py"),
                       "--spec-dir", str(experiment / "specs" / run_name),
                       "--output-dir", str(experiment / "runs" / run_name),
                       "--log-dir", str(experiment / "logs" / "transfer" / run_name),
                       "--model", config.get("checkpoint_path", "Cosmos3-Nano"),
                       "--seed", str(config.get("seed", 2026)),
                       "--gateway-url", config.get("gateway_url", "http://127.0.0.1:8002"),
                       "--upstream-url", config.get("upstream_url", "http://127.0.0.1:8000")]
            if config.get("gateway_poll_interval"):
                command.extend(["--poll-interval", str(config["gateway_poll_interval"])])
            if config.get("gateway_timeout_sec"):
                command.extend(["--timeout-sec", str(config["gateway_timeout_sec"])])
        elif transfer_backend == "local_cosmos_framework":
            command = [python, str(HERE / "run_transfer.py"), "--spec-dir", str(experiment / "specs" / run_name),
                       "--output-dir", str(experiment / "runs" / run_name),
                       "--log-dir", str(experiment / "logs" / "transfer" / run_name),
                       "--checkpoint-path", config.get("checkpoint_path", "Cosmos3-Nano"),
                       "--seed", str(config.get("seed", 2026))]
            if config.get("transfer_config_file"):
                command.extend(["--config-file", config["transfer_config_file"]])
            if config.get("transfer_experiment_overrides"):
                command.append("--experiment-overrides")
                command.extend(str(v) for v in config["transfer_experiment_overrides"])
            if not config.get("no_diffusion_cache", True):
                command.append("--diffusion-cache")
        else:
            raise SystemExit(f"Unsupported transfer_backend: {transfer_backend}")
        for scene in args.scene:
            command.extend(["--scene", scene])
        env = os.environ.copy()
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        run(command, cwd=framework, env=env)
    elif args.stage == "verify":
        run([python, str(HERE / "verify_runs.py"), "--run-root", str(experiment / "runs" / run_name),
             "--views-root", str(views), "--out", str(experiment / "evaluations" / run_name),
             "--control", control])
    elif args.stage == "compare":
        command = [python, str(HERE / "build_comparisons.py"), "--views-root", str(views),
                   "--run-root", str(experiment / "runs" / run_name),
                   "--out", str(experiment / "comparisons" / run_name)]
        for scene in args.scene:
            command.extend(["--scene", scene])
        if config.get("comparison_font"):
            command.extend(["--font", config["comparison_font"]])
        run(command)


if __name__ == "__main__":
    main()

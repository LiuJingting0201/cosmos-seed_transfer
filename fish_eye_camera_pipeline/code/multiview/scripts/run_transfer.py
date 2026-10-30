#!/usr/bin/env python3
"""Run independent Cosmos Transfer specs with one process per available GPU."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading


def detect_gpus(requested: str) -> list[str]:
    ids = [x.strip() for x in requested.split(",") if x.strip()]
    if ids:
        return ids
    visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if visible:
        return visible
    probe = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=True)
    return [str(i) for i, line in enumerate(probe.stdout.splitlines()) if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-path", default="Cosmos3-Nano")
    ap.add_argument("--config-file")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--experiment-overrides", nargs="*", default=[])
    ap.add_argument("--scene", action="append", default=[])
    cache = ap.add_mutually_exclusive_group()
    cache.add_argument("--no-diffusion-cache", dest="no_diffusion_cache", action="store_true")
    cache.add_argument("--diffusion-cache", dest="no_diffusion_cache", action="store_false")
    ap.set_defaults(no_diffusion_cache=True)
    ap.add_argument("--guardrails", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    selected = set(args.scene)
    specs = []
    for spec in sorted(args.spec_dir.glob("*.json")):
        if selected and not any(spec.stem.startswith(f"{scene}__") for scene in selected):
            continue
        if (args.output_dir / spec.stem / "vision.mp4").exists():
            print(f"{spec.stem}: already complete")
        else:
            specs.append(spec)
    if not specs:
        print("No Transfer work pending")
        return

    gpus = detect_gpus(args.gpus)
    if not gpus:
        raise SystemExit("No GPU detected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    work: queue.Queue[Path] = queue.Queue()
    for spec in specs:
        work.put(spec)
    failures: list[str] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                spec = work.get_nowait()
            except queue.Empty:
                return
            tag = spec.stem
            command = [
                sys.executable, "-m", "cosmos_framework.scripts.inference",
                "--parallelism-preset=latency", "-i", str(spec), "-o", str(args.output_dir),
                "--checkpoint-path", args.checkpoint_path, "--seed", str(args.seed),
            ]
            if args.config_file:
                command.extend(["--config-file", args.config_file])
            checkpoint_dir = Path(args.checkpoint_path)
            if checkpoint_dir.is_dir() and (checkpoint_dir / "tokenizer.json").is_file():
                command.extend([
                    "--experiment-overrides",
                    "model.config.vlm_config.tokenizer.config_variant=hf",
                    f"model.config.vlm_config.tokenizer.pretrained_model_name={checkpoint_dir}",
                ])
            if args.experiment_overrides:
                if "--experiment-overrides" not in command:
                    command.append("--experiment-overrides")
                command.extend(args.experiment_overrides)
            if not args.guardrails:
                command.append("--no-guardrails")
            if args.no_diffusion_cache:
                command.append("--no-diffusion-cache")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
            cache_root = args.output_dir.parent.parent / ".cache"
            env["XDG_CACHE_HOME"] = str(cache_root / "xdg" / tag)
            env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor" / tag)
            Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
            Path(env["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            with (args.log_dir / f"{tag}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                with lock:
                    failures.append(tag)
            else:
                print(f"[gpu {gpu}] {tag} complete", flush=True)
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"Transfer failed for {len(failures)} view(s): {', '.join(failures)}")


if __name__ == "__main__":
    main()

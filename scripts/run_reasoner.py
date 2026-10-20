#!/usr/bin/env python3
"""Run the bundled native Reasoner adapter over view prompts with bounded GPU workers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-set-dir", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-path", default="Cosmos3-Nano")
    ap.add_argument("--gpus", default="", help="comma-separated device ids; auto-detect when empty")
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--resolution-w", type=int, default=640)
    ap.add_argument("--resolution-h", type=int, default=480)
    ap.add_argument("--aspect-ratio", default="4,3")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration-secs", type=int, default=10)
    ap.add_argument("--reasoner-seed", type=int, default=3407)
    args = ap.parse_args()

    helper = Path(__file__).with_name("upsample_with_native_reasoner.py")
    selected = set(args.scene)
    pending: list[Path] = []
    for scene_dir in sorted(p for p in args.prompt_set_dir.iterdir() if p.is_dir()):
        if selected and scene_dir.name not in selected:
            continue
        for view_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            if (view_dir / "plain_prompt.txt").exists() and not (view_dir / "upsampled_prompt.json").exists():
                pending.append(view_dir)
    if not pending:
        print("No Reasoner work pending")
        return

    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_ids = [x.strip() for x in visible.split(",") if x.strip()]
    if not gpu_ids:
        probe = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=True)
        gpu_ids = [str(i) for i, line in enumerate(probe.stdout.splitlines()) if line.strip()]
    if not gpu_ids:
        raise SystemExit("No GPU detected")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    work: queue.Queue[Path] = queue.Queue()
    for item in pending:
        work.put(item)
    failures: list[str] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                view_dir = work.get_nowait()
            except queue.Empty:
                return
            tag = f"{view_dir.parent.name}__{view_dir.name}"
            runtime = view_dir / "upsampler_runtime"
            runtime.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(helper),
                "--input", str(view_dir / "plain_prompt.txt"),
                "--output-dir", str(view_dir),
                "--runtime-dir", str(runtime),
                "--checkpoint-path", args.checkpoint_path,
                "--aspect-ratio", args.aspect_ratio,
                "--resolution-w", str(args.resolution_w),
                "--resolution-h", str(args.resolution_h),
                "--fps", str(args.fps),
                "--duration-secs", str(args.duration_secs),
                "--seed", str(args.reasoner_seed),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            # Executing the bundled adapter by path makes its own directory sys.path[0].
            # Add the configured framework working directory so a source checkout can
            # provide cosmos_framework without being installed as a site package.
            env["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(Path.cwd()), env.get("PYTHONPATH", "")) if part
            )
            with (args.log_dir / f"{tag}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                with lock:
                    failures.append(tag)
            else:
                print(f"[gpu {gpu}] {tag} complete", flush=True)
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"Reasoner failed for {len(failures)} view(s): {', '.join(failures)}")


if __name__ == "__main__":
    main()

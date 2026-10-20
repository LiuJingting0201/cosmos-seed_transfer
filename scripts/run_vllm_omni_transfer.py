#!/usr/bin/env python3
"""Run audited Transfer specs through the already-running vLLM-Omni gateway backend."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


BACKEND_NAME = "skill_gated_vllm_omni_gateway"


from prompt_safety import check_budget, check_echo


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command: list[str]) -> None:
    print("+ " + command[0] + " (full arguments retained in request artifacts)", flush=True)
    subprocess.run(command, check=True)


def spec_scene(name: str) -> str:
    return name.split("__", 1)[0] if "__" in name else ""


def discover_specs(spec_dir: Path, scenes: list[str]) -> list[Path]:
    specs = sorted(spec_dir.glob("*.json"))
    if scenes:
        wanted = set(scenes)
        specs = [p for p in specs if spec_scene(p.stem) in wanted]
    if not specs:
        raise SystemExit(f"no specs found under {spec_dir}")
    return specs


def negative_prompt(spec: dict) -> str:
    path = spec.get("negative_prompt_file")
    if not path:
        return ""
    payload = read_json(Path(path))
    return payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)


def submit(args: argparse.Namespace, spec_path: Path, *, preflight_only=False) -> tuple[str, Path] | None:
    spec = read_json(spec_path)
    name = spec.get("name") or spec_path.stem
    chosen = [key for key in ("edge", "seg", "depth", "blur") if spec.get(key)]
    if not chosen:
        raise ValueError("At least one edge, seg, depth, or blur hint is required")
    if "seg" in chosen and len(chosen) > 1:
        raise ValueError("Segmentation control must be submitted alone")
    control = "+".join(chosen)
    source = Path(spec["vision_path"])
    prompt_path = Path(spec["prompt_path"])
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = json.dumps(read_json(prompt_path), ensure_ascii=False, indent=2)
    max_frames = int(spec.get("max_frames") or spec.get("num_frames"))
    extra = {
        "resolution": str(spec.get("resolution", args.resolution)),
        "num_video_frames_per_chunk": int(spec.get("num_video_frames_per_chunk") or max_frames),
        "num_conditional_frames": int(spec.get("num_conditional_frames", 1)),
        "num_first_chunk_conditional_frames": int(spec.get("num_first_chunk_conditional_frames", 0)),
        "share_vision_temporal_positions": bool(spec.get("share_vision_temporal_positions", True)),
        "max_frames": max_frames,
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": False,
        "control_guidance": float(spec.get("control_guidance", 1.5)),
    }
    if "edge" in chosen:
        edge = spec["edge"]
        extra["edge"] = {
            "weight": float(edge.get("weight", args.edge_weight)),
            "preset_edge_threshold": edge.get("preset_edge_threshold", args.edge_threshold),
        }
        if edge.get("control_path"):
            control_path = Path(edge["control_path"])
            if not control_path.is_file():
                raise FileNotFoundError(f"Edge control does not exist: {control_path}")
            extra["edge"]["control_path"] = str(control_path)
    if "seg" in chosen:
        control_path = Path(spec["seg"]["control_path"])
        if not control_path.is_file():
            raise FileNotFoundError(f"Seg control does not exist: {control_path}")
        extra["seg"] = {
            "weight": float(spec["seg"].get("weight", 1.0)),
            "control_path": str(control_path),
        }
    if "depth" in chosen:
        depth = spec["depth"]
        extra["depth"] = {
            "weight": float(depth.get("weight", 1.0)),
        }
        if depth.get("control_path"):
            control_path = Path(depth["control_path"])
            if not control_path.is_file():
                raise FileNotFoundError(f"Depth control does not exist: {control_path}")
            extra["depth"]["control_path"] = str(control_path)
        else:
            extra["depth"]["auto_generate"] = True
            extra["depth"]["max_frames"] = max_frames
    if "blur" in chosen:
        blur = spec["blur"]
        extra["blur"] = {
            "weight": float(blur.get("weight", 1.0)),
            "preset_blur_strength": blur.get("preset_blur_strength", "medium"),
        }
    write_json(run_dir / "transfer_backend.json", {
        "backend": BACKEND_NAME,
        "spec_path": str(spec_path.resolve()),
        "prompt_path": str(prompt_path.resolve()),
        "vision_path": str(source.resolve()),
        "gateway_url": args.gateway_url,
        "upstream_url": args.upstream_url,
        "control": control,
    })
    write_json(run_dir / "params.json", {
        "spec": spec,
        "extra_params": extra,
        "backend": BACKEND_NAME,
    })
    write_json(run_dir / "extra_params.json", extra)
    (run_dir / "cosmos_prompt.txt").write_text(prompt_text + "\n", encoding="utf-8")

    check_budget(run_dir, prompt_text, negative_prompt(spec), extra, source)
    if preflight_only:
        return None

    async_json = run_dir / "async_submit.json"
    if async_json.exists():
        payload = read_json(async_json)
        check_echo(run_dir, payload, prompt_text)
        request_id = payload["id"]
        print(f"{name}: RESUME_REQUEST_ID={request_id}", flush=True)
        return request_id, run_dir
    headers = run_dir / "response.headers"
    run([
        "curl", "--fail-with-body", "--show-error", "--silent", "--max-time", "300",
        "--dump-header", str(headers),
        "-X", "POST", f"{args.gateway_url}/v1/videos",
        "--form-string", f"model={args.model}",
        "--form-string", f"prompt={prompt_text}",
        "--form-string", f"negative_prompt={negative_prompt(spec)}",
        "-F", f"input_reference=@{source};type=video/mp4",
        "--form-string", "input_mime_type=video/mp4",
        "--form-string", f"num_frames={max_frames}",
        "--form-string", f"fps={spec.get('fps', 16.0)}",
        "--form-string", f"num_inference_steps={int(spec.get('num_steps', args.steps))}",
        "--form-string", f"guidance_scale={float(spec.get('guidance', args.guidance_scale))}",
        "--form-string", f"flow_shift={float(spec.get('shift', args.flow_shift))}",
        "--form-string", f"seed={int(spec.get('seed', args.seed))}",
        "--form-string", "generate_sound=false",
        "--form-string", f"extra_params={json.dumps(extra, ensure_ascii=False)}",
        "-o", str(async_json),
    ])
    payload = read_json(async_json)
    check_echo(run_dir, payload, prompt_text)
    request_id = payload["id"]
    print(f"{name}: REQUEST_ID={request_id}", flush=True)
    return request_id, run_dir


def poll(args: argparse.Namespace, jobs: dict[str, tuple[str, Path]]) -> None:
    pending = dict(jobs)
    last_status: dict[str, str] = {}
    start = time.time()
    while pending:
        for name, (request_id, run_dir) in list(pending.items()):
            status_json = run_dir / "status.json"
            run([
                "curl", "--fail-with-body", "--show-error", "--silent", "--max-time", "30",
                f"{args.upstream_url}/v1/videos/{request_id}", "-o", str(status_json),
            ])
            payload = read_json(status_json)
            status = payload.get("status", "")
            if status != last_status.get(name):
                print(f"{name}: STATUS={status}", flush=True)
                last_status[name] = status
            if status == "completed":
                run([
                    "curl", "--fail-with-body", "--show-error", "--silent", "--max-time", "600",
                    f"{args.upstream_url}/v1/videos/{request_id}/content",
                    "-o", str(run_dir / "vision.mp4"),
                ])
                print(f"{name}: downloaded {run_dir / 'vision.mp4'}", flush=True)
                pending.pop(name)
            elif status in {"failed", "error"}:
                raise SystemExit(f"{name} failed: {json.dumps(payload, ensure_ascii=False)}")
        if pending:
            if time.time() - start > args.timeout_sec:
                raise SystemExit(f"timed out waiting for jobs: {sorted(pending)}")
            time.sleep(args.poll_interval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--gateway-url", default="http://127.0.0.1:8002")
    ap.add_argument("--upstream-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="/vemlp-demo-models/models/nvidia/Cosmos3-Nano/Cosmos3-Nano")
    ap.add_argument("--resolution", default="720")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    ap.add_argument("--flow-shift", type=float, default=10.0)
    ap.add_argument("--edge-weight", type=float, default=1.0)
    ap.add_argument("--edge-threshold", default="medium")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--timeout-sec", type=int, default=7200)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    specs = discover_specs(args.spec_dir, args.scene)
    # Validate every view before submitting any GPU generation job.
    for spec in specs:
        submit(args, spec, preflight_only=True)
    if args.preflight_only:
        print(f"preflight passed: {len(specs)} spec(s)", flush=True)
        return
    jobs = {spec.stem: submit(args, spec) for spec in specs}
    poll(args, jobs)


if __name__ == "__main__":
    main()

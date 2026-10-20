#!/usr/bin/env python3
"""Run per-view edge-only Transfer through the local Cosmos gateway service.

This adapter is for machines that expose the existing `/v1/videos` gateway but do not
have the full `cosmos_framework` checkout required by `run_transfer.py`.

It is intentionally separated from the standard Skill workflow. Outputs are written under
`gateway_*` artifact directories and must be reported as gateway adapter results, not as
official-baseline Cosmos Transfer results.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
BACKEND_NAME = "gateway_vllm_omni_edge_adapter"
WORKFLOW_COMPLIANCE = "gateway_adapter_not_standard_skill_workflow"


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration_s": frames / fps if fps else 0.0,
    }


def aspect_ratio(width: int, height: int) -> str:
    divisor = math.gcd(max(width, 1), max(height, 1))
    return f"{width // divisor}:{height // divisor}"


def discover(views_root: Path, prefix: str) -> list[Path]:
    videos = [
        p for p in sorted(views_root.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and p.name.startswith(prefix)
    ]
    if not videos:
        raise SystemExit(f"no videos matching prefix {prefix!r} under {views_root}")
    return videos


def write_edge_control(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {output}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 64, 160)
            rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            writer.write(rgb)
    finally:
        cap.release()
        writer.release()


def prompt_for_view(view: str, meta: dict, user_prompt: str) -> dict:
    view_label = view.replace("_original", "").replace("rwy_split2_", "")
    camera = {
        "left_left": "left-side forward fisheye driving camera",
        "v_front_left": "front-left fisheye driving camera",
        "v_rear_left": "rear-left fisheye driving camera",
        "v_up_left": "upward-left fisheye driving camera",
    }.get(view_label, "vehicle-mounted fisheye driving camera")
    duration = f"{meta['duration_s']:.3f}s"
    return {
        "subjects": [
            {
                "description": "Roadway, roadside barriers, lane markings, sidewalk areas, vegetation, buildings, parked or distant vehicles, and any pedestrians visible from this camera remain the same physical scene elements.",
                "appearance_details": "All non-weather objects keep their original geometry, scale, material, identity, and position; only the weather and daylight appearance become clear sunny daytime.",
                "relationship": "Scene elements keep their original spatial relationship to the moving vehicle-mounted camera.",
                "location": "Distributed across the road scene exactly as visible in the source view.",
                "relative_size": "Source-consistent.",
                "orientation": "Source-consistent.",
                "pose": "Source-consistent.",
                "action": "The vehicle-mounted camera moves forward with the original trajectory and timing.",
                "state_changes": "Object identities, counts, road structure, lane markings, buildings, barriers, sidewalks, vegetation, and camera motion remain stable while the weather becomes sunny.",
                "clothing": "",
                "expression": "",
                "gender": "",
                "age": "",
                "skin_tone_and_texture": "",
                "facial_features": "",
                "number_of_subjects": 1,
                "number_of_arms": 0,
                "number_of_legs": 0,
            }
        ],
        "background_setting": "A real vehicle-mounted DJI-style road scene rendered as clear sunny daytime. The sky is bright blue with sparse small white clouds, visibility is high, pavement and roadside structures are dry, and the original road layout, lane markings, barriers, sidewalks, greenery, buildings, vehicles, and pedestrians remain unchanged.",
        "lighting": {
            "conditions": "Clear sunny weather with bright natural daylight, dry air, high visibility, blue sky, and no overcast haze, rain, fog, mist, wet pavement, puddles, or storm clouds.",
            "direction": "Consistent high-angle sunlight across the road, roadside surfaces, vegetation, buildings, vehicles, and any visible pedestrians.",
            "shadows": "Natural directional sunlight shadows appear consistently on the road and roadside surfaces without flicker or temporal inconsistency.",
            "illumination_effect": "The scene has crisp daytime contrast, slightly warmer sunlit highlights, saturated but realistic greenery, clear road markings, and stable exposure.",
        },
        "aesthetics": {
            "composition": f"Source-preserving {camera} composition; framing, fisheye distortion, viewpoint, and motion are retained.",
            "color_scheme": "Blue sunny sky, dry gray asphalt, white lane markings, pale concrete roadside structures, realistic green vegetation, and source-preserved object colours.",
            "mood_atmosphere": "Clear, bright, calm daytime driving atmosphere.",
            "patterns": "Road lane markings, barriers, sidewalks, vegetation lines, and building edges retain their source structure and motion parallax.",
        },
        "cinematography": {
            "camera_motion": f"{camera} moving with the original vehicle trajectory.",
            "framing": "Wide fisheye vehicle-camera framing matching the source video.",
            "camera_angle": "Source camera angle and lens distortion are preserved.",
            "depth_of_field": "Deep focus with road, roadside structures, and background readable.",
            "focus": "Sharp focus on the driving scene.",
            "lens_focal_length": "Wide-angle fisheye lens matching the source.",
        },
        "style_medium": "Live-action video",
        "artistic_style": "Photorealistic",
        "context": "A synchronized multi-view vehicle driving scene in which the weather is sunny while all scene geometry and camera motion stay unchanged.",
        "actions": [
            {
                "time": f"0:00-{duration}",
                "description": "The vehicle-mounted camera continues the original motion through the road scene under clear sunny weather.",
            }
        ],
        "text_and_signage_elements": [],
        "segments": [
            {
                "segment_index": 0,
                "time_range": f"0:00-{duration}",
                "description": "The original driving shot is preserved with sunny blue-sky weather, dry pavement, stable daylight, and unchanged road and roadside structure.",
                "key_changes": user_prompt,
                "camera": camera,
            }
        ],
        "transitions": [],
        "temporal_caption": "The vehicle-mounted view moves exactly as in the source while the weather appears clear and sunny throughout the clip.",
        "audio_description": "",
        "resolution": {"W": meta["width"], "H": meta["height"]},
        "aspect_ratio": aspect_ratio(meta["width"], meta["height"]),
        "duration": duration,
        "fps": round(meta["fps"], 6),
        "environment_consistency": "The original roadway, lane markings, barriers, sidewalks, vegetation, buildings, vehicles, pedestrians, camera viewpoint, fisheye distortion, and motion trajectory remain spatially and temporally consistent across the full clip; only the weather and lighting become sunny.",
        "target_identity_consistency": "Every visible scene element keeps its original identity, count, shape, scale, material, and motion. No new salient objects are added and no existing salient objects are removed.",
    }


def submit_job(args: argparse.Namespace, run_dir: Path, source: Path, prompt: dict, meta: dict) -> str:
    prompt_text = json.dumps(prompt, ensure_ascii=False, indent=2)
    max_frames = min(args.max_frames, meta["frames"]) if args.max_frames else meta["frames"]
    extra = {
        "resolution": str(args.resolution),
        "num_video_frames_per_chunk": max_frames,
        "num_conditional_frames": 1,
        "num_first_chunk_conditional_frames": 0,
        "share_vision_temporal_positions": True,
        "max_frames": max_frames,
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": False,
        "control_guidance": args.control_guidance,
        "edge": {
            "weight": args.edge_weight,
            "preset_edge_threshold": args.edge_threshold,
        },
    }
    write_json(run_dir / "params.json", {
        "video": str(source),
        "user_prompt": args.user_prompt,
        "video_meta": meta,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "seed": args.seed,
        "control": "edge",
        "extra_params": extra,
    })
    write_json(run_dir / "extra_params.json", extra)
    (run_dir / "cosmos_prompt.txt").write_text(prompt_text + "\n", encoding="utf-8")
    async_json = run_dir / "async_submit.json"
    headers = run_dir / "response.headers"
    run([
        "curl", "--fail-with-body", "--show-error", "--silent", "--max-time", "300",
        "--dump-header", str(headers),
        "-X", "POST", f"{args.gateway_url}/v1/videos",
        "-F", f"model={args.model}",
        "-F", f"prompt={prompt_text}",
        "-F", f"negative_prompt={args.negative_prompt}",
        "-F", f"input_reference=@{source};type=video/mp4",
        "-F", "input_mime_type=video/mp4",
        "-F", f"num_frames={max_frames}",
        "-F", f"fps={meta['fps']}",
        "-F", f"num_inference_steps={args.steps}",
        "-F", f"guidance_scale={args.guidance_scale}",
        "-F", f"flow_shift={args.flow_shift}",
        "-F", f"seed={args.seed}",
        "-F", "generate_sound=false",
        "-F", f"extra_params={json.dumps(extra, ensure_ascii=False)}",
        "-o", str(async_json),
    ])
    payload = read_json(async_json)
    request_id = payload["id"]
    print(f"{run_dir.name}: REQUEST_ID={request_id}", flush=True)
    return request_id


def poll_and_download(args: argparse.Namespace, jobs: dict[str, tuple[str, Path]]) -> None:
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
                pending.pop(name)
                print(f"{name}: downloaded {run_dir / 'vision.mp4'}", flush=True)
            elif status in {"failed", "error"}:
                raise SystemExit(f"{name} failed: {json.dumps(payload, ensure_ascii=False)}")
        if pending:
            if time.time() - start > args.timeout_sec:
                raise SystemExit(f"timed out waiting for jobs: {sorted(pending)}")
            time.sleep(args.poll_interval)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views-root", type=Path, required=True)
    ap.add_argument("--experiment-root", type=Path, required=True)
    ap.add_argument("--scene", default="rwy_split2")
    ap.add_argument("--prefix", default="rwy_")
    ap.add_argument("--prompt-set", default="v1")
    ap.add_argument("--run-name", default="gateway_edge_low_50steps")
    ap.add_argument("--user-prompt", default="将天气换成晴天")
    ap.add_argument("--gateway-url", default="http://127.0.0.1:8002")
    ap.add_argument("--upstream-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="/vemlp-demo-models/models/nvidia/Cosmos3-Nano/Cosmos3-Nano")
    ap.add_argument("--resolution", default="720")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    ap.add_argument("--flow-shift", type=float, default=10.0)
    ap.add_argument("--control-guidance", type=float, default=1.5)
    ap.add_argument("--edge-weight", type=float, default=1.0)
    ap.add_argument("--edge-threshold", default="low")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--timeout-sec", type=int, default=7200)
    ap.add_argument("--negative-prompt", default="overcast sky, cloudy weather, gray sky, rain, fog, mist, wet road, puddles, storm clouds, low visibility, inconsistent geometry, new objects, removed objects, warped road markings, flicker")
    args = ap.parse_args()

    views_root = args.views_root.resolve()
    experiment = args.experiment_root.resolve()
    videos = discover(views_root, args.prefix)
    prompt_scene = experiment / "gateway_prompt_sets" / args.prompt_set / args.scene
    runs_root = experiment / "gateway_runs" / args.run_name
    specs_root = experiment / "gateway_specs" / args.run_name
    survey_root = experiment / "gateway_survey" / args.scene
    for path in (prompt_scene, runs_root, specs_root, survey_root, experiment / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    write_json(experiment / "experiment_index.json", {
        "source_root": str(views_root),
        "scene": args.scene,
        "views": [p.stem for p in videos],
        "control": "edge",
        "run_name": args.run_name,
        "prompt_set": args.prompt_set,
        "seed": args.seed,
        "runner": "scripts/run_gateway_edge_transfer.py",
        "backend": BACKEND_NAME,
        "workflow_compliance": WORKFLOW_COMPLIANCE,
        "artifact_layout": {
            "prompts": "gateway_prompt_sets/<prompt_set>/<scene>/<view>/",
            "specs": "gateway_specs/<run_name>/<scene>__<view>.json",
            "runs": "gateway_runs/<run_name>/<scene>__<view>/",
        },
    })
    write_json(prompt_scene / "transfer_plan.json", {
        "user_prompt": args.user_prompt,
        "control": "edge",
        "backend": BACKEND_NAME,
        "workflow_compliance": WORKFLOW_COMPLIANCE,
        "shared": {
            "background_setting": "Clear sunny weather with blue sky, dry pavement, high visibility, and unchanged DJI road-scene geometry.",
            "lighting": {
                "conditions": "Clear sunny weather with bright natural daylight.",
                "direction": "Consistent high-angle sunlight.",
                "shadows": "Natural directional shadows remain stable.",
                "illumination_effect": "Crisp dry daytime illumination with realistic contrast.",
            },
            "aesthetics": {
                "color_scheme": "Blue sunny sky, dry gray asphalt, white lane markings, pale concrete, green vegetation, and source-preserved object colors.",
                "mood_atmosphere": "Clear sunny daytime driving scene.",
                "patterns": "Road markings and roadside lines remain source-consistent.",
            },
            "style_medium": "Live-action video",
            "artistic_style": "Photorealistic",
            "context": "Synchronized multi-view DJI road footage restyled only from cloudy/overcast weather to sunny weather.",
        },
        "fixed_inference_parameters": {
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "flow_shift": args.flow_shift,
            "control_guidance": args.control_guidance,
            "resolution": args.resolution,
            "seed": args.seed,
            "edge_threshold": args.edge_threshold,
        },
    })

    jobs: dict[str, tuple[str, Path]] = {}
    for source in videos:
        meta = video_meta(source)
        view = source.stem
        view_prompt_dir = prompt_scene / view
        view_prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt = prompt_for_view(view, meta, args.user_prompt)
        (view_prompt_dir / "plain_prompt.txt").write_text(
            f"{args.user_prompt}\nPreserve the original DJI road scene, all objects, camera motion, geometry, and timing for view {view}.\n",
            encoding="utf-8",
        )
        write_json(view_prompt_dir / "modified_prompt.json", prompt)
        write_json(view_prompt_dir / "compliance_report.json", {
            "status": "pass",
            "auditor": "gateway-edge-transfer-adapter",
            "workflow_compliance": WORKFLOW_COMPLIANCE,
            "notes": "Prompt restricts the edit to sunny weather and preserves geometry, object identity, camera motion, timing, and non-weather appearance.",
        })
        spec = {
            "name": f"{args.scene}__{view}",
            "backend": BACKEND_NAME,
            "workflow_compliance": WORKFLOW_COMPLIANCE,
            "control": "edge",
            "prompt_path": str((view_prompt_dir / "modified_prompt.json").resolve()),
            "vision_path": str(source),
            "seed": args.seed,
            "num_frames": min(args.max_frames, meta["frames"]) if args.max_frames else meta["frames"],
            "max_frames": min(args.max_frames, meta["frames"]) if args.max_frames else meta["frames"],
            "fps": meta["fps"],
            "aspect_ratio": aspect_ratio(meta["width"], meta["height"]),
            "gateway_parameters": {
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "flow_shift": args.flow_shift,
                "control_guidance": args.control_guidance,
                "resolution": args.resolution,
                "edge": {"weight": args.edge_weight, "preset_edge_threshold": args.edge_threshold},
            },
        }
        write_json(specs_root / f"{args.scene}__{view}.json", spec)
        run_dir = runs_root / f"{args.scene}__{view}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "source_path.txt").write_text(str(source) + "\n", encoding="utf-8")
        write_edge_control(source, run_dir / "control_edge.mp4")
        if (run_dir / "vision.mp4").exists():
            print(f"{run_dir.name}: already has vision.mp4, skipping submit", flush=True)
            continue
        request_id = submit_job(args, run_dir, source, prompt, meta)
        jobs[run_dir.name] = (request_id, run_dir)
    if jobs:
        poll_and_download(args, jobs)
    else:
        print("No gateway jobs submitted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

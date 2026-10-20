#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time


PIPELINE = Path("/root/fangweiyi/multiview_repair_pipeline")
RUNNER = PIPELINE / "scripts/run_model_pipeline.py"
INPUT_ROOT = Path("/root/sam31/data/宇树数据集/10fps_30s")
ROOT = Path("/root/fangweiyi/showcases/10-10-new-cases-double")
BASELINE = Path(
    "/root/fangweiyi/showcases/full_batch_20261008/configs/edge_704_15steps.json"
)
NEGATIVE = Path(
    "/root/fangweiyi/showcases/full_batch_20261008/configs/"
    "general_restyle_negative_prompt.json"
)
SEEDS = (2026, 2027)
MAX_REPAIRS = 12
RETRY_SECONDS = 120

CASES = {
    "G1_WBT_Brainco_Make_The_Bed": {
        "prompt": (
            "仅将被子表面颜色统一改为鲜明红色，保持被子的材质、数量、形状、尺寸和"
            "位置不变，红色不得溢出到机器人或其他物体。将整体光照调整为暖光环境，"
            "并将地板替换为木质地板。保持机器人和人物外观、物体类别与数量、几何"
            "结构、位置、接触关系、动作、时序及相机运动不变。"
        ),
        "target": "quilt or blanket",
        "color": "vivid red",
    },
    "G1_WBT_Brainco_Pickup_Pillow": {
        "prompt": (
            "仅将枕头表面颜色统一改为鲜明绿色，保持枕头的材质、数量、形状、尺寸和"
            "位置不变，绿色不得溢出到机器人或其他物体。将整体光照调整为暖光环境，"
            "并将地板替换为木质地板。保持机器人和人物外观、物体类别与数量、几何"
            "结构、位置、接触关系、动作、时序及相机运动不变。"
        ),
        "target": "pillow",
        "color": "vivid green",
    },
    "G1_WBT_Dex1_Put_Clothes_into_Washing_Machine": {
        "prompt": (
            "将画面中的衣服改为鲜明黄色，将整体光照改为暖光环境，将地板替换为木质"
            "地板。保持机器人和人物外观、物体类别与数量、几何结构、位置、接触关系、"
            "动作、时序及相机运动不变。"
        ),
        "target": "clothes",
        "color": "vivid yellow",
    },
    "G1_WBT_Inspire_Put_Clothes_Into_Basket": {
        "prompt": (
            "将画面中的枕头改为鲜明红色，将整体光照改为暖光环境，将地板替换为木质"
            "地板。保持机器人和人物外观、物体类别与数量、几何结构、位置、接触关系、"
            "动作、时序及相机运动不变。"
        ),
        "target": "pillow",
        "color": "vivid red",
    },
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def run(command: list[str], log: Path) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write("+ " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode == 0


def stage(config_path: Path, name: str, log: Path) -> bool:
    print(f"{time.strftime('%FT%T')} {config_path.stem}: {name}", flush=True)
    return run(["python3", str(RUNNER), "--config", str(config_path), name], log)


def stage_until_success(config_path: Path, name: str, log: Path) -> None:
    while not stage(config_path, name, log):
        print(
            f"{time.strftime('%FT%T')} {config_path.stem}: {name} technical "
            f"failure; retry in {RETRY_SECONDS}s",
            flush=True,
        )
        time.sleep(RETRY_SECONDS)


def report_versions(experiment: Path) -> dict[str, int]:
    return {
        str(path): path.stat().st_mtime_ns
        for path in experiment.glob("prompt_sets/*/*/*/compliance_report.json")
    }


def audit_until_verdict(config_path: Path, log: Path, experiment: Path) -> bool:
    while True:
        before = report_versions(experiment)
        if stage(config_path, "audit", log):
            return True
        if report_versions(experiment) != before:
            return False
        print(
            f"{time.strftime('%FT%T')} {config_path.stem}: audit technical "
            f"failure; retry in {RETRY_SECONDS}s",
            flush=True,
        )
        time.sleep(RETRY_SECONDS)


def reports_pass(experiment: Path) -> bool:
    reports = list(experiment.glob("prompt_sets/v1/*/*/compliance_report.json"))
    return bool(reports) and all(
        json.loads(path.read_text()).get("status") == "pass" for path in reports
    )


def modified_prompts_complete(experiment: Path, scene: str) -> bool:
    scene_root = experiment / "prompt_sets" / "v1" / scene
    views = list((INPUT_ROOT / scene).glob("*.mp4"))
    prompts = list(scene_root.glob("*/modified_prompt.json"))
    if len(prompts) != len(views):
        return False
    try:
        return all(isinstance(json.loads(path.read_text()), dict) for path in prompts)
    except (OSError, json.JSONDecodeError):
        return False


def rebuild_edit_receipt(
    experiment: Path,
    scene: str,
    config_path: Path,
) -> None:
    scripts = str(PIPELINE / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    model_api = importlib.import_module("model_api")
    receipt = model_api.receipt
    validate_receipt = model_api.validate_receipt

    provenance = experiment / "provenance" / scene
    record = provenance / "edit.json"
    scene_root = experiment / "prompt_sets" / "v1" / scene
    views = sorted(path.stem for path in (INPUT_ROOT / scene).glob("*.mp4"))
    if record.exists():
        existing = validate_receipt(record)
        if len(existing.get("outputs", {})) == len(views):
            return
        record.unlink()

    def matching_calls() -> tuple[list[Path], list[str]]:
        calls = []
        missing = []
        for view in views:
            directory = scene_root / view
            finalized = json.loads((directory / "finalized_prompt.json").read_text())
            modified = json.loads((directory / "modified_prompt.json").read_text())
            candidates = sorted(
                (provenance / "edit" / view).glob("*"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            match = None
            for candidate in candidates:
                parsed_path = candidate / "parsed.json"
                call_path = candidate / "call.json"
                if not parsed_path.is_file() or not call_path.is_file():
                    continue
                parsed = json.loads(parsed_path.read_text())
                call = json.loads(call_path.read_text())
                replacements = parsed.get("replacements")
                if (
                    call.get("status") == "completed"
                    and parsed.get("view") == view
                    and isinstance(replacements, dict)
                    and {**finalized, **replacements} == modified
                ):
                    match = candidate
                    break
            if match is None:
                missing.append(view)
            else:
                calls.append(match)
        return calls, missing

    calls, missing = matching_calls()
    pipeline_module = importlib.import_module("run_model_pipeline")
    while missing:
        print(
            f"{scene}: retry edit only for missing evidence: {', '.join(missing)}",
            flush=True,
        )
        pipeline = pipeline_module.Pipeline(config_path)
        pipeline.scenes = {
            scene: {
                view: pipeline.scenes[scene][view]
                for view in missing
            }
        }
        try:
            pipeline.edit()
        except Exception as exc:
            print(
                f"{scene}: missing-view edit failed ({exc}); "
                f"retry in {RETRY_SECONDS}s",
                flush=True,
            )
            time.sleep(RETRY_SECONDS)
            continue
        record.unlink(missing_ok=True)
        calls, missing = matching_calls()

    outputs = [scene_root / view / "modified_prompt.json" for view in views]
    inputs = [
        *(scene_root / view / "finalized_prompt.json" for view in views),
        scene_root / "transfer_plan.json",
        config_path,
        PIPELINE / "references/prompt-schema.md",
        PIPELINE / "SKILL.md",
        experiment / "survey/view_metadata.json",
    ]
    for view in views:
        inputs.extend([
            INPUT_ROOT / scene / f"{view}.mp4",
            experiment / "survey" / scene / f"{view}.jpg",
        ])
    receipt(record, inputs, outputs, calls)
    validate_receipt(record)
    print(f"{scene}: rebuilt and validated edit receipt", flush=True)


def config(scene: str, seed: int, experiment: Path) -> dict:
    case = CASES[scene]
    target = case["target"]
    color = case["color"]
    return {
        "cosmos_checkout": "/root/sam31/cosmos",
        "views_root": str(INPUT_ROOT / scene),
        "experiment_root": str(experiment),
        "user_prompt": case["prompt"],
        "prompt_set": "v1",
        "run_name": f"seed_{seed}",
        "control": "edge",
        "seed": seed,
        "python": "python3",
        "checkpoint_path": (
            "/vemlp-demo-models/models/nvidia/Cosmos3-Nano/Cosmos3-Nano"
        ),
        "reasoner_model": "cosmos3-nano-reasoner",
        "reasoner_url": "http://127.0.0.1:8001/v1/chat/completions",
        "reasoner_max_tokens": 12000,
        "transfer_backend": "vllm_omni_gateway",
        "gateway_url": "http://127.0.0.1:8002",
        "upstream_url": "http://127.0.0.1:8000",
        "gateway_timeout_sec": 21600,
        "gateway_poll_interval": 20,
        "no_diffusion_cache": False,
        "editor_provider": "ark",
        "editor_model": "doubao-seed-evolving",
        "editor_url": "https://ark.cn-beijing.volces.com/api/v3/responses",
        "editor_api_key_env": "ARK_API_KEY",
        "editor_max_tokens": 16000,
        "baseline_spec": str(BASELINE),
        "negative_prompt_file": str(NEGATIVE),
        "revision_feedback": {
            "task_scope": (
                f"This is a tightly scoped restyle. Recolor only the visible {target} "
                f"to {color}. Preserve its material, count, geometry, size, position, "
                "contact, and motion. The target color must not bleed onto robots, "
                "people, or any other object. Use warm lighting and replace only the "
                "floor appearance with a wooden floor. Preserve every robot and person "
                "exactly, plus all object identities, geometry, contacts, motion, timing, "
                "and camera motion."
            ),
            "media": (
                "Every synchronized view is exactly 30 seconds at integer 10 fps. "
                "All actions, segments, and temporal captions must cover 0:00-0:30."
            ),
            "multiview": (
                "Files are four synchronized views of one scene. Contact-sheet columns "
                "are successive times, never a grid inside the source video. Trailing "
                "black contact-sheet cells are padding."
            ),
            "binding": (
                f"Use a concrete singular class key for each visible {target} and copy "
                f"the exact phrase '{color}' byte-for-byte into that subject's own "
                "subjects[] entry. Every visible robot hand or gripper needs a dedicated "
                "subjects[] entry carrying its exact source color and material. Explicitly "
                "own all non-target objects so the requested color cannot bleed onto them."
            ),
            "visibility": (
                "Describe only objects and robot or person parts actually visible in each "
                "camera. Do not infer hidden limbs, labels, contact, completed actions, or "
                "exact orientation from cropped regions."
            ),
            "length": (
                "Keep the complete native JSON concise while retaining every schema key."
            ),
        },
        "source_validation_facts": {
            "media": (
                "All four views are synchronized, 30s, 10 fps, H.264/yuv420p."
            ),
            "scope": (
                f"The requested {color} {target}, warm lighting, and wooden floor are "
                "intentional target outcomes and must not be reported as source-fidelity "
                "errors. Audit source fidelity for object identity and count, geometry, "
                "pose, contact, motion, timing, visibility, all non-target appearances, "
                "and the complete appearance of robots and people."
            ),
        },
    }


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def horizontal(inputs: list[Path], output: Path) -> None:
    tile_w = 1920 // len(inputs)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for path in inputs:
        command.extend(["-i", str(path)])
    filters = []
    labels = []
    for index in range(len(inputs)):
        label = f"v{index}"
        filters.append(
            f"[{index}:v]scale={tile_w}:360:force_original_aspect_ratio=decrease,"
            f"pad={tile_w}:360:(ow-iw)/2:(oh-ih)/2:black[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append("".join(labels) + f"hstack=inputs={len(inputs)}[out]")
    output.parent.mkdir(parents=True, exist_ok=True)
    command.extend([
        "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", str(output),
    ])
    subprocess.run(command, check=True)


def compare(source: Path, generated: Path, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-i", str(generated),
            "-filter_complex", "[0:v][1:v]vstack=inputs=2[out]",
            "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-shortest", str(output),
        ],
        check=True,
    )


def archive(scene: str, seed: int, experiment: Path, started: float) -> dict:
    source_dir = INPUT_ROOT / scene
    views = sorted(path.stem for path in source_dir.glob("*.mp4"))
    sources = [source_dir / f"{view}.mp4" for view in views]
    outputs = [
        experiment / "runs" / f"seed_{seed}" / f"{scene}__{view}" / "vision.mp4"
        for view in views
    ]
    media = []
    for source, output in zip(sources, outputs):
        if not output.is_file():
            raise RuntimeError(f"missing output: {output}")
        source_info = probe(source)
        output_info = probe(output)
        if int(source_info["nb_frames"]) != int(output_info["nb_frames"]):
            raise RuntimeError(f"frame count mismatch: {output}")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(output),
             "-f", "null", "-"],
            check=True,
            stdin=subprocess.DEVNULL,
        )
        media.append(output_info)

    out = ROOT / "archive" / scene / f"seed_{seed}"
    source_row = ROOT / "archive" / scene / f"{scene}_source_horizontal.mp4"
    generated_row = out / f"{scene}_generated_seed_{seed}.mp4"
    comparison = out / f"{scene}_comparison_seed_{seed}.mp4"
    if not source_row.exists():
        horizontal(sources, source_row)
    horizontal(outputs, generated_row)
    compare(source_row, generated_row, comparison)
    evidence = out / f"{scene}_evidence_seed_{seed}.tar.gz"
    subprocess.run(
        [
            "tar", "--exclude=vision.mp4", "-czf", str(evidence),
            "-C", str(experiment), "prompt_sets", "provenance", "specs", "logs",
        ],
        check=True,
    )
    result = {
        "scene": scene,
        "seed": seed,
        "prompt": CASES[scene]["prompt"],
        "views": views,
        "started_at": started,
        "completed_at": time.time(),
        "source_horizontal": str(source_row),
        "generated_horizontal": str(generated_row),
        "comparison": str(comparison),
        "evidence": str(evidence),
        "experiment": str(experiment),
        "media": media,
        "status": "generated_and_technically_valid_visual_review_pending",
    }
    result["elapsed_seconds"] = result["completed_at"] - started
    write_json(out / "result.json", result)
    write_json(ROOT / "publish_queue" / f"{scene}__seed_{seed}.json", result)
    return result


def process(scene: str, seed: int) -> dict:
    result_path = ROOT / "archive" / scene / f"seed_{seed}" / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    started = time.time()
    experiment = ROOT / "experiments" / scene / f"seed_{seed}"
    config_path = ROOT / "configs" / f"{scene}__seed_{seed}.json"
    log = ROOT / "logs" / f"{scene}__seed_{seed}.log"
    status = ROOT / "state" / f"{scene}__seed_{seed}.json"
    if not config_path.exists():
        write_json(config_path, config(scene, seed, experiment))
    write_json(status, {
        "scene": scene, "seed": seed, "status": "starting",
        "started_at": started,
    })

    if not experiment.exists():
        for name in ("init", "survey", "author", "reasoner", "finalize"):
            stage_until_success(config_path, name, log)
        if not stage(config_path, "edit", log):
            rebuild_edit_receipt(experiment, scene, config_path)
        prompt_ready = audit_until_verdict(config_path, log, experiment)
    else:
        prompt_ready = reports_pass(experiment)
        if not prompt_ready and modified_prompts_complete(experiment, scene):
            print(
                f"{scene} seed {seed}: resume from four valid modified prompts",
                flush=True,
            )
            rebuild_edit_receipt(experiment, scene, config_path)
            prompt_ready = audit_until_verdict(config_path, log, experiment)

    for attempt in range(1, MAX_REPAIRS + 1):
        if prompt_ready:
            break
        write_json(status, {
            "scene": scene, "seed": seed, "status": "repair",
            "repair_attempt": attempt, "started_at": started,
        })
        stage_until_success(config_path, "repair", log)
        prompt_ready = audit_until_verdict(config_path, log, experiment)
    if not prompt_ready:
        raise RuntimeError(
            f"{scene} seed {seed}: audit did not pass after {MAX_REPAIRS} repairs"
        )

    for name in ("check", "build-specs"):
        if not stage(config_path, name, log):
            raise RuntimeError(f"{scene} seed {seed}: {name} failed")
    write_json(status, {
        "scene": scene, "seed": seed, "status": "transfer",
        "started_at": started,
    })
    stage_until_success(config_path, "transfer", log)
    result = archive(scene, seed, experiment, started)
    write_json(status, {
        "scene": scene, "seed": seed, "status": "complete", "result": result,
    })
    print(f"{scene} seed {seed}: COMPLETE", flush=True)
    return result


def main() -> None:
    if not os.environ.get("ARK_API_KEY"):
        raise SystemExit("ARK_API_KEY is required")
    ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for scene in CASES:
        for seed in SEEDS:
            results.append(process(scene, seed))
    write_json(ROOT / "COMPLETE.json", {"results": results})
    print("ALL NEW CASES COMPLETE", flush=True)


if __name__ == "__main__":
    main()

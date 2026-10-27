#!/usr/bin/env python3
"""Run one bounded Cosmos prefix preview from a JSON configuration."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, **kwargs)


def video_meta(path: Path, *, count_frames: bool = True) -> dict[str, Any]:
    entries = "stream=width,height,avg_frame_rate,nb_frames,duration"
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        command += ["-count_frames"]
        entries += ",nb_read_frames"
    command += [
        "-show_entries",
        entries,
        "-of",
        "json",
        str(path),
    ]
    output = subprocess.check_output(
        command
    )
    streams = json.loads(output)["streams"]
    if not streams:
        raise ValueError(f"No video stream: {path}")
    return streams[0]


def require_path(config: dict[str, Any], key: str) -> Path:
    path = Path(str(config.get(key, ""))).expanduser()
    if not path.is_file():
        raise ValueError(f"{key} is not a file: {path}")
    return path.resolve()


def validate(config: dict[str, Any], frames: int) -> None:
    required = ("name", "source", "prompt", "output_root", "controls")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(config["name"])):
        raise ValueError("name must contain only letters, digits, dot, underscore, or hyphen")
    if frames < 5 or (frames - 1) % 4:
        raise ValueError("--frames must be at least 5 and satisfy frames = 4k + 1")
    chunk_frames = int(config.get("chunk_frames", 93))
    if frames > chunk_frames:
        raise ValueError(
            f"Prefix preview is limited to one chunk ({chunk_frames} frames); got {frames}"
        )
    source = require_path(config, "source")
    require_path(config, "prompt")
    source_meta = video_meta(source, count_frames=False)
    source_frames_value = source_meta.get("nb_frames")
    if source_frames_value and source_frames_value != "N/A":
        source_frames = int(source_frames_value)
    else:
        source_frames = round(
            float(source_meta["duration"]) * float(Fraction(source_meta["avg_frame_rate"]))
        )
    if frames > source_frames:
        raise ValueError(f"Requested {frames} frames but source has {source_frames}")
    controls = config["controls"]
    if not isinstance(controls, dict) or not controls:
        raise ValueError("controls must be a non-empty JSON object")
    for name, control in controls.items():
        if name not in {"edge", "seg", "depth", "blur"}:
            raise ValueError(f"Unsupported control {name!r}")
        if not isinstance(control, dict):
            raise ValueError(f"Control {name!r} must be an object")
        weight = float(control.get("weight", -1))
        if not 0 <= weight <= 2:
            raise ValueError(f"Control {name!r} weight must be in [0, 2]")
        if control.get("control_path"):
            path = Path(str(control["control_path"])).expanduser()
            if not path.is_file():
                raise ValueError(f"Control {name!r} path is not a file: {path}")


def clip_video(source: Path, destination: Path, frames: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and int(video_meta(destination)["nb_read_frames"]) == frames:
        return
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ]
    )
    actual = int(video_meta(destination)["nb_read_frames"])
    if actual != frames:
        raise RuntimeError(f"Clipped {destination} has {actual} frames, expected {frames}")


def prompt_text(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return json.dumps(load(path), ensure_ascii=False, indent=2)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt is empty: {path}")
    return text


def prepare_controls(
    controls: dict[str, dict[str, Any]], inputs: Path, frames: int
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    prepared: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name, original in controls.items():
        control = dict(original)
        source_value = control.get("control_path")
        if source_value:
            source = Path(str(source_value)).expanduser().resolve()
            destination = inputs / f"{name}.mp4"
            clip_video(source, destination, frames)
            control["control_path"] = str(destination)
            hashes[name] = sha256(destination)
        prepared[name] = control
    return prepared, hashes


def build_request(
    config: dict[str, Any],
    source: Path,
    controls: dict[str, dict[str, Any]],
    frames: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    fps = int(config.get("fps", 30))
    chunk_frames = int(config.get("chunk_frames", 93))
    guidance = float(config.get("guidance_scale", 3.0))
    flow_shift = float(config.get("flow_shift", 10.0))
    extra: dict[str, Any] = {
        "resolution": str(config.get("resolution", "720")),
        "fps": fps,
        "num_frames": frames,
        "max_frames": frames,
        "num_video_frames_per_chunk": chunk_frames,
        "num_conditional_frames": int(config.get("conditional_frames", 1)),
        "num_first_chunk_conditional_frames": int(
            config.get("first_chunk_conditional_frames", 0)
        ),
        "share_vision_temporal_positions": True,
        "use_resolution_template": False,
        "use_duration_template": False,
        "control_guidance": float(config.get("control_guidance", 1.5)),
        "guidance_scale": guidance,
        "flow_shift": flow_shift,
        "emphasize_control_in_prompt": bool(
            config.get("emphasize_control_in_prompt", False)
        ),
        **controls,
    }
    fields = {
        "model": str(config.get(
            "model", "/vemlp-demo-models/models/nvidia/Cosmos3-Nano/Cosmos3-Nano"
        )),
        "prompt": prompt_text(require_path(config, "prompt")),
        "negative_prompt": str(config.get("negative_prompt", "")),
        "size": str(config.get("size", "1280x720")),
        "fps": str(fps),
        "num_frames": str(frames),
        "num_inference_steps": str(int(config.get("num_inference_steps", 15))),
        "guidance_scale": str(guidance),
        "flow_shift": str(flow_shift),
        "seed": str(int(config.get("seed", 2026))),
        "generate_sound": "false",
        "input_mime_type": "video/mp4",
        "extra_params": json.dumps(extra),
    }
    return fields, extra


def create_review(source: Path, generated: Path, output_dir: Path) -> None:
    review = output_dir / "review_source_left_generated_right.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-i",
            str(generated),
            "-filter_complex",
            "[0:v]scale=640:360,setsar=1[left];"
            "[1:v]scale=640:360,setsar=1[right];"
            "[left][right]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(review),
        ]
    )
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(review),
            "-vf",
            "fps=5/3,scale=640:-1,tile=3x2",
            "-frames:v",
            "1",
            str(output_dir / "review_contact_sheet.jpg"),
        ]
    )


def preview(config_path: Path, frames: int, force: bool) -> Path:
    config = load(config_path)
    validate(config, frames)
    output_dir = (
        Path(str(config["output_root"])).expanduser().resolve()
        / "previews"
        / str(config["name"])
        / f"prefix_{frames:04d}"
    )
    complete = output_dir / "complete.json"
    request = output_dir / "request.json"
    if (
        request.exists()
        and not force
        and load(request).get("config_sha256") != sha256(config_path)
    ):
        raise ValueError(
            f"Config changed for existing preview {output_dir}; rerun with --force"
        )
    if complete.exists() and not force:
        print(f"ALREADY_COMPLETE {output_dir}", flush=True)
        return output_dir
    if force and output_dir.exists():
        shutil.rmtree(output_dir)

    inputs = output_dir / "inputs"
    source = inputs / "source.mp4"
    clip_video(require_path(config, "source"), source, frames)
    controls, control_hashes = prepare_controls(config["controls"], inputs, frames)
    fields, extra = build_request(config, source, controls, frames)
    gateway = str(config.get("gateway", "http://127.0.0.1:8002")).rstrip("/")
    upstream = str(config.get("upstream", "http://127.0.0.1:8000")).rstrip("/")
    save(
        output_dir / "request.json",
        {
            "config": str(config_path.resolve()),
            "config_sha256": sha256(config_path),
            "method": "POST",
            "url": f"{gateway}/v1/videos",
            "form_fields": fields,
            "input_sha256": {"source": sha256(source), **control_hashes},
            "extra_params": extra,
            "bounded_preview": {"frames": frames, "max_frames": int(config.get("chunk_frames", 93))},
        },
    )

    lock_path = Path(str(config.get(
        "gpu_lock", "/root/sam31/cosmos-full-suite-20260917/gpu-admission.lock"
    )))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        minimum_free_gib = int(config.get("minimum_free_gib", 10))
        if shutil.disk_usage(output_dir).free < minimum_free_gib * 1024**3:
            raise RuntimeError(f"Less than {minimum_free_gib} GiB free at {output_dir}")
        submit_path = output_dir / "async_submit.json"
        if not submit_path.exists():
            command = [
                "curl", "--fail-with-body", "-sS", "--max-time", "600",
                "-X", "POST", f"{gateway}/v1/videos",
            ]
            for key, value in fields.items():
                command += ["--form-string", f"{key}={value}"]
            command += [
                "-F", f"input_reference=@{source};type=video/mp4",
                "-o", str(submit_path),
            ]
            run(command)
        job_id = load(submit_path)["id"]
        started = time.time()
        while True:
            run(
                [
                    "curl", "--fail-with-body", "-sS", "--max-time", "60",
                    f"{upstream}/v1/videos/{job_id}", "-o", str(output_dir / "status.json"),
                ]
            )
            status = load(output_dir / "status.json")
            if status["status"] == "completed":
                break
            if status["status"] in {"failed", "error"}:
                raise RuntimeError(status)
            time.sleep(25)
        generated = output_dir / "native.mp4"
        run(
            [
                "curl", "--fail-with-body", "-sS", "--max-time", "1800",
                f"{upstream}/v1/videos/{job_id}/content", "-o", str(generated),
            ]
        )
        run(["ffmpeg", "-v", "error", "-i", str(generated), "-f", "null", "-"])
        metadata = video_meta(generated)
        if int(metadata["nb_read_frames"]) != frames:
            raise RuntimeError(
                f"Generated video has {metadata['nb_read_frames']} frames, expected {frames}"
            )
        create_review(source, generated, output_dir)
        save(
            complete,
            {
                "status": "completed",
                "job_id": job_id,
                "frames": frames,
                "elapsed_seconds": time.time() - started,
                "generated_sha256": sha256(generated),
                "metadata": metadata,
                "review_video": str(output_dir / "review_source_left_generated_right.mp4"),
                "review_contact_sheet": str(output_dir / "review_contact_sheet.jpg"),
                "full_run_started": False,
            },
        )
    print(f"COMPLETE {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a bounded one-chunk Cosmos prefix preview. Never starts a full run."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate config without submitting a job"
    )
    args = parser.parse_args()
    config = load(args.config)
    frames = args.frames or int(config.get("preview_frames", 93))
    validate(config, frames)
    if args.validate_only:
        print(f"VALID name={config['name']} frames={frames}", flush=True)
        return
    preview(args.config, frames, args.force)


if __name__ == "__main__":
    main()

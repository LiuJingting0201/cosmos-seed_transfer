#!/usr/bin/env python3
"""Verify downloaded API videos independently of unavailable control exports."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from harness import load_config, discover
from model_api import read_json, write_json, digest
from verify_runs import inspect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["experiment_root"])
    run = config.get("run_name", "edge_v1")
    out = root / "evaluations" / run
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for scene, videos in discover(Path(config["views_root"])):
        for source in videos:
            name = f"{scene}__{source.stem}"
            directory = root / "runs" / run / name
            entry = {"video_ok": False, "full_control_verification": False}
            try:
                spec = read_json(root / "specs" / run / f"{name}.json")
                status = read_json(directory / "status.json")
                source_info, _ = inspect(source)
                picks = {0, source_info["frames"] // 2, source_info["frames"] - 1}
                source_info, source_frames = inspect(source, picks)
                output_info, frames = inspect(directory / "vision.mp4", picks)
                control = directory / "control_edge.mp4"
                entry.update(
                    source=source_info, output=output_info, status=status.get("status"),
                    request_id=status.get("id"), spec_resolution_tier=spec["resolution"],
                    source_dimensions_preserved=(
                        source_info["width"] == output_info["width"]
                        and source_info["height"] == output_info["height"]
                    ),
                    control_export_present=control.exists(),
                    output_sha256=digest(directory / "vision.mp4"),
                    note="Control alignment is unverified; semantic quality requires visual inspection.",
                )
                expected_prompt = (directory / 'cosmos_prompt.txt').read_text().removesuffix('\n')
                budget = read_json(directory / 'prompt_token_audit.json')
                entry['prompt_transport_exact'] = status.get('prompt') == expected_prompt
                import hashlib
                entry['prompt_token_budget_verified'] = (budget.get('safe') is True and
                    budget['counts']['positive']['raw_sha256'] == hashlib.sha256(expected_prompt.encode()).hexdigest())
                entry["video_ok"] = (
                    entry['prompt_transport_exact'] and entry['prompt_token_budget_verified'] and
                    status.get("status") == "completed"
                    and source_info["frames"] == output_info["frames"] == spec["num_frames"]
                    and abs(output_info["fps"] - spec["fps"]) < 0.01
                    and abs(output_info["duration_s"] - source_info["duration_s"]) < 0.01
                    and output_info["width"] > 0 and output_info["height"] > 0
                    and not output_info["tail_frozen"]
                    and picks == frames.keys()
                )
                # Only review images are resized; the delivered MP4 remains untouched.
                rows = []
                for sampled in (source_frames, frames):
                    rows.append(np.concatenate([
                        np.asarray(Image.fromarray(sampled[i]).resize((320, 320)))
                        for i in sorted(picks)
                    ], axis=1))
                Image.fromarray(np.concatenate(rows, axis=0)).save(out / f"{name}.jpg")
            except Exception as exc:
                entry["error"] = str(exc)
            report[name] = entry
    passed = sum(x["video_ok"] for x in report.values())
    write_json(out / "api_video_verification.json", {
        "video_pass_count": passed, "expected_count": len(report),
        "full_control_verification": False, "views": report,
    })
    print(json.dumps({"video_pass_count": passed, "expected_count": len(report),
                      "report": str(out / "api_video_verification.json")}))
    raise SystemExit(0 if report and passed == len(report) else 1)


if __name__ == "__main__":
    main()

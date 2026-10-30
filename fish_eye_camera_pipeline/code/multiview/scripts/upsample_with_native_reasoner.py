#!/usr/bin/env python3
"""Run Cosmos3-Nano's native Reasoner prompt upsampler and preserve every stage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cosmos_framework.inference.structured_caption import parse_structured_caption


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", default="Cosmos3-Nano")
    parser.add_argument("--aspect-ratio", default="4,3")
    parser.add_argument("--resolution-w", type=int, default=640)
    parser.add_argument("--resolution-h", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration-secs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plain_prompt = args.input.read_text(encoding="utf-8").strip()
    if not plain_prompt:
        raise ValueError(f"Plain prompt is empty: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    input_json = args.runtime_dir / "reasoner_input.json"
    input_json.write_text(
        json.dumps(
            {
                "model_mode": "reasoner",
                "prompt": plain_prompt,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "top_k": 20,
                "top_p": 0.8,
                "repetition_penalty": 1.0,
                "presence_penalty": 1.5,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "cosmos_framework.scripts.inference",
        "--parallelism-preset=latency",
        "-i",
        str(input_json),
        "-o",
        str(args.runtime_dir),
        "--checkpoint-path",
        args.checkpoint_path,
        "--seed",
        str(args.seed),
        "--no-guardrails",
        "--no-use-torch-compile",
        "--no-use-cuda-graphs",
    ]
    subprocess.run(command, check=True)

    outputs = sorted(args.runtime_dir.rglob("sample_outputs.json"))
    if len(outputs) != 1:
        raise RuntimeError(f"expected one sample_outputs.json under {args.runtime_dir}, found {outputs}")
    content = json.loads(outputs[0].read_text(encoding="utf-8"))["outputs"][0]["content"]
    raw_response = content.get("reasoner_text")
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise RuntimeError(f"missing reasoner_text in {outputs[0]}")
    parsed = parse_structured_caption(raw_response)

    (args.output_dir / "upsampler_raw_response.txt").write_text(raw_response + "\n", encoding="utf-8")
    if parsed is None:
        raise ValueError("The native Reasoner response did not contain a valid JSON object.")
    (args.output_dir / "upsampled_prompt.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

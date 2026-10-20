#!/usr/bin/env python3
"""Track both wrist-camera gripper jaws with SAM2.1 video memory."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor


MODEL = "/vemlp-demo-models/models/facebook/facebook/sam2.1-hiera-large"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok or (args.max_frames and len(frames) >= args.max_frames):
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError("No source frames decoded")
    height, width = frames[0].shape[:2]
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

    processor = Sam2VideoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = Sam2VideoModel.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    session = processor.init_video_session(
        video=pil_frames,
        inference_device="cuda",
        processing_device="cuda",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    input_points = [[
        [
            [int(width * 0.11), int(height * 0.79)],
            [int(width * 0.20), int(height * 0.88)],
            [int(width * 0.31), int(height * 0.61)],
            [int(width * 0.04), int(height * 0.50)],
            [int(width * 0.39), int(height * 0.90)],
        ],
        [
            [int(width * 0.88), int(height * 0.79)],
            [int(width * 0.94), int(height * 0.88)],
            [int(width * 0.70), int(height * 0.61)],
            [int(width * 0.97), int(height * 0.48)],
            [int(width * 0.59), int(height * 0.90)],
        ],
    ]]
    input_labels = [[[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]]
    processor.add_inputs_to_inference_session(
        session, frame_idx=0, obj_ids=[1, 2],
        input_points=input_points, input_labels=input_labels,
    )

    masks: dict[int, np.ndarray] = {}
    scores: dict[int, list[float]] = {}
    with torch.inference_mode():
        iterator = model.propagate_in_video_iterator(
            session, start_frame_idx=0, max_frame_num_to_track=len(frames),
            show_progress_bar=True,
        )
        for output in iterator:
            processed = processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[(height, width)],
                binarize=True,
            )[0]
            array = processed.detach().cpu().numpy().astype(bool).reshape(-1, height, width)
            index = int(output.frame_idx)
            tracked = np.any(array, axis=0)
            frame = frames[index]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            channel_range = (
                frame.max(axis=2).astype(np.int16) - frame.min(axis=2).astype(np.int16)
            )
            material = (gray < 90) & (channel_range < 75)
            selected = tracked & material
            selected = cv2.dilate(
                selected.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ).astype(bool) & tracked
            masks[index] = selected
            scores[index] = (
                output.object_score_logits.detach().cpu().float().numpy().reshape(-1).tolist()
            )
    if sorted(masks) != list(range(len(frames))):
        raise RuntimeError(f"Incomplete tracking: {len(masks)} of {len(frames)} frames")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
        "-c:v", "ffv1", str(args.output),
    ], stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    for index in range(len(frames)):
        encoder.stdin.write((masks[index] * 255).astype(np.uint8).tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("Mask encoding failed")

    indexes = sorted(set(range(0, len(frames), 30)) | {len(frames) - 1})
    tiles = []
    for index in indexes:
        overlay = frames[index].copy()
        overlay[masks[index]] = (
            overlay[masks[index]] * 0.45 + np.array([0, 255, 0]) * 0.55
        ).astype(np.uint8)
        tiles.append(np.hstack([
            cv2.resize(frames[index], (320, 240)),
            cv2.resize(overlay, (320, 240)),
        ]))
    rows = []
    for offset in range(0, len(tiles), 2):
        row = tiles[offset:offset + 2]
        if len(row) == 1:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(str(args.contact), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])

    payload = {
        "source": str(args.source.resolve()),
        "model": MODEL,
        "method": "SAM2.1 video tracking from two first-frame wrist-gripper boxes",
        "frames": len(frames),
        "fps": fps,
        "seed_points_xy": input_points[0],
        "seed_point_labels": input_labels[0],
        "empty_frames": sum(not mask.any() for mask in masks.values()),
        "mean_area_fraction": float(np.mean([mask.mean() for mask in masks.values()])),
        "object_score_logits": scores,
    }
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "object_score_logits"}, indent=2))


if __name__ == "__main__":
    main()

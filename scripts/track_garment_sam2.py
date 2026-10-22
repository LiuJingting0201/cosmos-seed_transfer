#!/usr/bin/env python3
"""Track one garment with SAM2.1 from an auditable point-prompt seed."""

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


def point(value: str) -> tuple[int, int]:
    try:
        x_text, y_text = value.split(",", 1)
        return int(x_text), int(y_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected X,Y") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude-mask", type=Path)
    parser.add_argument("--seed-frame", type=int, required=True)
    parser.add_argument("--positive", type=point, action="append", required=True)
    parser.add_argument("--negative", type=point, action="append", default=[])
    parser.add_argument("--processing-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--min-object-score", type=float, default=0.0)
    parser.add_argument("--min-area-fraction", type=float, default=0.0002)
    parser.add_argument("--max-area-fraction", type=float, default=0.9)
    parser.add_argument("--exclude-dilate", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.source))
    exclude_capture = (
        cv2.VideoCapture(str(args.exclude_mask)) if args.exclude_mask else None
    )
    if not capture.isOpened() or (exclude_capture and not exclude_capture.isOpened()):
        raise RuntimeError("Failed to open source or exclude mask")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    excludes: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
        if exclude_capture:
            exclude_ok, exclude_frame = exclude_capture.read()
            if not exclude_ok:
                raise RuntimeError(f"Exclude-mask decode failed at frame {len(frames) - 1}")
            excludes.append(exclude_frame)
    capture.release()
    if exclude_capture:
        exclude_capture.release()
    if not frames:
        raise RuntimeError("No source frames decoded")
    if not 0 <= args.seed_frame < len(frames):
        raise ValueError("--seed-frame is outside the video")

    height, width = frames[0].shape[:2]
    all_points = args.positive + args.negative
    if any(not (0 <= x < width and 0 <= y < height) for x, y in all_points):
        raise ValueError("One or more prompt points are outside the source frame")
    points = [[[list(value) for value in all_points]]]
    labels = [[[1] * len(args.positive) + [0] * len(args.negative)]]
    pil_frames = [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames
    ]

    processor = Sam2VideoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = Sam2VideoModel.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    session = processor.init_video_session(
        video=pil_frames,
        inference_device="cuda",
        processing_device=args.processing_device,
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    processor.add_inputs_to_inference_session(
        session,
        frame_idx=args.seed_frame,
        obj_ids=[1],
        input_points=points,
        input_labels=labels,
    )

    raw_masks: dict[int, np.ndarray] = {}
    scores: dict[int, float] = {}
    with torch.inference_mode():
        directions = [
            (args.seed_frame, len(frames) - args.seed_frame, False),
            (args.seed_frame - 1, args.seed_frame, True),
        ]
        for start, count, reverse in directions:
            if count <= 0:
                continue
            iterator = model.propagate_in_video_iterator(
                session,
                start_frame_idx=start,
                max_frame_num_to_track=count,
                reverse=reverse,
                show_progress_bar=True,
            )
            for output in iterator:
                processed = processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=[(height, width)],
                    binarize=True,
                )[0]
                index = int(output.frame_idx)
                raw_masks[index] = (
                    processed.detach().cpu().numpy().astype(bool).reshape(
                        -1, height, width
                    )[0]
                )
                scores[index] = float(
                    output.object_score_logits.detach().cpu().float().numpy().reshape(-1)[0]
                )
    if sorted(raw_masks) != list(range(len(frames))):
        raise RuntimeError(f"Incomplete tracking: {len(raw_masks)} of {len(frames)} frames")

    exclude_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * args.exclude_dilate + 1, 2 * args.exclude_dilate + 1),
    )
    masks: dict[int, np.ndarray] = {}
    gate_reasons: dict[int, str] = {}
    for index, raw_mask in raw_masks.items():
        mask = raw_mask.copy()
        if excludes:
            excluded = cv2.resize(
                (np.max(excludes[index], axis=2) > 32).astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            mask[cv2.dilate(excluded, exclude_kernel) > 0] = False
        area = float(mask.mean())
        if scores[index] < args.min_object_score:
            mask[:] = False
            gate_reasons[index] = "low_object_score"
        elif not args.min_area_fraction <= area <= args.max_area_fraction:
            mask[:] = False
            gate_reasons[index] = "area_out_of_range"
        masks[index] = mask

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
            "-c:v", "ffv1", str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    if encoder.stdin is None:
        raise RuntimeError("Failed to open FFmpeg stdin")
    for index in range(len(frames)):
        encoder.stdin.write((masks[index] * 255).astype(np.uint8).tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("Mask encoding failed")

    indexes = np.linspace(0, len(frames) - 1, 12).astype(int).tolist()
    tiles = []
    for index in indexes:
        overlay = frames[index].copy()
        overlay[masks[index]] = np.rint(
            overlay[masks[index]] * 0.4 + np.array([0, 255, 0]) * 0.6
        ).astype(np.uint8)
        status = gate_reasons.get(index, f"score {scores[index]:.2f}")
        cv2.putText(
            overlay, f"frame {index} {status}", (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        tiles.append(cv2.resize(overlay, (320, 240)))
    contact_sheet = np.vstack(
        [np.hstack(tiles[offset:offset + 4]) for offset in range(0, 12, 4)]
    )
    args.contact.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.contact), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])

    areas = [float(masks[index].mean()) for index in range(len(frames))]
    payload = {
        "source": str(args.source.resolve()),
        "model": MODEL,
        "method": "SAM2.1 single-garment bidirectional point-prompt tracking",
        "frames": len(frames),
        "fps": fps,
        "resolution": [width, height],
        "seed_frame": args.seed_frame,
        "processing_device": args.processing_device,
        "seed_points_xy": all_points,
        "seed_point_labels": labels[0][0],
        "confidence_gate": {
            "min_object_score": args.min_object_score,
            "area_fraction": [args.min_area_fraction, args.max_area_fraction],
        },
        "empty_frames": sum(not masks[index].any() for index in range(len(frames))),
        "mean_area_fraction": float(np.mean(areas)),
        "gate_reasons": gate_reasons,
        "object_score_logits": scores,
        "output": str(args.output.resolve()),
        "contact": str(args.contact.resolve()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {key: value for key, value in payload.items() if key != "object_score_logits"},
        indent=2,
    ))


if __name__ == "__main__":
    main()

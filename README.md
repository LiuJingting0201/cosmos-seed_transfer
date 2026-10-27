# Multiview Cosmos Transfer Pipeline

Current as of **2026-10-27 12:02:41 +0800**.

This repository is a reproducible snapshot and experiment toolkit for
multi-camera video-to-video editing with NVIDIA Cosmos3 Transfer. It started as
the gated per-view pipeline used for the Unitree Dex1 washing-machine
experiments and now also contains the follow-up tools developed to diagnose and
reduce shadow hallucination, robot drift, garment identity swaps, temporal
color drift, and the cost of testing a full video before checking its visual
direction.

The central engineering rule is:

> Models describe the requested semantic change and generate candidates;
> deterministic code validates media, preserves provenance, and applies exact
> pixel operations where generative consistency is insufficient.

## What Is Current

The repository contains four related layers. They should not be treated as one
monolithic pipeline.

1. **Gated multiview prompt and Transfer pipeline**

   Surveys every view, authors and expands structured prompts, audits
   cross-view consistency, builds specs from an official baseline, runs each
   camera independently, and verifies the generated media.
2. **Transfer backends**

   Supports the standard local `cosmos_framework` runtime and an already
   running vLLM-Omni/Cosmos gateway. The standalone gateway smoke-test adapter
   remains intentionally separate from the audited standard artifact tree.
3. **Dex1 investigation and repair tools**

   Reproducible Edge/Depth ablations, SAM-based robot and garment masks,
   source-pixel compositing, deterministic Lab chroma, operation-bound garment
   identities, and review builders.
4. **Bounded prefix preview**

   Runs at most one Cosmos chunk so a prompt/control design can be reviewed
   before committing GPU time to a complete video.

Model weights, credentials, source videos, generated videos, and experiment
outputs are deliberately excluded from Git.

## Core Architecture

The full model-assisted pipeline is:

```text
init
  -> survey
  -> author
  -> reasoner
  -> finalize
  -> edit
  -> audit
  -> repair (when needed)
  -> check
  -> build-specs
  -> transfer
  -> verify
  -> compare
```

### Prompt and provenance gates

- `survey` samples the complete videos rather than reasoning from filenames or
  a few opening frames.
- `author` creates one structured prompt per view while maintaining a shared
  scene identity block.
- the native Cosmos Reasoner expands per-view prompts;
- `finalize` normalizes the schema and injects byte-identical shared fields;
- `audit` and `repair` use fresh visual evidence and retain model call records;
- `check` enforces schema, object/color binding, shared fields, and passing
  compliance reports;
- `build-specs` copies fixed inference settings from the selected baseline and
  writes a hash-bound `provenance/READY.json`;
- `transfer` verifies those hashes before submitting generation.

`scripts/run_model_pipeline.py` is the end-to-end orchestrator.
`scripts/harness.py` exposes the lower-level deterministic stages.

### Multiview strategy

Each synchronized camera is generated as an independent Cosmos job. Inputs are
never tiled into a montage before inference because that reduces per-view pixel
budget and does not create true cross-view consistency. Consistency is instead
enforced through:

- shared scene, lighting, material, robot, and end-effector descriptions;
- per-view visibility and camera descriptions;
- one seed across all views;
- explicit persistent object identities and operation intervals;
- post-run cross-view review.

A shared seed is not a shared latent trajectory. Cross-view agreement remains a
semantic constraint and must be reviewed separately from codec/media validity.

## Quick Start

### 1. Configure the runtime

Create a runtime JSON based on `configs/dex1_seed_2026.runtime.json` and update
its machine-specific paths:

```json
{
  "cosmos_checkout": "/path/to/cosmos",
  "views_root": "/path/to/scene-or-scenes",
  "experiment_root": "/new/unique/experiment",
  "user_prompt": "Describe only the intended edit",
  "control": "edge",
  "seed": 2026,
  "transfer_backend": "local_cosmos_framework"
}
```

Experiment roots are append-only evidence directories. Use a new path for a
new run instead of overwriting an earlier result.

### 2. Check the local Cosmos runtime

```bash
python3 scripts/check_standard_runtime.py \
  --cosmos-checkout /path/to/cosmos
```

The standard backend requires the Cosmos3 checkout, model weights, CUDA, and
the checkout's Python environment. Media tooling uses `av`, `imageio`, `numpy`,
`Pillow`, OpenCV, `ffmpeg`, and `ffprobe`. SAM-based repair scripts additionally
require their corresponding PyTorch model runtime.

### 3. Build an audited prompt/spec set

```bash
python3 scripts/run_model_pipeline.py \
  --config /path/to/runtime.json \
  prompt
```

This executes through `build-specs` and stops before video generation. Review
the prompt artifacts and `provenance/READY.json` before continuing.

### 4. Generate and verify

```bash
python3 scripts/run_model_pipeline.py --config /path/to/runtime.json transfer
python3 scripts/run_model_pipeline.py --config /path/to/runtime.json verify
python3 scripts/run_model_pipeline.py --config /path/to/runtime.json compare
```

For manual stage-by-stage operation, artifact layout, and configuration keys,
see [`references/portable-runtime.md`](references/portable-runtime.md).

## Prefix Preview

`scripts/run_prefix_preview.py` is the current preflight entry point for a
gateway-served Cosmos3 pipeline. It:

- clips the source and every file-backed control to a prefix;
- accepts Edge, Depth, Segmentation, or Blur control dictionaries;
- submits exactly one bounded job;
- records the request, config hash, input hashes, server status, output hash,
  metadata, and elapsed time;
- produces a source/generated side-by-side MP4 and a contact sheet;
- never starts a full-video run.

The preview frame count must satisfy `frames = 4k + 1` and cannot exceed
`chunk_frames` (93 by default).

```bash
# Validate paths and bounds without submitting.
python3 scripts/run_prefix_preview.py \
  --config configs/popcorn_red_cup_edge_blur_prefix.json \
  --validate-only

# Run the configured one-chunk preview.
python3 scripts/run_prefix_preview.py \
  --config configs/popcorn_red_cup_edge_blur_prefix.json
```

Outputs are written to:

```text
<output_root>/previews/<name>/prefix_<frames>/
|-- inputs/
|-- request.json
|-- async_submit.json
|-- status.json
|-- native.mp4
|-- review_source_left_generated_right.mp4
|-- review_contact_sheet.jpg
`-- complete.json
```

`configs/popcorn_red_cup_edge_blur_prefix.json` captures the 2026-10-27
development-machine experiment that used Edge `0.9` plus Blur `0.1` to recolor
a popcorn cup. It is an exact runtime record with machine-specific paths, not a
portable dataset config.

## Backends

### Standard local Cosmos

`scripts/run_transfer.py` invokes:

```text
python -m cosmos_framework.scripts.inference
```

Specs inherit resolution, sampling, guidance, shift, chunking, and control
settings from the chosen official baseline. This is the reference path for the
portable skill.

### Audited vLLM-Omni gateway

When `transfer_backend` is `vllm_omni_gateway`,
`scripts/run_vllm_omni_transfer.py` consumes the standard audited specs and
writes standard `runs/` artifacts. It supports gateway controls exposed as
Edge, Segmentation, Depth, and Blur, validates every spec before submitting any
job, and stores request parameters and server responses.

### Standalone gateway adapter

`scripts/run_gateway_edge_transfer.py` is a feasibility/smoke-test adapter. It
bypasses the complete author/Reasoner/audit/spec gate and therefore writes to
`gateway_prompt_sets/`, `gateway_specs/`, and `gateway_runs/`. Do not report its
outputs as standard gated-pipeline results. See
[`references/gateway-vllm-omni.md`](references/gateway-vllm-omni.md).

## Experiment Evolution

| Date | Commit | Change |
| --- | --- | --- |
| 2026-10-20 | `bc6e6ba` | Initial reproducible snapshot: multiview survey, structured prompt pipeline, provenance gates, fixed baseline spec, local/gateway Transfer, verification, and batch runners. |
| 2026-10-20 | `c02e0e1` | Added A-E gripper-shadow ablation and stricter vLLM-Omni submission records. |
| 2026-10-20 | `6b01a03` | Added full four-view Edge+Depth experiment preparation and experiment archival. |
| 2026-10-20 | `a35b84b` | Documented the Dex1 shadow diagnosis and observed tradeoffs. |
| 2026-10-20 | `1725f1c` | Added SAM/optical-flow gripper masks and source-pixel compositing to preserve robot pixels exactly. |
| 2026-10-21 | `811ce2e` | Added second-garment tracking and generated-region recoloring to repair identity transitions. |
| 2026-10-21 | `fa00c6d` | Tightened black-garment identity preservation in wrist views. |
| 2026-10-21 | `e749e40` | Added persistent garment IDs and a multiview identity finalization workflow. |
| 2026-10-21 | `6436bcc` | Added P1/P2/P3 prompt ablations and synchronized review matrices. |
| 2026-10-22 | `d5a9bf3` | Added color-anchor/SAM garment masks and deterministic Lab chroma while preserving generated luminance. |
| 2026-10-22 | `678501b` | Added causal garment event intervals and mask padding constrained by grasp/release state. |
| 2026-10-22 | `2c68917` | Added operation-bound A/B/C identities and head-to-wrist appearance binding for four-view Edge+Depth runs. |
| 2026-10-27 | working tree | Added the bounded, config-driven prefix preview used before full-video generation. |

The investigation details are in
[`docs/dex1_shadow_ablation.md`](docs/dex1_shadow_ablation.md) and
[`docs/gripper_preservation.md`](docs/gripper_preservation.md).

## Repair Tools

The repair utilities are explicit post-generation experiments, not hidden parts
of the main generator:

- `prepare_shadow_ablation.py`: reproducible prompt/control variants for the
  gripper-shadow hallucination;
- `prepare_multiview_edge_depth.py`: full four-view Edge+Depth follow-up;
- `build_gripper_preservation_mask.py` and
  `track_wrist_grippers_sam2.py`: robot/gripper preservation masks;
- `composite_preserved_region.py`: copy source RGB inside reviewed masks;
- `prepare_multiview_garment_identity.py`: persistent garment identity specs;
- `prepare_garment_prompt_ablation.py`: controlled prompt variants;
- `track_garment_sam2.py`, `build_color_anchor_garment_mask.py`, and
  `track_second_garment_sam2.py`: garment mask generation/tracking;
- `apply_deterministic_chroma.py`: Lab-space chroma replacement that preserves
  generated luminance;
- `pad_interval_mask.py`: causal interval padding without crossing operation
  boundaries;
- `prepare_crossview_edge_depth_ablation.py`: operation-bound cross-view
  identity prompts.

Every pixel-level output should retain input hashes, media metadata, mask
coverage, and output hashes. Masks require visual review before compositing or
recoloring.

## Repository Map

```text
configs/               Baseline, runtime, event, and prefix-preview configs
docs/                  Experiment findings and repair notes
references/            Runtime, schema, backend, and Cosmos implementation notes
runners/               Historical batch orchestration
scripts/               Main pipeline, backend adapters, ablations, and repairs
tokenizer_snapshot/    Frozen tokenizer/VAE configuration for prompt accounting
SKILL.md               Operational rules for per-view Cosmos Transfer
prompt_budget.py       Prompt/token budget audit helper
```

## Known Limits

- Prompt and pre-generation audits cannot detect hallucinations visible only in
  generated frames. Post-generation review remains mandatory.
- Edge/Depth/Segmentation controls constrain generation but do not freeze RGB
  pixels. Exact preservation requires reviewed masks and compositing.
- Independent per-view generation cannot guarantee geometric cross-view
  consistency.
- A one-chunk prefix validates only that view and time range. It does not prove
  full-sequence temporal stability or later object visibility.
- Deterministic chroma and source-pixel compositing are 2D postprocesses; they
  do not add 3D consistency or recover an incorrect generated geometry.
- Several checked-in configs intentionally record development-machine absolute
  paths. Update them for another host and never commit credentials.

## Safety and Data Policy

- Source media is read-only.
- Use unique output roots for experiments and retries.
- Do not commit API keys, `.env` files, model weights, source datasets,
  generated videos, logs, or runtime state.
- Keep semantic success, cross-view success, and technical media validity as
  separate evaluation results.

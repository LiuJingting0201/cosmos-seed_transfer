# Cosmos3 checkout adapter

This skill no longer assumes a checkout or dataset location. Supply them in the harness
configuration as `cosmos_checkout`, `views_root`, and `experiment_root`.

The checkout is expected to contain:

```text
<cosmos_checkout>/
|-- packages/cosmos3/
`-- cookbooks/cosmos3/generator/transfer/
    |-- specs/edge.json
    |-- specs/depth.json
    `-- assets/negative_prompt.json
```

The framework Python environment must import `cosmos_framework`; normally it is the
checkout's `packages/cosmos3/.venv`. The harness discovers this layout from the configured
checkout and never embeds the host path in the skill.

For a nonstandard checkout, set these optional config keys:

- `framework_root`: directory from which `python -m cosmos_framework...` is run.
- `transfer_cookbook`: directory holding official specs and assets.
- `python`: interpreter with Cosmos and the media-tool dependencies installed.
- `control_root`: directory containing precomputed depth control videos.

## Spec construction

Copy the official spec for the chosen modality. The builder may change only the sample
name, prompt and input paths, output location, control hint/path, seed, and media-derived
frame count, FPS, and aspect ratio. It must preserve guidance, control guidance, shift,
step count, sampler, resolution tier, and chunking.

Keep `emphasize_control_in_prompt` false. Enabling it adds an instruction that every
visible structure align with the control signal, which fights appearance transfer.

Edge uses the official preset threshold and computes its control from `vision_path`.
Depth uses one precomputed `control_path` per view. Never place both hints in one spec.

## Prompt finalization

The framework passes the entire prompt JSON to the generator. It does not remove
`scene_imagination`, despite the Reasoner prompt assigning that responsibility to the
deployment layer. Remove the field while writing `modified_prompt.json`.

## Seed behavior

Set the seed both in the spec and on the inference command. The framework derives chunk
seeds from it and uses a device-local generator, so concurrently running views do not
perturb each other's random state. A shared seed does not itself enforce cross-view
agreement because every input and control map follows a different denoising trajectory.

## Evaluation

Technical checks:

- process exit status is zero;
- frame count, duration, dimensions, and FPS match the source/spec;
- the tail is present, decodable, and not frozen.

Visual checks:

- silhouettes, articulated geometry, trajectory, timing, and camera motion remain intact;
- background/material and lighting changes remain visible throughout;
- target colours are unmistakable, persistent, and do not leak;
- robot, gripper, cables, fixtures, and all preserved objects keep their real appearance;
- the views agree on background, lighting, robot, end effector, and target object colours.

When comparing prompt versions, keep the source video and all fixed inference parameters
identical. Report semantic and cross-view failures even when media validation passes.

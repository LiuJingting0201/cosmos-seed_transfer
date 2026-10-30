---
name: cosmos-transfer-per-view
description: Run Cosmos3 video Transfer over separate synchronized camera videos, generating each view as its own job and keeping the restyle consistent across views. Use for multi-camera scenes, robot high-angle and wrist views, single videos, and already-existing montage inputs.
---

# Cosmos Transfer — per-view generation

Generate every source view as its own Transfer sample. Do not create a montage before
inference: it divides the pixel budget and does not guarantee cross-view consistency. An
already-existing montage is one input view; a review montage may be created after Transfer.

## Scope comes from the user

Change what the user asked to change, and nothing else.

- If the user named specific changes, do only those. An unrequested recolour or material
  change corrupts the comparison even when it looks attractive.
- If the user named no target, design the background material, lighting treatment,
  per-object colour mapping, and preserved elements, then ask before running.
- Skip that confirmation only when the user delegated the design or approved an
  unattended batch. State the chosen design in the handoff.

Everything outside the requested change still needs an accurate description so it
survives unchanged. Restricted scope means fewer changes, not a thinner prompt.

## Invariants

- Preserve object identity and count, geometry, proportions, pose, motion trajectory,
  action order and timing, and camera viewpoint and motion.
- Do not add or remove salient objects or redesign robots, tools, hands, or articulated
  parts.
- Source media is read-only. Write every experiment and retry to a unique output path.
- Run only when the user requested execution. Use one candidate per view by default and
  do not launch expensive semantic retries without authorization.

## Backend boundary

The standard Skill workflow uses the local Cosmos Transfer environment:
`scripts/harness.py` builds official-baseline specs and `scripts/run_transfer.py` calls
`python -m cosmos_framework.scripts.inference`.

The gateway adapter `scripts/run_gateway_edge_transfer.py` is a separate vLLM-Omni
smoke-test backend for servers that expose `/v1/videos` but do not have
`cosmos_framework` available. Its outputs must be labelled as gateway adapter results,
not as standard Skill results, and it writes to `gateway_prompt_sets/`, `gateway_specs/`,
and `gateway_runs/` instead of the standard artifact directories.

Use the gateway adapter only when the user accepts that backend explicitly. Read
[references/gateway-vllm-omni.md](references/gateway-vllm-omni.md) before running it.

If the Skill gates have already produced audited `modified_prompt.json` files and
official-baseline `specs/`, `scripts/harness.py` may run the final transfer stage through
the already-running vLLM-Omni gateway by setting `transfer_backend:
vllm_omni_gateway`. This consumes standard `specs/` and writes standard `runs/`, while
recording `backend: skill_gated_vllm_omni_gateway` in each run directory. It is distinct
from the ungated smoke-test adapter above.

## Inference parameters are fixed

- Copy the selected control modality's official spec. Never tune `guidance`,
  `control_guidance`, `shift`, `num_steps`, sampler, resolution tier, or chunking to
  influence appearance.
- Only `name`, prompt path, input path, output location, control hint/path, seed, and
  source-derived frame count, FPS, and aspect ratio may differ from the baseline.
- Keep `emphasize_control_in_prompt` false for appearance Transfer.
- Use one seed for every view of a scene. A shared seed fixes initial noise only; the
  different inputs and control maps diverge immediately, so prompt identity is the actual
  cross-view consistency mechanism.
- When diffusion cache must be disabled, pass `--no-diffusion-cache`. The bundled runner
  and harness disable it by default.

## Control modality

Use exactly one modality for a scene: `edge` or `depth`, never both, and use it for every
view. Ask when unspecified; default to `edge`. Edge control is computed from the input;
depth requires one precomputed control video per view.

## Prompt schema and field split

Read [references/prompt-schema.md](references/prompt-schema.md) before authoring prompts.

Write the shared block once and copy these strings byte-for-byte into every view:

- `background_setting`, `context`, `style_medium`, `artistic_style`
- all four `lighting` fields
- `aesthetics.color_scheme`, `mood_atmosphere`, and `patterns`
- the robot subject's `description` and `appearance_details`
- the dedicated gripper or hand subject's `description` and `appearance_details`

These may differ per view: cinematography, composition, visible `subjects[]`, subject
location/size/orientation/pose/action/state changes, actions, segments, and temporal
caption.

## Read the footage before writing

Sample roughly one frame per second across each entire clip. Record per view:

- background surfaces, lighting, every foreground object and its count;
- each visible object's left-to-right order;
- action direction, acting arm, motion, and whether the final action remains incomplete;
- camera motion and the real appearance of every element that must not change.

The Reasoner is text-only and cannot recover visual facts missing from the plain prompt.
Do not claim an occluded or off-frame object is visible.

Per-view left-to-right ordering is the binding between same-shaped objects and target
colours. Keep this ordering in per-view fields only. View-relative phrases in a shared
field are invalid because that text is broadcast to every camera.

## Isolate authoring and audit by scene

For a large batch, use one isolated authoring context per scene that covers all views of
that scene. When subagent execution is available, use one subagent per scene, not one per
view, so the same author owns the shared block.

After Reasoner expansion and finalization, use a fresh isolated audit context per scene.
The audit edits `modified_prompt.json` in place; do not re-run Reasoner merely to apply a
correction. Re-run it only when the design itself changes.

Tell the auditor that source-to-target colour changes are intentional. It must check:

- the robot and end effector's real materials and colours;
- action direction, acting arm, and actions falsely described as completed;
- object counts and identities;
- significant unowned regions or invented objects;
- camera motion, media metadata, and per-view ordering.

An audit reporting no findings should still demonstrate that it inspected the frames and
prompt files; silence alone is not evidence of a clean scene.

Record a passing or failing `compliance_report.json` for each view. A scene that fails is
not ready to run.

## Head words bind colours

The structural checker assigns a subject to a colour class using the class key's final
word. Before prompt generation, verify that:

- a class head word does not occur in an unrelated preserved object, robot, gripper, or
  another class description;
- each class noun contains its own expected head word;
- target colours are distinct across source-colour groups of one class;
- every visible subject in the class carries exactly one target colour in its own
  `subjects[]` entry.

This prevents an unrelated phrase such as “gripper housing blocks” from being interpreted
as an uncoloured member of a `block` class.

## Mandatory consistency check

After all `modified_prompt.json` files exist and before building specs, run:

```bash
python3 <skill-dir>/scripts/check_prompts.py <scene-prompt-dir> --require-audit
```

The check asserts the shared fields, robot and dedicated end-effector entries, colour
binding, target-colour uniqueness, schema completeness, audit reports, and absence of
view-relative phrases in shared fields.

## Workflow

1. Create a unique experiment and survey every source view.
2. Save one `transfer_plan.json` per scene and one `plain_prompt.txt` per view.
3. Run the native Cosmos Reasoner per view, preserving raw and parsed responses.
4. Finalize prompts: remove `scene_imagination`, normalize schema/media fields, and inject
   the shared block verbatim.
5. Audit each scene and edit `modified_prompt.json` directly.
6. Run the mandatory consistency check; do not build specs for a failing scene.
7. Build one spec per view from the official baseline and inspect its allowed diff.
8. Run every view independently with bounded GPU concurrency.
9. Verify process status, codec-readable frame count, dimensions, FPS, duration, control
   alignment, and non-frozen tail; inspect beginning, middle, and end frames.
10. Report technical success, semantic adherence, and cross-view agreement separately.

For configuration, bundled scripts, dependencies, and artifact layout, read
[references/portable-runtime.md](references/portable-runtime.md). For Cosmos spec
construction, media handling, and evaluation details, read
[references/local-cosmos3.md](references/local-cosmos3.md).

## Iteration policy

Diagnose weak output before requesting a retry:

- robot recoloured: its real appearance is vague or absent from the colour scheme;
- gripper or hand drifted: it lacks a dedicated shared subject entry or foreground pin;
- requested change is weak: the shared block is too thin;
- views disagree: a shared string drifted or view-relative text entered the shared block;
- an object is unpainted or colours swap: ordering is missing, binding failed, or same-
  class target hues are too close.

Do not compensate by tuning fixed inference parameters. Report partial semantic or
cross-view failure honestly even when the media files are technically valid.

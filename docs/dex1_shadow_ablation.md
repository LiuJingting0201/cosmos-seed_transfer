# Dex1 Gripper-Shadow Ablation

This experiment investigates a failure in
`G1_WBT_Dex1_Put_Clothes_into_Washing_Machine/wrist_left__file-000`: a hard
gripper shadow on a garment can be regenerated as a second physical gripper-like
structure.

## Fixed settings

- Source range: first 80 frames at 10 FPS
- Seed: `2026`
- Resolution setting: `704`
- Inference steps: `15`
- Guidance: `3.0`
- Control guidance: `1.5`
- Chunk length: `80`
- Target edit: yellow garments and wood-grain floor

## Variants

| ID | Name | Prompt and negative prompt | Control |
| --- | --- | --- | --- |
| A | `A_baseline_edge_medium` | Original audited prompt and negative prompt | Edge, Canny `100/200`, weight `1.0` |
| B | `B_shadow_prompt_negative_edge_medium` | Adds flat-shadow identity constraints and shadow-object negatives | Edge, Canny `100/200`, weight `1.0` |
| C | `C_shadow_prompt_negative_edge_high` | Same as B | Edge, Canny `200/300`, weight `1.0` |
| D | `D_shadow_prompt_negative_cleaned_edge` | Same as B | Precomputed Canny `100/200`; suppresses edges inside an eroded red-garment mask while preserving the lower gripper region |
| E | `E_alt_shadow_prompt_edge_depth` | Same as B | Edge weight `0.6` plus automatically generated Depth weight `0.4` |

The originally proposed Edge+Depth+Seg variant was not submitted because no
reviewed segmentation control existed. E is the executable Edge+Depth
alternative supported by the deployed gateway.

## Initial observation

C gave the best balance in the sampled frames: the weaker shadow edges were
filtered while strong object boundaries and garment motion remained relatively
stable. B improved semantic interpretation but retained the complete hard edge.
D removed the problematic internal edge more aggressively, but also removed
useful garment structure and increased shape drift. E reduced reliance on Edge
by adding relative-depth control, but the medium Edge hint still retained the
gripper-shaped shadow.

These are visual observations, not an automated cross-view or contact-fidelity
score. Full videos require temporal review before selecting a production
configuration.

## Reproduction

Prepare the five short variants:

```bash
python3 scripts/prepare_shadow_ablation.py \
  --source-video /path/to/wrist_left__file-000.mp4 \
  --source-prompt /path/to/repaired_prompt.json \
  --source-spec /path/to/wrist_left_spec.json \
  --source-negative /path/to/general_restyle_negative_prompt.json \
  --output-root /path/to/ablation
```

Run each generated spec directory with
`scripts/run_vllm_omni_transfer.py`. The launcher validates every spec before
submission and records request parameters and server responses in the run
directory.

The full four-view Edge+Depth follow-up is prepared by
`scripts/prepare_multiview_edge_depth.py`. It applies the same Edge/Depth
weights, shadow-aware negative prompt, and view-independent shadow identity
rule to all four views while preserving each view's audited camera description.

Generated videos, source videos, model weights, and credentials are excluded
from Git.

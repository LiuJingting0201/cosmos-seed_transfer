# Multiview Cosmos Pipeline Snapshot

Snapshot date: 2026-10-20

This repository freezes the multiview Cosmos pipeline used for the Unitree
washing-machine experiments. It includes the prompt pipeline, provenance
gates, batch runner, baseline Transfer spec, negative prompt, and an exact
launcher for the Dex1 experiment.

## Runtime flow

`init -> survey -> author -> reasoner -> finalize -> edit -> audit/repair -> check -> build-specs -> transfer -> archive`

## Snapshot sources

- Pipeline: `/root/fangweiyi/multiview_repair_pipeline`
- Shared runner: `/root/fangweiyi/showcases/run_new_cases_twice.py`
- Baseline: `/root/fangweiyi/showcases/full_batch_20261008/configs/edge_704_15steps.json`
- Negative prompt: `/root/fangweiyi/showcases/full_batch_20261008/configs/general_restyle_negative_prompt.json`

## Known issue

In the Dex1 seed-2026 result, the third camera view redraws the gripper shadow
as a gripper-like mechanical object. The current pre-generation audit does not
inspect generated videos, so this defect passes the existing technical gates.

## Sensitive data

Credentials, model weights, source videos, generated videos, and experiment
artifacts are intentionally excluded.

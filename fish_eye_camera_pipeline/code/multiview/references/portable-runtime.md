# Portable runtime and harness

## What is bundled

```text
scripts/
|-- harness.py                         gated experiment driver
|-- survey_views.py                    one-frame-per-second contact sheets
|-- upsample_with_native_reasoner.py   native Cosmos Reasoner adapter
|-- run_reasoner.py                    bounded multi-GPU Reasoner runner
|-- finalize_prompts.py                schema cleanup and shared-block injection
|-- check_prompts.py                   cross-view structural gate
|-- build_specs.py                     official-baseline spec builder
|-- run_transfer.py                    bounded per-view Transfer runner
|-- verify_runs.py                     streaming media verification/contact sheets
|-- build_comparisons.py               post-run source/Transfer review mosaics
|-- check_standard_runtime.py          standard local Transfer dependency check
`-- run_gateway_edge_transfer.py       separate vLLM-Omni gateway smoke-test adapter
```

The Python media tools need `av`, `imageio`, `numpy`, and `Pillow`; install them from
`requirements-tools.txt`. Reasoner and Transfer additionally require a working Cosmos3
checkout, weights, CUDA, and the checkout's Python environment. Post-run comparison
videos require `ffmpeg` and `ffprobe`.

`run_gateway_edge_transfer.py` is not part of the standard gated workflow below. It is a
separate backend for servers that can serve `/v1/videos` through vLLM-Omni but cannot
import `cosmos_framework`. It writes to `gateway_prompt_sets/`, `gateway_specs/`, and
`gateway_runs/`. Read `references/gateway-vllm-omni.md` before using it.

## Configuration

Copy `assets/experiment.example.json`. Relative host paths resolve relative to the config
file, so a project can commit a config without machine-specific absolute paths.

Required keys:

- `cosmos_checkout`: Cosmos repository root.
- `views_root`: either scene directories containing per-view videos, or one scene dir.
- `experiment_root`: unique output experiment directory.

Common optional keys:

- `prompt_set` (`v1`), `run_name` (`edge_v1`), `control` (`edge`), `seed` (`2026`).
- `checkpoint_path` (`Cosmos3-Nano`). The resolution tier and sampling parameters always
  remain those of the official baseline.
- `python`, `framework_root`, `transfer_cookbook`, `baseline_spec`,
  `negative_prompt_file`, and `control_root` for nonstandard layouts.
- `comparison_font` to add source/Transfer labels using a chosen font file. Without it,
  the comparison video is still generated but has no text labels.

Machine-specific paths belong only in this runtime config. Generated specs resolve paths
from it; the skill itself does not assume a home directory, checkout, dataset, or
execution backend.

Before using the standard local Transfer path on a new machine, run:

```bash
python3 <skill-dir>/scripts/check_standard_runtime.py --cosmos-checkout <cosmos_checkout>
```

The standard chain is ready only when this reports `standard_transfer_ready: true`.

## Artifact layout

```text
<experiment_root>/
|-- experiment_index.json
|-- survey/<scene>/<view>.jpg
|-- prompt_sets/<version>/<scene>/
|   |-- transfer_plan.json
|   `-- <view>/
|       |-- plain_prompt.txt
|       |-- upsampler_raw_response.txt
|       |-- upsampled_prompt.json
|       |-- modified_prompt.json
|       `-- compliance_report.json
|-- specs/<run>/<scene>__<view>.json
|-- runs/<run>/<scene>__<view>/
|   |-- vision.mp4
|   |-- control_edge.mp4 or control_depth.mp4
|   |-- sample_args.json
|   `-- sample_outputs.json
|-- evaluations/<run>/verification.json
|-- comparisons/<run>/
`-- logs/
```

Harness code stays inside the skill, never under an experiment's `outputs`.

Gateway adapter artifacts intentionally use a separate namespace under the same
experiment root:

```text
gateway_prompt_sets/
gateway_specs/
gateway_runs/
```

## Harness commands

```bash
H="<skill-dir>/scripts/harness.py"
C="./cosmos-transfer.json"

python3 "$H" --config "$C" init
python3 "$H" --config "$C" survey
python3 "$H" --config "$C" status

# After plain_prompt.txt exists for every view:
python3 "$H" --config "$C" reasoner

python3 "$H" --config "$C" finalize
# Pause here for the independent per-scene audit.
python3 "$H" --config "$C" check
python3 "$H" --config "$C" build-specs

python3 "$H" --config "$C" transfer

python3 "$H" --config "$C" verify
python3 "$H" --config "$C" compare
```

`transfer` disables diffusion cache by default. Set
`no_diffusion_cache` to `false` only when intentionally testing cache behavior.

## Audit gate

Finalization is mechanical, not an audit. The harness writes an `AUDIT_REQUIRED` marker.
After fresh per-scene audit contexts edit every `modified_prompt.json` and write passing
`compliance_report.json`, run `check`. Only a successful check writes `AUDIT_DONE`.
`build-specs` refuses to proceed without that marker.

## Verification

`verify` streams frames and checks source, generated video, and control video for frame
count, FPS, dimensions, duration, and a non-frozen tail. It also writes start/middle/end
comparison sheets. Technical PASS does not imply semantic or cross-view PASS; inspect the
sheets and report those separately.

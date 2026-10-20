# Gateway vLLM-Omni adapter

This is a separate smoke-test backend for servers that expose Cosmos3 through
`vllm-omni` and `cosmos_control_service.gateway`, but do not have the local
`cosmos_framework` Transfer environment installed.

Do not report this backend as the standard `cosmos-transfer-per-view` Skill workflow.
It bypasses Reasoner expansion, manual audit gating, official baseline spec copying, and
`python -m cosmos_framework.scripts.inference`.

## When to use it

Use `scripts/run_gateway_edge_transfer.py` only when all of these are true:

- the user explicitly accepts a gateway adapter or a quick feasibility run;
- the server has `/v1/videos` available through vLLM-Omni or the local gateway;
- the selected control is `edge`;
- the result will be labelled as `gateway_vllm_omni_edge_adapter`.

If the task is to validate the standard Skill chain, use `scripts/harness.py` and
`scripts/run_transfer.py` instead.

## Artifact layout

Gateway artifacts intentionally use a different namespace:

```text
<experiment_root>/
|-- experiment_index.json
|-- gateway_prompt_sets/<prompt_set>/<scene>/<view>/
|-- gateway_specs/<run>/<scene>__<view>.json
|-- gateway_runs/<run>/<scene>__<view>/
|   |-- vision.mp4
|   |-- control_edge.mp4
|   |-- params.json
|   |-- extra_params.json
|   |-- async_submit.json
|   `-- status.json
`-- logs/
```

Standard Skill artifacts remain under `prompt_sets/`, `specs/`, `runs/`, and
`evaluations/`.

## Example

```bash
python3 scripts/run_gateway_edge_transfer.py \
  --views-root /root/sam31/data/dji_weather_rwy \
  --experiment-root /root/sam31/cosmos-transfer-per-view/outputs/dji_weather_rwy_gateway_edge_20260914 \
  --scene rwy_split2 \
  --prefix rwy_ \
  --user-prompt "将天气换成晴天" \
  --run-name gateway_edge_low_50steps \
  --steps 50 \
  --guidance-scale 3.0 \
  --control-guidance 1.5 \
  --flow-shift 10.0 \
  --resolution 720 \
  --edge-threshold low \
  --edge-weight 1.0 \
  --seed 2026 \
  --max-frames 80
```

The command writes `workflow_compliance: gateway_adapter_not_standard_skill_workflow`
into `experiment_index.json`, specs, and compliance reports.

## Standard-chain dependency check

Before claiming the standard local Transfer workflow is available on a server, verify:

```bash
python3 - <<'PY'
mods = ["cosmos_framework", "cosmos_control_service"]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__file__", None))
    except Exception as e:
        print(m, "MISSING", repr(e))
PY

test -f /path/to/cosmos/cookbooks/cosmos3/generator/transfer/specs/edge.json
test -f /path/to/cosmos/cookbooks/cosmos3/generator/transfer/assets/negative_prompt.json
```

`cosmos_control_service` alone is not enough for the standard chain. The standard chain
requires a Python environment that can import `cosmos_framework` plus the official
Transfer baseline spec files.

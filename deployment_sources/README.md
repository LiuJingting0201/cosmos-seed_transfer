# Deployment source snapshot

Read-only snapshot copied from the running development environment
`root@101.126.75.237:26659` on 2026-10-28. These files are source snapshots,
not an installable package and are not wired into the pipeline.

## Contents

- `scripts/start_services.sh`: service launcher used for the local Cosmos stack.
- `cosmos_control_service/gateway.py`: Gateway HTTP proxy.
- `cosmos_control_service/depth_server.py`: Depth control service.
- `vllm_omni/entrypoints/openai/api_server.py`: video API and job lifecycle.
- `vllm_omni/entrypoints/openai/stores.py`: in-memory video/task registries.
- `vllm_omni/entrypoints/openai/storage.py`: local output storage manager.

The launcher was copied from
`/root/sam31/cosmos-gpt6-local-20260916/single/cmd/start_services.sh`.
The two checked remote launcher copies had identical MD5 checksums.
The Python files were copied from `/usr/local/lib/python3.12/dist-packages`.

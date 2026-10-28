#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/root/sam31/reasoner_pipeline/logs/service_logs/$(date +%Y%m%d_%H%M%S)}"
MODEL_PATH="${MODEL_PATH:-/vemlp-demo-models/models/nvidia/Cosmos3-Nano/Cosmos3-Nano}"
DEPTH_WEIGHTS="${DEPTH_WEIGHTS:-/vemlp-demo-models/models/depth-anything/depth-anything/Video-Depth-Anything-Large/video_depth_anything_vitl.pth}"
DEPTH_CACHE_DIR="${DEPTH_CACHE_DIR:-/var/tmp/cosmos3-control/cache}"

START_COSMOS="${START_COSMOS:-0}"
START_REASONER="${START_REASONER:-1}"
START_DEPTH="${START_DEPTH:-0}"
START_GATEWAY="${START_GATEWAY:-0}"

COSMOS_GPUS="${COSMOS_GPUS:-0,1,2,3}"
REASONER_GPUS="${REASONER_GPUS:-4,5,6,7}"
REASONER_TP="${REASONER_TP:-4}"
REASONER_GPU_MEMORY_UTILIZATION="${REASONER_GPU_MEMORY_UTILIZATION:-0.85}"
REASONER_MAX_MODEL_LEN="${REASONER_MAX_MODEL_LEN:-32768}"
DEPTH_GPU="${DEPTH_GPU:-4}"

REASONER_FPS="${REASONER_FPS:-4}"
REASONER_NUM_FRAMES="${REASONER_NUM_FRAMES:-128}"

mkdir -p "$RUN_ROOT"/{cosmos,reasoner,depth,gateway} "$DEPTH_CACHE_DIR"
echo "$RUN_ROOT" > /root/sam31/reasoner_pipeline/state/latest_service_run_root.txt

export VLLM_SITE="${VLLM_SITE:-/usr/local/lib/python3.12/dist-packages}"
export LD_LIBRARY_PATH="$VLLM_SITE/nvidia/cu13/lib:$VLLM_SITE/nvidia/cublas/lib:$VLLM_SITE/nvidia/cuda_cupti/lib:$VLLM_SITE/nvidia/cuda_nvrtc/lib:$VLLM_SITE/nvidia/cuda_runtime/lib:$VLLM_SITE/nvidia/cudnn/lib:$VLLM_SITE/nvidia/cufft/lib:$VLLM_SITE/nvidia/cufile/lib:$VLLM_SITE/nvidia/curand/lib:$VLLM_SITE/nvidia/cusolver/lib:$VLLM_SITE/nvidia/cusparse/lib:$VLLM_SITE/nvidia/cusparselt/lib:$VLLM_SITE/nvidia/nccl/lib:$VLLM_SITE/nvidia/nvjitlink/lib:$VLLM_SITE/nvidia/nvshmem/lib:$VLLM_SITE/nvidia/nvtx/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/vemlp-demo-models/models}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

is_healthy() {
  curl -fsS --max-time 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1
}

wait_health() {
  local port="$1"
  local pid_file="$2"
  local log_file="$3"
  for _ in $(seq 1 180); do
    if is_healthy "$port"; then
      echo "Port $port is ready"
      return 0
    fi
    if [[ -s "$pid_file" ]] && ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      echo "Service on port $port exited" >&2
      tail -n 200 "$log_file" >&2 || true
      return 1
    fi
    sleep 10
  done
  echo "Timed out waiting for port $port" >&2
  tail -n 200 "$log_file" >&2 || true
  return 1
}

if [[ "$START_COSMOS" == "1" || "$START_COSMOS" == "true" ]]; then
  if is_healthy 8000; then
    echo "Port 8000 already healthy; skip starting Cosmos3."
  else
    (
      export CUDA_VISIBLE_DEVICES="$COSMOS_GPUS"
      export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA
      nohup /usr/local/bin/vllm-omni serve "$MODEL_PATH" \
        --omni \
        --model-class-name Cosmos3OmniDiffusersPipeline \
        --allowed-local-media-path / \
        --host 0.0.0.0 \
        --port 8000 \
        --init-timeout 1800 \
        --num-gpus 4 \
        --usp 4 \
        --ulysses-mode advanced_uaa \
        --ring-degree 1 \
        --cfg-parallel-size 1 \
        --no-guardrails \
        --enable-diffusion-pipeline-profiler \
        --diffusion-attention-backend TORCH_SDPA \
        > "$RUN_ROOT/cosmos/server.log" 2>&1 < /dev/null &
      echo $! > "$RUN_ROOT/cosmos/server.pid"
    )
    wait_health 8000 "$RUN_ROOT/cosmos/server.pid" "$RUN_ROOT/cosmos/server.log"
  fi
fi

if [[ "$START_REASONER" == "1" || "$START_REASONER" == "true" ]]; then
  if is_healthy 8001; then
    echo "Port 8001 already healthy; skip starting Reasoner."
  else
    (
      export CUDA_VISIBLE_DEVICES="$REASONER_GPUS"
      nohup /usr/local/bin/vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8001 \
        --served-model-name cosmos3-nano-reasoner \
        --tensor-parallel-size "$REASONER_TP" \
        --mm-encoder-tp-mode data \
        --async-scheduling \
        --disable-custom-all-reduce \
        --attention-backend TRITON_ATTN \
        --mm-encoder-attn-backend TORCH_SDPA \
        --dtype bfloat16 \
        --gpu-memory-utilization "$REASONER_GPU_MEMORY_UTILIZATION" \
        --max-model-len "$REASONER_MAX_MODEL_LEN" \
        --max-num-seqs 4 \
        --allowed-local-media-path / \
        --limit-mm-per-prompt '{"image":16,"video":1}' \
        --media-io-kwargs "{\"video\":{\"fps\":${REASONER_FPS},\"num_frames\":${REASONER_NUM_FRAMES}}}" \
        > "$RUN_ROOT/reasoner/server.log" 2>&1 < /dev/null &
      echo $! > "$RUN_ROOT/reasoner/server.pid"
    )
    wait_health 8001 "$RUN_ROOT/reasoner/server.pid" "$RUN_ROOT/reasoner/server.log"
  fi
fi

if [[ "$START_DEPTH" == "1" || "$START_DEPTH" == "true" ]]; then
  if is_healthy 8101; then
    echo "Port 8101 already healthy; skip starting Depth Worker."
  else
    cd /opt/vllm-omni-cosmos3
    (
      export CUDA_VISIBLE_DEVICES="$DEPTH_GPU"
      nohup python3 -m cosmos_control_service.depth_server \
        --weights "$DEPTH_WEIGHTS" \
        --encoder vitl \
        --device cuda \
        --cache-dir "$DEPTH_CACHE_DIR" \
        --allowed-input-root / \
        --host 127.0.0.1 \
        --port 8101 \
        > "$RUN_ROOT/depth/server.log" 2>&1 < /dev/null &
      echo $! > "$RUN_ROOT/depth/server.pid"
    )
    wait_health 8101 "$RUN_ROOT/depth/server.pid" "$RUN_ROOT/depth/server.log"
  fi
fi

if [[ "$START_GATEWAY" == "1" || "$START_GATEWAY" == "true" ]]; then
  if is_healthy 8002; then
    echo "Port 8002 already healthy; skip starting Gateway."
  else
    cd /opt/vllm-omni-cosmos3
    nohup python3 -m cosmos_control_service.gateway \
      --upstream-url http://127.0.0.1:8000 \
      --depth-worker-url http://127.0.0.1:8101 \
      --cache-dir "$DEPTH_CACHE_DIR" \
      --host 0.0.0.0 \
      --port 8002 \
      > "$RUN_ROOT/gateway/server.log" 2>&1 < /dev/null &
    echo $! > "$RUN_ROOT/gateway/server.pid"
    wait_health 8002 "$RUN_ROOT/gateway/server.pid" "$RUN_ROOT/gateway/server.log"
  fi
fi

for port in 8000 8001 8002 8101; do
  printf "port %s: " "$port"
  is_healthy "$port" && echo healthy || echo not_ready
done
echo "RUN_ROOT=$RUN_ROOT"

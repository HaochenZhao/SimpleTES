#!/usr/bin/env bash
# Shared cluster settings for the circle-packing SimpleTES reproduction.

export HEAD_IP="${HEAD_IP:-28.33.242.47}"
if [[ -n "${NODE_IP_LIST:-}" ]]; then
  ALL_IPS=()
  while read -r ip; do
    [[ -n "${ip}" ]] && ALL_IPS+=("${ip}")
  done < <(tr ',' '\n' <<<"${NODE_IP_LIST}" | cut -d: -f1)
else
  ALL_IPS=(28.33.242.47 28.33.244.15 28.33.244.19 28.33.243.17)
fi
export ALL_IPS

export PYTHON="${PYTHON:-/opt/conda/envs/torch-base/bin/python}"
export VLLM_BIN="${VLLM_BIN:-/opt/conda/envs/torch-base/bin/vllm}"
export MODEL_PATH="${MODEL_PATH:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/cache/hf/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a}"
export SERVED_NAME="${SERVED_NAME:-gpt-oss-120b}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export BASE_PORT="${BASE_PORT:-8100}"
export FRONTEND_PORT="${FRONTEND_PORT:-8000}"
export SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15}"
export REPRO_DIR="${REPRO_DIR:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_circle_packing_26}"
export LOG_DIR="${LOG_DIR:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_circle_packing_26/logs}"
export SIMPLES_DIR="${SIMPLES_DIR:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES}"
export HTTP_PROXY="${HTTP_PROXY:-http://star-proxy.oa.com:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://star-proxy.oa.com:3128}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,.oa.com,.woa.com,${HEAD_IP},28.33.244.15,28.33.244.19,28.33.243.17}"
export NO_PROXY="$no_proxy"

mkdir -p "${LOG_DIR}"

#!/usr/bin/env bash
# 8 TP=1 replicas on the launcher only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
PACK="${PACKING_DIR}"
IP="${HEAD_IP}"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Drop local burn so vLLM can take the 8 cards.
if [[ -f /tmp/gpu_burn.pid ]]; then
  kill "$(cat /tmp/gpu_burn.pid)" 2>/dev/null || true
  rm -f /tmp/gpu_burn.pid
fi
pkill -f 'repro_circle_packing_26/burn_node.py' >/dev/null 2>&1 || true

mkdir -p /tmp/tiktoken-rs-cache /root/vocab "${LOG_DIR}"
if [[ -d /apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/cache/tiktoken-rs-cache ]]; then
  cp -a /apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/cache/tiktoken-rs-cache/. /tmp/tiktoken-rs-cache/ || true
  cp -a /apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/cache/tiktoken-rs-cache/. /root/vocab/ || true
fi

for gpu in $(seq 0 7); do
  if [[ -f /tmp/vllm_gpu_${gpu}.pid ]]; then
    kill "$(cat /tmp/vllm_gpu_${gpu}.pid)" 2>/dev/null || true
    rm -f /tmp/vllm_gpu_${gpu}.pid
  fi
done

for gpu in $(seq 0 7); do
  port=$((BASE_PORT + gpu))
  logf="${LOG_DIR}/vllm_${IP}_gpu${gpu}.log"
  if [[ -f "${logf}" ]]; then
    mv "${logf}" "${logf}.prev.$(date +%Y%m%d_%H%M%S)" || true
  fi
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export HF_HOME=/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/cache/hf
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-rs-cache
  export VLLM_HOST_IP="${IP}"
  export no_proxy="localhost,127.0.0.1,${IP}"
  export NO_PROXY="localhost,127.0.0.1,${IP}"
  nohup "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${port}" \
    --served-model-name "${SERVED_NAME}" "openai/${SERVED_NAME}" \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --distributed-executor-backend mp \
    --disable-log-requests \
    </dev/null > "${logf}" 2>&1 &
  echo $! > "/tmp/vllm_gpu_${gpu}.pid"
  log "vLLM gpu${gpu} ${IP}:${port} pid=$(cat /tmp/vllm_gpu_${gpu}.pid)"
  sleep 8
done

export no_proxy='*'
log "waiting for 8 /v1/models"
for i in $(seq 1 180); do
  ok=0
  for gpu in $(seq 0 7); do
    if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$((BASE_PORT + gpu))/v1/models" >/dev/null 2>&1; then
      ok=$((ok + 1))
    fi
  done
  log "ready ${ok}/8 (try ${i})"
  if [[ "${ok}" -eq 8 ]]; then
    echo "vllm_head_ready"
    exit 0
  fi
  sleep 10
done
echo "vllm_head_timeout"
exit 1

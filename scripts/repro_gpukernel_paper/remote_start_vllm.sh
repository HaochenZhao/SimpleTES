#!/usr/bin/env bash
# Run ON a vLLM node. TP-sharded replicas so H20 can serve paper 65536 context.
set -euo pipefail
IP="${1:?node ip}"
MAX_LEN="${2:-65536}"
LOG_DIR="${3:?log dir}"
VLLM_BIN="${4:?vllm bin}"
MODEL_PATH="${5:?model path}"
SERVED_NAME="${6:-gpt-oss-120b}"
BASE_PORT="${7:-8100}"
TP="${8:-2}"

n_replicas=$((8 / TP))
if [[ "${n_replicas}" -lt 1 ]]; then
  echo "invalid TP=${TP}"
  exit 1
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [[ -f /tmp/gpu_burn.pid ]]; then
  kill "$(cat /tmp/gpu_burn.pid)" 2>/dev/null || true
  rm -f /tmp/gpu_burn.pid
fi
pkill -f 'repro_circle_packing_26/burn_node.py' >/dev/null 2>&1 || true

for gpu in $(seq 0 7); do
  if [[ -f /tmp/vllm_gpu_${gpu}.pid ]]; then
    kill "$(cat /tmp/vllm_gpu_${gpu}.pid)" 2>/dev/null || true
    rm -f /tmp/vllm_gpu_${gpu}.pid
  fi
done
for r in $(seq 0 7); do
  if [[ -f /tmp/vllm_rep_${r}.pid ]]; then
    kill "$(cat /tmp/vllm_rep_${r}.pid)" 2>/dev/null || true
    rm -f /tmp/vllm_rep_${r}.pid
  fi
done
pkill -f 'vllm serve' >/dev/null 2>&1 || true

for _ in $(seq 1 120); do
  max_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if($1+0>m)m=$1+0} END{print int(m)}')"
  if [[ "${max_mib}" -lt 4000 ]]; then
    log "gpus_free max_mib=${max_mib}"
    break
  fi
  log "waiting_gpu_free max_mib=${max_mib}"
  sleep 2
done

mkdir -p /tmp/tiktoken-rs-cache /root/vocab "${LOG_DIR}"
if [[ -d /apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/cache/tiktoken-rs-cache ]]; then
  cp -a /apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/cache/tiktoken-rs-cache/. /tmp/tiktoken-rs-cache/ || true
  cp -a /apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/cache/tiktoken-rs-cache/. /root/vocab/ || true
fi

sleep 2

for r in $(seq 0 $((n_replicas - 1))); do
  port=$((BASE_PORT + r))
  gpus="$(seq -s, $((r * TP)) $((r * TP + TP - 1)))"
  logf="${LOG_DIR}/vllm_${IP}_tp${TP}_r${r}.log"
  if [[ -f "${logf}" ]]; then
    mv "${logf}" "${logf}.prev.$(date +%Y%m%d_%H%M%S)" || true
  fi
  export CUDA_VISIBLE_DEVICES="${gpus}"
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
    --max-model-len "${MAX_LEN}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 8 \
    --distributed-executor-backend mp \
    --disable-log-requests \
    </dev/null > "${logf}" 2>&1 &
  echo $! > "/tmp/vllm_rep_${r}.pid"
  log "vLLM r${r} ${IP}:${port} gpus=${gpus} pid=$(cat /tmp/vllm_rep_${r}.pid) tp=${TP} max_len=${MAX_LEN}"
  sleep 12
done

export no_proxy='*'
log "waiting for ${n_replicas} /v1/models on ${IP}"
for i in $(seq 1 180); do
  ok=0
  for r in $(seq 0 $((n_replicas - 1))); do
    if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$((BASE_PORT + r))/v1/models" >/dev/null 2>&1; then
      ok=$((ok + 1))
    fi
  done
  log "ready ${ok}/${n_replicas} (try ${i})"
  if [[ "${ok}" -eq "${n_replicas}" ]]; then
    echo "vllm_node_ready ${IP} tp=${TP} replicas=${n_replicas} max_len=${MAX_LEN}"
    exit 0
  fi
  sleep 10
done
echo "vllm_node_timeout ${IP}"
tail -n 40 "${LOG_DIR}/vllm_${IP}_tp${TP}_r0.log" || true
exit 1

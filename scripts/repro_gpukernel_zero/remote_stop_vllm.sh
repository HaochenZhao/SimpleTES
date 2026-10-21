#!/usr/bin/env bash
# Run ON a node. Stop local vLLM replicas, then hold GPUs with burn until kernel starts.
set -euo pipefail
LOG_DIR="${1:?log dir}"
PYTHON="${2:?python}"
BURN="${3:?burn script}"
TAG="${4:-node}"
mkdir -p "${LOG_DIR}"
for g in $(seq 0 7); do
  if [[ -f /tmp/vllm_gpu_${g}.pid ]]; then
    kill "$(cat /tmp/vllm_gpu_${g}.pid)" 2>/dev/null || true
    rm -f /tmp/vllm_gpu_${g}.pid
  fi
done
pkill -f 'vllm serve' >/dev/null 2>&1 || true
if [[ -f /tmp/gpu_burn.pid ]]; then
  kill "$(cat /tmp/gpu_burn.pid)" 2>/dev/null || true
  rm -f /tmp/gpu_burn.pid
fi
for _ in $(seq 1 90); do
  max_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if($1+0>m)m=$1+0} END{print int(m)}')"
  if [[ "${max_mib}" -lt 4000 ]]; then
    echo "vllm_released max_mib=${max_mib}"
    break
  fi
  echo "waiting_vllm_release max_mib=${max_mib}"
  sleep 2
done
export BURN_HOLD_GIB=40
export BURN_MATRIX=8192
nohup "${PYTHON}" "${BURN}" </dev/null > "${LOG_DIR}/burn_${TAG}.log" 2>&1 &
echo $! > /tmp/gpu_burn.pid
disown || true
echo "vllm_stopped burn_pid=$(cat /tmp/gpu_burn.pid)"

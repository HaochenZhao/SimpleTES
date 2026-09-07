#!/usr/bin/env bash
# Restart the 8 TP=1 replicas on one node after harmony vocab is in place.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

IP="${1:?node ip}"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

remote() {
  if [[ "${IP}" == "${HEAD_IP}" ]]; then
    bash -lc "$*"
  else
    # shellcheck disable=SC2086
    ssh ${SSH_OPTS} "root@${IP}" "$*"
  fi
}

remote "mkdir -p /tmp/tiktoken-rs-cache /root/vocab; cp -a /apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/cache/tiktoken-rs-cache/. /tmp/tiktoken-rs-cache/; cp -a /apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/cache/tiktoken-rs-cache/. /root/vocab/; for g in \$(seq 0 7); do if [[ -f /tmp/vllm_gpu_\${g}.pid ]]; then kill \"\$(cat /tmp/vllm_gpu_\${g}.pid)\" 2>/dev/null || true; rm -f /tmp/vllm_gpu_\${g}.pid; fi; done"

for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
  port=$((BASE_PORT + gpu))
  logf="${LOG_DIR}/vllm_${IP}_gpu${gpu}.log"
  remote "export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HOME=/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/cache/hf
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-rs-cache
export HTTP_PROXY=${HTTP_PROXY}
export HTTPS_PROXY=${HTTPS_PROXY}
export http_proxy=${HTTP_PROXY}
export https_proxy=${HTTPS_PROXY}
export no_proxy=localhost,127.0.0.1,${IP}
export NO_PROXY=localhost,127.0.0.1,${IP}
export VLLM_HOST_IP=${IP}
nohup ${VLLM_BIN} serve ${MODEL_PATH} \
  --host 0.0.0.0 --port ${port} \
  --served-model-name ${SERVED_NAME} openai/${SERVED_NAME} \
  --max-model-len 32768 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --distributed-executor-backend mp \
  --disable-log-requests \
  > ${logf} 2>&1 &
echo \$! > /tmp/vllm_gpu_${gpu}.pid"
  log "relaunched ${IP}:${port} gpu${gpu}"
  sleep 8
done

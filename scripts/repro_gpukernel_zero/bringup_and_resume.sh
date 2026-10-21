#!/usr/bin/env bash
# New-cluster bring-up: 8 vLLM on launcher, 3 kernel nodes 4+4, resume 27648.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
STATUS="${KERNEL_LOG_DIR}/bringup.status"
INSTANCE="checkpoints/gpukernel_zero_trimul/2026-09-08/instance-bd60257b"
OCCUPY="${HERE}/occupy_node.sh"
BURN="${SIMPLES_DIR}/scripts/repro_circle_packing_26/burn_node.py"
mkdir -p "${KERNEL_LOG_DIR}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${STATUS}"; }

log "HEAD=${HEAD_IP} LLM=${LLM_IPS} KERNELS=${KERNEL_NODES} ALL=${ALL_IPS[*]}"

log "rotate shared logs (quota)"
bash "${HERE}/rotate_quota_logs.sh" >> "${STATUS}" 2>&1 || true

log "occupy remote kernel nodes"
IFS=',' read -r -a KN <<< "${KERNEL_NODES}"
for ip in "${KN[@]}"; do
  (
    timeout 40 ssh ${SSH_OPTS} -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
      "root@${ip}" "bash ${OCCUPY} ${KERNEL_LOG_DIR} ${PYTHON} ${BURN} ${ip}" \
      > "${KERNEL_LOG_DIR}/occupy_${ip}.out" 2>&1 \
      || echo "occupy_ssh_fail ${ip}" >> "${KERNEL_LOG_DIR}/occupy_${ip}.out"
  ) &
done

log "write server_info.yaml"
bash "${HERE}/write_server_info.sh" >> "${STATUS}" 2>&1

log "start 8 vLLM on launcher ${HEAD_IP}"
if ! bash "${HERE}/start_vllm_head.sh" >> "${STATUS}" 2>&1; then
  log "vLLM start failed"
  exit 1
fi

log "nginx launcher-only + light keepalive"
bash "${HERE}/reload_nginx_launcher_only.sh" >> "${STATUS}" 2>&1 || log "nginx_fail"
export KEEPALIVE_WORKERS=8
bash "${HERE}/switch_keepalive_light.sh" >> "${STATUS}" 2>&1 || true

wait || true
log "start kernel servers 4+4"
if ! bash "${HERE}/start_kernels_split.sh" 4 >> "${STATUS}" 2>&1; then
  log "kernel start failed"
  exit 1
fi

if [[ -f "${KERNEL_LOG_DIR}/resume.log" ]]; then
  mv "${KERNEL_LOG_DIR}/resume.log" "${KERNEL_LOG_DIR}/resume.log.prev.$(date +%Y%m%d_%H%M%S)"
fi

log "resume search from ${INSTANCE}"
cd "${SIMPLES_DIR}"
nohup bash "${HERE}/run_simpletes.sh" --resume "${INSTANCE}" --eval-concurrency 24 \
  > "${KERNEL_LOG_DIR}/resume.log" 2>&1 &
echo $! > "${KERNEL_LOG_DIR}/resume.pid"
log "resume_pid=$(cat "${KERNEL_LOG_DIR}/resume.pid")"
log "BRINGUP_DONE"

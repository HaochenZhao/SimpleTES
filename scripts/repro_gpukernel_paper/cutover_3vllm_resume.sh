#!/usr/bin/env bash
# Cut over to 3 vLLM + 2 eval, then resume instance-7bf43cb3.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
STATUS="${KERNEL_LOG_DIR}/cutover_3vllm.status"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${STATUS}"; }

: > "${STATUS}"
log "LLM=${LLM_IPS}"
log "KERNELS=${KERNEL_NODES}"

if pgrep -f 'python3 main.py .*init_program_naive' >/dev/null 2>&1; then
  log "search still running, refuse to overlap"
  exit 1
fi

log "write server_info.yaml for 2 eval nodes"
bash "${HERE}/write_server_info.sh" | tee -a "${STATUS}"

export no_proxy='*'
for ip in 28.33.242.50 28.33.245.72; do
  if ! curl --noproxy '*' -fsS --max-time 5 "http://${ip}:${KERNEL_PORT}/health" >/dev/null 2>&1; then
    log "kernel ${ip} UNHEALTHY, restarting 4+4"
    bash "${HERE}/start_kernels_split.sh" 4 | tee -a "${STATUS}"
    break
  fi
  log "kernel ${ip} healthy"
done

log "start vLLM 49152 on 28.33.241.189 and 28.33.240.27"
fail=0
bash "${HERE}/start_vllm_node.sh" 28.33.241.189 > "${KERNEL_LOG_DIR}/start_vllm_28.33.241.189.out" 2>&1 &
p1=$!
bash "${HERE}/start_vllm_node.sh" 28.33.240.27 > "${KERNEL_LOG_DIR}/start_vllm_28.33.240.27.out" 2>&1 &
p2=$!
if ! wait "${p1}"; then
  log "vLLM 28.33.241.189 FAILED"
  tail -n 40 "${KERNEL_LOG_DIR}/start_vllm_28.33.241.189.out" | tee -a "${STATUS}" || true
  fail=1
else
  log "vLLM 28.33.241.189 ready"
fi
if ! wait "${p2}"; then
  log "vLLM 28.33.240.27 FAILED"
  tail -n 40 "${KERNEL_LOG_DIR}/start_vllm_28.33.240.27.out" | tee -a "${STATUS}" || true
  fail=1
else
  log "vLLM 28.33.240.27 ready"
fi
if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

log "reload nginx -> 24 backends"
bash "${HERE}/reload_nginx_llm.sh" | tee -a "${STATUS}"

if ! curl --noproxy '*' -fsS --max-time 8 "http://127.0.0.1:${FRONTEND_PORT}/v1/models" >/dev/null 2>&1; then
  log "nginx/vllm UNHEALTHY"
  exit 1
fi
log "nginx/vllm ok"

log "smoke token-forcing"
if ! python3 "${HERE}/smoke_token_forcing.py" >> "${STATUS}" 2>&1; then
  log "token-forcing smoke failed"
  tail -n 40 "${STATUS}"
  exit 1
fi
log "token-forcing smoke ok"

if [[ -f "${KERNEL_LOG_DIR}/simpletes.log" ]]; then
  mv "${KERNEL_LOG_DIR}/simpletes.log" "${KERNEL_LOG_DIR}/simpletes.log.prev.$(date +%Y%m%d_%H%M%S)"
fi

log "resume 7bf43cb3 gen=96 eval=16"
cd "${SIMPLES_DIR}"
nohup bash "${HERE}/resume_simpletes.sh" \
  checkpoints/gpukernel_trimul_paper/2026-09-17/instance-7bf43cb3 \
  > "${KERNEL_LOG_DIR}/simpletes.log" 2>&1 &
echo $! > "${KERNEL_LOG_DIR}/simpletes.pid"
log "simpletes_pid=$(cat "${KERNEL_LOG_DIR}/simpletes.pid")"
sleep 8
if kill -0 "$(cat "${KERNEL_LOG_DIR}/simpletes.pid")" 2>/dev/null; then
  log "CUTOVER_DONE"
else
  log "SEARCH_PID_DEAD"
  tail -n 80 "${KERNEL_LOG_DIR}/simpletes.log" | tee -a "${STATUS}" || true
  exit 1
fi

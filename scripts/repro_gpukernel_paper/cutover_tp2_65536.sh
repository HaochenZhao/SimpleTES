#!/usr/bin/env bash
# Restart 3 LLM nodes as TP=2 / 65536, reload nginx, resume search.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
STATUS="${KERNEL_LOG_DIR}/cutover_tp2_65536.status"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${STATUS}"; }

: > "${STATUS}"
log "LLM=${LLM_IPS} TP=${VLLM_TP} max_len=${MAX_MODEL_LEN} replicas/node=${VLLM_REPLICAS_PER_NODE}"

if pgrep -f 'python3 main.py .*init_program_naive' >/dev/null 2>&1; then
  log "search still running, refuse"
  exit 1
fi

export no_proxy='*'
for ip in 28.33.242.50 28.33.245.72; do
  if ! curl --noproxy '*' -fsS --max-time 5 "http://${ip}:${KERNEL_PORT}/health" >/dev/null 2>&1; then
    log "kernel ${ip} UNHEALTHY"
    exit 1
  fi
  log "kernel ${ip} healthy"
done

fail=0
IFS=',' read -r -a llm <<< "${LLM_IPS}"
pids=()
for ip in "${llm[@]}"; do
  log "start vLLM ${ip}"
  bash "${HERE}/start_vllm_node.sh" "${ip}" > "${KERNEL_LOG_DIR}/start_vllm_${ip}.out" 2>&1 &
  pids+=($!)
done
for i in "${!pids[@]}"; do
  ip="${llm[$i]}"
  if ! wait "${pids[$i]}"; then
    log "vLLM ${ip} FAILED"
    tail -n 50 "${KERNEL_LOG_DIR}/start_vllm_${ip}.out" | tee -a "${STATUS}" || true
    fail=1
  else
    log "vLLM ${ip} ready"
  fi
done
if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

log "reload nginx"
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

log "resume from latest ckpt context=65536 reasoning=50000 gen=48"
cd "${SIMPLES_DIR}"
nohup bash "${HERE}/resume_simpletes.sh" \
  checkpoints/gpukernel_trimul_paper/2026-09-17/instance-7bf43cb3 \
  > "${KERNEL_LOG_DIR}/simpletes.log" 2>&1 &
echo $! > "${KERNEL_LOG_DIR}/simpletes.pid"
log "simpletes_pid=$(cat "${KERNEL_LOG_DIR}/simpletes.pid")"
sleep 10
if kill -0 "$(cat "${KERNEL_LOG_DIR}/simpletes.pid")" 2>/dev/null; then
  log "CUTOVER_DONE"
else
  log "SEARCH_PID_DEAD"
  tail -n 80 "${KERNEL_LOG_DIR}/simpletes.log" | tee -a "${STATUS}" || true
  exit 1
fi

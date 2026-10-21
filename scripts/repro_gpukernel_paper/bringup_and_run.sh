#!/usr/bin/env bash
# Occupy 4 nodes, start 3 kernel servers + launcher vLLM, then paper-aligned search.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
STATUS="${KERNEL_LOG_DIR}/bringup.status"
PACK_LOG="${PACKING_DIR}/logs"
mkdir -p "${KERNEL_LOG_DIR}" "${PACK_LOG}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${STATUS}"; }

: > "${STATUS}"
log "HEAD=${HEAD_IP} LLM=${LLM_IPS} KERNELS=${KERNEL_NODES} idle=28.33.240.27"
log "paper cfg: C=32 K=8 N=51200 rpucg λ=1.0 γ=0.8 temp=1.0 reasoning=high token-forcing ctx=49152"

if pgrep -f 'checkpoints/gpukernel_trimul_paper' >/dev/null 2>&1; then
  log "paper search already running, refuse to overlap"
  exit 1
fi

log "occupy 4 experiment nodes"
bash "${HERE}/occupy_now.sh" >> "${STATUS}" 2>&1 || true

log "write official server_info.yaml"
bash "${HERE}/write_server_info.sh" >> "${STATUS}" 2>&1

log "start official kernel servers 4+4"
if ! bash "${HERE}/start_kernels_split.sh" 4 >> "${STATUS}" 2>&1; then
  log "kernel start failed"
  tail -n 50 "${STATUS}"
  exit 1
fi

log "start vLLM on launcher max-model-len=49152"
if ! bash "${HERE}/start_vllm_head.sh" >> "${STATUS}" 2>&1; then
  log "vllm start failed"
  tail -n 40 "${STATUS}"
  exit 1
fi

log "nginx launcher-only"
export NODE_IP_LIST="${LLM_IPS}"
# shellcheck source=../repro_circle_packing_26/cluster_env.sh
source "${PACKING_DIR}/cluster_env.sh"
bash "${PACKING_DIR}/gen_nginx_conf.sh" >> "${STATUS}" 2>&1
if [[ -f "${PACK_LOG}/nginx.pid" ]] && kill -0 "$(cat "${PACK_LOG}/nginx.pid")" 2>/dev/null; then
  nginx -s reload -c "${PACKING_DIR}/nginx.conf" -p "${PACKING_DIR}/"
else
  nginx -s quit >/dev/null 2>&1 || true
  nginx -c "${PACKING_DIR}/nginx.conf" -p "${PACKING_DIR}/"
fi
log "nginx backends=${ALL_IPS[*]} port=${FRONTEND_PORT}"

# Restore paper cluster env after nginx helper overwrote ALL_IPS.
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

export no_proxy='*'
if ! curl --noproxy '*' -fsS --max-time 8 "http://127.0.0.1:${FRONTEND_PORT}/v1/models" >/dev/null 2>&1; then
  log "nginx/vllm UNHEALTHY"
  exit 1
fi
log "nginx/vllm ok"

log "smoke token-forcing completions"
if ! python3 "${HERE}/smoke_token_forcing.py" >> "${STATUS}" 2>&1; then
  log "token-forcing smoke failed"
  tail -n 40 "${STATUS}"
  exit 1
fi
log "token-forcing smoke ok"

if [[ -f "${KERNEL_LOG_DIR}/simpletes.log" ]]; then
  mv "${KERNEL_LOG_DIR}/simpletes.log" "${KERNEL_LOG_DIR}/simpletes.log.prev.$(date +%Y%m%d_%H%M%S)"
fi

log "start paper-aligned search"
cd "${SIMPLES_DIR}"
nohup bash "${HERE}/run_simpletes.sh" \
  > "${KERNEL_LOG_DIR}/simpletes.log" 2>&1 &
echo $! > "${KERNEL_LOG_DIR}/simpletes.pid"
log "simpletes_pid=$(cat "${KERNEL_LOG_DIR}/simpletes.pid")"
sleep 5
if kill -0 "$(cat "${KERNEL_LOG_DIR}/simpletes.pid")" 2>/dev/null; then
  log "BRINGUP_DONE"
else
  log "SEARCH_PID_DEAD"
  tail -n 50 "${KERNEL_LOG_DIR}/simpletes.log" | tee -a "${STATUS}" || true
  exit 1
fi

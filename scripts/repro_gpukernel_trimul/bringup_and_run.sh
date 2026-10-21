#!/usr/bin/env bash
# Switch 3 eval nodes to official guarded kernel servers, then start paper-scale search.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
STATUS="${KERNEL_LOG_DIR}/bringup.status"
PACK_LOG="${PACKING_DIR}/logs"
mkdir -p "${KERNEL_LOG_DIR}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${STATUS}"; }

: > "${STATUS}"
log "HEAD=${HEAD_IP} LLM=${LLM_IPS} KERNELS=${KERNEL_NODES}"

log "truncate oversized shared logs in place"
for f in \
  "${PACK_LOG}/nginx_access.log" \
  "${PACK_LOG}/nginx_error.log" \
  "${PACK_LOG}/keepalive.log" \
  "${PACK_LOG}"/vllm_*.log \
  "${KERNEL_LOG_DIR}/simpletes.log"
do
  [[ -f "${f}" && ! -L "${f}" ]] || continue
  sz="$(stat -c%s "${f}" 2>/dev/null || echo 0)"
  if [[ "${sz}" -gt 5000000 ]]; then
    : > "${f}"
    log "truncated ${f} was_bytes=${sz}"
  fi
done

if pgrep -f 'gpukernel_zero_trimul' >/dev/null 2>&1; then
  log "zero search still running, refuse to overlap"
  exit 1
fi

export no_proxy='*'
if curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:${FRONTEND_PORT}/v1/models" >/dev/null 2>&1; then
  log "nginx/vllm ok"
else
  log "nginx/vllm UNHEALTHY"
  exit 1
fi

log "write official server_info.yaml"
bash "${HERE}/write_server_info.sh" >> "${STATUS}" 2>&1

log "start official kernel servers 4+4"
if ! bash "${HERE}/start_kernels_split.sh" 4 >> "${STATUS}" 2>&1; then
  log "kernel start failed"
  tail -n 40 "${STATUS}"
  exit 1
fi

if [[ -f "${KERNEL_LOG_DIR}/simpletes.log" ]]; then
  mv "${KERNEL_LOG_DIR}/simpletes.log" "${KERNEL_LOG_DIR}/simpletes.log.prev.$(date +%Y%m%d_%H%M%S)"
fi

log "start guarded paper-scale search"
cd "${SIMPLES_DIR}"
nohup bash "${HERE}/run_simpletes.sh" \
  > "${KERNEL_LOG_DIR}/simpletes.log" 2>&1 &
echo $! > "${KERNEL_LOG_DIR}/simpletes.pid"
log "simpletes_pid=$(cat "${KERNEL_LOG_DIR}/simpletes.pid")"
sleep 3
if kill -0 "$(cat "${KERNEL_LOG_DIR}/simpletes.pid")" 2>/dev/null; then
  log "BRINGUP_DONE"
else
  log "SEARCH_PID_DEAD"
  tail -n 40 "${KERNEL_LOG_DIR}/simpletes.log" | tee -a "${STATUS}" || true
  exit 1
fi

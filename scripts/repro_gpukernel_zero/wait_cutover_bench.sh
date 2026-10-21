#!/usr/bin/env bash
# After the 9216 checkpoint: stop search, 8-way launcher LLM, 3 kernel nodes,
# bench compile/eval GPU splits, resume with the fastest split.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
STATUS="${KERNEL_LOG_DIR}/cutover.status"
LOCK="${KERNEL_LOG_DIR}/cutover.lock"
INSTANCE="checkpoints/gpukernel_zero_trimul/2026-09-08/instance-bd60257b"
BENCH_DIR="${HERE}/results/split_bench"
mkdir -p "${KERNEL_LOG_DIR}" "${BENCH_DIR}"

if [[ -f "${LOCK}" ]] && kill -0 "$(cat "${LOCK}")" 2>/dev/null; then
  echo "cutover already running pid=$(cat "${LOCK}")"
  exit 0
fi
echo $$ > "${LOCK}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${STATUS}"
}

log "armed: waiting for 9216 checkpoint"

while ! "${PYTHON}" "${HERE}/check_9216.py" >> "${STATUS}" 2>&1; do
  sleep 20
done
log "9216 checkpoint present — stopping search"

bash "${HERE}/stop_search.sh" >> "${STATUS}" 2>&1 || log "stop_search_nonzero"
log "search stop returned"

log "reload nginx to launcher 8-way"
bash "${HERE}/reload_nginx_launcher_only.sh" >> "${STATUS}" 2>&1 || log "nginx_reload_failed"

log "stop vLLM on 244.15 / 244.19"
bash "${HERE}/stop_vllm_eval_nodes.sh" >> "${STATUS}" 2>&1 || log "stop_vllm_failed"

log "write 3-eval server_info.yaml"
bash "${HERE}/write_server_info.sh" >> "${STATUS}" 2>&1

SPLITS=(1 2 3 4)
for n in "${SPLITS[@]}"; do
  split="${n}+$((8 - n))"
  log "start kernels split=${split}"
  if ! bash "${HERE}/start_kernels_split.sh" "${n}" >> "${STATUS}" 2>&1; then
    log "start_failed split=${split}"
    continue
  fi
  out="${BENCH_DIR}/split_${n}.json"
  log "bench split=${split}"
  if ! "${PYTHON}" "${HERE}/bench_eval_split.py" --split "${split}" --concurrency 24 --timeout 600 --out "${out}" >> "${STATUS}" 2>&1; then
    log "bench_failed split=${split}"
  fi
done

JSONS=()
for f in "${BENCH_DIR}"/split_*.json; do
  [[ -f "${f}" ]] && JSONS+=("${f}")
done
if [[ "${#JSONS[@]}" -eq 0 ]]; then
  log "no_bench_results fallback_split=2+6"
  WIN_N=2
else
  WINNER_LINE="$("${PYTHON}" "${HERE}/pick_best_split.py" "${JSONS[@]}" | tail -n 1)"
  log "bench_summary winner=${WINNER_LINE}"
  WIN_N="${WINNER_LINE%%+*}"
  if [[ -z "${WIN_N}" || ! "${WIN_N}" =~ ^[0-9]+$ ]]; then
    WIN_N=2
  fi
fi

log "restart kernels on winner ${WIN_N}+$((8 - WIN_N))"
bash "${HERE}/start_kernels_split.sh" "${WIN_N}" >> "${STATUS}" 2>&1 || log "winner_start_failed"

if [[ -f "${KERNEL_LOG_DIR}/resume.log" ]]; then
  mv "${KERNEL_LOG_DIR}/resume.log" "${KERNEL_LOG_DIR}/resume.log.prev.$(date +%Y%m%d_%H%M%S)"
fi

log "resume search eval-concurrency=24"
cd "${SIMPLES_DIR}"
nohup bash "${HERE}/run_simpletes.sh" --resume "${INSTANCE}" --eval-concurrency 24 \
  > "${KERNEL_LOG_DIR}/resume.log" 2>&1 &
echo $! > "${KERNEL_LOG_DIR}/resume.pid"
log "resume_pid=$(cat "${KERNEL_LOG_DIR}/resume.pid") winner=${WIN_N}+$((8 - WIN_N))"
log "CUTOVER_DONE"

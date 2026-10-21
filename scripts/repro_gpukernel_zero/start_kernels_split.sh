#!/usr/bin/env bash
# Start gpukernel_zero servers on all 3 eval nodes with the same compile/eval GPU split.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

N_COMPILE="${1:?compile gpu count 1-5}"
IFS=',' read -r -a NODES <<< "${KERNEL_NODES}"

compile=""
evalg=""
for i in $(seq 0 7); do
  if [[ "${i}" -lt "${N_COMPILE}" ]]; then
    compile+="${compile:+,}${i}"
  else
    evalg+="${evalg:+,}${i}"
  fi
done
echo "split compile=${compile} eval=${evalg} nodes=${KERNEL_NODES}"

ssh_cmd() {
  local ip="$1"
  shift
  # shellcheck disable=SC2086
  timeout 240 ssh ${SSH_OPTS} -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
    "root@${ip}" "$@"
}

for ip in "${NODES[@]}"; do
  (
    echo "[${ip}] starting kernel"
    ssh_cmd "${ip}" "bash ${HERE}/remote_start_kernel.sh ${KERNEL_LOG_DIR} ${PYTHON} ${SIMPLES_DIR} ${KERNEL_PORT} ${compile} ${evalg} ${ip}" \
      > "${KERNEL_LOG_DIR}/start_kernel_${ip}.out" 2>&1 \
      || echo "[${ip}] start_ssh_failed" >> "${KERNEL_LOG_DIR}/start_kernel_${ip}.out"
  ) &
done
wait

export no_proxy='*'
ok=0
for ip in "${NODES[@]}"; do
  healthy=0
  for _ in $(seq 1 60); do
    if curl --noproxy '*' -fsS --max-time 5 "http://${ip}:${KERNEL_PORT}/health" >/dev/null 2>&1; then
      echo "[${ip}] healthy compile=${compile} eval=${evalg}"
      healthy=1
      ok=$((ok + 1))
      break
    fi
    sleep 2
  done
  if [[ "${healthy}" -eq 0 ]]; then
    echo "[${ip}] UNHEALTHY"
    tail -n 30 "${KERNEL_LOG_DIR}/kernel_${ip}.log" || true
    tail -n 20 "${KERNEL_LOG_DIR}/start_kernel_${ip}.out" || true
  fi
done

if [[ "${ok}" -ne "${#NODES[@]}" ]]; then
  echo "kernel_health_failed ${ok}/${#NODES[@]}"
  exit 1
fi
echo "all_kernels_healthy split=${N_COMPILE}+$((8 - N_COMPILE))"

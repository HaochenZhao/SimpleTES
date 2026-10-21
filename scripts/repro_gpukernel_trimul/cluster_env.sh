#!/usr/bin/env bash
# Official guarded TriMul. Reuses the current 1 LLM + 3 kernel allocation.

PACKING_DIR="${PACKING_DIR:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_circle_packing_26}"
# shellcheck source=../repro_circle_packing_26/cluster_env.sh
source "${PACKING_DIR}/cluster_env.sh"

if [[ ${#ALL_IPS[@]} -ge 4 ]]; then
  export LLM_IPS="${LLM_IPS:-${ALL_IPS[0]}}"
  export KERNEL_NODE="${KERNEL_NODE:-${ALL_IPS[1]}}"
  export KERNEL_NODES="${KERNEL_NODES:-${ALL_IPS[1]},${ALL_IPS[2]},${ALL_IPS[3]}}"
else
  export LLM_IPS="${LLM_IPS:-${HEAD_IP}}"
  export KERNEL_NODE="${KERNEL_NODE:-28.33.241.232}"
  export KERNEL_NODES="${KERNEL_NODES:-28.33.241.232,28.33.244.228,28.33.244.27}"
fi
export KERNEL_PORT="${KERNEL_PORT:-8000}"
export KERNEL_REPRO_DIR="${KERNEL_REPRO_DIR:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_gpukernel_trimul}"
export KERNEL_LOG_DIR="${KERNEL_LOG_DIR:-${KERNEL_REPRO_DIR}/logs}"
export no_proxy="${no_proxy},${KERNEL_NODES},${KERNEL_NODE}"
export NO_PROXY="$no_proxy"

mkdir -p "${KERNEL_LOG_DIR}" "${KERNEL_REPRO_DIR}/results"

kernel_remote() {
  local ip="$1"
  shift
  if [[ "${ip}" == "${HEAD_IP}" ]]; then
    bash -lc "$*"
  else
    # shellcheck disable=SC2086
    ssh ${SSH_OPTS} "root@${ip}" "$*"
  fi
}

#!/usr/bin/env bash
# Isolated unguarded TriMul search. Does not modify datasets/gpukernel.

PACKING_DIR="${PACKING_DIR:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_circle_packing_26}"
# shellcheck source=../repro_circle_packing_26/cluster_env.sh
source "${PACKING_DIR}/cluster_env.sh"

export KERNEL_NODE="${KERNEL_NODE:-28.33.243.17}"
export KERNEL_PORT="${KERNEL_PORT:-8000}"
export LLM_IPS="${LLM_IPS:-28.33.242.47,28.33.244.15,28.33.244.19}"
export KERNEL_ZERO_DIR="${KERNEL_ZERO_DIR:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_gpukernel_zero}"
export KERNEL_LOG_DIR="${KERNEL_LOG_DIR:-${KERNEL_ZERO_DIR}/logs}"
export no_proxy="${no_proxy},${KERNEL_NODE}"
export NO_PROXY="$no_proxy"

mkdir -p "${KERNEL_LOG_DIR}" "${KERNEL_ZERO_DIR}/results"

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

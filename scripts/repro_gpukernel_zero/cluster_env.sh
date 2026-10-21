#!/usr/bin/env bash
# Isolated unguarded TriMul search. Does not modify datasets/gpukernel.

PACKING_DIR="${PACKING_DIR:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_circle_packing_26}"
# shellcheck source=../repro_circle_packing_26/cluster_env.sh
source "${PACKING_DIR}/cluster_env.sh"

# Launcher = first node (vLLM). The other three are exclusive kernel/eval.
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
export KERNEL_ZERO_DIR="${KERNEL_ZERO_DIR:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_gpukernel_zero}"
export KERNEL_LOG_DIR="${KERNEL_LOG_DIR:-${KERNEL_ZERO_DIR}/logs}"
export no_proxy="${no_proxy},${KERNEL_NODES}"
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

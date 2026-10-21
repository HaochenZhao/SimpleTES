#!/usr/bin/env bash
# Paper-aligned TriMul on 3 of the current Taiji 4x8 H20 nodes.
# 28.33.242.226 is left free.
#   LLM  : 28.33.243.157
#          4x TP=2 replicas, max-model-len 65536 (H20 cannot fit 65536 at TP=1)
#   eval : 28.33.245.196, 28.33.243.114

PACKING_DIR="${PACKING_DIR:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_circle_packing_26}"
TRIMUL_DIR="${TRIMUL_DIR:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_gpukernel_trimul}"

# Only the 3 occupied nodes. Do not pull in 28.33.242.226.
export NODE_IP_LIST="28.33.243.157:8,28.33.245.196:8,28.33.243.114:8"
export HEAD_IP="28.33.243.157"
# shellcheck source=../repro_circle_packing_26/cluster_env.sh
source "${PACKING_DIR}/cluster_env.sh"

export LLM_IPS="28.33.243.157"
export KERNEL_NODE="28.33.245.196"
export KERNEL_NODES="28.33.245.196,28.33.243.114"
export KERNEL_PORT="${KERNEL_PORT:-8000}"
export KERNEL_REPRO_DIR="${KERNEL_REPRO_DIR:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_gpukernel_paper}"
export KERNEL_LOG_DIR="${KERNEL_LOG_DIR:-${KERNEL_REPRO_DIR}/logs}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
# Paper raises the window to 65536 when evolved programs no longer fit 49152.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export VLLM_TP="${VLLM_TP:-2}"
export VLLM_REPLICAS_PER_NODE="${VLLM_REPLICAS_PER_NODE:-4}"
export no_proxy="${no_proxy},${LLM_IPS},${KERNEL_NODES},${KERNEL_NODE},${HEAD_IP}"
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

#!/usr/bin/env bash
# Start TP-sharded vLLM on one node (including the launcher).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
IP="${1:?node ip}"
MAX_LEN="${MAX_MODEL_LEN:-65536}"
TP="${VLLM_TP:-2}"
CMD="bash ${HERE}/remote_start_vllm.sh ${IP} ${MAX_LEN} ${KERNEL_LOG_DIR} ${VLLM_BIN} ${MODEL_PATH} ${SERVED_NAME} ${BASE_PORT} ${TP}"

if [[ "${IP}" == "${HEAD_IP}" ]]; then
  timeout 2400 bash -lc "${CMD}"
else
  # shellcheck disable=SC2086
  timeout 2400 ssh ${SSH_OPTS} -o ServerAliveInterval=15 -o ServerAliveCountMax=8 \
    "root@${IP}" "${CMD}"
fi

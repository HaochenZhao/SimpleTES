#!/usr/bin/env bash
set -euo pipefail
LOG=/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_gpukernel_zero/logs
OCC=/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_gpukernel_zero/occupy_node.sh
BURN=/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_circle_packing_26/burn_node.py
PY=/opt/conda/envs/torch-base/bin/python
for ip in 28.33.244.15 28.33.244.19; do
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
    -o ServerAliveInterval=3 -o ServerAliveCountMax=2 \
    "root@${ip}" "bash ${OCC} ${LOG} ${PY} ${BURN} ${ip}" \
    > "${LOG}/ssh_occupy_${ip}.out" 2>&1 || echo "fail ${ip}" > "${LOG}/ssh_occupy_${ip}.out"
done
echo LLM_OCCUPY_DONE

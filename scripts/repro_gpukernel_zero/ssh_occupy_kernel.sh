#!/usr/bin/env bash
set -euo pipefail
LOG=/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_gpukernel_zero/logs
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
  -o ServerAliveInterval=3 -o ServerAliveCountMax=2 \
  root@28.33.243.17 \
  "bash /apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_gpukernel_zero/occupy_node.sh ${LOG} /opt/conda/envs/torch-base/bin/python /apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_circle_packing_26/burn_node.py 28.33.243.17"
echo SSH_OCCUPY_DONE
